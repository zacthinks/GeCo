"""Bounded, reproducible candidate tasks shared by Code and Scale acquisition.

This module samples tasks, never judgments, models, predictions, or rankings.
An acquisition scorer explicitly acknowledges the tasks it actually evaluated.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from array import array
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

DEFAULT_SCALE_CANDIDATE_BUDGET = 1_000
MAX_EXHAUSTIVE_SCALE_PAIRS = 1_000_000
MAX_BATCH_SIZE = 256


def resolve_candidate_budget(kind: str, budget: int | None) -> int:
    """None is type-specific compatibility default; zero is always exhaustive."""
    if kind not in {"code", "scale"}:
        raise ValueError("Candidate tasks require a Code or Scale.")
    if budget is None:
        return 0 if kind == "code" else DEFAULT_SCALE_CANDIDATE_BUDGET
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("Candidate evaluation budget must be a nonnegative integer.")
    return budget


def _permutation(size: int, seed: int) -> Iterator[int]:
    """A constant-memory seeded full-cycle permutation of [0, size)."""
    if size <= 0:
        return
    generator = random.Random(seed)
    offset = generator.randrange(size)
    if size == 1:
        yield 0
        return
    step = generator.randrange(1, size)
    while math.gcd(step, size) != 1:
        step = generator.randrange(1, size)
    for rank in range(size):
        yield (offset + step * rank) % size


def _unrank_pair(rank: int, n: int) -> tuple[int, int]:
    """Zero-based lexicographic unordered-pair rank, using integer arithmetic."""
    discriminant = (2 * n - 1) ** 2 - 8 * rank
    root = math.isqrt(discriminant)
    ceil_root = root + (root * root != discriminant)
    first = (2 * n - 1 - ceil_root) // 2
    first_rank = first * (2 * n - first - 1) // 2
    return first, first + 1 + rank - first_rank


@dataclass(frozen=True)
class CandidateBatch:
    sequence: int
    # Codes: (observation_id,); Scales: (low_observation_id, high_observation_id).
    tasks: tuple[tuple[int, ...], ...]


@dataclass
class CandidateRun:
    """One-shot bounded sampler; Phase 3B owns scores and frozen rankings.

    batches() produces distinct task IDs in stable seeded order. After scoring,
    acknowledge_evaluated(batch, completed_tasks) counts *only* successfully
    evaluated distinct tasks. This avoids equating sampling with EIG computation.
    """
    construct_type: str
    construct_id: int
    definition_version_id: int
    definition_version_number: int
    eligibility_fingerprint: str
    eligibility_context: dict[str, Any]
    seed: int
    requested_budget: int
    sampling_strategy: str
    possible_task_count: int
    eligible_observation_count: int
    source_fit_id: int | None
    source_configuration_id: int | None
    batch_size: int
    _ids: array = field(repr=False)
    _database: Any = field(repr=False)
    sampled_count: int = 0
    evaluated_count: int = 0
    inspected_task_count: int = 0
    exhausted: bool = False
    unavailable_reason: str | None = None
    _started: bool = field(default=False, repr=False)
    _pending_batch: CandidateBatch | None = field(default=None, repr=False)
    _acknowledged: bool = field(default=False, repr=False)

    @property
    def effective_limit(self) -> int:
        if self.requested_budget == 0:
            return self.possible_task_count
        return min(self.requested_budget, self.possible_task_count)

    def provenance(self) -> dict[str, Any]:
        """Attach to a scorer's persisted evaluated identities and rankings."""
        return {
            "construct_type": self.construct_type,
            "construct_id": self.construct_id,
            "definition_version_id": self.definition_version_id,
            "definition_version_number": self.definition_version_number,
            "eligibility_fingerprint": self.eligibility_fingerprint,
            "eligibility_context": self.eligibility_context,
            "random_seed": self.seed,
            "requested_budget": self.requested_budget,
            "sampling_strategy": self.sampling_strategy,
            "possible_task_count": self.possible_task_count,
            "sampled_candidate_count": self.sampled_count,
            "evaluated_candidate_count": self.evaluated_count,
            "inspected_task_count": self.inspected_task_count,
            "exhausted": self.exhausted,
            "unavailable_reason": self.unavailable_reason,
            "source_fit_id": self.source_fit_id,
            "source_configuration_id": self.source_configuration_id,
        }

    def acknowledge_evaluated(
        self, batch: CandidateBatch,
        completed_tasks: Sequence[tuple[int, ...]] | None = None,
    ) -> None:
        """Credit a batch once, including an explicitly partial scorer result."""
        if batch is not self._pending_batch or self._acknowledged:
            raise ValueError("Only the current, unacknowledged candidate batch may be credited.")
        tasks = list(batch.tasks if completed_tasks is None else completed_tasks)
        if len(set(tasks)) != len(tasks) or not set(tasks).issubset(batch.tasks):
            raise ValueError("Evaluated tasks must be distinct members of the sampled batch.")
        self.evaluated_count += len(tasks)
        self._acknowledged = True

    def batches(self) -> Iterator[CandidateBatch]:
        if self._started:
            raise RuntimeError("A candidate run can only be consumed once.")
        self._started = True
        limit = self.effective_limit
        if limit == 0:
            self.exhausted = True
            self.unavailable_reason = (
                "At least two eligible observations are required."
                if self.construct_type == "scale" else "No eligible unjudged observations."
            )
            return

        buffer: list[tuple[int, ...]] = []
        sequence = 0
        permutation = _permutation(self.possible_task_count, self.seed)
        # Each SQL exclusion check contains at most MAX_BATCH_SIZE canonical pairs.
        for positions_start in range(0, self.possible_task_count, self.batch_size):
            count = min(self.batch_size, self.possible_task_count - positions_start)
            ranks = [next(permutation) for _ in range(count)]
            self.inspected_task_count += count
            if self.construct_type == "code":
                accepted = [(int(self._ids[i]),) for i in ranks]
            else:
                pairs = []
                for rank in ranks:
                    i, j = _unrank_pair(rank, len(self._ids))
                    low, high = sorted((int(self._ids[i]), int(self._ids[j])))
                    pairs.append((low, high))
                judged = self._database.scale_judged_candidate_pairs(
                    self.construct_id, pairs,
                )
                accepted = [pair for pair in pairs if pair not in judged]

            for task in accepted:
                if self.sampled_count + len(buffer) >= limit:
                    break
                buffer.append(task)
                if len(buffer) == self.batch_size or self.sampled_count + len(buffer) == limit:
                    batch = CandidateBatch(sequence, tuple(buffer))
                    self.sampled_count += len(buffer)
                    sequence += 1
                    buffer = []
                    self._pending_batch = batch
                    self._acknowledged = False
                    yield batch
            if self.sampled_count >= limit:
                # A finite budget does not claim that every candidate was inspected.
                self.exhausted = self.inspected_task_count >= self.possible_task_count
                return

        if buffer:
            batch = CandidateBatch(sequence, tuple(buffer))
            self.sampled_count += len(buffer)
            self._pending_batch = batch
            self._acknowledged = False
            yield batch
        self.exhausted = True
        if self.sampled_count == 0:
            self.unavailable_reason = (
                "Every eligible Scale pair has an existing judgment."
                if self.construct_type == "scale" else "No eligible unjudged observations."
            )


def prepare_candidate_run(
    project: Any, construct_id: int, *,
    budget: int | None = None, seed: int = 0, batch_size: int = 128,
    session_id: int | None = None,
    source_fit_id: int | None = None,
    source_configuration_id: int | None = None,
) -> CandidateRun:
    """Freeze the eligible atomic-ID population, then sample tasks lazily.

    Scale pools are O(N) IDs, not O(N**2) pairs. The costly exhaustive request
    is refused explicitly before enumerating any pairs when it exceeds the cap.
    Existing comparison history (all Scale versions, including bootstrap and
    unsure) is excluded in indexed bounded batches, not loaded into memory.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Random seed must be an integer.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise ValueError(f"Batch size must be between 1 and {MAX_BATCH_SIZE}.")
    construct = project.construct(int(construct_id))
    kind = str(construct["type"])
    selected_budget = resolve_candidate_budget(kind, budget)
    eligibility = (
        project.scale_eligibility(int(construct_id)) if kind == "scale"
        else {"code_recommendation_revision": project.database.recommendation_revision(int(construct_id))}
    )
    ids = project.database.acquisition_eligible_atomic_ids(
        kind, int(construct_id),
        eligibility_spec=(eligibility.get("eligibility") if kind == "scale" else None),
        session_id=session_id if kind == "code" else None,
    )
    n = len(ids)
    possible = n * (n - 1) // 2 if kind == "scale" else n
    if kind == "scale" and selected_budget == 0 and possible > MAX_EXHAUSTIVE_SCALE_PAIRS:
        raise ValueError(
            f"Exhaustive Scale acquisition would examine up to {possible:,} pairs "
            f"from {n:,} eligible observations (before judged-pair exclusions). "
            f"This exceeds the {MAX_EXHAUSTIVE_SCALE_PAIRS:,}-pair safety limit. "
            "Choose a finite candidate-evaluation budget; no random subset "
            "has been substituted for Exhaustive."
        )
    # Explicitly encode the population and eligibility dependencies in the
    # fingerprint, so identical seeds are only comparable on identical states.
    digest = hashlib.sha256()
    context_bytes = json.dumps(
        {"construct": int(construct_id), "version_id": int(construct["version_id"]),
         "eligibility": eligibility, "session_id": session_id if kind == "code" else None},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    digest.update(context_bytes)
    for observation_id in ids:
        digest.update(int(observation_id).to_bytes(8, "little", signed=True))
    if kind == "scale":
        # Append-only event ID is a cheap revision token, including judgments
        # from older Scale definition versions and Code-to-Scale bootstrapping.
        digest.update(str(project.database.scale_judgment_revision(int(construct_id))).encode())
    return CandidateRun(
        construct_type=kind,
        construct_id=int(construct_id),
        definition_version_id=int(construct["version_id"]),
        definition_version_number=int(construct["version_number"]),
        eligibility_fingerprint=digest.hexdigest(),
        eligibility_context=json.loads(json.dumps(eligibility, sort_keys=True)),
        seed=seed, requested_budget=selected_budget,
        sampling_strategy="seeded_affine_task_permutation_v1",
        possible_task_count=possible, eligible_observation_count=n,
        source_fit_id=source_fit_id,
        source_configuration_id=source_configuration_id,
        batch_size=batch_size, _ids=ids, _database=project.database,
    )
