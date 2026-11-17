"""Translate effective GeCo Scale judgments into the Davidson numerical contract.

This module does not implement a likelihood, optimizer, posterior or acquisition.
The estimator receives a compact geometry-row matrix with local row-index pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy import sparse

from geometric_coder.geometry.base import Matrix


# The Phase 0B numerical contract: first higher, parity, second higher.
_OUTCOME_CODES = {"low_higher": 0, "parity": 1, "high_higher": 2}


@dataclass(frozen=True)
class ScaleTrainingData:
    """A compact, ordered representation of current human comparison evidence."""

    features: Matrix
    pairs: np.ndarray
    outcomes: np.ndarray
    observation_ids: tuple[int, ...]
    comparison_event_ids: tuple[int, ...]

    @property
    def n_comparisons(self) -> int:
        return len(self.comparison_event_ids)


def _atomic_positions(project: Any, observation_ids: Sequence[int]) -> dict[int, int]:
    """Resolve observation IDs, not unit IDs, to zero-based geometry row positions.

    Never assume an observation's durable ID equals a corpus row or unit ID.
    Resolve only judged members, in bounded SQL batches without reading text.
    """
    unique = list(dict.fromkeys(int(value) for value in observation_ids))
    positions: dict[int, int] = {}
    with project.database.connect() as connection:
        for offset in range(0, len(unique), 900):
            batch = unique[offset:offset + 900]
            if not batch:
                continue
            placeholders = ",".join("?" for _ in batch)
            rows = connection.execute(
                "SELECT observation_id, row_position FROM units "
                f"WHERE observation_id IN ({placeholders})",
                batch,
            ).fetchall()
            positions.update(
                (int(row["observation_id"]), int(row["row_position"]))
                for row in rows
            )
    if len(positions) != len(unique):
        missing = set(unique) - set(positions)
        raise ValueError(f"Scale evidence references non-atomic or missing units: {sorted(missing)}")
    return positions


def build_scale_training_data(
    project: Any,
    scale_id: int,
    geometry_id: int,
    *,
    comparisons: Sequence[dict[str, Any]] | None = None,
) -> ScaleTrainingData:
    """Adapt current effective judgments, including repeats and parity, to rows.

    The pair array is expressed in canonical low-ID/high-ID orientation;
    outcome 0 means the low-ID observation is higher. Reversing the original
    A/B presentation therefore never reverses the measurement semantics.

    The feature reference consists of distinct currently compared observations,
    in the geometry's corpus order. Its row order is explicitly frozen by the
    fitted numerical estimator. No full-corpus text or dense sparse copy is made.
    """
    events = (
        project.scale_training_comparisons(int(scale_id))
        if comparisons is None else list(comparisons)
    )
    if not events:
        return ScaleTrainingData(
            features=np.empty((0, 0), dtype=np.float64),
            pairs=np.empty((0, 2), dtype=np.int64),
            outcomes=np.empty(0, dtype=np.int64),
            observation_ids=(),
            comparison_event_ids=(),
        )

    involved = [
        int(value)
        for event in events
        for value in (event["observation_low_id"], event["observation_high_id"])
    ]
    positions = _atomic_positions(project, involved)
    ordered_ids = tuple(sorted(positions, key=lambda obs: positions[obs]))
    local_rows = {observation_id: index for index, observation_id in enumerate(ordered_ids)}

    pair_rows: list[tuple[int, int]] = []
    outcomes: list[int] = []
    event_ids: list[int] = []
    for event in events:
        low, high = int(event["observation_low_id"]), int(event["observation_high_id"])
        if low >= high:
            raise ValueError("Scale event has invalid canonical observation order.")
        outcome = str(event["canonical_outcome"])
        if outcome not in _OUTCOME_CODES:
            raise ValueError(f"Unexpected training comparison outcome: {outcome}")
        pair_rows.append((local_rows[low], local_rows[high]))
        outcomes.append(_OUTCOME_CODES[outcome])
        event_ids.append(int(event["comparison_event_id"]))

    features = project.geometry_page_rows(
        int(geometry_id), [positions[value] for value in ordered_ids]
    )
    if features.ndim != 2 or features.shape[0] != len(ordered_ids):
        raise ValueError("Selected geometry rows do not match Scale observations.")
    numeric = features.data if sparse.issparse(features) else np.asarray(features)
    if not np.isfinite(numeric).all():
        raise ValueError("Scale training geometry contains non-finite values.")
    return ScaleTrainingData(
        features=features,
        pairs=np.asarray(pair_rows, dtype=np.int64),
        outcomes=np.asarray(outcomes, dtype=np.int64),
        observation_ids=ordered_ids,
        comparison_event_ids=tuple(event_ids),
    )
