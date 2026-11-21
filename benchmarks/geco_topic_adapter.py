#!/usr/bin/env python3
"""Persisted linked-TeAL GeCo topic read benchmark.

The A/B comparison exercises the same GeometricCoder.topic_model_rows calls,
against the same persisted project, with a legacy single-read provider wrapper
and the new TeAL batch-capable provider. No mocked artifact values are used.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse

from geometric_coder.topic_artifacts import TopicModelRows


class LegacyProvider:
    """Expose only the original provider capability while delegating real reads."""

    def __init__(self, provider):
        self._provider = provider
        self.calls = 0

    def read_artifact(self, artifact_id, keys):
        self.calls += 1
        return self._provider.read_artifact(artifact_id, keys)


def seed(project, artifact_id, values, columns, *, keys):
    from text_analysis_lab.core.writer import create_artifact_writer

    if isinstance(values, pd.DataFrame):
        kind = "table"
    elif sparse.issparse(values):
        kind = "sparse_matrix"
    else:
        kind = "dense_matrix"
    writer = create_artifact_writer(
        artifact_type=kind,
        artifact_dir=project.storage.artifact_dir(artifact_id),
        artifact_id=artifact_id,
        label=artifact_id,
        lineage_mode="new_key",
        basis_artifact_ids=(),
    )
    writer.write({"keys": keys, "data": (
        values if kind == "table" else {"values": values, "columns": columns}
    )})
    writer.finalize()
    project.catalog.register_artifact(
        artifact_id=artifact_id, artifact_type=kind, label=artifact_id,
        lineage_mode="new_key", status="complete", basis_artifact_ids=(),
    )
    return project.get_artifact(artifact_id)


def create_fixture(root, row_count):
    import text_analysis_lab as teal
    from text_analysis_lab.translators.argmax import ArgMax

    rng = np.random.default_rng(20261121)
    project = teal.Project.create(root, name="linked_topic_adapter_benchmark")
    try:
        keys = pd.DataFrame({"row_id": np.arange(row_count, dtype=np.int64)})
        docs = seed(
            project, "benchmark_documents",
            pd.DataFrame({"text": ["sample topic document"] * row_count}),
            ["text"], keys=keys,
        )
        columns = [f"topic_{i}" for i in range(8)]
        raw = rng.random((row_count, len(columns)), dtype=np.float32)
        raw /= raw.sum(axis=1, keepdims=True)
        memberships = {
            "dense": seed(project, "membership_dense", raw, columns, keys=keys),
            "sparse": seed(
                project, "membership_sparse",
                sparse.csr_matrix(raw * (raw >= 0.14)), columns, keys=keys,
            ),
        }
        dominant = {
            name: project.translate(ArgMax(), matrix)["output"]
            for name, matrix in memberships.items()
        }
        descriptions = seed(
            project, "topic_descriptions",
            rng.random((len(columns), 6), dtype=np.float32),
            [f"word_{i}" for i in range(6)],
            keys=pd.DataFrame({"topic_id": np.arange(len(columns), dtype=np.int64)}),
        )
        linked = project.geco.create(
            "linked", documents=docs, text_field="text",
            topic_models={
                name: {"membership": matrix, "dominant": dominant[name],
                       "description": descriptions}
                for name, matrix in memberships.items()
            },
        )
        linked.close()
    finally:
        project.close()


def equal_rows(a: TopicModelRows, b: TopicModelRows) -> None:
    assert a.topic_model_id == b.topic_model_id
    assert a.topic_ids == b.topic_ids
    assert a.matched_positions == b.matched_positions
    assert a.topic_labels == b.topic_labels
    assert sparse.issparse(a.memberships) == sparse.issparse(b.memberships)
    if sparse.issparse(a.memberships):
        assert (a.memberships != b.memberships).nnz == 0
    else:
        np.testing.assert_array_equal(a.memberships, b.memberships)
    pd.testing.assert_frame_equal(a.dominant, b.dominant, check_exact=True)
    assert (a.descriptions is None) == (b.descriptions is None)
    if a.descriptions is not None:
        assert a.descriptions.kind == b.descriptions.kind
        assert list(a.descriptions.columns) == list(b.descriptions.columns)
        assert a.descriptions.row_ids == b.descriptions.row_ids
        np.testing.assert_array_equal(a.descriptions.values, b.descriptions.values)


def measure(coder, provider, topic, positions):
    coder.external_provider = provider
    start = time.perf_counter_ns()
    rows = coder.topic_model_rows(
        topic, positions=positions, include_description=True,
    )
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    return rows, elapsed_ms


def run(root, row_count, repetitions):
    import text_analysis_lab as teal

    project = teal.Project.open(root)
    try:
        with project.geco.open("linked") as linked:
            coder = linked.coder
            provider = linked.external_provider
            assert provider is not None
            assert callable(getattr(provider, "read_artifacts", None))
            legacy = LegacyProvider(provider)
            sessions = {"opens": 0}
            original_session = provider._project_session

            @contextmanager
            def count_session():
                sessions["opens"] += 1
                with original_session() as active:
                    yield active

            provider._project_session = count_session
            cases = []
            for size in (1, 100, 5000, row_count):
                if size > row_count:
                    continue
                # Exact same keys and GeCo position order on each A/B call.
                positions = np.random.default_rng(311 + size).choice(
                    row_count, size=size, replace=False
                ).tolist()
                for topic in ("dense", "sparse"):
                    # First-use times are recorded separately; they are not
                    # fresh-process cold starts or evidence of cold speedup.
                    before = sessions["opens"]
                    reference, sequential_first = measure(coder, legacy, topic, positions)
                    sequential_first_sessions = sessions["opens"] - before
                    before = sessions["opens"]
                    actual, batched_first = measure(coder, provider, topic, positions)
                    batched_first_sessions = sessions["opens"] - before
                    equal_rows(reference, actual)
                    assert sequential_first_sessions == 3, sequential_first_sessions
                    assert batched_first_sessions == 1, batched_first_sessions

                    samples = {"sequential": [], "batched": []}
                    for iteration in range(repetitions):
                        order = ("sequential", "batched") if iteration % 2 == 0 else (
                            "batched", "sequential"
                        )
                        for variant in order:
                            active = legacy if variant == "sequential" else provider
                            before = sessions["opens"]
                            observed, duration = measure(coder, active, topic, positions)
                            equal_rows(reference, observed)
                            expected_sessions = 3 if variant == "sequential" else 1
                            assert sessions["opens"] - before == expected_sessions
                            samples[variant].append(duration)
                    a = statistics.median(samples["sequential"])
                    b = statistics.median(samples["batched"])
                    cases.append({
                        "topic": topic, "keys": size, "repetitions": repetitions,
                        "first_use_ms": {
                            "sequential": sequential_first, "batched": batched_first,
                        },
                        "provider_sessions_per_call": {"sequential": 3, "batched": 1},
                        "warm_ms": samples,
                        "median_ms": {"sequential": a, "batched": b},
                        "median_gain_percent": 100 * (a - b) / a,
                        "covered_rows": len(reference.matched_positions),
                        "topics": len(reference.topic_ids),
                        "equal_results": True,
                    })
            return cases
    finally:
        project.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=50000)
    parser.add_argument("--repetitions", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("topic-adapter-results.json"))
    parser.add_argument("--project", type=Path, default=Path("topic-adapter-project"))
    args = parser.parse_args()
    if args.rows < 5000 or args.repetitions < 2:
        parser.error("rows must be >=5000 and repetitions >=2")
    create_fixture(args.project, args.rows)
    cases = run(args.project, args.rows, args.repetitions)
    output = {
        "benchmark": "GeCo topic_model_rows: sequential versus optional batched TeAL reads",
        "rows": args.rows, "python": sys.version, "platform": platform.platform(),
        "note": "First-use is same-process, not fresh-interpreter cold startup.",
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    for case in cases:
        print(
            f"{case['topic']} keys={case['keys']}: "
            f"sequential={case['median_ms']['sequential']:.2f}ms, "
            f"batch={case['median_ms']['batched']:.2f}ms, "
            f"gain={case['median_gain_percent']:.1f}%; "
            f"sessions=3->1; exact output match",
            flush=True,
        )


if __name__ == "__main__":
    main()
