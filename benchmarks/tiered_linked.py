"""Tiered GeCo backend read benchmarks: A 1k, B 50k, C 300k.

Run: uv run python -m benchmarks.tiered_linked --tier a --output /tmp/a.json
Real linked mode is opt-in and does not call GeCo or TeAL mutation APIs.
This is NOT a browser-latency, process-RSS, or full TeAL synthetic I/O test.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import random
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from benchmarks.tiered_report import environment, measure, save_report, skipped
from geometric_coder import GeometricCoder
from geometric_coder.topic_artifacts import ArtifactRows

TIERS = {"a": 1_000, "b": 50_000, "c": 300_000}
DOCUMENT_SIZE = 16


def _absolute_positions(keys: list[dict[str, Any]]) -> np.ndarray:
    return np.fromiter(
        (int(key["document"]) * DOCUMENT_SIZE + int(key["sentence"]) for key in keys),
        dtype=np.int64,
        count=len(keys),
    )


class SyntheticProvider:
    """Keyed in-memory math, never an approximation of TeAL artifact I/O."""

    def geometry_matrix(self, external_ref: Any, keys: list[dict[str, Any]]) -> np.ndarray:
        return self.view_coordinates(external_ref, keys)

    def view_coordinates(self, external_ref: Any, keys: list[dict[str, Any]]) -> np.ndarray:
        positions = _absolute_positions(keys)
        return np.column_stack((np.sin(positions / 101), np.cos(positions / 127)))

    def read_artifact(
        self, artifact_id: str, requested_keys: list[dict[str, Any]] | None,
    ) -> ArtifactRows:
        if requested_keys is None:
            raise AssertionError("Benchmark never requests unbounded artifact rows")
        indices = _absolute_positions(requested_keys)
        topics = indices % 16
        if artifact_id == "synthetic_membership":
            values = sparse.csr_matrix(
                (np.ones(len(indices)), (np.arange(len(indices)), topics)),
                shape=(len(indices), 16),
            )
            return ArtifactRows(
                kind="matrix", values=values,
                columns=[f"topic_{i}" for i in range(16)],
                positions=list(range(len(indices))),
            )
        if artifact_id == "synthetic_argmax":
            return ArtifactRows(
                kind="table",
                values=pd.DataFrame({
                    "max_feature": [f"topic_{i}" for i in topics],
                    "max_value": np.ones(len(indices)),
                }),
                columns=["max_feature", "max_value"],
                positions=list(range(len(indices))),
            )
        raise KeyError(artifact_id)


def synthetic_project(directory: Path, units: int) -> GeometricCoder:
    if units <= 0:
        raise ValueError("units must be positive")
    data = pd.DataFrame({
        "document": [i // DOCUMENT_SIZE for i in range(units)],
        "sentence": [i % DOCUMENT_SIZE for i in range(units)],
        "text": [f"Example sentence {i}" for i in range(units)],
        "category": ["A" if i % 3 == 0 else "B" for i in range(units)],
    })
    project = GeometricCoder.create_external(
        project_dir=directory, data=data, keys=["document", "sentence"],
        text="text", metadata=["category"], external_provider=SyntheticProvider(),
    )
    gid = project.register_external_geometry(
        name="synthetic2d", external_ref={"artifact_id": "synthetic_geom"},
    )
    project.register_external_view(
        geometry_id=gid, name="synthetic_view",
        external_ref={"artifact_id": "synthetic_view"},
    )
    project.register_topic_model(
        name="synthetic_topic_model",
        membership_artifact_id="synthetic_membership",
        dominant_artifact_id="synthetic_argmax",
    )
    return project


def _json_bytes(value: Any) -> int:
    return len(json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def exercise(
    project: GeometricCoder, *,
    tier: str, seed: int = 1729, page_size: int = 64,
    repeats: int = 5, synthetic: bool = False,
    filter_field: str | None = None, filter_value: str | None = None,
) -> dict[str, Any]:
    """One bounded operation family per tier; all oracles outside timing."""
    if page_size < 1 or page_size > 5_000:
        raise ValueError("page_size must be between 1 and 5000")
    units = project.database.unit_count()
    if units < 1:
        raise ValueError("empty project")
    positions = random.Random(seed).sample(range(units), min(units, page_size))
    db = project.database
    operations: dict[str, dict[str, Any]] = {}

    def check_keys(rows: Any) -> None:
        assert isinstance(rows, list) and len(rows) == len(positions)
        if synthetic:
            assert rows == [
                {"document": p // DOCUMENT_SIZE, "sentence": p % DOCUMENT_SIZE}
                for p in positions
            ]

    operations["sqlite_key_page"] = measure(
        "sqlite_key_page", lambda: db.unit_keys_at_positions(positions),
        check_keys,
        lambda rows: {"row_count": len(rows), "serialized_result_bytes_local": _json_bytes(rows)},
        warm_repeats=repeats,
    )

    def check_rows(rows: Any) -> None:
        assert [int(row["row_position"]) for row in rows] == positions
        if synthetic:
            assert [row["text"] for row in rows] == [
                f"Example sentence {p}" for p in positions
            ]
            assert [row["metadata"]["category"] for row in rows] == [
                "A" if p % 3 == 0 else "B" for p in positions
            ]

    operations["sqlite_text_page"] = measure(
        "sqlite_text_page", lambda: db.unit_rows_at_positions(positions),
        check_rows,
        lambda rows: {"row_count": len(rows), "serialized_result_bytes_local": _json_bytes(rows)},
        warm_repeats=repeats,
    )

    hierarchy = db.hierarchy_levels()
    bounded_context = "max_units" in inspect.signature(project.context).parameters
    if not hierarchy:
        operations["context_page"] = skipped("context_page", "no_hierarchy")
    elif not synthetic and not bounded_context:
        operations["context_page"] = skipped(
            "context_page", "unbounded_context_on_real_workspace_is_unsafe",
        )
    else:
        focal = db.unit_rows_at_positions([positions[0]])[0]
        focal_id = int(focal["unit_id"])
        kwargs = {"level": int(hierarchy[0]["level_index"]), "window": 1}
        if bounded_context:
            kwargs["max_units"] = 200
        max_rows = 200 if bounded_context else DOCUMENT_SIZE

        def check_context(rows: Any) -> None:
            assert 1 <= len(rows) <= max_rows
            assert focal_id in {int(row["unit_id"]) for row in rows}
            if synthetic:
                assert {int(row["row_position"]) // DOCUMENT_SIZE for row in rows} == {
                    positions[0] // DOCUMENT_SIZE
                }

        operations["context_page"] = measure(
            "context_page", lambda: project.context(focal_id, **kwargs),
            check_context,
            lambda rows: {
                "row_count": len(rows), "serialized_result_bytes_local": _json_bytes(rows),
            },
            warm_repeats=repeats,
        )

    if filter_field is None or filter_value is None:
        operations["filtered_sql_page"] = skipped(
            "filtered_sql_page", "no_categorical_filter_configured",
        )
    else:
        clauses = [{
            "field": filter_field, "type": "categorical",
            "operator": "in", "value": [filter_value],
        }]

        def check_filtered(result: Any) -> None:
            page, count, page_number = result
            assert page_number == 1 and len(page) <= page_size
            assert list(page) == sorted(set(page))
            assert all(0 <= p < units for p in page)
            assert 0 <= count <= units
            if synthetic:
                assert count == (units + 2) // 3
                assert page == list(range(0, units, 3))[:page_size]

        operations["filtered_sql_page"] = measure(
            "filtered_sql_page",
            lambda: db.filtered_unit_page(clauses, page_size=page_size, page_number=1),
            check_filtered,
            lambda result: {
                "row_count": len(result[0]), "matching_rows": result[1],
                "serialized_result_bytes_local": _json_bytes(result),
            },
            warm_repeats=repeats,
        )

    views = project.views()
    if views:
        view_id = int(views[0]["view_id"])

        def check_view(rows: Any) -> None:
            matrix = np.asarray(rows)
            assert matrix.ndim == 2 and matrix.shape[0] == len(positions)
            assert np.isfinite(matrix).all()
            if synthetic:
                np.testing.assert_allclose(
                    matrix,
                    np.column_stack((
                        np.sin(np.asarray(positions) / 101),
                        np.cos(np.asarray(positions) / 127),
                    )),
                    rtol=1e-5, atol=1e-6,
                )

        operations["view_coordinate_page"] = measure(
            "view_coordinate_page",
            lambda: project.view_coordinate_rows(view_id, positions),
            check_view,
            lambda rows: {
                "row_count": len(rows),
                "dimensions": int(np.asarray(rows).shape[1]),
                "serialized_result_bytes_local": _json_bytes(np.asarray(rows).tolist()),
            },
            warm_repeats=repeats,
        )
    else:
        operations["view_coordinate_page"] = skipped(
            "view_coordinate_page", "no_registered_legacy_view",
        )

    topic_models = project.topic_models()
    if topic_models:
        model_id = int(topic_models[0]["topic_model_id"])

        def check_topic(rows: Any, requested: list[int]) -> None:
            assert len(rows.matched_positions) <= len(requested)
            assert tuple(sorted(set(rows.matched_positions))) == rows.matched_positions
            assert all(0 <= p < len(requested) for p in rows.matched_positions)
            assert rows.memberships.shape == (
                len(rows.matched_positions), len(rows.topic_ids)
            )
            assert len(rows.dominant) == len(rows.matched_positions)
            if synthetic:
                assert rows.matched_positions == tuple(range(len(requested)))
                assert rows.dominant["max_feature"].tolist() == [
                    f"topic_{p % 16}" for p in requested
                ]

        for name, requested in (
            ("topic_provider_page", positions), ("topic_provider_selector", positions[:1]),
        ):
            operations[name] = measure(
                name,
                lambda p=requested: project.topic_model_rows(model_id, positions=p),
                lambda rows, p=requested: check_topic(rows, p),
                lambda rows: {
                    "row_count": len(rows.matched_positions),
                    "topic_count": len(rows.topic_ids),
                    "serialized_result_bytes_local": None,
                    "payload_note": "provider packet is not a Dash JSON payload",
                },
                warm_repeats=repeats,
            )
    else:
        for name in ("topic_provider_page", "topic_provider_selector"):
            operations[name] = skipped(name, "no_registered_topic_model")

    return {"tier": tier, "units": units, "seed": seed, "page_size": page_size,
            "warm_repeats": repeats, "operations": operations}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.repeats < 1 or args.units is not None and args.units < 1:
        raise ValueError("repeats and units must be positive")
    if bool(args.geco_path) != bool(args.teal_path):
        raise ValueError("real mode requires both --geco-path and --teal-path")
    requested_units = args.units if args.units is not None else TIERS[args.tier]
    with ExitStack() as stack:
        started = time.perf_counter()
        if args.geco_path is not None:
            # No GeometricCoder.open() migration/index update on the real workspace.
            # The benchmark invokes only read methods; use snapshots if the installed
            # TeAL version performs any opening-time cache/migration activity.
            from text_analysis_lab.core.project import Project
            from text_analysis_lab.integrations.geco import TeALGeCoProvider

            teal = Project.open(args.teal_path)
            stack.callback(teal.close)
            project = GeometricCoder(
                args.geco_path, external_provider=TeALGeCoProvider(teal),
            )
            mode = "real_teal_linked"
            field, value = args.filter_field, args.filter_value
            dataset_id = args.dataset_id
        else:
            root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            project = synthetic_project(root / "tiered.geco", requested_units)
            mode = "synthetic_external_provider_no_teal_io"
            field = args.filter_field or "category"
            value = args.filter_value if args.filter_value is not None else "A"
            dataset_id = hashlib.sha256(
                f"synthetic-v1:{requested_units}:{args.seed}".encode()
            ).hexdigest()[:16]
        setup_seconds = round(time.perf_counter() - started, 5)
        outcome = exercise(
            project, tier=args.tier, seed=args.seed, page_size=args.page_size,
            repeats=args.repeats, synthetic=args.geco_path is None,
            filter_field=field, filter_value=value,
        )
    return {
        "schema_version": 1,
        "environment": environment(),
        "fixture": {
            "mode": mode, "dataset_id": dataset_id, "seed": args.seed,
            "actual_units": outcome["units"],
            "fixture_setup_seconds_excluded": setup_seconds,
            "synthetic_fixture_version": "v1" if args.geco_path is None else None,
        },
        "execution": outcome,
        "cold_process": {"status": "not_measured", "reason": "requires isolated process"},
        "limitations": [
            "Backend API calls only; no browser paint or click-to-visible timing.",
            "Local JSON serialization is not Dash HTTP or network byte measurement.",
            "Synthetic external provider does not exercise TeAL DuckDB/artifact I/O.",
            "tracemalloc reports per-operation Python allocations, not process RSS.",
            "Timing includes tracemalloc overhead; do not use noisy CI speed gates.",
            "Compare only matching revisions, machines, environments and datasets.",
            "For real projects, use snapshots to guard against provider opening-time writes.",
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=tuple(TIERS), default="a")
    parser.add_argument("--units", type=int, help="override tier fixture size")
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5, help="warm samples; >=20 for p95")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--geco-path", type=Path)
    parser.add_argument("--teal-path", type=Path)
    parser.add_argument("--dataset-id", help="explicit real dataset identity for comparisons")
    parser.add_argument("--filter-field", help="real project categorical field")
    parser.add_argument("--filter-value", help="real project categorical value")
    parser.add_argument("--output", type=Path, help="JSON destination (default stdout only)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print(save_report(run(args), args.output), end="", flush=True)


if __name__ == "__main__":
    main()
