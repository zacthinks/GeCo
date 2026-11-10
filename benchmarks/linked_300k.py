"""Repeatable read-only 300k-unit GeCo scalability benchmark.

Examples
--------
uv run python benchmarks/linked_300k.py --units 300000
uv run python benchmarks/linked_300k.py --geco-path /path/to/workspace.geco --teal-path /path/to/teal --output results.json

Synthetic mode uses an external provider with deterministic in-memory arrays to
isolate GeCo's SQLite and UI behavior. Real linked mode uses TeALGeCoProvider
and measures DuckDB/key resolution and artifact I/O too. Real linked mode never
changes codes, sessions, annotations, artifacts, or notebook data.
"""
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
import tracemalloc
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from geometric_coder import GeometricCoder
from geometric_coder.topic_artifacts import ArtifactRows
from geometric_coder.ui.app import create_app


class SyntheticProvider:
    """Deterministic, subset-resolving external source; no full-array copy."""

    def geometry_matrix(self, external_ref: Any, keys: list[dict[str, Any]]) -> np.ndarray:
        indices = np.fromiter(
            (int(key["sentence"]) for key in keys), dtype=np.int64,
        )
        return np.column_stack((np.sin(indices / 101), np.cos(indices / 127)))

    def view_coordinates(self, external_ref: Any, keys: list[dict[str, Any]]) -> np.ndarray:
        return self.geometry_matrix(external_ref, keys)

    def read_artifact(
        self, artifact_id: str, requested_keys: list[dict[str, Any]] | None,
    ) -> ArtifactRows:
        if requested_keys is None:
            raise ValueError("Synthetic benchmark does not include descriptions")
        idx = np.fromiter(
            (int(key["sentence"]) for key in requested_keys),
            dtype=np.int64,
        )
        topics = idx % 16
        if artifact_id == "synthetic_membership":
            return ArtifactRows(
                kind="matrix",
                values=sparse.csr_matrix(
                    (np.ones(len(idx)), (np.arange(len(idx)), topics)),
                    shape=(len(idx), 16),
                ),
                columns=[f"topic_{j}" for j in range(16)],
                positions=list(range(len(idx))),
            )
        if artifact_id == "synthetic_argmax":
            return ArtifactRows(
                kind="table",
                values=pd.DataFrame({
                    "max_feature": [f"topic_{j}" for j in topics],
                    "max_value": np.ones(len(idx)),
                }),
                columns=["max_feature", "max_value"],
                positions=list(range(len(idx))),
            )
        raise KeyError(artifact_id)


def synthetic_project(directory: Path, count: int) -> GeometricCoder:
    data = pd.DataFrame({
        "document": ["large"] * count,
        "sentence": np.arange(count),
        "text": [f"Example sentence {i}" for i in range(count)],
        "category": ["A" if i % 3 == 0 else "B" for i in range(count)],
    })
    provider = SyntheticProvider()
    project = GeometricCoder.create_external(
        project_dir=directory,
        data=data, keys=["document", "sentence"], text="text",
        metadata=["category"], external_provider=provider,
    )
    geometry_id = project.register_external_geometry(
        name="synthetic2d", external_ref={"artifact_id": "synthetic_geom"},
    )
    project.register_external_view(
        geometry_id=geometry_id, name="synthetic_view",
        external_ref={"artifact_id": "synthetic_view"},
    )
    project.register_topic_model(
        name="synthetic_topic_model",
        membership_artifact_id="synthetic_membership",
        dominant_artifact_id="synthetic_argmax",
    )
    project.create_code("benchmark")
    project.create_session("benchmark")
    return project


def timed(label: str, fn: Any, repeats: int = 3) -> dict[str, Any]:
    samples: list[float] = []
    detail: Any = None
    for _ in range(repeats):
        start = time.perf_counter()
        detail = fn()
        samples.append(round(time.perf_counter() - start, 5))
    print(f"{label:32} median={statistics.median(samples):.4f}s   {samples}", flush=True)
    return {
        "seconds": samples,
        "median_seconds": statistics.median(samples),
        "result_size": (
            len(detail) if hasattr(detail, "__len__")
            else None
        ),
    }


def exercise(project: GeometricCoder, *, page_size: int, repeats: int,
             filter_field: str | None, filter_value: str | None) -> dict[str, Any]:
    result: dict[str, Any] = {"units": project.database.unit_count()}
    n = result["units"]
    if n == 0:
        raise RuntimeError("Empty workspace is not a valid large-scale benchmark")
    positions = [int(x) for x in np.linspace(
        0, n - 1, num=min(n, page_size), dtype=np.int64,
    )]
    rows = project.database.unit_rows_at_positions(positions[:1])
    focal_unit_id = int(rows[0]["unit_id"])
    result["unit_key_page"] = timed(
        "SQLite selected keys",
        lambda: project.database.unit_keys_at_positions(positions),
        repeats=repeats,
    )
    result["unit_text_page"] = timed(
        "SQLite selected text/metadata",
        lambda: project.database.unit_rows_at_positions(positions),
        repeats=repeats,
    )

    hierarchy = project.database.hierarchy_levels()
    if hierarchy:
        level = int(hierarchy[0]["level_index"])
        result["context_200"] = timed(
            "Context (max 200 units)",
            lambda: project.context(focal_unit_id, level=level, window=1, max_units=200),
            repeats=repeats,
        )

    codes = project.codes()
    sessions = project.sessions()
    if codes and sessions:
        code = int(codes[0]["code_id"])
        session = int(sessions[0]["session_id"])
        result["random"] = timed(
            "Develop random selection",
            lambda: project.database.random_available_unit(
                code_id=code, session_id=session,
            ),
            repeats=repeats,
        )
    if filter_field and filter_value is not None:
        clauses = [{
            "field": filter_field, "type": "categorical",
            "operator": "in", "value": [filter_value],
        }]
        result["filter_count"] = timed(
            "Filtered SQLite count",
            lambda: project.database.filtered_unit_count(clauses),
            repeats=repeats,
        )
        result["filter_first_page"] = timed(
            "Filtered SQL page",
            lambda: project.database.filtered_unit_page(
                clauses, page_size=page_size, page_number=1,
            ),
            repeats=repeats,
        )
        result["filter_result_index"] = timed(
            "Cached-position source query",
            lambda: project.database.filtered_unit_positions(clauses),
            repeats=repeats,
        )

    views = project.views()
    if views:
        view_id = int(views[0]["view_id"])
        result["view_page"] = timed(
            "Externally backed view page",
            lambda: project.view_coordinate_rows(view_id, positions),
            repeats=repeats,
        )
    models = project.topic_models()
    if models:
        model_id = int(models[0]["topic_model_id"])
        result["topic_page"] = timed(
            "Externally backed topic page",
            lambda: project.topic_model_rows(model_id, positions=positions),
            repeats=repeats,
        )
        result["topic_selector"] = timed(
            "Topic selector single row",
            lambda: project.topic_model_rows(model_id, positions=[0]),
            repeats=repeats,
        )

    result["dash_setup"] = timed(
        "Dash app initialization",
        lambda: create_app(project),
        repeats=1,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geco-path", type=Path)
    parser.add_argument("--teal-path", type=Path)
    parser.add_argument("--units", type=int, default=300_000)
    parser.add_argument("--page-size", type=int, default=5_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--filter-field")
    parser.add_argument("--filter-value")
    parser.add_argument("--output", type=Path, default=Path("linked_300k_results.json"))
    args = parser.parse_args()
    if bool(args.geco_path) != bool(args.teal_path):
        parser.error("--geco-path and --teal-path must be provided together")
    tracemalloc.start()
    with ExitStack() as stack:
        if args.geco_path is not None:
            from text_analysis_lab.core.project import Project
            from text_analysis_lab.integrations.geco import TeALGeCoProvider
            teal_project = stack.enter_context(Project.open(args.teal_path))
            project = GeometricCoder.open(
                args.geco_path, external_provider=TeALGeCoProvider(teal_project),
            )
            field = args.filter_field
            value = args.filter_value
            mode = "real_teal_linked"
        else:
            directory = Path(stack.enter_context(tempfile.TemporaryDirectory())) / "synthetic.geco"
            setup_start = time.perf_counter()
            project = synthetic_project(directory, args.units)
            print(f"Synthetic workspace setup: {time.perf_counter()-setup_start:.2f}s")
            field = args.filter_field or "category"
            value = args.filter_value if args.filter_value is not None else "A"
            mode = "synthetic_external_provider"
        timings = exercise(
            project, page_size=args.page_size, repeats=args.repeats,
            filter_field=field, filter_value=value,
        )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    report = {
        "mode": mode, "page_size": args.page_size,
        "repeats": args.repeats, "peak_python_tracemalloc_bytes": peak,
        "timings": timings,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {args.output.resolve()} (Python peak {peak/1048576:.1f} MiB)")


if __name__ == "__main__":
    main()
