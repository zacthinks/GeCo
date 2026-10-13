"""Phase 2A/2B pilot: inspect real Dash routes and sample backend workload.

This is diagnostic, not a performance assertion, browser-paint benchmark, or
small-laptop qualification. Python/SQLite work and HTTP response sizes are
reported separately. It never edits an existing research project.

Run:
    uv run python benchmarks/phase2a_runtime_pilot.py --sizes 1000 10000
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from geometric_coder import GeometricCoder
from geometric_coder.ui import create_app


def _sample(fn: Callable[[], Any], repeats: int) -> dict[str, Any]:
    wall: list[float] = []
    cpu: list[float] = []
    for _ in range(repeats):
        before_wall, before_cpu = time.perf_counter(), time.process_time()
        value = fn()
        assert value is not None
        wall.append((time.perf_counter() - before_wall) * 1000)
        cpu.append((time.process_time() - before_cpu) * 1000)
    ordered = sorted(wall)
    return {
        "first_ms": round(wall[0], 3),
        "median_ms": round(statistics.median(wall), 3),
        "p95_ms": round(ordered[max(0, int(np.ceil(0.95 * len(ordered))) - 1)], 3),
        "median_cpu_ms": round(statistics.median(cpu), 3),
        "samples": len(wall),
    }


def _http(client: Any, path: str) -> bytes:
    response = client.get(path)
    if response.status_code != 200:
        raise AssertionError(
            f"GET {path}: HTTP {response.status_code}: "
            f"{response.get_data(as_text=True)[:800]}"
        )
    return response.get_data()


def _count_ids(root: Any) -> dict[str, int]:
    totals = {"static": 0, "pattern": 0, "judgment": 0}

    def walk(item: Any) -> None:
        if isinstance(item, list):
            for part in item:
                walk(part)
        elif isinstance(item, dict):
            value = item.get("id")
            if isinstance(value, str):
                totals["static"] += 1
            elif isinstance(value, dict):
                totals["pattern"] += 1
                if value.get("type") == "focus-coding-judgment":
                    totals["judgment"] += 1
            for part in item.values():
                walk(part)

    walk(root)
    return totals


def _exercise_focus_render(app: Any, session_id: int, unit_id: int) -> bytes:
    """POST the genuine multi-output Dash render callback; no browser involved."""
    item = next(
        (key, value)
        for key, value in app.callback_map.items()
        if "focus-coding-progress-summary.children" in key
    )
    output_key, spec = item
    outputs = spec["output"]
    if not isinstance(outputs, (tuple, list)):
        outputs = [outputs]
    serialized_outputs = [
        {"id": output.component_id, "property": output.component_property}
        for output in outputs
    ]
    values = {
        "focus-coding-current-unit-store.data": unit_id,
        "focus-coding-refresh-store.data": 0,
        "focus-coding-session-store.data": session_id,
    }
    inputs = [
        {**x, "value": values[f"{x['id']}.{x['property']}"]}
        for x in spec["inputs"]
    ]
    state = [
        {**x, "value": values[f"{x['id']}.{x['property']}"]}
        for x in spec["state"]
    ]
    response = app.server.test_client().post(
        "/_dash-update-component",
        json={
            "output": output_key,
            "outputs": serialized_outputs,
            "inputs": inputs,
            "state": state,
            "changedPropIds": ["focus-coding-current-unit-store.data"],
        },
    )
    if response.status_code != 200:
        raise AssertionError(
            f"Focus render callback HTTP {response.status_code}: "
            f"{response.get_data(as_text=True)[:1000]}"
        )
    return response.get_data()


def _sql_plan(project: GeometricCoder) -> list[str]:
    with project.database.connect() as connection:
        result = connection.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM current_annotations "
            "ORDER BY observation_id, code_id"
        ).fetchall()
    return [str(row["detail"]) for row in result]


def _peak_rss_kib() -> int | None:
    if os.name == "nt":
        return None
    import resource
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if platform.system() == "Linux" else int(raw / 1024)


def _case(size: int, repeats: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    tokens = ("lesson", "teacher", "student", "dialogue", "reading", "feedback")
    texts = [
        " ".join(tokens[int(x)] for x in rng.integers(0, len(tokens), 10))
        for _ in range(size)
    ]
    with tempfile.TemporaryDirectory(prefix="geco-phase2a-") as root:
        project = GeometricCoder.create(
            project_dir=Path(root) / "pilot.geco",
            data=pd.DataFrame({
                "row_id": range(size),
                "text": texts,
                "group": [f"group_{i % 8}" for i in range(size)],
            }),
            keys=["row_id"],
            text="text",
            metadata=["group"],
        )
        code_a = project.create_code("reviewed")
        code_b = project.create_code("secondary")
        task_size = min(200, size)
        session_id = project.configure_focus(
            codes=[code_a, code_b],
            unit_ids=list(range(1, task_size + 1)),
            allow_unsure=False,
        )
        # Modest, deterministic existing coding activity. This intentionally
        # creates out-of-task annotations in the larger case, because the
        # current task-progress path reads the project-wide current view.
        annotate_count = min(500, size // 20)
        for unit in project.units()[:annotate_count]:
            project.annotate(
                int(unit["observation_id"]),
                code_a,
                "positive",
                origin="human_focus_coder",
            )

        backend: dict[str, Any] = {}
        backend["current_annotations"] = _sample(
            project.database.current_annotations, repeats
        )
        backend["focus_progress"] = _sample(
            lambda: project.focus_progress(session_id), repeats
        )
        state = dict(project.database.get_session(session_id)["state"])
        backend["session_state_sql_write"] = _sample(
            lambda: project.update_session_state(session_id, state) or True,
            repeats,
        )
        baseline_sql_plan = _sql_plan(project)
        std_initial = _sample(lambda: create_app(project), 1)
        app = create_app(project)
        client = app.server.test_client()
        standard_layout = _http(client, "/_dash-layout")
        standard_deps = _http(client, "/_dash-dependencies")
        standard_http = {
            "route_layout_bytes": len(standard_layout),
            "route_dependencies_bytes": len(standard_deps),
            "registered_callback_map_entries": len(app.callback_map),
            "dependency_records": len(json.loads(standard_deps)),
            "initial_layout_ids": _count_ids(json.loads(standard_layout)),
            "layout_http": _sample(lambda: _http(client, "/_dash-layout"), repeats),
            "dependencies_http": _sample(
                lambda: _http(client, "/_dash-dependencies"), repeats
            ),
        }
        focus_app = create_app(project, mode="focus", focus_session_id=session_id)
        focus_client = focus_app.server.test_client()
        focus_layout = _http(focus_client, "/_dash-layout")
        focus_deps = _http(focus_client, "/_dash-dependencies")
        focus_initial = _exercise_focus_render(focus_app, session_id, 1)
        focus_decoded = json.loads(focus_initial)
        focus_http = {
            "route_layout_bytes": len(focus_layout),
            "route_dependencies_bytes": len(focus_deps),
            "registered_callback_map_entries": len(focus_app.callback_map),
            "dependency_records": len(json.loads(focus_deps)),
            "initial_layout_ids": _count_ids(json.loads(focus_layout)),
            "render_response_bytes": len(focus_initial),
            "render_dynamic_ids": _count_ids(focus_decoded),
            "render_callback_http": _sample(
                lambda: _exercise_focus_render(focus_app, session_id, 1),
                repeats,
            ),
        }
        output = {
            "size": size,
            "task_units": task_size,
            "required_codes": 2,
            "annotation_count": annotate_count,
            "backend": backend,
            "standard_dash": standard_http,
            "focus_dash": focus_http,
            "current_annotations_query_plan": baseline_sql_plan,
            "process_peak_rss_kib": _peak_rss_kib(),
            "standard_app_build": std_initial,
        }
    gc.collect()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 10000])
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.sizes or min(args.sizes) < 30 or args.repeats < 2:
        parser.error("Each size must be >= 30, with at least two repeats.")
    result = {
        "runner": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "numpy": np.__version__,
        },
        "limitations": [
            "Dash Flask test client, not a real browser or click-to-paint",
            "HTTP callback timing includes server dispatch and JSON serialization",
            "No loaded geometry/view, matrix provider, Apply draft or classifier fit",
            "Python process peak RSS is cumulative, not an operation-level RAM delta",
            "Setup ingestion and seeded human judgments are excluded from samples",
            "No low-power laptop, historical baseline, concurrent user or remote provider",
            "This pilot samples a few operations, not the exhaustive Phase 2B register",
        ],
        "runs": [_case(size, args.repeats, args.seed) for size in args.sizes],
    }
    serialized = json.dumps(result, sort_keys=True, indent=2)
    print(serialized, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
