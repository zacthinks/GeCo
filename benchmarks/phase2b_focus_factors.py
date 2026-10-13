"""Controlled Focus Coding scaling diagnostics. No production mutations.

Run: uv run python benchmarks/phase2b_focus_factors.py --repeats 20
All projects are disposable. CI timing results are descriptive, not gates.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sqlite3
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from geometric_coder import GeometricCoder
from geometric_coder.ui import create_app
from phase2a_runtime_pilot import _exercise_focus_render, _judgment_ids, _component_text


CASES = (
    ("base-small", 1000, 40, 0, 100, 2),
    ("base-large", 10000, 40, 0, 100, 2),
    ("active-small-high", 1000, 400, 0, 100, 2),
    ("active-large-high", 10000, 400, 0, 100, 2),
    ("history-large", 10000, 40, 400, 100, 2),
    ("task-large", 10000, 40, 0, 400, 2),
    ("codes-large", 10000, 40, 0, 100, 4),
    ("crossed-task-codes", 10000, 40, 0, 400, 4),
)


def _measure(fn: Callable[[], Any], repeats: int, validate=None) -> dict[str, Any]:
    walls, cpus = [], []
    for _ in range(repeats):
        t0, c0 = time.perf_counter(), time.process_time()
        value = fn()
        walls.append((time.perf_counter() - t0) * 1000)
        cpus.append((time.process_time() - c0) * 1000)
        if validate is not None:
            validate(value)
    sorted_wall = sorted(walls)
    return {
        "first_ms": round(walls[0], 3),
        "median_ms": round(statistics.median(walls), 3),
        "p95_ms": round(sorted_wall[max(0, int(np.ceil(.95 * repeats)) - 1)], 3),
        "median_cpu_ms": round(statistics.median(cpus), 3),
        "repeats": repeats,
    }


def _trace_sql(project: GeometricCoder, fn: Callable[[], Any]) -> int:
    # Instrument a separate invocation, never a timed one.
    statements = []
    original = project.database.connect

    from contextlib import contextmanager

    @contextmanager
    def traced_connect():
        with original() as connection:
            connection.set_trace_callback(statements.append)
            try:
                yield connection
            finally:
                connection.set_trace_callback(None)

    project.database.connect = traced_connect
    try:
        fn()
    finally:
        project.database.connect = original
    return len(statements)


def _validate_render(payload: bytes, task: int, codes: list[int], first_text: str):
    obj = json.loads(payload)
    assert obj["multi"] is True
    reply = obj["response"]

    def field(name, prop="children"):
        return reply[name][prop]

    assert field("focus-coding-progress-summary") == f"0 / {task} documents complete"
    assert field("focus-coding-judgment-summary") == (
        f"0 / {task * len(codes)} required judgments resolved"
    )
    assert field("focus-coding-progress-bar", "value") == 0
    assert field("focus-coding-progress-bar", "max") == task
    assert field("focus-coding-position") == f"Document 1 of {task}"
    assert field("focus-coding-text") == first_text
    assert "row_id: " in _component_text(field("focus-coding-key"))
    assert "0" in _component_text(field("focus-coding-key"))
    cards = field("focus-coding-code-cards")
    assert len(cards) == len(codes)
    assert _judgment_ids(cards) == {
        (code, value) for code in codes for value in ("positive", "negative")
    }


def _case(spec, repeats, seed):
    name, size, active, extra_history, task_size, code_count = spec
    rng = np.random.default_rng(seed)
    tokens = ("teacher", "student", "reading", "feedback", "dialogue", "learning")
    texts = [
        " ".join(tokens[int(j)] for j in rng.integers(0, len(tokens), 7))
        for _ in range(size)
    ]
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="geco-phase2b-") as temp:
        project = GeometricCoder.create(
            project_dir=Path(temp) / "factor.geco",
            data=pd.DataFrame({"row_id": range(size), "text": texts}),
            keys=["row_id"], text="text", metadata=[],
        )
        codes = [project.create_code(f"focus_{i}") for i in range(code_count)]
        session = project.configure_focus(
            codes=codes,
            unit_ids=list(range(1, task_size + 1)),
            allow_unsure=False,
        )
        units = project.units()
        assert len(units) == size
        # Write assignment identities ONLY outside the finite Focus task.
        # Repeated history events revise the same identity (not create pairs).
        outside = units[task_size:task_size + active]
        assert len(outside) == active
        for unit in outside:
            project.annotate(int(unit["observation_id"]), codes[0], "positive")
        for i in range(extra_history):
            project.annotate(
                int(outside[i % active]["observation_id"]),
                codes[0], "negative" if i % 2 else "positive",
            )
        with project.database.connect() as db:
            n_events = int(db.execute("SELECT COUNT(*) FROM annotation_events").fetchone()[0])
            n_current = int(db.execute("SELECT COUNT(*) FROM current_annotations").fetchone()[0])
            plan = [
                row["detail"] for row in db.execute(
                    "EXPLAIN QUERY PLAN SELECT * FROM current_annotations "
                    "ORDER BY observation_id, code_id"
                )
            ]
        assert n_events == active + extra_history, (name, n_events)
        assert n_current == active, (name, n_current)
        expected_unresolved = task_size * len(codes)

        def check_progress(value):
            assert value["total_units"] == task_size
            assert value["total_judgments"] == expected_unresolved
            assert value["resolved_judgments"] == 0
            assert value["unresolved_count"] == expected_unresolved
            assert value["complete_units"] == 0

        check_progress(project.focus_progress(session))
        app = create_app(project, mode="focus", focus_session_id=session)
        def render():
            return _exercise_focus_render(app, session, 1)

        validate = lambda payload: _validate_render(payload, task_size, codes, texts[0])
        validate(render())
        setup_seconds = round(time.perf_counter() - started, 3)

        ops = {
            "focus_task": lambda: project.focus_task(session),
            "current_annotations": project.current_annotations,
            "focus_progress": lambda: project.focus_progress(session),
        }
        # Counts are from separate traced invocations.
        sql_counts = {k: _trace_sql(project, fn) for k, fn in ops.items()}
        timings = {
            k: _measure(fn, repeats, check_progress if k == "focus_progress" else None)
            for k, fn in ops.items()
        }
        sql_counts["focus_render_http"] = _trace_sql(project, render)
        render_times = _measure(render, repeats, validate)
        return {
            "case": name,
            "N": size, "A": active, "H": extra_history,
            "T": task_size, "C": code_count,
            "events": n_events, "current_pairs": n_current,
            "setup_seconds": setup_seconds,
            "query_plan": plan,
            "sql_statements": sql_counts,
            "timings": {**timings, "focus_render_http": render_times},
            "render_bytes": len(render()),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--cases", nargs="*", choices=[x[0] for x in CASES])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 5:
        parser.error("repeats must be >= 5")
    selected = [spec for spec in CASES if not args.cases or spec[0] in args.cases]
    report = {
        "runner": {
            "platform": platform.platform(), "python": platform.python_version(),
            "cpu_count": os.cpu_count(), "numpy": np.__version__,
        },
        "seed": args.seed,
        "limitations": [
            "No real browser, paint, external provider, trained model or geometry",
            "Each case builds its own disposable project, but cases run sequentially in one process",
            "Fixture setup time is not API callback time",
            "Cold process initialization is not independently profiled",
            "A/H pairs use one outside-task code; events are not independent random annotations",
            "CI p95 estimates are noisy; no speed pass/fail threshold or user-laptop guarantee",
        ],
        "runs": [_case(spec, args.repeats, args.seed) for spec in selected],
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text, flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
