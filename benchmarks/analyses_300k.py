"""300k-observation chart-specific acceptance benchmark with real SQLite paging.

Usage: uv run --no-sync python benchmarks/analyses_300k.py --units 300000
This tests the continuous Analyses computation and Plotly output, NOT Scale
candidate sampling. The synthetic project contains only atomic IDs and typed
metadata; no text, model fit, or TeAL geometry is allocated.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import date
import json
from pathlib import Path
import sqlite3
import tempfile
import time
import tracemalloc

from geometric_coder.analyses import analysis_figure, compute_analysis


class Database:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()


class Project:
    def __init__(self, path: Path):
        self.database = Database(path)
        self.metadata = {
            "metadata_columns": ["score", "rating", "group", "day"],
            "key_columns": ["id"],
        }


def populate(path: Path, units: int) -> dict[str, int]:
    counts = {"score": 0, "rating": 0, "group": 0, "complete": 0}
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "CREATE TABLE units (observation_id INTEGER PRIMARY KEY, "
            "row_position INTEGER NOT NULL, metadata_json TEXT NOT NULL)"
        )
        conn.execute("CREATE INDEX units_row_position ON units(row_position)")
        batch = []
        for i in range(units):
            score = None if i % 20 == 0 else float(i % 100)
            rating = None if i % 25 == 0 else float((i % 100) * 2 + 3)
            group = None if i % 17 == 0 else "G" + str(i % 10)
            # Cross multiple calendar years without parsing text/geometry.
            day = f"{2024 + (i // 120000):04d}-{(i // 10000) % 12 + 1:02d}-15"
            counts["score"] += score is not None
            counts["rating"] += rating is not None
            counts["group"] += group is not None
            counts["complete"] += score is not None and rating is not None and group is not None
            batch.append((i + 1, i, json.dumps({
                "score": score, "rating": rating, "group": group, "day": day,
            }, separators=(",", ":"))))
            if len(batch) == 5000:
                conn.executemany("INSERT INTO units VALUES (?, ?, ?)", batch)
                batch.clear()
        if batch:
            conn.executemany("INSERT INTO units VALUES (?, ?, ?)", batch)
        conn.commit()
    finally:
        conn.close()
    return counts


def metadata(field: str, kind: str) -> dict:
    return {"variable": {"type": "metadata", "field": field, "value": kind}}


def recipe(kind: str, **roles) -> dict:
    return {"schema_version": 2, "kind": kind, "roles": roles}


def cases() -> dict[str, dict]:
    score, rating = metadata("score", "numeric"), metadata("rating", "numeric")
    group = metadata("group", "categorical")
    month = {**metadata("day", "temporal"),
             "transform": {"kind": "calendar", "unit": "month"}}
    numeric_group = {**metadata("rating", "numeric"),
                     "transform": {"kind": "numeric_bins", "method": "equal_width", "bins": 8}}
    result = {
        "scatter": recipe("scatter", x=score, y=rating, color=group),
        "histogram": recipe("histogram", values=score, group=month),
        "histogram_quantile": recipe("histogram", values=score, group=group),
        "summary_mean": recipe("summary_bar", values=score, group=group),
        "summary_median": recipe("summary_bar", values=score, group=group),
        "correlation": recipe("correlation_heatmap", values=[score, rating]),
        "value_heatmap": recipe("value_heatmap", values=score, rows=group, columns=month),
        "numeric_group": recipe("summary_bar", values=score, group=numeric_group),
    }
    result["histogram"]["binning"] = {
        "kind": "histogram_bins", "method": "equal_width", "bins": 20,
    }
    result["histogram_quantile"]["binning"] = {
        "kind": "histogram_bins", "method": "quantile", "bins": 10,
    }
    result["summary_median"]["statistic"] = "median"
    return result


def benchmark(units: int, max_payload_bytes: int = 1_000_000) -> dict:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "analyses.sqlite"
        expected = populate(path, units)
        project = Project(path)
        records = {}
        for name, spec in cases().items():
            tracemalloc.start()
            started = time.perf_counter()
            result = compute_analysis(project, spec)
            elapsed = time.perf_counter() - started
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            figure = analysis_figure(result)
            # This is the actual Plotly figure transferred by Dash.
            payload_bytes = len(figure.to_json().encode("utf-8"))
            result_bytes = len(json.dumps(result).encode("utf-8"))
            assert result["coverage"]["total"] == units, name
            assert result["coverage"]["observed"][0] == expected["score"], name
            assert result_bytes <= max_payload_bytes, (name, result_bytes)
            assert payload_bytes <= max_payload_bytes, (name, payload_bytes)
            if name == "scatter":
                paired = sum(i % 20 != 0 and i % 25 != 0 and i % 17 != 0
                             for i in range(units))
                assert result["eligible"] == paired, result["eligible"]
                assert result["sampled"] == min(paired, 2000)
                assert sum(len(trace.x) for trace in figure.data) == result["sampled"]
            elif name.startswith("histogram"):
                assert sum(sum(values) for values in result["counts"].values()) <= expected["score"]
                assert result["coverage"]["observed"][0] == expected["score"]
            elif name.startswith("summary") or name == "numeric_group":
                assert sum(result["denominators"]) <= expected["score"]
            elif name == "correlation":
                assert result["samples"][0][0] == expected["score"]
                assert result["samples"][1][1] == expected["rating"]
            elif name == "value_heatmap":
                assert sum(map(sum, result["samples"])) <= expected["score"]
            assert len(result["sources"]) == len(result["coverage"]["observed"])
            records[name] = {
                "seconds": round(elapsed, 3),
                "peak_tracemalloc_mib": round(peak / 1024**2, 3),
                "result_bytes": result_bytes,
                "plotly_bytes": payload_bytes,
                "warnings": result["warnings"],
            }
        return {"observations": units, "expected": expected, "charts": records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--units", type=int, default=300_000)
    parser.add_argument("--max-payload-bytes", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.units < 1 or args.max_payload_bytes < 1:
        parser.error("units and payload limit must be positive")
    print(json.dumps(benchmark(args.units, args.max_payload_bytes), indent=2))


if __name__ == "__main__":
    main()
