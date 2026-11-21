"""300k-observation chart-specific acceptance benchmark with real SQLite paging.

Usage: uv run --no-sync python benchmarks/analyses_300k.py --units 300000
This tests the continuous Analyses computation and Plotly output, NOT Scale
candidate sampling. The synthetic project contains only atomic IDs and typed
metadata; no text, model fit, or TeAL geometry is allocated.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import defaultdict
import math
from statistics import median
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
        "unified_bar": {
            "schema_version": 2, "kind": "bar_chart",
            "roles": {"values": [score], "group": group, "color": month},
            "statistic": "sum",
        },
    }
    result["histogram"]["binning"] = {
        "kind": "histogram_bins", "method": "equal_width", "bins": 20,
    }
    result["histogram_quantile"]["binning"] = {
        "kind": "histogram_bins", "method": "quantile", "bins": 10,
    }
    result["summary_median"]["statistic"] = "median"
    return result



def _synthetic_rows(units: int):
    """Independent fixture oracle, not using GeCo chart grouping/bin kernels."""
    for i in range(units):
        yield (
            None if i % 20 == 0 else float(i % 100),
            None if i % 25 == 0 else float((i % 100) * 2 + 3),
            None if i % 17 == 0 else "G" + str(i % 10),
            f"{2024 + (i // 120000):04d}-{(i // 10000) % 12 + 1:02d}",
        )


def _expected_cell_statistics(units: int):
    grouped = defaultdict(list)
    by_month = defaultdict(list)
    heatmap = defaultdict(list)
    pairs = []
    for score, rating, group, month in _synthetic_rows(units):
        if score is not None and group is not None:
            grouped[group].append(score)
            heatmap[(group, month)].append(score)
        if score is not None:
            by_month[month].append(score)
        if score is not None and rating is not None:
            pairs.append((score, rating))
    return grouped, by_month, heatmap, pairs


def _bin_oracle(value: float, edges: list[float]) -> int | None:
    if value < edges[0] or value > edges[-1]:
        return None
    return min(len(edges) - 2, bisect_right(edges, value) - 1)


def _assert_approximate_median(actual: float, observed: list[float]) -> None:
    """Compare sampled medians by population rank, not arbitrary value units.

    A 5k sample can move across a large gap in a discrete distribution while
    changing the population quantile rank negligibly. The 3-percentage-point
    band is a conservative sampling-approximation acceptance tolerance (near
    the 99.9% DKW bound for 5k random draws), not an inferential CI: the
    production sample is deterministic and hash-selected.
    """
    if len(observed) <= 5000:
        assert math.isclose(actual, median(observed), abs_tol=1e-9), actual
        return
    ordered = sorted(observed)
    below = bisect_left(ordered, actual) / len(ordered)
    at_or_below = bisect_right(ordered, actual) / len(ordered)
    assert below <= 0.53 and at_or_below >= 0.47, (
        actual, below, at_or_below,
    )


def _assert_chart_values(name: str, result: dict, units: int, oracle) -> None:
    """Require exact second-pass counts, not merely non-exceedance.

    The synthetic oracle reconstructs values from the fixture generator, not
    GeCo's chart rows, group reducers, or histogram aggregation functions.
    """
    grouped, by_month, heatmap, pairs = oracle
    expected_score = sum(score is not None for score, _, _, _ in _synthetic_rows(units))
    expected_rating = sum(rating is not None for _, rating, _, _ in _synthetic_rows(units))
    if name.startswith("histogram"):
        edges = result["edges"]
        expected_counts = defaultdict(lambda: [0] * (len(edges) - 1))
        excluded = missing_group = 0
        for score, _, group, month in _synthetic_rows(units):
            if score is None:
                continue
            category = month if name == "histogram" else group
            if category is None:
                missing_group += 1
                continue
            index = _bin_oracle(score, edges)
            if index is None:
                excluded += 1
                continue
            expected_counts[category][index] += 1
        assert result["counts"] == dict(expected_counts), name
        assert result["excluded_range"] == excluded, name
        assert result["missing_group"] == missing_group, name
        assert result["denominators"] == {
            group: sum(counts) for group, counts in expected_counts.items()
        }, name
    elif name in ("summary_mean", "summary_median"):
        expected_groups = sorted(grouped)
        assert result["categories"] == expected_groups, name
        assert result["denominators"] == [len(grouped[g]) for g in expected_groups], name
        if name == "summary_mean":
            for actual, group in zip(result["values"], expected_groups, strict=True):
                assert math.isclose(actual, sum(grouped[group]) / len(grouped[group]),
                                    abs_tol=1e-9), (name, group, actual)
        else:
            for actual, group in zip(result["values"], expected_groups, strict=True):
                # The sampled median is an estimate of the 50th percentile;
                # numeric error depends on measurement scale and value gaps.
                _assert_approximate_median(actual, grouped[group])
    elif name == "correlation":
        assert result["samples"] == [
            [expected_score, len(pairs)], [len(pairs), expected_rating]
        ], name
        if len(pairs) >= 2:
            xs, ys = zip(*pairs)
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            covariance = sum((x - mx) * (y - my) for x, y in pairs)
            variance_x = sum((x - mx) ** 2 for x in xs)
            variance_y = sum((y - my) ** 2 for y in ys)
            expected_r = covariance / math.sqrt(variance_x * variance_y)
            assert math.isclose(result["values"][0][1], expected_r, abs_tol=1e-10), name
    elif name == "value_heatmap":
        assert result["rows"] == sorted(grouped), name
        assert result["columns"] == sorted({month for _, _, _, month in _synthetic_rows(units)}), name
        for i, group in enumerate(result["rows"]):
            for j, month in enumerate(result["columns"]):
                cell = heatmap.get((group, month), [])
                assert result["samples"][i][j] == len(cell), (name, group, month)
                expected_value = sum(cell) / len(cell) if cell else None
                actual = result["values"][i][j]
                assert (actual is None if expected_value is None else
                        math.isclose(actual, expected_value, abs_tol=1e-9)), (name, group, month)
    elif name == "unified_bar":
        expected_groups = sorted(grouped)
        assert result["categories"] == expected_groups, name
        # The independent fixture oracle computes expected per-group monthly
        # values without calling GeCo bar aggregation or grouping helpers.
        by_month_series = {
            record["name"]: record for record in result["series"]
        }
        expected_months = sorted(
            {month for _, _, _, month in _synthetic_rows(units)}
        )
        assert set(by_month_series) == {
            f"score / {month}" for month in expected_months
            if any(heatmap.get((group, month)) for group in expected_groups)
        }, name
        for month in expected_months:
            series = by_month_series.get(f"score / {month}")
            if series is None:
                continue
            for index, group in enumerate(expected_groups):
                values = heatmap.get((group, month), [])
                assert series["denominators"][index] == len(values), (
                    name, group, month, "denominator"
                )
                assert math.isclose(
                    series["values"][index], sum(values), abs_tol=1e-9
                ), (name, group, month, "sum")
    elif name == "numeric_group":
        edges = result["transformations"]["group"]["edges"]
        by_bin = defaultdict(list)
        for score, rating, _, _ in _synthetic_rows(units):
            if score is None or rating is None:
                continue
            index = _bin_oracle(rating, edges)
            if index is not None:
                by_bin[index].append(score)
        # Group labels are intervals; they appear in ascending-bin order.
        assert len(result["categories"]) == len(by_bin), name
        for i, bin_index in enumerate(sorted(by_bin)):
            values = by_bin[bin_index]
            assert result["denominators"][i] == len(values), (name, bin_index)
            assert math.isclose(result["values"][i], sum(values) / len(values),
                                abs_tol=1e-9), (name, bin_index)


def benchmark(units: int, max_payload_bytes: int = 1_000_000) -> dict:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "analyses.sqlite"
        expected = populate(path, units)
        project = Project(path)
        records = {}
        oracle = _expected_cell_statistics(units)
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
            _assert_chart_values(name, result, units, oracle)
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
