"""Fast deterministic checks for the tiered benchmark runner (no large CI fixtures)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# The benchmark directory is intentionally not an installed application package.
# pytest's console entrypoint does not always add the checkout root to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.tiered_linked import (
    TIERS, SyntheticProvider, exercise, parse_args, run, synthetic_project,
)
from benchmarks.tiered_report import _percentile_95, measure, save_report


def test_report_first_and_warm_are_separate_and_small_samples_have_no_p95():
    seen = []

    def action():
        seen.append(len(seen))
        return [1, 2, 3]

    result = measure(
        "case", action, lambda rows: rows == [1, 2, 3] or (_ for _ in ()).throw(
            AssertionError("bad result")
        ),
        lambda rows: {"row_count": len(rows)}, warm_repeats=3,
    )
    assert len(seen) == 4
    assert result["first_call_proxy"]["state"] == "first_call_same_process_not_cold"
    assert result["warm"]["repetitions"] == 3
    assert result["warm"]["p95_wall_ms"] is None
    assert result["warm"]["p95_status"] == "insufficient_samples"
    assert result["metrics"]["process_rss"] == "not_measured"
    assert result["warm"]["python_allocation_peak_bytes_max"] >= 0


def test_report_rejects_changed_shape_and_invalid_sample_count():
    with pytest.raises(ValueError):
        measure("bad", lambda: [], lambda _: None, lambda _: {}, warm_repeats=0)
    counter = iter(range(4))
    with pytest.raises(AssertionError, match="shape changed"):
        measure(
            "unstable", lambda: next(counter), lambda _: None,
            lambda value: {"item": value}, warm_repeats=2,
        )


def test_percentile_only_with_enough_samples():
    assert _percentile_95(list(range(19))) is None
    assert _percentile_95(list(range(20))) == 18


@pytest.mark.parametrize("units", [18, 65])
def test_synthetic_reads_are_aligned_and_bounded(tmp_path, units):
    project = synthetic_project(tmp_path / "tiered.geco", units)
    outcome = exercise(
        project, tier="a", seed=19, page_size=7,
        repeats=1, synthetic=True, filter_field="category", filter_value="A",
    )
    assert outcome["units"] == units
    assert set(outcome["operations"]) == {
        "sqlite_key_page", "sqlite_text_page", "context_page",
        "filtered_sql_page", "view_coordinate_page",
        "topic_provider_page", "topic_provider_selector",
    }
    assert all(record["status"] == "ok" for record in outcome["operations"].values())
    assert outcome["operations"]["filtered_sql_page"]["result"]["matching_rows"] == (units + 2) // 3
    assert outcome["operations"]["topic_provider_selector"]["result"]["row_count"] == 1
    assert outcome["operations"]["context_page"]["result"]["row_count"] <= 16


def test_unavailable_or_unsafe_modes_are_reported(tmp_path, monkeypatch):
    project = synthetic_project(tmp_path / "tiered.geco", 20)
    project.views = lambda: []
    project.topic_models = lambda: []
    outcome = exercise(project, tier="a", page_size=4, repeats=1)
    assert outcome["operations"]["view_coordinate_page"]["status"] == "unavailable"
    assert outcome["operations"]["topic_provider_page"]["status"] == "unavailable"
    # Bounded GeCo context is now available without an unbounded materialization.
    assert outcome["operations"]["context_page"]["status"] == "ok"
    assert outcome["operations"]["context_page"]["result"]["row_count"] <= 200
    assert outcome["operations"]["filtered_sql_page"]["status"] == "unavailable"


def test_provider_never_requests_unbounded_rows():
    provider = SyntheticProvider()
    with pytest.raises(AssertionError, match="unbounded"):
        provider.read_artifact("synthetic_membership", None)
    keys = [{"document": 2, "sentence": 3}, {"document": 0, "sentence": 4}]
    values = provider.read_artifact("synthetic_argmax", keys)
    assert values.values["max_feature"].tolist() == ["topic_3", "topic_4"]
    assert values.positions == [0, 1]


def test_cli_tiers_and_json_report(tmp_path):
    assert TIERS == {"a": 1000, "b": 50000, "c": 300000}
    assert parse_args(["--tier", "b"]).tier == "b"
    with pytest.raises(SystemExit):
        parse_args(["--tier", "heavy"])
    destination = tmp_path / "out.json"
    value = {"status": "ok", "p95": None}
    output = save_report(value, destination)
    assert json.loads(output) == value
    assert json.loads(destination.read_text()) == value


def test_synthetic_provider_preserves_unsorted_key_alignment():
    provider = SyntheticProvider()
    keys = [
        {"document": 2, "sentence": 5},
        {"document": 0, "sentence": 1},
        {"document": 1, "sentence": 7},
    ]
    expected = np.array([37, 1, 23])
    values = provider.view_coordinates({}, keys)
    np.testing.assert_allclose(values[:, 0], np.sin(expected / 101))

def test_small_report_round_trip_without_real_teal(tmp_path):
    args = parse_args(["--tier", "a", "--units", "24", "--page-size", "5", "--repeats", "1"])
    report = run(args)
    assert report["schema_version"] == 1
    assert report["fixture"]["actual_units"] == 24
    assert report["execution"]["operations"]["sqlite_key_page"]["status"] == "ok"
    assert report["cold_process"]["status"] == "not_measured"
    saved = save_report(report, tmp_path / "sample.json")
    assert json.loads(saved) == json.loads((tmp_path / "sample.json").read_text())


@pytest.mark.skipif(
    not (os.environ.get("GECO_BENCH_SNAPSHOT") and os.environ.get("TEAL_BENCH_SNAPSHOT")),
    reason="opt-in: supply GECO_BENCH_SNAPSHOT and TEAL_BENCH_SNAPSHOT copies",
)
def test_optional_real_teal_linked_read_snapshot():
    """Only execute with disposable project snapshots, never canonical datasets."""
    args = parse_args([
        "--tier", "a", "--repeats", "1", "--page-size", "3",
        "--geco-path", os.environ["GECO_BENCH_SNAPSHOT"],
        "--teal-path", os.environ["TEAL_BENCH_SNAPSHOT"],
        "--dataset-id", "linked-opt-in-test",
    ])
    report = run(args)
    assert report["fixture"]["mode"] == "real_teal_linked"
    assert report["execution"]["operations"]["sqlite_key_page"]["status"] == "ok"
