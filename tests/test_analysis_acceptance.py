"""Phase 4 scientific acceptance: saved workflows and transparent chart sources."""
from __future__ import annotations

import pandas as pd
import pytest

from geometric_coder import GeometricCoder
from geometric_coder.analyses import compute_analysis, analysis_figure
from geometric_coder import analysis_continuous


def field(name, value):
    return {"variable": {"type": "metadata", "field": name, "value": value}}


def test_continuous_sources_missing_reasons_and_no_zero_imputation(tmp_path, monkeypatch):
    project = GeometricCoder.create(
        project_dir=tmp_path / "acceptance.geco",
        data=pd.DataFrame({
            "id": [101, 103, 107, 109, 113],
            "text": ["a", "b", "c", "d", "e"],
            "score": [0.0, 2.0, None, 6.0, 8.0],
            "group": ["A", "A", "A", "B", None],
            "day": ["2025-12-31", "2026-01-01", "2026-01-02",
                    "2026-02-01", "2026-02-02"],
        }), keys=["id"], text="text",
        metadata=["score", "group", "day"], geometries=None,
    )
    code = project.create_code("Human judgment")
    ids = [row["observation_id"] for row in project.database.list_units()]
    project.annotate(ids[0], code, "positive")
    project.annotate(ids[1], code, "unsure")
    project.annotate(ids[3], code, "negative")
    values = {"variable": {"type": "code", "code_id": code,
                            "source": "labels", "value": "binary"}}
    spec = {"schema_version": 2, "kind": "scatter",
            "roles": {"x": field("score", "numeric"), "y": values},
            "max_points": 2}
    monkeypatch.setattr(project, "train_classifier",
                        lambda *a, **k: pytest.fail("Analyses trained a classifier"))
    result = compute_analysis(project, spec)
    assert result["eligible"] == 2
    assert result["sampled"] == 2
    assert result["coverage"] == {"total": 5, "complete": 2, "observed": [4, 2]}
    assert result["sources"][0]["missing_reasons"] == {"missing_metadata": 1}
    assert result["sources"][1]["missing_reasons"] == {"unsure": 1, "unreviewed": 2}
    assert result["sources"][1]["provenance"]["source"] == "labels"
    assert result["sources"][1]["provenance"]["fit_status"] == "human_only"
    assert {point["id"] for point in result["points"]} == {ids[0], ids[3]}
    assert len(analysis_figure(result).data) == 1

    # Verify source provenance also survives aggregated histograms, without
    # double-counting missingness during their second full-corpus pass.
    hist = {"schema_version": 2, "kind": "histogram",
            "roles": {"values": field("score", "numeric"),
                      "group": field("day", "temporal") |
                               {"transform": {"kind": "calendar", "unit": "month"}}}}
    chart = compute_analysis(project, hist)
    assert chart["coverage"]["total"] == 5
    assert chart["sources"][0]["missing_reasons"] == {"missing_metadata": 1}
    assert chart["sources"][1]["missing_reasons"] == {}
    assert sum(map(sum, chart["counts"].values())) == 4
    assert "2025-12" in chart["groups"]
    assert "2026-02" in chart["groups"]

    saved = project.save_analysis("Scientific chart", spec)
    assert project.run_analysis(saved)["sources"] == result["sources"]
    updated = {"schema_version": 2, "kind": "summary_bar",
               "roles": {"values": field("score", "numeric"),
                         "group": field("group", "categorical")},
               "statistic": "mean"}
    project.save_analysis("Updated scientific chart", updated, analysis_id=saved)
    assert project.analysis(saved)["spec"]["kind"] == "summary_bar"
    assert project.run_analysis(saved)["denominators"] == [2, 1]
    project.delete_analysis(saved)
    assert all(entry["analysis_id"] != saved for entry in project.analyses())


def test_model_or_source_changes_during_scan_are_rejected(tmp_path, monkeypatch):
    project = GeometricCoder.create(
        project_dir=tmp_path / "source-change.geco",
        data=pd.DataFrame({"id": [1, 2, 3], "text": ["a", "b", "c"],
                           "score": [1.0, 2.0, 3.0]}),
        keys=["id"], text="text", metadata=["score"], geometries=None,
    )
    spec = {"schema_version": 2, "kind": "histogram",
            "roles": {"values": field("score", "numeric")}}
    original = analysis_continuous._pages
    calls = 0

    def changed_pages(*args, **kwargs):
        nonlocal calls
        calls += 1
        for page in original(*args, **kwargs):
            if calls == 2:
                page["columns"][0]["provenance"] = {
                    **page["columns"][0]["provenance"], "source_changed": True,
                }
            yield page

    monkeypatch.setattr(analysis_continuous, "_pages", changed_pages)
    with pytest.raises(ValueError, match="source or fitted model changed"):
        compute_analysis(project, spec)


def test_chart_performance_harness_smoke():
    # The full 300k workload runs as a separate required CI step.
    from benchmarks.analyses_300k import benchmark
    output = benchmark(120)
    assert output["observations"] == 120
    assert set(output["charts"]) == {
        "scatter", "histogram", "histogram_quantile", "summary_mean",
        "summary_median", "correlation", "value_heatmap", "numeric_group",
    }
    assert all(chart["plotly_bytes"] < 1_000_000
               for chart in output["charts"].values())
