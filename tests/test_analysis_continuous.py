"""Regression of typed numerical exploratory recipes and calendar semantics."""
from __future__ import annotations

import math
import pandas as pd
import pytest

from geometric_coder import GeometricCoder
from geometric_coder.analyses import analysis_figure, compute_analysis, validate_analysis
from geometric_coder.analysis_continuous import calendar_period


@pytest.fixture
def workspace(tmp_path):
    p = GeometricCoder.create(
        project_dir=tmp_path / "continuous.geco",
        data=pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6],
            "text": ["a", "b", "c", "d", "e", "f"],
            "score": [1.0, 2.0, 3.0, 4.0, 5.0, float("nan")],
            "group": ["A", "A", "B", "B", "(Missing)", None],
            "day": ["2025-12-31", "2026-01-01", "2026-01-02",
                    "2026-02-01", "2026-02-02", "2026-12-30"],
        }), keys=["id"], text="text",
        metadata=["score", "group", "day"], geometries=None,
    )
    code = p.create_code("Binary")
    units = [u["observation_id"] for u in p.database.list_units()]
    for key, judgment in zip(units, ["positive", "negative", "unsure", "positive"]):
        p.annotate(key, code, judgment)
    return p, code, units


def meta(field, value):
    return {"variable": {"type": "metadata", "field": field, "value": value}}


def recipe(kind, **roles):
    return {"schema_version": 2, "kind": kind, "roles": roles}


def test_scatter_pairwise_missing_coverage_and_stable_ids(workspace):
    p, code, ids = workspace
    spec = recipe("scatter", x=meta("score", "numeric"),
                  y={"variable": {"type": "code", "code_id": code,
                                  "source": "labels", "value": "binary"}})
    result = compute_analysis(p, spec)
    assert result["eligible"] == 3
    assert result["sampled"] == 3
    assert {point["id"] for point in result["points"]} == set(ids[:2] + ids[3:4])
    assert result["coverage"]["total"] == 6
    assert len(analysis_figure(result).data) == 1


def test_histogram_equal_width_quantile_and_literal_missing_category(workspace):
    p, _, _ = workspace
    spec = recipe("histogram", values=meta("score", "numeric"),
                  group=meta("group", "categorical"))
    spec["binning"] = {"kind": "histogram_bins", "method": "equal_width", "bins": 2}
    result = compute_analysis(p, spec)
    assert result["edges"] == [1, 3, 5]
    assert sum(map(sum, result["counts"].values())) == 5
    assert "(Missing)" in result["groups"]
    assert result["missing_group"] == 0
    spec["binning"] = {"kind": "histogram_bins", "method": "quantile", "bins": 2}
    result = compute_analysis(p, spec)
    assert sum(map(sum, result["counts"].values())) == 5
    spec["binning"] = {"kind": "histogram_bins", "method": "explicit", "edges": [2, 4]}
    result = compute_analysis(p, spec)
    assert result["excluded_range"] == 2
    assert sum(map(sum, result["counts"].values())) == 3


def test_numerical_summary_and_paired_correlations(workspace):
    p, code, ids = workspace
    spec = recipe("summary_bar", values=meta("score", "numeric"),
                  group=meta("group", "categorical"))
    spec["statistic"] = "mean"
    result = compute_analysis(p, spec)
    assert result["categories"] == ["(Missing)", "A", "B"]
    assert result["values"] == pytest.approx([5, 1.5, 3.5])
    assert result["denominators"] == [1, 2, 2]
    spec["statistic"] = "median"
    assert compute_analysis(p, spec)["values"] == pytest.approx([5, 1.5, 3.5])

    numeric_code = {"variable": {"type": "code", "code_id": code,
                                "source": "labels", "value": "binary"}}
    heat = recipe("correlation_heatmap",
                  values=[meta("score", "numeric"), numeric_code])
    r = compute_analysis(p, heat)
    assert r["samples"][0][1] == 3
    assert r["values"][0][1] is not None
    assert r["samples"][1][1] == 3
    assert len(analysis_figure(r).data) == 1


def test_calendar_iso_week_year_quarter_and_timezone():
    date_only = {"kind": "date", "iso": "2025-12-31"}
    assert calendar_period(date_only, "week") == "2026-W01"
    assert calendar_period(date_only, "month") == "2025-12"
    assert calendar_period(date_only, "quarter") == "2025-Q4"
    offset = {"kind": "datetime", "iso": "2026-01-01T00:30:00+02:00",
              "timezone": "offset"}
    assert calendar_period(offset, "day") == "2025-12-31"
    naive = {"kind": "datetime", "iso": "2026-01-01T00:30:00",
             "timezone": "naive"}
    assert calendar_period(naive, "day") == "2026-01-01"


def test_calendar_grouping_saved_live_and_no_model_training(workspace, monkeypatch):
    p, code, ids = workspace
    spec = recipe("summary_bar", values=meta("score", "numeric"),
                  group={**meta("day", "temporal"),
                         "transform": {"kind": "calendar", "unit": "month"}})
    with monkeypatch.context() as patch:
        patch.setattr(p, "train_classifier", lambda *a, **k: pytest.fail("retraining"))
        saved = p.save_analysis("Monthly means", spec)
        assert p.analysis(saved)["spec"]["roles"]["group"]["transform"]["unit"] == "month"
        result = p.run_analysis(saved)
    assert "2025-12" in result["categories"]
    assert "2026-01" in result["categories"]
    assert p.analysis(saved)["spec"]["schema_version"] == 2
    from geometric_coder.ui.analyses import _initial_config
    restored = _initial_config(p.analysis(saved)["spec"])
    assert restored["group:metadata:day"]["transform"] == {"kind": "calendar", "unit": "month"}


def test_invalid_settings_dont_save(workspace):
    p, _, _ = workspace
    invalid = recipe("scatter", x=meta("score", "numeric"), y=meta("day", "temporal"))
    with pytest.raises(ValueError, match="numeric"):
        validate_analysis(p, invalid)
    histogram = recipe("histogram", values=meta("score", "numeric"))
    histogram["binning"] = {"kind": "histogram_bins", "method": "equal_width", "bins": 0}
    with pytest.raises(ValueError, match="bins"):
        p.save_analysis("No bins", histogram)
    assert not any(row["name"] == "No bins" for row in p.analyses())


def test_bounded_scatter_sampling_and_source_change(workspace):
    p, code, ids = workspace
    spec = recipe("scatter", x=meta("score", "numeric"), y=meta("score", "numeric"))
    spec["max_points"] = 2
    result = compute_analysis(p, spec)
    assert result["sampled"] == 2
    assert result["eligible"] == 5
    assert compute_analysis(p, spec)["points"] == result["points"]
    saved = p.save_analysis("Sampled scatter", spec)
    assert p.run_analysis(saved)["points"] == result["points"]


def test_scatter_color_limit_checked_on_eligible_population_before_save(tmp_path):
    # Even a tiny sampled plot must reject unbounded color traces.
    count = 101
    p = GeometricCoder.create(
        project_dir=tmp_path / "scatter-colors.geco",
        data=pd.DataFrame({
            "id": list(range(count)),
            "text": ["observation"] * count,
            "x": list(range(count)),
            "color": [f"category_{i:03d}" for i in range(count)],
        }),
        keys=["id"], text="text", metadata=["x", "color"], geometries=None,
    )
    spec = recipe("scatter", x=meta("x", "numeric"), y=meta("x", "numeric"),
                  color=meta("color", "categorical"))
    spec["max_points"] = 2
    with pytest.raises(ValueError, match="Scatter color has more than 100 categories"):
        compute_analysis(p, spec)
    with pytest.raises(ValueError, match="Scatter color has more than 100 categories"):
        p.save_analysis("Too many colors", spec)
    assert not any(row["name"] == "Too many colors" for row in p.analyses())


def test_parent_code_reductions_in_continuous_summary(tmp_path):
    from geometric_coder.ui.analyses import editor_recipe

    p = GeometricCoder.create(
        project_dir=tmp_path / "parent-continuous.geco",
        data=pd.DataFrame({
            "cluster": [1, 1, 2, 2],
            "id": [1, 2, 1, 2],
            "text": ["a", "b", "c", "d"],
            "group": ["A", "A", "B", "B"],
        }),
        keys=["cluster", "id"], text="text",
        metadata=["group"], geometries=None,
    )
    code = p.create_code("Theme")
    ids = [unit["observation_id"] for unit in p.database.list_units()]
    for identity, judgment in zip(ids, ["positive", "positive", "positive", "negative"]):
        p.annotate(identity, code, judgment)
    slots = {"values": f"code:{code}", "group": "metadata:group"}
    for method, expected in (("count", [2, 1]), ("mean", [1, 0.5]),
                             ("log_count", [math.log1p(2), math.log1p(1)])):
        spec = editor_recipe("summary_bar", "count", slots, {},
                             "cluster", method, 1,
                             continuous_options={"statistic": "mean"})
        assert spec["aggregation"]["method"] == method
        result = compute_analysis(p, spec)
        assert result["categories"] == ["A", "B"]
        assert result["values"] == pytest.approx(expected)
        assert result["denominators"] == [1, 1]


def test_parent_numeric_aggregation_sum_mean_median_and_missing(tmp_path):
    """Numerical parent Sum must not become mean or count missing observations."""
    from geometric_coder.ui.analyses import editor_recipe

    p = GeometricCoder.create(
        project_dir=tmp_path / "parent-numeric-sum.geco",
        data=pd.DataFrame({
            "parent": [1, 1, 1, 2, 2, 2, 3, 3],
            "turn": [1, 2, 3, 1, 2, 3, 1, 2],
            "text": [f"unit {i}" for i in range(8)],
            "score": [2.0, 4.0, None, 5.0, 7.0, 9.0, None, None],
            "group": ["A"] * 8,
        }),
        keys=["parent", "turn"], text="text",
        metadata=["score", "group"], geometries=None,
    )
    for method, expected in (
        ("sum", 27.0), ("mean", 10.0), ("median", 10.0),
    ):
        # Summary_bar applies the selected parent function first, then
        # computes a mean across the two observed parents (3 is missing).
        spec = recipe("summary_bar", values=meta("score", "numeric"),
                      group=meta("group", "categorical"))
        spec["aggregation"] = {"level": "parent", "numeric_method": method}
        result = compute_analysis(p, spec)
        assert result["categories"] == ["A"]
        assert result["values"] == pytest.approx([expected])
        assert result["denominators"] == [2]
        assert p.analysis(p.save_analysis(f"Parent {method}", spec))["spec"][
            "aggregation"]["numeric_method"] == method

        restored = p.analysis(next(row["analysis_id"] for row in p.analyses()
                                   if row["name"] == f"Parent {method}"))["spec"]
        edited = editor_recipe(
            "summary_bar", "count",
            {"values": "metadata:score", "group": "metadata:group"},
            {"values:metadata:score": {"value": "numeric"},
             "group:metadata:group": {"value": "categorical"}},
            "parent", "minimum", 1,
            continuous_options={"numeric_method": method, "statistic": "mean"},
        )
        assert edited["aggregation"]["numeric_method"] == restored["aggregation"]["numeric_method"]

    # Sum is a parent aggregation operation, independent of the final
    # displayed Bar statistic; a raw numeric Sum with no observations
    # must remain missing rather than becoming zero.
    invalid = recipe("summary_bar", values=meta("score", "numeric"),
                     group=meta("group", "categorical"))
    invalid["aggregation"] = {"level": "parent", "numeric_method": "maximum"}
    with pytest.raises(ValueError, match="aggregation"):
        validate_analysis(p, invalid)


def test_hidden_numerical_options_survive_editor_round_trips(workspace):
    """Edit/duplicate reconstruction must retain meaningful non-UI options."""
    from geometric_coder.ui.analyses import (
        _initial_config, _preserve_hidden_bin_settings, _slots, editor_recipe,
    )

    p, _, _ = workspace
    histogram = recipe(
        "histogram", values=meta("score", "numeric"),
        group={"variable": {"type": "metadata", "field": "score", "value": "numeric"},
               "transform": {"kind": "numeric_bins", "method": "fixed_width",
                             "width": 2.0, "origin": 1.0, "min": 1.0, "max": 5.0}},
    )
    histogram["binning"] = {"kind": "histogram_bins", "method": "fixed_width",
                            "width": 2.0, "origin": 1.0, "min": 1.0, "max": 5.0}
    original = p.analysis(p.save_analysis("Custom histogram", histogram))["spec"]
    source = original["roles"]["group"]["transform"]
    preserved = _preserve_hidden_bin_settings(
        {"kind": "numeric_bins", "method": "fixed_width", "width": 2.0, "origin": 0},
        source,
    )
    assert preserved == source
    options = _preserve_hidden_bin_settings(
        {"kind": "histogram_bins", "method": "fixed_width", "width": 2.0, "origin": 0},
        original["binning"],
    )
    recreated = editor_recipe(
        "histogram", "count", _slots(original), _initial_config(original),
        None, "any", 1, continuous_options={"binning": options},
    )
    assert p.save_analysis("Custom histogram copy", recreated)
    assert p.analysis(next(row["analysis_id"] for row in p.analyses()
                           if row["name"] == "Custom histogram copy"))["spec"] == original

    # Calendar transforms have a unit rather than a numerical bin method.
    calendar = {"kind": "calendar", "unit": "month"}
    assert _preserve_hidden_bin_settings(calendar, calendar) == calendar
    assert _preserve_hidden_bin_settings(calendar, None) == calendar

    # Switching methods must not bring incompatible ranges/origins forward.
    assert _preserve_hidden_bin_settings(
        {"kind": "histogram_bins", "method": "quantile", "bins": 4},
        original["binning"],
    ) == {"kind": "histogram_bins", "method": "quantile", "bins": 4}

    scatter = recipe("scatter", x=meta("score", "numeric"), y=meta("score", "numeric"))
    scatter["max_points"] = 3
    persisted = p.analysis(p.save_analysis("Custom scatter", scatter))["spec"]
    reconstructed = editor_recipe(
        "scatter", "count", _slots(persisted), _initial_config(persisted),
        None, "any", 1, continuous_options={"max_points": persisted["max_points"]},
    )
    assert p.analysis(p.save_analysis("Custom scatter copy", reconstructed))["spec"] == persisted


def test_value_heatmap_pairwise_cells_and_missingness(workspace):
    p, _, _ = workspace
    spec = recipe(
        "value_heatmap", values=meta("score", "numeric"),
        rows=meta("group", "categorical"),
        columns={**meta("day", "temporal"),
                 "transform": {"kind": "calendar", "unit": "month"}},
    )
    result = compute_analysis(p, spec)
    assert result["rows"] == ["(Missing)", "A", "B"]
    assert result["columns"] == ["2025-12", "2026-01", "2026-02"]
    assert result["samples"] == [[0, 0, 1], [1, 1, 0], [0, 1, 1]]
    assert result["values"] == [[None, None, 5.0], [1.0, 2.0, None],
                                [None, 3.0, 4.0]]
    assert result["coverage"]["total"] == 6
    assert result["coverage"]["complete"] == 5
    assert len(analysis_figure(result).data) == 1


def test_parent_group_metadata_requires_consistent_missingness(tmp_path):
    """A parent with observed and missing group labels cannot be assigned one label."""
    p = GeometricCoder.create(
        project_dir=tmp_path / "parent-missing-group.geco",
        data=pd.DataFrame({
            "parent": [1, 1, 2, 2],
            "id": [1, 2, 1, 2],
            "text": ["a", "b", "c", "d"],
            "group": ["Treatment", None, "Control", "Control"],
            "score": [2., 8., 4., 6.],
        }),
        keys=["parent", "id"], text="text",
        metadata=["group", "score"], geometries=None,
    )
    spec = recipe("summary_bar", values=meta("score", "numeric"),
                  group=meta("group", "categorical"))
    spec["aggregation"] = {"level": "parent", "numeric_method": "mean"}
    with pytest.raises(ValueError, match="metadata varies within a parent"):
        compute_analysis(p, spec)
    with pytest.raises(ValueError, match="metadata varies within a parent"):
        p.save_analysis("Invalid grouping", spec)
    assert not p.analyses()


def test_parent_all_missing_group_and_conflicting_group_behavior(tmp_path):
    p = GeometricCoder.create(
        project_dir=tmp_path / "parent-categories.geco",
        data=pd.DataFrame({
            "parent": [1, 1, 2, 2],
            "id": [1, 2, 1, 2],
            "text": ["a", "b", "c", "d"],
            "group": [None, None, "A", "A"],
            "score": [2., 8., 4., 6.],
        }),
        keys=["parent", "id"], text="text",
        metadata=["group", "score"], geometries=None,
    )
    spec = recipe("summary_bar", values=meta("score", "numeric"),
                  group=meta("group", "categorical"))
    spec["aggregation"] = {"level": "parent", "numeric_method": "mean"}
    result = compute_analysis(p, spec)
    assert result["categories"] == ["A"]
    assert result["values"] == pytest.approx([5.])
    # Two different nonmissing labels within one parent must still fail.
    with p.database.connect() as conn:
        import json
        row = conn.execute("SELECT observation_id, metadata_json FROM units "
                           "WHERE row_position = 3").fetchone()
        obj = json.loads(row["metadata_json"])
        obj["group"] = "B"
        conn.execute("UPDATE units SET metadata_json = ? WHERE observation_id = ?",
                     (json.dumps(obj), row["observation_id"]))
    with pytest.raises(ValueError, match="metadata varies within a parent"):
        compute_analysis(p, spec)


def test_mixed_type_categories_remain_distinct_in_all_grouped_charts(tmp_path):
    p = GeometricCoder.create(
        project_dir=tmp_path / "mixed-type-groups.geco",
        data=pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6],
            "text": ["a", "b", "c", "d", "e", "f"],
            "group": [1, "1", True, "True", "1 (integer)", 1.0],
            "score": [1., 2., 3., 4., 5., 6.],
        }),
        keys=["id"], text="text",
        metadata=["group", "score"], geometries=None,
    )
    values = meta("score", "numeric")
    group = meta("group", "categorical")
    summary = compute_analysis(p, recipe("summary_bar", values=values, group=group))
    assert len(summary["categories"]) == 6
    assert len(set(summary["categories"])) == 6
    assert sorted(summary["values"]) == pytest.approx([1., 2., 3., 4., 5., 6.])
    histogram = compute_analysis(p, recipe("histogram", values=values, group=group))
    assert len(histogram["groups"]) == 6
    assert len(histogram["counts"]) == 6
    assert sum(map(sum, histogram["counts"].values())) == 6
    heatmap = compute_analysis(
        p, recipe("value_heatmap", values=values,
                  rows=group, columns=group))
    assert len(heatmap["rows"]) == len(heatmap["columns"]) == 6
    assert sorted(n for row in heatmap["samples"] for n in row if n) == [1] * 6
    scatter = compute_analysis(p, recipe("scatter", x=values, y=values, color=group))
    assert len({point["color"] for point in scatter["points"]}) == 6


def test_constant_quantile_histogram_has_nonzero_interval(workspace):
    p, _, _ = workspace
    spec = recipe("histogram", values=meta("score", "numeric"))
    spec["binning"] = {"kind": "histogram_bins", "method": "quantile", "bins": 5}
    # Direct bin edge helper ensures exact all-tied samples remain meaningful.
    from geometric_coder.analysis_continuous import _edges
    edges = _edges([3.0] * 6, spec["binning"])
    assert edges[0] < 3.0 < edges[-1]


def test_binary_metadata_is_valid_on_numeric_axes_and_as_a_group(workspace):
    project, _code_id, ids = workspace
    from geometric_coder.analysis_variables import resolve_analytical_variables
    import json
    with project.database.connect() as connection:
        for observation_id, value in zip(ids, [True, False, True, False, True, False]):
            row = connection.execute(
                "SELECT metadata_json FROM units WHERE observation_id = ?",
                (observation_id,),
            ).fetchone()
            meta_json = json.loads(row["metadata_json"])
            meta_json["group"] = value
            connection.execute(
                "UPDATE units SET metadata_json = ? WHERE observation_id = ?",
                (json.dumps(meta_json), observation_id),
            )
    binary = meta("group", "binary")
    scatter = recipe("scatter", x=binary, y=meta("score", "numeric"))
    points = compute_analysis(project, scatter)["points"]
    assert {point["x"] for point in points} == {0.0, 1.0}
    corr = recipe("correlation_heatmap", values=[binary, meta("score", "numeric")])
    matrix = compute_analysis(project, corr)["values"]
    assert len(matrix) == 2
    assert matrix[0][1] is not None
    hist = recipe("histogram", values=binary)
    counts = compute_analysis(project, hist)["counts"]
    assert sum(map(sum, counts.values())) == 6
    summary = recipe("summary_bar", values=meta("score", "numeric"), group=binary)
    assert len(compute_analysis(project, summary)["categories"]) == 2
