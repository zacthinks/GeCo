"""Researcher-facing unified Bar chart regressions.

Legacy saved analyses stay on their original engines; new charts use typed v2
recipes with exact counts/sums, observed denominators and bounded median paths.
"""
from __future__ import annotations

import pandas as pd
import pytest

from geometric_coder import GeometricCoder
from geometric_coder.analyses import analysis_figure, compute_analysis, validate_analysis


@pytest.fixture
def workspace(tmp_path):
    project = GeometricCoder.create(
        project_dir=tmp_path / "bar.geco",
        data=pd.DataFrame({
            "id": list(range(1, 7)),
            "text": [f"observation {i}" for i in range(1, 7)],
            "condition": ["A", "A", "A", "B", "B", "B"],
            "site": ["North", "South", "North", "North", "South", "North"],
            "rating": [0.0, 2.0, None, 6.0, 8.0, None],
        }),
        keys=["id"], text="text", metadata=["condition", "site", "rating"],
        geometries=None,
    )
    code = project.create_code("Positive experience")
    ids = [row["observation_id"] for row in project.database.list_units()]
    for observation_id, label in zip(
        ids, ["positive", "negative", "unsure", "negative", "positive", None]
    ):
        if label is not None:
            project.annotate(observation_id, code, label)
    return project, code


def meta(field, value="categorical"):
    return {"type": "metadata", "field": field, "value": value}


def code_value(code_id):
    return {"type": "code", "code_id": code_id, "source": "labels", "value": "binary"}


def chart(values, statistic="count", group=None, color=None):
    roles = {"values": [{"variable": v} for v in values]}
    if group is not None:
        roles["group"] = {"variable": group}
    if color is not None:
        roles["color"] = {"variable": color}
    return {"schema_version": 2, "kind": "bar_chart", "roles": roles,
            "statistic": statistic}


def test_categorical_frequency_with_no_code_required(workspace):
    p, _ = workspace
    result = compute_analysis(p, chart([meta("condition")]))
    assert result["categories"] == ["A", "B"]
    assert result["series"][0]["values"] == [3, 3]
    assert result["coverage"]["total"] == 6
    assert result["coverage"]["observed"] == [6]
    assert len(analysis_figure(result).data) == 1
    assert result["xaxis_title"] == "condition"
    assert analysis_figure(result).layout.xaxis.title.text == "condition"


def test_binary_code_sum_is_positive_count_with_real_denominators(workspace):
    p, code = workspace
    result = compute_analysis(p, chart([code_value(code)], "sum", meta("condition")))
    assert result["categories"] == ["A", "B"]
    assert result["xaxis_title"] == "condition"
    assert analysis_figure(result).layout.xaxis.title.text == "condition"
    assert result["series"][0]["values"] == [1, 1]
    assert result["series"][0]["denominators"] == [2, 2]
    assert result["coverage"]["observed"][0] == 4
    assert sum(result["sources"][0]["missing_reasons"].values()) == 2


def test_binary_distribution_has_only_discrete_categories(workspace):
    p, code = workspace
    result = compute_analysis(p, chart([code_value(code)]))
    assert result["categories"] == ["Absent", "Present"]
    assert result["series"][0]["values"] == [2, 2]
    fig = analysis_figure(result)
    assert fig.layout.xaxis.type == "category"
    assert fig.layout.xaxis.title.text == "Positive experience"
    assert list(fig.data[0].x) == ["Absent", "Present"]
    assert 0.5 not in fig.data[0].x


def test_numeric_mean_preserves_zero_and_missing(workspace):
    p, _ = workspace
    result = compute_analysis(p, chart([meta("rating", "numeric")], "mean", meta("condition")))
    assert result["categories"] == ["A", "B"]
    assert result["series"][0]["values"] == pytest.approx([1.0, 7.0])
    assert result["series"][0]["denominators"] == [2, 2]
    assert result["coverage"]["observed"][0] == 4
    assert result["coverage"]["total"] == 6


def test_multiple_groupings_form_distinct_series_and_counts(workspace):
    p, _ = workspace
    result = compute_analysis(
        p, chart([meta("condition")], group=meta("site"))
    )
    assert set(result["categories"]) == {"North", "South"}
    labels = {item["name"] for item in result["series"]}
    assert any("A" in label for label in labels)
    assert any("B" in label for label in labels)
    assert sum(sum(s["values"]) for s in result["series"]) == 6


def test_color_splits_numerical_summary_without_changing_totals(workspace):
    p, _ = workspace
    result = compute_analysis(p, chart(
        [meta("rating", "numeric")], "sum", meta("condition"), meta("site")
    ))
    assert len(result["series"]) == 2
    assert sum(sum(s["values"]) for s in result["series"]) == pytest.approx(16.0)


def test_non_numeric_sum_and_unsupported_transform_rejected(workspace):
    p, code = workspace
    with pytest.raises(ValueError, match="numerical"):
        validate_analysis(p, chart([meta("condition")], "sum"))
    recipe = chart([code_value(code)])
    recipe["roles"]["group"] = {
        "variable": meta("rating", "numeric")
    }
    with pytest.raises(ValueError, match="numeric binning"):
        validate_analysis(p, recipe)


def test_legacy_bar_and_distribution_still_execute_unchanged(workspace):
    p, code = workspace
    v1 = {"kind": "bar", "measure": "count",
          "variables": [code_value(code), {"type": "metadata", "field": "condition"}]}
    v2 = {"schema_version": 2, "kind": "bar", "measure": "count",
          "roles": {"values": [{"variable": code_value(code)}],
                    "group": {"variable": meta("condition")}}}
    assert compute_analysis(p, v1) == compute_analysis(p, v2)
    old_dist = {"kind": "distribution", "measure": "count", "variables": [code_value(code)]}
    assert compute_analysis(p, old_dist)["histograms"][0]["counts"] == [2, 2]
    assert [v["kind"] for v in (p.analysis(
        p.save_analysis(f"Old {i}", spec))["spec"] for i, spec in enumerate((v1, old_dist))
    )] == ["bar", "distribution"]

def test_code_and_numeric_metadata_with_identical_names_remain_separate(workspace):
    p, _ = workspace
    code = p.create_code("rating")
    units = [unit["observation_id"] for unit in p.database.list_units()]
    for index, label in enumerate(("positive", "negative", "positive",
                                  "negative", "positive", "negative")):
        p.annotate(units[index], code, label)
    result = compute_analysis(
        p, chart([code_value(code), meta("rating", "numeric")],
                 "sum", meta("condition"))
    )
    assert result["categories"] == ["A", "B"]
    assert len(result["series"]) == 2
    by_name = {row["name"]: row for row in result["series"]}
    assert set(by_name) == {"rating (Code)", "rating (Metadata)"}
    assert by_name["rating (Code)"]["values"] == [2, 1]
    assert by_name["rating (Code)"]["denominators"] == [3, 3]
    assert by_name["rating (Metadata)"]["values"] == pytest.approx([2.0, 14.0])
    assert by_name["rating (Metadata)"]["denominators"] == [2, 2]


def test_total_variable_limit_includes_both_grouping_roles(workspace):
    p, first_code = workspace
    # The limit test needs genuinely distinct descriptors: repeated Values
    # are invalid regardless of whether the total stays below 20.
    codes = [first_code] + [p.create_code(f"Other value {i}") for i in range(18)]
    valid = chart([code_value(code) for code in codes[:18]], "sum",
                  meta("condition"), meta("site"))
    assert len(validate_analysis(p, valid)["roles"]["values"]) == 18
    invalid = chart([code_value(code) for code in codes], "sum",
                    meta("condition"), meta("site"))
    with pytest.raises(ValueError, match="21 analytical variables.*maximum is 20"):
        validate_analysis(p, invalid)


@pytest.mark.parametrize("statistic", ["count", "sum", "mean", "median"])
def test_duplicate_code_values_are_rejected_before_execution(workspace, statistic):
    p, code = workspace
    duplicated = chart([code_value(code), code_value(code)],
                       statistic, meta("condition"))
    with pytest.raises(ValueError, match="Values variable may be selected only once"):
        validate_analysis(p, duplicated)
    with pytest.raises(ValueError, match="Values variable may be selected only once"):
        compute_analysis(p, duplicated)
    with pytest.raises(ValueError, match="Values variable may be selected only once"):
        p.save_analysis(f"Invalid repeated Values {statistic}", duplicated)


def test_duplicate_metadata_values_rejected_but_same_variable_can_group(workspace):
    p, _ = workspace
    with pytest.raises(ValueError, match="Values variable may be selected only once"):
        validate_analysis(p, chart([meta("rating", "numeric")] * 2,
                                   "sum", meta("condition")))
    # Role reuse is legitimate; only repetition within Values is rejected.
    result = compute_analysis(p, chart([meta("condition")], group=meta("condition")))
    assert result["categories"] == ["A", "B"]
    assert sum(sum(series["values"]) for series in result["series"]) == 6


def test_variable_identity_distinguishes_prediction_source_and_representation():
    from geometric_coder.analysis_bars import _variable_identity
    descriptor = {"type": "code", "code_id": 7, "source": "labels", "value": "binary"}
    predicted = {**descriptor, "source": "classifier", "predictor_id": 21}
    probability = {**predicted, "value": "probability"}
    assert len({_variable_identity(value) for value in (descriptor, predicted, probability)}) == 3


def test_multiple_categorical_values_without_group_have_generic_value_axis(workspace):
    p, code = workspace
    result = compute_analysis(p, chart([code_value(code), meta("condition")]))
    assert result["xaxis_title"] == "Value"
    assert analysis_figure(result).layout.xaxis.title.text == "Value"


def test_numerical_values_without_group_have_values_axis(workspace):
    p, _ = workspace
    result = compute_analysis(p, chart([meta("rating", "numeric")], "sum"))
    assert result["xaxis_title"] == "Values"
    assert analysis_figure(result).layout.xaxis.title.text == "Values"
