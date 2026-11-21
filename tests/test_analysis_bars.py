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
    assert result["categories"] == ["condition"]
    assert result["series"][0]["values"] == [6]
    assert result["coverage"]["total"] == 6
    assert result["coverage"]["observed"] == [6]
    assert len(analysis_figure(result).data) == 1
    assert result["xaxis_title"] == "Values"
    assert analysis_figure(result).layout.xaxis.title.text == "Values"
    grouped = compute_analysis(p, chart([meta("condition")], group=meta("condition")))
    assert grouped["categories"] == ["A", "B"]
    assert grouped["series"][0]["values"] == [3, 3]


def test_binary_metadata_count_is_category_frequency_with_optional_roles(workspace):
    project, _ = workspace
    import json

    ids = [row["observation_id"] for row in project.database.list_units()]
    raw = [True, False, True, False, True, False]
    with project.database.connect() as conn:
        for observation_id, value in zip(ids, raw, strict=True):
            row = conn.execute(
                "SELECT metadata_json FROM units WHERE observation_id = ?",
                (observation_id,),
            ).fetchone()
            metadata = json.loads(row["metadata_json"])
            metadata["rating"] = value
            conn.execute(
                "UPDATE units SET metadata_json = ? WHERE observation_id = ?",
                (json.dumps(metadata), observation_id),
            )

    binary = meta("rating", "binary")
    plain = compute_analysis(project, chart([binary], "count"))
    assert plain["categories"] == ["rating"]
    assert plain["series"][0]["values"] == [6]
    assert plain["series"][0]["denominators"] == [6]
    explicitly_grouped = compute_analysis(project, chart([binary], "count", group=binary))
    assert explicitly_grouped["categories"] == ["0.0", "1.0"]
    assert explicitly_grouped["series"][0]["values"] == [3, 3]

    grouped = compute_analysis(project, chart(
        [binary], "count", group=meta("condition"), color=meta("site"),
    ))
    assert grouped["categories"] == ["A", "B"]
    assert len(grouped["series"]) == 2
    assert {series["name"] for series in grouped["series"]} == {
        "rating / North", "rating / South",
    }
    assert sum(sum(series["values"]) for series in grouped["series"]) == 6

    summed = compute_analysis(project, chart([binary], "sum"))
    assert summed["series"][0]["values"] == [3.0]
    assert summed["series"][0]["denominators"] == [6]


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
    assert result["categories"] == ["Positive experience"]
    assert result["series"][0]["values"] == [4]
    assert result["series"][0]["denominators"] == [4]
    fig = analysis_figure(result)
    assert fig.layout.xaxis.type == "category"
    assert fig.layout.xaxis.title.text == "Values"
    assert list(fig.data[0].x) == ["Positive experience"]
    grouped = compute_analysis(p, chart([code_value(code)], group=code_value(code)))
    assert grouped["categories"] == ["Absent", "Present"]
    assert grouped["series"][0]["values"] == [2, 2]
    colored = compute_analysis(p, chart([code_value(code)], color=code_value(code)))
    assert set(s["name"] for s in colored["series"]) == {
        "Positive experience / Absent", "Positive experience / Present"
    }
    assert sum(sum(s["values"]) for s in colored["series"]) == 4


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
    assert len(result["series"]) == 1
    assert result["series"][0]["name"] == "condition"
    assert result["series"][0]["values"] == [4, 2]
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
    assert result["xaxis_title"] == "Values"
    assert analysis_figure(result).layout.xaxis.title.text == "Values"
    assert result["categories"] == ["Positive experience", "condition"]
    assert [v for series in result["series"] for v in series["values"] if v] == [4, 6]


def test_numerical_values_without_group_have_values_axis(workspace):
    p, _ = workspace
    result = compute_analysis(p, chart([meta("rating", "numeric")], "sum"))
    assert result["xaxis_title"] == "Values"
    assert analysis_figure(result).layout.xaxis.title.text == "Values"


def test_categorical_count_with_group_and_color_computes_and_saves(workspace):
    project, _ = workspace
    spec = chart([meta("condition")], "count", meta("site"), meta("condition"))
    computed = compute_analysis(project, spec)
    assert computed["categories"] == ["North", "South"]
    by_name = {series["name"]: series for series in computed["series"]}
    assert set(by_name) == {
        "condition / A", "condition / B",
    }
    assert by_name["condition / A"]["values"] == [2, 1]
    assert by_name["condition / B"]["values"] == [2, 1]
    assert by_name["condition / A"]["denominators"] == [2, 1]
    assert len(analysis_figure(computed).data) == 2
    saved_id = project.save_analysis("Categorical Group and Color", spec)
    assert project.analysis(saved_id)["spec"]["roles"] == validate_analysis(
        project, spec
    )["roles"]
    assert project.run_analysis(saved_id)["series"] == computed["series"]


def test_sum_missing_cells_remain_null_but_observed_zero_is_zero(workspace):
    project, code = workspace
    spec = chart([code_value(code)], "sum", meta("condition"), meta("site"))
    result = compute_analysis(project, spec)
    assert result["categories"] == ["A", "B"]
    by_name = {row["name"]: row for row in result["series"]}
    # A/North has one Present and one Unsure; B/North has an observed
    # Absent and an unreviewed judgment. A/South is observed Absent.
    # B/South is observed Present. All four cells have evidence.
    assert by_name["Positive experience / North"]["values"] == [1.0, 0.0]
    assert by_name["Positive experience / South"]["values"] == [0.0, 1.0]
    assert by_name["Positive experience / North"]["denominators"] == [1, 1]
    assert by_name["Positive experience / South"]["denominators"] == [1, 1]

    # Make B/North wholly missing while retaining the group and color series
    # elsewhere. No B/North evidence must produce None, not numeric zero.
    units = project.database.list_units()
    project.annotate(units[3]["observation_id"], code, "unsure")
    missing = compute_analysis(project, spec)
    by_name = {row["name"]: row for row in missing["series"]}
    assert by_name["Positive experience / North"]["values"] == [1.0, None]
    assert by_name["Positive experience / North"]["denominators"] == [1, 0]
    assert by_name["Positive experience / South"]["values"] == [0.0, 1.0]
    assert by_name["Positive experience / South"]["denominators"] == [1, 1]
    fig = analysis_figure(missing)
    north = next(trace for trace in fig.data if trace.name == "Positive experience / North")
    assert list(north.y) == [1.0, None]
    assert list(north.customdata) == [1, 0]

    saved_id = project.save_analysis("Missing-aware Sum", spec)
    assert project.run_analysis(saved_id)["series"] == missing["series"]


def test_count_by_condition_is_observed_count_not_implicit_code_labels(workspace):
    project, code = workspace
    spec = chart([code_value(code)], "count", group=meta("condition"))
    result = compute_analysis(project, spec)
    assert result["categories"] == ["A", "B"]
    assert len(result["series"]) == 1
    assert result["series"][0]["values"] == [2, 2]
    assert result["series"][0]["denominators"] == [2, 2]
    # An Unsure and an unreviewed Code judgment are not observed.
    assert result["coverage"]["observed"][0] == 4
    assert compute_analysis(
        project, chart([code_value(code)], "sum", group=meta("condition"))
    )["series"][0]["values"] == [1.0, 1.0]


def test_count_allows_mixed_categorical_and_numeric_values(workspace):
    project, code = workspace
    result = compute_analysis(project, chart([
        code_value(code), meta("condition"), meta("rating", "numeric"),
    ], "count", group=meta("condition")))
    assert result["categories"] == ["A", "B"]
    by_name = {row["name"]: row for row in result["series"]}
    assert by_name["Positive experience"]["values"] == [2, 2]
    assert by_name["condition"]["values"] == [3, 3]
    assert by_name["rating"]["values"] == [2, 2]


def test_toy_fixture_count_17_present_15_absent_per_condition(tmp_path):
    conditions = ["Control", "Dialogue", "Hints"]
    rows = []
    labels = []
    for condition in conditions:
        for i in range(40):
            rows.append({"id": len(rows) + 1, "text": f"{condition} {i}",
                         "condition": condition})
            labels.append("positive" if i < 17 else "negative" if i < 32
                          else "unsure" if i < 34 else None)
    project = GeometricCoder.create(
        project_dir=tmp_path / "toy-count.geco",
        data=pd.DataFrame(rows),
        keys=["id"], text="text", metadata=["condition"], geometries=None,
    )
    code = project.create_code("Positive experience")
    for row, label in zip(project.database.list_units(), labels, strict=True):
        if label is not None:
            project.annotate(row["observation_id"], code, label)
    by_condition = chart([code_value(code)], "count", group=meta("condition"))
    result = compute_analysis(project, by_condition)
    assert result["categories"] == conditions
    assert len(result["series"]) == 1
    assert result["series"][0]["values"] == [32, 32, 32]
    assert result["series"][0]["denominators"] == [32, 32, 32]
    assert result["coverage"]["observed"][0] == 96
    summed = compute_analysis(
        project, chart([code_value(code)], "sum", group=meta("condition"))
    )
    assert summed["series"][0]["values"] == [17.0, 17.0, 17.0]
    # Code categories appear only with an explicit Group or Color role.
    colored = compute_analysis(
        project, chart([code_value(code)], "count",
                       group=meta("condition"), color=code_value(code))
    )
    assert {s["name"] for s in colored["series"]} == {
        "Positive experience / Present", "Positive experience / Absent"
    }
    assert {tuple(s["values"]) for s in colored["series"]} == {
        (17, 17, 17), (15, 15, 15)
    }
    saved_id = project.save_analysis("Observed labels by condition", by_condition)
    assert project.run_analysis(saved_id)["series"] == result["series"]


def test_parent_positive_counts_keep_numeric_group_labels(tmp_path):
    """Parent counts of 0 and 1 are numeric, not atomic Absent/Present labels."""
    project = GeometricCoder.create(
        project_dir=tmp_path / "parent-bar.geco",
        data=pd.DataFrame({
            "interview": ["A"] * 3 + ["B"] * 3 + ["C"] * 3,
            "turn": [1, 2, 3] * 3,
            "text": [f"observation {i}" for i in range(9)],
        }),
        keys=["interview", "turn"], text="text", metadata=[], geometries=None,
    )
    code = project.create_code("Positive experience")
    for unit, label in zip(
        project.database.list_units(),
        ["negative", "negative", "negative",
         "positive", "negative", "negative",
         "positive", "positive", "negative"],
        strict=True,
    ):
        project.annotate(unit["observation_id"], code, label)
    spec = chart([code_value(code)], group=code_value(code))
    spec["aggregation"] = {
        "level": "interview", "method": "count",
        "threshold": 1, "numeric_method": "mean",
    }
    result = compute_analysis(project, spec)
    assert result["categories"] == ["0.0", "1.0", "2.0"]
    assert result["series"][0]["values"] == [1, 1, 1]
    assert "Absent" not in result["categories"]
    assert "Present" not in result["categories"]

    # Truly binary parent reductions retain the readable labels.
    spec["aggregation"]["method"] = "any"
    binary_result = compute_analysis(project, spec)
    assert binary_result["categories"] == ["Absent", "Present"]
    assert binary_result["series"][0]["values"] == [1, 2]


def test_parent_categorical_count_values_need_presence_not_constant_category(tmp_path):
    """Count Values may vary within a parent; grouping metadata may not."""
    project = GeometricCoder.create(
        project_dir=tmp_path / "categorical-parents.geco",
        data=pd.DataFrame({
            "interview": ["A"] * 3 + ["B"] * 3 + ["C"] * 3,
            "turn": [1, 2, 3] * 3,
            "text": [f"observation {i}" for i in range(9)],
            "topic": ["helpful", None, "difficult", None, None, None,
                      "difficult", "helpful", None],
            "condition": ["Control"] * 6 + ["Treatment"] * 3,
        }),
        keys=["interview", "turn"], text="text",
        metadata=["topic", "condition"], geometries=None,
    )
    spec = chart([meta("topic")], group=meta("condition"))
    spec["aggregation"] = {
        "level": "interview", "method": "any",
        "threshold": 1, "numeric_method": "mean",
    }
    result = compute_analysis(project, spec)
    assert result["categories"] == ["Control", "Treatment"]
    # A and C each have observed topics, B is wholly missing.
    assert result["series"][0]["values"] == [1, 1]
    assert result["series"][0]["denominators"] == [1, 1]

    # Removing Group should count exactly the two observed parents.
    del spec["roles"]["group"]
    ungrouped = compute_analysis(project, spec)
    assert ungrouped["series"][0]["values"] == [2]

    # The Group role still enforces a stable category inside each parent.
    spec["roles"]["group"] = {"variable": meta("topic")}
    with pytest.raises(ValueError, match="grouping metadata varies within a parent"):
        compute_analysis(project, spec)
