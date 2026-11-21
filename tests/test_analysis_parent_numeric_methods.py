"""Regression coverage for numerical parent-grain aggregation functions."""
import math

import pandas as pd
import pytest

from geometric_coder import GeometricCoder
from geometric_coder.analyses import compute_analysis, validate_analysis
from geometric_coder.ui.analyses import editor_recipe


@pytest.fixture
def workspace(tmp_path):
    project = GeometricCoder.create(
        project_dir=tmp_path / "numeric-parent-methods.geco",
        data=pd.DataFrame({
            "parent": [1, 1, 1, 1, 2, 2, 2, 3, 3],
            "turn": [1, 2, 3, 4, 1, 2, 3, 1, 2],
            "text": [f"unit {i}" for i in range(9)],
            "score": [1., 2., 3., None, 4., 6., 8., None, None],
            "group": ["A"] * 9,
        }),
        keys=["parent", "turn"], text="text",
        metadata=["score", "group"], geometries=None,
    )
    return project


def numeric_spec(method, **settings):
    return {
        "schema_version": 2, "kind": "summary_bar",
        "roles": {
            "values": {"variable": {"type": "metadata", "field": "score", "value": "numeric"}},
            "group": {"variable": {"type": "metadata", "field": "group", "value": "categorical"}},
        },
        "aggregation": {"level": "parent", "numeric_method": method, **settings},
    }


@pytest.mark.parametrize(("method", "settings", "expected"), [
    ("mean", {}, 4.),
    ("median", {}, 4.),
    ("sum", {}, 12.),
    ("minimum", {}, 2.5),
    ("maximum", {}, 5.5),
    ("count", {}, 3.),
    ("standard_deviation", {}, (math.sqrt(2 / 3) + math.sqrt(8 / 3)) / 2),
    ("percentile", {"numeric_percentile": 25}, 3.25),
    ("percentile", {"numeric_percentile": 0}, 2.5),
    ("percentile", {"numeric_percentile": 100}, 5.5),
    ("range", {}, 3.),
    ("proportion_within_bounds", {"numeric_bounds": {
        "lower": 2, "lower_inclusive": True, "upper": 6, "upper_inclusive": False,
    }}, .5),
    ("proportion_within_bounds", {"numeric_bounds": {
        "lower": 2, "lower_inclusive": False, "upper": 6, "upper_inclusive": True,
    }}, .5),
    ("proportion_within_bounds", {"numeric_bounds": {"upper": 2}}, 1 / 3),
    ("proportion_within_bounds", {"numeric_bounds": {"lower": 7}}, 1 / 6),
    ("proportion_within_bounds", {"numeric_bounds": {
        "lower": 2, "upper": 2,
    }}, 1 / 6),
])
def test_numerical_parent_methods(workspace, method, settings, expected):
    spec = numeric_spec(method, **settings)
    result = compute_analysis(workspace, spec)
    assert result["categories"] == ["A"]
    assert result["denominators"] == [2]  # All-missing parent is not a zero.
    assert result["values"] == pytest.approx([expected])
    saved = workspace.save_analysis(f"Parent {method}", spec)
    assert workspace.analysis(saved)["spec"]["aggregation"] == validate_analysis(
        workspace, spec,
    )["aggregation"]


@pytest.mark.parametrize(("settings", "message"), [
    ({"numeric_method": "unknown"}, "aggregation"),
    ({"numeric_method": "percentile", "numeric_percentile": -1}, "percentile"),
    ({"numeric_method": "percentile", "numeric_percentile": 101}, "percentile"),
    ({"numeric_method": "percentile", "numeric_percentile": float("nan")}, "percentile"),
    ({"numeric_method": "mean", "numeric_percentile": 50}, "Percentile"),
    ({"numeric_method": "mean", "numeric_bounds": {"lower": 1}}, "Bounds"),
    ({"numeric_method": "proportion_within_bounds"}, "bound"),
    ({"numeric_method": "proportion_within_bounds", "numeric_bounds": {}}, "bound"),
    ({"numeric_method": "proportion_within_bounds", "numeric_bounds": {"lower": 5, "upper": 3}}, "Lower"),
    ({"numeric_method": "proportion_within_bounds", "numeric_bounds": {"lower": None}}, "finite"),
    ({"numeric_method": "proportion_within_bounds", "numeric_bounds": {"upper": float("inf")}}, "finite"),
    ({"numeric_method": "proportion_within_bounds", "numeric_bounds": {"lower": 1, "lower_inclusive": 0}}, "boolean"),
    ({"numeric_method": "proportion_within_bounds", "numeric_bounds": {"lower_inclusive": True, "upper": 1}}, "corresponding"),
])
def test_invalid_parent_settings(workspace, settings, message):
    method = settings.pop("numeric_method")
    with pytest.raises(ValueError, match=message):
        validate_analysis(workspace, numeric_spec(method, **settings))


@pytest.mark.parametrize("method,options", [
    ("percentile", {"numeric_percentile": 75}),
    ("proportion_within_bounds", {"numeric_lower": 2, "numeric_upper": 6,
                                  "numeric_lower_inclusive": False, "numeric_upper_inclusive": True}),
    ("proportion_within_bounds", {"numeric_upper": 4, "numeric_upper_inclusive": False}),
])
def test_editor_round_trip_numeric_parent_settings(workspace, method, options):
    spec = editor_recipe(
        "summary_bar", "count",
        {"values": "metadata:score", "group": "metadata:group"},
        {"values:metadata:score": {"value": "numeric"},
         "group:metadata:group": {"value": "categorical"}},
        "parent", "minimum", 1,
        continuous_options={"numeric_method": method, "statistic": "mean", **options},
    )
    aggregation = validate_analysis(workspace, spec)["aggregation"]
    assert aggregation["numeric_method"] == method
    assert (aggregation.get("numeric_percentile") == 75) if method == "percentile" else (
        "numeric_bounds" in aggregation
    )


def test_unified_bar_accepts_same_parent_contract(workspace):
    spec = {
        "schema_version": 2, "kind": "bar_chart",
        "roles": {"values": [{"variable": {"type": "metadata", "field": "score", "value": "numeric"}}]},
        "statistic": "mean",
        "aggregation": {"level": "parent", "numeric_method": "proportion_within_bounds",
                        "numeric_bounds": {"lower": 2, "upper": 6, "upper_inclusive": False}},
    }
    result = compute_analysis(workspace, spec)
    assert result["series"][0]["values"] == pytest.approx([.5])
    assert result["series"][0]["denominators"] == [2]
