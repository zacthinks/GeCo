"""Phase 2 metadata-filter predicate equivalence and edge cases."""

import numpy as np
import pytest

from geometric_coder.ui.app import (
    _evaluate_metadata_filters, _metadata_value_matches,
)


@pytest.mark.parametrize(
    ("kind", "condition", "value", "value2", "raw_values"),
    [
        ("categorical", "in", ["A", "b"], None, ["a", "B", "c", None, ""]),
        ("categorical", "not_in", ["A", "b"], None, ["a", "B", "c", None, ""]),
        ("text", "contains", "AB", None, ["abacus", "CAB", "no", None]),
        ("text", "excludes", "AB", None, ["abacus", "CAB", "no", None]),
        ("text", "eq", "AB", None, ["ab", "a", "AB", None]),
        ("text", "neq", "AB", None, ["ab", "a", "AB", None]),
        ("text", "starts", "AB", None, ["abacus", "CAB", "no", None]),
        ("text", "ends", "AB", None, ["AB", "CAB", "no", None]),
        ("text", "regex", "^a.*z$", None, ["abz", "ABZ", "az", "no", None]),
        ("numeric", "between", "2", "5", [1, 2, 4, 6, None, np.nan]),
        ("numeric", "lt", "3", None, [1, 2, 4, None, np.nan]),
        ("numeric", "gte", "3", None, [1, 3, 4, None, np.nan]),
        ("datetime", "between", "2026-01-01", "2026-02-01",
         ["2025-12-31", "2026-01-15", "2026-02-03", None]),
        ("datetime", "lt", "2026-02-01", None,
         ["2026-01-15", "2026-02-03", None]),
        ("boolean", "true", None, None, [True, False, 0, 1, None]),
        ("boolean", "false", None, None, [True, False, 0, 1, None]),
        ("text", "missing", None, None, ["a", "", None, np.nan]),
        ("text", "not_missing", None, None, ["a", "", None, np.nan]),
    ],
)
def test_prepared_clause_matches_original(kind, condition, value, value2, raw_values):
    units = [{"metadata": {"x": raw}} for raw in raw_values]
    actual, errors = _evaluate_metadata_filters(units, [{
        "field": "x", "type": kind, "operator": condition,
        "value": value, "value2": value2,
    }], operator="and")
    expected = [
        _metadata_value_matches(raw, kind=kind, condition=condition,
                                value=value, value2=value2)
        for raw in raw_values
    ]
    assert errors == []
    assert actual.tolist() == expected


@pytest.mark.parametrize("operator", ["and", "or", "xor"])
def test_combined_filters_preserve_boolean_logic(operator):
    units = [{"metadata": {"color": "A" if i % 2 else "B", "score": i}}
             for i in range(12)]
    filters = [
        {"field": "color", "type": "categorical", "operator": "in",
         "value": ["a"]},
        {"field": "score", "type": "numeric", "operator": "gte",
         "value": "5"},
        {"field": "score", "type": "numeric", "operator": "lt",
         "value": "9"},
    ]
    actual, errors = _evaluate_metadata_filters(units, filters, operator=operator)
    clause_values = [
        [_metadata_value_matches(row["metadata"].get(clause["field"]),
                                 kind=clause["type"], condition=clause["operator"],
                                 value=clause["value"], value2=None)
         for row in units]
        for clause in filters
    ]
    expected = []
    for columns in zip(*clause_values):
        expected.append(
            all(columns) if operator == "and"
            else any(columns) if operator == "or"
            else sum(columns) == 1
        )
    assert errors == []
    assert actual.tolist() == expected


def test_invalid_regex_and_numeric_errors_match_prior_behavior():
    for kind, condition, invalid in [
        ("text", "regex", "["),
        ("numeric", "gte", "not a number"),
        ("datetime", "lt", "not a date"),
    ]:
        filters = [{"field": "x", "type": kind, "operator": condition,
                    "value": invalid}]
        missing_only, errors = _evaluate_metadata_filters(
            [{"metadata": {"x": None}}], filters, operator="and"
        )
        assert missing_only.tolist() == [False]
        assert errors == []
        nonmissing, errors = _evaluate_metadata_filters(
            [{"metadata": {"x": "2026-01-01" if kind == "datetime" else "3"}}],
            filters, operator="and"
        )
        assert nonmissing.tolist() == [False]
        assert len(errors) == 1 and errors[0].startswith("x: ")
