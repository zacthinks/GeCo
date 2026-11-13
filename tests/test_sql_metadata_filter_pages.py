"""SQL metadata filter equivalence and bounded page regression tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geometric_coder import GeometricCoder
from geometric_coder.ui.app import _evaluate_metadata_filters, _sql_metadata_page


@pytest.fixture
def project(tmp_path):
    n = 120
    data = pd.DataFrame({
        "key": list(range(n)),
        "text": [f"Text {i}" for i in range(n)],
        "category": ["Green" if i % 3 == 0 else "Blue" for i in range(n)],
        "score": [i * 0.5 for i in range(n)],
        "note": ["Hi FRIEND" if i % 4 == 0 else "Other" for i in range(n)],
        "active": [i % 2 == 0 for i in range(n)],
    })
    return GeometricCoder.create(
        project_dir=tmp_path / "filtered.geco", data=data, keys=["key"],
        text="text", metadata=["category", "score", "note", "active"],
        geometries=None,
    )


@pytest.mark.parametrize(
    "filters,operator",
    [
        ([{"field": "category", "type": "categorical", "operator": "in", "value": ["green"]}], "and"),
        ([{"field": "category", "type": "categorical", "operator": "not_in", "value": ["BLUE"]}], "and"),
        ([{"field": "score", "type": "numeric", "operator": "between", "value": "4", "value2": "13"}], "and"),
        ([{"field": "score", "type": "numeric", "operator": "gte", "value": "35"}], "and"),
        ([{"field": "note", "type": "text", "operator": "contains", "value": "friend"}], "and"),
        ([{"field": "note", "type": "text", "operator": "starts", "value": "hi"}], "and"),
        ([{"field": "note", "type": "text", "operator": "ends", "value": "other"}], "and"),
        ([{"field": "note", "type": "text", "operator": "regex", "value": "^hi.*end$"}], "and"),
        ([{"field": "active", "type": "boolean", "operator": "true"}], "and"),
        ([{"field": "active", "type": "boolean", "operator": "false"}], "and"),
        ([{"field": "category", "type": "categorical", "operator": "missing"}], "and"),
        ([{"field": "category", "type": "categorical", "operator": "not_missing"}], "and"),
        ([
            {"field": "category", "type": "categorical", "operator": "in", "value": ["Green"]},
            {"field": "active", "type": "boolean", "operator": "true"},
        ], "or"),
        ([
            {"field": "category", "type": "categorical", "operator": "in", "value": ["Green"]},
            {"field": "active", "type": "boolean", "operator": "true"},
        ], "xor"),
    ],
)
def test_sql_filter_matches_python_oracle(project, filters, operator):
    expected_mask, errors = _evaluate_metadata_filters(
        project.units(), filters, operator=operator,
    )
    assert not errors
    expected = np.flatnonzero(expected_mask).tolist()
    received, count, page = project.database.filtered_unit_page(
        filters, operator=operator, page_size=250, page_number=1,
    )
    assert received == expected
    assert count == len(expected)
    assert page == 1


def test_filtered_page_does_not_fetch_corpus_or_call_python_filter(project, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Filtered page attempted Python full-corpus scan")

    monkeypatch.setattr(project.database, "list_units", forbidden)
    import geometric_coder.ui.app as app
    monkeypatch.setattr(app, "_evaluate_metadata_filters", forbidden)
    flt = [{"field": "category", "type": "categorical", "operator": "in", "value": ["green"]}]
    positions, page, count = _sql_metadata_page(
        project, total_units=120, filters=flt, operator="and",
        uncoded_only=False, page_size=7, page_number=3,
    )
    assert count == 40
    assert page.page_count == 6
    assert len(positions) == 7
    assert positions.tolist() == [42, 45, 48, 51, 54, 57, 60]


def test_sql_pages_preserve_uncoded_positive_semantics(project):
    code = project.create_code("example")
    project.annotate(1, code, "positive")
    project.annotate(2, code, "negative")
    result, total, _ = project.database.filtered_unit_page(
        [], uncoded_only=True, page_size=125, page_number=1,
    )
    assert total == 119
    assert 0 not in result
    assert 1 in result


def test_sql_query_rejects_invalid_columns_or_values_safely(project):
    cases = [
        {"field": "category", "type": "text", "operator": "regex", "value": "["},
        {"field": "score", "type": "numeric", "operator": "gte", "value": "bad"},
    ]
    for clause in cases:
        with pytest.raises((ValueError, TypeError)):
            project.database.filtered_unit_count([clause])


def test_compact_matching_positions_reproduce_sql_page_sequence(project):
    filters = [{"field": "category", "type": "categorical", "operator": "in", "value": ["Green"]}]
    positions = project.database.filtered_unit_positions(filters)
    assert len(positions) == 40
    assert positions == list(range(0, 120, 3))
    for page in range(1, 7):
        rows, total, selected = project.database.filtered_unit_page(
            filters, page_size=7, page_number=page,
        )
        assert total == 40
        assert rows == positions[(page - 1) * 7 : page * 7]
        assert selected == page


def test_explore_callbacks_share_cached_filter_position_index():
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] /
              "src/geometric_coder/ui/app.py").read_text(encoding="utf-8")
    assert "def cached_metadata_positions(" in source
    assert "positions, page, filter_match_count = cached_sql_metadata_page(" in source
    assert "positions, _page, _match_count = cached_sql_metadata_page(" in source
