"""Temporal metadata resolution, retaining imported ISO date/time semantics."""
from __future__ import annotations

import json
from datetime import date, datetime, timezone, timedelta

import pandas as pd
import pytest

from geometric_coder import GeometricCoder
from geometric_coder.analysis_variables import (
    MAX_RESOLUTION_OBSERVATIONS,
    resolve_analytical_variables,
    validate_variable,
)


def _ref(value="temporal"):
    return {"type": "metadata", "field": "when", "value": value}


def _resolve(project, ids, refs=None):
    return resolve_analytical_variables(
        project, [_ref()] if refs is None else refs, ids
    )


@pytest.fixture
def dates_project(tmp_path):
    # Both Python dates and pandas Timestamps flow through _json_value during
    # ordinary import, not a synthetic metadata update or a new persistence API.
    data = pd.DataFrame({
        "id": [9, 3, 12, 5, 20, 30],
        "text": ["a", "b", "c", "d", "e", "f"],
        "when": [
            date(2024, 2, 29),
            pd.Timestamp("2026-12-31 23:59:59.123456789+05:30"),
            pd.Timestamp("2027-01-01 00:00:00.123456789"),
            "2026-11-17T08:30:15Z",
            None,
            "01/02/2026",
        ],
    })
    project = GeometricCoder.create(
        project_dir=tmp_path / "temporal.geco", data=data,
        keys=["id"], text="text", metadata=["when"], geometries=None,
    )
    ids = [u["observation_id"] for u in project.database.list_units()]
    return project, ids


def test_imported_python_date_pandas_timestamps_and_iso_utc(dates_project):
    project, ids = dates_project
    # Verify that the test is exercising GeCo's actual string serialization.
    rows = project.database.list_units()
    assert rows[0]["metadata"]["when"] == "2024-02-29"
    assert rows[1]["metadata"]["when"] == "2026-12-31 23:59:59.123456789+05:30"
    assert rows[2]["metadata"]["when"] == "2027-01-01 00:00:00.123456789"

    result = _resolve(project, ids)
    column = result["columns"][0]
    assert column["descriptor"] == {
        "type": "metadata", "field": "when", "value": "temporal",
        "numeric_strings": False,
    }
    assert column["kind"] == "temporal"
    assert column["provenance"]["format"] == "iso8601_tagged"
    assert column["values"] == [
        {"kind": "date", "iso": "2024-02-29"},
        {
            "kind": "datetime",
            "iso": "2026-12-31T23:59:59.123456789+05:30",
            "timezone": "offset",
        },
        {
            "kind": "datetime",
            "iso": "2027-01-01T00:00:00.123456789",
            "timezone": "naive",
        },
        {
            "kind": "datetime",
            "iso": "2026-11-17T08:30:15+00:00",
            "timezone": "offset",
        },
        None,
        None,
    ]
    assert column["missing_reasons"] == [
        None, None, None, None, "missing_metadata", "ambiguous_temporal",
    ]
    assert column["coverage"] == {
        "observed": 4, "missing": 2,
        "by_reason": {"missing_metadata": 1, "ambiguous_temporal": 1},
    }
    assert len(column["warnings"]) == 2
    assert "mix dates and timestamps" in column["warnings"][0]
    assert "mix naive and offset-aware" in column["warnings"][1]
    assert result["warnings"] == column["warnings"]


def test_temporal_caller_order_duplicates_and_cross_variable_alignment(dates_project):
    project, ids = dates_project
    selected = [ids[3], ids[1], ids[5], ids[0], ids[1], ids[4]]
    result = _resolve(project, selected, [_ref(), _ref("categorical")])
    assert result["observation_ids"] == selected
    assert result["row_count"] == 6
    temporal, categorical = result["columns"]
    assert temporal["values"][1] == temporal["values"][4]
    assert categorical["values"][1] == categorical["values"][4]
    assert temporal["values"][2] is None
    assert temporal["missing_reasons"][2] == "ambiguous_temporal"
    assert categorical["values"][2] == "01/02/2026"
    assert temporal["coverage"] == {
        "observed": 4, "missing": 2,
        "by_reason": {"ambiguous_temporal": 1, "missing_metadata": 1},
    }


@pytest.mark.parametrize(("raw", "expected", "reason"), [
    ("2024-02-29", {"kind": "date", "iso": "2024-02-29"}, None),
    ("2026-12-31", {"kind": "date", "iso": "2026-12-31"}, None),
    ("2027-01-01", {"kind": "date", "iso": "2027-01-01"}, None),
    ("2026-03-01T02:03", {"kind": "datetime",
        "iso": "2026-03-01T02:03:00", "timezone": "naive"}, None),
    ("2026-03-01 02:03:04.123456789-04:30", {"kind": "datetime",
        "iso": "2026-03-01T02:03:04.123456789-04:30",
        "timezone": "offset"}, None),
    ("2026-03-01T02:03:04Z", {"kind": "datetime",
        "iso": "2026-03-01T02:03:04+00:00", "timezone": "offset"}, None),
    ("2026-03-01t02:03:04z", {"kind": "datetime",
        "iso": "2026-03-01T02:03:04+00:00", "timezone": "offset"}, None),
    ("", None, "missing_metadata"),
    ("   ", None, "missing_metadata"),
    (None, None, "missing_metadata"),
    ("2026-02-29", None, "invalid_temporal"),
    ("2024-13-01", None, "invalid_temporal"),
    ("2026-11-31", None, "invalid_temporal"),
    ("2026-03-01T25:00:00", None, "invalid_temporal"),
    ("2026-03-01T01:60:00", None, "invalid_temporal"),
    ("2026-03-01T12:30:60", None, "invalid_temporal"),
    ("2026-03-01T10:00:00+24:00", None, "invalid_temporal"),
    ("2026-03-01T10:00:00-00:00", None, "ambiguous_temporal"),
    ("2026-03-01T10:00:00+0530", None, "invalid_temporal"),
    ("2026-03-01T10:00:00.1234567890Z", None, "invalid_temporal"),
    ("01/02/2026", None, "ambiguous_temporal"),
    ("1-2-2026", None, "ambiguous_temporal"),
    ("March 1, 2026", None, "invalid_temporal"),
    ("2026/03/01", None, "invalid_temporal"),
    ("2026-03-01 10:00 EST", None, "invalid_temporal"),
    (1730000000, None, "invalid_temporal"),
    (True, None, "invalid_temporal"),
    ({"year": 2026}, None, "invalid_temporal"),
])
def test_strict_temporal_input_and_missing_reason(raw, expected, reason):
    from geometric_coder.analysis_variables import _temporal_value
    assert _temporal_value(raw) == (expected, reason)


def test_temporal_empty_request_is_bounded_and_never_loads_full_corpus(
    dates_project, monkeypatch,
):
    project, ids = dates_project
    monkeypatch.setattr(project.database, "list_units", lambda: 1 / 0)
    result = _resolve(project, [])
    column = result["columns"][0]
    assert column["values"] == []
    assert column["coverage"] == {
        "observed": 0, "missing": 0, "by_reason": {},
    }
    assert result["observation_ids"] == []
    with pytest.raises(ValueError, match="at most"):
        _resolve(project, [ids[0]] * (MAX_RESOLUTION_OBSERVATIONS + 1))
    with pytest.raises(ValueError, match="Unknown or non-atomic"):
        _resolve(project, [ids[0], 999999999])


def test_explicit_metadata_interpretation_and_v1_analysis_unchanged(dates_project):
    project, ids = dates_project
    assert validate_variable(project, {"type": "metadata", "field": "when"})["value"] == "categorical"
    for invalid in (
        {"type": "metadata", "field": "when", "value": "timestamp"},
        {"type": "metadata", "field": "when", "value": "temporal", "numeric_strings": True},
        {"type": "metadata", "field": "nonexistent", "value": "temporal"},
    ):
        with pytest.raises(ValueError):
            validate_variable(project, invalid)

    categorical = _resolve(project, ids, [_ref("categorical")])["columns"][0]
    assert categorical["values"][5] == "01/02/2026"
    assert categorical["values"][0] == "2024-02-29"
    temporal = _resolve(project, ids)["columns"][0]
    assert temporal["values"][0] == {"kind": "date", "iso": "2024-02-29"}
    assert temporal["values"][5] is None

    code = project.create_code("Legacy test Code")
    project.annotate(ids[0], code, "positive")
    analysis = project.save_analysis(
        "Unchanged legacy bar",
        {"kind": "bar", "variables": [
            {"type": "code", "code_id": code, "source": "labels", "value": "binary"},
            {"type": "metadata", "field": "when"},
        ]},
    )
    before = project.run_analysis(analysis)
    assert project.analysis(analysis)["spec"]["schema_version"] == 1
    _resolve(project, ids)
    assert project.run_analysis(analysis) == before
    reopened = GeometricCoder.open(project.project_dir)
    assert reopened.run_analysis(analysis) == before


def test_temporal_resolver_never_writes_evidence(dates_project, monkeypatch):
    project, ids = dates_project
    monkeypatch.setattr(project, "train_scale", lambda *a, **kw: 1 / 0)
    monkeypatch.setattr(project, "train_classifier", lambda *a, **kw: 1 / 0)
    with project.database.connect() as conn:
        before = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("annotation_events", "scale_comparison_events", "scaler_fits")
        }
    _resolve(project, ids)
    with project.database.connect() as conn:
        assert before == {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in before
        }
