"""Regression checks for bounded Phase 2 database read paths."""

from pathlib import Path
from contextlib import contextmanager

import pandas as pd
import pytest

from geometric_coder import GeometricCoder
from geometric_coder.ui.app import _render_memo_preview


def _project(tmp_path: Path) -> GeometricCoder:
    return GeometricCoder.create(
        project_dir=tmp_path / "reads.geco",
        data=pd.DataFrame({
            "doc_id": [10, 20, 30, 40],
            "text": ["alpha", "beta", "gamma", "delta"],
        }),
        keys=["doc_id"],
        text="text",
    )


def test_memo_versions_preserve_historical_values_and_references(tmp_path: Path, monkeypatch) -> None:
    project = _project(tmp_path)
    memo_id = project.create_memo(
        title="First", body_markdown="old text",
        hashtags=["alpha", "beta"], unit_references=[1, 2],
    )
    project.update_memo(
        memo_id, title="Second", body_markdown="new text",
        hashtags=["beta"], unit_references=[2, 4],
    )
    project.update_memo(
        memo_id, title="Third", body_markdown="final text",
        hashtags=[], unit_references=[],
    )
    expected = [project.memo(memo_id, version) for version in (1, 2, 3)]

    def reject_get_memo(*args, **kwargs):
        raise AssertionError("memo version history must not perform N individual get_memo calls")

    monkeypatch.setattr(project.database, "get_memo", reject_get_memo)
    assert project.memo_versions(memo_id) == expected
    with pytest.raises(KeyError):
        project.memo_versions(987654321)


def test_batched_unit_key_lookup_preserves_identity_and_missing(tmp_path: Path) -> None:
    project = _project(tmp_path)
    keys = project.database.unit_keys_for_ids([3, 1, 3, 99999])
    assert list(keys) == [1, 3]
    assert keys[1] == {"doc_id": 10}
    assert keys[3] == {"doc_id": 30}
    assert project.database.unit_keys_for_ids([]) == {}


def test_memo_renderer_resolves_duplicate_and_missing_refs_without_single_lookup(
    tmp_path: Path, monkeypatch
) -> None:
    project = _project(tmp_path)
    from dash import dcc, html

    def reject_single_unit(*args, **kwargs):
        raise AssertionError("memo preview must use batched keys")

    monkeypatch.setattr(project, "unit", reject_single_unit)
    preview = _render_memo_preview(
        dcc, html, project,
        "See [[unit:1]], [[unit:3]], [[unit:1]], [[unit:99999]].",
    )
    assert preview.children.count("doc_id: 10") == 2
    assert "doc_id: 30" in preview.children
    assert "unit 99999" in preview.children


def test_batched_lookup_chunks_beyond_sqlite_variable_limit(tmp_path: Path) -> None:
    project = _project(tmp_path)
    keys = project.database.unit_keys_for_ids(list(range(1, 1100)))
    assert list(keys) == [1, 2, 3, 4]


def test_classifier_coverage_matches_missing_id_oracle(tmp_path: Path) -> None:
    project = _project(tmp_path)
    fit_id = 999999
    missing_ids = project.database.missing_classifier_prediction_observation_ids(fit_id)
    assert bool(missing_ids)
    assert project.database.classifier_predictions_complete(fit_id) == (not missing_ids)


def test_memo_version_query_count_is_independent_of_history(tmp_path: Path, monkeypatch):
    project = _project(tmp_path)
    memo_id = project.create_memo(
        title="Version 1", body_markdown="initial", hashtags=["start"],
        unit_references=[1],
    )
    for version in range(2, 12):
        project.update_memo(
            memo_id, title=f"Version {version}", body_markdown="updated",
            hashtags=["tag"], unit_references=[2],
        )
    statements = []
    original_connect = project.database.connect

    @contextmanager
    def traced_connect():
        with original_connect() as connection:
            connection.set_trace_callback(statements.append)
            yield connection

    monkeypatch.setattr(project.database, "connect", traced_connect)
    rows = project.memo_versions(memo_id)
    selects = [sql for sql in statements if sql.lstrip().upper().startswith("SELECT")]
    assert len(rows) == 11
    assert len(selects) == 3
    assert [row["version_number"] for row in rows] == list(range(1, 12))


def test_code_and_memo_search_query_counts_are_independent_of_collection_size(
    tmp_path: Path, monkeypatch
) -> None:
    project = _project(tmp_path)
    for index in range(9):
        project.create_code(f"Code {index}", f"Description {index}")
        project.create_memo(
            title=f"Memo {index}", body_markdown=f"Evidence {index}",
            hashtags=["shared", f"tag{index}"],
            unit_references=[1, 2],
        )
    statements = []
    original_connect = project.database.connect

    @contextmanager
    def traced_connect():
        with original_connect() as connection:
            connection.set_trace_callback(statements.append)
            yield connection

    monkeypatch.setattr(project.database, "connect", traced_connect)
    code_results = project.search_codes()
    code_selects = [sql for sql in statements if sql.lstrip().upper().startswith("SELECT")]
    statements.clear()
    memo_results = project.search_memos()
    memo_selects = [sql for sql in statements if sql.lstrip().upper().startswith("SELECT")]
    assert len(code_results) == len(memo_results) == 9
    assert len(code_selects) == 2
    assert len(memo_selects) == 3
