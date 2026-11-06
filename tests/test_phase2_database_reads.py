"""Regression checks for bounded Phase 2 database read paths."""

from pathlib import Path

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
