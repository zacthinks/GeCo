"""Persistence, alignment, and migration regression checks."""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from geometric_coder import GeometricCoder


@pytest.fixture
def project(tmp_path):
    return GeometricCoder.create(
        project_dir=tmp_path / "workspace.geco",
        data=pd.DataFrame({"id": [9, 3, 5], "text": ["a", "b", "c"]}),
        keys=["id"], text="text", geometries=None,
    )


def _geometry(project):
    return project.database.register_geometry(
        name="sample", class_name="CountGeometry", modality="text", public=True,
        config={}, dependency_names=[], matrix_path="", matrix_sparse=False,
        state_path=None, supports_query=True, supports_text_transform=True,
    )


def test_named_saved_search_survives_reopen_and_rename(project):
    geometry = _geometry(project)
    saved = project.save_semantic_search(
        geometry_id=geometry, query="  example ", name="Find examples",
        scores=[0.7, 0.1, 0.4], range_values=[0.1, 0.7],
    )
    reopened = GeometricCoder.open(project.project_dir)
    loaded = reopened.load_semantic_search(saved)
    assert loaded["query"] == "example"
    assert loaded["scores"] == pytest.approx([0.7, 0.1, 0.4])
    assert loaded["range"] == [0.1, 0.7]
    reopened.rename_semantic_search(saved, "Examples")
    assert reopened.list_semantic_searches()[0]["name"] == "Examples"
    with pytest.raises(ValueError, match="already exists"):
        reopened.save_semantic_search(
            geometry_id=geometry, query="test", name="examples", scores=[0, 0, 0]
        )
    reopened.delete_semantic_search(saved)
    assert reopened.list_semantic_searches() == []


def test_auto_results_are_bounded_by_session_and_geometry(project):
    gid = _geometry(project)
    session_id = project.sessions()[0]["session_id"]
    first = project.save_semantic_search(
        geometry_id=gid, query="one", scores=[1, 0, 0], session_id=session_id,
    )
    latest = project.save_semantic_search(
        geometry_id=gid, query="two", scores=[0, 1, 0], session_id=session_id,
    )
    assert first != latest
    assert project.saved_semantic_searches.latest(session_id, gid)["query"] == "two"
    assert len(project.list_semantic_searches(session_id=session_id)) == 1
    with pytest.raises(KeyError):
        project.load_semantic_search(first)


def test_stale_result_rejected_after_ordered_universe_change(project):
    gid = _geometry(project)
    saved = project.save_semantic_search(
        geometry_id=gid, query="example", name="Saved", scores=[0.1, 0.2, 0.3],
    )
    with project.database.connect() as con:
        con.execute("UPDATE units SET user_key_json = ? WHERE row_position = 0", ('{"id": 99}',))
    with pytest.raises(ValueError, match="stale"):
        project.load_semantic_search(saved)


def test_schema_15_migrates_without_erasing_existing_records(project):
    path = project.project_dir / "project.sqlite3"
    gid = _geometry(project)
    with sqlite3.connect(path) as con:
        con.execute("DROP TABLE semantic_search_results")
        con.execute("UPDATE project SET schema_version=15")
    reopened = GeometricCoder.open(project.project_dir)
    assert reopened.database.project_metadata()["schema_version"] == 16
    assert reopened.database.get_geometry(gid)["name"] == "sample"
    assert reopened.list_semantic_searches() == []



def test_restore_reports_stale_and_does_not_restore_semantic_coloring(project):
    from geometric_coder.ui.app import (
        _restore_semantic_snapshot, _restored_semantic_color_mode,
    )
    gid = _geometry(project)
    sid = int(project.sessions()[0]["session_id"])
    project.save_semantic_search(
        geometry_id=gid, session_id=sid, query="original",
        scores=[0.2, 0.6, 0.9],
    )
    scores, status = _restore_semantic_snapshot(project, sid, gid, "original")
    assert scores["sample"] == pytest.approx([0.2, 0.6, 0.9])
    assert "Restored" in status
    assert _restored_semantic_color_mode("semantic", scores) == "semantic"

    with project.database.connect() as con:
        con.execute("UPDATE units SET user_key_json = ? WHERE row_position = 0", ('{"id": 99}',))
    stale, status = _restore_semantic_snapshot(project, sid, gid, "original")
    assert stale == {}
    assert "stale" in status.lower() and "Run Search" in status
    assert _restored_semantic_color_mode("semantic", stale) == "plain"


def test_restore_mismatched_and_missing_query_retains_request(project):
    from geometric_coder.ui.app import _restore_semantic_snapshot
    gid = _geometry(project)
    sid = int(project.sessions()[0]["session_id"])
    scores, status = _restore_semantic_snapshot(project, sid, gid, "question")
    assert scores == {} and "Run Search" in status
    project.save_semantic_search(
        geometry_id=gid, session_id=sid, query="completed", scores=[0.1, 0.2, 0.3],
    )
    scores, status = _restore_semantic_snapshot(project, sid, gid, "edited")
    assert scores == {} and "query has changed" in status


def test_restored_topic_selection_cannot_take_over_semantic_coloring():
    from geometric_coder.ui.app import _topic_restore_snapshot, _is_restored_topic_selection
    snapshot = _topic_restore_snapshot(
        12, {"topic_model_id": 42, "topic_component_id": "topic-a",
             "topic_color_mode": "semantic"},
    )
    assert _is_restored_topic_selection(42, "dominant", "topic-a", snapshot, 12)
    assert _is_restored_topic_selection(42, "dominant", "topic-a", snapshot, 13)
    assert not _is_restored_topic_selection(42, "intensity", "topic-a", snapshot, 12)
    assert not _is_restored_topic_selection(21, "dominant", "topic-a", snapshot, 12)
