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
