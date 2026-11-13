"""Query-centric semantic-search persistence, alignment, and restoration."""
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


def _geometry(project, name="sample"):
    return project.database.register_geometry(
        name=name, class_name="CountGeometry", modality="text", public=True,
        config={}, dependency_names=[], matrix_path="", matrix_sparse=False,
        state_path=None, supports_query=True, supports_text_transform=True,
    )


def test_queries_deduplicate_and_cache_across_geometries_and_sessions(project):
    store = project.saved_semantic_searches
    first = _geometry(project)
    second = _geometry(project, "other")
    qid = store.store("  example  ", first, [0.7, 0.1, 0.4])
    assert store.register("example") == qid
    assert store.store("example", second, [0.2, 0.3, 0.9]) == qid
    assert [x["query"] for x in store.queries()] == ["example"]
    reopened = GeometricCoder.open(project.project_dir)
    cache = reopened.saved_semantic_searches
    assert cache.find("example", first)["scores"] == pytest.approx([0.7, 0.1, 0.4])
    assert cache.find("example", second)["scores"] == pytest.approx([0.2, 0.3, 0.9])
    assert cache.find("other", first) is None


def test_delete_query_cascades_all_geometry_caches(project):
    store = project.saved_semantic_searches
    a = _geometry(project)
    b = _geometry(project, "second")
    qid = store.store("query", a, [0.1, 0.2, 0.3])
    store.store("query", b, [0.3, 0.2, 0.1])
    store.delete_query(qid)
    assert store.queries() == []
    assert store.find("query", a) is None
    assert store.find("query", b) is None


def test_stale_result_rejected_after_ordered_universe_change(project):
    gid = _geometry(project)
    store = project.saved_semantic_searches
    store.store("example", gid, [0.1, 0.2, 0.3])
    with project.database.connect() as con:
        con.execute("UPDATE units SET user_key_json = ? WHERE row_position = 0",
                    ('{"id": 99}',))
    with pytest.raises(ValueError, match="stale"):
        store.find("example", gid)


def test_schema_15_migrates_to_query_catalog(project):
    path = project.project_dir / "project.sqlite3"
    gid = _geometry(project)
    with sqlite3.connect(path) as con:
        con.execute("DROP TABLE saved_analyses")
        con.execute("DROP TABLE semantic_search_results")
        con.execute("DROP TABLE semantic_queries")
        con.execute("UPDATE project SET schema_version=15")
    reopened = GeometricCoder.open(project.project_dir)
    assert reopened.database.project_metadata()["schema_version"] == 19
    assert reopened.database.get_geometry(gid)["name"] == "sample"
    assert reopened.saved_semantic_searches.queries() == []


def test_legacy_v16_saved_results_migrate_without_loss(project):
    from geometric_coder.storage import database as storage
    gid = _geometry(project)
    path = project.project_dir / "project.sqlite3"
    with sqlite3.connect(path) as con:
        con.execute("DROP TABLE saved_analyses")
        con.execute("DROP TABLE semantic_search_results")
        con.execute("DROP TABLE semantic_queries")
        for statement in storage._SAVED_SEARCH_SQL:
            con.execute(statement)
        con.execute("UPDATE project SET schema_version=16")
    # Create the legacy result with its old schema for the migration exercise.
    from geometric_coder.search_results import SavedSemanticSearches
    import numpy as np
    import zlib
    store = SavedSemanticSearches(project)
    signature, count, _ = store._signature(gid)
    with sqlite3.connect(path) as con:
        con.execute(
            """INSERT INTO semantic_search_results
               (kind,name,session_id,geometry_id,query,signature,unit_count,scores_blob,
                range_json,created_at) VALUES
               ('named','Saved',NULL,?,?,?,?,?,'null','2026-11-02T00:00:00Z')""",
            (gid, "example", signature, count,
             zlib.compress(np.asarray([0.1,0.2,0.3], dtype="<f8").tobytes())),
        )
    reopened = GeometricCoder.open(project.project_dir)
    assert reopened.saved_semantic_searches.find("example", gid)["scores"] == pytest.approx([0.1,0.2,0.3])


def test_session_resume_drops_legacy_semantic_and_topic_state(project):
    from geometric_coder.ui.app import _resumable_session_state
    state = {
        "geometry_id": 7, "view_id": 13, "literal_query": "kept",
        "semantic_query": "old draft", "semantic_active_query": "old search",
        "semantic_range": [0.2, 0.8], "topic_color_mode": "semantic",
        "topic_model_id": 6, "topic_component_id": "topic-a",
    }
    resumed = _resumable_session_state(state)
    assert resumed == {"geometry_id": 7, "view_id": 13, "literal_query": "kept"}
    gid = _geometry(project)
    project.saved_semantic_searches.store("old search", gid, [0.2, 0.6, 0.9])
    assert project.saved_semantic_searches.find("old search", gid) is not None



def test_topic_selection_restore_guard():
    from geometric_coder.ui.app import _topic_restore_snapshot, _is_restored_topic_selection
    snapshot = _topic_restore_snapshot(
        12, {"topic_model_id": 42, "topic_component_id": "topic-a",
             "topic_color_mode": "semantic"},
    )
    assert _is_restored_topic_selection(42, "dominant", "topic-a", snapshot, 12)
    assert _is_restored_topic_selection(42, "dominant", "topic-a", snapshot, 13)
    assert not _is_restored_topic_selection(42, "intensity", "topic-a", snapshot, 12)


def test_inflight_deleted_query_cannot_be_resurrected(project):
    gid = _geometry(project)
    cache = project.saved_semantic_searches
    old_id = cache.register("question")
    cache.delete_query(old_id)
    with pytest.raises(KeyError, match="deleted"):
        cache.store("question", gid, [0.1, 0.2, 0.3], expected_query_id=old_id)
    assert cache.queries() == []
    assert cache.find("question", gid) is None
    new_id = cache.register("question")
    assert new_id != old_id
    with pytest.raises(KeyError, match="deleted"):
        cache.store("question", gid, [0.1, 0.2, 0.3], expected_query_id=old_id)
    assert cache.find("question", gid) is None


def test_late_query_and_geometry_result_is_rejected(project):
    from geometric_coder.ui.app import _current_semantic_scores
    from geometric_coder.ui.app import _resumable_session_state
    payload_a = {"query": "A", "geometry_id": 5,
                 "query_id": 11, "scores": {"sample": [0.1, 0.2, 0.3]}}
    payload_b = {"query": "B", "geometry_id": 5,
                 "query_id": 12, "scores": {"sample": [0.4, 0.5, 0.6]}}
    assert _current_semantic_scores(payload_b, "B", 5) == payload_b["scores"]
    assert _current_semantic_scores(payload_a, "B", 5) == {}
    assert _current_semantic_scores(payload_a, "A", 6) == {}
    assert _current_semantic_scores(payload_a, "", 5) == {}
    assert _resumable_session_state({"semantic_active_query": "A"}) == {}
