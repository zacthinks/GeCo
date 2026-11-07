"""Phase 3 semantic-result transport: vector stays server-side."""

import inspect
import json

import pandas as pd
import pytest

from geometric_coder import CountGeometry, GeometricCoder
from geometric_coder.ui.app import create_app


def _callbacks(app):
    return {
        inspect.unwrap(entry["callback"]).__name__: inspect.unwrap(entry["callback"])
        for entry in app.callback_map.values()
        if callable(entry.get("callback"))
    }


def test_semantic_candidate_carries_identity_but_no_corpus_vector(tmp_path):
    project = GeometricCoder.create(
        project_dir=tmp_path / "semantic-transport.geco",
        data=pd.DataFrame({
            "id": [10, 20, 30],
            "text": ["red maple", "blue sky", "green forest"],
        }),
        keys=["id"], text="text",
        geometries={"lexical": CountGeometry(lemmatize=False, weighting="tfidf")},
    )
    geometry = project.database.get_geometry_by_name("lexical")
    geometry_id = int(geometry["geometry_id"])
    query_id = project.saved_semantic_searches.store(
        "forest", geometry_id, [0.15, 0.2, 0.95],
    )
    app = create_app(project)
    callbacks = _callbacks(app)
    candidate = callbacks["display_active_semantic"]("forest", geometry_id)
    assert candidate["query"] == "forest"
    assert candidate["geometry_id"] == geometry_id
    assert candidate["query_id"] == query_id
    assert isinstance(candidate["token"], str) and candidate["token"]
    assert candidate["bounds"] == pytest.approx([0.15, 0.95])
    assert "scores" not in candidate
    assert len(json.dumps(candidate)) < 250

    selector = {
        field: candidate[field]
        for field in ("query", "geometry_id", "query_id", "token", "bounds")
    }
    slider = callbacks["update_semantic_range"]
    active = slider(selector, geometry_id, "forest", None)
    assert active[0:2] == pytest.approx((0.15, 0.95))
    assert active[5] == {"display": "block"}
    assert slider(selector, geometry_id, "different query", None)[5] == {"display": "none"}

    project.saved_semantic_searches.delete_query(query_id)
    assert slider(selector, geometry_id, "forest", None)[5] == {"display": "none"}


def test_saved_query_replaced_with_same_text_does_not_reuse_old_token(tmp_path):
    project = GeometricCoder.create(
        project_dir=tmp_path / "semantic-replacement.geco",
        data=pd.DataFrame({"id": [1, 2], "text": ["one", "two"]}),
        keys=["id"], text="text",
        geometries={"lexical": CountGeometry(lemmatize=False, weighting="tfidf")},
    )
    geometry_id = int(project.database.get_geometry_by_name("lexical")["geometry_id"])
    old_id = project.saved_semantic_searches.store("needle", geometry_id, [0.2, 0.7])
    callbacks = _callbacks(create_app(project))
    old = callbacks["display_active_semantic"]("needle", geometry_id)
    old_selector = {key: old[key] for key in ("query", "geometry_id", "query_id", "token")}
    project.saved_semantic_searches.delete_query(old_id)
    new_id = project.saved_semantic_searches.store("needle", geometry_id, [0.8, 0.1])
    assert new_id != old_id
    assert callbacks["update_semantic_range"](old_selector, geometry_id, "needle", None)[5] == {
        "display": "none"
    }
    new = callbacks["display_active_semantic"]("needle", geometry_id)
    assert new["query_id"] == new_id
    assert new["token"] != old["token"]
