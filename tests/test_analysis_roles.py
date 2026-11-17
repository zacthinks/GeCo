"""Phase 2 role recipe compatibility and execution boundaries."""
from __future__ import annotations

import pandas as pd
import pytest

from geometric_coder import GeometricCoder
from geometric_coder.analyses import compute_analysis, validate_analysis
from geometric_coder.ui.analyses import _initial_config, _slots, editor_recipe


@pytest.fixture
def project(tmp_path):
    workspace = GeometricCoder.create(
        project_dir=tmp_path / "roles.geco",
        data=pd.DataFrame({
            "id": [1, 2, 3, 4],
            "text": ["a", "b", "c", "d"],
            "category": ["A", "A", "B", "B"],
            "date": ["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01"],
            "year": [2023, 2024, 2025, 2026],
        }),
        keys=["id"], text="text",
        metadata=["category", "date", "year"], geometries=None,
    )
    first = workspace.create_code("First")
    second = workspace.create_code("Second")
    ids = [u["observation_id"] for u in workspace.database.list_units()]
    for index, label in enumerate(("positive", "negative", "unsure", "positive")):
        workspace.annotate(ids[index], first, label)
    for index, label in enumerate(("positive", "positive", "negative", "unsure")):
        workspace.annotate(ids[index], second, label)
    return workspace, first, second, ids


def code(code_id):
    return {"type": "code", "code_id": code_id, "source": "labels", "value": "binary"}


def entry(variable):
    return {"variable": variable}


def test_v2_bar_matches_v1_sql_and_saves_live_recipe(project):
    p, a, b, ids = project
    legacy = {
        "kind": "bar", "measure": "count",
        "variables": [code(a), code(b), {"type": "metadata", "field": "category"}],
    }
    v2 = {
        "schema_version": 2, "kind": "bar", "measure": "count",
        "roles": {"values": [entry(code(a)), entry(code(b))],
                  "group": entry({"type": "metadata", "field": "category"})},
    }
    assert compute_analysis(p, v2) == compute_analysis(p, legacy)
    key = p.save_analysis("Role-based bar", v2)
    assert p.analysis(key)["spec"]["schema_version"] == 2
    assert p.run_analysis(key)["counts"] == [[1, 1], [2, 0]]
    p.annotate(ids[1], a, "positive")
    assert p.run_analysis(key)["counts"][0] == [2, 1]


def test_heatmap_and_distribution_match_legacy_outputs(project):
    p, a, b, _ = project
    for kind, measure, variables in (
        ("heatmap", "jaccard", [code(a), code(b)]),
        ("heatmap", "correlation", [code(a), {"type": "metadata", "field": "category"}]),
        ("distribution", "count", [code(a)]),
    ):
        roles = {
            "indicators" if kind == "heatmap" else "values": [
                entry(variable) for variable in variables
            ],
        }
        v1 = {"kind": kind, "measure": measure, "variables": variables}
        v2 = {"schema_version": 2, "kind": kind, "measure": measure, "roles": roles}
        assert compute_analysis(p, v2) == compute_analysis(p, v1)


def test_unsupported_or_misleading_role_configurations_are_rejected(project):
    p, a, b, _ = project
    scale = p.create_scale("Continuous", "Description")
    base = {"schema_version": 2, "kind": "bar", "measure": "count"}
    bad_entries = [
        entry({"type": "scale", "scale_id": scale, "value": "position"}),
        entry({**code(a), "value": "probability"}),
        entry({"type": "metadata", "field": "date", "value": "temporal"}),
        entry({"type": "metadata", "field": "year", "value": "numeric"}),
    ]
    for bad in bad_entries:
        with pytest.raises(ValueError):
            validate_analysis(p, {**base, "roles": {"values": [bad]}})
    for bad in bad_entries[2:]:
        with pytest.raises(ValueError, match="Phase 3"):
            validate_analysis(p, {**base, "roles": {
                "values": [entry(code(a))], "group": bad,
            }})
    with pytest.raises(ValueError, match="at least two"):
        validate_analysis(p, {
            "schema_version": 2, "kind": "heatmap",
            "roles": {"indicators": [entry(code(a))]},
        })
    with pytest.raises(ValueError, match="transformations"):
        validate_analysis(p, {**base, "roles": {"values": [{
            **entry(code(a)), "transform": {"kind": "bin"},
        }]}})
    with pytest.raises(ValueError, match="once"):
        validate_analysis(p, {**base, "roles": {
            "values": [entry(code(a)), entry(code(a))],
        }})
    with pytest.raises(ValueError, match="schema version"):
        validate_analysis(p, {"schema_version": 99, "kind": "bar"})


def test_legacy_restoration_and_role_config_isolation(project):
    p, a, b, _ = project
    legacy = {"schema_version": 1, "kind": "bar", "measure": "count",
              "variables": [code(a), {"type": "metadata", "field": "category"}]}
    slots = _slots(legacy)
    assert slots == {"values": [f"code:{a}"], "group": "metadata:category"}
    config = _initial_config(legacy)
    assert f"values:code:{a}" in config
    assert f"group:metadata:category" in config
    recipe = editor_recipe("bar", "count", slots, config, "none", "any", 1, legacy=True)
    assert compute_analysis(p, recipe) == compute_analysis(p, legacy)
    # A variable's configuration belongs to (role, identity), never to a slot
    # index that can later refer to a different Code.
    configs = {
        f"values:code:{a}": {**code(a), "source": "labels"},
        f"values:code:{b}": {**code(b), "source": "hybrid", "predictor_id": 17},
    }
    recipe = editor_recipe("bar", "count", {"values": [f"code:{a}"]},
                           configs, None, "any", 1)
    assert recipe["roles"]["values"][0]["variable"]["source"] == "labels"
    recipe = editor_recipe("bar", "count", {"values": [f"code:{b}"]},
                           configs, None, "any", 1)
    assert recipe["roles"]["values"][0]["variable"]["predictor_id"] == 17


def test_v1_id_rename_duplicate_and_outputs_remain_unchanged(project):
    p, a, b, _ = project
    old = {"kind": "heatmap", "measure": "count",
           "variables": [code(a), code(b)]}
    identity = p.save_analysis("Original", old)
    before = p.run_analysis(identity)
    assert p.analysis(identity)["spec"]["schema_version"] == 1
    assert p.save_analysis("Renamed", old, analysis_id=identity) == identity
    assert p.run_analysis(identity) == before
    clone = p.save_analysis("Copy", p.analysis(identity)["spec"])
    assert clone != identity
    p.delete_analysis(clone)
    assert p.analysis(identity)["name"] == "Renamed"


def test_fixed_committee_binary_is_executable_without_retraining(tmp_path, monkeypatch):
    from geometric_coder import CountGeometry

    p = GeometricCoder.create(
        project_dir=tmp_path / "committee-roles.geco",
        data=pd.DataFrame({
            "id": list(range(6)),
            "text": ["good clear", "great useful", "nice clear",
                     "bad confusing", "poor unclear", "weak bad"],
        }),
        keys=["id"], text="text",
        geometries={"lexical": CountGeometry(lemmatize=False, weighting="tfidf")},
    )
    code_id = p.create_code("Sentiment")
    p.ensure_default_classifier_specs(code_id=code_id)
    classifier = int(p.classifier_specs(code_id)[0]["classifier_spec_id"])
    ids = [row["observation_id"] for row in p.database.list_units()]
    p.annotate(ids[0], code_id, "positive")
    p.annotate(ids[3], code_id, "negative")
    p.train_classifier(code_id=code_id, classifier_spec_id=classifier)
    committee = p.create_classifier_committee(
        code_id=code_id, name="Only member",
        classifier_spec_ids=[classifier], aggregation="mean",
    )
    ref = {
        "type": "code", "code_id": code_id, "source": "predictions",
        "value": "binary", "predictor_kind": "committee",
        "predictor_id": committee,
    }
    spec = {"schema_version": 2, "kind": "bar", "measure": "count",
            "roles": {"values": [entry(ref)]}}
    def forbidden(*args, **kwargs):
        raise AssertionError("A preview must not train a model or change evidence.")
    monkeypatch.setattr(p, "train_classifier", forbidden)
    monkeypatch.setattr(p, "train_scale", forbidden)
    result = compute_analysis(p, spec)
    assert result["counts"][0][0] <= result["denominators"][0][0]
    assert result["denominators"][0][0] == 6
    assert p.save_analysis("Committee", spec) > 0
    bad = {**spec, "aggregation": {"level": "id", "method": "any"}}
    with pytest.raises(ValueError):
        validate_analysis(p, bad)
