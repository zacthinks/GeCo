"""Phase 2 role recipe compatibility and execution boundaries."""
from __future__ import annotations

import pandas as pd
import pytest

from geometric_coder import GeometricCoder
from geometric_coder.analyses import compute_analysis, validate_analysis
from geometric_coder.ui.analyses import (
    _initial_config, _slots, _reconcile_live_source_choices,
    compatible_aggregation_method, editor_recipe,
)


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


@pytest.mark.parametrize(
    ("kind", "measure", "method", "expected"),
    [
        ("distribution", "count", "count", "count"),
        ("distribution", "count", "mean", "mean"),
        ("heatmap", "correlation", "mean", "mean"),
        ("heatmap", "jaccard", "mean", "minimum"),
        ("heatmap", "count", "log_count", "minimum"),
        ("bar", "count", "count", "minimum"),
        ("bar", "count", "minimum", "minimum"),
        ("bar", "count", "any", "minimum"),
        ("scatter", "count", "count", "count"),
        ("histogram", "count", "log_count", "log_count"),
        ("summary_bar", "count", "mean", "mean"),
        ("correlation_heatmap", "count", "count", "count"),
        ("value_heatmap", "count", "log_count", "log_count"),
    ],
)
def test_chart_aggregation_method_compatibility(kind, measure, method, expected):
    assert compatible_aggregation_method(kind, measure, method) == expected


def test_chart_switch_normalizes_only_incompatible_aggregations(tmp_path):
    p = GeometricCoder.create(
        project_dir=tmp_path / "chart-switch.geco",
        data=pd.DataFrame({
            "group": ["A", "A", "B", "B"],
            "id": [1, 2, 1, 2],
            "text": ["one", "two", "three", "four"],
        }),
        keys=["group", "id"], text="text", geometries=None,
    )
    a = p.create_code("First")
    b = p.create_code("Second")
    ids = [row["observation_id"] for row in p.database.list_units()]
    p.annotate(ids[0], a, "positive")
    p.annotate(ids[1], a, "negative")
    p.annotate(ids[2], b, "positive")
    p.annotate(ids[3], b, "negative")
    codes = [f"code:{a}", f"code:{b}"]
    for kind, measure, slots, expected in (
        ("distribution", "count", {"values": codes}, "count"),
        ("bar", "count", {"values": codes}, "minimum"),
        ("heatmap", "correlation", {"indicators": codes}, "count"),
        ("heatmap", "jaccard", {"indicators": codes}, "minimum"),
    ):
        recipe = editor_recipe(kind, measure, slots, {}, "group", "count", 1)
        assert recipe["aggregation"]["method"] == expected
        result = compute_analysis(p, recipe)
        assert result["kind"] == kind

    # A saved bar threshold reduction is valid and must survive editing.
    existing = {
        "kind": "bar", "measure": "count",
        "variables": [code(a)],
        "aggregation": {"level": "group", "method": "minimum", "threshold": 2},
    }
    slots = _slots(existing)
    configs = _initial_config(existing)
    edited = editor_recipe(
        "bar", "count", slots, configs, "group", "minimum", 2, legacy=True,
    )
    assert compute_analysis(p, edited) == compute_analysis(p, existing)
    assert edited["aggregation"]["method"] == "minimum"
    assert edited["aggregation"]["threshold"] == 2


def test_any_and_inclusive_threshold_one_are_equivalent_for_legacy_and_v2(project):
    p, a, _, _ = project
    legacy = {
        "kind": "bar", "measure": "count",
        "variables": [code(a)],
        "aggregation": {"level": "group", "method": "any"},
    }
    threshold = {**legacy, "aggregation": {
        "level": "group", "method": "minimum", "threshold": 1,
    }}
    assert compute_analysis(p, legacy) == compute_analysis(p, threshold)
    assert editor_recipe(
        "bar", "count", {"values": [f"code:{a}"]},
        {f"values:code:{a}": code(a)}, "group", "any", 1,
        legacy=True,
    )["aggregation"]["method"] == "minimum"


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


def test_v2_save_rejects_data_dependent_chart_limits_before_writing(tmp_path):
    # A valid role schema can expand past the live chart engine's limits.
    # Exercise the real save boundary without training models or changing v1.
    row_count = 101
    meta = [f"flag_{i}" for i in range(13)]
    data = {
        "id": list(range(row_count)),
        "text": ["observation"] * row_count,
        "category": [f"group_{i:03d}" for i in range(row_count)],
    }
    for field in meta:
        data[field] = ["A" if i % 2 else "B" for i in range(row_count)]
    p = GeometricCoder.create(
        project_dir=tmp_path / "role-limits.geco",
        data=pd.DataFrame(data),
        keys=["id"], text="text",
        metadata=["category", *meta], geometries=None,
    )
    theme = p.create_code("Theme")
    group_entry = entry({"type": "metadata", "field": "category",
                         "value": "categorical"})
    bar = {
        "schema_version": 2, "kind": "bar", "measure": "count",
        "roles": {"values": [entry(code(theme))], "group": group_entry},
    }
    with pytest.raises(ValueError, match=r"(maximum 100|more than 100 categories)"):
        p.save_analysis("Too many groups", bar)
    assert p.analyses() == []

    heatmap = {
        "schema_version": 2, "kind": "heatmap", "measure": "jaccard",
        "roles": {"indicators": [
            entry({"type": "metadata", "field": field, "value": "categorical"})
            for field in meta
        ]},
    }
    # Thirteen two-category fields produce 26 indicators (> 24).
    with pytest.raises(ValueError, match="Too many encoded variables"):
        p.save_analysis("Oversized heatmap", heatmap)
    assert p.analyses() == []

    # An individual field can also exceed the 24-category heatmap bound.
    heatmap["roles"]["indicators"] = [
        group_entry,
        entry({"type": "metadata", "field": meta[0], "value": "categorical"}),
    ]
    with pytest.raises(ValueError, match="too many categories"):
        p.save_analysis("Oversized indicator", heatmap)
    assert p.analyses() == []

    # Supported metadata configurations remain saveable; a failing update
    # must not replace an existing valid definition.
    bar["roles"]["group"] = entry({
        "type": "metadata", "field": meta[0], "value": "categorical",
    })
    analysis_id = p.save_analysis("Valid role analysis", bar)
    saved = p.analysis(analysis_id)["spec"]
    assert p.run_analysis(analysis_id)["categories"] == ["A", "B"]
    bar["roles"]["group"] = group_entry
    with pytest.raises(ValueError, match=r"(maximum 100|more than 100 categories)"):
        p.save_analysis("Overwrite rejected", bar, analysis_id=analysis_id)
    assert p.analysis(analysis_id)["name"] == "Valid role analysis"
    assert p.analysis(analysis_id)["spec"] == saved


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
    untrained_ref = {
        "type": "code", "code_id": code_id, "source": "predictions",
        "value": "binary", "predictor_id": classifier,
    }
    untrained_spec = {
        "schema_version": 2, "kind": "bar", "measure": "count",
        "roles": {"values": [entry(untrained_ref)]},
    }
    # A new recipe must not persist before its selected fit exists. Legacy
    # v1 creation remains allowed (and the definition remains viewable).
    with pytest.raises(ValueError, match="no trained prediction fit"):
        p.save_analysis("Not ready", untrained_spec)
    assert p.analyses() == []
    legacy = p.save_analysis(
        "Legacy pending", {"kind": "bar", "variables": [untrained_ref]}
    )
    assert p.analysis(legacy)["spec"]["schema_version"] == 1
    with pytest.raises(ValueError, match="No trained prediction source"):
        p.run_analysis(legacy)
    p.train_classifier(code_id=code_id, classifier_spec_id=classifier)
    assert p.save_analysis("Now ready", untrained_spec) > 0
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


@pytest.mark.parametrize("missing_prediction", [False, True])
def test_committee_jaccard_matches_single_member_classifier_for_zero_union(
    tmp_path, missing_prediction,
):
    """Observed all-negative pairs are zero; unobserved pairs remain undefined."""
    from geometric_coder import CountGeometry

    p = GeometricCoder.create(
        project_dir=tmp_path / "zero-union.geco",
        data=pd.DataFrame({
            "id": list(range(6)),
            "text": ["good clear", "great useful", "nice clear",
                     "bad confusing", "poor unclear", "weak bad"],
        }),
        keys=["id"], text="text",
        geometries={"lexical": CountGeometry(lemmatize=False, weighting="tfidf")},
    )
    ids = [row["observation_id"] for row in p.database.list_units()]
    pairs = []
    for name in ("Theme A", "Theme B"):
        cid = p.create_code(name)
        p.annotate(ids[0], cid, "positive")
        p.annotate(ids[3], cid, "negative")
        p.ensure_default_classifier_specs(code_id=cid)
        spec_id = int(p.classifier_specs(code_id=cid)[0]["classifier_spec_id"])
        p.train_classifier(code_id=cid, classifier_spec_id=spec_id)
        fit_id = int(p.database.latest_classifier_fit(
            code_id=cid, classifier_spec_id=spec_id
        )["classifier_fit_id"])
        committee = p.create_classifier_committee(
            code_id=cid, name=f"{name} single-member committee",
            classifier_spec_ids=[spec_id], aggregation="mean",
        )
        pairs.append((cid, spec_id, fit_id, committee))

    # Force identical fully observed negative model predictions without changing
    # human evidence or fitting any new model.
    with p.database.connect() as conn:
        for _, _, fit_id, _ in pairs:
            conn.execute(
                "UPDATE classifier_predictions SET predicted_label = 'negative', "
                "probability = 0.1 WHERE classifier_fit_id = ?", (fit_id,),
            )
        if missing_prediction:
            # No pairwise-complete observations: remove predictions from the
            # second model, while the first remains fully observed.
            conn.execute(
                "DELETE FROM classifier_predictions WHERE classifier_fit_id = ?",
                (pairs[1][2],),
            )

    classifier_refs = [
        {**code(cid), "source": "predictions", "predictor_id": spec_id}
        for cid, spec_id, _, _ in pairs
    ]
    committee_refs = [
        {**code(cid), "source": "predictions", "predictor_kind": "committee",
         "predictor_id": committee}
        for cid, _, _, committee in pairs
    ]
    old = compute_analysis(p, {
        "kind": "heatmap", "measure": "jaccard", "variables": classifier_refs,
    })
    new = compute_analysis(p, {
        "schema_version": 2, "kind": "heatmap", "measure": "jaccard",
        "roles": {"indicators": [entry(ref) for ref in committee_refs]},
    })
    assert new["samples"] == old["samples"]
    assert new["values"] == old["values"]
    assert new["samples"][0][1] == (0 if missing_prediction else 6)
    if missing_prediction:
        assert new["values"][0][1] is None
    else:
        assert new["values"][0][1] == 0.0


def test_committee_bar_rejects_category_101_before_next_page(monkeypatch):
    from geometric_coder import analysis_roles

    class FakeProject:
        def code(self, cid):
            return {"name": "Theme"}

    def pages(_project, _descriptors):
        yield {
            "row_count": 101,
            "columns": [
                {"values": [0.0] * 101},
                {"values": [f"category-{i:03d}" for i in range(101)]},
            ],
            "warnings": [],
        }
        raise AssertionError("Must stop before reading subsequent pages")

    monkeypatch.setattr(analysis_roles, "_pages", pages)
    spec = {
        "kind": "bar", "measure": "count",
        "roles": {
            "values": [entry(code(1))],
            "group": entry({"type": "metadata", "field": "category",
                            "value": "categorical"}),
        },
    }
    with pytest.raises(ValueError, match="more than 100 categories"):
        analysis_roles._committee_analysis(FakeProject(), spec)


def test_mounted_source_and_predictor_override_stale_role_store():
    key = "values:code:17"
    other = "group:metadata:category"
    stale = {
        key: {"source": "labels", "value": "binary",
              "predictor_kind": "classifier", "predictor_id": 5},
        other: {"value": "categorical"},
    }
    live = _reconcile_live_source_choices(
        stale, ["predictions"], [{"index": key}],
        ["committee:9"], [{"index": key}],
    )
    assert live[key]["source"] == "predictions"
    assert live[key]["predictor_kind"] == "committee"
    assert live[key]["predictor_id"] == 9
    assert live[other] == stale[other]
    assert stale[key]["source"] == "labels"

    cleared = _reconcile_live_source_choices(
        live, ["hybrid"], [{"index": key}], [None], [{"index": key}],
    )
    assert cleared[key]["source"] == "hybrid"
    assert "predictor_id" not in cleared[key]
    assert "predictor_kind" not in cleared[key]
