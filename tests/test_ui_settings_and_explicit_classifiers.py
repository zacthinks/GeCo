from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from geometric_coder import CountGeometry, GeometricCoder
from geometric_coder.ui.app import _read_ui_settings, _save_ui_settings


def _project(tmp_path: Path) -> GeometricCoder:
    return GeometricCoder.create(
        project_dir=tmp_path / "explicit-classifiers.geco",
        data=pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6],
            "text": [
                "clear helpful", "good useful", "helpful positive",
                "bad unclear", "poor confusing", "negative unhelpful",
            ],
        }),
        keys=["id"],
        text="text",
        geometries={"lexical": CountGeometry(lemmatize=False, weighting="tfidf")},
    )


def test_new_code_and_reopen_do_not_create_classifiers(tmp_path: Path) -> None:
    project = _project(tmp_path)
    code_id = project.create_code("example", "Example concept")
    assert project.classifier_specs(code_id=code_id) == []
    reopened = GeometricCoder.open(project.project_dir)
    assert reopened.classifier_specs(code_id=code_id) == []
    geometry_id = int(reopened.geometries(public_only=True)[0]["geometry_id"])
    explicit_id = reopened.create_classifier_spec(
        code_id=code_id,
        name="chosen by researcher",
        geometry_id=geometry_id,
    )
    assert [int(row["classifier_spec_id"]) for row in reopened.classifier_specs(code_id=code_id)] == [explicit_id]
    again = GeometricCoder.open(reopened.project_dir)
    assert [int(row["classifier_spec_id"]) for row in again.classifier_specs(code_id=code_id)] == [explicit_id]


def test_cv_setting_defaults_to_five_and_persists_per_project(tmp_path: Path) -> None:
    root = tmp_path / "settings.geco"
    root.mkdir()
    assert _read_ui_settings(root) == {"cv_folds": 5, "candidate_evaluation_budget": -1}
    _save_ui_settings(root, cv_folds=3)
    assert _read_ui_settings(root) == {"cv_folds": 3, "candidate_evaluation_budget": -1}
    with pytest.raises(ValueError, match="between 2 and 10"):
        _save_ui_settings(root, cv_folds=1)
    assert _read_ui_settings(root) == {"cv_folds": 3, "candidate_evaluation_budget": -1}
    (root / "ui_settings.json").write_text('{"cv_folds": "garbled"}', encoding="utf-8")
    assert _read_ui_settings(root) == {"cv_folds": 5, "candidate_evaluation_budget": -1}



def test_custom_candidate_budget_requires_a_value_before_saving(tmp_path: Path) -> None:
    import inspect

    pytest.importorskip("dash")
    from geometric_coder.ui.app import create_app

    project = _project(tmp_path)
    app = create_app(project)

    def find(component, component_id):
        if getattr(component, "id", None) == component_id:
            return component
        children = getattr(component, "children", None)
        if not isinstance(children, (list, tuple)):
            children = [children] if children is not None else []
        for child in children:
            found = find(child, component_id)
            if found is not None:
                return found
        return None

    field = find(app.layout, "geco-settings-candidate-custom")
    assert field is not None
    assert field.value is None

    callback = next(
        inspect.unwrap(spec["callback"])
        for output, spec in app.callback_map.items()
        if "geco-settings-candidate-status.children" in output
    )
    assert "Enter a whole-number" in callback("custom", field.value)
    assert _read_ui_settings(project.project_dir)["candidate_evaluation_budget"] == -1

    assert "Evaluate up to 25" in callback("custom", 25)
    assert _read_ui_settings(project.project_dir)["candidate_evaluation_budget"] == 25
    assert find(create_app(project).layout, "geco-settings-candidate-custom").value == 25


def test_cv_folds_pass_through_saved_evidence_training(tmp_path: Path) -> None:
    project = _project(tmp_path)
    code_id = project.create_code("example", "Example concept")
    geometry_id = int(project.geometries(public_only=True)[0]["geometry_id"])
    spec_id = project.create_classifier_spec(
        code_id=code_id, name="manual classifier", geometry_id=geometry_id,
    )
    for index, unit in enumerate(project.units()):
        project.annotate(
            int(unit["observation_id"]),
            code_id,
            "positive" if index < 3 else "negative",
        )
    result = project.train_classifiers_after_evidence_change(
        code_id=code_id, classifier_spec_ids=[spec_id], tune=True, folds=2,
    )
    assert result["status"] == "trained"
    runs = project.classifier_tuning_runs(code_id=code_id, classifier_spec_id=spec_id)
    assert runs and int(runs[-1]["folds"]) == 2


def test_no_implicit_classifier_creation_paths_remain() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "geometric_coder"
    project = (root / "project.py").read_text(encoding="utf-8")
    app = (root / "ui" / "app.py").read_text(encoding="utf-8")
    representations = (root / "ui" / "representation_manager.py").read_text(encoding="utf-8")
    assert "project.ensure_default_classifier_specs()" not in project
    assert "self.ensure_default_classifier_specs()" not in project
    assert "self.ensure_default_classifier_specs(code_id=code_id)" not in project
    assert "project.ensure_default_classifier_specs(" not in app
    assert "coder.ensure_default_classifier_specs(" not in representations


def test_finite_code_budget_limits_indexed_classifier_acquisition(tmp_path: Path) -> None:
    from geometric_coder.explore import deterministic_page_permutation_parameters

    project = GeometricCoder.create(
        project_dir=tmp_path / "finite-code-budget.geco",
        data=pd.DataFrame({
            "id": list(range(30)),
            "text": [f"helpful example {i}" if i % 2 else f"poor example {i}" for i in range(30)],
        }),
        keys=["id"], text="text",
        geometries={"lexical": CountGeometry(lemmatize=False, weighting="tfidf")},
    )
    code = project.create_code("Judgment")
    geometry_id = int(project.geometries(public_only=True)[0]["geometry_id"])
    spec = project.create_classifier_spec(
        code_id=code, name="budgeted", geometry_id=geometry_id,
    )
    units = project.units()
    for index in (0, 2):
        project.annotate(int(units[index]["observation_id"]), code, "positive")
    for index in (1, 3):
        project.annotate(int(units[index]["observation_id"]), code, "negative")
    project.train_classifier(code_id=code, classifier_spec_id=spec)
    fit_id = int(project.database.latest_classifier_fit(
        code_id=code, classifier_spec_id=spec,
    )["classifier_fit_id"])
    session = project.create_session("Budget test")
    budget = 6
    _save_ui_settings(project.project_dir, candidate_evaluation_budget=budget)

    total = len(units)
    seed = (fit_id * 2654435761 + code * 2246822519 + 1) & 0xFFFFFFFF
    multiplier, shift = deterministic_page_permutation_parameters(total, seed)
    sample_positions = {
        (multiplier * rank + shift) % total for rank in range(budget)
    }
    position_by_unit = {int(unit["unit_id"]): i for i, unit in enumerate(units)}

    def choose():
        return project.focus_recommendation(
            code_id=code, session_id=session,
            strategy="likely_positive", active_classifier_spec_id=spec,
        )

    first = choose()
    assert first is not None
    assert position_by_unit[first["unit_id"]] in sample_positions
    assert choose()["unit_id"] == first["unit_id"]
    project.record_visit(session, first["unit_id"], method="test")
    second = choose()
    if second is not None:
        assert second["unit_id"] != first["unit_id"]
        assert position_by_unit[second["unit_id"]] in sample_positions

    # The sampled window never widens after visits or labels. With every
    # candidate in the window unavailable, the bounded recommendation ends.
    for position in sample_positions:
        project.record_visit(session, int(units[position]["unit_id"]), method="test")
    assert choose() is None

    # Explicit exhaustive mode restores the ordinary indexed full-fit ranking.
    _save_ui_settings(project.project_dir, candidate_evaluation_budget=0)
    unrestricted = choose()
    assert unrestricted is not None
    assert position_by_unit[unrestricted["unit_id"]] not in sample_positions

    # Prepared fixed committees use the same finite candidate contract,
    # including disagreement (which is not a single-classifier strategy).
    second_spec = project.create_classifier_spec(
        code_id=code, name="budgeted second", geometry_id=geometry_id,
    )
    project.train_classifier(code_id=code, classifier_spec_id=second_spec)
    committee = project.create_classifier_committee(
        code_id=code, name="bounded mean",
        classifier_spec_ids=[spec, second_spec], aggregation="mean",
    )
    fixed_budget = 12
    _save_ui_settings(project.project_dir, candidate_evaluation_budget=fixed_budget)
    fixed_seed = (
        committee * 2654435761 + code * 2246822519 + 3
    ) & 0xFFFFFFFF
    a, b = deterministic_page_permutation_parameters(total, fixed_seed)
    fixed_window = {(a * rank + b) % total for rank in range(fixed_budget)}
    for strategy in ("likely_positive", "classifier_disagreement"):
        selected = project.focus_recommendation(
            code_id=code, session_id=session,
            strategy=strategy, committee_id=committee,
            recommendation_source="committee",
        )
        assert selected is not None
        assert position_by_unit[selected["unit_id"]] in fixed_window

    # The default remains the same full indexed ranking as explicit Exhaustive.
    _save_ui_settings(project.project_dir, candidate_evaluation_budget=-1)
    assert choose()["unit_id"] == unrestricted["unit_id"]


def test_code_budgeted_source_validates_strategy_and_budget(tmp_path: Path) -> None:
    project = _project(tmp_path)
    with pytest.raises(ValueError, match="positive integer"):
        project.database.ranked_budgeted_code_recommendation(
            source="classifier", source_id=1, code_id=1,
            session_id=1, strategy="likely_positive", budget=0,
        )
    with pytest.raises(ValueError, match="prepared fixed"):
        project.database.ranked_budgeted_code_recommendation(
            source="classifier", source_id=1, code_id=1,
            session_id=1, strategy="classifier_disagreement", budget=2,
        )


def _budgeted_committee_fixture(tmp_path: Path):
    project = GeometricCoder.create(
        project_dir=tmp_path / "committee-code-budget.geco",
        data=pd.DataFrame({
            "id": list(range(30)),
            "text": [
                f"helpful clear excellent example {i}" if i % 2
                else f"poor confusing weak example {i}"
                for i in range(30)
            ],
        }),
        keys=["id"], text="text",
        geometries={"lexical": CountGeometry(lemmatize=False, weighting="tfidf")},
    )
    code = project.create_code("Quality")
    geometry_id = int(project.geometries(public_only=True)[0]["geometry_id"])
    first = project.create_classifier_spec(
        code_id=code, name="committee logistic", geometry_id=geometry_id,
    )
    second = project.create_classifier_spec(
        code_id=code, name="committee NB", geometry_id=geometry_id,
        algorithm="complement_nb",
    )
    units = project.units()
    for i in (1, 3, 5):
        project.annotate(int(units[i]["observation_id"]), code, "positive")
    for i in (0, 2, 4):
        project.annotate(int(units[i]["observation_id"]), code, "negative")
    project.train_classifier(code_id=code, classifier_spec_id=first)
    project.train_classifier(code_id=code, classifier_spec_id=second)
    learned = project.create_classifier_committee(
        code_id=code, name="budgeted stack",
        classifier_spec_ids=[first, second], aggregation="logistic_stack",
    )
    learned_fit = project.train_classifier_committee(
        code_id=code, committee_id=learned,
    )
    fixed = project.create_classifier_committee(
        code_id=code, name="budgeted fixed",
        classifier_spec_ids=[first, second], aggregation="mean",
    )
    project.prepare_selected_committee(code_id=code, committee_id=fixed)
    classifier_fit_id = int(project.database.latest_classifier_fit(
        code_id=code, classifier_spec_id=first,
    )["classifier_fit_id"])
    return (project, code, units, first, classifier_fit_id,
            learned, int(learned_fit["committee_fit_id"]), fixed)


def test_finite_code_budget_learned_committee_stays_in_frozen_window(
    tmp_path: Path,
) -> None:
    from geometric_coder.explore import deterministic_page_permutation_parameters

    project, code, units, _, _, committee, fit_id, _ = (
        _budgeted_committee_fixture(tmp_path)
    )
    session = project.create_session("Learned budget")
    budget = 12
    _save_ui_settings(project.project_dir, candidate_evaluation_budget=budget)
    seed = (fit_id * 2654435761 + code * 2246822519 + 2) & 0xFFFFFFFF
    multiplier, shift = deterministic_page_permutation_parameters(len(units), seed)
    window = {
        (multiplier * rank + shift) % len(units) for rank in range(budget)
    }
    by_unit = {int(row["unit_id"]): i for i, row in enumerate(units)}

    def choose(strategy="committee_uncertainty"):
        return project.focus_recommendation(
            code_id=code, session_id=session, strategy=strategy,
            committee_id=committee, recommendation_source="committee",
            auto_retrain=False,
        )

    first = choose()
    assert first is not None
    assert by_unit[first["unit_id"]] in window
    assert first["committee_probability"] is not None
    assert choose()["unit_id"] == first["unit_id"]

    project.record_visit(session, int(first["unit_id"]), method="test")
    second = choose()
    assert second is not None and second["unit_id"] != first["unit_id"]
    assert by_unit[second["unit_id"]] in window
    project.annotate(
        int(units[by_unit[second["unit_id"]]]["observation_id"]), code, "positive",
    )
    assert choose()["unit_id"] != second["unit_id"]

    # Mark every sampled candidate visited: the finite window cannot expand.
    for position in window:
        project.record_visit(session, int(units[position]["unit_id"]), method="test")
    assert choose() is None
    assert choose("likely_positive") is None

    _save_ui_settings(project.project_dir, candidate_evaluation_budget=0)
    exhaustive = choose()
    assert exhaustive is not None
    assert by_unit[exhaustive["unit_id"]] not in window
    _save_ui_settings(project.project_dir, candidate_evaluation_budget=-1)
    assert choose()["unit_id"] == exhaustive["unit_id"]


@pytest.mark.parametrize("budget_extra", [0, 7])
def test_full_corpus_code_budget_reuses_indexed_rankings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, budget_extra: int,
) -> None:
    import geometric_coder.explore as explore

    project, code, units, classifier_spec, classifier_fit, learned, learned_fit, fixed = (
        _budgeted_committee_fixture(tmp_path)
    )
    session = project.create_session("Full-budget equivalence")
    database = project.database
    sources = [
        ("classifier", classifier_fit, ["likely_positive", "likely_negative",
                                        "classifier_uncertainty"]),
        ("learned_committee", learned_fit,
         ["likely_positive", "likely_negative", "committee_uncertainty"]),
        ("fixed_committee", fixed,
         ["likely_positive", "likely_negative", "committee_uncertainty",
          "classifier_disagreement"]),
    ]

    def exhaustive(source, source_id, strategy):
        if source == "classifier":
            return database.ranked_classifier_recommendation(
                classifier_fit_id=source_id, code_id=code,
                session_id=session, strategy=strategy,
            )
        if source == "learned_committee":
            return database.ranked_classifier_committee_recommendation(
                committee_fit_id=source_id, code_id=code,
                session_id=session, strategy=strategy,
            )
        return database.ranked_prepared_committee_recommendation(
            committee_id=source_id, code_id=code,
            session_id=session, strategy=strategy,
        )

    # Verify both tie-breaking/ranking equivalence and that the expensive
    # candidate permutation never runs when the budget covers the corpus.
    def forbid_permutation(*args, **kwargs):
        raise AssertionError("full-corpus budget must use existing indexes")

    monkeypatch.setattr(
        explore, "deterministic_page_permutation_parameters", forbid_permutation,
    )
    budget = len(units) + budget_extra
    _save_ui_settings(project.project_dir, candidate_evaluation_budget=budget)
    for source, source_id, strategies in sources:
        for strategy in strategies:
            expected = exhaustive(source, source_id, strategy)
            actual = database.ranked_budgeted_code_recommendation(
                source=source, source_id=source_id, code_id=code,
                session_id=session, strategy=strategy, budget=budget,
            )
            assert actual == expected
            if source == "classifier":
                recommendation = project.focus_recommendation(
                    code_id=code, session_id=session, strategy=strategy,
                    active_classifier_spec_id=classifier_spec,
                )
            else:
                recommendation = project.focus_recommendation(
                    code_id=code, session_id=session, strategy=strategy,
                    committee_id=learned if source == "learned_committee" else fixed,
                    recommendation_source="committee",
                )
            assert recommendation is not None
            assert recommendation["unit_id"] == expected["unit_id"]

    # Human labels and visits are still live eligibility filters.
    first = exhaustive("classifier", classifier_fit, "likely_positive")
    assert first is not None
    project.record_visit(session, first["unit_id"], method="test")
    after_visit = exhaustive("classifier", classifier_fit, "likely_positive")
    assert after_visit is not None and after_visit != first
    assert database.ranked_budgeted_code_recommendation(
        source="classifier", source_id=classifier_fit, code_id=code,
        session_id=session, strategy="likely_positive", budget=budget,
    ) == after_visit
    project.annotate(after_visit["observation_id"], code, "negative")
    after_label = exhaustive("classifier", classifier_fit, "likely_positive")
    assert after_label is not None and after_label != after_visit
    assert database.ranked_budgeted_code_recommendation(
        source="classifier", source_id=classifier_fit, code_id=code,
        session_id=session, strategy="likely_positive", budget=budget,
    ) == after_label


def test_forest_creation_controls_and_feature_parser(tmp_path):
    import inspect
    pytest.importorskip("dash")
    from geometric_coder.ui.app import create_app, _forest_feature_input

    app = create_app(_project(tmp_path))

    def find(component, component_id):
        if getattr(component, "id", None) == component_id:
            return component
        children = getattr(component, "children", None)
        for child in children if isinstance(children, (list, tuple)) else [children]:
            if child is not None:
                match = find(child, component_id)
                if match is not None:
                    return match

    assert "random_forest" in {
        option["value"] for option in find(app.layout, "classifier-spec-algorithm").options
    }
    for suffix, value in (
        ("trees", 100), ("depth", 8), ("leaf", 1), ("features", "sqrt"),
        ("weight", "balanced"), ("seed", 0), ("threshold", 0.5),
    ):
        assert find(app.layout, "classifier-spec-forest-" + suffix).value == value
    callback = next(
        inspect.unwrap(spec["callback"]) for key, spec in app.callback_map.items()
        if "classifier-spec-forest-options.style" in key
    )
    assert callback("random_forest") == ({}, {"display": "none"})
    assert callback("logistic_l2") == ({"display": "none"}, {})
    assert _forest_feature_input("all") is None
    assert type(_forest_feature_input("1")) is int
    assert type(_forest_feature_input("1.0")) is float
    with pytest.raises(ValueError):
        _forest_feature_input("invalid")


def test_forest_training_cv_reuse_and_reopen(tmp_path, monkeypatch):
    import numpy as np
    import geometric_coder.project as module
    project = _project(tmp_path)
    code = project.create_code("forest quality")
    geometry = int(project.geometries(public_only=True)[0]["geometry_id"])
    spec = project.create_classifier_spec(
        code_id=code, name="forest", geometry_id=geometry,
        algorithm="random_forest", hyperparameters={"n_estimators": 9, "max_features": 0.5},
    )
    units = project.units()
    for index in (0, 1, 3):
        project.annotate(int(units[index]["observation_id"]), code,
                         "positive" if index < 3 else "negative")
    first = project.train_classifier(code_id=code, classifier_spec_id=spec, tune=True)
    assert first["training_selection"]["reason"] == "insufficient_cv_data"
    project.annotate(int(units[4]["observation_id"]), code, "negative")
    calls = []
    original = module.fit_classifier

    def record(**kwargs):
        calls.append((kwargs["training_matrix"].shape[0], kwargs["scoring_matrix"].shape[0]))
        return original(**kwargs)

    monkeypatch.setattr(module, "fit_classifier", record)
    fit = project.train_classifier(code_id=code, classifier_spec_id=spec, tune=True, folds=2)
    selection = fit["training_selection"]
    assert selection["tuned"]
    assert selection["folds"] == 2
    assert len(calls) == 2 * len(selection["results"]) + 1
    assert all(pair == (2, 2) for pair in calls[:-1])
    assert calls[-1] == (4, len(units))
    assert selection["results"][0]["hyperparameters"]["max_features"] == 0.5
    assert all(row["hyperparameters"]["n_estimators"] == 9 for row in selection["results"])
    calls.clear()
    reopened = GeometricCoder.open(project.project_dir)
    retained = reopened.train_classifier(code_id=code, classifier_spec_id=spec)
    assert retained["classifier_fit_id"] == fit["classifier_fit_id"]
    assert calls == []
    exported = reopened.export_classifier("forest")
    matrix = reopened.geometry_matrix(geometry)
    np.testing.assert_allclose(exported.predict_proba([matrix]), [
        row["probability"] for row in reopened.database.classifier_predictions(
            int(fit["classifier_fit_id"]), atomic_only=True,
        )
    ])
    session = reopened.create_session("Forest")
    assert reopened.focus_recommendation(
        code_id=code, session_id=session, strategy="likely_positive",
        active_classifier_spec_id=spec,
    ) is not None
    reopened.create_apply_run(code_id=code, source_kind="classifier", source_id=spec)
    assert calls == []
