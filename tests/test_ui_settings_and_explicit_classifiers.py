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
    project.annotate(int(units[0]["observation_id"]), code, "positive")
    project.annotate(int(units[1]["observation_id"]), code, "negative")
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
