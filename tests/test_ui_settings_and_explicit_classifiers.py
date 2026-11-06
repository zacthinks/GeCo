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
    assert _read_ui_settings(root) == {"cv_folds": 5}
    _save_ui_settings(root, cv_folds=3)
    assert _read_ui_settings(root) == {"cv_folds": 3}
    with pytest.raises(ValueError, match="between 2 and 10"):
        _save_ui_settings(root, cv_folds=1)
    assert _read_ui_settings(root) == {"cv_folds": 3}
    (root / "ui_settings.json").write_text('{"cv_folds": "garbled"}', encoding="utf-8")
    assert _read_ui_settings(root) == {"cv_folds": 5}


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
