from __future__ import annotations

from pathlib import Path

import pandas as pd

from geometric_coder import CountGeometry, GeometricCoder


def _project(tmp_path: Path) -> GeometricCoder:
    data = pd.DataFrame(
        {
            "row_id": list(range(8)),
            "text": [
                "excellent clear helpful",
                "excellent useful strong",
                "helpful clear useful",
                "bad confusing weak",
                "terrible unclear weak",
                "bad difficult confusing",
                "excellent clear",
                "bad weak",
            ],
        }
    )
    return GeometricCoder.create(
        project_dir=tmp_path / "training.geco",
        data=data,
        keys=["row_id"],
        text="text",
        metadata=[],
        geometries={
            "lexical": CountGeometry(lemmatize=False, weighting="tfidf")
        },
    )


def test_train_uses_defaults_until_cv_is_possible_then_tunes(tmp_path: Path) -> None:
    project = _project(tmp_path)
    code_id = project.create_code("quality")
    observations = project.database.list_observations()
    classifier = project.classifier_specs(code_id=code_id)[0]
    classifier_id = int(classifier["classifier_spec_id"])

    project.annotate(int(observations[0]["observation_id"]), code_id, "positive")
    project.annotate(int(observations[3]["observation_id"]), code_id, "negative")
    first = project.train_classifier(code_id=code_id, classifier_spec_id=classifier_id)
    assert first["training_selection"]["tuned"] is False
    assert first["training_selection"]["reason"] == "insufficient_cv_data"

    project.annotate(int(observations[1]["observation_id"]), code_id, "positive")
    project.annotate(int(observations[4]["observation_id"]), code_id, "negative")
    second = project.train_classifier(code_id=code_id, classifier_spec_id=classifier_id)
    assert second["training_selection"]["tuned"] is True
    assert second["training_selection"]["folds"] == 2
    assert project.classifier_status(
        code_id=code_id, classifier_spec_ids=[classifier_id]
    )["classifiers"][0]["status"] == "current"


def test_training_current_fit_does_not_rerun_cv(tmp_path: Path) -> None:
    project = _project(tmp_path)
    code_id = project.create_code("quality")
    observations = project.database.list_observations()
    classifier_id = int(project.classifier_specs(code_id=code_id)[0]["classifier_spec_id"])
    for position in (0, 1):
        project.annotate(int(observations[position]["observation_id"]), code_id, "positive")
    for position in (3, 4):
        project.annotate(int(observations[position]["observation_id"]), code_id, "negative")

    trained = project.train_classifier(code_id=code_id, classifier_spec_id=classifier_id)
    assert trained["training_selection"]["tuned"] is True
    before = len(project.classifier_tuning_runs(code_id=code_id, classifier_spec_id=classifier_id))

    reused = project.train_classifier(code_id=code_id, classifier_spec_id=classifier_id)
    after = len(project.classifier_tuning_runs(code_id=code_id, classifier_spec_id=classifier_id))
    assert reused["training_selection"]["reason"] == "current_fit_reused"
    assert before == after


def test_new_families_train_through_common_pipeline(tmp_path: Path) -> None:
    project = _project(tmp_path)
    code_id = project.create_code("quality")
    observations = project.database.list_observations()
    for position in (0, 1, 2):
        project.annotate(int(observations[position]["observation_id"]), code_id, "positive")
    for position in (3, 4, 5):
        project.annotate(int(observations[position]["observation_id"]), code_id, "negative")
    geometry_id = int(project.database.list_geometries(public_only=True)[0]["geometry_id"])

    algorithms = [
        "logistic_l1",
        "logistic_elasticnet",
        "complement_nb",
        "linear_svm",
        "decision_tree",
        "knn",
    ]
    ids = [
        project.create_classifier_spec(
            code_id=code_id,
            name=algorithm,
            geometry_id=geometry_id,
            algorithm=algorithm,
        )
        for algorithm in algorithms
    ]
    fits = project.train_classifiers(code_id=code_id, classifier_spec_ids=ids)
    assert set(fits) == set(ids)
    assert all(fit["score_kind"] == "probability" for fit in fits.values())
    assert all(fit["training_selection"]["tuned"] for fit in fits.values())


def test_auto_training_respects_train_all_scope(tmp_path: Path) -> None:
    project = _project(tmp_path)
    code_id = project.create_code("quality")
    observations = project.database.list_observations()
    for position in (0, 1):
        project.annotate(int(observations[position]["observation_id"]), code_id, "positive")
    for position in (3, 4):
        project.annotate(int(observations[position]["observation_id"]), code_id, "negative")
    geometry_id = int(project.database.list_geometries(public_only=True)[0]["geometry_id"])
    default_id = int(project.classifier_specs(code_id=code_id)[0]["classifier_spec_id"])
    tree_id = project.create_classifier_spec(
        code_id=code_id,
        name="tree",
        geometry_id=geometry_id,
        algorithm="decision_tree",
    )
    session_id = project.create_session("Develop")

    project.focus_recommendation(
        code_id=code_id,
        session_id=session_id,
        strategy="likely_positive",
        active_classifier_spec_id=default_id,
        auto_retrain=True,
        auto_train_all=True,
    )
    statuses = {
        int(row["classifier_spec_id"]): row["status"]
        for row in project.classifier_status(code_id=code_id)["classifiers"]
    }
    assert statuses[default_id] == "current"
    assert statuses[tree_id] == "current"

    # New evidence makes both retained fits stale. With Train all cleared, the
    # automatic refresh should update only the active classifier.
    project.annotate(int(observations[2]["observation_id"]), code_id, "positive")
    project.focus_recommendation(
        code_id=code_id,
        session_id=session_id,
        strategy="likely_positive",
        active_classifier_spec_id=default_id,
        auto_retrain=True,
        auto_train_all=False,
    )
    statuses = {
        int(row["classifier_spec_id"]): row["status"]
        for row in project.classifier_status(code_id=code_id)["classifiers"]
    }
    assert statuses[default_id] == "current"
    assert statuses[tree_id] == "stale"


def test_mixed_family_committee_and_apply_use_probability_outputs(tmp_path: Path) -> None:
    project = _project(tmp_path)
    code_id = project.create_code("quality")
    observations = project.database.list_observations()
    for position in (0, 1, 2):
        project.annotate(int(observations[position]["observation_id"]), code_id, "positive")
    for position in (3, 4, 5):
        project.annotate(int(observations[position]["observation_id"]), code_id, "negative")
    geometry_id = int(project.database.list_geometries(public_only=True)[0]["geometry_id"])
    logistic_id = int(project.classifier_specs(code_id=code_id)[0]["classifier_spec_id"])
    tree_id = project.create_classifier_spec(
        code_id=code_id,
        name="tree",
        geometry_id=geometry_id,
        algorithm="decision_tree",
    )
    project.train_classifiers(
        code_id=code_id, classifier_spec_ids=[logistic_id, tree_id]
    )
    committee_id = project.create_classifier_committee(
        code_id=code_id,
        name="mixed",
        classifier_spec_ids=[logistic_id, tree_id],
        aggregation="mean",
    )
    apply_run_id = project.create_apply_run(
        code_id=code_id,
        source_kind="committee",
        source_id=committee_id,
        threshold=0.5,
    )
    proposals = project.apply_proposals(apply_run_id)
    assert proposals
    assert all(0.0 <= float(row["probability"]) <= 1.0 for row in proposals)
