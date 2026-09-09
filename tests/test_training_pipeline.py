from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

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
    first = project.train_classifier(code_id=code_id, classifier_spec_id=classifier_id, tune=True)
    assert first["training_selection"]["tuned"] is False
    assert first["training_selection"]["reason"] == "insufficient_cv_data"

    project.annotate(int(observations[1]["observation_id"]), code_id, "positive")
    project.annotate(int(observations[4]["observation_id"]), code_id, "negative")
    second = project.train_classifier(code_id=code_id, classifier_spec_id=classifier_id, tune=True)
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

    trained = project.train_classifier(code_id=code_id, classifier_spec_id=classifier_id, tune=True)
    assert trained["training_selection"]["tuned"] is True
    before = len(project.classifier_tuning_runs(code_id=code_id, classifier_spec_id=classifier_id))

    reused = project.train_classifier(code_id=code_id, classifier_spec_id=classifier_id)
    after = len(project.classifier_tuning_runs(code_id=code_id, classifier_spec_id=classifier_id))
    assert reused["training_selection"]["reason"] == "current_fit_reused"
    assert before == after


def test_training_defaults_to_current_hyperparameters_without_cv(tmp_path: Path) -> None:
    project = _project(tmp_path)
    code_id = project.create_code("quality")
    observations = project.database.list_observations()
    classifier_id = int(project.classifier_specs(code_id=code_id)[0]["classifier_spec_id"])
    for position in (0, 1, 2):
        project.annotate(int(observations[position]["observation_id"]), code_id, "positive")
    for position in (3, 4, 5):
        project.annotate(int(observations[position]["observation_id"]), code_id, "negative")

    trained = project.train_classifier(code_id=code_id, classifier_spec_id=classifier_id)

    assert trained["training_selection"]["tuned"] is False
    assert trained["training_selection"]["reason"] == "tuning_disabled"
    assert project.classifier_tuning_runs(
        code_id=code_id, classifier_spec_id=classifier_id
    ) == []


def test_explicit_tuning_reruns_cv_even_when_fit_is_current(tmp_path: Path) -> None:
    project = _project(tmp_path)
    code_id = project.create_code("quality")
    observations = project.database.list_observations()
    classifier_id = int(project.classifier_specs(code_id=code_id)[0]["classifier_spec_id"])
    for position in (0, 1):
        project.annotate(int(observations[position]["observation_id"]), code_id, "positive")
    for position in (3, 4):
        project.annotate(int(observations[position]["observation_id"]), code_id, "negative")

    fast = project.train_classifier(code_id=code_id, classifier_spec_id=classifier_id)
    assert fast["training_selection"]["reason"] == "tuning_disabled"
    assert project.classifier_tuning_runs(
        code_id=code_id, classifier_spec_id=classifier_id
    ) == []

    tuned = project.train_classifier(
        code_id=code_id,
        classifier_spec_id=classifier_id,
        tune=True,
        retune_current=True,
    )
    runs = project.classifier_tuning_runs(
        code_id=code_id, classifier_spec_id=classifier_id
    )
    assert tuned["training_selection"]["tuned"] is True
    assert tuned["training_selection"]["folds"] == 2
    assert len(runs) == 1


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
    fits = project.train_classifiers(code_id=code_id, classifier_spec_ids=ids, tune=True)
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


def test_auto_training_uses_explicit_tuning_choice(tmp_path: Path) -> None:
    project = _project(tmp_path)
    code_id = project.create_code("quality")
    observations = project.database.list_observations()
    classifier_id = int(project.classifier_specs(code_id=code_id)[0]["classifier_spec_id"])
    for position in (0, 1):
        project.annotate(int(observations[position]["observation_id"]), code_id, "positive")
    for position in (3, 4):
        project.annotate(int(observations[position]["observation_id"]), code_id, "negative")
    session_id = project.create_session("Develop")

    project.focus_recommendation(
        code_id=code_id,
        session_id=session_id,
        strategy="likely_positive",
        active_classifier_spec_id=classifier_id,
        auto_retrain=True,
    )
    assert project.classifier_tuning_runs(
        code_id=code_id, classifier_spec_id=classifier_id
    ) == []

    # Merely leaving tuning enabled must not rerun CV before every recommendation
    # when the retained fit is already current.
    project.focus_recommendation(
        code_id=code_id,
        session_id=session_id,
        strategy="likely_positive",
        active_classifier_spec_id=classifier_id,
        auto_retrain=True,
        tune_hyperparameters=True,
    )
    assert project.classifier_tuning_runs(
        code_id=code_id, classifier_spec_id=classifier_id
    ) == []

    # Once new evidence makes the fit stale, automatic training honors the tuning
    # checkbox and performs CV before the refresh.
    project.annotate(int(observations[2]["observation_id"]), code_id, "positive")
    project.focus_recommendation(
        code_id=code_id,
        session_id=session_id,
        strategy="likely_positive",
        active_classifier_spec_id=classifier_id,
        auto_retrain=True,
        tune_hyperparameters=True,
    )
    assert len(
        project.classifier_tuning_runs(
            code_id=code_id, classifier_spec_id=classifier_id
        )
    ) == 1


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


def test_rename_classifier_preserves_fit_and_updates_live_identity(tmp_path: Path) -> None:
    project = _project(tmp_path)
    code_id = project.create_code("quality")
    observations = project.database.list_observations()
    classifier = project.classifier_specs(code_id=code_id)[0]
    classifier_id = int(classifier["classifier_spec_id"])
    old_name = str(classifier["name"])

    for position in (0, 1):
        project.annotate(int(observations[position]["observation_id"]), code_id, "positive")
    for position in (3, 4):
        project.annotate(int(observations[position]["observation_id"]), code_id, "negative")

    fit = project.train_classifier(code_id=code_id, classifier_spec_id=classifier_id)
    fit_id = int(fit["classifier_fit_id"])
    committee_id = project.create_classifier_committee(
        code_id=code_id,
        name="quality committee",
        classifier_spec_ids=[classifier_id],
        aggregation="mean",
    )

    project.rename_classifier_spec(classifier_id, name="quality renamed")

    renamed = project.database.get_classifier_spec(classifier_id)
    assert renamed["name"] == "quality renamed"
    assert int(renamed["classifier_spec_id"]) == classifier_id

    status = project.classifier_status(
        code_id=code_id, classifier_spec_ids=[classifier_id]
    )["classifiers"][0]
    assert status["status"] == "current"
    assert int(status["classifier_fit_id"]) == fit_id
    assert status["name"] == "quality renamed"

    reused = project.train_classifier(code_id=code_id, classifier_spec_id=classifier_id)
    assert int(reused["classifier_fit_id"]) == fit_id
    assert reused["classifier_name"] == "quality renamed"
    assert reused["training_selection"]["reason"] == "current_fit_reused"

    prediction_geometry = project.classifier_prediction_geometry(
        code_id=code_id, classifier_fit_ids=[fit_id]
    )
    assert prediction_geometry["classifier_names"] == ["quality renamed"]
    assert prediction_geometry["axis_titles"][0] == "quality renamed"

    # A rename is present-day metadata. The retained fit keeps the name recorded
    # when it was created for historical provenance.
    retained_fit = project.database.latest_classifier_fit(
        code_id=code_id, classifier_spec_id=classifier_id
    )
    assert retained_fit is not None
    assert retained_fit["classifier_name"] == old_name
    assert project.export_classifier("quality renamed") is not None
    with pytest.raises(KeyError, match="Unknown active classifier name"):
        project.export_classifier(old_name)

    committee = next(
        row for row in project.classifier_committees(code_id)
        if int(row["committee_id"]) == committee_id
    )
    assert [member["classifier_name"] for member in committee["members"]] == [
        "quality renamed"
    ]


def test_rename_classifier_requires_project_wide_unique_active_name(tmp_path: Path) -> None:
    project = _project(tmp_path)
    code_a = project.create_code("quality")
    code_b = project.create_code("relevance")
    first_id = int(project.classifier_specs(code_id=code_a)[0]["classifier_spec_id"])
    second_id = int(project.classifier_specs(code_id=code_b)[0]["classifier_spec_id"])

    project.rename_classifier_spec(first_id, name="shared name")
    with pytest.raises(ValueError, match="active predictor named 'shared name' already exists \\(classifier\\)"):
        project.rename_classifier_spec(second_id, name="shared name")

    assert project.database.get_classifier_spec(first_id)["name"] == "shared name"
    assert project.database.get_classifier_spec(second_id)["name"] != "shared name"


def test_rename_classifier_can_claim_an_archived_name_by_bumping_archive(tmp_path: Path) -> None:
    project = _project(tmp_path)
    code_id = project.create_code("quality")
    geometry_id = int(project.database.list_geometries(public_only=True)[0]["geometry_id"])

    archived_id = project.create_classifier_spec(
        code_id=code_id,
        name="old",
        geometry_id=geometry_id,
        algorithm="decision_tree",
    )
    project.delete_classifier_spec(archived_id)
    assert project.database.get_classifier_spec(archived_id)["name"] == "old_deleted"

    active_id = project.create_classifier_spec(
        code_id=code_id,
        name="current",
        geometry_id=geometry_id,
        algorithm="decision_tree",
    )
    project.rename_classifier_spec(active_id, name="old_deleted")

    assert project.database.get_classifier_spec(active_id)["name"] == "old_deleted"
    assert project.database.get_classifier_spec(archived_id)["name"] == "old_deleted_1"


def test_deleted_classifier_cannot_be_renamed(tmp_path: Path) -> None:
    project = _project(tmp_path)
    code_id = project.create_code("quality")
    classifier_id = int(project.classifier_specs(code_id=code_id)[0]["classifier_spec_id"])
    project.delete_classifier_spec(classifier_id)

    with pytest.raises(ValueError, match="Deleted classifiers cannot be renamed"):
        project.rename_classifier_spec(classifier_id, name="revived")


def test_fast_training_loads_atomic_geometry_once_and_current_reuse_zero_times(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path)
    code_id = project.create_code("quality")
    observations = project.database.list_observations()
    classifier_id = int(project.classifier_specs(code_id=code_id)[0]["classifier_spec_id"])
    for position in (0, 1, 2):
        project.annotate(int(observations[position]["observation_id"]), code_id, "positive")
    for position in (3, 4, 5):
        project.annotate(int(observations[position]["observation_id"]), code_id, "negative")

    original = project.geometry_matrix
    calls = 0

    def counted_geometry_matrix(geometry):
        nonlocal calls
        calls += 1
        return original(geometry)

    monkeypatch.setattr(project, "geometry_matrix", counted_geometry_matrix)

    trained = project.train_classifier(code_id=code_id, classifier_spec_id=classifier_id)
    assert trained["training_selection"]["reason"] == "tuning_disabled"
    assert calls == 1

    calls = 0
    reused = project.train_classifier(code_id=code_id, classifier_spec_id=classifier_id)
    assert reused["training_selection"]["reason"] == "current_fit_reused"
    assert calls == 0


def _trained_logistic_stack_committee(tmp_path: Path):
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
        name="tree stack member",
        geometry_id=geometry_id,
        algorithm="decision_tree",
    )
    project.train_classifiers(
        code_id=code_id, classifier_spec_ids=[logistic_id, tree_id]
    )
    committee_id = project.create_classifier_committee(
        code_id=code_id,
        name="learned committee",
        classifier_spec_ids=[logistic_id, tree_id],
        aggregation="logistic_stack",
    )
    fit = project.train_classifier_committee(code_id=code_id, committee_id=committee_id)
    session_id = project.create_session("Develop")
    return project, code_id, committee_id, int(fit["committee_fit_id"]), session_id


def test_current_learned_committee_training_does_not_reload_member_score_maps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, code_id, committee_id, fit_id, _ = _trained_logistic_stack_committee(tmp_path)
    calls = 0
    original = project.database.classifier_score_maps

    def counted(fit_ids):
        nonlocal calls
        calls += 1
        return original(fit_ids)

    monkeypatch.setattr(project.database, "classifier_score_maps", counted)
    reused = project.train_classifier_committee(code_id=code_id, committee_id=committee_id)
    assert int(reused["committee_fit_id"]) == fit_id
    assert calls == 0


def test_learned_committee_uncertainty_uses_committee_vector_then_only_selected_member_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, code_id, committee_id, _, session_id = _trained_logistic_stack_committee(tmp_path)
    vector_calls = 0
    point_calls = 0
    original_vectors = project.database.classifier_atomic_score_vectors
    original_point = project.database.classifier_scores_for_unit

    def counted_vectors(fit_ids):
        nonlocal vector_calls
        vector_calls += 1
        return original_vectors(fit_ids)

    def counted_point(fit_ids, unit_id):
        nonlocal point_calls
        point_calls += 1
        return original_point(fit_ids, unit_id)

    monkeypatch.setattr(project.database, "classifier_atomic_score_vectors", counted_vectors)
    monkeypatch.setattr(project.database, "classifier_scores_for_unit", counted_point)
    result = project.focus_recommendation(
        code_id=code_id,
        session_id=session_id,
        strategy="committee_uncertainty",
        committee_id=committee_id,
        recommendation_source="committee",
        auto_retrain=False,
    )
    assert result is not None
    assert vector_calls == 0
    assert point_calls == 1
    assert set(result["probabilities"]) == {
        project.classifier_specs(code_id=code_id)[0]["name"],
        "tree stack member",
    }
    assert result["committee_probability"] is not None
    assert 0.0 <= float(result["committee_probability"]) <= 1.0


def test_learned_committee_disagreement_batches_member_vectors_once_and_reports_committee_probability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, code_id, committee_id, _, session_id = _trained_logistic_stack_committee(tmp_path)
    vector_calls = 0
    original_vectors = project.database.classifier_atomic_score_vectors

    def counted_vectors(fit_ids):
        nonlocal vector_calls
        vector_calls += 1
        return original_vectors(fit_ids)

    monkeypatch.setattr(project.database, "classifier_atomic_score_vectors", counted_vectors)
    result = project.focus_recommendation(
        code_id=code_id,
        session_id=session_id,
        strategy="classifier_disagreement",
        committee_id=committee_id,
        recommendation_source="committee",
        auto_retrain=False,
    )
    assert result is not None
    assert vector_calls == 1
    assert len(result["probabilities"]) == 2
    assert result["committee_probability"] is not None


def test_rename_committee_preserves_learned_fit_and_uses_shared_predictor_namespace(
    tmp_path: Path,
) -> None:
    project, code_id, committee_id, fit_id, _ = _trained_logistic_stack_committee(tmp_path)
    before = project.database.get_classifier_committee(committee_id)
    before_updated = before["updated_at"]

    project.rename_classifier_committee(committee_id, name="renamed learned committee")
    renamed = project.database.get_classifier_committee(committee_id)
    assert renamed["name"] == "renamed learned committee"
    assert renamed["updated_at"] == before_updated
    assert int(
        project.database.latest_classifier_committee_fit(
            committee_id=committee_id, code_id=code_id
        )["committee_fit_id"]
    ) == fit_id
    assert project.classifier_committee_status(
        code_id=code_id, committee_id=committee_id
    )["status"] == "current"

    other_id = project.create_classifier_committee(
        code_id=code_id,
        name="other committee",
        classifier_spec_ids=[
            int(row["classifier_spec_id"])
            for row in project.classifier_specs(code_id=code_id)
        ],
        aggregation="mean",
    )
    with pytest.raises(ValueError, match="active predictor named 'renamed learned committee' already exists \\(committee\\)"):
        project.rename_classifier_committee(other_id, name="renamed learned committee")

    project.delete_classifier_committee(committee_id)
    project.rename_classifier_committee(other_id, name="renamed learned committee")
    assert project.database.get_classifier_committee(other_id)["name"] == "renamed learned committee"


def test_current_auto_retrained_learned_committee_recommendation_avoids_full_member_score_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, code_id, committee_id, _, session_id = _trained_logistic_stack_committee(tmp_path)
    full_score_map_calls = 0
    atomic_vector_calls = 0
    original_full = project.database.classifier_score_maps
    original_atomic = project.database.classifier_atomic_score_vectors

    def counted_full(fit_ids):
        nonlocal full_score_map_calls
        full_score_map_calls += 1
        return original_full(fit_ids)

    def counted_atomic(fit_ids):
        nonlocal atomic_vector_calls
        atomic_vector_calls += 1
        return original_atomic(fit_ids)

    monkeypatch.setattr(project.database, "classifier_score_maps", counted_full)
    monkeypatch.setattr(project.database, "classifier_atomic_score_vectors", counted_atomic)
    result = project.focus_recommendation(
        code_id=code_id,
        session_id=session_id,
        strategy="committee_uncertainty",
        committee_id=committee_id,
        recommendation_source="committee",
        auto_retrain=True,
        auto_train_all=False,
    )
    assert result is not None
    assert full_score_map_calls == 0
    assert atomic_vector_calls == 0


def test_fixed_committee_recommendation_reports_member_and_committee_probabilities(
    tmp_path: Path,
) -> None:
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
        code_id=code_id, name="tree fixed member", geometry_id=geometry_id, algorithm="decision_tree"
    )
    project.train_classifiers(code_id=code_id, classifier_spec_ids=[logistic_id, tree_id])
    committee_id = project.create_classifier_committee(
        code_id=code_id,
        name="fixed mean",
        classifier_spec_ids=[logistic_id, tree_id],
        aggregation="mean",
    )
    result = project.focus_recommendation(
        code_id=code_id,
        session_id=project.create_session("Develop"),
        strategy="likely_positive",
        committee_id=committee_id,
        recommendation_source="committee",
    )
    assert result is not None
    assert len(result["probabilities"]) == 2
    expected = sum(float(value) for value in result["probabilities"].values()) / 2.0
    assert float(result["committee_probability"]) == pytest.approx(expected)
