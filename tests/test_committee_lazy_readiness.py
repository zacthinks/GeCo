"""Committee lifecycle: derived fixed rankings and learned fit freshness."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest

from geometric_coder import CountGeometry, GeometricCoder


def _trained_committee(tmp_path, aggregation="harmonic_mean"):
    project = GeometricCoder.create(
        project_dir=tmp_path / f"committee-{aggregation}.geco",
        data=pd.DataFrame({
            "row_id": list(range(12)),
            "text": [
                "bright useful clear", "helpful excellent clear", "excellent kind",
                "wonderful clear", "helpful informative", "good clear",
                "awful confusing", "bad weak unclear", "terrible disappointing",
                "poor misleading", "difficult unclear", "weak bad",
            ],
        }),
        keys=["row_id"], text="text", metadata=[],
        geometries={"lexical": CountGeometry(lemmatize=False, weighting="tfidf")},
    )
    code_id = project.create_code("quality")
    geom_id = int(project.geometries(public_only=True)[0]["geometry_id"])
    first = project.create_classifier_spec(
        code_id=code_id, name="main quality", geometry_id=geom_id
    )
    second = project.create_classifier_spec(
        code_id=code_id, name="backup quality", geometry_id=geom_id,
        algorithm="complement_nb",
    )
    observations = project.database.list_observations()
    for index in (0, 1):
        project.annotate(int(observations[index]["observation_id"]), code_id, "positive")
    for index in (6, 7):
        project.annotate(int(observations[index]["observation_id"]), code_id, "negative")
    a = project.train_classifier(code_id=code_id, classifier_spec_id=first)
    b = project.train_classifier(code_id=code_id, classifier_spec_id=second)
    committee_id = project.create_classifier_committee(
        code_id=code_id, name=f"quality {aggregation}",
        classifier_spec_ids=[first, second], aggregation=aggregation,
    )
    return project, code_id, (first, second), (a, b), committee_id, observations


def _ranking_recommend(project, code_id, committee_id, *, strategy="likely_positive"):
    return project.focus_recommendation(
        code_id=code_id, session_id=project.create_session("recommend"),
        strategy=strategy, committee_id=committee_id,
        recommendation_source="committee", auto_retrain=False,
    )


@pytest.mark.parametrize("aggregation", [
    "mean", "median", "minimum", "maximum", "harmonic_mean", "geometric_mean"
])
def test_fixed_committee_first_use_builds_and_reuses_derived_ranking(
    tmp_path, aggregation, monkeypatch
):
    project, code_id, _, _, committee_id, _ = _trained_committee(tmp_path, aggregation)
    assert project.database.prepared_committee_ranking(committee_id) is None
    assert project.classifier_committee_status(
        code_id=code_id, committee_id=committee_id
    )["status"] == "current"
    first = _ranking_recommend(project, code_id, committee_id)
    assert first and first["committee_probability"] is not None
    cached = project.database.prepared_committee_ranking(committee_id)
    assert cached and cached["prediction_count"] == 12
    assert first["stale_committee"] is False

    def forbid_rebuild(*args, **kwargs):
        raise AssertionError("unchanged rankings must not be recomputed")

    monkeypatch.setattr(project.database, "save_prepared_committee_ranking", forbid_rebuild)
    again = _ranking_recommend(
        project, code_id, committee_id, strategy="classifier_disagreement"
    )
    assert again and again["committee_probability"] is not None
    project.rename_classifier_committee(committee_id, name=f"renamed {aggregation}")
    assert _ranking_recommend(project, code_id, committee_id)
    assert project.database.prepared_committee_ranking(committee_id) == cached


def test_fixed_committee_stale_evidence_keeps_valid_frozen_ranking(tmp_path, monkeypatch):
    project, code_id, _, _, committee_id, observations = _trained_committee(tmp_path)
    assert _ranking_recommend(project, code_id, committee_id)
    prepared = project.database.prepared_committee_ranking(committee_id)
    project.annotate(int(observations[2]["observation_id"]), code_id, "positive")
    assert project.classifier_committee_status(
        code_id=code_id, committee_id=committee_id
    )["status"] == "stale"

    def forbid_rebuild(*args, **kwargs):
        raise AssertionError("human evidence alone does not change frozen fit rankings")

    monkeypatch.setattr(project.database, "save_prepared_committee_ranking", forbid_rebuild)
    result = _ranking_recommend(project, code_id, committee_id)
    assert result and result["stale_committee"] is True
    assert project.database.prepared_committee_ranking(committee_id) == prepared


def test_fixed_committee_refit_replaces_cache_on_first_next_use(tmp_path):
    project, code_id, (first, _), _, committee_id, observations = _trained_committee(tmp_path)
    assert _ranking_recommend(project, code_id, committee_id)
    before = project.database.prepared_committee_ranking(committee_id)
    project.annotate(int(observations[2]["observation_id"]), code_id, "positive")
    newer = project.train_classifier(code_id=code_id, classifier_spec_id=first)
    assert int(newer["classifier_fit_id"]) != int(before["member_fit_ids"][0])
    # The obsolete numeric index is deleted with the changed fit; no eager
    # replacement is computed until a recommendation first needs it.
    assert project.database.prepared_committee_ranking(committee_id) is None
    result = _ranking_recommend(project, code_id, committee_id)
    assert result and result["stale_committee"] is True  # other member is still stale
    after = project.database.prepared_committee_ranking(committee_id)
    assert after["member_fit_ids"][0] == int(newer["classifier_fit_id"])
    assert after["member_fit_ids"] != before["member_fit_ids"]


def test_fixed_committee_missing_member_fit_is_not_trained(tmp_path):
    project, code_id, specs, _, committee_id, _ = _trained_committee(tmp_path)
    third = project.create_classifier_spec(
        code_id=code_id, name="untrained", geometry_id=int(
            project.geometries(public_only=True)[0]["geometry_id"]
        )
    )
    project.update_classifier_committee(
        committee_id, name="expanded", classifier_spec_ids=[*specs, third],
        aggregation="harmonic_mean"
    )
    assert project.classifier_committee_status(
        code_id=code_id, committee_id=committee_id
    )["status"] == "not_trained"
    with pytest.raises(ValueError, match="Train committee members"):
        _ranking_recommend(project, code_id, committee_id)


def test_learned_committee_stale_fit_remains_usable_without_retraining(tmp_path, monkeypatch):
    project, code_id, specs, _, committee_id, observations = _trained_committee(
        tmp_path, "logistic_stack"
    )
    fitted = project.train_classifier_committee(
        code_id=code_id, committee_id=committee_id
    )
    assert project.classifier_committee_status(
        code_id=code_id, committee_id=committee_id
    )["status"] == "current"
    project.annotate(int(observations[2]["observation_id"]), code_id, "positive")
    assert project.classifier_committee_status(
        code_id=code_id, committee_id=committee_id
    )["status"] == "stale"

    def forbid_training(*args, **kwargs):
        raise AssertionError("recommendations must not silently train a committee")

    monkeypatch.setattr(project, "train_classifier_committee", forbid_training)
    result = _ranking_recommend(project, code_id, committee_id)
    assert result and result["stale_committee"] is True
    current = project.database.latest_classifier_committee_fit(
        committee_id=committee_id, code_id=code_id
    )
    assert current["committee_fit_id"] == fitted["committee_fit_id"]


def test_first_use_by_multiple_request_threads_publishes_one_complete_ranking(tmp_path):
    project, code_id, _, _, committee_id, _ = _trained_committee(tmp_path)
    def prepare(_):
        return project.prepare_selected_committee(
            code_id=code_id, committee_id=committee_id
        )
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(prepare, range(4)))
    assert sum(bool(row["prepared"]) for row in results) == 1
    assert all(row["unit_count"] == 12 for row in results)
    prepared = project.database.prepared_committee_ranking(committee_id)
    assert prepared is not None and prepared["prediction_count"] == 12


def test_definition_change_deletes_only_disposable_ranking(tmp_path):
    project, code_id, specs, _, committee_id, _ = _trained_committee(tmp_path)
    assert _ranking_recommend(project, code_id, committee_id)
    assert project.database.prepared_committee_ranking(committee_id)
    project.update_classifier_committee(
        committee_id, name="new mean", classifier_spec_ids=list(reversed(specs)),
        aggregation="mean"
    )
    assert project.database.prepared_committee_ranking(committee_id) is None
    assert project.classifier_committee_status(
        code_id=code_id, committee_id=committee_id
    )["status"] == "current"
    assert _ranking_recommend(project, code_id, committee_id)
    prepared = project.database.prepared_committee_ranking(committee_id)
    assert prepared["aggregation"] == "mean"


def test_frozen_learned_scores_not_attributed_to_retrained_member(tmp_path):
    project, code_id, (first, _), _, committee_id, observations = _trained_committee(
        tmp_path, "logistic_stack"
    )
    old_committee_fit = project.train_classifier_committee(
        code_id=code_id, committee_id=committee_id
    )
    project.annotate(int(observations[2]["observation_id"]), code_id, "positive")
    project.train_classifier(code_id=code_id, classifier_spec_id=first)
    assert project.classifier_committee_status(
        code_id=code_id, committee_id=committee_id
    )["status"] == "stale"
    frozen = _ranking_recommend(project, code_id, committee_id)
    assert frozen and frozen["stale_committee"] is True
    assert frozen["committee_probability"] is not None
    # The aggregation is from the old trained committee, NOT an application
    # of its coefficients to today's newly fitted member predictions.
    assert frozen["probabilities"] == {}
    assert frozen["classifier_fit_ids"] == {}
    assert project.database.latest_classifier_committee_fit(
        committee_id=committee_id, code_id=code_id
    )["committee_fit_id"] == old_committee_fit["committee_fit_id"]


def test_apply_from_stale_stacker_records_frozen_member_ids(tmp_path):
    project, code_id, (first, _), _, committee_id, observations = _trained_committee(
        tmp_path, "logistic_stack"
    )
    frozen = project.train_classifier_committee(
        code_id=code_id, committee_id=committee_id
    )
    original_ids = frozen["training_snapshot"]["member_fit_ids"]
    project.annotate(int(observations[2]["observation_id"]), code_id, "positive")
    newer = project.train_classifier(code_id=code_id, classifier_spec_id=first)
    assert int(newer["classifier_fit_id"]) not in original_ids
    apply_id = project.create_apply_run(
        code_id=code_id, source_kind="committee",
        source_id=committee_id, threshold=0.5,
    )
    draft = project.apply_run(apply_id)
    assert draft["committee_fit_id"] == frozen["committee_fit_id"]
    assert sorted(draft["classifier_fit_ids"]) == sorted(original_ids)
    assert int(newer["classifier_fit_id"]) not in draft["classifier_fit_ids"]


def test_stale_learned_committee_uses_frozen_scores_with_unfitted_new_member(tmp_path):
    project, code_id, (first, second), _, committee_id, _ = _trained_committee(
        tmp_path, "logistic_stack"
    )
    frozen = project.train_classifier_committee(
        code_id=code_id, committee_id=committee_id
    )
    third = project.create_classifier_spec(
        code_id=code_id, name="brand new committee member",
        geometry_id=int(project.geometries(public_only=True)[0]["geometry_id"])
    )
    project.update_classifier_committee(
        committee_id, name="expanded learner",
        classifier_spec_ids=[first, second, third],
        aggregation="logistic_stack",
    )
    assert project.classifier_committee_status(
        code_id=code_id, committee_id=committee_id
    )["status"] == "stale"
    candidate = _ranking_recommend(project, code_id, committee_id)
    assert candidate and candidate["stale_committee"] is True
    assert candidate["committee_probability"] is not None
    assert candidate["probabilities"] == {}
    draft_id = project.create_apply_run(
        code_id=code_id, source_kind="committee", source_id=committee_id,
    )
    assert project.apply_run(draft_id)["committee_fit_id"] == frozen["committee_fit_id"]
