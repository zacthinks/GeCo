"""Regression tests for cross-fitted learned committee probabilities and provenance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import geometric_coder.committee_stacking as stacking
from geometric_coder import CountGeometry, GeometricCoder
from geometric_coder.committee_stacking import MemberTrainingData, out_of_fold_probabilities


def _data() -> tuple[np.ndarray, np.ndarray]:
    ids = np.arange(12, dtype=int)
    labels = np.asarray([0] * 6 + [1] * 6, dtype=int)
    return ids, labels


def test_oof_excludes_heldout_ids_for_every_member_and_is_deterministic(monkeypatch):
    ids, labels = _data()
    # First column identifies the observation even when geometries differ.
    dense = np.column_stack([ids, np.arange(12) ** 2, np.ones(12)])
    wide = sparse.csr_matrix(np.column_stack([ids, np.arange(12), np.ones(12), labels]))
    members = [
        MemberTrainingData("logistic_l2", {"feature_scaling": "fixed"},
                           tuple(ids), labels, dense),
        MemberTrainingData("linear_svm", {"probability": True},
                           tuple(ids), labels, wide),
    ]
    observed = []
    original = stacking.fit_classifier

    def guarded(*, training_matrix, scoring_matrix, labels, **kwargs):
        train = training_matrix.toarray() if sparse.issparse(training_matrix) else training_matrix
        held = scoring_matrix.toarray() if sparse.issparse(scoring_matrix) else scoring_matrix
        train_ids = set(train[:, 0].astype(int).tolist())
        held_ids = set(held[:, 0].astype(int).tolist())
        assert train_ids.isdisjoint(held_ids)
        assert set(np.unique(labels)) == {0, 1}
        observed.append((train_ids, held_ids))
        return original(
            training_matrix=training_matrix, scoring_matrix=scoring_matrix,
            labels=labels, **kwargs,
        )

    monkeypatch.setattr(stacking, "fit_classifier", guarded)
    first = out_of_fold_probabilities(
        observation_ids=ids, labels=labels, members=members,
        n_splits=5, random_state=0,
    )
    assert first.shape == (12, 2)
    assert len(observed) == 10
    assert np.isfinite(first).all() and ((0 <= first) & (first <= 1)).all()
    second = out_of_fold_probabilities(
        observation_ids=ids, labels=labels, members=members,
        n_splits=5, random_state=0,
    )
    np.testing.assert_allclose(first, second, rtol=0, atol=0)


def test_member_only_evidence_does_not_include_common_heldout_ids(monkeypatch):
    ids = np.arange(8, dtype=int)
    labels = np.asarray([0] * 4 + [1] * 4, dtype=int)
    member_ids = tuple(list(ids) + [8, 9])
    member_y = np.r_[labels, [0, 1]]
    x = np.arange(10, dtype=float).reshape(-1, 1)
    seen = []
    original = stacking.fit_classifier

    def guarded(*, training_matrix, scoring_matrix, labels, **kwargs):
        seen.append((set(training_matrix[:, 0].astype(int)),
                     set(scoring_matrix[:, 0].astype(int))))
        assert seen[-1][0].isdisjoint(seen[-1][1])
        return original(training_matrix=training_matrix, scoring_matrix=scoring_matrix,
                        labels=labels, **kwargs)

    monkeypatch.setattr(stacking, "fit_classifier", guarded)
    values = out_of_fold_probabilities(
        observation_ids=ids, labels=labels,
        members=[MemberTrainingData("logistic_l2", {}, member_ids, member_y, x)],
        n_splits=4, random_state=0,
    )
    assert values.shape == (8, 1)
    assert all({8, 9}.issubset(train) for train, _ in seen)


def test_oof_rejects_too_few_examples_and_unavailable_probabilities():
    ids = np.arange(4, dtype=int)
    y = np.array([0, 0, 0, 1])
    x = np.column_stack([ids, ids])
    member = MemberTrainingData("logistic_l2", {}, tuple(ids), y, x)
    with pytest.raises(ValueError, match="at least two Present and two Absent"):
        out_of_fold_probabilities(
            observation_ids=ids, labels=y, members=[member],
            n_splits=2, random_state=0,
        )
    y = np.asarray([0, 0, 1, 1])
    margin_only = MemberTrainingData("linear_svm", {"probability": False},
                                     tuple(ids), y, x)
    with pytest.raises(ValueError, match="must provide probabilities"):
        out_of_fold_probabilities(
            observation_ids=ids, labels=y, members=[margin_only],
            n_splits=2, random_state=0,
        )


def _project(tmp_path: Path) -> tuple[GeometricCoder, int, int]:
    data = pd.DataFrame({
        "key": list(range(10)),
        "text": [
            "good clear strong", "good useful", "clear strong helpful",
            "very good", "good excellent", "bad weak unclear",
            "bad confusing", "weak terrible", "bad hard", "unclear weak",
        ],
    })
    project = GeometricCoder.create(
        project_dir=tmp_path / "oof.geco", data=data, keys=["key"], text="text",
        metadata=[], geometries={"lexical": CountGeometry(lemmatize=False, weighting="tfidf")},
    )
    code_id = project.create_code("quality")
    geometry_id = int(project.geometries()[0]["geometry_id"])
    first_id = project.create_classifier_spec(
        code_id=code_id, name="scaled member", geometry_id=geometry_id,
        algorithm="logistic_l2", hyperparameters={"feature_scaling": "fixed"},
    )
    second_id = project.create_classifier_spec(
        code_id=code_id, name="tree member", geometry_id=geometry_id,
        algorithm="decision_tree", hyperparameters={"max_depth": None},
    )
    rows = project.database.list_observations()
    for i in range(5):
        project.annotate(int(rows[i]["observation_id"]), code_id, "positive")
    for i in range(5, 10):
        project.annotate(int(rows[i]["observation_id"]), code_id, "negative")
    project.train_classifiers(code_id=code_id, classifier_spec_ids=[first_id, second_id])
    committee_id = project.create_classifier_committee(
        code_id=code_id, name="cross fitted", classifier_spec_ids=[first_id, second_id],
        aggregation="logistic_stack",
    )
    return project, code_id, committee_id


def test_committee_oof_meta_inputs_not_in_sample_and_reopen_export(tmp_path, monkeypatch):
    project, code_id, committee_id = _project(tmp_path)
    recorded = []
    original = stacking.fit_classifier

    def record(*, training_matrix, scoring_matrix, labels, **kwargs):
        recorded.append(np.asarray(scoring_matrix.toarray() if sparse.issparse(scoring_matrix)
                                   else scoring_matrix).copy())
        return original(training_matrix=training_matrix, scoring_matrix=scoring_matrix,
                        labels=labels, **kwargs)

    monkeypatch.setattr(stacking, "fit_classifier", record)
    fit = project.train_classifier_committee(code_id=code_id, committee_id=committee_id)
    assert len(recorded) == 10  # five shared folds for each of two members
    scheme = fit["training_snapshot"]["stacking_scheme"]
    assert scheme["method"] == "stratified_oof"
    assert scheme["folds"] == 5
    assert scheme["seed"] == 0
    assert len(scheme["training_observation_ids"]) == 10
    reused = project.train_classifier_committee(code_id=code_id, committee_id=committee_id)
    assert reused["committee_fit_id"] == fit["committee_fit_id"]
    assert len(recorded) == 10

    ref = next(ref for ref in project.predictors(code_id)
               if ref.kind == "committee" and ref.id == committee_id)
    exported = project.export_predictor(ref)
    member_ids = [int(x) for x in fit["member_fit_ids"]]
    maps = project.database.classifier_score_maps(member_ids)
    rows = project.database.list_observations()
    in_sample = np.column_stack([
        [maps[fit_id][int(row["observation_id"])] for row in rows]
        for fit_id in member_ids
    ])
    expected = exported.stacker.predict_proba(in_sample)[:, 1]
    np.testing.assert_allclose(
        exported.predict_proba([project.geometry_matrix(exported.sources[0].geometry_id)]),
        expected,
    )
    reopened = GeometricCoder.open(project.project_dir)
    assert reopened.classifier_committee_status(
        code_id=code_id, committee_id=committee_id
    )["status"] == "current"
    again = reopened.export_predictor(ref)
    np.testing.assert_allclose(
        exported.predict_proba([project.geometry_matrix(exported.sources[0].geometry_id)]),
        again.predict_proba([reopened.geometry_matrix(again.sources[0].geometry_id)]),
    )


def test_logistic_stacker_is_fitted_on_oof_not_cached_in_sample_scores(
    tmp_path, monkeypatch,
):
    import geometric_coder.project as project_module

    project, code_id, committee_id = _project(tmp_path)
    members = project.database.get_classifier_committee(committee_id)["members"]
    member_ids = [
        int(project.database.latest_classifier_fit(
            code_id=code_id, classifier_spec_id=int(member["classifier_spec_id"])
        )["classifier_fit_id"])
        for member in members
    ]
    score_maps = project.database.classifier_score_maps(member_ids)
    labeled = project._eligible_training_annotations(
        code_id, project.database.current_annotations(code_id)
    )
    ids = [int(row["observation_id"]) for row in labeled]
    cached = np.column_stack([
        [score_maps[fit_id][observation_id] for observation_id in ids]
        for fit_id in member_ids
    ])
    captured = []
    original = project_module.fit_classifier

    def watch_stacker(*, algorithm, hyperparameters, training_matrix, labels,
                      scoring_matrix):
        if algorithm == "logistic_l2" and training_matrix.shape[1] == 2:
            captured.append(np.asarray(training_matrix).copy())
        return original(
            algorithm=algorithm, hyperparameters=hyperparameters,
            training_matrix=training_matrix, labels=labels,
            scoring_matrix=scoring_matrix,
        )

    monkeypatch.setattr(project_module, "fit_classifier", watch_stacker)
    project.train_classifier_committee(code_id=code_id, committee_id=committee_id)
    assert len(captured) == 1
    assert captured[0].shape == cached.shape
    assert np.max(np.abs(captured[0] - cached)) > 1e-6


def test_legacy_in_sample_stacker_becomes_stale(tmp_path):
    project, code_id, committee_id = _project(tmp_path)
    fit = project.train_classifier_committee(code_id=code_id, committee_id=committee_id)
    legacy = dict(fit["training_snapshot"])
    legacy.pop("stacking_scheme")
    with project.database.connect() as connection:
        connection.execute(
            "UPDATE classifier_committee_fits SET training_snapshot_json = ? "
            "WHERE committee_fit_id = ?",
            (json.dumps(legacy), int(fit["committee_fit_id"])),
        )
    assert project.classifier_committee_status(
        code_id=code_id, committee_id=committee_id,
    )["status"] == "stale"
    retrained = project.train_classifier_committee(code_id=code_id, committee_id=committee_id)
    assert retrained["committee_fit_id"] != fit["committee_fit_id"]
    assert retrained["training_snapshot"]["stacking_scheme"]["method"] == "stratified_oof"
