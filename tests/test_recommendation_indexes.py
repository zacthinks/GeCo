"""Regression tests for frozen-score, indexed active-learning retrieval."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from geometric_coder import CountGeometry, GeometricCoder


def _trained_project(tmp_path: Path):
    data = pd.DataFrame(
        {
            "row_id": range(8),
            "text": [
                "positive delightful wonderful",
                "positive great excellent",
                "positive helpful good",
                "negative awful poor",
                "negative terrible disappointing",
                "negative bad difficult",
                "positive kind clear",
                "negative wrong weak",
            ],
        }
    )
    project = GeometricCoder.create(
        project_dir=tmp_path / "indexed.geco",
        data=data,
        keys=["row_id"],
        text="text",
        metadata=[],
        geometries={"lexical": CountGeometry(lemmatize=False, weighting="tfidf")},
    )
    code_id = project.create_code("quality")
    obs = project.database.list_observations()
    for idx in (0, 1):
        project.annotate(int(obs[idx]["observation_id"]), code_id, "positive")
    for idx in (3, 4):
        project.annotate(int(obs[idx]["observation_id"]), code_id, "negative")
    spec_id = int(project.classifier_specs(code_id=code_id)[0]["classifier_spec_id"])
    fit = project.train_classifier(code_id=code_id, classifier_spec_id=spec_id)
    session = project.create_session("Develop")
    return project, code_id, spec_id, int(fit["classifier_fit_id"]), session


@pytest.mark.parametrize(
    "strategy,reverse",
    [
        ("likely_positive", True),
        ("likely_negative", False),
        ("classifier_uncertainty", None),
    ],
)
def test_indexed_recommendation_matches_frozen_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, strategy: str, reverse: bool | None
) -> None:
    project, code_id, spec_id, fit_id, session = _trained_project(tmp_path)
    scores = project.database.classifier_predictions(fit_id, atomic_only=True)
    annotated = {
        int(row["observation_id"])
        for row in project.database.current_annotations(code_id)
        if row["unit_id"] is not None
    }
    remaining = [
        row for row in scores if int(row["observation_id"]) not in annotated
    ]
    if reverse is None:
        expected = sorted(
            remaining,
            key=lambda row: (-float(row["uncertainty"]), int(row["observation_id"])),
        )
    else:
        expected = sorted(
            remaining,
            key=lambda row: (
                -float(row["probability"]) if reverse else float(row["probability"]),
                int(row["observation_id"]),
            ),
        )

    def full_scan_forbidden(*args, **kwargs):
        raise AssertionError("A full prediction score vector was loaded on a button click")

    monkeypatch.setattr(project.database, "classifier_atomic_score_vectors", full_scan_forbidden)
    monkeypatch.setattr(project, "_latest_classifier_scores", full_scan_forbidden)
    for row in expected:
        result = project.focus_recommendation(
            code_id=code_id,
            session_id=session,
            strategy=strategy,
            active_classifier_spec_id=spec_id,
            auto_retrain=False,
        )
        assert result is not None
        assert int(result["unit_id"]) == int(row["unit_id"])
        project.record_visit(session, int(result["unit_id"]), method="test")
    assert project.focus_recommendation(
        code_id=code_id,
        session_id=session,
        strategy=strategy,
        active_classifier_spec_id=spec_id,
        auto_retrain=False,
    ) is None


def test_cached_scores_remain_usable_after_new_label_without_retrain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, code_id, spec_id, fit_id, session = _trained_project(tmp_path)
    first = project.focus_recommendation(
        code_id=code_id, session_id=session, strategy="likely_positive",
        active_classifier_spec_id=spec_id, auto_retrain=False,
    )
    assert first is not None
    project.annotate(
        int(project.observation_id_for_unit(int(first["unit_id"]))),
        code_id,
        "positive",
    )

    def fail_fit(*args, **kwargs):
        raise AssertionError("Recommendation must not retrain a stale classifier")

    monkeypatch.setattr(project, "train_classifier", fail_fit)
    second = project.focus_recommendation(
        code_id=code_id, session_id=session, strategy="likely_positive",
        active_classifier_spec_id=spec_id, auto_retrain=False,
    )
    assert second is not None
    assert second["unit_id"] != first["unit_id"]
    assert second["stale_classifiers"]
    assert int(project.database.latest_classifier_fit(
        code_id=code_id, classifier_spec_id=spec_id
    )["classifier_fit_id"]) == fit_id


def test_ranked_index_only_scores_its_requested_fit(tmp_path: Path) -> None:
    project, code_id, spec_id, fit_id, session = _trained_project(tmp_path)
    plan = []
    with project.database.connect() as connection:
        plan = [
            str(row["detail"])
            for row in connection.execute(
                """EXPLAIN QUERY PLAN
                SELECT cp.observation_id FROM classifier_predictions AS cp
                WHERE cp.classifier_fit_id = ?
                  AND cp.probability IS NOT NULL
                ORDER BY cp.probability DESC, cp.observation_id DESC
                LIMIT 1""",
                (fit_id,),
            )
        ]
    assert any("classifier_predictions_" in part for part in plan)
    result = project.database.ranked_classifier_recommendation(
        classifier_fit_id=fit_id, code_id=code_id,
        session_id=session, strategy="classifier_uncertainty",
    )
    assert result is not None
    assert 0 <= result["probability"] <= 1


def test_existing_project_installs_indexes_on_reopen(tmp_path: Path) -> None:
    project, code_id, spec_id, fit_id, session = _trained_project(tmp_path)
    path = project.project_dir
    with project.database.connect() as connection:
        connection.execute("DROP INDEX classifier_predictions_probability_rank_idx")
        connection.execute("DROP INDEX classifier_predictions_probability_desc_rank_idx")
        connection.execute("DROP INDEX classifier_predictions_uncertainty_rank_idx")
    reopened = GeometricCoder.open(path)
    with reopened.database.connect() as connection:
        names = {
            str(row["name"])
            for row in connection.execute("PRAGMA index_list('classifier_predictions')")
        }
    assert {
        "classifier_predictions_probability_rank_idx",
        "classifier_predictions_probability_desc_rank_idx",
        "classifier_predictions_uncertainty_rank_idx",
    }.issubset(names)
    assert reopened.database.ranked_classifier_recommendation(
        classifier_fit_id=fit_id, code_id=code_id, session_id=session,
        strategy="likely_positive",
    ) is not None


def test_new_ingestion_requires_prediction_completion_before_selection(tmp_path: Path) -> None:
    project, code_id, spec_id, fit_id, session = _trained_project(tmp_path)
    project.ingest(pd.DataFrame({"row_id": [8], "text": ["another helpful document"]}))
    with pytest.raises(ValueError, match="missing predictions"):
        project.focus_recommendation(
            code_id=code_id, session_id=session,
            strategy="likely_positive", active_classifier_spec_id=spec_id,
            auto_retrain=False,
        )
    project.train_classifier(code_id=code_id, classifier_spec_id=spec_id)
    assert project.focus_recommendation(
        code_id=code_id, session_id=session,
        strategy="likely_positive", active_classifier_spec_id=spec_id,
        auto_retrain=False,
    ) is not None


def test_repeated_recommendations_skip_corpus_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, code_id, spec_id, fit_id, session = _trained_project(tmp_path)
    first = project.focus_recommendation(
        code_id=code_id, session_id=session,
        strategy="classifier_uncertainty", active_classifier_spec_id=spec_id,
    )
    assert first is not None

    def forbidden(*args, **kwargs):
        raise AssertionError("Repeated ready-snapshot retrieval loaded full corpus")

    monkeypatch.setattr(project, "units", forbidden)
    monkeypatch.setattr(project.database, "current_annotations", forbidden)
    monkeypatch.setattr(project.database, "seen_unit_ids", forbidden)
    monkeypatch.setattr(project.database, "missing_classifier_prediction_observation_ids", forbidden)
    monkeypatch.setattr(project.database, "classifier_atomic_score_vectors", forbidden)
    monkeypatch.setattr(project, "_latest_classifier_scores", forbidden)
    project.record_visit(session, int(first["unit_id"]), method="test")
    result = project.focus_recommendation(
        code_id=code_id, session_id=session,
        strategy="classifier_uncertainty", active_classifier_spec_id=spec_id,
        auto_retrain=False,
    )
    assert result is not None
    assert result["unit_id"] != first["unit_id"]


def test_training_coverage_is_updated_with_predictions(tmp_path: Path) -> None:
    project, code_id, spec_id, fit_id, session = _trained_project(tmp_path)
    assert not project.database.has_missing_atomic_classifier_predictions(fit_id)
    project.ingest(pd.DataFrame({"row_id": [8], "text": ["another useful document"]}))
    assert project.database.has_missing_atomic_classifier_predictions(fit_id)
    project.train_classifier(code_id=code_id, classifier_spec_id=spec_id)
    assert not project.database.has_missing_atomic_classifier_predictions(fit_id)


@pytest.mark.parametrize("strategy", ["likely_positive", "likely_negative", "classifier_uncertainty"])
@pytest.mark.parametrize("auto_retrain", [False, True])
@pytest.mark.parametrize("label_count", [0, 1])
def test_bootstrap_random_fallback_before_two_classes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    strategy: str, auto_retrain: bool, label_count: int,
) -> None:
    data = pd.DataFrame({
        "row_id": list(range(4)),
        "text": ["good clear", "bad unclear", "helpful text", "difficult text"],
    })
    project = GeometricCoder.create(
        project_dir=tmp_path / "bootstrap.geco", data=data,
        keys=["row_id"], text="text", metadata=[],
        geometries={"lexical": CountGeometry(lemmatize=False, weighting="tfidf")},
    )
    code_id = project.create_code("quality")
    spec_id = int(project.classifier_specs(code_id=code_id)[0]["classifier_spec_id"])
    if label_count:
        observation = project.database.list_observations()[0]
        project.annotate(int(observation["observation_id"]), code_id, "positive")
    session = project.create_session("Develop")

    def train_forbidden(*args, **kwargs):
        raise AssertionError("Bootstrap recommendations cannot attempt fitting")

    monkeypatch.setattr(project, "train_classifier", train_forbidden)
    monkeypatch.setattr(project, "train_classifiers", train_forbidden)
    result = project.focus_recommendation(
        code_id=code_id, session_id=session, strategy=strategy,
        active_classifier_spec_id=spec_id, auto_retrain=auto_retrain, random_seed=8,
    )
    assert result is not None
    assert result["strategy"] == "random"
    assert result["fallback"] is True
    assert result["probabilities"] == {}
    project.record_visit(session, int(result["unit_id"]), method="test")
    second = project.focus_recommendation(
        code_id=code_id, session_id=session, strategy=strategy,
        active_classifier_spec_id=spec_id, auto_retrain=auto_retrain, random_seed=8,
    )
    assert second is not None
    assert second["unit_id"] != result["unit_id"]


def test_out_of_order_apply_retractions_advance_freshness_revision(tmp_path: Path) -> None:
    project, code_id, spec_id, fit_id, _session = _trained_project(tmp_path)
    db = project.database
    before = db.recommendation_revision(code_id)
    with db.connect() as connection:
        run = connection.execute(
            """INSERT INTO apply_runs(
                code_id, aggregation, source_kind, source_id, threshold,
                eligibility, status, created_at
            ) VALUES (?, 'mean', 'classifier', ?, 0.5, 'unreviewed', 'draft', ?)""",
            (code_id, spec_id, "2026-10-11T00:00:00+00:00"),
        )
        run_id = int(run.lastrowid)
        op_ids = []
        for _ in range(2):
            operation = connection.execute(
                """INSERT INTO apply_commit_operations(apply_run_id, scope, created_at)
                VALUES (?, 'explicit', ?)""",
                (run_id, "2026-10-11T00:00:00+00:00"),
            )
            op_ids.append(int(operation.lastrowid))
        event_ids = [
            int(row["annotation_event_id"])
            for row in connection.execute(
                "SELECT annotation_event_id FROM annotation_events WHERE code_id = ? "
                "ORDER BY annotation_event_id LIMIT 2",
                (code_id,),
            )
        ]
        connection.execute(
            """INSERT INTO annotation_event_retractions(
                annotation_event_id, commit_operation_id, retracted_at
            ) VALUES (?, ?, ?)""",
            (event_ids[0], op_ids[1], "2026-10-11T00:00:01+00:00"),
        )
    middle = db.recommendation_revision(code_id)
    assert middle != before
    with db.connect() as connection:
        connection.execute(
            """INSERT INTO annotation_event_retractions(
                annotation_event_id, commit_operation_id, retracted_at
            ) VALUES (?, ?, ?)""",
            (event_ids[1], op_ids[0], "2026-10-11T00:00:02+00:00"),
        )
    assert db.recommendation_revision(code_id) != middle
