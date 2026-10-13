"""Phase 1E: live eligibility, ranking and negative-work closure checks.

These tests verify behavior, never hard-code a CI machine's timing thresholds.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from geometric_coder import CountGeometry, GeometricCoder


def _trained(tmp_path: Path):
    data = pd.DataFrame({
        "row_id": list(range(12)),
        "text": [
            "bright useful clear", "helpful excellent clear", "excellent kind",
            "wonderful clear", "helpful informative", "good clear",
            "awful confusing", "bad weak unclear", "terrible disappointing",
            "poor misleading", "difficult unclear", "weak bad",
        ],
    })
    project = GeometricCoder.create(
        project_dir=tmp_path / "phase1e.geco",
        data=data, keys=["row_id"], text="text", metadata=[],
        geometries={"lexical": CountGeometry(lemmatize=False, weighting="tfidf")},
    )
    code = project.create_code("quality")
    observations = project.database.list_observations()
    for idx in (0, 1):
        project.annotate(int(observations[idx]["observation_id"]), code, "positive")
    for idx in (6, 7):
        project.annotate(int(observations[idx]["observation_id"]), code, "negative")
    spec = int(project.classifier_specs(code_id=code)[0]["classifier_spec_id"])
    fit = project.train_classifier(code_id=code, classifier_spec_id=spec)
    return project, code, spec, int(fit["classifier_fit_id"])


@pytest.mark.parametrize("strategy", [
    "likely_positive", "likely_negative", "classifier_uncertainty",
])
def test_indexed_scores_match_independent_oracle_across_visits_and_sessions(
    tmp_path: Path, strategy: str,
) -> None:
    project, code, spec, fit = _trained(tmp_path)
    a = project.create_session("visited")
    b = project.create_session("independent")
    rows = project.database.classifier_predictions(fit, atomic_only=True)
    reviewed = {
        int(row["observation_id"])
        for row in project.current_annotations(code)
        if row["unit_id"] is not None
    }
    eligible = [
        row for row in rows if int(row["observation_id"]) not in reviewed
    ]
    if strategy == "likely_positive":
        key = lambda r: (-float(r["probability"]), int(r["observation_id"]))
    elif strategy == "likely_negative":
        key = lambda r: (float(r["probability"]), int(r["observation_id"]))
    else:
        key = lambda r: (-float(r["uncertainty"]), int(r["observation_id"]))
    ordered = sorted(eligible, key=key)
    assert len(ordered) > 2

    for expected in ordered[:3]:
        result = project.focus_recommendation(
            code_id=code, session_id=a, strategy=strategy,
            active_classifier_spec_id=spec, auto_retrain=False,
        )
        assert result is not None
        assert not result["fallback"]
        assert result["unit_id"] == int(expected["unit_id"])
        assert fit in result["classifier_fit_ids"].values()
        project.record_visit(a, int(result["unit_id"]), method="phase1e")

    # Session B has not visited anything and must see the first candidate.
    independent = project.focus_recommendation(
        code_id=code, session_id=b, strategy=strategy,
        active_classifier_spec_id=spec, auto_retrain=False,
    )
    assert independent is not None
    assert independent["unit_id"] == int(ordered[0]["unit_id"])

    # A new saved coding judgment invalidates eligibility for both sessions,
    # without refitting or overwriting either session's frozen score snapshot.
    newly_coded = ordered[3]
    project.annotate(int(newly_coded["observation_id"]), code, "unsure")
    next_a = project.focus_recommendation(
        code_id=code, session_id=a, strategy=strategy,
        active_classifier_spec_id=spec, auto_retrain=False,
    )
    assert next_a is not None
    assert next_a["unit_id"] == int(ordered[4]["unit_id"])
    assert next_a["stale_classifiers"]
    assert int(project.database.latest_classifier_fit(
        code_id=code, classifier_spec_id=spec
    )["classifier_fit_id"]) == fit


@pytest.mark.parametrize("order", [
    "cp.probability DESC, cp.observation_id ASC",
    "cp.probability ASC, cp.observation_id ASC",
    "cp.uncertainty DESC, cp.observation_id ASC",
])
def test_actual_classifier_eligibility_sql_uses_ranked_index(
    tmp_path: Path, order: str,
) -> None:
    project, code, _spec, fit = _trained(tmp_path)
    session = project.create_session("SQL plan")
    # Match the actual ranked_classifier_recommendation SELECT, JOIN, NOT
    # EXISTS filters and ORDER BY, not an unfiltered surrogate benchmark.
    sql = f"""
        EXPLAIN QUERY PLAN
        SELECT u.unit_id, cp.observation_id, cp.probability,
               cp.uncertainty, cp.decision_score
        FROM classifier_predictions AS cp
        JOIN units AS u ON u.observation_id = cp.observation_id
        WHERE cp.classifier_fit_id = ?
          AND cp.probability IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM navigation_events AS ne
              WHERE ne.session_id = ? AND ne.unit_id = u.unit_id
          )
          AND NOT EXISTS (
              SELECT 1 FROM annotation_events AS ae
              WHERE ae.observation_id = cp.observation_id
                AND ae.code_id = ?
                AND NOT EXISTS (
                    SELECT 1 FROM annotation_event_retractions AS aer
                    WHERE aer.annotation_event_id = ae.annotation_event_id
                )
          )
        ORDER BY {order}
        LIMIT 1
    """
    with project.database.connect() as conn:
        details = [str(row["detail"]) for row in conn.execute(sql, (fit, session, code))]
    assert any("classifier_predictions_" in detail for detail in details), details
    assert not any("USE TEMP B-TREE FOR ORDER BY" in detail for detail in details), details


def test_warm_recommendations_never_fit_or_load_full_score_vectors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, code, spec, _fit = _trained(tmp_path)
    session = project.create_session("warm")
    first = project.focus_recommendation(
        code_id=code, session_id=session, strategy="likely_positive",
        active_classifier_spec_id=spec, auto_retrain=False,
    )
    assert first is not None
    project.record_visit(session, int(first["unit_id"]), method="warmup")

    def forbidden(*args, **kwargs):
        raise AssertionError("Ready-snapshot click triggered forbidden heavy work")

    for obj, name in [
        (project, "train_classifier"),
        (project, "train_classifiers"),
        (project, "_classifier_scoring_matrix"),
        (project, "_latest_classifier_scores"),
        (project.database, "classifier_predictions"),
        (project.database, "classifier_atomic_score_vectors"),
        (project.database, "current_annotations"),
    ]:
        monkeypatch.setattr(obj, name, forbidden)

    for strategy in ("likely_negative", "classifier_uncertainty", "likely_positive"):
        result = project.focus_recommendation(
            code_id=code, session_id=session, strategy=strategy,
            active_classifier_spec_id=spec, auto_retrain=False,
        )
        assert result is not None
        project.record_visit(session, int(result["unit_id"]), method="warm")
