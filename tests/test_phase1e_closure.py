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

    # Unsure is a saved assignment and excludes a recommendation candidate,
    # but is *not* eligible positive/negative training evidence. It must not
    # misreport a still-current classifier as stale.
    project.annotate(int(ordered[3]["observation_id"]), code, "unsure")
    after_unsure = project.focus_recommendation(
        code_id=code, session_id=a, strategy=strategy,
        active_classifier_spec_id=spec, auto_retrain=False,
    )
    assert after_unsure is not None
    assert after_unsure["unit_id"] == int(ordered[4]["unit_id"])
    assert after_unsure["stale_classifiers"] == []

    # A later eligible Present judgment genuinely changes the training
    # snapshot, excludes that candidate and keeps the previous fit available.
    project.annotate(int(ordered[4]["observation_id"]), code, "positive")
    after_present = project.focus_recommendation(
        code_id=code, session_id=a, strategy=strategy,
        active_classifier_spec_id=spec, auto_retrain=False,
    )
    assert after_present is not None
    assert after_present["unit_id"] == int(ordered[5]["unit_id"])
    assert after_present["stale_classifiers"]
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


def test_dash_training_inbox_routes_success_reuse_and_failure_by_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the *registered* six-output callback, not just a request helper."""
    import inspect

    from dash import no_update

    from geometric_coder.ui.app import create_app

    project, code, spec, _fit = _trained(tmp_path)
    app = create_app(project)
    candidates = [
        entry["callback"].__wrapped__
        for output, entry in app.callback_map.items()
        if "focus-auto-training-status.children" in output
        and "apply-training-status.children" in output
    ]
    assert len(candidates) == 1
    dispatch = candidates[0]
    inbox = inspect.getclosurevars(dispatch).nonlocals["training_inbox"]

    def request(origin: str) -> dict:
        return {
            "code_id": code,
            "classifier_spec_ids": [spec],
            "tune": False,
            "origin": origin,
        }

    # A current model is reused and Apply receives its own saved acknowledgment.
    inbox.enqueue(request("apply"))
    reused = dispatch({"nonce": "apply"}, 0)
    assert len(reused) == 6
    assert "Apply batch saved" in reused[5]
    assert "current" in reused[5]
    assert reused[1] is no_update
    assert reused[2] is no_update

    # A new eligible Explore judgment makes the fit stale. Its request trains
    # the selected classifier and advances the model refresh exactly once.
    obs = project.database.list_observations()
    project.annotate(int(obs[2]["observation_id"]), code, "positive")
    inbox.enqueue(request("explore"))
    trained = dispatch({"nonce": "explore"}, 0)
    assert "trained 1 selected classifier" in trained[2]
    assert trained[1] == 1
    assert trained[5] is no_update

    # Training errors may not erase a saved Code Center decision, nor claim
    # a successful model refresh. The origin-specific feedback remains visible.
    def fail_fit(**kwargs):
        raise RuntimeError("simulated fitter failure")

    monkeypatch.setattr(project, "train_classifiers_after_evidence_change", fail_fit)
    inbox.enqueue(request("code_center"))
    failed = dispatch({"nonce": "error"}, 1)
    assert "Saved judgment. Automatic training failed:" in failed[3]
    assert "simulated fitter failure" in failed[3]
    assert failed[1] is no_update
    assert failed[5] is no_update


def test_develop_summary_tab_return_skips_expensive_hidden_refreshes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply drafts and Explore navigation cannot force hidden code-wide scans."""
    from dash import no_update

    from geometric_coder.ui.app import create_app

    project, code, _spec, _fit = _trained(tmp_path)
    app = create_app(project)
    summaries = [
        entry["callback"].__wrapped__
        for output, entry in app.callback_map.items()
        if "focus-code-definition.children" in output
        and "focus-label-counts.children" in output
    ]
    assert len(summaries) == 1
    summary = summaries[0]
    first = summary(code, 0, 0, "focus")
    assert "Positive 2" in str(first[1])

    def forbid(*args, **kwargs):
        raise AssertionError("Hidden Develop scanned all code assignments")

    with monkeypatch.context() as patch:
        patch.setattr(project.database, "annotation_counts", forbid)
        patch.setattr(project, "current_annotations", forbid)
        patch.setattr(project, "training_label_counts", forbid)
        for workspace in ("apply", "explore", "codes"):
            assert summary(code, 0, 0, workspace) == (
                no_update, no_update, no_update
            )

    # While the checkbox is off, Explore changes still become visible on
    # returning to Develop. This does not involve any training request.
    project.annotate(
        int(project.database.list_observations()[2]["observation_id"]),
        code, "unsure", origin="human_direct",
    )
    after = summary(code, 0, 0, "focus")
    assert "Unsure 1" in str(after[1])


@pytest.mark.parametrize("source", ["classifier", "fixed", "learned"])
def test_nearly_exhausted_session_preserves_rank_order_without_refitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str,
) -> None:
    """A heavily visited session must reach its final eligible candidate."""
    project, code, spec, fit_id = _trained(tmp_path)
    session = project.create_session("near exhaustion")
    committee_id = None

    if source != "classifier":
        observations = project.database.list_observations()
        project.annotate(int(observations[2]["observation_id"]), code, "positive")
        project.annotate(int(observations[8]["observation_id"]), code, "negative")
        fit = project.train_classifier(code_id=code, classifier_spec_id=spec)
        fit_id = int(fit["classifier_fit_id"])
        geometry_id = int(project.database.get_classifier_spec(spec)["geometry_id"])
        member_id = project.create_classifier_spec(
            code_id=code, name="phase1e member", geometry_id=geometry_id,
            algorithm="decision_tree",
        )
        project.train_classifier(code_id=code, classifier_spec_id=member_id)
        committee_id = project.create_classifier_committee(
            code_id=code, name="phase1e near-exhaustion committee",
            classifier_spec_ids=[spec, member_id],
            aggregation="mean" if source == "fixed" else "logistic_stack",
        )
        if source == "fixed":
            project.prepare_selected_committee(code_id=code, committee_id=committee_id)
        else:
            project.train_classifier_committee(code_id=code, committee_id=committee_id)

    if source == "classifier":
        ranked_rows = project.database.classifier_predictions(fit_id, atomic_only=True)
    elif source == "fixed":
        with project.database.connect() as conn:
            ranked_rows = [
                dict(row) for row in conn.execute(
                    """SELECT pcs.observation_id, u.unit_id, pcs.probability
                    FROM prepared_committee_scores AS pcs
                    JOIN units AS u ON u.observation_id = pcs.observation_id
                    WHERE pcs.committee_id = ?""",
                    (committee_id,),
                )
            ]
    else:
        fitted = project.database.latest_classifier_committee_fit(
            committee_id=int(committee_id), code_id=code
        )
        assert fitted is not None
        ranked_rows = project.database.classifier_committee_predictions(
            int(fitted["committee_fit_id"])
        )

    annotated = {
        int(row["observation_id"]) for row in project.current_annotations(code)
        if row["unit_id"] is not None
    }
    eligible = sorted(
        (
            row for row in ranked_rows
            if row["unit_id"] is not None
            and row["probability"] is not None
            and int(row["observation_id"]) not in annotated
        ),
        key=lambda row: (-float(row["probability"]), int(row["observation_id"])),
    )
    assert len(eligible) >= 3
    for row in eligible[:-1]:
        project.record_visit(session, int(row["unit_id"]), method="phase1e exhausted")

    def forbid_fit(*args, **kwargs):
        raise AssertionError("Recommendation clicked a training path")

    monkeypatch.setattr(project, "train_classifier", forbid_fit)
    monkeypatch.setattr(project, "train_classifiers", forbid_fit)
    monkeypatch.setattr(project, "train_classifier_committee", forbid_fit)

    options = dict(
        code_id=code, session_id=session, strategy="likely_positive",
        auto_retrain=False,
    )
    if source == "classifier":
        options["active_classifier_spec_id"] = spec
    else:
        options["committee_id"] = committee_id
        options["recommendation_source"] = "committee"

    last = project.focus_recommendation(**options)
    assert last is not None
    assert int(last["unit_id"]) == int(eligible[-1]["unit_id"])
    project.record_visit(session, int(last["unit_id"]), method="phase1e exhausted")
    assert project.focus_recommendation(**options) is None

    # Exhaustion is per session, not persistent global censorship.
    fresh_session = project.create_session("fresh")
    first = project.focus_recommendation(**{**options, "session_id": fresh_session})
    assert first is not None
    assert int(first["unit_id"]) == int(eligible[0]["unit_id"])
