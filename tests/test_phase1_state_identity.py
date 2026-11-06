from __future__ import annotations

from pathlib import Path

import pandas as pd

from geometric_coder import GeometricCoder
from geometric_coder.ui.app import (
    _apply_rows_match_draft,
    _matching_recommendation_fit_id,
    _prediction_geometry_matches_code,
)


def test_recommendation_fit_requires_exact_atomic_code_session_and_unit() -> None:
    recommended = {
        "unit_id": 10,
        "code_id": 5,
        "session_id": 3,
        "classifier_fit_ids": {"classifier-a": 41},
    }
    match = lambda rec, **kw: _matching_recommendation_fit_id(
        rec,
        unit_id=kw.get("unit_id", 10),
        code_id=kw.get("code_id", 5),
        session_id=kw.get("session_id", 3),
        selected_unit_ids=kw.get("selected_unit_ids", [10]),
    )
    assert match(recommended) == 41
    assert match(recommended, unit_id=11) is None
    assert match(recommended, code_id=6) is None
    assert match(recommended, session_id=4) is None
    assert match(recommended, selected_unit_ids=[9, 10]) is None
    assert match({"unit_id": 10, "classifier_fit_ids": {"a": 41}}) is None
    assert match({**recommended, "classifier_fit_ids": {"a": 41, "b": 42}}) is None
    assert match({**recommended, "classifier_fit_ids": {}}) is None
    assert match(None) is None


def test_prediction_snapshot_rejects_other_code_and_missing_identity() -> None:
    selected = {"cache_key": "abc", "code_id": 7, "classifier_fit_ids": [31]}
    assert _prediction_geometry_matches_code(selected, 7)
    assert not _prediction_geometry_matches_code(selected, 8)
    assert not _prediction_geometry_matches_code(selected, None)
    assert not _prediction_geometry_matches_code({}, 7)
    assert not _prediction_geometry_matches_code(None, 7)


def test_apply_row_identity_rejects_previous_draft_data() -> None:
    rows = [{"unit_id": 1, "apply_run_id": 10}, {"unit_id": 2, "apply_run_id": 10}]
    assert _apply_rows_match_draft(rows, 10)
    assert not _apply_rows_match_draft(rows, 11)
    assert not _apply_rows_match_draft([rows[0], {"unit_id": 3, "apply_run_id": 11}], 10)
    assert not _apply_rows_match_draft([{"unit_id": 1}], 10)
    assert _apply_rows_match_draft([], 10)
    assert _apply_rows_match_draft(None, 10)


def test_session_ui_patch_preserves_independent_focus_task(tmp_path: Path) -> None:
    project = GeometricCoder.create(
        project_dir=tmp_path / "state.geco",
        data=pd.DataFrame({"doc_id": [10, 20], "text": ["alpha", "beta"]}),
        keys=["doc_id"],
        text="text",
    )
    code_id = int(project.create_code("concept", "Example")["code_id"])
    session_id = project.configure_focus(codes=[code_id], unit_ids=[2, 1])
    focus_before = project.focus_task(session_id)
    state_before = project.database.get_session(session_id)["state"]
    assert "focus_task" in state_before

    project.patch_session_state(session_id, {
        "literal_query": "alpha",
        "explore_uncoded_only": True,
        "focus_train_all": True,
        "focus_auto_retrain": False,
        "focus_tune_hyperparameters": True,
    })
    state_after = project.database.get_session(session_id)["state"]
    assert state_after["focus_task"] == state_before["focus_task"]
    assert project.focus_task(session_id)["required_unit_ids"] == focus_before["required_unit_ids"]
    assert state_after["literal_query"] == "alpha"
    assert state_after["explore_uncoded_only"] is True
    assert state_after["focus_train_all"] is True
    assert state_after["focus_auto_retrain"] is False
    assert state_after["focus_tune_hyperparameters"] is True


def test_ui_persist_and_restore_use_symmetric_session_ownership() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "geometric_coder" / "ui" / "app.py"
    ).read_text(encoding="utf-8")
    persist = source.split("def persist_session_state(", 1)[1].split("\n    return app", 1)[0]
    restore = source.split("def restore_session(", 1)[1].split("\n    @app.callback(", 1)[0]
    assert "project.patch_session_state(" in persist
    assert "project.update_session_state(" not in persist
    for component in (
        "explore-uncoded-only",
        "focus-train-scope",
        "focus-auto-retrain",
        "focus-tune-hyperparameters",
    ):
        assert f'Output("{component}", "value")' in source
    for field in (
        "explore_uncoded_only",
        "focus_train_all",
        "focus_auto_retrain",
        "focus_tune_hyperparameters",
    ):
        assert f'state.get("{field}"' in restore
    assert 'Input("focus-code-dropdown", "value")' in source
    assert "def _prediction_geometry_matches_code(" in source


def test_ui_guards_apply_events_and_prediction_clicks_against_stale_identity() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "geometric_coder" / "ui" / "app.py"
    ).read_text(encoding="utf-8")
    assert '"apply_run_id": int(draft_id)' in source
    assert "_apply_rows_match_draft([row], int(draft_id))" in source
    assert "_apply_rows_match_draft(table_rows, int(draft_id))" in source
    render = source.split("def render_prediction_geometry(", 1)[1].split("\n    @app.callback(", 1)[0]
    assert "_prediction_geometry_matches_code(refresh_state, code_id)" in render
    click = source.split("def open_prediction_geometry_point(", 1)[1].split("\n    @app.callback(", 1)[0]
    assert "_prediction_geometry_matches_code(prediction_refresh, code_id)" in click
