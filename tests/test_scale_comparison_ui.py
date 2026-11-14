"""End-to-end callback contracts for manual human Scale comparisons."""
from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

from geometric_coder import GeometricCoder
from geometric_coder.ui.scale_comparisons import (
    _prepare_scale_task, _restore_scale_task, _scale_task_snapshot,
)

dash = pytest.importorskip("dash")
from dash import no_update
from dash._callback_context import context_value
from dash._utils import AttributeDict


def _project(tmp_path: Path, *, count: int = 80) -> GeometricCoder:
    project = GeometricCoder.create(
        project_dir=tmp_path / "pair-ui.geco",
        data=pd.DataFrame({
            "key": list(range(count)),
            "text": [f"observation {i} " + "context " * 4 for i in range(count)],
            "group": ["alpha" if i < count // 2 else "beta" for i in range(count)],
        }),
        keys=["key"], text="text", metadata=["group"], geometries=None,
    )
    return project


def _callback(app, name: str):
    for record in app.callback_map.values():
        if "callback" not in record:
            continue
        fn = inspect.unwrap(record["callback"])
        if fn.__name__ == name:
            return fn
    raise AssertionError(f"Missing callback: {name}")


def _invoke(callback, trigger: str, *args, **kwargs):
    token = context_value.set(AttributeDict(
        triggered_inputs=[{"prop_id": f"{trigger}.n_clicks", "value": 1}]
    ))
    try:
        return callback(*args, **kwargs)
    finally:
        context_value.reset(token)


def _manage(
    callback, trigger: str, *, scale: int, session: int,
    task: dict | None = None, rows: list | None = None, selection: list | None = None,
    history: list | None = None, history_selection: list | None = None,
    repeat_ack: list | None = None, reviewer: str | None = None,
    focal: int | None = None,
):
    return _invoke(
        callback, trigger, scale, session, *([0] * 11),
        rows or [], selection or [], task, reviewer, repeat_ack or [],
        history or [], history_selection or [], focal,
    )


def test_manual_comparison_with_correction_and_independent_repeats(tmp_path: Path, monkeypatch) -> None:
    from geometric_coder.ui.app import create_app

    project = _project(tmp_path)
    scale = project.create_scale("Argument quality", "Operational definition")
    assert project.codes() == []
    session = int(project.sessions()[0]["session_id"])
    app = create_app(project)
    browse = _callback(app, "browse_scale_units")
    manage = _callback(app, "manage_scale_comparison")
    display = _callback(app, "display_scale_task")
    listing = _callback(app, "refresh_scale_history")

    def forbidden(*args, **kwargs):
        raise AssertionError("Scale judgment attempted classifier fitting")
    monkeypatch.setattr(project, "train_classifier", forbidden)
    monkeypatch.setattr(project, "train_classifiers", forbidden)

    rows, selected_rows, status = browse(scale, 1, "focus")
    assert len(rows) == 20 and selected_rows == []
    assert "Page 1" in status
    a, b = rows[2]["observation_id"], rows[7]["observation_id"]
    task, _, _ = _manage(manage, "scale-select-first", scale=scale, session=session, rows=rows, selection=[2])
    task, _, _ = _manage(manage, "scale-select-second", scale=scale, session=session, task=task, rows=rows, selection=[7])
    task, _, _ = _manage(manage, "scale-prepare", scale=scale, session=session, task=task, reviewer="researcher")
    assert task["status"] == "ready"
    assert {task["first"], task["second"]} == {a, b}
    assert task["first"] != task["second"]
    assert task["snapshot"]["version_number"] == 1
    assert len(task["submission_key"]) >= 16
    selection, first_display, second_display, warning, note, *disabled = display(task, [], scale, 0)
    assert "First selection" in selection
    assert first_display and second_display
    assert not any(disabled)
    assert "No previous" in note
    assert not warning

    judged, message, _ = _manage(manage, "scale-judge-first", scale=scale, session=session, task=task)
    assert judged["status"] == "saved" and "Saved event" in message
    saved = project.scale_comparisons(scale)
    assert len(saved) == 1
    assert saved[0]["presented_first_id"] == task["first"]
    assert saved[0]["presented_second_id"] == task["second"]
    assert saved[0]["origin"] == "human_direct"
    assert saved[0]["reviewer_key"] == "researcher"
    assert saved[0]["outcome"] == "first_higher"
    assert saved[0]["submission_key"] == task["submission_key"]
    assert saved[0]["selection_provenance"]["mode"] == "independent"
    assert saved[0]["session_id"] == session
    retried, _, _ = _manage(manage, "scale-judge-first", scale=scale, session=session, task=judged)
    assert retried["event_id"] == judged["event_id"]
    assert len(project.scale_comparisons(scale)) == 1

    history, _, _ = listing(scale, judged, "focus")
    assert len(history) == 1
    correction, _, _ = _manage(
        manage, "scale-correct", scale=scale, session=session, task=judged,
        history=history, history_selection=[0], reviewer="researcher",
    )
    assert correction["mode"] == "correction"
    assert correction["first"] == task["first"]
    assert correction["second"] == task["second"]
    corrected, _, _ = _manage(
        manage, "scale-judge-parity", scale=scale, session=session, task=correction,
    )
    assert corrected["status"] == "saved"
    history_rows = project.scale_comparisons(scale)
    assert len(history_rows) == 2
    assert history_rows[1]["supersedes_event_id"] == history_rows[0]["comparison_event_id"]
    assert history_rows[1]["outcome"] == "parity"
    assert len(project.scale_comparisons(scale, effective_only=True)) == 1
    history, _, _ = listing(scale, corrected, "focus")
    assert history[0]["state"] == "Effective"
    assert history[1]["state"] == "Superseded"

    repeated, _, _ = _manage(
        manage, "scale-new-repeat", scale=scale, session=session, task=corrected,
    )
    assert repeated["mode"] == "independent"
    _, _, _, _, prior, *disabled = display(repeated, [], scale, 0)
    assert "Confirm an independent repeat" in prior
    assert all(disabled)
    _, rejection, _ = _manage(
        manage, "scale-judge-second", scale=scale, session=session, task=repeated,
    )
    assert "Confirm" in rejection
    repeated_saved, _, _ = _manage(
        manage, "scale-judge-second", scale=scale, session=session, task=repeated,
        repeat_ack=["repeat"],
    )
    assert repeated_saved["status"] == "saved"
    assert len(project.scale_comparisons(scale, effective_only=True)) == 2
    assert project.scale_comparisons(scale)[-1]["supersedes_event_id"] is None

    next_repeat, _, _ = _manage(
        manage, "scale-new-repeat", scale=scale, session=session, task=repeated_saved,
    )
    unsure, _, _ = _manage(
        manage, "scale-judge-unsure", scale=scale, session=session, task=next_repeat,
        repeat_ack=["repeat"],
    )
    assert unsure["status"] == "saved"
    assert project.scale_comparisons(scale)[-1]["outcome"] == "unsure"
    assert len(project.scale_training_comparisons(scale)) == 2


def test_eligibility_self_pair_and_stale_definition(tmp_path: Path) -> None:
    from geometric_coder.ui.app import create_app

    project = _project(tmp_path)
    scale = project.create_scale(
        "Strength", "v1", eligibility={
            "kind": "filtered",
            "metadata_clauses": [{
                "field": "group", "type": "categorical",
                "operator": "in", "value": ["alpha"],
            }],
        },
    )
    session = int(project.sessions()[0]["session_id"])
    app = create_app(project)
    browse = _callback(app, "browse_scale_units")
    manage = _callback(app, "manage_scale_comparison")
    page1, _, _ = browse(scale, 1, "focus")
    page4, _, _ = browse(scale, 4, "focus")
    assert all(x["eligibility"] == "Eligible" for x in page1)
    assert all(x["eligibility"] == "Excluded" for x in page4)
    initial = _restore_scale_task(project, scale, session, None)
    initial["picked_a"] = initial["picked_b"] = page1[0]["observation_id"]
    with pytest.raises(ValueError, match="different"):
        _prepare_scale_task(project, initial, None)
    initial["picked_b"] = page4[0]["observation_id"]
    with pytest.raises(ValueError, match="eligib"):
        _prepare_scale_task(project, initial, None)
    initial["picked_b"] = page1[1]["observation_id"]
    task = _prepare_scale_task(project, initial, None)
    project.update_scale(scale, "Strength revised", "v2")
    _, status, _ = _manage(
        manage, "scale-judge-first", scale=scale, session=session, task=task,
    )
    assert "definition or eligibility changed" in status
    assert project.scale_comparisons(scale) == []
    assert all(display is True for display in _callback(
        app, "display_scale_task"
    )(task, [], scale, 1)[-4:])


def test_session_restoration_keeps_orientation_and_saved_pair(tmp_path: Path) -> None:
    from geometric_coder.ui.app import create_app

    project = _project(tmp_path)
    scale = project.create_scale("Clearness")
    session = int(project.sessions()[0]["session_id"])
    other_session = int(project.create_session("Other"))
    app = create_app(project)
    manage = _callback(app, "manage_scale_comparison")
    rows, _, _ = _callback(app, "browse_scale_units")(scale, 1, "focus")
    task, _, _ = _manage(manage, "scale-select-first", scale=scale, session=session, rows=rows, selection=[1])
    task, _, _ = _manage(manage, "scale-select-second", scale=scale, session=session, task=task, rows=rows, selection=[2])
    task, _, _ = _manage(manage, "scale-prepare", scale=scale, session=session, task=task)
    snapshot = (task["first"], task["second"], task["submission_key"])
    assert project.database.get_session(session)["state"]["scale_comparison_task"] == task

    for active in (other_session, session):
        token = context_value.set(AttributeDict(
            triggered_inputs=[{"prop_id": "active-session-store.data", "value": active}]
        ))
        try:
            reopened, _, _ = manage(
                scale, active, *([0] * 11), [], [], None, None, [], [], [], None,
            )
        finally:
            context_value.reset(token)
        if active == other_session:
            assert reopened["status"] == "selecting"
        else:
            assert (reopened["first"], reopened["second"], reopened["submission_key"]) == snapshot


def test_history_labels_nonhuman_and_earlier_definitions(tmp_path: Path) -> None:
    from geometric_coder.ui.app import create_app

    project = _project(tmp_path)
    scale = project.create_scale("Measure")
    first = project.observation_id_for_unit(1)
    second = project.observation_id_for_unit(2)
    machine = project.record_scale_comparison(
        scale, first, second, "first_higher", origin="model_apply",
    )
    project.update_scale(scale, "Measure", "refined")
    project.record_scale_comparison(
        scale, second, first, "unsure", origin="human_direct",
    )
    app = create_app(project)
    history, _, _ = _callback(app, "refresh_scale_history")(scale, None, "focus")
    assert len(history) == 2
    assert any(row["origin"].startswith("Nonhuman") for row in history)
    assert {int(row["version"]) for row in history} == {1, 2}
    manage = _callback(app, "manage_scale_comparison")
    _, status, _ = _manage(
        manage, "scale-correct", scale=scale,
        session=int(project.sessions()[0]["session_id"]),
        history=history, history_selection=[1],
    )
    assert "earlier Scale definition" in status
    assert machine in {row["comparison_event_id"] for row in project.scale_comparisons(scale)}


def test_large_project_page_is_bounded(tmp_path: Path, monkeypatch) -> None:
    from geometric_coder.ui.app import create_app

    project = _project(tmp_path, count=10000)
    scale = project.create_scale("Quality")
    app = create_app(project)

    def forbidden(*args, **kwargs):
        raise AssertionError("Comparison browser materialized corpus")
    monkeypatch.setattr(project, "units", forbidden)
    monkeypatch.setattr(project.database, "list_observations", forbidden)
    monkeypatch.setattr(project.database, "iter_unit_texts", forbidden)
    monkeypatch.setattr(project, "geometry_matrix", forbidden)

    rows, _, _ = _callback(app, "browse_scale_units")(scale, 500, "focus")
    assert len(rows) == 20
    assert rows[-1]["unit_id"] == 10000
    assert rows[0]["eligibility"] == "Eligible"
