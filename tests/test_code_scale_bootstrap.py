"""Code-to-Scale bootstrap creates ordinary, complete, transactional human comparisons."""

from __future__ import annotations

import inspect
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from geometric_coder import GeometricCoder


def _project(tmp_path: Path, count: int = 9) -> GeometricCoder:
    return GeometricCoder.create(
        project_dir=tmp_path / "code-bootstrap.geco",
        data=pd.DataFrame({
            "key": list(range(count)),
            "text": [f"example {index}" for index in range(count)],
        }),
        keys=["key"], text="text", metadata=[], geometries=None,
    )


def _labels(project: GeometricCoder, code: int) -> tuple[list[int], list[int]]:
    observations = [project.observation_id_for_unit(index) for index in range(1, 10)]
    present, absent = observations[:3], observations[3:6]
    for observation_id in present:
        project.annotate(observation_id, code, "positive", origin="human_direct")
    for observation_id in absent:
        project.annotate(observation_id, code, "negative", origin="human_direct")
    return present, absent


def _callback(app, name: str):
    for record in app.callback_map.values():
        if "callback" in record:
            function = inspect.unwrap(record["callback"])
            if function.__name__ == name:
                return function
    raise AssertionError(name)


def test_exact_cartesian_nine_normal_records_and_correction(tmp_path: Path) -> None:
    project = _project(tmp_path)
    code = project.create_code("Argument present", "Original definition")
    present, absent = _labels(project, code)
    before = project.code(code).copy()
    before_labels = project.current_annotations(code)
    preview = project.code_scale_bootstrap_preview(code)
    assert (preview["present"], preview["absent"], preview["comparisons"]) == (3, 3, 9)

    saved = project.create_scale_from_code(
        "Argument quality", "Higher means more quality.", code,
        expected_preview=preview, submission_key="a" * 32,
    )
    assert saved["comparisons_created"] == 9
    assert saved["retried"] is False
    scale = saved["scale_id"]
    events = project.scale_comparisons(scale)
    assert len(events) == 9
    assert {(row["presented_first_id"], row["presented_second_id"]) for row in events} == {
        (first, second) for first in present for second in absent
    }
    assert all(row["outcome"] == "first_higher" for row in events)
    assert all(row["origin"] == "human_direct" for row in events)
    assert all(row["selection_provenance"] == {} for row in events)
    assert all(row["supersedes_event_id"] is None for row in events)
    assert len(project.scale_training_comparisons(scale)) == 9
    assert project.scale_eligibility(scale)["eligibility"]["kind"] == "all"
    assert project.code(code) == before
    assert project.current_annotations(code) == before_labels

    correction = project.record_scale_comparison(
        scale, present[0], absent[0], "parity",
        supersedes_event_id=events[0]["comparison_event_id"],
    )
    history = project.scale_comparisons(scale)
    assert len(history) == 10
    assert history[-1]["comparison_event_id"] == correction
    assert len(project.scale_comparisons(scale, effective_only=True)) == 9
    assert len(project.scale_training_comparisons(scale)) == 9
    # Correcting a row later does not make the original creation retry look partial.
    retried = project.create_scale_from_code(
        "Argument quality", "Higher means more quality.", code,
        expected_preview=preview, submission_key="a" * 32,
    )
    assert retried["scale_id"] == scale
    assert retried["comparisons_created"] == 9
    assert retried["retried"] is True
    assert len(project.scale_comparisons(scale)) == 10


def test_filter_effective_human_annotations_and_atomic_kinds(tmp_path: Path) -> None:
    project = _project(tmp_path)
    code = project.create_code("Binary")
    ids = [project.observation_id_for_unit(index) for index in range(1, 10)]
    # Human states can be superseded, including by nonhuman or unsure labels.
    project.annotate(ids[0], code, "positive")
    project.annotate(ids[1], code, "positive")
    project.annotate(ids[2], code, "positive")
    project.annotate(ids[3], code, "negative")
    project.annotate(ids[4], code, "negative")
    project.annotate(ids[5], code, "negative")
    project.annotate(ids[6], code, "positive", origin="model_apply")
    project.annotate(ids[7], code, "unsure")
    project.annotate(ids[0], code, "negative", origin="human_direct")
    project.annotate(ids[2], code, "unsure")
    project.annotate(ids[3], code, "positive", origin="model_apply")
    with project.database.connect() as connection:
        extra = connection.execute(
            "INSERT INTO observations(kind, text, created_at) "
            "VALUES ('span', 'synthetic span', '2026-01-01')"
        ).lastrowid
    project.annotate(int(extra), code, "positive", origin="human_direct")
    preview = project.code_scale_bootstrap_preview(code)
    assert (preview["present"], preview["absent"], preview["comparisons"]) == (1, 3, 3)
    result = project.create_scale_from_code(
        "Rank", "", code, expected_preview=preview, submission_key="b" * 32,
    )
    rows = project.scale_comparisons(result["scale_id"])
    assert len(rows) == 3
    assert {row["presented_first_id"] for row in rows} == {ids[1]}
    assert {row["presented_second_id"] for row in rows} == {ids[0], ids[4], ids[5]}
    assert all(row["origin"] == "human_direct" for row in rows)


def test_retries_and_changed_previews_do_not_duplicate_or_partially_write(tmp_path: Path) -> None:
    project = _project(tmp_path)
    code = project.create_code("Binary")
    _labels(project, code)
    preview = project.code_scale_bootstrap_preview(code)
    key = "c" * 32
    first = project.create_scale_from_code(
        "Quality", "definition", code, expected_preview=preview, submission_key=key,
    )
    retried = project.create_scale_from_code(
        "Quality", "definition", code, expected_preview=preview, submission_key=key,
    )
    assert retried == {**first, "retried": True}
    assert len(project.constructs()) == 2
    assert len(project.scale_comparisons(first["scale_id"])) == 9
    with pytest.raises(ValueError, match="different Scale details"):
        project.create_scale_from_code(
            "Other quality", "definition", code,
            expected_preview=preview, submission_key=key,
        )

    # A stale preflight must fail before inserting a Scale or any comparisons.
    changed = project.create_code("Other Code")
    project.annotate(project.observation_id_for_unit(7), code, "positive")
    with pytest.raises(ValueError, match="changed after preview"):
        project.create_scale_from_code(
            "Stale", "", code, expected_preview=preview,
            submission_key="d" * 32,
        )
    assert len([row for row in project.constructs() if row["type"] == "scale"]) == 1
    assert project.code(changed)["name"] == "Other Code"


def test_forced_mid_batch_failure_rolls_back_scale_and_all_events(tmp_path: Path) -> None:
    project = _project(tmp_path)
    code = project.create_code("Binary")
    _labels(project, code)
    preview = project.code_scale_bootstrap_preview(code)
    with project.database.connect() as connection:
        connection.execute(
            """CREATE TRIGGER reject_bootstrap_midway
               BEFORE INSERT ON scale_comparison_events
               WHEN (SELECT COUNT(*) FROM scale_comparison_events) >= 4
               BEGIN SELECT RAISE(ABORT, 'forced comparison failure'); END"""
        )
    before = project.constructs()
    with pytest.raises(sqlite3.IntegrityError, match="forced comparison failure"):
        project.create_scale_from_code(
            "Must not appear", "", code, expected_preview=preview,
            submission_key="e" * 32,
        )
    assert project.constructs() == before
    with project.database.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM scale_comparison_events"
        ).fetchone()[0] == 0
        connection.execute("DROP TRIGGER reject_bootstrap_midway")
    result = project.create_scale_from_code(
        "Must not appear", "", code, expected_preview=preview,
        submission_key="e" * 32,
    )
    assert result["comparisons_created"] == 9


def test_empty_group_and_safety_limit_leave_database_untouched(tmp_path: Path) -> None:
    project = _project(tmp_path)
    code = project.create_code("Binary")
    project.annotate(project.observation_id_for_unit(1), code, "positive")
    empty = project.code_scale_bootstrap_preview(code)
    assert (empty["present"], empty["absent"], empty["comparisons"]) == (1, 0, 0)
    with pytest.raises(ValueError, match="at least one human Present"):
        project.create_scale_from_code(
            "Empty", "", code, expected_preview=empty, submission_key="f" * 32,
        )
    _labels(project, code)
    # Lower only the test's cap; do not actually insert a very large batch.
    project.database._MAX_CODE_BOOTSTRAP_COMPARISONS = 4
    too_large = project.code_scale_bootstrap_preview(code)
    assert too_large["comparisons"] == 9
    assert too_large["limit"] == 4
    with pytest.raises(ValueError, match="above the 4 safety limit"):
        project.create_scale_from_code(
            "Too large", "", code, expected_preview=too_large,
            submission_key="1" * 32,
        )
    assert project.constructs() == [row for row in project.constructs() if row["type"] == "code"]


def test_paged_bootstrap_history_exposes_and_corrects_older_events(tmp_path: Path) -> None:
    pytest.importorskip("dash")
    from dash._callback_context import context_value
    from dash._utils import AttributeDict
    from geometric_coder.ui.app import create_app

    project = _project(tmp_path, count=18)
    code = project.create_code("Binary")
    for unit_id in range(1, 10):
        project.annotate(
            project.observation_id_for_unit(unit_id), code, "positive",
        )
    for unit_id in range(10, 19):
        project.annotate(
            project.observation_id_for_unit(unit_id), code, "negative",
        )
    preview = project.code_scale_bootstrap_preview(code)
    assert preview["comparisons"] == 81
    result = project.create_scale_from_code(
        "Eighty-one comparisons", "", code, expected_preview=preview,
        submission_key="8" * 32,
    )
    scale = result["scale_id"]
    app = create_app(project)
    history_callback = _callback(app, "refresh_scale_history")
    page1, _, message1 = history_callback(scale, None, "focus", 1)
    page2, _, message2 = history_callback(scale, None, "focus", 2)
    assert len(page1) == 60
    assert len(page2) == 21
    assert "page 2 of 2" in message2
    assert "81 recorded" in message1
    assert not ({row["comparison_event_id"] for row in page1}
                & {row["comparison_event_id"] for row in page2})
    old_event = page2[-1]["comparison_event_id"]
    assert project.scale_comparison_event(scale, old_event)["is_superseded"] is False

    manage = _callback(app, "manage_scale_comparison")
    session = int(project.sessions()[0]["session_id"])
    token = context_value.set(AttributeDict(
        triggered_inputs=[{"prop_id": "scale-correct.n_clicks", "value": 1}]
    ))
    try:
        prepared, msg, _ = manage(
            scale, session, *([0] * 11),
            [], [], None, None, [], page2, [len(page2) - 1], None,
        )
    finally:
        context_value.reset(token)
    assert prepared["supersedes_event_id"] == old_event
    assert "Correcting event" in msg
    project.record_scale_comparison(
        scale, prepared["first"], prepared["second"], "parity",
        supersedes_event_id=old_event,
    )
    assert project.scale_comparison_event(scale, old_event)["is_superseded"] is True


def test_shared_dialog_bootstrap_preview_and_explicit_confirmation(tmp_path: Path) -> None:
    pytest.importorskip("dash")
    from dash._callback_context import context_value
    from dash._utils import AttributeDict
    from geometric_coder.ui.app import create_app

    project = _project(tmp_path)
    code = project.create_code("Source")
    _labels(project, code)
    app = create_app(project)
    preview_callback = _callback(app, "preview_code_bootstrap")
    modal_callback = _callback(app, "manage_code_modal")
    snapshot, explanation, confirmed = preview_callback(
        code, ["enabled"], "scale", "center",
    )
    assert "3 human Present" in explanation
    assert "3 human Absent" in explanation
    assert "9 comparisons" in explanation
    assert confirmed == []
    assert snapshot["preview"]["code_id"] == code

    token = context_value.set(AttributeDict(
        triggered_inputs=[{"prop_id": "code-modal-create-button.n_clicks", "value": 1}]
    ))
    try:
        args = [0, 0, 1, 0, 0, 1, "center", "Scale from Code", "Definition",
                0, 0, [], "scale", ["enabled"], code]
        unconfirmed = modal_callback(*args, [], snapshot, None)
        assert unconfirmed[0]["display"] == "flex"
        assert "confirm" in unconfirmed[-1].lower()
        assert [row["type"] for row in project.constructs()] == ["code"]

        created = modal_callback(*args, ["confirmed"], snapshot, None)
        assert created[0]["display"] == "none"
        scale = next(row["construct_id"] for row in project.constructs() if row["type"] == "scale")
        assert len(project.scale_comparisons(scale)) == 9
        repeated = modal_callback(*args, ["confirmed"], snapshot, None)
        assert repeated[0]["display"] == "none"
        assert len(project.scale_comparisons(scale)) == 9

        # Unchecked optional bootstrap still creates a regular independent Scale.
        ordinary = modal_callback(
            0, 0, 1, 0, 0, 1, "center", "Ordinary", "", 0, 0, [],
            "scale", [], None, [], None, None,
        )
        assert ordinary[0]["display"] == "none"
        regular = next(row["construct_id"] for row in project.constructs()
                       if row["name"] == "Ordinary")
        assert project.scale_comparisons(regular) == []
    finally:
        context_value.reset(token)
