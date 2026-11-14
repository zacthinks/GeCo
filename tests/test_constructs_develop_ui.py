"""Constructs Center and typed Develop selection regression tests."""

from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

from geometric_coder import GeometricCoder
from geometric_coder.ui.app import _construct_options, _selected_focus_construct_id


def _project(tmp_path: Path) -> GeometricCoder:
    return GeometricCoder.create(
        project_dir=tmp_path / "typed-ui.geco",
        data=pd.DataFrame({"key": [1, 2], "text": ["first", "second"]}),
        keys=["key"], text="text", metadata=[], geometries=None,
    )


def _callback(app, output_fragment: str):
    for output, spec in app.callback_map.items():
        if output_fragment in output:
            return inspect.unwrap(spec["callback"])
    raise AssertionError(f"Missing callback output: {output_fragment}")


def test_construct_selection_restores_old_and_new_sessions(tmp_path: Path) -> None:
    project = _project(tmp_path)
    first = project.create_code("Code")
    scale = project.create_scale("Scale")
    second = project.create_code("Another")
    rows = project.constructs()

    assert _selected_focus_construct_id(rows, {"focus_code_id": first}) == first
    assert _selected_focus_construct_id(rows, {"focus_construct_id": scale, "focus_code_id": first}) == scale
    assert _selected_focus_construct_id(rows, {"focus_construct_id": second}) == second
    assert _selected_focus_construct_id(rows, {"focus_code_id": scale}) is None
    assert _selected_focus_construct_id(rows, {"focus_construct_id": 9999}) is None
    assert [row["value"] for row in _construct_options(rows)] == [first, scale, second]
    assert [row["label"].split(" · ")[-1] for row in _construct_options(rows)] == [
        "Code", "Scale", "Code",
    ]


def test_scale_created_and_revised_without_binary_codes(tmp_path: Path) -> None:
    project = _project(tmp_path)
    scale = project.create_scale("Quality", "first definition")
    assert project.codes() == []
    assert project.construct(scale)["type"] == "scale"
    project.update_scale(scale, "Argument quality", "second definition")
    assert [row["description"] for row in project.scale_versions(scale)] == [
        "first definition", "second definition",
    ]


def test_construct_selector_callback_hides_binary_develop_for_scales(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("dash")
    from geometric_coder.ui.app import create_app

    project = _project(tmp_path)
    code = project.create_code("Code", "binary definition")
    scale = project.create_scale("Scale", "continuous definition")
    app = create_app(project)
    callback = _callback(app, "focus-scale-heading.children")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Scale route called a Code-only classifier API")

    monkeypatch.setattr(project, "classifier_specs", forbidden)
    monkeypatch.setattr(project, "classifier_status", forbidden)
    selected_scale = callback(scale)
    assert selected_scale[0] is None
    assert selected_scale[2] == {}
    assert selected_scale[3] == "Scale: Scale"
    assert "continuous definition" in selected_scale[4]
    assert selected_scale[-1] == {"display": "none"}

    selected_code = callback(code)
    assert selected_code[0] == code
    assert selected_code[1] == {}
    assert selected_code[2] == {"display": "none"}
    assert selected_code[-1] == {}

    cleared = callback(None)
    assert cleared[0] is None
    assert cleared[2] == {"display": "none"}


def test_center_lists_both_types_and_guards_binary_content(tmp_path: Path) -> None:
    pytest.importorskip("dash")
    from geometric_coder.ui.app import create_app

    project = _project(tmp_path)
    code = project.create_code("Code")
    scale = project.create_scale("Scale")
    app = create_app(project)
    filter_list = _callback(app, "code-center-table.data")
    rows, _, status = filter_list("", "full", [], 0, 0, 0, None)
    assert [(row["construct_id"], row["type"]) for row in rows] == [
        (code, "Code"), (scale, "Scale"),
    ]
    assert "2 constructs" in status
    display = _callback(app, "code-center-type-label.children")
    assert display(scale)[-1] == {"display": "none"}
    assert display(code)[-1] == {}


def test_ui_callbacks_persist_and_restore_scale_and_legacy_code(tmp_path: Path) -> None:
    pytest.importorskip("dash")
    from geometric_coder.ui.app import create_app

    project = _project(tmp_path)
    code = project.create_code("Code")
    scale = project.create_scale("Scale")
    app = create_app(project)
    callbacks = [
        inspect.unwrap(entry["callback"])
        for entry in app.callback_map.values() if "callback" in entry
    ]
    persist = next(func for func in callbacks if func.__name__ == "persist_session_state")
    restore = next(func for func in callbacks if func.__name__ == "restore_session")
    session_id = int(project.sessions()[0]["session_id"])
    fields = dict(
        session_id=session_id, geometry_id=None, view_id=None, focal_unit_id=None,
        navigation_history=None, literal_query="", literal_options=[],
        metadata_filters=[], metadata_filter_operator="and", explore_uncoded_values=[],
        context_level=0, context_window=1, page_size="5000", page_number=1,
        navigation_new_only=[], workspace="focus", focus_construct_id=scale,
        focus_code_id=code, focus_classifier_spec_id=123, focus_committee_id=456,
        focus_train_scope_values=[], focus_auto_retrain_values=[], 
        focus_tune_hyperparameter_values=[], focus_context_level=0,
        focus_context_window=1, explore_code_palette=[], presentation_search="",
    )
    persist(**fields)
    saved = project.database.get_session(session_id)["state"]
    assert saved["focus_construct_id"] == scale
    assert saved["focus_code_id"] is None
    assert saved["focus_classifier_spec_id"] is None
    assert saved["focus_committee_id"] is None
    restored = restore(session_id)
    assert restored[23] == scale
    assert restored[24] is None

    project.patch_session_state(
        session_id, {"focus_construct_id": None, "focus_code_id": code}
    )
    restored_legacy = restore(session_id)
    assert restored_legacy[23] == code
    assert restored_legacy[24] == code


def test_scale_creation_and_version_update_through_center_callbacks(tmp_path: Path) -> None:
    pytest.importorskip("dash")
    from dash._callback_context import context_value
    from dash._utils import AttributeDict
    from geometric_coder.ui.app import create_app

    project = _project(tmp_path)
    app = create_app(project)
    callbacks = [
        inspect.unwrap(entry["callback"])
        for entry in app.callback_map.values() if "callback" in entry
    ]
    create = next(func for func in callbacks if func.__name__ == "manage_code_modal")
    save = next(func for func in callbacks if func.__name__ == "save_code_center_version")
    token = context_value.set(AttributeDict(
        triggered_inputs=[{"prop_id": "code-modal-create-button.n_clicks", "value": 1}]
    ))
    try:
        # The shared modal is usable from Constructs without creating any Code.
        result = create(0, 0, 1, 0, 0, 1, "center", "Quality", "Initial",
                        0, 0, [], "scale")
        assert result[0] == {"display": "none"}
        assert result[6] is not None
        assert project.codes() == []
        scale = project.constructs()[0]["construct_id"]
        assert project.scale(scale)["name"] == "Quality"
        # The exact same modal can create a Scale directly in Develop.
        result = create(0, 1, 0, 0, 0, 1, "focus", "Clarity", "Definition",
                        0, 0, [], "scale")
        assert result[6] == project.constructs()[1]["construct_id"]
        assert project.codes() == []
    finally:
        context_value.reset(token)
    save(1, scale, "Quality revised", "Revised", 0, 0)
    assert [v["description"] for v in project.scale_versions(scale)] == [
        "Initial", "Revised",
    ]


def test_scale_revision_refreshes_develop_without_resetting_code_state(tmp_path: Path) -> None:
    """A saved revision updates a selected Scale but not Code classifier selections."""
    pytest.importorskip("dash")
    from dash import no_update
    from dash._callback_context import context_value
    from dash._utils import AttributeDict
    from geometric_coder.ui.app import create_app

    project = _project(tmp_path)
    code = project.create_code("Binary", "code definition")
    scale = project.create_scale("Argument quality", "definition v1")
    app = create_app(project)
    route = _callback(app, "focus-scale-heading.children")
    save = next(
        inspect.unwrap(entry["callback"])
        for entry in app.callback_map.values()
        if "callback" in entry
        and inspect.unwrap(entry["callback"]).__name__ == "save_code_center_version"
    )

    selected = route(scale)
    assert selected[3] == "Scale: Argument quality"
    assert selected[4] == "definition v1"
    assert "Definition version 1" in selected[5]

    # The real save path increments focus-refresh-store. The Construct selection
    # remains unchanged while Dash fires the refresh input.
    saved = save(1, scale, "Argument quality revised", "definition v2", 0, 0)
    assert saved[1] == 1

    token = context_value.set(AttributeDict(
        triggered_inputs=[{"prop_id": "focus-refresh-store.data", "value": 1}]
    ))
    try:
        revised = route(scale, 1)
        unchanged_code = route(code, 1)
    finally:
        context_value.reset(token)

    assert revised[0] is no_update
    assert revised[1] is no_update
    assert revised[2] is no_update
    assert revised[3] == "Scale: Argument quality revised"
    assert revised[4] == "definition v2"
    assert "Definition version 2" in revised[5]
    assert all(value is no_update for value in revised[6:])
    assert all(value is no_update for value in unchanged_code)

    # Explicit selection changes still switch the Code-only controls normally.
    token = context_value.set(AttributeDict(
        triggered_inputs=[{"prop_id": "focus-construct-dropdown.value", "value": code}]
    ))
    try:
        code_selected = route(code, 1)
    finally:
        context_value.reset(token)
    assert code_selected[0] == code
    assert code_selected[1] == {}
    assert code_selected[2] == {"display": "none"}


def test_ui_source_keeps_legacy_code_callbacks_and_new_shared_state() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "geometric_coder" / "ui" / "app.py"
    ).read_text(encoding="utf-8")
    assert 'label="Constructs"' in source
    assert '"focus_construct_id": valid_construct' in source
    assert '"focus_code_id": selected_code' in source
    assert 'Input("focus-construct-dropdown", "value")' in source
    assert 'Input("focus-code-dropdown", "value")' in source
    assert 'project.create_scale(' in source
    assert 'project.update_scale(' in source
    assert "project.scale_versions(" in source
    assert 'id="code-modal-construct-type"' in source
    assert 'Create Construct' in source
    assert 'New Construct' in source
