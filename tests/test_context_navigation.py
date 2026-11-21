"""Regression coverage for contiguous context spans and context-text navigation."""
from __future__ import annotations

import inspect
import json
import shutil
import threading

import pandas as pd
import pytest

from geometric_coder import GeometricCoder
from geometric_coder.ui.app import (
    _append_navigation_history,
    _move_navigation_history,
    _normalized_span_selection,
    _render_context_units,
    _selection_observation_id,
    _span_eligible_order,
    _update_span_selection,
)


def _rows(count=5, *, document="same"):
    return [
        {
            "unit_id": i,
            "row_position": i - 1,
            "user_key": {"document": document, "sentence": i},
            "text": f"Context observation {i}",
        }
        for i in range(1, count + 1)
    ]


def _project(tmp_path, count=5, *, focal=3):
    project = GeometricCoder.create(
        project_dir=tmp_path / "reader.geco",
        data=pd.DataFrame({
            "document": ["same"] * count,
            "sentence": range(1, count + 1),
            "text": [f"Context observation {i}" for i in range(1, count + 1)],
        }),
        keys=["document", "sentence"], text="text", metadata=[], geometries=None,
    )
    session = project.sessions()[0]
    project.patch_session_state(int(session["session_id"]), {"focal_unit_id": focal})
    return project


def _callbacks(app):
    return {
        inspect.unwrap(item["callback"]).__name__: inspect.unwrap(item["callback"])
        for item in app.callback_map.values() if "callback" in item
    }


def test_five_unit_interval_endpoints_are_independent():
    units = _rows()
    current = [3]
    for clicked, checked, expected in (
        (1, True, [1, 2, 3]),
        (5, True, [1, 2, 3, 4, 5]),
        (2, False, [3, 4, 5]),
        (4, False, [3]),
        (3, False, [3]),
        (2, False, [3]),
    ):
        current = _update_span_selection(
            units, focal_unit_id=3, current_unit_ids=current,
            clicked_unit_id=clicked, checked=checked,
        )
        assert current == expected
    assert _normalized_span_selection(units, 3, [1, 3, 5]) == [1, 2, 3, 4, 5]
    # Dash can report re-rendered, already-checked interior rows. These are
    # no-ops, not instructions to retract a previously expanded endpoint.
    for clicked in (2, 4):
        assert _update_span_selection(
            units, focal_unit_id=3, current_unit_ids=[1, 2, 3, 4, 5],
            clicked_unit_id=clicked, checked=True,
        ) == [1, 2, 3, 4, 5]


def test_span_cannot_cross_parent_boundary_and_does_not_persist(tmp_path):
    pytest.importorskip("dash")
    from dash import dcc, html

    units = _rows(5)
    units[0]["user_key"]["document"] = "previous"
    units[-1]["user_key"]["document"] = "next"
    assert _span_eligible_order(units, 3) == [2, 3, 4]
    assert _normalized_span_selection(units, 3, [1, 3, 5]) == [3]
    assert _normalized_span_selection(units, 3, [2, 3, 4]) == [2, 3, 4]
    assert _update_span_selection(
        units, focal_unit_id=3, current_unit_ids=[3],
        clicked_unit_id=5, checked=True,
    ) == [3]
    output = _render_context_units(
        html, dcc, units, focal_unit_id=3, key_columns=["document", "sentence"],
        selected_unit_ids=[2, 3, 4], checkbox_type="explore-span-checkbox",
        navigation_surface="explore",
    )
    checklists = [part.children[0] for part in output
                  if getattr(part, "className", "").startswith("context-unit-row")]
    assert [check.options[0]["disabled"] for check in checklists] == [
        True, False, True, False, True,
    ]
    assert [check.value for check in checklists] == [[], [2], [3], [4], []]

    project = _project(tmp_path)
    assert _selection_observation_id(project, 3, [2, 3]) is None
    assert project.span_observation([2, 3]) is None


def test_explore_focus_checkbox_callback_transitions(tmp_path):
    pytest.importorskip("dash")
    from dash._callback_context import context_value
    from dash._utils import AttributeDict
    from geometric_coder.ui.app import create_app

    app = create_app(_project(tmp_path))
    callbacks = _callbacks(app)
    for surface in ("explore", "focus"):
        selected = [3]
        for clicked, checked, expected in (
            (1, True, [1, 2, 3]),
            (5, True, [1, 2, 3, 4, 5]),
            (2, False, [3, 4, 5]),
            (4, False, [3]),
        ):
            identity = {"type": f"{surface}-span-checkbox", "unit_id": clicked}
            token = context_value.set(AttributeDict(triggered_inputs=[{
                "prop_id": json.dumps(identity) + ".value",
                "value": [clicked] if checked else [],
            }]))
            try:
                selected = callbacks[f"update_{surface}_span_selection"](
                    [[clicked] if checked else []], [identity],
                    3, 0, 1, selected,
                )
            finally:
                context_value.reset(token)
            assert selected == expected, surface


def test_context_jump_records_one_visit_and_history(tmp_path):
    pytest.importorskip("dash")
    from dash import no_update
    from geometric_coder.ui.app import create_app

    project = _project(tmp_path)
    app = create_app(project)
    callbacks = _callbacks(app)
    session = int(project.sessions()[0]["session_id"])
    assert "navigate_from_context" in callbacks
    history = {"items": [3], "index": 0}
    target, history = callbacks["navigate_from_context"](
        {"surface": "explore", "unit_id": 1}, 3, session, None, None,
        history, "explore",
    )
    assert target == 1
    assert history == {"items": [3, 1], "index": 1}
    assert project.database.seen_unit_ids(session) == {1}
    assert callbacks["navigate_from_context"](
        {"surface": "explore", "unit_id": 1}, 1, session, None, None,
        history, "explore",
    ) == (no_update, no_update)
    assert callbacks["navigate_from_context"](
        {"surface": "focus", "unit_id": 2}, 1, session, None, None,
        history, "explore",
    ) == (no_update, no_update)
    back = _move_navigation_history(history, -1)
    assert back == (3, {"items": [3, 1], "index": 0})
    assert _append_navigation_history(back[1], 2) == {"items": [3, 2], "index": 1}
    focal, focus_history = callbacks["navigate_from_context"](
        {"surface": "focus", "unit_id": 5}, 1, session, None, None,
        history, "focus",
    )
    assert focal == 5
    assert focus_history["items"] == [3, 1, 5]


def test_200_unit_interval_expands_contiguously(tmp_path):
    from geometric_coder.ui.context_expansion import read_context

    project = _project(tmp_path, 450, focal=225)
    initial, truncated = read_context(project, 225, level=0, window=1, limit=200)
    assert len(initial) == 200 and truncated
    selected = _update_span_selection(
        initial, focal_unit_id=225, current_unit_ids=[225],
        clicked_unit_id=int(initial[0]["unit_id"]), checked=True,
    )
    assert selected == list(range(selected[0], 226))
    expanded, _ = read_context(project, 225, level=0, window=1, limit=400)
    assert _normalized_span_selection(expanded, 225, selected) == selected
    farther = int(expanded[-1]["unit_id"])
    selected = _update_span_selection(
        expanded, focal_unit_id=225, current_unit_ids=selected,
        clicked_unit_id=farther, checked=True,
    )
    assert selected == list(range(selected[0], farther + 1))


@pytest.mark.parametrize("surface", ["explore", "focus"])
def test_context_text_has_keyboard_destination_without_checkbox_bubbling(surface):
    pytest.importorskip("dash")
    from dash import dcc, html

    output = _render_context_units(
        html, dcc, _rows(), focal_unit_id=3,
        key_columns=["document", "sentence"], checkbox_type=f"{surface}-span-checkbox",
        navigation_surface=surface,
    )
    rows = [x for x in output if getattr(x, "className", "").startswith("context-unit-row")]
    text = rows[0].children[1]
    assert text.tabIndex == 0
    assert text.role == "button"
    assert text.__dict__["data-geco-context-unit"] == "1"
    assert text.__dict__["data-geco-context-surface"] == surface
    assert rows[0].children[0].id["unit_id"] == 1


def test_live_browser_doubleclick_navigation_preserves_span_and_history(tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    chrome = next(
        (path for executable in ("google-chrome", "chromium", "chromium-browser")
         if (path := shutil.which(executable)) is not None), None,
    )
    if chrome is None:
        pytest.skip("Chrome/Chromium not installed")
    from werkzeug.serving import make_server
    from geometric_coder.ui.app import create_app

    project = _project(tmp_path, 260, focal=110)
    code_id = project.create_code("Reader navigation")
    session = int(project.sessions()[0]["session_id"])
    project.patch_session_state(session, {
        "focus_construct_id": code_id, "focus_code_id": code_id,
    })
    app = create_app(project)
    server = make_server("127.0.0.1", 0, app.server, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with playwright.sync_playwright() as runner:
            browser = runner.chromium.launch(
                executable_path=chrome, headless=True, args=["--no-sandbox"],
            )
            try:
                page = browser.new_page(viewport={"width": 1400, "height": 1000})
                page.goto(f"http://127.0.0.1:{server.server_port}/")
                # The default context level is atomic (one row). Use the
                # document level to exercise the bounded 200-unit reader.
                page.wait_for_function(
                    "() => !!window.dash_clientside?.set_props && !!document.querySelector('#context-level')"
                )
                page.evaluate(
                    "() => window.dash_clientside.set_props('context-level', {value: 0})"
                )
                rows = page.locator("#context-content .context-unit-row")
                page.wait_for_function(
                    "() => document.querySelectorAll('#context-content .context-unit-row').length === 200",
                    timeout=20000,
                )
                assert rows.count() == 200
                # Only checkboxes, not text click handlers, mutate span selection.
                selected = page.locator("#context-content .span-unit-checkbox input:checked")
                assert selected.count() == 1
                first = rows.first
                first.locator("input").check()
                page.wait_for_function("""() => {
                    const rows = [...document.querySelectorAll('#context-content .context-unit-row')];
                    const focal = rows.findIndex(row => row.querySelector('.focal-context-unit'));
                    return focal >= 0 &&
                        rows.slice(0, focal + 1).every(row => row.querySelector('input').checked);
                }""", timeout=15000)
                # Double-click text (never checkbox) and check visit / history.
                target = rows.last.locator("p.context-unit")
                target_text = target.inner_text()
                target.dblclick()
                page.wait_for_function(
                    "text => document.querySelector('#focal-key')?.innerText.includes(text)",
                    arg=target_text.replace("Context observation ", "sentence: "),
                    timeout=15000,
                )
                page.wait_for_function(
                    "() => document.querySelectorAll('#context-content .span-unit-checkbox input:checked').length === 1",
                    timeout=15000,
                )
                assert page.locator("#nav-back").is_enabled()
                page.locator("#nav-back").click()
                page.wait_for_function(
                    "() => document.querySelector('#focal-key')?.innerText.includes('sentence: 110')",
                    timeout=15000,
                )
                # Focus uses the same reading navigation gesture.
                page.get_by_text("Develop", exact=True).first.click()
                page.evaluate(
                    "() => window.dash_clientside.set_props('focus-context-level', {value: 0})"
                )
                focus_text = page.locator("#focus-context-content p.context-unit")
                page.wait_for_function(
                    "() => document.querySelectorAll('#focus-context-content p.context-unit').length > 1",
                    timeout=15000,
                )
                focus_text.first.press("Enter")
                page.wait_for_function(
                    "() => !document.querySelector('#focus-focal-key')?.innerText.includes('sentence: 110')",
                    timeout=15000,
                )
                assert page.locator("#focus-context-content .span-unit-checkbox input:checked").count() == 1
            finally:
                browser.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)
