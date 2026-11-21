"""Incremental reader context must remain bounded and support long-span selection."""
from __future__ import annotations

import inspect

import pandas as pd
import pytest

from geometric_coder import GeometricCoder
from geometric_coder.ui.context_expansion import (
    EXPANDED_CONTEXT_MAX, INITIAL_CONTEXT_UNITS,
    advance_limit, context_key, current_limit, read_context, truncation_notice,
)
from geometric_coder.ui.app import _normalized_span_selection, _update_span_selection


def _project(tmp_path, count=650):
    return GeometricCoder.create(
        project_dir=tmp_path / "expanded.geco",
        data=pd.DataFrame({
            "document": ["same"] * count,
            "sentence": range(count),
            "text": [f"Research observation {i}. " + "Long context " * 6 for i in range(count)],
        }), keys=["document", "sentence"], text="text",
        metadata=[], geometries=None,
    )


def test_expansion_limited_stepwise_and_reset_by_context_identity():
    key = context_key(310, 0, "1")
    state = advance_limit(None, key, expand=False)
    assert current_limit(state, key) == INITIAL_CONTEXT_UNITS
    state = advance_limit(state, key, expand=True)
    assert current_limit(state, key) == 400
    assert current_limit(state, context_key(311, 0, "1")) == 200
    assert current_limit(state, context_key(310, 1, "1")) == 200
    assert current_limit(state, context_key(310, 0, "2")) == 200
    for _ in range(100):
        state = advance_limit(state, key, expand=True)
    assert current_limit(state, key) == EXPANDED_CONTEXT_MAX
    assert advance_limit(state, key, expand=True)["limit"] == EXPANDED_CONTEXT_MAX
    assert advance_limit(state, context_key(311, 0, "1"), expand=False)["limit"] == 200
    assert current_limit({"key": key, "limit": -100}, key) == 200
    assert current_limit({"key": key, "limit": 100000}, key) == EXPANDED_CONTEXT_MAX


def test_expanding_enables_longer_contiguous_span_without_mutating_evidence(tmp_path):
    project = _project(tmp_path)
    focal_id = 310
    key = context_key(focal_id, 0, "1")
    initial, truncated = read_context(project, focal_id, level=0, window=1, limit=200)
    assert len(initial) == 200 and truncated
    assert focal_id in [row["unit_id"] for row in initial]
    assert 490 not in [row["unit_id"] for row in initial]

    state = advance_limit(None, key, expand=False)
    state = advance_limit(state, key, expand=True)
    expanded, more = read_context(
        project, focal_id, level=0, window=1, limit=current_limit(state, key),
    )
    assert more
    assert len(expanded) == 400
    assert [row["unit_id"] for row in initial] == [
        row["unit_id"] for row in expanded
        if row["unit_id"] in {x["unit_id"] for x in initial}
    ]
    assert 490 in [row["unit_id"] for row in expanded]
    selected = _update_span_selection(
        expanded, focal_unit_id=focal_id,
        current_unit_ids=[focal_id], clicked_unit_id=490, checked=True,
    )
    assert selected == list(range(focal_id, 491))
    # Selection remains contiguous and renderable after another expansion.
    larger = read_context(project, focal_id, level=0, window=1, limit=600)[0]
    assert _normalized_span_selection(larger, focal_id, selected) == selected
    # Expansion only reads; it never creates a saved coding observation.
    assert project.observation(focal_id)["kind"] == "atomic"


def test_truncation_boundary_and_lookahead_are_bounded(tmp_path, monkeypatch):
    project = _project(tmp_path, count=620)
    requested = []
    context = project.context

    def spy(*args, **kwargs):
        requested.append(kwargs["max_units"])
        return context(*args, **kwargs)

    monkeypatch.setattr(project, "context", spy)
    small, hidden = read_context(project, 1, level=0, window=1, limit=200)
    assert len(small) == 200 and hidden
    assert requested == [201]
    larger, hidden = read_context(project, 1, level=0, window=1, limit=600)
    assert len(larger) == 600 and hidden
    complete, hidden = read_context(project, 1, level=0, window=1, limit=1000)
    assert len(complete) == 620 and not hidden
    assert requested == [201, 601, 1001]
    from dash import html
    assert truncation_notice(html, 200, True, 200).className == "context-truncated"
    assert truncation_notice(html, 620, False, 1000) is None


def test_registered_explore_focus_memo_and_scale_expansion_callbacks(tmp_path):
    pytest.importorskip("dash")
    from geometric_coder.ui.app import create_app

    project = _project(tmp_path)
    app = create_app(project)
    funcs = {
        inspect.unwrap(item["callback"]).__name__: inspect.unwrap(item["callback"])
        for item in app.callback_map.values() if "callback" in item
    }
    for name in (
        "expand_explore_context", "expand_focus_context", "expand_memo_context",
        "expand_scale_context", "expand_scale_prediction_context",
        "update_reading_panel", "update_focus_reading_panel",
        "update_memo_presentation_reader", "display_scale_task",
        "inspect_scale_prediction_point",
    ):
        assert name in funcs, name
    focal = 310
    for name in ("update_reading_panel", "update_focus_reading_panel"):
        label, children, metadata = funcs[name](focal, 0, 1)
        assert len([item for item in children if getattr(item, "className", None) == "context-truncated"]) == 1
        extended_state = {"key": context_key(focal, 0, 1), "limit": 400}
        label2, extended_children, _ = funcs[name](focal, 0, 1, extended_state, [focal, 490])
        checklist_ids = []
        for item in extended_children:
            for inner in getattr(item, "children", []) if isinstance(getattr(item, "children", None), list) else []:
                identifier = getattr(inner, "id", None)
                if isinstance(identifier, dict) and identifier.get("unit_id"):
                    checklist_ids.append(identifier["unit_id"])
        assert len(checklist_ids) == 400
        assert 490 in checklist_ids

    # A leaf-level window of five genuinely contains only five units;
    # it must not advertise truncation or fabricate an expansion.
    memo = funcs["update_memo_presentation_reader"]
    _, short, _ = memo("#geco-presentation-unit-310")
    assert isinstance(short, list)
    assert not any(getattr(item, "className", None) == "context-truncated" for item in short)
    _, longer, _ = memo("#geco-presentation-unit-310", {"key": context_key(310), "limit": 400})
    assert len(longer) == len(short)


def test_explore_and_focus_checkbox_callbacks_select_beyond_initial_cap(tmp_path):
    pytest.importorskip("dash")
    import json
    from dash._callback_context import context_value
    from dash._utils import AttributeDict
    from geometric_coder.ui.app import create_app

    project = _project(tmp_path)
    app = create_app(project)
    funcs = {
        inspect.unwrap(item["callback"]).__name__: inspect.unwrap(item["callback"])
        for item in app.callback_map.values() if "callback" in item
    }
    expanded = {"key": context_key(310, 0, 1), "limit": 400}
    for prefix in ("explore", "focus"):
        clicked = {"type": f"{prefix}-span-checkbox", "unit_id": 490}
        token = context_value.set(AttributeDict(
            triggered_inputs=[{"prop_id": json.dumps(clicked) + ".value", "value": [490]}],
        ))
        try:
            selected = funcs[f"update_{prefix}_span_selection"](
                [[490]], [clicked], 310, 0, 1, [310], expanded,
            )
        finally:
            context_value.reset(token)
        assert selected == list(range(310, 491)), prefix


def test_expansion_callbacks_do_not_fire_on_focal_or_prediction_changes(tmp_path):
    """Only deliberate load-more clicks can write expansion state."""
    pytest.importorskip("dash")
    from geometric_coder.ui.app import create_app

    project = _project(tmp_path)
    app = create_app(project)
    expected = {
        "expand_explore_context": "explore-context-expand",
        "expand_focus_context": "focus-context-expand",
        "expand_memo_context": "memo-context-expand",
        "expand_scale_context": "scale-context-expand",
        "expand_scale_prediction_context": "scale-prediction-context-expand",
    }
    observed = {}
    for record in app.callback_map.values():
        if "callback" not in record:
            continue
        name = inspect.unwrap(record["callback"]).__name__
        if name in expected:
            observed[name] = [entry["id"] for entry in record["inputs"]]
    assert observed == {name: [button] for name, button in expected.items()}


def test_memo_presentation_expands_large_single_level_group(tmp_path):
    pytest.importorskip("dash")
    from geometric_coder.ui.app import create_app

    count = 650
    project = GeometricCoder.create(
        project_dir=tmp_path / "single-level.geco",
        data=pd.DataFrame({
            "document": ["shared"] * count,
            "text": [f"Memo evidence {i}." for i in range(count)],
        }), keys=["document"], text="text", metadata=[], geometries=None,
    )
    app = create_app(project)
    reader = next(
        inspect.unwrap(record["callback"])
        for record in app.callback_map.values() if "callback" in record
        and inspect.unwrap(record["callback"]).__name__ == "update_memo_presentation_reader"
    )
    _, short, _ = reader("#geco-presentation-unit-310")
    assert sum(getattr(item, "className", None) == "context-truncated" for item in short) == 1
    _, longer, _ = reader(
        "#geco-presentation-unit-310", {"key": context_key(310), "limit": 400},
    )
    assert len(longer) > len(short)
    assert sum(getattr(item, "className", None) == "context-truncated" for item in longer) == 1
