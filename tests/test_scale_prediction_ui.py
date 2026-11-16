"""Scale prediction page and researcher-facing comparison layout regressions."""
from __future__ import annotations

from array import array

import pytest

from geometric_coder.ui.scale_prediction import (
    scale_prediction_figure,
    scale_prediction_page,
)


class _Database:
    def __init__(self, ids):
        self.ids = ids
        self.seen = []

    def acquisition_eligible_atomic_ids(self, kind, scale_id, *, eligibility_spec):
        assert kind == "scale"
        assert scale_id == 7
        assert eligibility_spec == {"kind": "all"}
        return array("q", self.ids)


class _Project:
    def __init__(self, ids=range(1, 10), status="current", uncertainty=True):
        self.database = _Database(ids)
        self.status = status
        self.uncertainty = uncertainty
        self.scored = []

    def scale_model_status(self, scale_id):
        assert scale_id == 7
        return {
            "status": self.status,
            "fit": {"scaler_fit_id": 13} if self.status in {"current", "stale"} else None,
        }

    def scale_eligibility(self, scale_id):
        assert scale_id == 7
        return {"eligibility": {"kind": "all"}}

    def scale_scores(self, scale_id, ids, *, allow_stale, include_uncertainty):
        assert scale_id == 7 and allow_stale and include_uncertainty
        self.scored.append(list(ids))
        return [{
            "observation_id": item,
            "score": float(item - 5),
            "epistemic_sd": float(item) / 10 if self.uncertainty else None,
            "scaler_fit_id": 13,
        } for item in ids]

    def observation(self, item):
        return {
            "kind": "atomic", "observation_id": item, "unit_id": item,
            "user_key": {"source": f"example-{item}"},
        }


def test_scale_plot_scores_only_requested_page_and_retains_signed_positions():
    project = _Project()
    result = scale_prediction_page(project, 7, page_number=2, page_size=3)
    assert result["pages"] == 3
    assert result["page"] == 2
    assert project.scored == [[4, 5, 6]]
    assert [row["score"] for row in result["records"]] == [-1.0, 0.0, 1.0]
    figure = scale_prediction_figure(result)
    assert figure.data[0].x == (-1.0, 0.0, 1.0)
    assert figure.data[0].y == pytest.approx((0.4, 0.5, 0.6))
    assert [tuple(item) for item in figure.data[0].customdata] == [
        (4, 4), (5, 5), (6, 6),
    ]
    assert "epistemic" in figure.layout.yaxis.title.text.lower()
    assert "centered" in figure.layout.xaxis.title.text.lower()


def test_missing_posterior_is_not_disguised_as_zero_uncertainty():
    project = _Project(ids=[1, 2], uncertainty=False, status="stale")
    page = scale_prediction_page(project, 7, page_size=2)
    assert not page["uncertainty_available"]
    assert "Stale" in page["message"]
    assert "last saved fit" in page["message"]
    assert "uncertainty unavailable" in page["message"]
    plot = scale_prediction_figure(page)
    assert "not uncertainty" in plot.layout.yaxis.title.text.lower()
    assert list(plot.data[0].y) == [1, 2]
    assert [tuple(item) for item in plot.data[0].customdata] == [(1, 1), (2, 2)]


def test_untrained_does_not_score_and_empty_eligibility_is_reported():
    project = _Project(status="not_trained")
    page = scale_prediction_page(project, 7)
    assert page["records"] == []
    assert "train" in page["message"].lower()
    assert project.scored == []
    other = _Project(ids=[])
    empty = scale_prediction_page(other, 7)
    assert empty["records"] == []
    assert "No eligible" in empty["message"]
    assert scale_prediction_figure(empty).data == ()


def test_prediction_rejects_misaligned_score_results_and_forged_fit():
    project = _Project(ids=[1, 2])
    project.scale_scores = lambda *args, **kwargs: [
        {"observation_id": 2, "score": 0, "epistemic_sd": 1, "scaler_fit_id": 13},
        {"observation_id": 1, "score": 0, "epistemic_sd": 1, "scaler_fit_id": 13},
    ]
    with pytest.raises(ValueError, match="observation IDs"):
        scale_prediction_page(project, 7, page_size=2)
    project = _Project(ids=[1])
    project.scale_scores = lambda *args, **kwargs: [
        {"observation_id": 1, "score": 0, "epistemic_sd": 1, "scaler_fit_id": 999},
    ]
    with pytest.raises(ValueError, match="unexpected fit"):
        scale_prediction_page(project, 7)


def test_scale_comparison_controls_are_right_aligned_without_replacing_judgment_ids():
    dash = pytest.importorskip("dash")
    from geometric_coder.ui.scale_comparisons import comparison_layout
    from dash import dcc, html, dash_table

    layout = comparison_layout(dcc=dcc, html=html, dash_table=dash_table)

    def find(node, wanted):
        if getattr(node, "id", None) == wanted:
            return node
        children = getattr(node, "children", None)
        if children is None:
            return None
        if not isinstance(children, (tuple, list)):
            children = [children]
        for child in children:
            result = find(child, wanted)
            if result is not None:
                return result
        return None

    main = next(component for component in layout.children
                if getattr(component, "className", None) == "scale-comparison-main")
    grid = next(component for component in main.children
                if getattr(component, "className", None) == "scale-comparison-grid")
    assert [getattr(child, "id", None) or child.className
            for child in grid.children] == [
        "scale-reading-first", "scale-reading-second", "scale-judgment-actions",
    ]
    for name in [
        "scale-judge-first", "scale-judge-second", "scale-judge-parity",
        "scale-judge-unsure", "scale-next-pair",
        "scale-prediction-map", "scale-prediction-page-state",
        "scale-prediction-selection",
        "scale-context-level", "scale-context-window", "scale-prediction-preview",
    ]:
        assert find(layout, name) is not None
    actions = grid.children[2]
    assert actions.className == "scale-judgment-actions"
    assert find(actions, "scale-next-pair") is not None
    assert find(layout, "scale-next-pair").className == "primary-action scale-next-pair"
    assert getattr(find(layout, "scale-next-pair"), "disabled", False) is False
    main_children = main.children
    panel = next(child for child in main_children
                 if getattr(child, "className", None) == "focus-coding-panel scale-prediction-panel")
    assert panel.__class__.__name__ == "Div"
    assert panel.children[0].className == "scale-prediction-header"
    assert panel.children[0].children[0].children == "Scale prediction positions"


def test_real_dash_callbacks_are_registered_and_untrained_plot_is_safe(tmp_path):
    pytest.importorskip("dash")
    import inspect
    import pandas as pd
    from dash import html
    from geometric_coder import GeometricCoder
    from geometric_coder.ui.app import create_app

    project = GeometricCoder.create(
        project_dir=tmp_path / "prediction-callback.geco",
        data=pd.DataFrame({"key": [1, 2, 3], "text": ["A", "B", "C"]}),
        keys=["key"], text="text", geometries=None,
    )
    scale = project.create_scale("Clarity", "Continuous")
    code = project.create_code("Relevant")
    app = create_app(project)
    by_name = {
        inspect.unwrap(record["callback"]).__name__: inspect.unwrap(record["callback"])
        for record in app.callback_map.values()
        if "callback" in record
    }
    assert "render_scale_predictions" in by_name
    assert "inspect_scale_prediction_point" in by_name
    assert "select_scale_prediction_point" in by_name
    assert "step_scale_prediction_page" in by_name

    render = by_name["render_scale_predictions"]
    plot, message, page, pages = render(scale, 1, 0, "", "", None)
    assert plot.data == ()
    assert "train" in message.lower()
    assert page is None and pages == 1

    plot, message, page, pages = render(code, 1, 0, "", "", None)
    assert plot.data == () and message == "" and page is None
    inspect_point = by_name["inspect_scale_prediction_point"]
    cleared = inspect_point(
        {"points": [{"customdata": [1, 1]}]},
        {"scale_id": scale, "fit_id": 999, "observation_ids": [1]},
        code,
        0,
        "1",
    )
    assert isinstance(cleared, html.P)
    assert cleared.children == "Select a plotted observation to inspect it."


def test_prediction_preview_resets_when_page_state_is_cleared(tmp_path):
    pytest.importorskip("dash")
    import inspect
    import pandas as pd
    from dash import html
    from geometric_coder import GeometricCoder
    from geometric_coder.ui.app import create_app

    project = GeometricCoder.create(
        project_dir=tmp_path / "preview-reset.geco",
        data=pd.DataFrame({"key": [1, 2], "text": ["A", "B"]}),
        keys=["key"], text="text", geometries=None,
    )
    scale = project.create_scale("Clarity")
    app = create_app(project)
    preview = next(
        inspect.unwrap(entry["callback"])
        for entry in app.callback_map.values()
        if "callback" in entry
        and inspect.unwrap(entry["callback"]).__name__ == "inspect_scale_prediction_point"
    )
    stale_selection = {
        "scale_id": scale, "fit_id": 999, "page": 1,
        "observation_id": 1, "unit_id": 1,
    }
    cleared = preview(stale_selection, None, scale, 0, "1")
    assert isinstance(cleared, html.P)
    assert cleared.children == "Select a plotted observation to inspect it."

    output = next(
        entry for key, entry in app.callback_map.items()
        if "scale-prediction-preview.children" in key
    )
    names = [item["id"] for item in output["inputs"]]
    assert "scale-prediction-page-state" in names
    assert "scale-prediction-selection" in names
    assert "scale-prediction-map" not in names
    assert "focus-construct-dropdown" in names
    assert "scale-context-level" in names
    assert "scale-context-window" in names


def test_prediction_selection_lives_across_clickdata_clearing_and_context_changes(
    tmp_path, monkeypatch,
):
    """Exercise actual registered Dash callbacks, not merely the pure validator."""
    pytest.importorskip("dash")
    import inspect
    import pandas as pd
    from dash import html, no_update
    from dash._callback_context import context_value
    from dash._utils import AttributeDict
    from geometric_coder import GeometricCoder
    from geometric_coder.ui.app import create_app

    project = GeometricCoder.create(
        project_dir=tmp_path / "preview-persistence.geco",
        data=pd.DataFrame({"key": [1, 2], "text": ["First text", "Second text"]}),
        keys=["key"], text="text", geometries=None,
    )
    scale = project.create_scale("Clarity")
    other = project.create_scale("Other")
    app = create_app(project)

    monkeypatch.setattr(
        project, "scale_model_status",
        lambda scale_id: {"status": "current", "fit": {"scaler_fit_id": 13}},
    )
    monkeypatch.setattr(
        project, "scale_scores",
        lambda scale_id, ids, **kwargs: [
            {"observation_id": identity, "score": float(index - 1),
             "epistemic_sd": 0.1 * (index + 1), "scaler_fit_id": 13}
            for index, identity in enumerate(ids)
        ],
    )
    by_name = {
        inspect.unwrap(entry["callback"]).__name__: inspect.unwrap(entry["callback"])
        for entry in app.callback_map.values() if "callback" in entry
    }
    render = by_name["render_scale_predictions"]
    select = by_name["select_scale_prediction_point"]
    preview = by_name["inspect_scale_prediction_point"]

    figure, _, page_state, _ = render(scale, 1, 0, "", "", None)
    assert figure.data[0].x == (-1.0, 0.0)
    observation_id, unit_id, render_token = figure.data[0].customdata[0]
    clicked = {"points": [{"customdata": [observation_id, unit_id, render_token]}]}
    assert render_token == page_state["render_token"]
    no_side_effects = (
        len(project.scale_comparisons(scale)),
        len(project.scale_comparisons(other)),
    )
    def call_selection(trigger, click_data, page, construct, previous):
        token = context_value.set(AttributeDict(
            triggered_inputs=[{"prop_id": trigger, "value": click_data}]
        ))
        try:
            return select(click_data, page, construct, previous)
        finally:
            context_value.reset(token)

    selected = call_selection(
        "scale-prediction-map.clickData", clicked, page_state, scale, None,
    )
    assert selected == {
        "scale_id": scale, "fit_id": 13, "page": 1, "page_size": 128,
        "observation_id": observation_id, "unit_id": unit_id,
    }
    first_preview = preview(selected, page_state, scale, 0, "1")
    assert isinstance(first_preview, html.Div)
    assert first_preview.children[2].children == "First text"

    # Plotly clears clickData while rebuilding; the selected store must survive.
    assert call_selection(
        "scale-prediction-map.clickData", None, page_state, scale, selected,
    ) is no_update
    _, _, rerendered, _ = render(scale, 1, 1, "", "", None)
    assert rerendered["render_token"] != page_state["render_token"]
    assert rerendered["observation_ids"] == page_state["observation_ids"]
    assert call_selection(
        "scale-prediction-page-state.data", None, rerendered, scale, selected,
    ) is no_update
    assert isinstance(preview(selected, rerendered, scale, 0, "3"), html.Div)
    assert isinstance(preview(selected, rerendered, scale, 0, "1"), html.Div)
    # The server can temporarily clear page-state while replacing a figure.
    # This is not a valid reason to erase the current selection/inspector.
    assert call_selection(
        "scale-prediction-page-state.data", None, None, scale, selected,
    ) is no_update
    assert preview(selected, None, scale, 0, "1") is no_update

    # Stale/fabricated click identities never change the saved inspector.
    fake_click = {"points": [{"customdata": [999999, unit_id, render_token]}]}
    assert call_selection(
        "scale-prediction-map.clickData", fake_click, rerendered, scale, selected,
    ) is no_update
    # A stale old-figure click must not become the first selected point after
    # refreshing, even if the same observation remains on the new page.
    assert call_selection(
        "scale-prediction-map.clickData", clicked, rerendered, scale, selected,
    ) is no_update
    new_point = list(figure.data[0].customdata[1])
    new_point[2] = rerendered["render_token"]
    chosen_second = call_selection(
        "scale-prediction-map.clickData",
        {"points": [{"customdata": new_point}]}, rerendered, scale, selected,
    )
    assert chosen_second["observation_id"] != selected["observation_id"]
    assert chosen_second["unit_id"] == new_point[1]
    # The previous selection is still the observation in the preview until
    # the new valid click explicitly replaces it.
    assert (
        len(project.scale_comparisons(scale)),
        len(project.scale_comparisons(other)),
    ) == no_side_effects

    # Departing the page, switching constructs, or replacing the fit clears it.
    departed = {**page_state, "page": 2, "observation_ids": []}
    assert call_selection(
        "scale-prediction-page-state.data", None, departed, scale, selected,
    ) is None
    assert isinstance(preview(selected, departed, scale, 0, "1"), html.P)
    assert call_selection(
        "focus-construct-dropdown.value", None, page_state, other, selected,
    ) is None
    assert isinstance(preview(selected, page_state, other, 0, "1"), html.P)
    monkeypatch.setattr(
        project, "scale_model_status",
        lambda scale_id: {"status": "current", "fit": {"scaler_fit_id": 14}},
    )
    assert call_selection(
        "scale-prediction-page-state.data", None, page_state, scale, selected,
    ) is None
    assert isinstance(preview(selected, page_state, scale, 0, "1"), html.P)


def test_scale_selected_judgments_are_green_without_binary_outcome_colors():
    from pathlib import Path
    css = (
        Path(__file__).resolve().parents[1]
        / "src" / "geometric_coder" / "ui" / "assets" / "geco.css"
    ).read_text()
    selected = css.split(".scale-judgment-button.judgment-selected {", 1)[1].split("}", 1)[0]
    assert "var(--geco-green-600)" in selected
    assert "var(--geco-green-700)" in selected
    assert "#425e70" not in css
    assert "#344c5b" not in css
    assert ".scale-judgment-button.judgment-selected:focus-visible" in css
    assert ".scale-next-pair-wrap" in css
    assert "align-items: stretch" in css


def test_scale_prediction_page_size_is_adjustable_and_bounded():
    from geometric_coder.ui.scale_prediction import MAX_SCALE_PLOT_PAGE_SIZE
    assert MAX_SCALE_PLOT_PAGE_SIZE == 5000
    project = _Project(ids=range(1, 251))
    page = scale_prediction_page(project, 7, page_number=3, page_size=100)
    assert page["page"] == 3
    assert page["pages"] == 3
    assert len(page["records"]) == 50
    assert project.scored == [list(range(201, 251))]
    with pytest.raises(ValueError, match="page size"):
        scale_prediction_page(project, 7, page_size=5001)


def test_scale_prediction_toolbar_matches_code_pagination_pattern():
    from dash import dcc, html, dash_table
    from geometric_coder.ui.scale_comparisons import comparison_layout

    def find(node, identifier):
        if getattr(node, "id", None) == identifier:
            return node
        children = getattr(node, "children", None)
        if children is None:
            return None
        for child in children if isinstance(children, (list, tuple)) else [children]:
            if isinstance(child, str):
                continue
            found = find(child, identifier)
            if found is not None:
                return found
        return None

    layout = comparison_layout(dcc=dcc, html=html, dash_table=dash_table)
    size = find(layout, "scale-prediction-page-size")
    assert size.value == "128" and size.debounce is True
    assert find(layout, "scale-prediction-previous") is not None
    assert find(layout, "scale-prediction-next") is not None
    assert find(layout, "scale-prediction-page") is not None
    assert find(layout, "scale-prediction-refresh") is not None
    assert find(layout, "scale-bootstrap-open") is not None
    assert find(layout, "scale-bootstrap-modal").style == {"display": "none"}
