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
        "scale-context-level", "scale-context-window", "scale-prediction-preview",
    ]:
        assert find(layout, name) is not None


def test_real_dash_callbacks_are_registered_and_untrained_plot_is_safe(tmp_path):
    pytest.importorskip("dash")
    import inspect
    import pandas as pd
    from dash import no_update
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
    assert "step_scale_prediction_page" in by_name

    render = by_name["render_scale_predictions"]
    plot, message, page, pages = render(scale, 1, 0, "", "", None)
    assert plot.data == ()
    assert "train" in message.lower()
    assert page is None and pages == 1

    plot, message, page, pages = render(code, 1, 0, "", "", None)
    assert plot.data == () and message == "" and page is None
    inspect_point = by_name["inspect_scale_prediction_point"]
    assert inspect_point(
        {"points": [{"customdata": [1, 1]}]},
        {"scale_id": scale, "fit_id": 999, "observation_ids": [1]},
        code,
        0,
        "1",
    ) is no_update
