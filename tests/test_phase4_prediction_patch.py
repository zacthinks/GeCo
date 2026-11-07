"""Phase 4: Develop judgment patches use bounded server-side point indices."""

from pathlib import Path

from geometric_coder.ui.app import (
    _build_prediction_geometry_figure,
    _compact_prediction_geometry,
    _prediction_geometry_page_index,
)


def _points():
    return [
        {
            "observation_id": 1, "unit_id": 1, "kind": "atomic",
            "x": 0.1, "y": 0.2, "assignment": None,
        },
        {
            "observation_id": 8, "unit_id": 8, "kind": "atomic",
            "x": 0.3, "y": 0.7, "assignment": "positive",
        },
        {
            "observation_id": 15, "unit_id": None, "kind": "span",
            "x": 0.6, "y": 0.4, "assignment": "negative",
        },
        {
            "observation_id": 22, "unit_id": None, "kind": "teaching_example",
            "x": 0.8, "y": 0.9, "assignment": "unsure",
        },
    ]


def test_prediction_index_identifies_all_rendered_observation_kinds():
    figure = _build_prediction_geometry_figure(
        geometry={"classifier_fit_ids": [9], "axis_titles": ["X", "Y"]},
        points=_points(), page_index=0,
    )
    lookup = _prediction_geometry_page_index(figure)
    assert set(lookup) == {1, 8, 15, 22}
    assert len(lookup) == 4
    # Four legend traces contain no real observations. Atomic points occupy
    # a shared trace and each derived kind occupies its own trace.
    assert lookup == {1: (4, 0), 8: (4, 1), 15: (5, 0), 22: (6, 0)}
    for observation_id, (trace_index, position) in lookup.items():
        assert int(figure.data[trace_index].customdata[position][0]) == observation_id


def test_prediction_index_is_page_scoped_without_training_or_scoring():
    geometry = {"classifier_fit_ids": [9], "axis_titles": ["X", "Y"]}
    first = _build_prediction_geometry_figure(
        geometry=geometry, points=_points()[:2], page_index=0
    )
    second = _build_prediction_geometry_figure(
        geometry=geometry, points=_points()[2:], page_index=1
    )
    assert set(_prediction_geometry_page_index(first)) == {1, 8}
    assert set(_prediction_geometry_page_index(second)) == {15, 22}


def test_develop_annotation_callback_never_sends_whole_prediction_figure():
    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "geometric_coder" / "ui" / "app.py"
    ).read_text()
    assert 'State("focus-prediction-map", "figure")' not in source
    assert 'State("prediction-geometry-page-token", "data")' in source
    assert 'Output("prediction-geometry-page-token", "data")' in source
    assert 'page[1] == str(prediction_refresh.get("cache_key"))' in source


def test_prediction_cache_drops_unneeded_payload_but_preserves_plot_and_fits():
    from copy import deepcopy

    points = _points()
    for point in points:
        point["text"] = "long observation text" * 200
        point["scores"] = {"classifier A": 0.9, "classifier B": 0.2}
    geometry = {
        "classifier_fit_ids": [9, 10],
        "axis_titles": ["A", "B"],
        "method": "two_classifier_scores",
        "points": points,
    }
    before = _build_prediction_geometry_figure(
        geometry=geometry, points=geometry["points"], page_index=2
    ).to_plotly_json()
    compact = _compact_prediction_geometry(deepcopy(geometry))
    assert compact["classifier_fit_ids"] == [9, 10]
    assert compact["method"] == "two_classifier_scores"
    assert all("text" not in p and "scores" not in p for p in compact["points"])
    after = _build_prediction_geometry_figure(
        geometry=compact, points=compact["points"], page_index=2
    ).to_plotly_json()
    assert after == before
