"""Phase 3: compact Explore overlay frames preserve chart semantics."""

import json

import numpy as np

from geometric_coder.ui.app import (
    _geometry_overlay_frame,
    _geometry_overlay_payload,
)
from geometric_coder.ui.figures import build_geometry_figure


def test_overlay_frame_matches_original_plotly_overlay():
    units = [
        {"unit_id": 10, "text": "alpha", "user_key": {"doc": "A"}},
        {"unit_id": 20, "text": "beta", "user_key": {"doc": "B"}},
        {"unit_id": 30, "text": "gamma", "user_key": {"doc": "C"}},
    ]
    figure = build_geometry_figure(
        coordinates=np.array([[0, 1], [1, 2], [2, 3]], dtype=float),
        units=units,
        eligible=[True, False, True],
        seen_unit_ids={10},
        focal_unit_id=20,
        geometry_name="example",
        view_name="test",
    )
    full = figure.to_plotly_json()
    compact = _geometry_overlay_frame(figure)
    assert compact is not None
    assert len(compact["data"]) == 4
    for trace in compact["data"]:
        assert "marker" not in trace
        assert "layout" not in trace
    for seen, focal in [({10}, 20), ({20, 30}, 30), (set(), None)]:
        assert _geometry_overlay_payload(
            compact, seen_unit_ids=seen, focal_unit_id=focal
        ) == _geometry_overlay_payload(
            full, seen_unit_ids=seen, focal_unit_id=focal
        )
    assert len(json.dumps(compact, default=lambda x: x.tolist())) < len(
        json.dumps(full, default=lambda x: x.tolist())
    )


def test_overlay_frame_keeps_id_stable_1d_jitter_and_hover():
    units = [
        {"unit_id": 100 + n, "text": f"t{n}", "user_key": {"doc": n}}
        for n in range(4)
    ]
    figure = build_geometry_figure(
        coordinates=np.array([[0], [1], [2], [3]], dtype=float),
        units=units,
        eligible=[True, False, True, False],
        jitter_strength=1.5,
        jitter_spread=1.0,
    )
    frame = _geometry_overlay_frame(figure)
    assert frame is not None
    payload = _geometry_overlay_payload(
        frame, seen_unit_ids={100, 103}, focal_unit_id=102
    )
    full_payload = _geometry_overlay_payload(
        figure.to_plotly_json(), seen_unit_ids={100, 103}, focal_unit_id=102
    )
    assert payload == full_payload
