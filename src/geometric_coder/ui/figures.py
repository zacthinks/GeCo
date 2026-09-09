"""Plotly figure construction for GeCo workspaces."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Any

import numpy as np
import plotly.graph_objects as go


EXPLORE_FILTERED_TRACE_INDEX = 0
EXPLORE_ELIGIBLE_TRACE_INDEX = 1
EXPLORE_VISITED_TRACE_INDEX = 2
EXPLORE_FOCAL_TRACE_INDEX = 3
EXPLORE_TRACE_NAMES = (
    "Filtered",
    "Eligible",
    "Visited overlay",
    "Focal overlay",
)


def build_geometry_figure(
    *,
    coordinates: np.ndarray,
    units: Sequence[dict[str, Any]],
    eligible: Sequence[bool],
    seen_unit_ids: Collection[int] = (),
    focal_unit_id: int | None = None,
    semantic_scores: Sequence[float] | None = None,
    geometry_name: str = "Geometry",
    view_name: str = "View",
    page_label: str | None = None,
    uirevision: str | None = None,
) -> go.Figure:
    """Build a click-responsive Explore map with stable trace structure.

    The base geometry is deliberately independent of navigation state. Point fill
    is reserved for filtering and semantic similarity. Two fixed overlay traces
    carry visited and focal outlines. Keeping those overlay traces present even
    when empty lets Dash update navigation highlighting with a tiny partial
    figure update instead of rebuilding and retransmitting the full map.
    """
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape (n, 2)")
    if coordinates.shape[0] != len(units):
        raise ValueError("coordinates and units must contain the same number of rows")
    eligible_array = np.asarray(eligible, dtype=bool)
    if eligible_array.shape != (len(units),):
        raise ValueError("eligible must contain one value per unit")

    unit_ids = np.asarray([int(unit["unit_id"]) for unit in units], dtype=int)
    seen = (
        np.isin(unit_ids, np.fromiter(seen_unit_ids, dtype=int))
        if seen_unit_ids
        else np.zeros(len(units), dtype=bool)
    )
    hover = [_hover_key(unit) for unit in units]
    semantic = None if semantic_scores is None else np.asarray(semantic_scores, dtype=float)
    if semantic is not None and semantic.shape != (len(units),):
        raise ValueError("semantic_scores must contain one value per unit")
    focal_mask = (
        unit_ids == int(focal_unit_id)
        if focal_unit_id is not None
        else np.zeros(len(units), dtype=bool)
    )

    figure = go.Figure()

    def add_base_points(
        positions: np.ndarray,
        *,
        name: str,
        fill: str | np.ndarray,
        opacity: float,
        outline: str,
        outline_width: float,
    ) -> None:
        marker: dict[str, Any] = {
            "size": 8,
            "opacity": opacity,
            "symbol": "circle",
            "color": fill.tolist() if isinstance(fill, np.ndarray) else fill,
            "line": {"width": outline_width, "color": outline},
        }
        figure.add_trace(
            go.Scattergl(
                x=coordinates[positions, 0].tolist(),
                y=coordinates[positions, 1].tolist(),
                mode="markers",
                name=name,
                customdata=unit_ids[positions].tolist(),
                hovertext=[hover[int(index)] for index in positions],
                hovertemplate="%{hovertext}<extra></extra>",
                marker=marker,
            )
        )

    filtered_positions = np.flatnonzero(~eligible_array)
    active_positions = np.flatnonzero(eligible_array)
    add_base_points(
        filtered_positions,
        name=EXPLORE_TRACE_NAMES[EXPLORE_FILTERED_TRACE_INDEX],
        fill="#d5d8dc",
        opacity=0.38,
        outline="#aeb6bf",
        outline_width=0.7,
    )
    active_fill: str | np.ndarray
    if semantic is None:
        active_fill = "#111111"
    else:
        active_fill = semantic[active_positions]
    add_base_points(
        active_positions,
        name=EXPLORE_TRACE_NAMES[EXPLORE_ELIGIBLE_TRACE_INDEX],
        fill=active_fill,
        opacity=0.9,
        outline="#111111",
        outline_width=0.55,
    )

    def add_outline_overlay(
        positions: np.ndarray,
        *,
        name: str,
        outline: str,
        outline_width: float,
        size: float,
    ) -> None:
        figure.add_trace(
            go.Scattergl(
                x=coordinates[positions, 0].tolist(),
                y=coordinates[positions, 1].tolist(),
                mode="markers",
                name=name,
                customdata=unit_ids[positions].tolist(),
                hovertext=[hover[int(index)] for index in positions],
                hovertemplate="%{hovertext}<extra></extra>",
                marker={
                    "size": size,
                    "opacity": 1.0,
                    "symbol": "circle",
                    "color": "rgba(0,0,0,0)",
                    "line": {"width": outline_width, "color": outline},
                },
            )
        )

    add_outline_overlay(
        np.flatnonzero(seen),
        name=EXPLORE_TRACE_NAMES[EXPLORE_VISITED_TRACE_INDEX],
        outline="#d62728",
        outline_width=2.2,
        size=9.5,
    )
    add_outline_overlay(
        np.flatnonzero(focal_mask),
        name=EXPLORE_TRACE_NAMES[EXPLORE_FOCAL_TRACE_INDEX],
        outline="#178547",
        outline_width=2.5,
        size=11.5,
    )

    # Apply one shared semantic colorscale and show one conventional Plotly
    # colorbar on the base eligible trace. Navigation overlays stay colorless.
    if semantic is not None:
        finite = semantic[np.isfinite(semantic)]
        color_min = float(np.min(finite)) if finite.size else 0.0
        color_max = float(np.max(finite)) if finite.size else 1.0
        if np.isclose(color_min, color_max):
            color_max = color_min + 1e-9
        trace = figure.data[EXPLORE_ELIGIBLE_TRACE_INDEX]
        trace.marker.colorscale = [
            [0.0, "#2457c5"],
            [1.0, "#f2b134"],
        ]
        trace.marker.cmin = color_min
        trace.marker.cmax = color_max
        trace.marker.showscale = bool(active_positions.size)
        if active_positions.size:
            color_mid = (color_min + color_max) / 2.0
            trace.marker.colorbar = {
                "title": {"text": "Cosine similarity", "side": "right"},
                "tickmode": "array",
                "tickvals": [color_min, color_mid, color_max],
                "ticktext": [
                    f"{color_min:.3f}",
                    f"{color_mid:.3f}",
                    f"{color_max:.3f}",
                ],
                "thickness": 14,
                "len": 0.72,
                "x": 1.015,
                "xpad": 8,
            }

    title = f"{geometry_name} · {view_name}"
    if page_label:
        title = f"{title} · {page_label}"
    figure.update_layout(
        title={"text": title, "x": 0.01},
        template="plotly_white",
        margin={"l": 20, "r": 92 if semantic is not None else 20, "t": 56, "b": 20},
        hovermode="closest",
        hoverdistance=8,
        clickmode="event",
        dragmode="zoom",
        uirevision=uirevision or f"{geometry_name}:{view_name}:{page_label or ''}",
        showlegend=False,
        xaxis={"visible": False, "showgrid": False, "zeroline": False},
        yaxis={"visible": False, "showgrid": False, "zeroline": False, "scaleanchor": "x"},
    )
    return figure


def _hover_key(unit: dict[str, Any]) -> str:
    """Return a compact, non-textual tooltip from the user-provided key."""
    return " · ".join(f"{name}: {value}" for name, value in unit["user_key"].items())
