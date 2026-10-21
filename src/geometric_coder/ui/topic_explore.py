"""Topic-based Explore point coloring from validated, key-aligned provider rows.

This is a view model, not a model fitter or a new model registry. It keeps the
difference between no coverage and an observed all-zero/noise assignment.
"""

from __future__ import annotations

import colorsys
import hashlib
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import sparse

from geometric_coder.topic_artifacts import TopicModelRows

ColorMode = Literal["plain", "semantic", "dominant", "intensity"]

UNCOVERED_COLOR = "#bcc5cf"
UNASSIGNED_COLOR = "#626b76"


@dataclass(frozen=True)
class TopicPointColors:
    colors: tuple[str, ...]
    legend: tuple[tuple[str, str], ...]
    status: str


def normalize_color_mode(value: Any) -> ColorMode:
    """Do not restore unknown session modes from newer or broken versions."""
    return value if value in {"plain", "semantic", "dominant", "intensity"} else "plain"


def component_label(component: str, labels: dict[str, str]) -> str:
    """Keep TeAL's immutable component identity visible through local aliases."""
    alias = labels.get(component)
    return f"{alias} ({component})" if alias else component


def component_color(component: str) -> str:
    """Stable categorical color by immutable feature ID, independent of name/order."""
    digest = hashlib.sha256(component.encode("utf-8")).digest()
    hue = int.from_bytes(digest[:4], "big") / 2**32
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.65, 0.78)
    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


def _gradient(value: float) -> str:
    """Light blue to dark indigo, normalized over the selected page's values."""
    left = (222, 235, 252)
    right = (36, 65, 155)
    bounded = float(np.clip(value, 0.0, 1.0))
    rgb = tuple(round(a + (b - a) * bounded) for a, b in zip(left, right, strict=True))
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def topic_point_colors(
    rows: TopicModelRows,
    *,
    mode: Literal["dominant", "intensity"],
    selected_topic: str | None,
    requested_count: int,
) -> TopicPointColors:
    """Produce one color for each requested GeCo key, without densifying N x K.

    matched_positions indexes requested keys, not source-artifact physical rows.
    The TeAL ArgMax result decides assigned versus unassigned status. Sparse
    membership extraction reads only the selected column for intensity mode.
    """
    if requested_count < 0:
        raise ValueError("requested_count must be nonnegative")
    if mode not in {"dominant", "intensity"}:
        raise ValueError("Topic coloring requires dominant or intensity mode.")
    if len(rows.matched_positions) != rows.memberships.shape[0]:
        raise ValueError("Matched topic positions and membership rows disagree.")
    if len(rows.dominant) != len(rows.matched_positions):
        raise ValueError("Matched ArgMax positions and membership rows disagree.")
    if mode == "intensity" and selected_topic not in rows.topic_ids:
        raise ValueError("Select a valid topic before coloring by intensity.")

    colors = [UNCOVERED_COLOR] * requested_count
    observed: set[str] = set()
    assigned_indices: list[int] = []
    assigned_values: list[float] = []

    column_values: np.ndarray | None = None
    if mode == "intensity":
        component_index = rows.topic_ids.index(str(selected_topic))
        column = rows.memberships[:, component_index]
        if sparse.issparse(column):
            column = column.toarray()
        column_values = np.asarray(column, dtype=float).reshape(-1)

    for source_row, requested_index in enumerate(rows.matched_positions):
        if requested_index < 0 or requested_index >= requested_count:
            raise ValueError("Topic key position is outside the requested page.")
        winner = rows.dominant.iloc[source_row]["max_feature"]
        if pd.isna(winner):
            colors[requested_index] = UNASSIGNED_COLOR
            continue
        component = str(winner)
        if component not in rows.topic_ids:
            raise ValueError("ArgMax references an unknown topic component.")
        if mode == "dominant":
            colors[requested_index] = component_color(component)
            observed.add(component)
        else:
            assert column_values is not None
            assigned_indices.append(int(requested_index))
            assigned_values.append(float(column_values[source_row]))

    if mode == "intensity":
        if assigned_values:
            floor = min(0.0, min(assigned_values))
            ceiling = max(assigned_values)
            span = ceiling - floor
            for position, value in zip(assigned_indices, assigned_values, strict=True):
                colors[position] = _gradient((value - floor) / span if span > 0 else 0)
            scale_label = f"Selected topic: page range {floor:.3g}–{ceiling:.3g} (raw weight)"
        else:
            scale_label = "No assigned topic memberships on this page."
        legend = (
            ("#deeafc", "Low selected-topic weight"),
            ("#24419b", "High selected-topic weight"),
            (UNASSIGNED_COLOR, "Covered, unassigned/noise"),
            (UNCOVERED_COLOR, "Outside Topic Model coverage"),
        )
        return TopicPointColors(tuple(colors), legend, scale_label)

    legend = tuple(
        (component_color(component), component_label(component, rows.topic_labels))
        for component in rows.topic_ids
        if component in observed
    ) + (
        (UNASSIGNED_COLOR, "Covered, unassigned/noise"),
        (UNCOVERED_COLOR, "Outside Topic Model coverage"),
    )
    return TopicPointColors(
        tuple(colors), legend, f"{len(observed)} dominant topics shown on this page."
    )
