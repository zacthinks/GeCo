"""Bounded continuous-Scale prediction pages for the Develop workspace.

The server scores only the visible eligible atomic observations. Scale scores
are centered scalar utilities, never Code probabilities or binary assignments.
"""
from __future__ import annotations

from math import ceil
from typing import Any

import plotly.graph_objects as go


MAX_SCALE_PLOT_PAGE_SIZE = 500
DEFAULT_SCALE_PLOT_PAGE_SIZE = 128


def scale_prediction_page(
    project: Any, scale_id: int, *,
    page_number: int = 1,
    page_size: int = DEFAULT_SCALE_PLOT_PAGE_SIZE,
) -> dict[str, Any]:
    """Score one stable-ID page without materializing a corpus-wide geometry."""
    if not 1 <= int(page_size) <= MAX_SCALE_PLOT_PAGE_SIZE:
        raise ValueError("Scale prediction page size must be between 1 and 500.")
    size = int(page_size)
    status = project.scale_model_status(int(scale_id))
    state = str(status["status"])
    if state not in {"current", "stale"} or status.get("fit") is None:
        reason = str(status.get("reason") or state.replace("_", " "))
        return {
            "status": state, "message": (
                "No usable fitted Scale for prediction: " + reason +
                ". Save judgments and train the Scale first."
            ),
            "records": [], "page": 1, "pages": 1, "eligible_count": 0,
            "fit_id": None,
        }

    eligibility = project.scale_eligibility(int(scale_id))["eligibility"]
    ids = project.database.acquisition_eligible_atomic_ids(
        "scale", int(scale_id), eligibility_spec=eligibility,
    )
    count = len(ids)
    pages = max(1, ceil(count / size))
    current_page = min(max(1, int(page_number)), pages)
    selected = [
        int(value) for value in
        ids[(current_page - 1) * size:current_page * size]
    ]
    fit_id = int(status["fit"]["scaler_fit_id"])
    if not selected:
        return {
            "status": state, "message": "No eligible atomic observations.",
            "records": [], "page": current_page, "pages": pages,
            "eligible_count": count, "fit_id": fit_id,
        }

    scored = project.scale_scores(
        int(scale_id), selected, allow_stale=True, include_uncertainty=True,
    )
    if [int(row["observation_id"]) for row in scored] != selected:
        raise ValueError("Scale score results do not match the requested observation IDs.")
    records = []
    for row in scored:
        if int(row["scaler_fit_id"]) != fit_id:
            raise ValueError("Scale score results came from an unexpected fit.")
        observation_id = int(row["observation_id"])
        observation = project.observation(observation_id)
        if observation["kind"] != "atomic":
            raise ValueError("Scale prediction pages must contain atomic observations.")
        source_key = " / ".join(
            f"{key}: {value}" for key, value in observation["user_key"].items()
        )
        records.append({
            "observation_id": observation_id,
            "unit_id": int(observation["unit_id"]),
            "label": source_key or f"Observation {observation_id}",
            "score": float(row["score"]),
            "epistemic_sd": row["epistemic_sd"],
        })
    available = all(row["epistemic_sd"] is not None for row in records)
    message = (
        f"Page {current_page:,} of {pages:,} · "
        f"{len(records):,} of {count:,} eligible observations shown · "
        f"{'Stale' if state == 'stale' else 'Current'} fit {fit_id}."
    )
    if state == "stale":
        message += " Scores use the last saved fit; newer evidence has not been fitted."
    if not available:
        message += " Posterior positional uncertainty unavailable; showing a score distribution."
    return {
        "status": state, "message": message, "records": records,
        "page": current_page, "pages": pages, "eligible_count": count,
        "fit_id": fit_id, "uncertainty_available": available,
    }


def scale_prediction_figure(page: dict[str, Any]) -> go.Figure:
    """Plot fitted positions against epistemic SD, or an honest 1D histogram."""
    rows = page.get("records") or []
    figure = go.Figure()
    available = bool(page.get("uncertainty_available"))
    if rows and available:
        figure.add_trace(go.Scattergl(
            x=[row["score"] for row in rows],
            y=[row["epistemic_sd"] for row in rows],
            customdata=[
                [row["observation_id"], row["unit_id"]] for row in rows
            ],
            text=[row["label"] for row in rows],
            mode="markers",
            marker={"size": 9, "opacity": 0.7},
            hovertemplate=(
                "%{text}<br>Scale position: %{x:.3g}"
                "<br>Epistemic SD: %{y:.3g}<extra></extra>"
            ),
        ))
    elif rows:
        figure.add_trace(go.Histogram(
            x=[row["score"] for row in rows],
            nbinsx=min(30, max(5, len(rows) // 3)),
            hovertemplate="Scale position: %{x:.3g}<br>Count: %{y}<extra></extra>",
        ))
        figure.add_annotation(
            text="Positional uncertainty unavailable; histogram of sampled page only",
            xref="paper", yref="paper", x=0.5, y=1.08,
            showarrow=False, font={"size": 11},
        )
    figure.update_layout(
        template="plotly_white",
        height=380,
        margin={"l": 65, "r": 25, "t": 52, "b": 72},
        showlegend=False,
        clickmode="event",
        uirevision=(
            f"scale-prediction:{page.get('fit_id')}:{page.get('page')}:"
            f"{'sd' if available else 'histogram'}"
        ),
        xaxis_title="Predicted scalar position (centered utility units)",
        yaxis_title=(
            "Positional epistemic uncertainty (posterior SD)"
            if available else "Observations on this page"
        ),
    )
    return figure
