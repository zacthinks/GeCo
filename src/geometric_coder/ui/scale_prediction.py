"""Bounded continuous-Scale prediction pages for the Develop workspace.

The server scores only the visible eligible atomic observations. Scale scores
are centered scalar utilities, never Code probabilities or binary assignments.
"""
from __future__ import annotations

from math import ceil
from typing import Any

import plotly.graph_objects as go


MAX_SCALE_PLOT_PAGE_SIZE = 5000
DEFAULT_SCALE_PLOT_PAGE_SIZE = 1000


def scale_prediction_page(
    project: Any, scale_id: int, *,
    page_number: int = 1,
    page_size: int = DEFAULT_SCALE_PLOT_PAGE_SIZE,
) -> dict[str, Any]:
    """Score one stable-ID page without materializing a corpus-wide geometry."""
    if not 1 <= int(page_size) <= MAX_SCALE_PLOT_PAGE_SIZE:
        raise ValueError("Scale prediction page size must be between 1 and 5000.")
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


def scale_prediction_figure(
    page: dict[str, Any], *, render_token: str | None = None,
) -> go.Figure:
    """Plot fitted positions against epistemic SD, or an honest 1D histogram."""
    rows = page.get("records") or []
    figure = go.Figure()
    available = bool(page.get("uncertainty_available"))
    if rows and available:
        figure.add_trace(go.Scattergl(
            x=[row["score"] for row in rows],
            y=[row["epistemic_sd"] for row in rows],
            customdata=[
                [row["observation_id"], row["unit_id"]] + (
                    [render_token] if render_token is not None else []
                ) for row in rows
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
        # Display identities as selectable points without fabricating
        # posterior uncertainty. The vertical coordinate is page order only.
        figure.add_trace(go.Scattergl(
            x=[row["score"] for row in rows],
            y=list(range(1, len(rows) + 1)),
            customdata=[
                [row["observation_id"], row["unit_id"]] + (
                    [render_token] if render_token is not None else []
                ) for row in rows
            ],
            text=[row["label"] for row in rows],
            mode="markers",
            marker={"size": 9, "opacity": 0.7},
            hovertemplate=(
                "%{text}<br>Scale position: %{x:.3g}"
                "<br>Page order: %{y} (not uncertainty)<extra></extra>"
            ),
        ))
        figure.add_annotation(
            text="Posterior uncertainty unavailable; vertical position is page order",
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
            f"{'sd' if available else 'page-order'}"
        ),
        xaxis_title="Predicted scalar position (centered utility units)",
        yaxis_title=(
            "Positional epistemic uncertainty (posterior SD)"
            if available else "Page order (not uncertainty)"
        ),
    )
    return figure


def _validated_prediction_observation(
    project: Any, selection: Any, page_state: Any, construct_id: Any,
) -> dict[str, Any] | None:
    """Revalidate a stored, page-scoped identity against the live Scale and fit.

    Plotly clickData is deliberately not part of the durable selection contract.
    """
    if not isinstance(selection, dict) or not isinstance(page_state, dict):
        return None
    try:
        scale_id = int(construct_id)
        observation_id = int(selection["observation_id"])
        unit_id = int(selection["unit_id"])
        if any(int(item["scale_id"]) != scale_id for item in (selection, page_state)):
            return None
        if int(selection["fit_id"]) != int(page_state["fit_id"]):
            return None
        if int(selection["page"]) != int(page_state["page"]):
            return None
        if int(selection.get("page_size", DEFAULT_SCALE_PLOT_PAGE_SIZE)) != int(
            page_state.get("page_size", DEFAULT_SCALE_PLOT_PAGE_SIZE)
        ):
            return None
        if observation_id not in page_state["observation_ids"]:
            return None
        if project.construct(scale_id)["type"] != "scale":
            return None
        state = project.scale_model_status(scale_id)
        fit = state.get("fit")
        if state["status"] not in {"current", "stale"} or fit is None:
            return None
        if int(fit["scaler_fit_id"]) != int(selection["fit_id"]):
            return None
        observation = project.observation(observation_id)
        if observation["kind"] != "atomic" or int(observation["unit_id"]) != unit_id:
            return None
        if observation_id not in project.eligible_scale_observation_ids(
            scale_id, [observation_id],
        ):
            return None
        return observation
    except (KeyError, IndexError, TypeError, ValueError, OSError):
        return None


def register_scale_prediction_callbacks(app: Any, project: Any) -> None:
    """Wire bounded Scale scoring and point inspection into the Develop tab."""
    from dash import Input, Output, State, callback_context, no_update
    from geometric_coder.ui.app import (
        _metadata_table, _render_context_units,
    )
    from dash import html, dcc
    import plotly.graph_objects as go
    import uuid
    from geometric_coder.ui.context_expansion import (
        advance_limit, context_key, current_limit, read_context, truncation_notice,
    )

    @app.callback(
        Output("scale-prediction-page", "value", allow_duplicate=True),
        Input("scale-prediction-previous", "n_clicks"),
        Input("scale-prediction-next", "n_clicks"),
        State("scale-prediction-page", "value"),
        State("scale-prediction-page", "max"),
        prevent_initial_call=True,
    )
    def step_scale_prediction_page(
        previous: int | None, following: int | None,
        current: int | None, maximum: int | None,
    ) -> int:
        del previous, following
        page = max(1, int(current or 1))
        total = max(1, int(maximum or 1))
        return (max(1, page - 1) if callback_context.triggered_id
                == "scale-prediction-previous"
                else min(total, page + 1))

    @app.callback(
        Output("scale-prediction-map", "figure"),
        Output("scale-prediction-status", "children"),
        Output("scale-prediction-page-state", "data"),
        Output("scale-prediction-page", "max"),
        Input("focus-construct-dropdown", "value"),
        Input("scale-prediction-page", "value"),
        Input("scale-prediction-refresh", "n_clicks"),
        Input("scale-model-train-status", "children"),
        Input("scale-model-auto-status", "children"),
        Input("scale-model-geometry", "value"),
        Input("scale-prediction-page-size", "value"),
    )
    def render_scale_predictions(
        construct_id: int | None, page: int | None, refresh: int | None,
        train_message: Any, automatic_message: Any, geometry_id: Any,
        page_size: Any = None,
    ) -> tuple[Any, str, Any, int]:
        del refresh, train_message, automatic_message, geometry_id
        # Tab navigation must not score the corpus or replace a valid plot.
        # A client-side resize reconciles WebGL dimensions when Develop returns.
        try:
            construct = project.construct(int(construct_id)) if construct_id is not None else None
            if construct is None or construct["type"] != "scale":
                return go.Figure(), "", None, 1
            size = int(page_size or DEFAULT_SCALE_PLOT_PAGE_SIZE)
            result = scale_prediction_page(
                project, int(construct_id), page_number=int(page or 1),
                page_size=size,
            )
            token = uuid.uuid4().hex
            figure = scale_prediction_figure(result, render_token=token)
            page_state = {
                "scale_id": int(construct_id),
                "fit_id": result.get("fit_id"),
                "page": result["page"],
                "page_size": size,
                "render_token": token,
                "observation_ids": [
                    record["observation_id"] for record in result["records"]
                ],
            } if result.get("fit_id") is not None else None
            return figure, result["message"], page_state, result["pages"]
        except (KeyError, TypeError, ValueError, RuntimeError, OSError) as error:
            return go.Figure(), f"Scale prediction unavailable: {error}", None, 1

    @app.callback(
        Output("scale-prediction-selection", "data"),
        Input("scale-prediction-map", "clickData"),
        Input("scale-prediction-page-state", "data"),
        Input("focus-construct-dropdown", "value"),
        State("scale-prediction-selection", "data"),
        prevent_initial_call=True,
    )
    def select_scale_prediction_point(
        click_data: Any, page_state: Any, construct_id: Any, previous: Any,
    ) -> Any:
        # A figure replacement may emit clickData=None; that is NOT a deselection.
        # Page, construct, eligibility and fit changes can invalidate the selection.
        # The figure callback can briefly publish a null page snapshot during a
        # redraw. A missing snapshot is not a page change: retain the last
        # selection while the same Scale and fitted model remain active.
        if page_state is None and isinstance(previous, dict):
            try:
                scale_id = int(construct_id)
                status = project.scale_model_status(scale_id)
                fit = status.get("fit")
                if (
                    int(previous["scale_id"]) == scale_id
                    and project.construct(scale_id)["type"] == "scale"
                    and status["status"] in {"current", "stale"}
                    and fit is not None
                    and int(fit["scaler_fit_id"]) == int(previous["fit_id"])
                ):
                    return no_update
            except (KeyError, ValueError, TypeError, OSError):
                pass
        valid_previous = _validated_prediction_observation(
            project, previous, page_state, construct_id,
        ) is not None
        if "scale-prediction-map.clickData" in callback_context.triggered_prop_ids and click_data:
            try:
                clicked = click_data["points"][0]["customdata"]
                # Plotly can replay clickData from the previous figure after
                # a refresh. Require an identity from the exact rendered page.
                if not (
                    isinstance(page_state, dict)
                    and isinstance(clicked, (list, tuple))
                    and len(clicked) == 3
                    and clicked[2] == page_state.get("render_token")
                ):
                    return no_update
                observation_id, unit_id = int(clicked[0]), int(clicked[1])
                candidate = {
                    "scale_id": int(construct_id),
                    "fit_id": int(page_state["fit_id"]),
                    "page": int(page_state["page"]),
                    "page_size": int(page_state["page_size"]),
                    "observation_id": observation_id,
                    "unit_id": unit_id,
                }
            except (KeyError, IndexError, TypeError, ValueError):
                candidate = None
            if _validated_prediction_observation(
                project, candidate, page_state, construct_id,
            ) is not None:
                return no_update if candidate == previous else candidate
        if previous is not None and not valid_previous:
            return None
        return no_update

    @app.callback(
        Output("scale-prediction-context-expansion-store", "data"),
        Input("scale-prediction-context-expand", "n_clicks"),
        State("scale-prediction-selection", "data"),
        State("scale-context-level", "value"),
        State("scale-context-window", "value"),
        State("scale-prediction-context-expansion-store", "data"),
        prevent_initial_call=True,
    )
    def expand_scale_prediction_context(clicks, selection, level, window, state):
        del clicks
        focal = selection.get("unit_id") if isinstance(selection, dict) else None
        return advance_limit(
            state, context_key(focal, level, window),
            expand=True,
        )

    @app.callback(
        Output("scale-prediction-preview", "children"),
        Input("scale-prediction-selection", "data"),
        Input("scale-prediction-page-state", "data"),
        Input("focus-construct-dropdown", "value"),
        Input("scale-context-level", "value"),
        Input("scale-context-window", "value"),
        Input("scale-prediction-context-expansion-store", "data"),
        prevent_initial_call=True,
    )
    def inspect_scale_prediction_point(
        selection: Any, page_state: Any, construct_id: Any,
        level: Any, window: Any, expansion: Any = None,
    ) -> Any:
        empty = html.P("Select a plotted observation to inspect it.")
        if page_state is None and isinstance(selection, dict):
            # Preserve the currently displayed inspector through a temporary
            # figure/page-state gap. Genuine fit or Construct loss clears it.
            try:
                scale_id = int(construct_id)
                state = project.scale_model_status(scale_id)
                fit = state.get("fit")
                if (
                    int(selection["scale_id"]) == scale_id
                    and project.construct(scale_id)["type"] == "scale"
                    and state["status"] in {"current", "stale"}
                    and fit is not None
                    and int(fit["scaler_fit_id"]) == int(selection["fit_id"])
                ):
                    return no_update
            except (KeyError, TypeError, ValueError, OSError):
                pass
        observation = _validated_prediction_observation(
            project, selection, page_state, construct_id,
        )
        if observation is None:
            return empty
        try:
            observation_id = int(selection["observation_id"])
            unit_id = int(selection["unit_id"])
            limit = current_limit(expansion, context_key(unit_id, level, window))
            context, truncated = read_context(
                project, unit_id,
                level=int(level if level is not None else 0),
                window=max(1, min(100, int(window or 1))) if str(window or "1").isdigit() else 1,
                limit=limit,
            )
            reading = _render_context_units(
                html, dcc, context, focal_unit_id=unit_id,
                key_columns=list(project.metadata["key_columns"]),
                selected_unit_ids=[unit_id], checkbox_type="scale-prediction-context",
                span_selection_enabled=False,
            )
            label = " › ".join(f"{key}: {value}" for key, value in observation["user_key"].items())
            return html.Div([
                html.H4("Selected observation"),
                html.Strong(label or f"Observation {observation_id}"),
                html.P(str(observation["text"])),
                html.Details([html.Summary("Context"), html.Div([*([truncation_notice(html, len(context), truncated, limit)] if truncated else []), *reading])]),
                html.Details([html.Summary("Metadata"), _metadata_table(html, observation["metadata"])]),
            ])
        except (KeyError, IndexError, TypeError, ValueError, OSError):
            return empty
