"""Focal Topic Model inspection and local label controls embedded in Explore.

Scientific values and component identities come from provider-backed keyed artifacts.
Model and topic aliases are the only GeCo-local edits; no separate topic tab.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import sparse

from geometric_coder.exceptions import ConfigurationError
from geometric_coder.topic_artifacts import TopicModelRows

if TYPE_CHECKING:
    from geometric_coder.project import GeometricCoder


def topic_name(component_id: str, aliases: dict[str, str]) -> str:
    """Show the original immutable feature name even after a local rename."""
    alias = aliases.get(component_id)
    return f"{alias} ({component_id})" if alias else component_id


def _dense_row(values: Any, index: int) -> np.ndarray:
    row = values[index]
    if sparse.issparse(row):
        row = row.toarray()
    return np.asarray(row, dtype=float).reshape(-1)


def topic_inspection(
    rows: TopicModelRows, *, selected_component: str | None = None
) -> dict[str, Any]:
    """Summarize a single requested observation, preserving coverage semantics.

    Callers request exactly one original atomic position. An empty matched set
    means outside model coverage, not a row of zero memberships.
    """
    if len(rows.matched_positions) > 1 or rows.matched_positions not in ((), (0,)):
        raise ConfigurationError("Topic inspection requires one requested observation.")
    aliases = rows.topic_labels
    component = (
        selected_component if selected_component in rows.topic_ids
        else rows.topic_ids[0]
    )
    weights = _dense_row(rows.memberships, 0) if rows.matched_positions else None
    topics = [
        {
            "id": topic_id,
            "name": topic_name(topic_id, aliases),
            "weight": float(weights[i]) if weights is not None else None,
        }
        for i, topic_id in enumerate(rows.topic_ids)
    ]
    if weights is None:
        coverage, dominant, max_value = "outside", None, None
    else:
        winner = rows.dominant.iloc[0]["max_feature"]
        if pd.isna(winner):
            coverage, dominant, max_value = "unassigned", None, None
        else:
            coverage = "assigned"
            dominant = str(winner)
            max_value = float(rows.dominant.iloc[0]["max_value"])

    descriptors: list[dict[str, Any]] | None = None
    if rows.descriptions is not None:
        index = rows.topic_ids.index(component)
        scores = _dense_row(rows.descriptions.values, index)
        indices = np.argsort(-scores, kind="stable")[:10]
        descriptors = [
            {"name": str(rows.descriptions.columns[i]), "weight": float(scores[i])}
            for i in indices
        ]
    return {
        "topics": topics,
        "coverage": coverage,
        "dominant": dominant,
        "max_value": max_value,
        "selected_component": component,
        "descriptors": descriptors,
    }


def register_topic_details_callbacks(
    app: Any,
    project: GeometricCoder,
    *,
    html: Any,
    Input: Any,
    Output: Any,
    State: Any,
    callback_context: Any,
    no_update: Any,
    units: list[dict[str, Any]],
) -> None:
    """Wire the panel without changing the Explore/Develop/Apply callbacks."""
    unit_positions = {int(unit["unit_id"]): i for i, unit in enumerate(units)}
    by_unit = {int(unit["unit_id"]): unit for unit in units}

    @app.callback(
        Output("topic-model-new-name", "value"),
        Input("topic-model-selector", "value"),
        Input("topic-model-refresh-store", "data"),
    )
    def model_name(model_id: int | None, _revision: int) -> str:
        return str(project.topic_model(model_id)["name"]) if model_id is not None else ""

    @app.callback(
        Output("topic-component-new-label", "value"),
        Input("topic-model-selector", "value"),
        Input("topic-component-selector", "value"),
        Input("topic-model-refresh-store", "data"),
    )
    def topic_label(model_id: int | None, component: str | None, _revision: int) -> str:
        if model_id is None or component is None:
            return ""
        return str(project.topic_model(model_id)["topic_labels"].get(component, ""))

    @app.callback(
        Output("topic-focal-key", "children"),
        Output("topic-memberships", "children"),
        Output("topic-descriptors", "children"),
        Input("topic-model-selector", "value"),
        Input("explore-focal-unit-store", "data"),
        Input("topic-component-selector", "value"),
        Input("topic-model-refresh-store", "data"),
    )
    def inspect(
        model_id: int | None,
        focal_unit_id: int | None,
        component: str | None,
        _revision: int,
    ) -> tuple[Any, Any, Any]:
        if model_id is None:
            return "", "No Topic Models registered in this project.", ""
        if not units:
            return "", "No atomic observations available.", ""
        position = unit_positions.get(int(focal_unit_id)) if focal_unit_id is not None else None
        if focal_unit_id is not None and position is None:
            return "", "The focal observation no longer exists.", ""
        try:
            # Read a single key. When no focal exists, sample one key only to
            # obtain the immutable topic axis and optional descriptors.
            rows = project.topic_model_rows(
                model_id,
                positions=[position if position is not None else 0],
                include_description=True,
            )
            summary = topic_inspection(rows, selected_component=component)
        except Exception as exc:
            # Missing external artifacts and provider failures must be visible,
            # never silently treated as uncovered/unassigned observations.
            return "", f"Unable to read registered Topic Model artifacts: {exc}", ""

        topic_id = summary["selected_component"]
        if position is None:
            focal_title: Any = "No focal observation selected."
            membership_display: Any = "Choose a point in Explore to inspect memberships."
        else:
            unit = by_unit[int(focal_unit_id)]
            focal_title = html.Strong(
                " › ".join(f"{key}: {value}" for key, value in unit["user_key"].items())
            )
            coverage = summary["coverage"]
            if coverage == "outside":
                membership_display = html.P(
                    "Outside model coverage: this observation has no fitted membership row."
                )
            else:
                if coverage == "unassigned":
                    status = "Covered but unassigned (all-zero/noise membership)."
                else:
                    status = (
                        "Dominant: "
                        + topic_name(summary["dominant"], rows.topic_labels)
                        + f" (max weight {summary['max_value']:.4g})."
                    )
                membership_display = html.Div(
                    [
                        html.P(status),
                        html.P(
                            "Membership weights are raw model values, not necessarily "
                            "calibrated probabilities."
                        ),
                        html.Table(
                            [
                                html.Thead(html.Tr([html.Th("Topic"), html.Th("Weight")])),
                                html.Tbody([
                                    html.Tr([
                                        html.Td(record["name"]),
                                        html.Td(f"{record['weight']:.4g}"),
                                    ])
                                    for record in summary["topics"]
                                ]),
                            ],
                            className="metadata-table",
                        ),
                    ]
                )

        if summary["descriptors"] is None:
            desc_display: Any = html.P("No topic descriptions registered for this model.")
        else:
            desc_display = html.Div([
                html.H4(f"Top features for {topic_name(topic_id, rows.topic_labels)}"),
                html.P("Original descriptor weights (not inferred topic names)."),
                html.Ol([
                    html.Li(f"{item['name']} — {item['weight']:.4g}")
                    for item in summary["descriptors"]
                ]),
            ])
        return focal_title, membership_display, desc_display

    @app.callback(
        Output("topic-model-refresh-store", "data"),
        Output("topic-action-status", "children"),
        Input("topic-model-rename-save", "n_clicks"),
        Input("topic-label-save", "n_clicks"),
        State("topic-model-selector", "value"),
        State("topic-model-new-name", "value"),
        State("topic-component-selector", "value"),
        State("topic-component-new-label", "value"),
        State("topic-model-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def rename(
        _model_clicks: int | None,
        _topic_clicks: int | None,
        model_id: int | None,
        name: str | None,
        component: str | None,
        label: str | None,
        revision: int | None,
    ) -> tuple[Any, str]:
        action = callback_context.triggered_id
        if model_id is None:
            return no_update, "Select a registered model first."
        try:
            if action == "topic-model-rename-save":
                project.rename_topic_model(int(model_id), name=name or "")
                message = "Model name saved locally in GeCo."
            elif action == "topic-label-save":
                if component is None or not units:
                    return no_update, "Select a topic first."
                # R1 persistence permits arbitrary alias keys; verify against
                # the source feature axis so the UI cannot create phantom topics.
                actual = project.topic_model_rows(int(model_id), positions=[0]).topic_ids
                if component not in actual:
                    return no_update, "This topic is not present in the model."
                project.rename_topic(
                    int(model_id), component_id=component, name=label or ""
                )
                message = "Topic label saved locally; TeAL artifacts were not changed."
            else:
                return no_update, ""
        except (ConfigurationError, RuntimeError, ValueError, KeyError) as exc:
            return no_update, str(exc)
        return int(revision or 0) + 1, message
