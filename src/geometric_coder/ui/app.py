"""Dash application for GeCo's Explore, Develop, Apply/Review, Codes, and Memos workspaces."""

from __future__ import annotations

from pathlib import Path
from datetime import UTC, date, datetime
import json
import re
import uuid
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, quote, unquote

import numpy as np
import plotly.graph_objects as go

from geometric_coder.classifiers import (
    SUPPORTED_CLASSIFIER_ALGORITHMS,
    classifier_algorithm_label,
    hyperparameter_summary,
)
from geometric_coder.exceptions import ConfigurationError
from geometric_coder.explore import (
    InvalidRegexError,
    combine_filters,
    deterministic_page,
    deterministic_page_with_inclusions,
    recommend_page_unit,
)
from geometric_coder.ui.figures import (
    EXPLORE_ELIGIBLE_TRACE_INDEX,
    EXPLORE_FILTERED_TRACE_INDEX,
    EXPLORE_FOCAL_TRACE_INDEX,
    EXPLORE_TRACE_NAMES,
    EXPLORE_VISITED_TRACE_INDEX,
    build_geometry_figure,
)

if TYPE_CHECKING:
    from geometric_coder.project import GeometricCoder


GECO_DASH_THEME: dict[str, str] = {
    "--Dash-Stroke-Strong": "#65766b",
    "--Dash-Stroke-Weak": "#d7e2da",
    "--Dash-Fill-Interactive-Strong": "#2f744a",
    "--Dash-Fill-Interactive-Weak": "#e7f1ea",
    "--Dash-Fill-Inverse-Strong": "#ffffff",
    "--Dash-Text-Primary": "#17251c",
    "--Dash-Text-Strong": "#102a19",
    "--Dash-Text-Weak": "#607067",
    "--Dash-Text-Disabled": "#9aa69e",
    "--Dash-Fill-Primary-Hover": "#eef6f0",
    "--Dash-Fill-Primary-Active": "#dcece1",
    "--Dash-Fill-Disabled": "#eef1ef",
    "--Dash-Shading-Strong": "rgba(16, 42, 25, 0.24)",
    "--Dash-Shading-Weak": "rgba(16, 42, 25, 0.12)",
}

GECO_AG_GRID_THEME: dict[str, str] = {
    "function": """themeQuartz.withParams({
        accentColor: '#2f744a',
        backgroundColor: '#ffffff',
        foregroundColor: '#17251c',
        borderColor: '#d7e2da',
        headerBackgroundColor: '#eef6f0',
        headerTextColor: '#173a27',
        oddRowBackgroundColor: '#fafcfb',
        selectedRowBackgroundColor: 'rgba(47, 116, 74, 0.12)',
        inputBackgroundColor: '#ffffff',
        inputBorder: { color: '#a5c8af', style: 'solid', width: 1 },
        inputTextColor: '#17251c',
        menuBackgroundColor: '#ffffff',
        menuTextColor: '#17251c',
        borderRadius: 6,
        wrapperBorderRadius: 8,
        fontFamily: 'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif',
        fontSize: 13,
        spacing: 6,
        rowVerticalPaddingScale: 0.55,
        headerVerticalPaddingScale: 0.75,
        listItemHeight: 30
    })"""
}

APPLY_DECISION_LABELS: dict[str, str] = {
    "pending": "Pending",
    "accept": "Accept proposal",
    "positive": "Positive",
    "negative": "Negative",
    "unsure": "Unsure",
    "unreviewed": "Leave unreviewed",
}
APPLY_DECISION_VALUES: dict[str, str] = {
    label: value for value, label in APPLY_DECISION_LABELS.items()
}

COMMITTEE_AGGREGATION_LABELS: dict[str, str] = {
    "mean": "Mean probability",
    "median": "Median probability (legacy)",
    "minimum": "Minimum probability",
    "maximum": "Maximum probability",
    "harmonic_mean": "Harmonic mean",
    "geometric_mean": "Geometric mean",
    "logistic_stack": "Logistic stacking",
}


def create_app(project: GeometricCoder) -> Any:
    """Create the local GeCo Dash application."""
    try:
        from dash import (
            ALL, Dash, Input, Output, Patch, State, callback_context, dash_table, dcc, html, no_update, set_props,
        )
        import dash_ag_grid as dag
    except ImportError as error:
        raise ConfigurationError(
            "The GeCo web interface is part of the standard installation, but a core "
            "UI dependency could not be imported. Reinstall geometric-coder with its "
            "default dependencies."
        ) from error

    metadata = project.metadata
    units = project.units()
    unit_ids = [int(unit["unit_id"]) for unit in units]
    unit_position_by_id = {unit_id: position for position, unit_id in enumerate(unit_ids)}
    observation_id_by_unit_id = {
        int(unit["unit_id"]): int(unit["observation_id"]) for unit in units
    }
    texts = [str(unit["text"]) for unit in units]
    metadata_columns = [str(value) for value in metadata.get("metadata_columns", [])]
    metadata_profiles = _build_metadata_profiles(units, metadata_columns)
    saved_filter_sets_path = project.project_dir / "saved_filter_sets.json"
    geometries = project.geometries(public_only=True)
    geometry_by_id = {int(record["geometry_id"]): record for record in geometries}
    geometry_id_by_name = {str(record["name"]): int(record["geometry_id"]) for record in geometries}
    views = project.views()
    view_by_id = {int(record["view_id"]): record for record in views}
    views_by_geometry: dict[int, list[dict[str, Any]]] = {}
    for view in views:
        views_by_geometry.setdefault(int(view["geometry_id"]), []).append(view)
    hierarchy = project.database.hierarchy_levels()
    hierarchy_summary = project.hierarchy_summary()
    codes = project.codes()
    classifier_specs = project.ensure_default_classifier_specs()
    prediction_geometry_cache: dict[str, dict[str, Any]] = {}

    sessions = project.sessions()
    initial_session = sessions[0]
    initial_state = dict(initial_session["state"])
    initial_geometry_id = _valid_geometry_id(initial_state.get("geometry_id"), geometries)
    initial_view_id = _valid_view_id(
        initial_state.get("view_id"),
        views_by_geometry.get(initial_geometry_id, []),
    )
    initial_semantic_query = str(initial_state.get("semantic_query", ""))
    initial_semantic_scores: dict[str, list[float]] = {}
    if initial_semantic_query:
        initial_semantic_scores = project.semantic_search(initial_semantic_query)
    initial_history = _normalize_navigation_history(
        initial_state.get("navigation_history"),
        focal_unit_id=initial_state.get("focal_unit_id"),
    )
    initial_explore_code_palette = _normalize_explore_code_palette(
        initial_state.get("explore_code_palette"),
        {int(code["code_id"]) for code in codes},
    )
    initial_seen_count = len(
        project.database.seen_unit_ids(int(initial_session["session_id"]))
    )
    initial_workspace = str(initial_state.get("workspace", "explore"))
    initial_focal_unit_id = initial_state.get("focal_unit_id")

    assets_folder = Path(__file__).parent / "assets"
    app = Dash(
        __name__,
        title="GeCo",
        update_title=None,
        assets_folder=str(assets_folder),
        suppress_callback_exceptions=True,
    )
    app.index_string = _geco_index_string()

    app.layout = html.Div(
        [
            dcc.Loading(
                id="global-working-indicator",
                delay_show=450,
                delay_hide=150,
                # Do not cover the interface for every lightweight Dash callback.
                # These targets correspond to operations that can legitimately take
                # long enough to need explicit feedback.
                target_components={
                    "semantic-scores-store": "data",
                    "view-refresh-store": "data",
                    "focus-refresh-store": "data",
                    "focus-model-refresh-store": "data",
                    "classifier-spec-refresh-store": "data",
                    "prediction-geometry-refresh-store": "data",
                    "teaching-example-refresh-store": "data",
                    "apply-refresh-store": "data",
                },
                # Dash's fullscreen prop applies only to built-in spinners.  The
                # custom GeCo indicator therefore owns its viewport overlay.
                parent_className="geco-working-loading-root",
                parent_style={"position": "static", "zIndex": "auto"},
                overlay_style={"visibility": "visible"},
                custom_spinner=html.Div(
                    html.Div(
                        [
                            html.Div(
                                html.Div(
                                    "🦎",
                                    className="geco-working-gecko",
                                    **{"aria-hidden": "true"},
                                ),
                                className="geco-working-track",
                                **{"aria-hidden": "true"},
                            ),
                            html.Strong("GeCo is working"),
                            html.Span(
                                "Please sit tight while this operation finishes."
                            ),
                        ],
                        className="geco-working-card",
                    ),
                    className="geco-working-overlay",
                    role="status",
                    **{
                        "aria-live": "polite",
                        "aria-busy": "true",
                        "aria-label": "GeCo is working",
                    },
                ),
                children=[
                dcc.Store(id="focal-unit-store", data=initial_focal_unit_id),
                dcc.Store(
                    id="explore-focal-unit-store",
                    data=initial_focal_unit_id if initial_workspace == "explore" else None,
                ),
                dcc.Store(
                    id="focus-focal-unit-store",
                    data=initial_focal_unit_id if initial_workspace == "focus" else None,
                ),
                dcc.Store(id="geometry-map-base-revision-store", data=0),
                dcc.Store(id="semantic-scores-store", data=initial_semantic_scores),
                dcc.Store(id="semantic-range", data=initial_state.get("semantic_range")),
                dcc.Store(
                    id="metadata-filters-store",
                    data=_normalize_metadata_filters(
                        initial_state.get("metadata_filters", []), metadata_profiles
                    ),
                ),
                dcc.Store(id="saved-filter-refresh-store", data=0),
                dcc.Store(id="active-session-store", data=int(initial_session["session_id"])),
                dcc.Store(id="navigation-history-store", data=initial_history),
                dcc.Store(id="desired-view-store", data=initial_view_id),
                dcc.Store(id="view-refresh-store", data=0),
                dcc.Store(
                    id="desired-semantic-range-store",
                    data=initial_state.get("semantic_range"),
                ),
                dcc.Store(id="focus-recommendation-store", data={}),
                dcc.Store(id="focus-refresh-store", data=0),
                dcc.Store(id="focus-model-refresh-store", data=0),
                dcc.Store(id="classifier-spec-refresh-store", data=0),
                dcc.Store(id="committee-refresh-store", data=0),
                dcc.Store(id="prediction-geometry-refresh-store", data=None),
                dcc.Store(id="apply-refresh-store", data=0),
                dcc.Store(id="memo-refresh-store", data=0),
                dcc.Store(id="active-memo-store", data=None),
                dcc.Store(id="explore-code-refresh-store", data=0),
                dcc.Store(
                    id="explore-code-palette-store",
                    data=initial_explore_code_palette,
                ),
                dcc.Store(
                    id="explore-code-row-count-store",
                    data=len(initial_explore_code_palette),
                ),
                dcc.Store(id="explore-annotation-refresh-store", data=0),
                dcc.Store(id="explore-map-filter-refresh-store", data=0),
                dcc.Store(
                    id="explore-span-selection-store",
                    data=(
                        [int(initial_state["focal_unit_id"])]
                        if initial_state.get("focal_unit_id") is not None
                        else []
                    ),
                ),
                dcc.Store(
                    id="focus-span-selection-store",
                    data=(
                        [int(initial_state["focal_unit_id"])]
                        if initial_state.get("focal_unit_id") is not None
                        else []
                    ),
                ),
                dcc.Store(id="teaching-example-refresh-store", data=0),
                dcc.Store(id="teaching-example-code-store", data=None),
                dcc.Store(id="code-modal-target-store", data=None),
                dcc.Store(id="memo-center-selected-store", data=None),
                dcc.Store(id="memo-center-version-store", data=None),
                dcc.Store(id="memo-center-edit-store", data=False),
                dcc.Store(id="code-center-selected-store", data=None),
                dcc.Store(id="code-center-version-store", data=None),
                dcc.Store(id="code-center-edit-store", data=False),
                dcc.Store(id="code-center-focal-store", data=None),
                    html.Header(
                        [
                            html.Div(
                                [
                                    html.Span(
                                        "🦎",
                                        className="app-gecko",
                                        role="img",
                                        **{"aria-label": "gecko"},
                                    ),
                                    html.Div(
                                        [
                                            html.H1("GeCo", className="app-title"),
                                            html.P(
                                                "Geometric Coder — A geometric coding platform for qualitative inquiry",
                                                className="app-subtitle",
                                            ),
                                        ]
                                    ),
                                ],
                                className="app-brand",
                            ),
                            html.Div(
                                _project_summary_children(
                                    html=html,
                                    modality=str(metadata["modality"]),
                                    key_columns=[str(value) for value in metadata["key_columns"]],
                                    hierarchy_summary=hierarchy_summary,
                                    atomic_count=len(units),
                                ),
                                className="project-summary",
                            ),
                        ],
                        id="app-header",
                        className="app-header",
                    ),
                    dcc.Tabs(
                id="workspace-tabs",
                value="explore",
                children=[
                    dcc.Tab(label="Explore", value="explore", className="workflow-tab", selected_className="workflow-tab-selected", children=_explore_layout(
                        dcc=dcc,
                        html=html,
                        geometries=geometries,
                        views=views_by_geometry.get(initial_geometry_id, []),
                        hierarchy=hierarchy,
                        initial_state=initial_state,
                        initial_geometry_id=initial_geometry_id,
                        initial_view_id=initial_view_id,
                        initial_semantic_scores=initial_semantic_scores,
                        geometry_by_id=geometry_by_id,
                        total_units=len(units),
                        sessions=sessions,
                        initial_session_id=int(initial_session["session_id"]),
                        initial_seen_count=initial_seen_count,
                        metadata_profiles=metadata_profiles,
                        allow_create_view=not project.is_external_backed,
                    )),
                    dcc.Tab(
                        label="Develop",
                        value="focus",
                        className="workflow-tab",
                        selected_className="workflow-tab-selected",
                        children=_focus_layout(
                            dcc=dcc,
                            html=html,
                            codes=codes,
                            geometries=geometries,
                            classifier_specs=classifier_specs,
                            hierarchy=hierarchy,
                            initial_state=initial_state,
                        ),
                    ),
                    dcc.Tab(
                        label="Apply and Review",
                        value="apply",
                        className="workflow-tab",
                        selected_className="workflow-tab-selected",
                        children=_apply_layout(
                            dcc=dcc,
                            html=html,
                            dag=dag,
                            codes=codes,
                            classifier_specs=classifier_specs,
                        ),
                    ),
                    dcc.Tab(
                        label="Codes",
                        value="codes",
                        className="workflow-tab",
                        selected_className="workflow-tab-selected",
                        children=_code_center_layout(
                            dcc=dcc, html=html, dash_table=dash_table,
                            geometries=geometries, views=views_by_geometry.get(initial_geometry_id, []),
                            initial_geometry_id=initial_geometry_id, initial_view_id=initial_view_id,
                            total_units=len(units),
                        ),
                    ),
                    dcc.Tab(
                        label="Memos",
                        value="memos",
                        className="workflow-tab",
                        selected_className="workflow-tab-selected",
                        children=_memo_center_layout(
                            dcc=dcc,
                            html=html,
                            dash_table=dash_table,
                        ),
                    ),
                ],
                    ),
                    dcc.Location(id="memo-navigation-location", refresh=False),
                    dcc.Location(id="memo-presentation-location", refresh=False),
                    _memo_presentation_layout(
                        dcc=dcc,
                        html=html,
                        views=views,
                        geometry_by_id=geometry_by_id,
                        initial_view_id=initial_view_id,
                        total_units=len(units),
                    ),
                    _code_modal(dcc=dcc, html=html),
                    _teaching_example_modal(
                        dcc=dcc, html=html, dash_table=dash_table
                    ),
                    _classifier_spec_modal(
                        dcc=dcc, html=html, geometries=geometries
                    ),
                    _classifier_manager_modal(
                        dcc=dcc, html=html, dash_table=dash_table
                    ),
                    _committee_modal(
                        dcc=dcc, html=html, dash_table=dash_table
                    ),
                    _testing_center_modal(
                        dcc=dcc, html=html, dash_table=dash_table
                    ),
                    _metadata_filter_modal(
                        dcc=dcc, html=html, metadata_profiles=metadata_profiles
                    ),
                    _save_filter_set_modal(dcc=dcc, html=html),
                    _load_filter_set_modal(dcc=dcc, html=html),
                ],
            ),
            html.Div(id="session-state-status", className="sr-only"),
        ],
        className="app-shell",
        style=GECO_DASH_THEME,
    )

    # Keep tab-specific focal mirrors on the client. Hidden workspaces should not
    # wake their server-side reading/model callbacks every time the global focal
    # unit changes in another workspace. Switching tabs copies the current focal
    # unit into that workspace exactly once.
    app.clientside_callback(
        """
        function(workspace, focalUnitId) {
            const noUpdate = window.dash_clientside.no_update;
            return [
                workspace === "explore" ? focalUnitId : noUpdate,
                workspace === "focus" ? focalUnitId : noUpdate
            ];
        }
        """,
        Output("explore-focal-unit-store", "data"),
        Output("focus-focal-unit-store", "data"),
        Input("workspace-tabs", "value"),
        Input("focal-unit-store", "data"),
    )

    app.clientside_callback(
        """
        function(annotationRefresh, uncodedValues, currentRefresh) {
            const enabled = Array.isArray(uncodedValues) && uncodedValues.includes("uncoded_only");
            if (!enabled) {
                return window.dash_clientside.no_update;
            }
            return (Number(currentRefresh) || 0) + 1;
        }
        """,
        Output("explore-map-filter-refresh-store", "data"),
        Input("explore-annotation-refresh-store", "data"),
        State("explore-uncoded-only", "value"),
        State("explore-map-filter-refresh-store", "data"),
        prevent_initial_call=True,
    )

    @app.callback(
        Output("session-dropdown", "options"),
        Output("session-dropdown", "value"),
        Output("new-session-title", "value"),
        Input("new-session-button", "n_clicks"),
        State("new-session-title", "value"),
        prevent_initial_call=True,
    )
    def create_session(n_clicks: int, title: str | None) -> tuple[list[dict[str, Any]], int, str]:
        del n_clicks
        clean_title = (title or "").strip() or f"Session {len(project.sessions()) + 1}"
        session_id = project.create_session(clean_title)
        return _session_options(project.sessions()), session_id, ""

    @app.callback(
        Output("active-session-store", "data"),
        Output("geometry-dropdown", "value"),
        Output("desired-view-store", "data"),
        Output("focal-unit-store", "data", allow_duplicate=True),
        Output("navigation-history-store", "data"),
        Output("literal-query", "value"),
        Output("literal-options", "value"),
        Output("semantic-query", "value"),
        Output("semantic-scores-store", "data", allow_duplicate=True),
        Output("desired-semantic-range-store", "data"),
        Output("metadata-filters-store", "data"),
        Output("metadata-filter-operator", "value"),
        Output("context-level", "value"),
        Output("context-window", "value"),
        Output("page-size", "value"),
        Output("page-number", "value"),
        Output("navigation-new-only", "value"),
        Output("workspace-tabs", "value"),
        Output("explore-code-palette-store", "data"),
        Output("explore-code-row-count-store", "data"),
        Output("focus-code-dropdown", "value", allow_duplicate=True),
        Output("focus-active-classifier-dropdown", "value", allow_duplicate=True),
        Output("focus-committee-dropdown", "value", allow_duplicate=True),
        Output("focus-context-level", "value"),
        Output("focus-context-window", "value"),
        Input("session-dropdown", "value"),
        prevent_initial_call=True,
    )
    def restore_session(session_id: int) -> tuple[Any, ...]:
        session = project.database.get_session(int(session_id))
        state = dict(session["state"])
        geometry_id = _valid_geometry_id(state.get("geometry_id"), geometries)
        view_id = _valid_view_id(state.get("view_id"), project.views(geometry_id))
        query = str(state.get("semantic_query", ""))
        scores = project.semantic_search(query) if query else {}
        literal_options = []
        if state.get("case_sensitive"):
            literal_options.append("case")
        if state.get("regex"):
            literal_options.append("regex")
        context_level = int(state.get("context_level", hierarchy[-1]["level_index"]))
        context_window = max(1, int(state.get("context_window", 1)))
        current_code_ids = {int(code["code_id"]) for code in project.codes()}
        requested_code = state.get("focus_code_id")
        focus_code_id = (
            int(requested_code)
            if requested_code is not None and int(requested_code) in current_code_ids
            else None
        )
        spec_ids = {
            int(record["classifier_spec_id"])
            for record in project.classifier_specs(code_id=focus_code_id)
        } if focus_code_id is not None else set()
        requested_classifier = state.get("focus_classifier_spec_id")
        focus_classifier = (
            int(requested_classifier)
            if requested_classifier is not None and int(requested_classifier) in spec_ids
            else (min(spec_ids) if spec_ids else None)
        )
        committee_ids = {
            int(record["committee_id"])
            for record in project.classifier_committees(focus_code_id)
        } if focus_code_id is not None else set()
        requested_committee = state.get("focus_committee_id")
        focus_committee = (
            int(requested_committee)
            if requested_committee is not None
            and int(requested_committee) in committee_ids
            else (min(committee_ids) if committee_ids else None)
        )
        history = _normalize_navigation_history(
            state.get("navigation_history"),
            focal_unit_id=state.get("focal_unit_id"),
        )
        explore_palette = _normalize_explore_code_palette(
            state.get("explore_code_palette"), current_code_ids
        )
        return (
            int(session_id),
            geometry_id,
            view_id,
            state.get("focal_unit_id"),
            history,
            str(state.get("literal_query", "")),
            literal_options,
            query,
            scores,
            state.get("semantic_range"),
            _normalize_metadata_filters(
                state.get("metadata_filters", []), metadata_profiles
            ),
            _normalize_boolean_operator(
                state.get("metadata_filter_operator", "and")
            ),
            context_level,
            context_window,
            str(max(1, int(state.get("page_size", 5000)))),
            max(1, int(state.get("page_number", 1))),
            (["new_only"] if state.get("navigation_new_only", True) else []),
            str(state.get("workspace", "explore")),
            explore_palette,
            len(explore_palette),
            focus_code_id,
            focus_classifier,
            focus_committee,
            int(state.get("focus_context_level", hierarchy[-1]["level_index"])),
            max(1, int(state.get("focus_context_window", 1))),
        )

    @app.callback(
        Output("view-dropdown", "options"),
        Output("view-dropdown", "value"),
        Input("geometry-dropdown", "value"),
        Input("view-refresh-store", "data"),
        State("view-dropdown", "value"),
        State("desired-view-store", "data"),
    )
    def update_view_options(
        geometry_id: int | None,
        refresh: int,
        current_view_id: int | None,
        desired_view_id: int | None,
    ) -> tuple[list[dict[str, Any]], int | None]:
        del refresh
        candidates = project.views(int(geometry_id)) if geometry_id is not None else []
        options = [{"label": view["name"], "value": int(view["view_id"])} for view in candidates]
        valid_ids = {option["value"] for option in options}
        if desired_view_id in valid_ids:
            value = desired_view_id
        elif current_view_id in valid_ids:
            value = current_view_id
        else:
            value = options[0]["value"] if options else None
        return options, value

    @app.callback(
        Output("projection-modal", "style"),
        Output("view-refresh-store", "data"),
        Output("desired-view-store", "data", allow_duplicate=True),
        Output("projection-create-status", "children"),
        Input("projection-open-button", "n_clicks"),
        Input("projection-cancel-button", "n_clicks"),
        Input("projection-cancel-footer", "n_clicks"),
        Input("projection-create-button", "n_clicks"),
        State("geometry-dropdown", "value"),
        State("projection-name", "value"),
        State("projection-method", "value"),
        State("projection-umap-neighbors", "value"),
        State("projection-umap-min-dist", "value"),
        State("projection-umap-metric", "value"),
        State("projection-umap-spread", "value"),
        State("projection-umap-repulsion", "value"),
        State("projection-umap-random-state", "value"),
        State("projection-pca-solver", "value"),
        State("projection-pca-whiten", "value"),
        State("projection-pca-random-state", "value"),
        State("projection-svd-algorithm", "value"),
        State("projection-svd-n-iter", "value"),
        State("projection-svd-random-state", "value"),
        State("view-refresh-store", "data"),
        prevent_initial_call=True,
        running=[
            (Output("projection-create-button", "disabled"), True, False),
            (
                Output("projection-progress-wrap", "style"),
                {"display": "block"},
                {"display": "none"},
            ),
        ],
    )
    def manage_projection_modal(
        open_clicks: int,
        cancel_clicks: int,
        cancel_footer_clicks: int,
        create_clicks: int,
        geometry_id: int | None,
        name: str | None,
        method: str | None,
        umap_neighbors: int | None,
        umap_min_dist: float | None,
        umap_metric: str | None,
        umap_spread: float | None,
        umap_repulsion: float | None,
        umap_random_state: int | None,
        pca_solver: str | None,
        pca_whiten: list[str] | None,
        pca_random_state: int | None,
        svd_algorithm: str | None,
        svd_n_iter: int | None,
        svd_random_state: int | None,
        refresh: int | None,
    ) -> tuple[dict[str, str], int | Any, int | Any, str | Any]:
        del open_clicks, cancel_clicks, cancel_footer_clicks, create_clicks
        triggered = callback_context.triggered_id
        if triggered == "projection-open-button":
            return {"display": "flex"}, no_update, no_update, ""
        if triggered in {"projection-cancel-button", "projection-cancel-footer"}:
            return {"display": "none"}, no_update, no_update, ""
        if geometry_id is None or not method:
            return {"display": "flex"}, int(refresh or 0), no_update, (
                "Choose a geometry and projection method."
            )
        try:
            parameters = _projection_parameters(
                method=str(method),
                umap_neighbors=umap_neighbors,
                umap_min_dist=umap_min_dist,
                umap_metric=umap_metric,
                umap_spread=umap_spread,
                umap_repulsion=umap_repulsion,
                umap_random_state=umap_random_state,
                pca_solver=pca_solver,
                pca_whiten=pca_whiten,
                pca_random_state=pca_random_state,
                svd_algorithm=svd_algorithm,
                svd_n_iter=svd_n_iter,
                svd_random_state=svd_random_state,
            )
            geometry_name = str(geometry_by_id[int(geometry_id)]["name"])
            clean_name = (name or "").strip() or (
                f"{geometry_name}_{method}_{len(project.views(int(geometry_id))) + 1}"
            )
            view_id = project.create_view(
                geometry_id=int(geometry_id),
                name=clean_name,
                method=str(method),
                parameters=parameters,
            )
        except (KeyError, TypeError, ValueError) as error:
            return {"display": "flex"}, int(refresh or 0), no_update, str(error)
        return (
            {"display": "none"},
            int(refresh or 0) + 1,
            view_id,
            f"Created 2D view {clean_name!r}.",
        )

    @app.callback(
        Output("projection-umap-fields", "style"),
        Output("projection-pca-fields", "style"),
        Output("projection-svd-fields", "style"),
        Input("projection-method", "value"),
    )
    def show_projection_fields(method: str | None) -> tuple[dict[str, str], ...]:
        hidden = {"display": "none"}
        shown = {"display": "grid"}
        return (
            shown if method == "umap" else hidden,
            shown if method == "pca" else hidden,
            shown if method in {"truncated_svd", "svd"} else hidden,
        )

    @app.callback(
        Output("semantic-scores-store", "data", allow_duplicate=True),
        Output("semantic-search-status", "children"),
        Output("semantic-query", "value", allow_duplicate=True),
        Output("semantic-range", "data", allow_duplicate=True),
        Output("desired-semantic-range-store", "data", allow_duplicate=True),
        Input("semantic-search-button", "n_clicks"),
        Input("semantic-clear-button", "n_clicks"),
        State("semantic-query", "value"),
        prevent_initial_call=True,
        running=[
            (Output("semantic-search-button", "disabled"), True, False),
            (Output("semantic-clear-button", "disabled"), True, False),
        ],
    )
    def run_semantic_search(
        search_clicks: int,
        clear_clicks: int,
        query: str | None,
    ) -> tuple[dict[str, list[float]], str, Any, None, None]:
        del search_clicks, clear_clicks
        if callback_context.triggered_id == "semantic-clear-button":
            return {}, "Semantic search cleared.", "", None, None
        clean_query = (query or "").strip()
        if not clean_query:
            return {}, "Semantic search cleared.", "", None, None
        scores = project.semantic_search(clean_query)
        if not scores:
            return (
                {},
                "No registered geometry supports semantic query transformation.",
                no_update,
                None,
                None,
            )
        return (
            scores,
            f"Computed semantic similarity in {len(scores)} geometries.",
            no_update,
            None,
            None,
        )

    @app.callback(
        Output("semantic-range-slider", "min"),
        Output("semantic-range-slider", "max"),
        Output("semantic-range-slider", "step"),
        Output("semantic-range-slider", "value"),
        Output("semantic-range-slider", "marks"),
        Output("semantic-range-wrap", "style"),
        Output("semantic-clear-button", "style"),
        Output("nav-most-similar", "style"),
        Output("nav-least-similar", "style"),
        Input("semantic-scores-store", "data"),
        Input("geometry-dropdown", "value"),
        Input("desired-semantic-range-store", "data"),
    )
    def update_semantic_range(
        all_scores: dict[str, list[float]] | None,
        geometry_id: int | None,
        desired_range: list[float] | None,
    ) -> tuple[
        float,
        float,
        float,
        list[float],
        dict[float, str],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ]:
        hidden = {"display": "none"}
        if geometry_id is None or not all_scores:
            return 0.0, 1.0, 0.001, [0.0, 1.0], {}, hidden, hidden, hidden, hidden
        geometry_name = str(geometry_by_id[int(geometry_id)]["name"])
        scores = all_scores.get(geometry_name)
        if not scores:
            return 0.0, 1.0, 0.001, [0.0, 1.0], {}, hidden, hidden, hidden, hidden
        values = np.asarray(scores, dtype=float)
        lower = float(np.nanmin(values))
        upper = float(np.nanmax(values))
        if np.isclose(lower, upper):
            upper = lower + 1e-9
        selected = [lower, upper]
        if desired_range and len(desired_range) == 2:
            requested_lower = max(lower, min(upper, float(desired_range[0])))
            requested_upper = max(lower, min(upper, float(desired_range[1])))
            selected = sorted([requested_lower, requested_upper])
        step = max((upper - lower) / 1000, 1e-9)
        marks = {
            lower: f"{lower:.3f}",
            upper: f"{upper:.3f}",
        }
        return (
            lower,
            upper,
            step,
            selected,
            marks,
            {"display": "block"},
            {"display": "inline-block"},
            {"display": "inline-block"},
            {"display": "inline-block"},
        )

    @app.callback(
        Output("semantic-range", "data", allow_duplicate=True),
        Input("semantic-range-slider", "value"),
        prevent_initial_call=True,
    )
    def synchronize_semantic_range(value: list[float] | None) -> list[float] | Any:
        if not value or len(value) != 2:
            return no_update
        return sorted([float(value[0]), float(value[1])])


    @app.callback(
        Output("metadata-filter-condition", "options"),
        Output("metadata-filter-condition", "value"),
        Output("metadata-filter-type-help", "children"),
        Output("metadata-filter-value", "type"),
        Output("metadata-filter-value", "placeholder"),
        Output("metadata-filter-value-second", "type"),
        Output("metadata-filter-category-values", "options"),
        Input("metadata-filter-field", "value"),
    )
    def configure_metadata_filter_field(
        field: str | None,
    ) -> tuple[list[dict[str, str]], Any, str, str, str, str, list[dict[str, Any]]]:
        profile = metadata_profiles.get(str(field), {"type": "text", "options": []})
        kind = str(profile["type"])
        options = _metadata_operator_options(kind)
        input_type = "date" if kind == "datetime" else "text"
        placeholder = "Enter a value"
        if kind == "numeric":
            input_type = "number"
            placeholder = "Enter a number"
        elif kind == "datetime":
            placeholder = "YYYY-MM-DD"
        category_options = [
            {"label": str(value), "value": value}
            for value in profile.get("options", [])
        ]
        return (
            options,
            options[0]["value"] if options else None,
            f"Detected as {kind} metadata.",
            input_type,
            placeholder,
            input_type,
            category_options,
        )

    @app.callback(
        Output("metadata-filter-value-wrap", "style"),
        Output("metadata-filter-value-second-wrap", "style"),
        Output("metadata-filter-category-wrap", "style"),
        Input("metadata-filter-field", "value"),
        Input("metadata-filter-condition", "value"),
    )
    def configure_metadata_filter_values(
        field: str | None,
        condition: str | None,
    ) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
        profile = metadata_profiles.get(str(field), {"type": "text"})
        kind = str(profile["type"])
        if condition in {"missing", "not_missing", "true", "false"}:
            return {"display": "none"}, {"display": "none"}, {"display": "none"}
        if kind == "categorical":
            return {"display": "none"}, {"display": "none"}, {"display": "block"}
        return (
            {"display": "block"},
            {"display": "block" if condition == "between" else "none"},
            {"display": "none"},
        )

    @app.callback(
        Output("metadata-filter-modal", "style"),
        Output("metadata-filters-store", "data", allow_duplicate=True),
        Output("metadata-filter-modal-status", "children"),
        Output("metadata-filter-value", "value"),
        Output("metadata-filter-value-second", "value"),
        Output("metadata-filter-category-values", "value"),
        Input("metadata-filter-add-button", "n_clicks"),
        Input("metadata-filter-modal-close", "n_clicks"),
        Input("metadata-filter-modal-cancel", "n_clicks"),
        Input("metadata-filter-modal-add", "n_clicks"),
        State("metadata-filter-field", "value"),
        State("metadata-filter-condition", "value"),
        State("metadata-filter-value", "value"),
        State("metadata-filter-value-second", "value"),
        State("metadata-filter-category-values", "value"),
        State("metadata-filters-store", "data"),
        prevent_initial_call=True,
    )
    def manage_metadata_filter_modal(
        open_clicks: int,
        close_clicks: int,
        cancel_clicks: int,
        add_clicks: int,
        field: str | None,
        condition: str | None,
        value: Any,
        value2: Any,
        category_values: list[Any] | None,
        current_filters: list[dict[str, Any]] | None,
    ) -> tuple[dict[str, str], Any, str, Any, Any, list[Any]]:
        del open_clicks, close_clicks, cancel_clicks, add_clicks
        triggered = callback_context.triggered_id
        if triggered == "metadata-filter-add-button":
            return {"display": "flex"}, no_update, "", None, None, []
        if triggered in {"metadata-filter-modal-close", "metadata-filter-modal-cancel"}:
            return {"display": "none"}, no_update, "", no_update, no_update, no_update
        profile = metadata_profiles.get(str(field))
        if profile is None or not condition:
            return {"display": "flex"}, no_update, "Choose a field and condition.", no_update, no_update, no_update
        kind = str(profile["type"])
        stored_value: Any = category_values if kind == "categorical" else value
        if condition not in {"missing", "not_missing", "true", "false"}:
            if kind == "categorical" and not category_values:
                return {"display": "flex"}, no_update, "Choose at least one value.", no_update, no_update, no_update
            if kind != "categorical" and (value is None or str(value).strip() == ""):
                return {"display": "flex"}, no_update, "Enter a filter value.", no_update, no_update, no_update
            if condition == "between" and (value2 is None or str(value2).strip() == ""):
                return {"display": "flex"}, no_update, "Enter both ends of the range.", no_update, no_update, no_update
        clause = {
            "id": uuid.uuid4().hex,
            "field": str(field),
            "type": kind,
            "operator": str(condition),
            "value": stored_value,
            "value2": value2,
        }
        candidate = [*(current_filters or []), clause]
        normalized = _normalize_metadata_filters(candidate, metadata_profiles)
        _mask, errors = _evaluate_metadata_filters(
            units, normalized, operator="and"
        )
        if errors:
            return {"display": "flex"}, no_update, errors[0], no_update, no_update, no_update
        return {"display": "none"}, normalized, "", None, None, []

    @app.callback(
        Output("metadata-filters-store", "data", allow_duplicate=True),
        Input("metadata-filter-clear-button", "n_clicks"),
        Input({"type": "remove-metadata-filter", "index": ALL}, "n_clicks"),
        State("metadata-filters-store", "data"),
        prevent_initial_call=True,
    )
    def remove_metadata_filters(
        clear_clicks: int,
        remove_clicks: list[int],
        current_filters: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]] | Any:
        del clear_clicks
        triggered = callback_context.triggered_id
        if triggered == "metadata-filter-clear-button":
            return []
        if isinstance(triggered, dict):
            # Pattern-matching callbacks also fire when chip buttons are mounted
            # or remounted. Only remove a clause after an actual click; otherwise
            # changing AND/OR/XOR or rendering a newly added chip can erase state.
            identifier = str(triggered.get("index", ""))
            clauses = list(current_filters or [])
            position = next(
                (
                    index
                    for index, clause in enumerate(clauses)
                    if str(clause.get("id")) == identifier
                ),
                None,
            )
            if (
                position is None
                or position >= len(remove_clicks or [])
                or int((remove_clicks or [])[position] or 0) < 1
            ):
                return no_update
            return [
                clause
                for clause in clauses
                if str(clause.get("id")) != identifier
            ]
        return no_update

    @app.callback(
        Output("active-metadata-filter-chips", "children"),
        Output("metadata-filter-status", "children"),
        Input("metadata-filters-store", "data"),
        Input("metadata-filter-operator", "value"),
    )
    def render_metadata_filters(
        filters: list[dict[str, Any]] | None,
        operator: str | None,
    ) -> tuple[list[Any], str]:
        clauses = _normalize_metadata_filters(filters or [], metadata_profiles)
        if not clauses:
            message = (
                "No metadata columns were supplied."
                if not metadata_profiles
                else "No active metadata filters."
            )
            return [html.Span("No active filters", className="filter-empty-state")], message
        mask, errors = _evaluate_metadata_filters(units, clauses, operator=operator)
        chips = []
        for clause in clauses:
            label = _filter_clause_label(clause)
            chips.append(
                html.Span(
                    [
                        html.Span(label, className="metadata-filter-chip-label"),
                        html.Button(
                            "×",
                            id={
                                "type": "remove-metadata-filter",
                                "index": str(clause["id"]),
                            },
                            n_clicks=0,
                            title="Remove this filter",
                        ),
                    ],
                    className="metadata-filter-chip",
                    title=label,
                )
            )
        if errors:
            return chips, "; ".join(errors)
        mode = _normalize_boolean_operator(operator).upper()
        return chips, f"{int(mask.sum()):,} of {len(units):,} units match · {mode}"

    @app.callback(
        Output("save-filter-set-modal", "style"),
        Output("save-filter-set-name", "value"),
        Output("save-filter-set-status", "children"),
        Output("saved-filter-refresh-store", "data", allow_duplicate=True),
        Input("metadata-filter-save-open-button", "n_clicks"),
        Input("save-filter-set-close", "n_clicks"),
        Input("save-filter-set-cancel", "n_clicks"),
        Input("save-filter-set-confirm", "n_clicks"),
        State("save-filter-set-name", "value"),
        State("metadata-filters-store", "data"),
        State("metadata-filter-operator", "value"),
        State("saved-filter-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def manage_save_filter_set(
        open_clicks: int,
        close_clicks: int,
        cancel_clicks: int,
        confirm_clicks: int,
        name: str | None,
        filters: list[dict[str, Any]] | None,
        operator: str | None,
        refresh: int | None,
    ) -> tuple[dict[str, str], Any, str, int]:
        del open_clicks, close_clicks, cancel_clicks, confirm_clicks
        triggered = callback_context.triggered_id
        current_refresh = int(refresh or 0)
        if triggered == "metadata-filter-save-open-button":
            return {"display": "flex"}, "", "", current_refresh
        if triggered in {"save-filter-set-close", "save-filter-set-cancel"}:
            return {"display": "none"}, "", "", current_refresh
        clean_name = (name or "").strip()
        clauses = _normalize_metadata_filters(filters or [], metadata_profiles)
        if not clean_name:
            return {"display": "flex"}, no_update, "Enter a name.", current_refresh
        if not clauses:
            return {"display": "flex"}, no_update, "Add at least one filter first.", current_refresh
        records = _read_saved_filter_sets(saved_filter_sets_path)
        records.append(
            {
                "id": uuid.uuid4().hex,
                "name": clean_name,
                "operator": _normalize_boolean_operator(operator),
                "filters": clauses,
                "created_at": datetime.now(UTC).isoformat(),
            }
        )
        _write_saved_filter_sets(saved_filter_sets_path, records)
        return {"display": "none"}, "", "", current_refresh + 1

    @app.callback(
        Output("saved-filter-set-list", "children"),
        Input("saved-filter-refresh-store", "data"),
    )
    def render_saved_filter_sets(refresh: int | None) -> list[Any]:
        del refresh
        records = _read_saved_filter_sets(saved_filter_sets_path)
        if not records:
            return [html.P("No filter collections have been saved.", className="filter-empty-state")]
        return [
            html.Div(
                [
                    html.Div(
                        [
                            html.Strong(str(record.get("name", "Untitled"))),
                            html.Small(
                                f"{len(record.get('filters', []))} filters · "
                                f"{_normalize_boolean_operator(record.get('operator')).upper()}"
                            ),
                        ],
                        className="saved-filter-set-description",
                    ),
                    html.Button(
                        "Load",
                        id={"type": "load-filter-set", "index": str(record.get("id"))},
                        n_clicks=0,
                    ),
                    html.Button(
                        "×",
                        id={"type": "delete-filter-set", "index": str(record.get("id"))},
                        n_clicks=0,
                        className="danger-icon-action",
                        title="Delete saved filter collection",
                    ),
                ],
                className="saved-filter-set-row",
            )
            for record in records
        ]

    @app.callback(
        Output("load-filter-set-modal", "style"),
        Output("metadata-filters-store", "data", allow_duplicate=True),
        Output("metadata-filter-operator", "value", allow_duplicate=True),
        Output("saved-filter-refresh-store", "data", allow_duplicate=True),
        Output("load-filter-set-status", "children"),
        Input("metadata-filter-load-open-button", "n_clicks"),
        Input("load-filter-set-close", "n_clicks"),
        Input("load-filter-set-done", "n_clicks"),
        Input({"type": "load-filter-set", "index": ALL}, "n_clicks"),
        Input({"type": "delete-filter-set", "index": ALL}, "n_clicks"),
        State("saved-filter-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def manage_load_filter_set(
        open_clicks: int,
        close_clicks: int,
        done_clicks: int,
        load_clicks: list[int],
        delete_clicks: list[int],
        refresh: int | None,
    ) -> tuple[dict[str, str], Any, Any, int, str]:
        del open_clicks, close_clicks, done_clicks
        triggered = callback_context.triggered_id
        current_refresh = int(refresh or 0)
        if triggered == "metadata-filter-load-open-button":
            return {"display": "flex"}, no_update, no_update, current_refresh, ""
        if isinstance(triggered, str) and triggered in {
            "load-filter-set-close",
            "load-filter-set-done",
        }:
            return {"display": "none"}, no_update, no_update, current_refresh, ""
        if not isinstance(triggered, dict):
            return no_update, no_update, no_update, current_refresh, ""
        identifier = str(triggered.get("index", ""))
        records = _read_saved_filter_sets(saved_filter_sets_path)
        selected_position = next(
            (
                index
                for index, record in enumerate(records)
                if str(record.get("id")) == identifier
            ),
            None,
        )
        if selected_position is None:
            return {"display": "flex"}, no_update, no_update, current_refresh, "Saved collection not found."
        clicked_values = (
            delete_clicks
            if triggered.get("type") == "delete-filter-set"
            else load_clicks
        )
        if (
            selected_position >= len(clicked_values or [])
            or int((clicked_values or [])[selected_position] or 0) < 1
        ):
            # Saved-set rows use pattern IDs. Mounting those buttons is not a
            # user action and must not load or delete anything.
            return no_update, no_update, no_update, current_refresh, ""
        selected = records[selected_position]
        if triggered.get("type") == "delete-filter-set":
            _write_saved_filter_sets(
                saved_filter_sets_path,
                [record for record in records if str(record.get("id")) != identifier],
            )
            return {"display": "flex"}, no_update, no_update, current_refresh + 1, "Deleted saved collection."
        clauses = _normalize_metadata_filters(
            selected.get("filters", []), metadata_profiles
        )
        return (
            {"display": "none"},
            clauses,
            _normalize_boolean_operator(selected.get("operator")),
            current_refresh,
            "",
        )

    @app.callback(
        Output("geometry-map", "figure"),
        Output("literal-search-status", "children"),
        Output("page-status", "children"),
        Output("geometry-map-base-revision-store", "data"),
        Input("view-dropdown", "value"),
        Input("geometry-dropdown", "value"),
        Input("literal-query", "value"),
        Input("literal-options", "value"),
        Input("semantic-scores-store", "data"),
        Input("semantic-range", "data"),
        Input("metadata-filters-store", "data"),
        Input("metadata-filter-operator", "value"),
        Input("explore-uncoded-only", "value"),
        Input("explore-map-filter-refresh-store", "data"),
        Input("session-dropdown", "value"),
        Input("page-size", "value"),
        Input("page-number", "value"),
        State("geometry-map", "relayoutData"),
        State("explore-focal-unit-store", "data"),
        State("geometry-map-base-revision-store", "data"),
    )
    def update_map(
        view_id: int | None,
        geometry_id: int | None,
        literal_query: str | None,
        literal_options: list[str] | None,
        all_semantic_scores: dict[str, list[float]] | None,
        semantic_range: list[float] | None,
        metadata_filters: list[dict[str, Any]] | None,
        metadata_filter_operator: str | None,
        uncoded_only_values: list[str] | None,
        map_filter_refresh: int | None,
        session_id: int,
        page_size: Any,
        page_number: Any,
        relayout_data: dict[str, Any] | None,
        focal_unit_id: int | None,
        base_revision: int | None,
    ) -> tuple[Any, str, str, int]:
        del map_filter_refresh
        next_revision = int(base_revision or 0) + 1
        if view_id is None or geometry_id is None:
            return _empty_figure("No computed geometry view is available."), "", "", next_revision
        options = set(literal_options or [])
        geometry_record = geometry_by_id[int(geometry_id)]
        semantic_scores = (all_semantic_scores or {}).get(str(geometry_record["name"]))
        try:
            filters = combine_filters(
                texts,
                literal_query=literal_query or "",
                case_sensitive="case" in options,
                regex="regex" in options,
                semantic_scores=semantic_scores,
                semantic_range=(
                    tuple(semantic_range)
                    if semantic_scores is not None and semantic_range
                    else None
                ),
            )
            literal_status = (
                f"{int(filters.literal_matches.sum()):,} literal matches"
                if literal_query
                else ""
            )
        except InvalidRegexError as error:
            filters = combine_filters(texts)
            literal_status = f"Invalid regular expression: {error}"

        size = _coerce_positive_int(page_size, default=5000)
        number = _coerce_positive_int(page_number, default=1)
        uncoded_mask = _explore_uncoded_mask(
            project,
            units,
            enabled="uncoded_only" in set(uncoded_only_values or []),
        )
        positions, page, combined_mask, _metadata_errors = _metadata_filtered_page(
            units,
            metadata_filters or [],
            operator=metadata_filter_operator,
            page_size=size,
            page_number=number,
            additional_mask=uncoded_mask,
        )
        coordinates = project.view_coordinates(int(view_id))[positions]
        page_units = [units[int(position)] for position in positions]
        page_eligible = filters.eligible[positions]
        page_semantic = (
            np.asarray(semantic_scores, dtype=float)[positions]
            if semantic_scores is not None
            else None
        )
        seen = project.database.seen_unit_ids(int(session_id))
        view = project.database.get_view(int(view_id))
        figure = build_geometry_figure(
            coordinates=coordinates,
            units=page_units,
            eligible=page_eligible,
            seen_unit_ids=seen,
            focal_unit_id=int(focal_unit_id) if focal_unit_id is not None else None,
            semantic_scores=page_semantic,
            geometry_name=str(geometry_record["name"]),
            view_name=str(view["name"]),
            page_label=f"Page {page.page_index + 1} of {page.page_count}",
            uirevision=f"view:{int(view_id)}:page:{page.page_index}",
        )
        if callback_context.triggered_id not in {
            "view-dropdown",
            "geometry-dropdown",
            "page-size",
            "page-number",
        }:
            _apply_relayout_ranges(figure, relayout_data)
        filter_match_count = int(combined_mask.sum())
        page_status = (
            f"Page {page.page_index + 1} of {page.page_count} · "
            f"{len(positions):,} points shown · "
            f"{filter_match_count:,} points match filters"
        )
        return figure, literal_status, page_status, next_revision

    @app.callback(
        Output("geometry-map", "figure", allow_duplicate=True),
        Output("session-progress", "children", allow_duplicate=True),
        Input("explore-focal-unit-store", "data"),
        Input("geometry-map-base-revision-store", "data"),
        Input("session-dropdown", "value"),
        State("geometry-map", "figure"),
        prevent_initial_call=True,
    )
    def update_map_navigation_overlays(
        focal_unit_id: int | None,
        base_revision: int | None,
        session_id: int,
        current_figure: dict[str, Any] | None,
    ) -> tuple[Any, str]:
        del base_revision
        seen = project.database.seen_unit_ids(int(session_id))
        progress = f"Visited {len(seen):,} of {len(units):,}"
        if callback_context.triggered_id == "session-dropdown":
            # The base-map callback will rebuild for the new session and then
            # advance geometry-map-base-revision-store. Patch only after that.
            return no_update, progress
        payload = _geometry_overlay_payload(
            current_figure,
            seen_unit_ids=seen,
            focal_unit_id=focal_unit_id,
        )
        if payload is None:
            return no_update, progress
        patch = Patch()
        for trace_index, values in payload.items():
            for property_name, property_value in values.items():
                patch["data"][int(trace_index)][property_name] = property_value
        return patch, progress

    @app.callback(
        Output("page-number", "max"),
        Output("page-number", "value", allow_duplicate=True),
        Input("page-previous", "n_clicks"),
        Input("page-next", "n_clicks"),
        Input("page-size", "value"),
        Input("metadata-filters-store", "data"),
        Input("metadata-filter-operator", "value"),
        Input("explore-uncoded-only", "value"),
        Input("explore-map-filter-refresh-store", "data"),
        State("page-number", "value"),
        prevent_initial_call=True,
    )
    def change_page(
        previous_clicks: int,
        next_clicks: int,
        page_size: Any,
        metadata_filters: list[dict[str, Any]] | None,
        metadata_filter_operator: str | None,
        uncoded_only_values: list[str] | None,
        map_filter_refresh: int | None,
        page_number: Any,
    ) -> tuple[int, int]:
        del previous_clicks, next_clicks, map_filter_refresh
        size = _coerce_positive_int(page_size, default=5000)
        metadata_mask, _errors = _evaluate_metadata_filters(
            units, metadata_filters or [], operator=metadata_filter_operator
        )
        uncoded_mask = _explore_uncoded_mask(
            project,
            units,
            enabled="uncoded_only" in set(uncoded_only_values or []),
        )
        combined_mask = metadata_mask & uncoded_mask
        page_count = deterministic_page(
            int(combined_mask.sum()), page_size=size
        ).page_count
        current = min(max(1, _coerce_positive_int(page_number, default=1)), page_count)
        triggered = callback_context.triggered_id
        if triggered in {
            "page-size",
            "metadata-filters-store",
            "metadata-filter-operator",
            "explore-uncoded-only",
            "explore-map-filter-refresh-store",
        }:
            current = 1
        elif triggered == "page-previous":
            current = max(1, current - 1)
        elif triggered == "page-next":
            current = min(page_count, current + 1)
        return page_count, current

    navigation_inputs = [
        Input("geometry-map", "clickData"),
        Input("nav-nearest", "n_clicks"),
        Input("nav-farthest", "n_clicks"),
        Input("nav-most-similar", "n_clicks"),
        Input("nav-least-similar", "n_clicks"),
        Input("nav-random", "n_clicks"),
    ]

    @app.callback(
        Output("focal-unit-store", "data", allow_duplicate=True),
        Output("navigation-history-store", "data", allow_duplicate=True),
        Output("navigation-status", "children"),
        *navigation_inputs,
        State("navigation-new-only", "value"),
        State("session-dropdown", "value"),
        State("geometry-dropdown", "value"),
        State("focal-unit-store", "data"),
        State("literal-query", "value"),
        State("literal-options", "value"),
        State("semantic-scores-store", "data"),
        State("semantic-range", "data"),
        State("metadata-filters-store", "data"),
        State("metadata-filter-operator", "value"),
        State("explore-uncoded-only", "value"),
        State("page-size", "value"),
        State("page-number", "value"),
        State("navigation-history-store", "data"),
        prevent_initial_call=True,
    )
    def navigate(
        click_data: dict[str, Any] | None,
        nearest_clicks: int,
        farthest_clicks: int,
        most_similar_clicks: int,
        least_similar_clicks: int,
        random_clicks: int,
        new_only_values: list[str] | None,
        session_id: int,
        geometry_id: int | None,
        focal_unit_id: int | None,
        literal_query: str | None,
        literal_options: list[str] | None,
        all_semantic_scores: dict[str, list[float]] | None,
        semantic_range: list[float] | None,
        metadata_filters: list[dict[str, Any]] | None,
        metadata_filter_operator: str | None,
        uncoded_only_values: list[str] | None,
        page_size: Any,
        page_number: Any,
        history: dict[str, Any] | None,
    ) -> tuple[int | None, dict[str, Any] | Any, str]:
        del (
            nearest_clicks,
            farthest_clicks,
            most_similar_clicks,
            least_similar_clicks,
            random_clicks,
        )
        triggered = callback_context.triggered_id
        if triggered == "geometry-map":
            unit_id = _clicked_unit_id(click_data)
            if unit_id is None:
                return no_update, no_update, no_update
            method = "plot_click"
        else:
            strategy_by_button = {
                "nav-nearest": "nearest",
                "nav-farthest": "farthest",
                "nav-most-similar": "most_similar",
                "nav-least-similar": "least_similar",
                "nav-random": "random",
            }
            strategy = strategy_by_button[str(triggered)]
            options = set(literal_options or [])
            geometry_record = (
                geometry_by_id[int(geometry_id)] if geometry_id is not None else None
            )
            semantic_scores = (
                (all_semantic_scores or {}).get(str(geometry_record["name"]))
                if geometry_record is not None
                else None
            )
            try:
                filters = combine_filters(
                    texts,
                    literal_query=literal_query or "",
                    case_sensitive="case" in options,
                    regex="regex" in options,
                    semantic_scores=semantic_scores,
                    semantic_range=(
                        tuple(semantic_range)
                        if semantic_scores is not None and semantic_range
                        else None
                    ),
                )
            except InvalidRegexError as error:
                return (
                    no_update,
                    no_update,
                    f"Fix the regular expression before navigating: {error}",
                )

            uncoded_mask = _explore_uncoded_mask(
                project,
                units,
                enabled="uncoded_only" in set(uncoded_only_values or []),
            )
            positions, _page, _combined_mask, _metadata_errors = _metadata_filtered_page(
                units,
                metadata_filters or [],
                operator=metadata_filter_operator,
                page_size=_coerce_positive_int(page_size, default=5000),
                page_number=_coerce_positive_int(page_number, default=1),
                additional_mask=uncoded_mask,
            )
            page_ids = np.asarray([unit_ids[int(position)] for position in positions], dtype=int)
            avoid_seen = "new_only" in (new_only_values or [])
            seen = project.database.seen_unit_ids(int(session_id)) if avoid_seen else set()
            if strategy in {"most_similar", "least_similar"}:
                if semantic_scores is None:
                    return no_update, no_update, "Run a semantic search in the selected geometry first."
                page_scores = np.asarray(semantic_scores, dtype=float)[positions]
                page_eligible = np.asarray(filters.eligible, dtype=bool)[positions]
                candidates = [
                    index
                    for index, unit_id_value in enumerate(page_ids.tolist())
                    if bool(page_eligible[index]) and int(unit_id_value) not in seen
                ]
                if not candidates:
                    qualifier = "new " if avoid_seen else ""
                    return (
                        no_update,
                        no_update,
                        f"No {qualifier}unit satisfies the current page and filters.",
                    )
                selected_index = (
                    max(candidates, key=lambda index: float(page_scores[index]))
                    if strategy == "most_similar"
                    else min(candidates, key=lambda index: float(page_scores[index]))
                )
                unit_id = int(page_ids[selected_index])
                method = str(strategy)
            else:
                matrix = None
                if strategy in {"nearest", "farthest"}:
                    if geometry_id is None:
                        return no_update, no_update, "Choose a geometry before using nearest or farthest."
                    if focal_unit_id is None:
                        return no_update, no_update, "Select a point on the current page first."
                    if int(focal_unit_id) not in set(page_ids.tolist()):
                        return (
                            no_update,
                            no_update,
                            "The focal point is not on the current page. "
                            "Select a point on this page first.",
                        )
                    matrix = project.geometry_matrix(int(geometry_id))

                try:
                    unit_id = recommend_page_unit(
                        strategy=strategy,  # type: ignore[arg-type]
                        unit_ids=unit_ids,
                        page_positions=positions,
                        seen_unit_ids=seen,
                        eligible=filters.eligible,
                        matrix=matrix,
                        focal_unit_id=(
                            int(focal_unit_id)
                            if focal_unit_id is not None
                            and int(focal_unit_id) in set(page_ids.tolist())
                            else None
                        ),
                    )
                except (KeyError, ValueError) as error:
                    return no_update, no_update, str(error)
                if unit_id is None:
                    qualifier = "new " if avoid_seen else ""
                    return (
                        no_update,
                        no_update,
                        f"No {qualifier}unit satisfies the current page and filters.",
                    )
                method = str(strategy)

        project.record_visit(
            int(session_id),
            unit_id,
            method=method,
            source_unit_id=int(focal_unit_id) if focal_unit_id is not None else None,
            geometry_id=int(geometry_id) if geometry_id is not None else None,
        )
        updated_history = _append_navigation_history(history, int(unit_id))
        return unit_id, updated_history, ""

    @app.callback(
        Output("focal-unit-store", "data", allow_duplicate=True),
        Output("navigation-history-store", "data", allow_duplicate=True),
        Input("nav-back", "n_clicks"),
        Input("nav-forward", "n_clicks"),
        State("navigation-history-store", "data"),
        State("session-dropdown", "value"),
        State("geometry-dropdown", "value"),
        prevent_initial_call=True,
    )
    def navigate_history(
        back_clicks: int,
        forward_clicks: int,
        history: dict[str, Any] | None,
        session_id: int,
        geometry_id: int | None,
    ) -> tuple[int | Any, dict[str, Any] | Any]:
        del back_clicks, forward_clicks
        direction = -1 if callback_context.triggered_id == "nav-back" else 1
        moved = _move_navigation_history(history, direction)
        if moved is None:
            return no_update, no_update
        unit_id, updated = moved
        project.record_visit(
            int(session_id),
            int(unit_id),
            method="history_back" if direction < 0 else "history_forward",
            geometry_id=int(geometry_id) if geometry_id is not None else None,
        )
        return int(unit_id), updated

    @app.callback(
        Output("nav-back", "disabled"),
        Output("nav-forward", "disabled"),
        Input("navigation-history-store", "data"),
    )
    def set_history_button_state(
        history: dict[str, Any] | None,
    ) -> tuple[bool, bool]:
        normalized = _normalize_navigation_history(history)
        index = int(normalized["index"])
        items = list(normalized["items"])
        return index <= 0, index < 0 or index >= len(items) - 1

    @app.callback(
        Output("focal-key", "children"),
        Output("context-content", "children"),
        Output("metadata-content", "children"),
        Input("explore-focal-unit-store", "data"),
        Input("context-level", "value"),
        Input("context-window", "value"),
    )
    def update_reading_panel(
        focal_unit_id: int | None,
        level_index: int,
        window: int,
    ) -> tuple[Any, Any, Any]:
        if focal_unit_id is None:
            placeholder = "Select a point or use a navigation command to begin reading."
            return "No focal unit", placeholder, ""
        focal = project.unit(int(focal_unit_id))
        context = project.context(int(focal_unit_id), level=int(level_index), window=int(window))
        key_label = " › ".join(f"{name}: {value}" for name, value in focal["user_key"].items())
        context_children = _render_context_units(
            html,
            dcc,
            context,
            focal_unit_id=int(focal_unit_id),
            key_columns=list(metadata["key_columns"]),
            selected_unit_ids=[int(focal_unit_id)],
            checkbox_type="explore-span-checkbox",
            span_selection_enabled=project.can_transform_new_observations(),
        )
        metadata_children = _metadata_table(html, focal["metadata"])
        return key_label, context_children, metadata_children

    @app.callback(
        Output("explore-span-selection-store", "data"),
        Input("explore-focal-unit-store", "data"),
        Input("context-level", "value"),
        Input("context-window", "value"),
    )
    def reset_explore_span_selection(
        focal_unit_id: int | None, level_index: int, window: int
    ) -> list[int]:
        del level_index, window
        return [int(focal_unit_id)] if focal_unit_id is not None else []

    @app.callback(
        Output("explore-span-selection-store", "data", allow_duplicate=True),
        Input({"type": "explore-span-checkbox", "unit_id": ALL}, "value"),
        State({"type": "explore-span-checkbox", "unit_id": ALL}, "id"),
        State("focal-unit-store", "data"),
        State("context-level", "value"),
        State("context-window", "value"),
        State("explore-span-selection-store", "data"),
        prevent_initial_call=True,
    )
    def update_explore_span_selection(
        values: list[list[int]], ids: list[dict[str, Any]],
        focal_unit_id: int | None, level_index: int, window: int,
        current: list[int] | None,
    ) -> Any:
        triggered = callback_context.triggered_id
        if focal_unit_id is None or not isinstance(triggered, dict):
            return no_update
        clicked = int(triggered.get("unit_id"))
        value_by_id = {
            int(identifier["unit_id"]): value
            for identifier, value in zip(ids or [], values or [], strict=False)
        }
        checked = bool(value_by_id.get(clicked))
        if not checked and clicked not in set(current or []):
            return no_update
        context = project.context(
            int(focal_unit_id), level=int(level_index), window=int(window)
        )
        updated = _update_span_selection(
            context, focal_unit_id=int(focal_unit_id), current_unit_ids=current,
            clicked_unit_id=clicked, checked=checked,
        )
        return updated if updated != list(current or []) else no_update

    @app.callback(
        Output("context-window", "value", allow_duplicate=True),
        Input("context-level", "value"),
        Input("context-window-minus", "n_clicks"),
        Input("context-window-plus", "n_clicks"),
        State("context-window", "value"),
        prevent_initial_call=True,
    )
    def update_context_window(
        level_index: int,
        minus_clicks: int,
        plus_clicks: int,
        current_window: int | None,
    ) -> int:
        del level_index, minus_clicks, plus_clicks
        triggered = callback_context.triggered_id
        if triggered == "context-level":
            return 1
        current = max(1, int(current_window or 1))
        if triggered == "context-window-minus":
            return max(1, current - 1)
        if triggered == "context-window-plus":
            return current + 1
        return current

    @app.callback(
        Output("memo-dropdown", "options"),
        Input("memo-refresh-store", "data"),
    )
    def refresh_memo_options(refresh: int | None) -> list[dict[str, Any]]:
        del refresh
        return _memo_options(project.memos())

    @app.callback(
        Output("active-memo-store", "data", allow_duplicate=True),
        Output("memo-title", "value", allow_duplicate=True),
        Output("memo-body", "value", allow_duplicate=True),
        Output("memo-status", "children", allow_duplicate=True),
        Input("memo-dropdown", "value"),
        prevent_initial_call=True,
    )
    def load_memo(memo_id: int | None) -> tuple[Any, str, str, str]:
        if memo_id is None:
            return None, "", "", ""
        memo = project.memo(int(memo_id))
        return int(memo_id), str(memo["title"]), str(memo["body_markdown"]), ""

    @app.callback(
        Output("memo-dropdown", "value", allow_duplicate=True),
        Output("active-memo-store", "data", allow_duplicate=True),
        Output("memo-title", "value", allow_duplicate=True),
        Output("memo-body", "value", allow_duplicate=True),
        Output("memo-status", "children", allow_duplicate=True),
        Input("memo-new-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def new_memo(n_clicks: int) -> tuple[None, None, str, str, str]:
        del n_clicks
        return None, None, "", "", "New memo ready."

    @app.callback(
        Output("memo-dropdown", "options", allow_duplicate=True),
        Output("memo-dropdown", "value", allow_duplicate=True),
        Output("active-memo-store", "data", allow_duplicate=True),
        Output("memo-refresh-store", "data", allow_duplicate=True),
        Output("memo-status", "children", allow_duplicate=True),
        Output("memo-title", "value", allow_duplicate=True),
        Output("memo-body", "value", allow_duplicate=True),
        Input("memo-save-button", "n_clicks"),
        State("active-memo-store", "data"),
        State("memo-title", "value"),
        State("memo-body", "value"),
        State("memo-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def save_memo(
        n_clicks: int,
        memo_id: int | None,
        title: str | None,
        body: str | None,
        refresh: int | None,
    ) -> tuple[Any, int | None, int | None, int, str, Any, Any]:
        del n_clicks
        clean_title = (title or "").strip()
        if not clean_title:
            return (
                no_update, no_update, no_update, int(refresh or 0),
                "Enter a memo title.", no_update, no_update,
            )
        clean_body = body or ""
        hashtags, references = _extract_memo_metadata(clean_body)
        if memo_id is None:
            saved_id = project.create_memo(
                title=clean_title,
                body_markdown=clean_body,
                hashtags=hashtags,
                unit_references=references,
            )
            message = f"Created memo {clean_title!r}."
        else:
            saved_id = int(memo_id)
            version_id = project.update_memo(
                saved_id,
                title=clean_title,
                body_markdown=clean_body,
                hashtags=hashtags,
                unit_references=references,
            )
            del version_id
            message = f"Saved a new version of {clean_title!r}."
        options = _memo_options(project.memos())
        return (
            options, saved_id, saved_id, int(refresh or 0) + 1,
            message, clean_title, clean_body,
        )

    app.clientside_callback(
        """
        function(nClicks, body, focalUnitId) {
            if (!nClicks || focalUnitId === null || focalUnitId === undefined) {
                return window.dash_clientside.no_update;
            }
            const textarea = document.getElementById("memo-body");
            const current = body || "";
            const start = textarea && Number.isInteger(textarea.selectionStart)
                ? textarea.selectionStart : current.length;
            const end = textarea && Number.isInteger(textarea.selectionEnd)
                ? textarea.selectionEnd : start;
            const token = `[[unit:${focalUnitId}]]`;
            const before = current.slice(0, start);
            const after = current.slice(end);
            const prefix = before && !/[\\s]$/.test(before) ? " " : "";
            const suffix = after && !/^[\\s]/.test(after) ? " " : "";
            const insertion = prefix + token + suffix;
            const next = before + insertion + after;
            const cursor = before.length + insertion.length;
            window.setTimeout(function() {
                const el = document.getElementById("memo-body");
                if (el) {
                    el.focus();
                    el.setSelectionRange(cursor, cursor);
                }
            }, 0);
            return next;
        }
        """,
        Output("memo-body", "value", allow_duplicate=True),
        Input("memo-insert-reference-button", "n_clicks"),
        State("memo-body", "value"),
        State("focal-unit-store", "data"),
        prevent_initial_call=True,
    )

    @app.callback(
        Output("memo-preview", "children"),
        Input("memo-body", "value"),
    )
    def preview_memo(body: str | None) -> Any:
        return _render_memo_preview(dcc, html, project, body or "")

    @app.callback(
        Output("memo-center-tags", "options"),
        Input("memo-refresh-store", "data"),
    )
    def refresh_memo_center_hashtags(refresh: int | None) -> list[dict[str, str]]:
        del refresh
        return [
            {"label": f"#{tag}", "value": tag}
            for tag in project.memo_hashtags()
        ]

    @app.callback(
        Output("memo-center-table", "data"),
        Output("memo-center-table", "selected_rows"),
        Output("memo-center-status", "children"),
        Input("memo-center-query", "value"),
        Input("memo-center-scope", "value"),
        Input("memo-center-tags", "value"),
        Input("memo-center-tag-operator", "value"),
        Input("memo-refresh-store", "data"),
        State("memo-center-selected-store", "data"),
    )
    def filter_memo_center(
        query: str | None,
        scope: str | None,
        hashtags: list[str] | None,
        operator: str | None,
        refresh: int | None,
        selected_memo_id: int | None,
    ) -> tuple[list[dict[str, Any]], list[int], str]:
        del refresh
        matches = project.search_memos(
            query=query or "",
            scope=str(scope or "full"),
            hashtags=hashtags or [],
            hashtag_operator=str(operator or "and"),
        )
        rows = [
            {
                "memo_id": int(memo["memo_id"]),
                "title": str(memo["title"]),
                "hashtags": " ".join(f"#{tag}" for tag in memo["hashtags"]),
                "updated_at": str(memo["updated_at"]),
            }
            for memo in matches
        ]
        selected_rows = [
            index for index, row in enumerate(rows)
            if selected_memo_id is not None and int(row["memo_id"]) == int(selected_memo_id)
        ]
        return rows, selected_rows[:1], f"{len(rows):,} memo{'s' if len(rows) != 1 else ''} found"

    @app.callback(
        Output("memo-center-selected-store", "data"),
        Output("memo-center-version-store", "data"),
        Output("memo-center-title", "children"),
        Output("memo-center-preview", "children"),
        Output("memo-version-label", "children"),
        Output("memo-version-previous", "disabled"),
        Output("memo-version-next", "disabled"),
        Output("memo-center-edit-button", "disabled"),
        Output("memo-center-edit-title", "value"),
        Output("memo-center-edit-body", "value"),
        Input("memo-center-table", "selected_rows"),
        State("memo-center-table", "data"),
        Input("memo-refresh-store", "data"),
    )
    def select_memo_center(
        selected_rows: list[int] | None, rows: list[dict[str, Any]] | None, refresh: int | None
    ) -> tuple[Any, ...]:
        del refresh
        if not selected_rows or not rows:
            empty = html.P("Choose a memo from the results to read it here.", className="memo-preview-empty")
            return None, None, "Select a memo", empty, "No version selected", True, True, True, "", ""
        index = int(selected_rows[0])
        if index < 0 or index >= len(rows):
            return None, None, "Select a memo", "", "No version selected", True, True, True, "", ""
        memo_id = int(rows[index]["memo_id"])
        versions = project.memo_versions(memo_id)
        memo = versions[-1]
        number = int(memo["version_number"])
        return (
            memo_id, number, str(memo["title"]),
            _render_memo_preview(dcc, html, project, str(memo["body_markdown"])),
            f"Version {number} of {len(versions)}", number <= 1, number >= len(versions), False,
            str(memo["title"]), str(memo["body_markdown"]),
        )

    @app.callback(
        Output("memo-center-version-store", "data", allow_duplicate=True),
        Output("memo-center-title", "children", allow_duplicate=True),
        Output("memo-center-preview", "children", allow_duplicate=True),
        Output("memo-version-label", "children", allow_duplicate=True),
        Output("memo-version-previous", "disabled", allow_duplicate=True),
        Output("memo-version-next", "disabled", allow_duplicate=True),
        Output("memo-center-edit-title", "value", allow_duplicate=True),
        Output("memo-center-edit-body", "value", allow_duplicate=True),
        Input("memo-version-previous", "n_clicks"),
        Input("memo-version-next", "n_clicks"),
        State("memo-center-selected-store", "data"),
        State("memo-center-version-store", "data"),
        prevent_initial_call=True,
    )
    def navigate_memo_versions(
        previous: int, next_clicks: int, memo_id: int | None, version_number: int | None
    ) -> tuple[Any, ...]:
        del previous, next_clicks
        if memo_id is None:
            return (no_update,) * 8
        versions = project.memo_versions(int(memo_id))
        current = int(version_number or len(versions))
        if callback_context.triggered_id == "memo-version-previous":
            target = max(1, current - 1)
        else:
            target = min(len(versions), current + 1)
        memo = project.memo(int(memo_id), target)
        return (
            target, str(memo["title"]),
            _render_memo_preview(dcc, html, project, str(memo["body_markdown"])),
            f"Version {target} of {len(versions)}", target <= 1, target >= len(versions),
            str(memo["title"]), str(memo["body_markdown"]),
        )

    @app.callback(
        Output("memo-center-edit-store", "data"),
        Output("memo-center-editor-wrap", "style"),
        Output("memo-center-preview", "style", allow_duplicate=True),
        Input("memo-center-edit-button", "n_clicks"),
        Input("memo-center-cancel-button", "n_clicks"),
        State("memo-center-selected-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_memo_center_edit(
        edit_clicks: int, cancel_clicks: int, memo_id: int | None
    ) -> tuple[bool, dict[str, str], dict[str, str]]:
        del edit_clicks, cancel_clicks
        editing = callback_context.triggered_id == "memo-center-edit-button" and memo_id is not None
        return (
            editing,
            ({"display": "block"} if editing else {"display": "none"}),
            ({"display": "none"} if editing else {}),
        )

    @app.callback(
        Output("memo-center-edit-preview", "children"),
        Input("memo-center-edit-body", "value"),
    )
    def preview_memo_center_edit(body: str | None) -> Any:
        return _render_memo_preview(dcc, html, project, body or "")

    @app.callback(
        Output("memo-refresh-store", "data", allow_duplicate=True),
        Output("memo-center-version-store", "data", allow_duplicate=True),
        Output("memo-center-title", "children", allow_duplicate=True),
        Output("memo-center-preview", "children", allow_duplicate=True),
        Output("memo-version-label", "children", allow_duplicate=True),
        Output("memo-version-previous", "disabled", allow_duplicate=True),
        Output("memo-version-next", "disabled", allow_duplicate=True),
        Output("memo-center-edit-store", "data", allow_duplicate=True),
        Output("memo-center-editor-wrap", "style", allow_duplicate=True),
        Output("memo-center-preview", "style", allow_duplicate=True),
        Output("memo-center-edit-status", "children"),
        Input("memo-center-save-button", "n_clicks"),
        State("memo-center-selected-store", "data"),
        State("memo-center-edit-title", "value"),
        State("memo-center-edit-body", "value"),
        State("memo-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def save_memo_center_version(
        clicks: int, memo_id: int | None, title: str | None, body: str | None, refresh: int | None
    ) -> tuple[Any, ...]:
        del clicks
        if memo_id is None:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, True, {"display": "none"}, {}, "Select a memo first."
        clean_title = (title or "").strip()
        if not clean_title:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, True, {"display": "block"}, {"display": "none"}, "Enter a memo title."
        clean_body = body or ""
        hashtags, references = _extract_memo_metadata(clean_body)
        project.update_memo(int(memo_id), title=clean_title, body_markdown=clean_body, hashtags=hashtags, unit_references=references)
        versions = project.memo_versions(int(memo_id))
        number = len(versions)
        memo = versions[-1]
        return (
            int(refresh or 0) + 1, number, clean_title,
            _render_memo_preview(dcc, html, project, clean_body),
            f"Version {number} of {number}", number <= 1, True, False, {"display": "none"}, {},
            f"Saved version {number}.",
        )


    @app.callback(
        Output("memo-presentation-link", "href"),
        Output("memo-presentation-link", "className"),
        Input("memo-center-selected-store", "data"),
        Input("memo-center-version-store", "data"),
    )
    def update_memo_presentation_link(
        memo_id: int | None, version_number: int | None
    ) -> tuple[str | None, str]:
        base_class = "memo-presentation-link"
        if memo_id is None or version_number is None:
            return None, f"{base_class} disabled"
        return (
            f"/?presentation=memo&memo_id={int(memo_id)}&version={int(version_number)}",
            base_class,
        )

    @app.callback(
        Output("memo-presentation-shell", "className"),
        Input("memo-presentation-location", "search"),
    )
    def show_memo_presentation(search: str | None) -> str:
        values = parse_qs((search or "").lstrip("?"))
        if values.get("presentation", [None])[0] == "memo":
            return "memo-presentation-shell memo-presentation-active"
        return "memo-presentation-shell"

    @app.callback(
        Output("memo-presentation-saved-filter", "options"),
        Input("memo-presentation-location", "pathname"),
        Input("saved-filter-refresh-store", "data"),
    )
    def refresh_presentation_saved_filters(
        pathname: str | None, refresh: int | None
    ) -> list[dict[str, str]]:
        del pathname, refresh
        options = [{"label": "All observations", "value": "__all__"}]
        options.extend(
            {
                "label": str(record.get("name", "Untitled filter")),
                "value": str(record.get("id")),
            }
            for record in _read_saved_filter_sets(saved_filter_sets_path)
        )
        return options

    @app.callback(
        Output("memo-presentation-title", "children"),
        Output("memo-presentation-version", "children"),
        Output("memo-presentation-body", "children"),
        Input("memo-presentation-location", "search"),
    )
    def load_memo_presentation(search: str | None) -> tuple[str, str, Any]:
        memo_id, requested_version = _parse_memo_presentation_request(search)
        if memo_id is None:
            return (
                "Memo presentation",
                "No memo selected",
                html.P(
                    "Open Presentation mode from a memo in Memo Center.",
                    className="memo-preview-empty",
                ),
            )
        try:
            versions = project.memo_versions(int(memo_id))
        except KeyError:
            versions = []
        if not versions:
            return (
                "Memo presentation",
                "Memo not found",
                html.P("This memo is unavailable.", className="memo-preview-empty"),
            )
        memo = _memo_version_by_number(versions, requested_version)
        number = int(memo["version_number"])
        return (
            str(memo["title"]),
            f"Version {number} of {len(versions)}",
            _render_memo_preview(
                dcc,
                html,
                project,
                str(memo["body_markdown"]),
                unit_hash_prefix="geco-presentation-unit-",
                link_hashtags=False,
            ),
        )

    @app.callback(
        Output("memo-presentation-page-number", "max"),
        Output("memo-presentation-page-number", "value"),
        Input("memo-presentation-page-previous", "n_clicks"),
        Input("memo-presentation-page-next", "n_clicks"),
        Input("memo-presentation-page-size", "value"),
        Input("memo-presentation-saved-filter", "value"),
        Input("memo-presentation-location", "hash"),
        State("memo-presentation-page-number", "value"),
    )
    def update_memo_presentation_page(
        previous_clicks: int,
        next_clicks: int,
        page_size: Any,
        saved_filter_id: str | None,
        location_hash: str | None,
        current_page: Any,
    ) -> tuple[int, int]:
        del previous_clicks, next_clicks
        size = _coerce_positive_int(page_size, default=5000)
        filters, operator = _presentation_filter_definition(
            saved_filter_sets_path,
            saved_filter_id,
            metadata_profiles,
        )
        mask, _errors = _evaluate_metadata_filters(units, filters, operator=operator)
        page_count = deterministic_page(int(mask.sum()), page_size=size).page_count
        current = min(max(1, _coerce_positive_int(current_page, default=1)), page_count)
        triggered = callback_context.triggered_id
        if triggered in {"memo-presentation-page-size", "memo-presentation-saved-filter"}:
            current = 1
        elif triggered == "memo-presentation-page-previous":
            current = max(1, current - 1)
        elif triggered == "memo-presentation-page-next":
            current = min(page_count, current + 1)
        elif triggered == "memo-presentation-location":
            focal_unit_id = _presentation_unit_id(location_hash)
            if focal_unit_id is not None and focal_unit_id in unit_position_by_id:
                target = _deterministic_page_number_for_position(
                    mask,
                    corpus_position=unit_position_by_id[focal_unit_id],
                    page_size=size,
                )
                if target is not None:
                    if target == current:
                        # Avoid emitting a redundant page-value update. Otherwise
                        # the map can redraw a second time with page-number as the
                        # apparent trigger and Plotly may discard the zoom camera.
                        return page_count, no_update
                    current = target
                else:
                    # Out-of-filter references are overlaid on the current page.
                    # Keep the page input untouched so focal selection is the only
                    # redraw trigger and the current camera can be preserved.
                    return page_count, no_update
        return page_count, current

    @app.callback(
        Output("memo-presentation-map", "figure"),
        Output("memo-presentation-page-status", "children"),
        Input("memo-presentation-view", "value"),
        Input("memo-presentation-saved-filter", "value"),
        Input("memo-presentation-page-size", "value"),
        Input("memo-presentation-page-number", "value"),
        Input("memo-presentation-location", "hash"),
        State("memo-presentation-map", "relayoutData"),
    )
    def update_memo_presentation_map(
        view_id: int | None,
        saved_filter_id: str | None,
        page_size: Any,
        page_number: Any,
        location_hash: str | None,
        relayout_data: dict[str, Any] | None,
    ) -> tuple[Any, str]:
        if view_id is None or int(view_id) not in view_by_id:
            return _empty_figure("No computed 2D view is available."), ""
        view_record = view_by_id[int(view_id)]
        geometry_id = int(view_record["geometry_id"])
        geometry_record = geometry_by_id.get(geometry_id)
        if geometry_record is None:
            return _empty_figure("This 2D view does not belong to a public geometry."), ""
        size = _coerce_positive_int(page_size, default=5000)
        number = _coerce_positive_int(page_number, default=1)
        filters, operator = _presentation_filter_definition(
            saved_filter_sets_path,
            saved_filter_id,
            metadata_profiles,
        )
        positions, page, mask, _errors = _metadata_filtered_page(
            units,
            filters,
            operator=operator,
            page_size=size,
            page_number=number,
        )
        focal_unit_id = _presentation_unit_id(location_hash)
        display_positions = positions.tolist()
        if focal_unit_id is not None and focal_unit_id in unit_position_by_id:
            focal_position = int(unit_position_by_id[focal_unit_id])
            if focal_position not in display_positions:
                display_positions.append(focal_position)
        display = np.asarray(display_positions, dtype=int)
        coordinates = project.view_coordinates(int(view_id))[display]
        page_units = [units[int(position)] for position in display]
        eligible = mask[display] if display.size else np.asarray([], dtype=bool)
        figure = build_geometry_figure(
            coordinates=coordinates,
            units=page_units,
            eligible=eligible,
            seen_unit_ids=(),
            focal_unit_id=focal_unit_id,
            geometry_name=str(geometry_record["name"]),
            view_name=str(view_record["name"]),
            page_label=f"Page {page.page_index + 1} of {page.page_count}",
            uirevision=f"memo-presentation:{int(view_id)}:page:{page.page_index}",
        )
        # Focal selection rebuilds the figure so an out-of-filter memo reference
        # can be overlaid. Preserve the user's current camera when only the focal
        # observation changes; deliberate view/filter/page changes should reset
        # the camera to the newly requested surface.
        if callback_context.triggered_id not in {
            "memo-presentation-view",
            "memo-presentation-saved-filter",
            "memo-presentation-page-size",
            "memo-presentation-page-number",
        }:
            _apply_relayout_ranges(figure, relayout_data)
        status = (
            f"Page {page.page_index + 1} of {page.page_count} · "
            f"{len(positions):,} filter points · {int(mask.sum()):,} total matches"
        )
        return figure, status

    @app.callback(
        Output("memo-presentation-location", "hash", allow_duplicate=True),
        Input("memo-presentation-map", "clickData"),
        prevent_initial_call=True,
    )
    def select_memo_presentation_map_point(
        click_data: dict[str, Any] | None,
    ) -> Any:
        unit_id = _clicked_unit_id(click_data)
        if unit_id is None:
            return no_update
        return f"#geco-presentation-unit-{int(unit_id)}"

    @app.callback(
        Output("memo-presentation-focal-key", "children"),
        Output("memo-presentation-text", "children"),
        Output("memo-presentation-metadata", "children"),
        Input("memo-presentation-location", "hash"),
    )
    def update_memo_presentation_reader(
        location_hash: str | None,
    ) -> tuple[Any, Any, Any]:
        unit_id = _presentation_unit_id(location_hash)
        if unit_id is None:
            return (
                "No focal observation",
                "Select a linked observation in the memo or click a point on the map.",
                "",
            )
        try:
            focal = project.unit(int(unit_id))
        except KeyError:
            return "Observation unavailable", "The referenced observation could not be loaded.", ""
        key_label = " › ".join(
            f"{name}: {value}" for name, value in focal["user_key"].items()
        )
        context_level = int(hierarchy[-1]["level_index"])
        context = project.context(int(unit_id), level=context_level, window=5)
        return (
            key_label,
            _render_readonly_context_units(
                html,
                context,
                focal_unit_id=int(unit_id),
                key_columns=list(metadata["key_columns"]),
            ),
            _metadata_table(html, focal["metadata"]),
        )

    @app.callback(
        Output("focal-unit-store", "data", allow_duplicate=True),
        Output("navigation-history-store", "data", allow_duplicate=True),
        Output("workspace-tabs", "value", allow_duplicate=True),
        Output("memo-center-tags", "value", allow_duplicate=True),
        Output("memo-navigation-location", "hash"),
        Input("memo-navigation-location", "hash"),
        State("session-dropdown", "value"),
        State("geometry-dropdown", "value"),
        State("focal-unit-store", "data"),
        State("navigation-history-store", "data"),
        prevent_initial_call=True,
    )
    def follow_memo_link(
        location_hash: str | None,
        session_id: int,
        geometry_id: int | None,
        focal_unit_id: int | None,
        history: dict[str, Any] | None,
    ) -> tuple[Any, Any, Any, Any, Any]:
        unit_match = re.fullmatch(r"#geco-unit-(\d+)", location_hash or "")
        if unit_match is not None:
            unit_id = int(unit_match.group(1))
            project.record_visit(
                int(session_id),
                unit_id,
                method="memo_reference",
                source_unit_id=int(focal_unit_id) if focal_unit_id is not None else None,
                geometry_id=int(geometry_id) if geometry_id is not None else None,
            )
            return (
                unit_id,
                _append_navigation_history(history, unit_id),
                "explore",
                no_update,
                "",
            )

        tag_match = re.fullmatch(r"#geco-tag-(.+)", location_hash or "")
        if tag_match is not None:
            return (
                no_update,
                no_update,
                "memos",
                [unquote(tag_match.group(1))],
                "",
            )
        return no_update, no_update, no_update, no_update, no_update

    @app.callback(
        Output("code-center-tags", "options"),
        Input("explore-code-refresh-store", "data"),
        Input("focus-refresh-store", "data"),
    )
    def refresh_code_center_hashtags(
        explore_refresh: int | None, focus_refresh: int | None
    ) -> list[dict[str, str]]:
        del explore_refresh, focus_refresh
        return [{"label": f"#{tag}", "value": tag} for tag in project.code_hashtags()]

    @app.callback(
        Output("code-center-table", "data"),
        Output("code-center-table", "selected_rows"),
        Output("code-center-status", "children"),
        Input("code-center-query", "value"),
        Input("code-center-scope", "value"),
        Input("code-center-tags", "value"),
        Input("explore-code-refresh-store", "data"),
        Input("focus-refresh-store", "data"),
        State("code-center-selected-store", "data"),
    )
    def filter_code_center(
        query: str | None, scope: str | None, hashtags: list[str] | None,
        explore_refresh: int | None, focus_refresh: int | None,
        selected_code_id: int | None,
    ) -> tuple[list[dict[str, Any]], list[int], str]:
        del explore_refresh, focus_refresh
        matches = project.search_codes(query=query or "", scope=str(scope or "full"), hashtags=hashtags or [])
        rows = [{
            "code_id": int(code["code_id"]), "name": str(code["name"]),
            "hashtags": " ".join(f"#{tag}" for tag in code.get("hashtags", [])),
        } for code in matches]
        selected_rows = [
            index for index, row in enumerate(rows)
            if selected_code_id is not None and int(row["code_id"]) == int(selected_code_id)
        ]
        return rows, selected_rows[:1], f"{len(rows):,} code{'s' if len(rows) != 1 else ''} found"

    @app.callback(
        Output("code-center-selected-store", "data"),
        Output("code-center-version-store", "data"),
        Output("code-center-title", "children"),
        Output("code-center-description", "children"),
        Output("code-version-label", "children"),
        Output("code-version-previous", "disabled"),
        Output("code-version-next", "disabled"),
        Output("code-center-edit-button", "disabled"),
        Output("code-center-edit-name", "value"),
        Output("code-center-edit-description", "value"),
        Input("code-center-table", "selected_rows"),
        State("code-center-table", "data"),
        Input("explore-code-refresh-store", "data"),
        Input("focus-refresh-store", "data"),
    )
    def select_code_center(
        selected_rows: list[int] | None, rows: list[dict[str, Any]] | None,
        explore_refresh: int | None, focus_refresh: int | None,
    ) -> tuple[Any, ...]:
        del explore_refresh, focus_refresh
        if not selected_rows or not rows:
            empty = html.P("Choose a code to inspect its description and positive units.", className="memo-preview-empty")
            return None, None, "Select a code", empty, "No version selected", True, True, True, "", ""
        index = int(selected_rows[0])
        if index < 0 or index >= len(rows):
            return None, None, "Select a code", "", "No version selected", True, True, True, "", ""
        code_id = int(rows[index]["code_id"])
        versions = project.code_versions(code_id)
        code = project.code(code_id)
        number = int(code["version_number"])
        description = str(code["description"])
        return (
            code_id, number, str(code["name"]),
            (_render_memo_preview(dcc, html, project, description) if description.strip() else None),
            f"Version {number} of {len(versions)}", number <= 1, number >= len(versions), False,
            str(code["name"]), description,
        )

    @app.callback(
        Output("code-center-version-store", "data", allow_duplicate=True),
        Output("code-center-title", "children", allow_duplicate=True),
        Output("code-center-description", "children", allow_duplicate=True),
        Output("code-version-label", "children", allow_duplicate=True),
        Output("code-version-previous", "disabled", allow_duplicate=True),
        Output("code-version-next", "disabled", allow_duplicate=True),
        Output("code-center-edit-name", "value", allow_duplicate=True),
        Output("code-center-edit-description", "value", allow_duplicate=True),
        Input("code-version-previous", "n_clicks"), Input("code-version-next", "n_clicks"),
        State("code-center-selected-store", "data"), State("code-center-version-store", "data"),
        prevent_initial_call=True,
    )
    def navigate_code_versions(
        previous: int, next_clicks: int, code_id: int | None, version_number: int | None
    ) -> tuple[Any, ...]:
        del previous, next_clicks
        if code_id is None:
            return (no_update,) * 8
        versions = project.code_versions(int(code_id))
        current = int(version_number or len(versions))
        target = max(1, current - 1) if callback_context.triggered_id == "code-version-previous" else min(len(versions), current + 1)
        code = project.code(int(code_id), target)
        description = str(code["description"])
        return (
            target, str(code["name"]),
            (_render_memo_preview(dcc, html, project, description) if description.strip() else None),
            f"Version {target} of {len(versions)}", target <= 1, target >= len(versions),
            str(code["name"]), description,
        )

    @app.callback(
        Output("code-center-edit-store", "data"),
        Output("code-center-editor-wrap", "style"),
        Output("code-center-description", "style", allow_duplicate=True),
        Input("code-center-edit-button", "n_clicks"), Input("code-center-cancel-button", "n_clicks"),
        State("code-center-selected-store", "data"), prevent_initial_call=True,
    )
    def toggle_code_center_edit(
        edit_clicks: int, cancel_clicks: int, code_id: int | None
    ) -> tuple[bool, dict[str, str], dict[str, str]]:
        del edit_clicks, cancel_clicks
        editing = callback_context.triggered_id == "code-center-edit-button" and code_id is not None
        return (
            editing,
            ({"display": "block"} if editing else {"display": "none"}),
            ({"display": "none"} if editing else {}),
        )

    @app.callback(Output("code-center-edit-preview", "children"), Input("code-center-edit-description", "value"))
    def preview_code_center_edit(description: str | None) -> Any:
        return _render_memo_preview(dcc, html, project, description or "")

    @app.callback(
        Output("explore-code-refresh-store", "data", allow_duplicate=True),
        Output("focus-refresh-store", "data", allow_duplicate=True),
        Output("code-center-version-store", "data", allow_duplicate=True),
        Output("code-center-title", "children", allow_duplicate=True),
        Output("code-center-description", "children", allow_duplicate=True),
        Output("code-version-label", "children", allow_duplicate=True),
        Output("code-version-previous", "disabled", allow_duplicate=True),
        Output("code-version-next", "disabled", allow_duplicate=True),
        Output("code-center-edit-store", "data", allow_duplicate=True),
        Output("code-center-editor-wrap", "style", allow_duplicate=True),
        Output("code-center-description", "style", allow_duplicate=True),
        Output("code-center-edit-status", "children"),
        Input("code-center-save-button", "n_clicks"),
        State("code-center-selected-store", "data"), State("code-center-edit-name", "value"),
        State("code-center-edit-description", "value"), State("explore-code-refresh-store", "data"),
        State("focus-refresh-store", "data"), prevent_initial_call=True,
    )
    def save_code_center_version(
        clicks: int, code_id: int | None, name: str | None, description: str | None,
        explore_refresh: int | None, focus_refresh: int | None,
    ) -> tuple[Any, ...]:
        del clicks
        if code_id is None:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, True, {"display": "none"}, {}, "Select a code first."
        clean_name = (name or "").strip()
        if not clean_name:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, True, {"display": "block"}, {"display": "none"}, "Enter a code label."
        clean_description = description or ""
        project.update_code(int(code_id), name=clean_name, description=clean_description)
        versions = project.code_versions(int(code_id))
        number = len(versions)
        return (
            int(explore_refresh or 0) + 1, int(focus_refresh or 0) + 1, number, clean_name,
            (_render_memo_preview(dcc, html, project, clean_description) if clean_description.strip() else None),
            f"Version {number} of {number}",
            number <= 1, True, False, {"display": "none"}, {}, f"Saved version {number}.",
        )

    @app.callback(
        Output("code-center-teaching-button", "disabled"),
        Output("focus-teaching-button", "disabled"),
        Input("code-center-selected-store", "data"),
        Input("focus-code-dropdown", "value"),
    )
    def enable_teaching_example_manager(
        code_center_id: int | None, focus_code_id: int | None
    ) -> tuple[bool, bool]:
        can_create = project.can_transform_new_observations()
        return code_center_id is None or not can_create, focus_code_id is None or not can_create

    @app.callback(
        Output("focus-teaching-counts", "children"),
        Input("focus-code-dropdown", "value"),
        Input("teaching-example-refresh-store", "data"),
    )
    def update_focus_teaching_counts(
        code_id: int | None, refresh: int | None
    ) -> Any:
        del refresh
        if code_id is None:
            return "Select a code to manage examples."
        examples = project.teaching_examples(int(code_id))
        active = [
            example
            for example in examples
            if str(example["teaching_status"]) == "active"
        ]
        present = sum(
            str(example["teaching_label"]) == "positive" for example in active
        )
        absent = sum(
            str(example["teaching_label"]) == "negative" for example in active
        )
        inactive = sum(
            str(example["teaching_status"]) == "inactive" for example in examples
        )
        children: list[Any] = [html.Div(f"Present {present:,} · Absent {absent:,}")]
        if inactive:
            children.append(html.Small(f"{inactive:,} inactive example{'s' if inactive != 1 else ''}"))
        return children

    @app.callback(
        Output("teaching-example-modal", "style"),
        Output("teaching-example-status", "children"),
        Output("teaching-example-refresh-store", "data", allow_duplicate=True),
        Output("teaching-example-text", "value"),
        Output("teaching-example-note", "value"),
        Output("teaching-example-code-store", "data"),
        Input("code-center-teaching-button", "n_clicks"),
        Input("focus-teaching-button", "n_clicks"),
        Input("teaching-example-close", "n_clicks"),
        Input("teaching-example-save", "n_clicks"),
        State("code-center-selected-store", "data"),
        State("focus-code-dropdown", "value"),
        State("teaching-example-code-store", "data"),
        State("teaching-example-text", "value"),
        State("teaching-example-label", "value"),
        State("teaching-example-note", "value"),
        State("teaching-example-assess-before-training", "value"),
        State("teaching-example-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def manage_teaching_example_modal(
        code_center_open_clicks: int,
        focus_open_clicks: int,
        close_clicks: int,
        save_clicks: int,
        code_center_id: int | None,
        focus_code_id: int | None,
        modal_code_id: int | None,
        text: str | None,
        label: str | None,
        note: str | None,
        assess_values: list[str] | None,
        refresh: int | None,
    ) -> tuple[dict[str, str], str, int, Any, Any, Any]:
        del code_center_open_clicks, focus_open_clicks, close_clicks, save_clicks
        triggered = callback_context.triggered_id
        if triggered == "teaching-example-close":
            return (
                {"display": "none"}, "", int(refresh or 0),
                no_update, no_update, modal_code_id,
            )
        if triggered == "code-center-teaching-button":
            selected_code_id = code_center_id
        elif triggered == "focus-teaching-button":
            selected_code_id = focus_code_id
        else:
            selected_code_id = modal_code_id
        if selected_code_id is None:
            return (
                {"display": "none"}, "Select a code first.", int(refresh or 0),
                no_update, no_update, no_update,
            )
        if triggered in {"code-center-teaching-button", "focus-teaching-button"}:
            return (
                {"display": "flex"}, "", int(refresh or 0),
                no_update, no_update, int(selected_code_id),
            )
        clean_text = (text or "").strip()
        if not clean_text:
            return (
                {"display": "flex"}, "Enter example text.", int(refresh or 0),
                no_update, no_update, int(selected_code_id),
            )
        try:
            project.create_teaching_example(
                code_id=int(selected_code_id), text=clean_text,
                label=str(label or "positive"), note=(note or "").strip(),
                assess_before_training="assess" in set(assess_values or []),
            )
        except Exception as error:
            return (
                {"display": "flex"}, str(error), int(refresh or 0),
                no_update, no_update, int(selected_code_id),
            )
        return (
            {"display": "flex"}, "Added an active teaching example.",
            int(refresh or 0) + 1, "", "", int(selected_code_id),
        )

    @app.callback(
        Output("teaching-example-table", "data"),
        Output("teaching-example-table", "selected_rows"),
        Input("teaching-example-code-store", "data"),
        Input("teaching-example-refresh-store", "data"),
    )
    def load_teaching_examples(
        code_id: int | None, refresh: int | None
    ) -> tuple[list[dict[str, Any]], list[int]]:
        del refresh
        if code_id is None:
            return [], []
        rows = [
            {
                "observation_id": int(example["observation_id"]),
                "text": str(example["text"]),
                "label": "Present" if example["teaching_label"] == "positive" else "Absent",
                "status": str(example["teaching_status"]).title(),
                "created": str(example["created_at"]),
            }
            for example in project.teaching_examples(int(code_id))
        ]
        return rows, []

    @app.callback(
        Output("teaching-example-status", "children", allow_duplicate=True),
        Output("teaching-example-refresh-store", "data", allow_duplicate=True),
        Input("teaching-example-apply-status", "n_clicks"),
        State("teaching-example-table", "selected_rows"),
        State("teaching-example-table", "data"),
        State("teaching-example-status-action", "value"),
        State("teaching-example-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def change_teaching_example_status(
        clicks: int, selected_rows: list[int] | None, rows: list[dict[str, Any]] | None,
        status: str | None, refresh: int | None,
    ) -> tuple[str, int]:
        del clicks
        if not selected_rows or not rows:
            return "Select an example first.", int(refresh or 0)
        index = int(selected_rows[0])
        if index < 0 or index >= len(rows):
            return "Select an example first.", int(refresh or 0)
        project.set_teaching_example_status(
            int(rows[index]["observation_id"]), str(status or "inactive")
        )
        action = "Deleted" if status == "deleted" else str(status or "inactive").title()
        return f"{action} the selected teaching example.", int(refresh or 0) + 1

    @app.callback(
        Output("teaching-example-status", "children", allow_duplicate=True),
        Output("teaching-example-refresh-store", "data", allow_duplicate=True),
        Input("teaching-example-apply-label", "n_clicks"),
        State("teaching-example-table", "selected_rows"),
        State("teaching-example-table", "data"),
        State("teaching-example-label-action", "value"),
        State("teaching-example-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def change_teaching_example_label(
        clicks: int, selected_rows: list[int] | None, rows: list[dict[str, Any]] | None,
        label: str | None, refresh: int | None,
    ) -> tuple[str, int]:
        del clicks
        if not selected_rows or not rows:
            return "Select an example first.", int(refresh or 0)
        index = int(selected_rows[0])
        if index < 0 or index >= len(rows):
            return "Select an example first.", int(refresh or 0)
        observation_id = int(rows[index]["observation_id"])
        selected_label = str(label or "positive")
        project.set_teaching_example_label(observation_id, selected_label)
        display = "Present" if selected_label == "positive" else "Absent"
        return f"Changed the selected teaching example to {display}.", int(refresh or 0) + 1

    @app.callback(
        Output("code-center-units-table", "data"),
        Input("code-center-selected-store", "data"),
        Input("workspace-tabs", "value"),
        Input("explore-annotation-refresh-store", "data"),
        Input("focus-refresh-store", "data"),
        Input("teaching-example-refresh-store", "data"),
    )
    def load_code_center_units(
        code_id: int | None,
        workspace: str | None,
        explore_refresh: int | None,
        focus_refresh: int | None,
        teaching_refresh: int | None,
    ) -> Any:
        del explore_refresh, focus_refresh, teaching_refresh
        if workspace != "codes":
            return no_update
        if code_id is None:
            return []
        return [
            {
                "observation_id": int(observation["observation_id"]),
                "kind": str(observation["kind"]).replace("_", " ").title(),
                "key": _observation_display_key(project, observation),
                "text": str(observation["text"]),
            }
            for observation in project.positive_observations_for_code(int(code_id))
        ]

    @app.callback(
        Output("code-center-view", "options"), Output("code-center-view", "value"),
        Input("code-center-geometry", "value"),
    )
    def update_code_center_views(geometry_id: int | None) -> tuple[list[dict[str, Any]], int | None]:
        if geometry_id is None:
            return [], None
        records = project.views(int(geometry_id))
        options = [{"label": row["name"], "value": int(row["view_id"])} for row in records]
        return options, (int(records[0]["view_id"]) if records else None)

    @app.callback(
        Output("code-center-map", "figure"),
        Output("code-center-page-status", "children"),
        Input("code-center-selected-store", "data"),
        Input("code-center-geometry", "value"),
        Input("code-center-view", "value"),
        Input("code-center-page-size", "value"),
        Input("code-center-page-number", "value"),
        Input("code-center-include-positives", "value"),
        Input("workspace-tabs", "value"),
        Input("explore-annotation-refresh-store", "data"),
        Input("focus-refresh-store", "data"),
        Input("teaching-example-refresh-store", "data"),
        State("code-center-focal-store", "data"),
        State("code-center-map", "relayoutData"),
    )
    def update_code_center_map(
        code_id: int | None,
        geometry_id: int | None,
        view_id: int | None,
        page_size: Any,
        page_number: Any,
        include_positive_values: list[str] | None,
        workspace: str | None,
        explore_refresh: int | None,
        focus_refresh: int | None,
        teaching_refresh: int | None,
        focal_unit_id: int | None,
        relayout_data: dict[str, Any] | None,
    ) -> tuple[Any, str]:
        del explore_refresh, focus_refresh, teaching_refresh
        if workspace != "codes":
            return no_update, no_update
        if geometry_id is None or view_id is None:
            return {}, ""
        positive_observations = (
            project.positive_observations_for_code(int(code_id))
            if code_id is not None else []
        )
        positive = {
            int(observation["unit_id"])
            for observation in positive_observations
            if observation["kind"] == "atomic" and observation["unit_id"] is not None
        }
        positive_positions = {
            unit_position_by_id[unit_id]
            for unit_id in positive
            if unit_id in unit_position_by_id
        }
        size = _coerce_positive_int(page_size, default=5000)
        number = _coerce_positive_int(page_number, default=1)
        include_all_positives = "include" in set(include_positive_values or [])
        page = deterministic_page_with_inclusions(
            len(units),
            page_size=size,
            page_index=number - 1,
            seed=0,
            include_positions=(positive_positions if include_all_positives else ()),
        )
        positions = page.positions
        page_units = [units[int(position)] for position in positions]
        page_eligible = [int(unit["unit_id"]) in positive for unit in page_units]
        geometry_record = geometry_by_id[int(geometry_id)]
        view_record = project.database.get_view(int(view_id))
        figure = build_geometry_figure(
            coordinates=project.view_coordinates(int(view_id))[positions],
            units=page_units,
            eligible=page_eligible,
            focal_unit_id=None,
            geometry_name=str(geometry_record["name"]),
            view_name=str(view_record["name"]),
            page_label=f"Page {page.page_index + 1} of {page.page_count}",
            uirevision=(
                f"code-center:{geometry_id}:{view_id}:{code_id}:"
                f"{page.page_index}:{int(include_all_positives)}"
            ),
        )
        for trace in figure.data:
            if trace.name == "Eligible":
                trace.name = "Positive atomic observations"
                trace.showlegend = True
            elif trace.name == "Filtered":
                trace.name = "Other atomic observations"
                trace.showlegend = True
        derived_positive = [
            observation for observation in positive_observations
            if observation["kind"] != "atomic"
        ]
        omitted_derived = 0
        for kind, symbol, label in (
            ("span", "triangle-up", "Positive spans"),
            ("teaching_example", "star", "Positive teaching examples"),
        ):
            observations_of_kind = [
                observation for observation in derived_positive
                if observation["kind"] == kind
            ]
            coordinates: list[list[float]] = []
            observation_ids: list[int] = []
            hover: list[str] = []
            for observation in observations_of_kind:
                try:
                    point = project.observation_view_coordinates(
                        int(observation["observation_id"]), int(view_id)
                    )[0]
                except (KeyError, ValueError):
                    omitted_derived += 1
                    continue
                coordinates.append([float(point[0]), float(point[1])])
                observation_ids.append(int(observation["observation_id"]))
                hover.append(_observation_display_key(project, observation))
            if coordinates:
                array = np.asarray(coordinates, dtype=float)
                marker_line = ["#111111" for _value in observation_ids]
                marker_width = [0.8 for _value in observation_ids]
                figure.add_trace(
                    go.Scattergl(
                        x=array[:, 0].tolist(), y=array[:, 1].tolist(),
                        mode="markers", name=label, customdata=observation_ids,
                        hovertext=hover, hovertemplate="%{hovertext}<extra></extra>",
                        marker={
                            "size": 10,
                            "symbol": symbol,
                            "color": "#111111",
                            "opacity": 0.96,
                            "line": {"color": marker_line, "width": marker_width},
                        },
                    )
                )

        focal_x: list[float] = []
        focal_y: list[float] = []
        focal_customdata: list[int] = []
        focal_hover: list[str] = []
        if focal_unit_id is not None:
            try:
                focal_observation = project.observation(int(focal_unit_id))
                focal_point = project.observation_view_coordinates(
                    int(focal_unit_id), int(view_id)
                )[0]
            except (KeyError, ValueError):
                pass
            else:
                focal_x = [float(focal_point[0])]
                focal_y = [float(focal_point[1])]
                focal_customdata = [int(focal_unit_id)]
                focal_hover = [_observation_display_key(project, focal_observation)]
        figure.add_trace(
            go.Scattergl(
                x=focal_x,
                y=focal_y,
                mode="markers",
                name="Focal observation",
                customdata=focal_customdata,
                hovertext=focal_hover,
                hovertemplate="%{hovertext}<extra></extra>",
                marker={
                    "size": 13,
                    "color": "rgba(0, 0, 0, 0)",
                    "line": {"color": "#178547", "width": 2.6},
                },
                showlegend=False,
            )
        )

        figure.update_layout(
            showlegend=True,
            legend={"orientation": "h", "y": -0.08},
        )
        # Keep the lightweight focal ring at a stable trace index so selecting
        # a point can patch only this trace instead of rebuilding the map.
        figure.data = (figure.data[-1], *figure.data[:-1])

        if callback_context.triggered_id not in {
            "code-center-selected-store",
            "code-center-geometry",
            "code-center-view",
            "code-center-page-size",
            "code-center-page-number",
            "code-center-include-positives",
        }:
            _apply_relayout_ranges(figure, relayout_data)
        extras = max(0, len(positions) - min(size, len(units) - page.page_index * size))
        status = (
            f"Page {page.page_index + 1} of {page.page_count} · "
            f"{len(positions):,} points shown"
        )
        if include_all_positives:
            status += f" · all {len(positive):,} positive units included"
            if extras:
                status += f" ({extras:,} added beyond the page sample)"
        if derived_positive:
            shown_derived = len(derived_positive) - omitted_derived
            status += f" · {shown_derived:,} derived positive observations shown"
            if omitted_derived:
                status += f" · {omitted_derived:,} unavailable in this view"
        return figure, status

    @app.callback(
        Output("code-center-map", "figure", allow_duplicate=True),
        Input("code-center-focal-store", "data"),
        State("code-center-view", "value"),
        State("code-center-map", "figure"),
        prevent_initial_call=True,
    )
    def update_code_center_focal_marker(
        observation_id: int | None,
        view_id: int | None,
        current_figure: dict[str, Any] | None,
    ) -> Any:
        if view_id is None:
            return no_update
        # Dash Patch updates require the target property and nested path to
        # already exist in the browser. code-center-map is initialized with a
        # stable focal-marker trace at data[0], and this guard also protects
        # transient/dynamic-layout states before that structure is available.
        if not current_figure or not current_figure.get("data"):
            return no_update
        patch = Patch()
        if observation_id is None:
            patch["data"][0]["x"] = []
            patch["data"][0]["y"] = []
            patch["data"][0]["customdata"] = []
            patch["data"][0]["hovertext"] = []
            return patch
        try:
            observation = project.observation(int(observation_id))
            point = project.observation_view_coordinates(
                int(observation_id), int(view_id)
            )[0]
        except (KeyError, ValueError):
            return no_update
        patch["data"][0]["x"] = [float(point[0])]
        patch["data"][0]["y"] = [float(point[1])]
        patch["data"][0]["customdata"] = [int(observation_id)]
        patch["data"][0]["hovertext"] = [
            _observation_display_key(project, observation)
        ]
        return patch

    @app.callback(
        Output("code-center-page-number", "max"),
        Output("code-center-page-number", "value", allow_duplicate=True),
        Input("code-center-page-previous", "n_clicks"),
        Input("code-center-page-next", "n_clicks"),
        Input("code-center-page-size", "value"),
        State("code-center-page-number", "value"),
        prevent_initial_call=True,
    )
    def change_code_center_page(
        previous_clicks: int,
        next_clicks: int,
        page_size: Any,
        page_number: Any,
    ) -> tuple[int, int]:
        del previous_clicks, next_clicks
        size = _coerce_positive_int(page_size, default=5000)
        page_count = deterministic_page(len(units), page_size=size).page_count
        current = min(max(1, _coerce_positive_int(page_number, default=1)), page_count)
        triggered = callback_context.triggered_id
        if triggered == "code-center-page-size":
            current = 1
        elif triggered == "code-center-page-previous":
            current = max(1, current - 1)
        elif triggered == "code-center-page-next":
            current = min(page_count, current + 1)
        return page_count, current

    @app.callback(
        Output("code-center-focal-store", "data"),
        Input("code-center-map", "clickData"),
        Input("code-center-units-table", "active_cell"),
        State("code-center-units-table", "data"),
        prevent_initial_call=True,
    )
    def select_code_center_unit(
        click_data: dict[str, Any] | None,
        active_cell: dict[str, Any] | None,
        rows: list[dict[str, Any]] | None,
    ) -> Any:
        if callback_context.triggered_id == "code-center-map":
            clicked = _clicked_unit_id(click_data)
            if clicked is None:
                return no_update
            return observation_id_by_unit_id.get(int(clicked), int(clicked))
        if active_cell and rows:
            row_index = int(active_cell.get("row", -1))
            if 0 <= row_index < len(rows):
                row = rows[row_index]
                return int(row.get("observation_id", row.get("unit_id")))
        return no_update

    @app.callback(Output("code-center-unit-preview", "children"), Input("code-center-focal-store", "data"))
    def preview_code_center_unit(observation_id: int | None) -> Any:
        if observation_id is None:
            return html.P(
                "Click a positive observation or a point to inspect it.",
                className="memo-preview-empty",
            )
        observation = project.observation(int(observation_id))
        return [
            html.Strong(_observation_display_key(project, observation)),
            html.Small(str(observation["kind"]).replace("_", " ").title()),
            html.P(str(observation["text"])),
        ]

    @app.callback(
        Output("code-center-assignment-positive", "disabled"),
        Output("code-center-assignment-negative", "disabled"),
        Output("code-center-assignment-unsure", "disabled"),
        Output("code-center-assignment-positive", "className"),
        Output("code-center-assignment-negative", "className"),
        Output("code-center-assignment-unsure", "className"),
        Input("code-center-selected-store", "data"),
        Input("code-center-focal-store", "data"),
        Input("workspace-tabs", "value"),
        Input("explore-annotation-refresh-store", "data"),
        Input("focus-refresh-store", "data"),
    )
    def style_code_center_assignment_buttons(
        code_id: int | None,
        observation_id: int | None,
        workspace: str | None,
        explore_refresh: int | None,
        focus_refresh: int | None,
    ) -> tuple[Any, Any, Any, Any, Any, Any]:
        del explore_refresh, focus_refresh
        if workspace != "codes":
            return (no_update,) * 6
        current = None
        disabled = code_id is None or observation_id is None
        if not disabled:
            observation = project.observation(int(observation_id))
            # Teaching examples have their own label/status manager. Revising
            # them here would desynchronize the teaching-example metadata.
            disabled = str(observation["kind"]) == "teaching_example"
            if not disabled:
                current = _current_annotation_value(
                    project, int(observation_id), int(code_id)
                )
        classes = _judgment_button_classes(
            current, prefix="code-center-assignment"
        )
        return disabled, disabled, disabled, *classes

    @app.callback(
        Output("code-center-assignment-status", "children"),
        Output("explore-annotation-refresh-store", "data", allow_duplicate=True),
        Output("focus-refresh-store", "data", allow_duplicate=True),
        Input("code-center-assignment-positive", "n_clicks"),
        Input("code-center-assignment-negative", "n_clicks"),
        Input("code-center-assignment-unsure", "n_clicks"),
        State("code-center-selected-store", "data"),
        State("code-center-focal-store", "data"),
        State("explore-annotation-refresh-store", "data"),
        State("focus-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def revise_code_center_assignment(
        positive_clicks: int,
        negative_clicks: int,
        unsure_clicks: int,
        code_id: int | None,
        observation_id: int | None,
        explore_refresh: int | None,
        focus_refresh: int | None,
    ) -> tuple[str, int, int]:
        del positive_clicks, negative_clicks, unsure_clicks
        if code_id is None or observation_id is None:
            return (
                "Select a code and observation first.",
                int(explore_refresh or 0),
                int(focus_refresh or 0),
            )
        observation = project.observation(int(observation_id))
        if str(observation["kind"]) == "teaching_example":
            return (
                "Revise teaching examples through the teaching-example manager.",
                int(explore_refresh or 0),
                int(focus_refresh or 0),
            )
        value_by_button = {
            "code-center-assignment-positive": "positive",
            "code-center-assignment-negative": "negative",
            "code-center-assignment-unsure": "unsure",
        }
        value = value_by_button[str(callback_context.triggered_id)]
        current = _current_annotation_value(
            project, int(observation_id), int(code_id)
        )
        if current == value:
            return "", int(explore_refresh or 0), int(focus_refresh or 0)
        project.annotate(
            int(observation_id),
            int(code_id),
            value,
            origin="human_code_center",
        )
        return (
            f"Revised the assignment to {value}.",
            int(explore_refresh or 0) + 1,
            int(focus_refresh or 0) + 1,
        )

    @app.callback(
        Output("focus-code-dropdown", "options"),
        Input("explore-code-refresh-store", "data"),
        Input("focus-refresh-store", "data"),
    )
    def refresh_code_options(
        explore_refresh: int | None, focus_refresh: int | None
    ) -> list[dict[str, Any]]:
        del explore_refresh, focus_refresh
        return _code_options(project.codes())

    @app.callback(
        Output("code-modal", "style"),
        Output("code-modal-target-store", "data"),
        Output("explore-code-refresh-store", "data", allow_duplicate=True),
        Output("focus-refresh-store", "data", allow_duplicate=True),
        Output("focus-code-dropdown", "options", allow_duplicate=True),
        Output("focus-code-dropdown", "value", allow_duplicate=True),
        Output("explore-code-palette-store", "data", allow_duplicate=True),
        Output("explore-code-row-count-store", "data", allow_duplicate=True),
        Output("code-modal-name", "value"),
        Output("code-modal-definition", "value"),
        Output("code-modal-status", "children"),
        Input("explore-new-code-button", "n_clicks"),
        Input("focus-new-code-button", "n_clicks"),
        Input("code-modal-cancel-button", "n_clicks"),
        Input("code-modal-cancel-footer", "n_clicks"),
        Input("code-modal-create-button", "n_clicks"),
        State("code-modal-target-store", "data"),
        State("code-modal-name", "value"),
        State("code-modal-definition", "value"),
        State("explore-code-refresh-store", "data"),
        State("focus-refresh-store", "data"),
        State("explore-code-palette-store", "data"),
        prevent_initial_call=True,
    )
    def manage_code_modal(
        explore_open: int,
        focus_open: int,
        cancel_header: int,
        cancel_footer: int,
        create_clicks: int,
        target: str | None,
        name: str | None,
        definition: str | None,
        explore_refresh: int | None,
        focus_refresh: int | None,
        explore_palette: list[int | None] | None,
    ) -> tuple[Any, ...]:
        del explore_open, focus_open, cancel_header, cancel_footer, create_clicks
        triggered = callback_context.triggered_id
        if triggered == "explore-new-code-button":
            return (
                {"display": "flex"}, "explore", no_update, no_update,
                no_update, no_update, no_update, no_update, "", "", "",
            )
        if triggered == "focus-new-code-button":
            return (
                {"display": "flex"}, "focus", no_update, no_update,
                no_update, no_update, no_update, no_update, "", "", "",
            )
        if triggered in {"code-modal-cancel-button", "code-modal-cancel-footer"}:
            return (
                {"display": "none"}, None, no_update, no_update,
                no_update, no_update, no_update, no_update, "", "", "",
            )

        clean_name = (name or "").strip()
        if not clean_name:
            return (
                {"display": "flex"}, target, no_update, no_update,
                no_update, no_update, no_update, no_update,
                name or "", definition or "", "Enter a code label.",
            )
        existing = {
            str(code["name"]).casefold(): int(code["code_id"])
            for code in project.codes()
        }
        if clean_name.casefold() in existing:
            code_id = existing[clean_name.casefold()]
        else:
            code_id = project.create_code(clean_name, (definition or "").strip())
        current_codes = project.codes()
        options = _code_options(current_codes)
        focus_value = code_id if target == "focus" else no_update
        palette_value: Any = no_update
        row_count_value: Any = no_update
        if target == "explore":
            palette_value = _place_code_in_explore_palette(
                explore_palette,
                code_id,
                {int(code["code_id"]) for code in current_codes},
            )
            row_count_value = len(palette_value)
        return (
            {"display": "none"},
            None,
            int(explore_refresh or 0) + 1,
            int(focus_refresh or 0) + 1,
            options,
            focus_value,
            palette_value,
            row_count_value,
            "",
            "",
            "",
        )

    @app.callback(
        Output("explore-code-palette-store", "data", allow_duplicate=True),
        Output("explore-code-row-count-store", "data", allow_duplicate=True),
        Input("explore-add-code-row", "n_clicks"),
        Input({"type": "explore-code-remove", "index": ALL}, "n_clicks"),
        State("explore-code-palette-store", "data"),
        prevent_initial_call=True,
    )
    def change_explore_code_palette_rows(
        add_clicks: int | None,
        remove_clicks: list[int] | None,
        palette: list[int | None] | None,
    ) -> tuple[Any, Any]:
        del remove_clicks
        valid_ids = {int(code["code_id"]) for code in project.codes()}
        current = _normalize_explore_code_palette(palette, valid_ids)
        triggered = callback_context.triggered_id
        triggered_value = (
            callback_context.triggered[0].get("value")
            if callback_context.triggered
            else None
        )
        if triggered == "explore-add-code-row":
            if int(add_clicks or 0) < 1:
                return no_update, no_update
            updated = [*current, None]
            return updated, len(updated)
        if isinstance(triggered, dict) and triggered.get("type") == "explore-code-remove":
            # Pattern-matched buttons fire when Dash mounts them. A newly mounted
            # remove button has n_clicks=0 and is not a user request to remove a row.
            if int(triggered_value or 0) < 1:
                return no_update, no_update
            index = int(triggered.get("index", -1))
            if index <= 0 or index >= len(current):
                return no_update, no_update
            updated = [value for position, value in enumerate(current) if position != index]
            return updated, len(updated)
        return no_update, no_update

    @app.callback(
        Output("explore-code-palette-store", "data", allow_duplicate=True),
        Input({"type": "explore-code-dropdown", "index": ALL}, "value"),
        State({"type": "explore-code-dropdown", "index": ALL}, "id"),
        State("explore-code-palette-store", "data"),
        prevent_initial_call=True,
    )
    def select_explore_palette_code(
        values: list[int | None],
        ids: list[dict[str, Any]],
        palette: list[int | None] | None,
    ) -> Any:
        del values, ids
        triggered = callback_context.triggered_id
        if not isinstance(triggered, dict):
            return no_update
        row_index = int(triggered.get("index", -1))
        triggered_value = (
            callback_context.triggered[0].get("value")
            if callback_context.triggered
            else None
        )
        valid_ids = {int(code["code_id"]) for code in project.codes()}
        current = _normalize_explore_code_palette(palette, valid_ids)
        if not 0 <= row_index < len(current):
            return no_update
        new_value = int(triggered_value) if triggered_value is not None else None
        if new_value is not None:
            selected_elsewhere = {
                int(value)
                for index, value in enumerate(current)
                if index != row_index and value is not None
            }
            if new_value in selected_elsewhere:
                # Keep one code per palette row without rerendering the whole
                # palette. Reset only the dropdown the user just changed.
                set_props(triggered, {"value": current[row_index]})
                return no_update
        if current[row_index] == new_value:
            return no_update
        updated = list(current)
        updated[row_index] = new_value
        return updated

    @app.callback(
        Output("explore-code-palette", "children"),
        Input("explore-code-row-count-store", "data"),
        Input("explore-code-refresh-store", "data"),
        State("explore-code-palette-store", "data"),
    )
    def render_explore_code_palette(
        row_count: int | None,
        code_refresh: int | None,
        palette: list[int | None] | None,
    ) -> list[Any]:
        del code_refresh
        current_codes = project.codes()
        valid_ids = {int(code["code_id"]) for code in current_codes}
        normalized = _explore_palette_for_row_count(palette, valid_ids, row_count)
        selected_codes = {int(value) for value in normalized if value is not None}
        rows: list[Any] = []
        for index, code_id in enumerate(normalized):
            options = [
                {"label": code["name"], "value": int(code["code_id"])}
                for code in current_codes
                if int(code["code_id"]) == code_id
                or int(code["code_id"]) not in selected_codes
            ]
            selector_children: list[Any] = [
                dcc.Dropdown(
                    id={"type": "explore-code-dropdown", "index": index},
                    options=options,
                    value=code_id,
                    placeholder="Select a code",
                    clearable=True,
                )
            ]
            if index > 0:
                selector_children.append(
                    html.Button(
                        "×",
                        id={"type": "explore-code-remove", "index": index},
                        n_clicks=0,
                        className="explore-code-remove",
                        title="Remove this coding row",
                        **{"aria-label": "Remove this coding row"},
                    )
                )
            rows.append(
                html.Div(
                    [
                        html.Div(selector_children, className="explore-code-selector-row"),
                        html.Div(
                            [
                                html.Button(
                                    "Present",
                                    id={"type": "explore-code-positive", "index": index},
                                    n_clicks=0,
                                    className="judgment-button positive-button",
                                    disabled=True,
                                ),
                                html.Button(
                                    "Absent",
                                    id={"type": "explore-code-negative", "index": index},
                                    n_clicks=0,
                                    className="judgment-button negative-button",
                                    disabled=True,
                                ),
                                html.Button(
                                    "Unsure",
                                    id={"type": "explore-code-unsure", "index": index},
                                    n_clicks=0,
                                    className="judgment-button unsure-button",
                                    disabled=True,
                                ),
                            ],
                            className="explore-code-buttons",
                        ),
                    ],
                    className="explore-code-entry",
                )
            )
        return rows

    @app.callback(
        Output({"type": "explore-code-dropdown", "index": ALL}, "options"),
        Input("explore-code-palette-store", "data"),
        Input("explore-code-refresh-store", "data"),
        Input({"type": "explore-code-dropdown", "index": ALL}, "id"),
    )
    def sync_explore_palette_dropdown_options(
        palette: list[int | None] | None,
        code_refresh: int | None,
        ids: list[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        del code_refresh
        current_codes = project.codes()
        valid_ids = {int(code["code_id"]) for code in current_codes}
        normalized = _normalize_explore_code_palette(palette, valid_ids)
        selected = {int(value) for value in normalized if value is not None}
        options_by_row: list[list[dict[str, Any]]] = []
        for component_id in ids or []:
            index = int(component_id.get("index", -1))
            value = normalized[index] if 0 <= index < len(normalized) else None
            options_by_row.append(
                [
                    {"label": code["name"], "value": int(code["code_id"])}
                    for code in current_codes
                    if int(code["code_id"]) == value
                    or int(code["code_id"]) not in selected
                ]
            )
        return options_by_row

    @app.callback(
        Output({"type": "explore-code-positive", "index": ALL}, "disabled"),
        Output({"type": "explore-code-negative", "index": ALL}, "disabled"),
        Output({"type": "explore-code-unsure", "index": ALL}, "disabled"),
        Output({"type": "explore-code-positive", "index": ALL}, "className"),
        Output({"type": "explore-code-negative", "index": ALL}, "className"),
        Output({"type": "explore-code-unsure", "index": ALL}, "className"),
        Input("explore-span-selection-store", "data"),
        Input("explore-annotation-refresh-store", "data"),
        Input("explore-code-palette-store", "data"),
        Input({"type": "explore-code-positive", "index": ALL}, "id"),
        State("explore-focal-unit-store", "data"),
    )
    def style_explore_palette_buttons(
        selected_unit_ids: list[int] | None,
        annotation_refresh: int | None,
        palette: list[int | None] | None,
        ids: list[dict[str, Any]],
        focal_unit_id: int | None,
    ) -> tuple[list[bool], list[bool], list[bool], list[str], list[str], list[str]]:
        del annotation_refresh
        valid_ids = {int(code["code_id"]) for code in project.codes()}
        normalized = _normalize_explore_code_palette(palette, valid_ids)
        values_by_code: dict[int, str] = {}
        if focal_unit_id is not None:
            observation_id = _selection_observation_id(
                project, int(focal_unit_id), selected_unit_ids
            )
            if observation_id is not None:
                values_by_code = {
                    int(row["code_id"]): str(row["value"])
                    for row in project.current_annotations_for_observation(observation_id)
                }
        disabled_values: list[bool] = []
        positive_classes: list[str] = []
        negative_classes: list[str] = []
        unsure_classes: list[str] = []
        for component_id in ids or []:
            index = int(component_id.get("index", -1))
            code_id = normalized[index] if 0 <= index < len(normalized) else None
            disabled = focal_unit_id is None or code_id is None
            disabled_values.append(disabled)
            current = values_by_code.get(int(code_id)) if code_id is not None else None
            positive, negative, unsure = _judgment_button_classes(
                current, prefix="explore-code"
            )
            positive_classes.append(positive)
            negative_classes.append(negative)
            unsure_classes.append(unsure)
        return (
            disabled_values,
            list(disabled_values),
            list(disabled_values),
            positive_classes,
            negative_classes,
            unsure_classes,
        )

    @app.callback(
        Output("explore-annotation-refresh-store", "data", allow_duplicate=True),
        Output("explore-code-status", "children", allow_duplicate=True),
        Input({"type": "explore-code-positive", "index": ALL}, "n_clicks"),
        Input({"type": "explore-code-negative", "index": ALL}, "n_clicks"),
        Input({"type": "explore-code-unsure", "index": ALL}, "n_clicks"),
        State("explore-focal-unit-store", "data"),
        State("explore-code-palette-store", "data"),
        State("explore-span-selection-store", "data"),
        State("explore-annotation-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def annotate_from_explore_palette(
        positive_clicks: list[int] | None,
        negative_clicks: list[int] | None,
        unsure_clicks: list[int] | None,
        focal_unit_id: int | None,
        palette: list[int | None] | None,
        selected_unit_ids: list[int] | None,
        refresh: int | None,
    ) -> tuple[Any, Any]:
        del positive_clicks, negative_clicks, unsure_clicks
        triggered = callback_context.triggered_id
        triggered_value = (
            callback_context.triggered[0].get("value")
            if callback_context.triggered
            else None
        )
        # Pattern-matched buttons can fire when a dynamic row is mounted. Only a
        # positive click count is an actual coding judgment.
        if not isinstance(triggered, dict) or int(triggered_value or 0) < 1:
            return no_update, no_update
        if focal_unit_id is None:
            return no_update, "Select a focal unit first."
        valid_ids = {int(code["code_id"]) for code in project.codes()}
        normalized = _normalize_explore_code_palette(palette, valid_ids)
        row_index = int(triggered.get("index", -1))
        code_id = normalized[row_index] if 0 <= row_index < len(normalized) else None
        if code_id is None:
            return no_update, "Select or create a code first."
        values = {
            "explore-code-positive": "positive",
            "explore-code-negative": "negative",
            "explore-code-unsure": "unsure",
        }
        value = values.get(str(triggered.get("type")))
        if value is None:
            return no_update, no_update
        selected = [int(item) for item in (selected_unit_ids or [focal_unit_id])]
        observation_id = _selection_observation_id(project, int(focal_unit_id), selected)
        current = _current_annotation_value(project, observation_id, int(code_id))
        if current == value:
            return no_update, no_update
        project.annotate_span(selected, int(code_id), value, origin="human_direct")
        return int(refresh or 0) + 1, ""

    @app.callback(
        Output("focus-active-area", "className"),
        Input("focus-code-dropdown", "value"),
    )
    def set_focus_active_state(code_id: int | None) -> str:
        return "focus-active-area" if code_id is not None else "focus-active-area focus-disabled"


    @app.callback(
        Output("classifier-spec-modal", "style"),
        Output("classifier-spec-status", "children"),
        Input("focus-new-classifier-button", "n_clicks"),
        Input("classifier-spec-close", "n_clicks"),
        Input("classifier-spec-save", "n_clicks"),
        State("classifier-spec-name", "value"),
        State("classifier-spec-geometry", "value"),
        State("classifier-spec-algorithm", "value"),
        State("focus-code-dropdown", "value"),
        prevent_initial_call=True,
    )
    def manage_classifier_spec_modal(
        open_clicks: int,
        close_clicks: int,
        save_clicks: int,
        name: str | None,
        geometry_id: int | None,
        algorithm: str | None,
        code_id: int | None,
    ) -> tuple[dict[str, str], str]:
        del open_clicks, close_clicks, save_clicks
        triggered = str(callback_context.triggered_id)
        if triggered == "focus-new-classifier-button":
            return {"display": "flex"}, ""
        if triggered == "classifier-spec-close":
            return {"display": "none"}, ""
        if code_id is None:
            return {"display": "flex"}, "Select a code first."
        if geometry_id is None:
            return {"display": "flex"}, "Select a parent geometry."
        try:
            selected_algorithm = str(algorithm or "logistic_l2")
            clean_name = (name or "").strip()
            if not clean_name:
                geometry = geometry_by_id[int(geometry_id)]
                code = project.code(int(code_id))
                clean_name = (
                    f"{code['name']} · {geometry['name']} · "
                    f"{classifier_algorithm_label(selected_algorithm)}"
                )
            project.create_classifier_spec(
                code_id=int(code_id),
                name=clean_name,
                geometry_id=int(geometry_id),
                algorithm=selected_algorithm,
                hyperparameters=None,
            )
        except (KeyError, ValueError) as error:
            return {"display": "flex"}, str(error)
        return {"display": "none"}, ""

    @app.callback(
        Output("classifier-spec-refresh-store", "data"),
        Input("classifier-spec-save", "n_clicks"),
        State("classifier-spec-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def refresh_after_classifier_spec_save(
        clicks: int, refresh: int | None
    ) -> int:
        del clicks
        return int(refresh or 0) + 1

    @app.callback(
        Output("classifier-manager-modal", "style"),
        Output("classifier-manager-table", "data"),
        Output("classifier-manager-status", "children"),
        Output("classifier-spec-refresh-store", "data", allow_duplicate=True),
        Output("committee-refresh-store", "data", allow_duplicate=True),
        Input("focus-manage-classifiers", "n_clicks"),
        Input("classifier-manager-close", "n_clicks"),
        Input("classifier-manager-rename", "n_clicks"),
        Input("classifier-manager-delete", "n_clicks"),
        State("classifier-manager-table", "selected_rows"),
        State("classifier-manager-table", "data"),
        State("classifier-manager-rename-name", "value"),
        State("focus-code-dropdown", "value"),
        State("classifier-spec-refresh-store", "data"),
        State("committee-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def manage_classifier_manager_modal(
        open_clicks: int,
        close_clicks: int,
        rename_clicks: int,
        delete_clicks: int,
        selected_rows: list[int] | None,
        table_data: list[dict[str, Any]] | None,
        rename_name: str | None,
        code_id: int | None,
        spec_refresh: int | None,
        committee_refresh: int | None,
    ) -> tuple[dict[str, str], list[dict[str, Any]], str, int, int]:
        del open_clicks, close_clicks, rename_clicks, delete_clicks
        triggered = str(callback_context.triggered_id)
        next_spec_refresh = int(spec_refresh or 0)
        next_committee_refresh = int(committee_refresh or 0)
        if triggered == "classifier-manager-close":
            return {"display": "none"}, [], "", next_spec_refresh, next_committee_refresh
        message = ""
        if triggered == "classifier-manager-rename":
            rows = table_data or []
            selection = selected_rows or []
            if not selection:
                message = "Select a classifier to rename."
            else:
                try:
                    row = rows[int(selection[0])]
                    project.rename_classifier_spec(
                        int(row["classifier_spec_id"]), name=str(rename_name or "")
                    )
                except (IndexError, KeyError, ValueError) as error:
                    message = str(error)
                else:
                    next_spec_refresh += 1
                    next_committee_refresh += 1
                    message = f"Classifier renamed to {str(rename_name).strip()!r}."
        if triggered == "classifier-manager-delete":
            rows = table_data or []
            selection = selected_rows or []
            if not selection:
                message = "Select a classifier to delete."
            else:
                try:
                    project.delete_classifier_spec(
                        int(rows[int(selection[0])]["classifier_spec_id"])
                    )
                except (KeyError, ValueError) as error:
                    message = str(error)
                else:
                    next_spec_refresh += 1
                    next_committee_refresh += 1
                    message = "Classifier deleted from active use."
        rows = [
            {
                "classifier_spec_id": int(row["classifier_spec_id"]),
                "name": str(row["name"]),
                "geometry": str(row["geometry_name"]),
                "algorithm": classifier_algorithm_label(str(row["algorithm"])),
                "settings": hyperparameter_summary(
                    str(row["algorithm"]), dict(row["hyperparameters"])
                ),
            }
            for row in (
                project.classifier_specs(code_id=int(code_id))
                if code_id is not None else []
            )
        ]
        return {"display": "flex"}, rows, message, next_spec_refresh, next_committee_refresh

    @app.callback(
        Output("classifier-manager-rename-name", "value"),
        Input("classifier-manager-table", "selected_rows"),
        Input("classifier-manager-table", "data"),
        prevent_initial_call=True,
    )
    def populate_classifier_rename_name(
        selected_rows: list[int] | None,
        table_data: list[dict[str, Any]] | None,
    ) -> str:
        rows = table_data or []
        selection = selected_rows or []
        if not selection:
            return ""
        try:
            return str(rows[int(selection[0])]["name"])
        except (IndexError, KeyError, TypeError, ValueError):
            return ""

    @app.callback(
        Output("committee-modal", "style"),
        Output("committee-members", "options"),
        Output("committee-table", "data"),
        Output("committee-status", "children"),
        Output("committee-refresh-store", "data", allow_duplicate=True),
        Input("focus-manage-committees", "n_clicks"),
        Input("committee-close", "n_clicks"),
        Input("committee-save", "n_clicks"),
        Input("committee-rename", "n_clicks"),
        Input("committee-delete", "n_clicks"),
        State("focus-code-dropdown", "value"),
        State("committee-name", "value"),
        State("committee-members", "value"),
        State("committee-aggregation", "value"),
        State("committee-table", "selected_rows"),
        State("committee-table", "data"),
        State("committee-manager-rename-name", "value"),
        State("committee-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def manage_committee_modal(
        open_clicks: int,
        close_clicks: int,
        save_clicks: int,
        rename_clicks: int,
        delete_clicks: int,
        code_id: int | None,
        name: str | None,
        member_ids: list[int] | None,
        aggregation: str | None,
        selected_rows: list[int] | None,
        table_data: list[dict[str, Any]] | None,
        rename_name: str | None,
        refresh: int | None,
    ) -> tuple[dict[str, str], list[dict[str, Any]], list[dict[str, Any]], str, int]:
        del open_clicks, close_clicks, save_clicks, rename_clicks, delete_clicks
        triggered = str(callback_context.triggered_id)
        if triggered == "committee-close":
            return {"display": "none"}, [], [], "", int(refresh or 0)
        if code_id is None:
            return {"display": "none"}, [], [], "Select a code first.", int(refresh or 0)
        message = ""
        next_refresh = int(refresh or 0)
        try:
            if triggered == "committee-save":
                clean_name = (name or "").strip()
                if not clean_name:
                    raise ValueError("Enter a committee name.")
                members = [int(value) for value in (member_ids or [])]
                if not members:
                    raise ValueError("Select at least one classifier.")
                project.create_classifier_committee(
                    code_id=int(code_id),
                    name=clean_name,
                    classifier_spec_ids=members,
                    aggregation=str(aggregation or "mean"),
                )
                next_refresh += 1
                message = "Committee saved."
            elif triggered == "committee-rename":
                rows = table_data or []
                selection = selected_rows or []
                if not selection:
                    raise ValueError("Select a committee to rename.")
                project.rename_classifier_committee(
                    int(rows[int(selection[0])]["committee_id"]),
                    name=str(rename_name or ""),
                )
                next_refresh += 1
                message = f"Committee renamed to {str(rename_name or '').strip()!r}."
            elif triggered == "committee-delete":
                rows = table_data or []
                selection = selected_rows or []
                if not selection:
                    raise ValueError("Select a committee to delete.")
                project.delete_classifier_committee(
                    int(rows[int(selection[0])]["committee_id"])
                )
                next_refresh += 1
                message = "Committee deleted."
        except (KeyError, ValueError) as error:
            message = str(error)
        specs = project.classifier_specs(code_id=int(code_id))
        member_options = [
            {
                "label": f"{row['name']} · {row['geometry_name']}",
                "value": int(row["classifier_spec_id"]),
            }
            for row in specs
        ]
        committees = project.classifier_committees(int(code_id))
        rows = [
            {
                "committee_id": int(row["committee_id"]),
                "name": str(row["name"]),
                "aggregation": COMMITTEE_AGGREGATION_LABELS.get(
                    str(row["aggregation"]), str(row["aggregation"])
                ),
                "members": ", ".join(
                    str(member["classifier_name"]) for member in row["members"]
                ),
            }
            for row in committees
        ]
        return {"display": "flex"}, member_options, rows, message, next_refresh

    @app.callback(
        Output("committee-manager-rename-name", "value"),
        Input("committee-table", "selected_rows"),
        Input("committee-table", "data"),
        prevent_initial_call=True,
    )
    def populate_committee_rename_name(
        selected_rows: list[int] | None,
        table_data: list[dict[str, Any]] | None,
    ) -> str:
        rows = table_data or []
        selection = selected_rows or []
        if not selection:
            return ""
        try:
            return str(rows[int(selection[0])]["name"])
        except (IndexError, KeyError, TypeError, ValueError):
            return ""

    @app.callback(
        Output("testing-center-modal", "style"),
        Output("testing-center-summary", "children"),
        Output("testing-center-table", "data"),
        Output("testing-center-events", "data"),
        Input("focus-testing-center-button", "n_clicks"),
        Input("testing-center-close", "n_clicks"),
        Input("focus-code-dropdown", "value"),
        Input("teaching-example-refresh-store", "data"),
        Input("focus-model-refresh-store", "data"),
        State("testing-center-modal", "style"),
        prevent_initial_call=True,
    )
    def manage_testing_center(
        open_clicks: int,
        close_clicks: int,
        code_id: int | None,
        teaching_refresh: int | None,
        model_refresh: int | None,
        current_style: dict[str, str] | None,
    ) -> tuple[dict[str, str], Any, list[dict[str, Any]], list[dict[str, Any]]]:
        del open_clicks, close_clicks, teaching_refresh, model_refresh
        triggered = str(callback_context.triggered_id)
        if triggered == "testing-center-close":
            return {"display": "none"}, "", [], []
        if triggered != "focus-testing-center-button" and (current_style or {}).get("display") == "none":
            return {"display": "none"}, "", [], []
        if code_id is None:
            return {"display": "flex"}, "Select a code first.", [], []
        summary = project.classifier_testing_summary(int(code_id))
        classifier_rows = [
            {
                "classifier": row["classifier_name"],
                "n": int(row["n"]),
                "accuracy": f"{float(row['accuracy']):.3f}",
                "brier": (
                    f"{float(row['brier']):.3f}" if row["brier"] is not None else "—"
                ),
            }
            for row in summary["classifiers"]
        ]
        event_rows = [
            {
                "event_type": row["event_type"],
                "classifier": row["classifier_name"],
                "human": row["human_label"],
                "predicted": row["predicted_label"],
                "probability": (
                    f"{float(row['probability']):.3f}"
                    if row["probability"] is not None
                    else "—"
                ),
            }
            for row in summary["events"]
        ]
        return (
            {"display": "flex"},
            f"{len(event_rows):,} stored diagnostic assessment events.",
            classifier_rows,
            event_rows,
        )

    @app.callback(
        Output("focus-active-classifier-dropdown", "options"),
        Output("focus-active-classifier-dropdown", "value"),
        Input("focus-code-dropdown", "value"),
        Input("classifier-spec-refresh-store", "data"),
        State("focus-active-classifier-dropdown", "value"),
    )
    def refresh_focus_classifier_specs(
        code_id: int | None,
        refresh: int | None,
        current_spec_id: int | None,
    ) -> tuple[list[dict[str, Any]], int | None]:
        del refresh
        if code_id is None:
            return [], None
        project.ensure_default_classifier_specs(code_id=int(code_id))
        rows = project.classifier_specs(code_id=int(code_id))
        options = [
            {
                "label": f"{row['name']} · {row['geometry_name']}",
                "value": int(row["classifier_spec_id"]),
            }
            for row in rows
        ]
        valid = {int(option["value"]) for option in options}
        value = (
            int(current_spec_id)
            if current_spec_id is not None and int(current_spec_id) in valid
            else (int(options[0]["value"]) if options else None)
        )
        return options, value

    @app.callback(
        Output("focus-committee-dropdown", "options"),
        Output("focus-committee-dropdown", "value"),
        Input("focus-code-dropdown", "value"),
        Input("committee-refresh-store", "data"),
        State("focus-committee-dropdown", "value"),
    )
    def refresh_focus_committees(
        code_id: int | None,
        refresh: int | None,
        current_committee_id: int | None,
    ) -> tuple[list[dict[str, Any]], int | None]:
        del refresh
        rows = project.classifier_committees(int(code_id)) if code_id is not None else []
        options = [
            {
                "label": f"{row['name']} ({len(row['members'])} classifiers)",
                "value": int(row["committee_id"]),
            }
            for row in rows
        ]
        valid = {int(option["value"]) for option in options}
        value = (
            int(current_committee_id)
            if current_committee_id is not None and int(current_committee_id) in valid
            else (int(options[0]["value"]) if options else None)
        )
        return options, value

    @app.callback(
        Output("focus-active-classifier-status", "children"),
        Output("focus-active-classifier-status", "className"),
        Output("focus-active-classifier-detail", "children"),
        Input("focus-code-dropdown", "value"),
        Input("focus-active-classifier-dropdown", "value"),
        Input("focus-refresh-store", "data"),
        Input("teaching-example-refresh-store", "data"),
        Input("focus-model-refresh-store", "data"),
        Input("classifier-spec-refresh-store", "data"),
    )
    def update_focus_active_classifier_status(
        code_id: int | None,
        classifier_spec_id: int | None,
        focus_refresh: int | None,
        teaching_refresh: int | None,
        model_refresh: int | None,
        spec_refresh: int | None,
    ) -> tuple[str, str, str]:
        del focus_refresh, teaching_refresh, model_refresh, spec_refresh
        if code_id is None:
            return "No code", "classifier-status-badge status-not-trained", ""
        if classifier_spec_id is None:
            return "No classifier", "classifier-status-badge status-not-trained", ""
        row = project.classifier_status(
            code_id=int(code_id), classifier_spec_ids=[int(classifier_spec_id)]
        )["classifiers"][0]
        status = str(row["status"])
        label = {
            "current": "Current",
            "stale": "Stale",
            "not_trained": "Not trained",
        }[status]
        algorithm = str(row["algorithm"])
        parameter_detail = hyperparameter_summary(algorithm, dict(row["hyperparameters"]))
        detail = f"{row['geometry_name']} · {classifier_algorithm_label(algorithm)}"
        if parameter_detail:
            detail += f" · {parameter_detail}"
        return label, f"classifier-status-badge status-{status.replace('_', '-')}", detail

    @app.callback(
        Output("focus-committee-status", "children"),
        Input("focus-code-dropdown", "value"),
        Input("focus-committee-dropdown", "value"),
        Input("focus-refresh-store", "data"),
        Input("teaching-example-refresh-store", "data"),
        Input("focus-model-refresh-store", "data"),
    )
    def update_focus_committee_status(
        code_id: int | None,
        committee_id: int | None,
        focus_refresh: int | None,
        teaching_refresh: int | None,
        model_refresh: int | None,
    ) -> Any:
        del focus_refresh, teaching_refresh, model_refresh
        if code_id is None:
            return "Select a code first."
        if committee_id is None:
            return "No committee selected."
        status = project.classifier_committee_status(
            code_id=int(code_id), committee_id=int(committee_id)
        )
        committee = status["committee"]
        counts = status["member_counts"]
        member_summary = (
            f"Members: {counts['current']} current · {counts['stale']} stale · "
            f"{counts['not_trained']} not trained"
        )
        aggregation_label = COMMITTEE_AGGREGATION_LABELS.get(
            str(committee["aggregation"]), str(committee["aggregation"])
        )
        if status["status"] == "no_training_required":
            return f"{aggregation_label} · No committee training required · {member_summary}"
        committee_fit_label = {
            "current": "Current",
            "stale": "Stale",
            "not_trained": "Not trained",
        }[str(status["status"])]
        return f"{aggregation_label} · Committee fit: {committee_fit_label} · {member_summary}"

    @app.callback(
        Output("focus-train-action-status", "children"),
        Output("focus-model-refresh-store", "data", allow_duplicate=True),
        Input("focus-train-active", "n_clicks"),
        State("focus-code-dropdown", "value"),
        State("focus-active-classifier-dropdown", "value"),
        State("focus-train-scope", "value"),
        State("focus-tune-hyperparameters", "value"),
        State("focus-model-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def train_active_classifier(
        clicks: int,
        code_id: int | None,
        classifier_spec_id: int | None,
        train_scope: list[str] | None,
        tune_values: list[str] | None,
        refresh: int | None,
    ) -> tuple[str, int]:
        del clicks
        if code_id is None or classifier_spec_id is None:
            return "Select a code and classifier first.", int(refresh or 0)
        train_all = "all" in set(train_scope or [])
        tune_hyperparameters = "tune" in set(tune_values or [])
        spec_ids = (
            [
                int(row["classifier_spec_id"])
                for row in project.classifier_specs(code_id=int(code_id))
            ]
            if train_all
            else [int(classifier_spec_id)]
        )
        try:
            fits = project.train_classifiers(
                code_id=int(code_id),
                classifier_spec_ids=spec_ids,
                tune=tune_hyperparameters,
                retune_current=tune_hyperparameters,
            )
        except (KeyError, ValueError) as error:
            return str(error), int(refresh or 0)
        selections = [dict(fit.get("training_selection", {})) for fit in fits.values()]
        tuned_count = sum(bool(selection.get("tuned")) for selection in selections)
        reused_count = sum(
            selection.get("reason") == "current_fit_reused" for selection in selections
        )
        fast_fit_count = sum(
            selection.get("reason") == "tuning_disabled" for selection in selections
        )
        insufficient_count = sum(
            selection.get("reason") == "insufficient_cv_data" for selection in selections
        )
        if train_all:
            parts = [f"Updated {len(fits):,} classifiers"]
            if tuned_count:
                parts.append(f"CV-tuned {tuned_count:,}")
            if fast_fit_count:
                parts.append(f"fit {fast_fit_count:,} with current hyperparameters")
            if insufficient_count:
                parts.append(
                    f"could not CV-tune {insufficient_count:,} with too little labeled data"
                )
            if reused_count:
                parts.append(f"reused {reused_count:,} current fits")
            message = ". ".join(parts) + "."
        else:
            fit = next(iter(fits.values()))
            selection = dict(fit.get("training_selection", {}))
            name = str(fit["classifier_name"])
            if selection.get("tuned"):
                message = (
                    f"CV-tuned {name} with {int(selection['folds'])}-fold "
                    f"cross-validation."
                )
                if selection.get("fit_reused"):
                    message += " The selected configuration was already fitted."
            elif selection.get("reason") == "insufficient_cv_data":
                message = (
                    f"CV tuning requested for {name}, but more labeled examples are "
                    "needed; using the current hyperparameters."
                )
            elif selection.get("reason") == "current_fit_reused":
                message = f"{name} is already current."
            elif selection.get("reason") == "tuning_disabled":
                message = f"Trained {name} with current hyperparameters."
            else:
                message = f"Trained {name}."
        return message, int(refresh or 0) + 1

    @app.callback(
        Output("focus-train-committee", "style"),
        Output("focus-train-committee", "disabled"),
        Input("focus-committee-dropdown", "value"),
        Input("committee-refresh-store", "data"),
    )
    def configure_committee_train_button(
        committee_id: int | None, refresh: int | None
    ) -> tuple[dict[str, str], bool]:
        del refresh
        if committee_id is None:
            return {"display": "none"}, True
        committee = project.database.get_classifier_committee(int(committee_id))
        trainable = str(committee["aggregation"]) == "logistic_stack"
        return ({} if trainable else {"display": "none"}), not trainable

    @app.callback(
        Output("focus-committee-train-status", "children"),
        Output("focus-model-refresh-store", "data", allow_duplicate=True),
        Input("focus-train-committee", "n_clicks"),
        State("focus-code-dropdown", "value"),
        State("focus-committee-dropdown", "value"),
        State("focus-model-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def train_selected_committee(
        clicks: int,
        code_id: int | None,
        committee_id: int | None,
        refresh: int | None,
    ) -> tuple[str, int]:
        del clicks
        if code_id is None or committee_id is None:
            return "Select a code and committee first.", int(refresh or 0)
        try:
            fit = project.train_classifier_committee(
                code_id=int(code_id), committee_id=int(committee_id)
            )
        except (KeyError, ValueError) as error:
            return str(error), int(refresh or 0)
        return (
            f"Trained learned committee aggregation #{int(fit['committee_fit_id'])}.",
            int(refresh or 0) + 1,
        )

    @app.callback(
        Output("focus-disagreement", "style"),
        Input("focus-recommendation-source", "value"),
    )
    def show_committee_disagreement(source: str | None) -> dict[str, str]:
        return {} if source == "committee" else {"display": "none"}

    focus_recommendation_inputs = [
        Input("focus-most-likely", "n_clicks"),
        Input("focus-least-likely", "n_clicks"),
        Input("focus-most-uncertain", "n_clicks"),
        Input("focus-disagreement", "n_clicks"),
        Input("focus-review-unsure", "n_clicks"),
        Input("focus-random", "n_clicks"),
    ]

    @app.callback(
        Output("focal-unit-store", "data", allow_duplicate=True),
        Output("navigation-history-store", "data", allow_duplicate=True),
        Output("focus-recommendation-status", "children"),
        Output("focus-recommendation-store", "data"),
        *focus_recommendation_inputs,
        State("focus-recommendation-source", "value"),
        State("focus-code-dropdown", "value"),
        State("active-session-store", "data"),
        State("focus-active-classifier-dropdown", "value"),
        State("focus-committee-dropdown", "value"),
        State("focus-train-scope", "value"),
        State("focus-auto-retrain", "value"),
        State("focus-tune-hyperparameters", "value"),
        State("focal-unit-store", "data"),
        State("navigation-history-store", "data"),
        prevent_initial_call=True,
    )
    def focus_recommend(
        most_likely_clicks: int,
        least_likely_clicks: int,
        most_uncertain_clicks: int,
        disagreement_clicks: int,
        review_unsure_clicks: int,
        random_clicks: int,
        recommendation_source: str | None,
        code_id: int | None,
        session_id: int,
        active_classifier_spec_id: int | None,
        committee_id: int | None,
        train_scope_values: list[str] | None,
        auto_retrain_values: list[str] | None,
        tune_hyperparameter_values: list[str] | None,
        focal_unit_id: int | None,
        history: dict[str, Any] | None,
    ) -> tuple[Any, Any, str, dict[str, Any]]:
        del most_likely_clicks, least_likely_clicks, most_uncertain_clicks
        del disagreement_clicks, review_unsure_clicks, random_clicks
        if code_id is None:
            return no_update, no_update, "Create or select a code first.", {}
        source = str(recommendation_source or "classifier")
        triggered = str(callback_context.triggered_id)
        strategy_by_button = {
            "focus-most-likely": "likely_positive",
            "focus-least-likely": "likely_negative",
            "focus-most-uncertain": (
                "committee_uncertainty"
                if source == "committee"
                else "classifier_uncertainty"
            ),
            "focus-disagreement": "classifier_disagreement",
            "focus-review-unsure": "review_unsure",
            "focus-random": "random",
        }
        strategy = strategy_by_button[triggered]
        if source == "committee" and committee_id is None and strategy not in {
            "random",
            "review_unsure",
        }:
            return no_update, no_update, "Select a classifier committee first.", {}
        try:
            result = project.focus_recommendation(
                code_id=int(code_id),
                session_id=int(session_id),
                strategy=strategy,
                active_classifier_spec_id=(
                    int(active_classifier_spec_id)
                    if active_classifier_spec_id is not None
                    else None
                ),
                committee_id=(int(committee_id) if committee_id is not None else None),
                recommendation_source=source,
                auto_retrain="auto" in set(auto_retrain_values or []),
                auto_train_all="all" in set(train_scope_values or []),
                tune_hyperparameters="tune" in set(tune_hyperparameter_values or []),
            )
        except (KeyError, ValueError) as error:
            return no_update, no_update, str(error), {}
        if result is None:
            if strategy == "review_unsure":
                return no_update, no_update, "This code has no unsure units to revisit.", {}
            return no_update, no_update, "No unseen, unreviewed units remain in this session.", {}

        fit_ids = list(result.get("classifier_fit_ids", {}).values())
        classifier_fit_id = int(fit_ids[0]) if len(fit_ids) == 1 else None
        project.record_visit(
            int(session_id),
            int(result["unit_id"]),
            method=f"focus_{result['strategy']}",
            source_unit_id=int(focal_unit_id) if focal_unit_id is not None else None,
            active_code_id=int(code_id),
            classifier_fit_id=classifier_fit_id,
        )
        if result["fallback"]:
            status = (
                "Both positive and negative labels are required. "
                "GeCo supplied a random unseen unit instead."
            )
        else:
            status = (
                f"Opened unit {int(result['unit_id']):,} via "
                f"{result['strategy'].replace('_', ' ')}."
            )
            if list(result.get("stale_classifiers", [])):
                status += (
                    " Using classifier scores from before the latest labels; "
                    "train when the current labeling batch is ready."
                )
        unit_id = int(result["unit_id"])
        return unit_id, _append_navigation_history(history, unit_id), status, result

    @app.callback(
        Output("focus-annotation-status", "children"),
        Output("focus-refresh-store", "data", allow_duplicate=True),
        Output("focus-prediction-map", "figure", allow_duplicate=True),
        Input("focus-label-positive", "n_clicks"),
        Input("focus-label-negative", "n_clicks"),
        Input("focus-label-unsure", "n_clicks"),
        State("focus-code-dropdown", "value"),
        State("focal-unit-store", "data"),
        State("focus-recommendation-store", "data"),
        State("focus-span-selection-store", "data"),
        State("focus-refresh-store", "data"),
        State("focus-prediction-map", "figure"),
        prevent_initial_call=True,
    )
    def annotate_focus_unit(
        positive_clicks: int,
        negative_clicks: int,
        unsure_clicks: int,
        code_id: int | None,
        focal_unit_id: int | None,
        recommendation: dict[str, Any] | None,
        selected_unit_ids: list[int] | None,
        refresh: int | None,
        prediction_figure: dict[str, Any] | None,
    ) -> tuple[str, int, Any]:
        del positive_clicks, negative_clicks, unsure_clicks
        if code_id is None or focal_unit_id is None:
            return (
                "Select a code and focal unit before labeling.",
                int(refresh or 0),
                no_update,
            )
        value_by_button = {
            "focus-label-positive": "positive",
            "focus-label-negative": "negative",
            "focus-label-unsure": "unsure",
        }
        value = value_by_button[str(callback_context.triggered_id)]
        selected = [int(item) for item in (selected_unit_ids or [focal_unit_id])]
        observation_id = _selection_observation_id(
            project, int(focal_unit_id), selected
        )
        current = _current_annotation_value(project, observation_id, int(code_id))
        if current == value:
            return "", int(refresh or 0), no_update
        fit_ids = list((recommendation or {}).get("classifier_fit_ids", {}).values())
        classifier_fit_id = int(fit_ids[0]) if len(fit_ids) == 1 else None
        observation_id, _event_id = project.annotate_span(
            selected,
            int(code_id),
            value,
            origin="human_active_learning",
            classifier_fit_id=classifier_fit_id,
        )
        for cached in prediction_geometry_cache.values():
            if int(cached.get("code_id", -1)) != int(code_id):
                continue
            for point in cached.get("points", []):
                if int(point["observation_id"]) == int(observation_id):
                    point["assignment"] = value

        color = {
            "positive": "#3f8d5d",
            "negative": "#a83f3f",
            "unsure": "#d39a22",
        }[value]
        patch: Any = no_update
        if prediction_figure:
            for trace_index, trace in enumerate(prediction_figure.get("data", [])):
                for position, customdata in enumerate(trace.get("customdata") or []):
                    if (
                        isinstance(customdata, (list, tuple))
                        and customdata
                        and int(customdata[0]) == int(observation_id)
                    ):
                        patch = Patch()
                        patch["data"][trace_index]["marker"]["color"][position] = color
                        patch["data"][trace_index]["marker"]["line"]["color"][position] = color
                        break
                if patch is not no_update:
                    break
        return "", int(refresh or 0) + 1, patch

    @app.callback(
        Output("focus-code-definition", "children"),
        Output("focus-label-counts", "children"),
        Output("focus-current-label", "children"),
        Input("focus-code-dropdown", "value"),
        Input("focus-refresh-store", "data"),
        Input("teaching-example-refresh-store", "data"),
        Input("focus-focal-unit-store", "data"),
    )
    def update_focus_code_summary(
        code_id: int | None,
        refresh: int,
        teaching_refresh: int | None,
        focal_unit_id: int | None,
    ) -> tuple[Any, Any, Any]:
        del refresh, teaching_refresh
        if code_id is None:
            return "No code selected.", "", ""
        code = project.database.get_code(int(code_id))
        counts = project.database.annotation_counts(int(code_id))
        atomic_reviewed = sum(
            1
            for row in project.current_annotations(int(code_id))
            if row["unit_id"] is not None
        )
        unreviewed = max(0, len(units) - atomic_reviewed)
        description = code["description"] or "No description has been written yet."
        count_text = (
            f"Positive {counts['positive']:,} · Negative {counts['negative']:,} · "
            f"Unsure {counts['unsure']:,} · Unreviewed {unreviewed:,}"
        )
        training_counts = project.training_label_counts(int(code_id))
        model_note = (
            "Model-ready"
            if training_counts["positive"] > 0 and training_counts["negative"] > 0
            else "Random sampling until both active positive and negative examples exist"
        )
        return description, [html.Div(count_text), html.Small(model_note)], ""

    @app.callback(
        Output("focus-label-positive", "disabled"),
        Output("focus-label-negative", "disabled"),
        Output("focus-label-unsure", "disabled"),
        Output("focus-label-positive", "className"),
        Output("focus-label-negative", "className"),
        Output("focus-label-unsure", "className"),
        Input("focus-code-dropdown", "value"),
        Input("focus-focal-unit-store", "data"),
        Input("focus-span-selection-store", "data"),
        Input("focus-refresh-store", "data"),
    )
    def style_focus_judgment_buttons(
        code_id: int | None,
        focal_unit_id: int | None,
        selected_unit_ids: list[int] | None,
        refresh: int | None,
    ) -> tuple[bool, bool, bool, str, str, str]:
        del refresh
        disabled = code_id is None or focal_unit_id is None
        current = None
        if not disabled:
            observation_id = _selection_observation_id(
                project, int(focal_unit_id), selected_unit_ids
            )
            current = _current_annotation_value(
                project, observation_id, int(code_id)
            )
        classes = _judgment_button_classes(current, prefix="focus-label")
        return disabled, disabled, disabled, *classes

    @app.callback(
        Output("focus-focal-key", "children"),
        Output("focus-context-content", "children"),
        Output("focus-metadata-content", "children"),
        Input("focus-focal-unit-store", "data"),
        Input("focus-context-level", "value"),
        Input("focus-context-window", "value"),
    )
    def update_focus_reading_panel(
        focal_unit_id: int | None,
        level_index: int,
        window: int,
    ) -> tuple[Any, Any, Any]:
        if focal_unit_id is None:
            return (
                "No focal unit",
                "Choose a recommendation to begin developing this code.",
                "",
            )
        focal = project.unit(int(focal_unit_id))
        context = project.context(int(focal_unit_id), level=int(level_index), window=int(window))
        key_label = " › ".join(f"{name}: {value}" for name, value in focal["user_key"].items())
        children = _render_context_units(
            html,
            dcc,
            context,
            focal_unit_id=int(focal_unit_id),
            key_columns=list(metadata["key_columns"]),
            selected_unit_ids=[int(focal_unit_id)],
            checkbox_type="focus-span-checkbox",
            span_selection_enabled=project.can_transform_new_observations(),
        )
        return key_label, children, _metadata_table(html, focal["metadata"])

    @app.callback(
        Output("focus-span-selection-store", "data"),
        Input("focus-focal-unit-store", "data"),
        Input("focus-context-level", "value"),
        Input("focus-context-window", "value"),
    )
    def reset_focus_span_selection(
        focal_unit_id: int | None, level_index: int, window: int
    ) -> list[int]:
        del level_index, window
        return [int(focal_unit_id)] if focal_unit_id is not None else []

    @app.callback(
        Output("focus-span-selection-store", "data", allow_duplicate=True),
        Input({"type": "focus-span-checkbox", "unit_id": ALL}, "value"),
        State({"type": "focus-span-checkbox", "unit_id": ALL}, "id"),
        State("focal-unit-store", "data"),
        State("focus-context-level", "value"),
        State("focus-context-window", "value"),
        State("focus-span-selection-store", "data"),
        prevent_initial_call=True,
    )
    def update_focus_span_selection(
        values: list[list[int]], ids: list[dict[str, Any]],
        focal_unit_id: int | None, level_index: int, window: int,
        current: list[int] | None,
    ) -> Any:
        triggered = callback_context.triggered_id
        if focal_unit_id is None or not isinstance(triggered, dict):
            return no_update
        clicked = int(triggered.get("unit_id"))
        value_by_id = {
            int(identifier["unit_id"]): value
            for identifier, value in zip(ids or [], values or [], strict=False)
        }
        checked = bool(value_by_id.get(clicked))
        if not checked and clicked not in set(current or []):
            return no_update
        context = project.context(
            int(focal_unit_id), level=int(level_index), window=int(window)
        )
        updated = _update_span_selection(
            context, focal_unit_id=int(focal_unit_id), current_unit_ids=current,
            clicked_unit_id=clicked, checked=checked,
        )
        return updated if updated != list(current or []) else no_update

    @app.callback(
        Output("focus-context-window", "value", allow_duplicate=True),
        Input("focus-context-level", "value"),
        Input("focus-context-window-minus", "n_clicks"),
        Input("focus-context-window-plus", "n_clicks"),
        State("focus-context-window", "value"),
        prevent_initial_call=True,
    )
    def update_focus_context_window(
        level_index: int,
        minus_clicks: int,
        plus_clicks: int,
        current_window: int | None,
    ) -> int:
        del level_index, minus_clicks, plus_clicks
        triggered = callback_context.triggered_id
        if triggered == "focus-context-level":
            return 1
        current = max(1, int(current_window or 1))
        if triggered == "focus-context-window-minus":
            return max(1, current - 1)
        if triggered == "focus-context-window-plus":
            return current + 1
        return current

    @app.callback(
        Output("focus-probabilities", "children"),
        Input("focus-recommendation-store", "data"),
        Input("focus-focal-unit-store", "data"),
    )
    def update_focus_probabilities(
        recommendation: dict[str, Any] | None,
        focal_unit_id: int | None,
    ) -> Any:
        data = recommendation or {}
        if focal_unit_id is None or int(data.get("unit_id", -1)) != int(focal_unit_id):
            return ""
        probabilities = dict(data.get("probabilities", {}))
        committee_probability = data.get("committee_probability")
        if not probabilities and committee_probability is None:
            return html.P("No model probabilities were used for this recommendation.")
        content: list[Any] = []
        if probabilities:
            rows = [
                html.Tr([html.Th(name), html.Td(f"{float(probability):.3f}")])
                for name, probability in probabilities.items()
            ]
            content.extend(
                [html.H4("Model probabilities"), html.Table(html.Tbody(rows))]
            )
        if committee_probability is not None:
            content.extend(
                [
                    html.H4("Committee probability"),
                    html.P(f"{float(committee_probability):.3f}"),
                ]
            )
        return content

    @app.callback(
        Output("focus-prediction-fits", "options"),
        Output("focus-prediction-fits", "value"),
        Input("focus-code-dropdown", "value"),
        Input("focus-model-refresh-store", "data"),
        Input("classifier-spec-refresh-store", "data"),
        State("focus-prediction-fits", "value"),
    )
    def refresh_prediction_fit_options(
        code_id: int | None,
        model_refresh: int | None,
        spec_refresh: int | None,
        current_values: list[int] | None,
    ) -> tuple[list[dict[str, Any]], list[int]]:
        del model_refresh, spec_refresh
        if code_id is None:
            return [], []
        status_rows = project.classifier_status(code_id=int(code_id))["classifiers"]
        options = [
            {
                "label": (
                    f"{row['name']} · {row['geometry_name']} · "
                    f"{str(row['status']).replace('_', ' ').title()}"
                ),
                # Store classifier specification IDs so an explicit retrain does
                # not silently replace the user's prediction-geometry selection.
                "value": int(row["classifier_spec_id"]),
                "disabled": row["classifier_fit_id"] is None,
            }
            for row in status_rows
        ]
        valid = {
            int(option["value"])
            for option in options
            if not bool(option.get("disabled"))
        }
        selected = [
            int(value) for value in (current_values or []) if int(value) in valid
        ]
        if not selected and valid:
            selected = [
                next(
                    int(option["value"])
                    for option in options
                    if int(option["value"]) in valid
                )
            ]
        return options, selected

    @app.callback(
        Output("prediction-geometry-refresh-store", "data"),
        Output("focus-prediction-status", "children"),
        Input("focus-prediction-refresh", "n_clicks"),
        State("focus-code-dropdown", "value"),
        State("focus-prediction-fits", "value"),
        prevent_initial_call=True,
    )
    def refresh_prediction_geometry(
        clicks: int,
        code_id: int | None,
        classifier_spec_ids: list[int] | None,
    ) -> tuple[dict[str, Any] | Any, str]:
        del clicks
        if code_id is None:
            return no_update, "Select a code first."
        spec_ids = [int(value) for value in (classifier_spec_ids or [])]
        if not spec_ids:
            return no_update, "Select at least one trained classifier."
        status_rows = project.classifier_status(
            code_id=int(code_id), classifier_spec_ids=spec_ids
        )["classifiers"]
        missing = [
            str(row["name"])
            for row in status_rows
            if row["classifier_fit_id"] is None
        ]
        if missing:
            return no_update, "Train these classifiers first: " + ", ".join(missing) + "."
        fit_ids = [int(row["classifier_fit_id"]) for row in status_rows]
        try:
            geometry = project.classifier_prediction_geometry(
                code_id=int(code_id), classifier_fit_ids=fit_ids
            )
        except (KeyError, ValueError) as error:
            return no_update, str(error)
        cache_key = uuid.uuid4().hex
        geometry["code_id"] = int(code_id)
        prediction_geometry_cache[cache_key] = geometry
        while len(prediction_geometry_cache) > 4:
            prediction_geometry_cache.pop(next(iter(prediction_geometry_cache)))
        method_labels = {
            "one_dimension_with_jitter": "one classifier with deterministic jitter",
            "two_classifier_scores": "two direct classifier-score axes",
            "pca": "PCA of classifier scores",
        }
        return (
            {
                "cache_key": cache_key,
                "code_id": int(code_id),
                "classifier_spec_ids": spec_ids,
                "classifier_fit_ids": fit_ids,
            },
            (
                f"Refreshed {len(geometry['points']):,} scored observations using "
                f"{method_labels[geometry['method']]}."
            ),
        )

    @app.callback(
        Output("focus-prediction-page-number", "max"),
        Output("focus-prediction-page-number", "value", allow_duplicate=True),
        Input("focus-prediction-page-previous", "n_clicks"),
        Input("focus-prediction-page-next", "n_clicks"),
        Input("focus-prediction-page-size", "value"),
        Input("focus-prediction-display-options", "value"),
        Input("prediction-geometry-refresh-store", "data"),
        State("focus-prediction-page-number", "value"),
        prevent_initial_call=True,
    )
    def change_prediction_geometry_page(
        previous_clicks: int,
        next_clicks: int,
        page_size: Any,
        display_options: list[str] | None,
        refresh_state: dict[str, Any] | None,
        page_number: Any,
    ) -> tuple[int, int]:
        del previous_clicks, next_clicks
        geometry = _prediction_geometry_from_cache(
            prediction_geometry_cache, refresh_state
        )
        visible = _prediction_geometry_visible_points(geometry, display_options)
        size = _coerce_positive_int(page_size, default=5000)
        page_count = deterministic_page(len(visible), page_size=size).page_count
        current = min(max(1, _coerce_positive_int(page_number, default=1)), page_count)
        triggered = callback_context.triggered_id
        if triggered in {
            "focus-prediction-page-size",
            "focus-prediction-display-options",
            "prediction-geometry-refresh-store",
        }:
            current = 1
        elif triggered == "focus-prediction-page-previous":
            current = max(1, current - 1)
        elif triggered == "focus-prediction-page-next":
            current = min(page_count, current + 1)
        return page_count, current

    @app.callback(
        Output("focus-prediction-map", "figure"),
        Output("focus-prediction-page-status", "children"),
        Input("prediction-geometry-refresh-store", "data"),
        Input("focus-prediction-page-size", "value"),
        Input("focus-prediction-page-number", "value"),
        Input("focus-prediction-display-options", "value"),
    )
    def render_prediction_geometry(
        refresh_state: dict[str, Any] | None,
        page_size: Any,
        page_number: Any,
        display_options: list[str] | None,
    ) -> tuple[Any, str]:
        geometry = _prediction_geometry_from_cache(
            prediction_geometry_cache, refresh_state
        )
        if not geometry:
            return go.Figure(), "Refresh the score view to display observations."
        visible = _prediction_geometry_visible_points(geometry, display_options)
        if not visible:
            return go.Figure(), "No observations match the selected display options."
        size = _coerce_positive_int(page_size, default=5000)
        number = _coerce_positive_int(page_number, default=1)
        include_coded = "coded" in set(display_options or [])
        coded_positions = {
            index
            for index, point in enumerate(visible)
            if point.get("assignment") in {"positive", "negative", "unsure"}
        }
        page = deterministic_page_with_inclusions(
            len(visible),
            page_size=size,
            page_index=number - 1,
            seed=0,
            include_positions=(coded_positions if include_coded else ()),
        )
        points = [visible[int(position)] for position in page.positions]
        figure = _build_prediction_geometry_figure(
            geometry=geometry,
            points=points,
            page_index=page.page_index,
        )
        extras = max(
            0,
            len(page.positions)
            - min(size, len(visible) - page.page_index * size),
        )
        status = (
            f"Page {page.page_index + 1} of {page.page_count} · "
            f"{len(points):,} points shown"
        )
        if include_coded:
            status += f" · all {len(coded_positions):,} coded observations included"
            if extras:
                status += f" ({extras:,} added beyond the page sample)"
        return figure, status

    @app.callback(
        Output("focus-prediction-status", "children", allow_duplicate=True),
        Input("focus-code-dropdown", "value"),
        Input("focus-model-refresh-store", "data"),
        Input("focus-refresh-store", "data"),
        Input("teaching-example-refresh-store", "data"),
        State("focus-prediction-fits", "value"),
        prevent_initial_call=True,
    )
    def mark_prediction_geometry_stale(
        code_id: int | None,
        model_refresh: int | None,
        focus_refresh: int | None,
        teaching_refresh: int | None,
        classifier_spec_ids: list[int] | None,
    ) -> Any:
        del model_refresh, focus_refresh, teaching_refresh
        if code_id is None or not classifier_spec_ids:
            return no_update
        status_rows = project.classifier_status(
            code_id=int(code_id),
            classifier_spec_ids=[int(value) for value in classifier_spec_ids],
        )["classifiers"]
        if any(row["classifier_fit_id"] is None for row in status_rows):
            return "One or more selected classifiers have not been trained."
        if any(str(row["status"]) == "stale" for row in status_rows):
            return (
                "Selected classifiers are stale because labels or settings changed. "
                "The plotted scores remain the last trained snapshot until you train and refresh."
            )
        return no_update

    @app.callback(
        Output("focal-unit-store", "data", allow_duplicate=True),
        Output("navigation-history-store", "data", allow_duplicate=True),
        Output("focus-span-selection-store", "data", allow_duplicate=True),
        Input("focus-prediction-map", "clickData"),
        State("active-session-store", "data"),
        State("focus-code-dropdown", "value"),
        State("prediction-geometry-refresh-store", "data"),
        State("focal-unit-store", "data"),
        State("navigation-history-store", "data"),
        prevent_initial_call=True,
    )
    def open_prediction_geometry_point(
        click_data: dict[str, Any] | None,
        session_id: int,
        code_id: int | None,
        prediction_refresh: dict[str, Any] | None,
        focal_unit_id: int | None,
        history: dict[str, Any] | None,
    ) -> tuple[Any, Any, Any]:
        if not click_data or not click_data.get("points"):
            return no_update, no_update, no_update
        customdata = click_data["points"][0].get("customdata")
        if not isinstance(customdata, (list, tuple)) or len(customdata) < 3:
            return no_update, no_update, no_update
        observation_id = int(customdata[0])
        kind = str(customdata[2])
        if kind == "teaching_example":
            return no_update, no_update, no_update
        if kind == "span":
            observation = project.observation(observation_id)
            member_unit_ids = [int(value) for value in observation["member_unit_ids"]]
            if not member_unit_ids:
                return no_update, no_update, no_update
            unit_id = member_unit_ids[len(member_unit_ids) // 2]
            selection = member_unit_ids
        else:
            if customdata[1] is None:
                return no_update, no_update, no_update
            unit_id = int(customdata[1])
            selection = [unit_id]
        fit_ids = [
            int(value)
            for value in (prediction_refresh or {}).get("classifier_fit_ids", [])
        ]
        project.record_visit(
            int(session_id),
            unit_id,
            method="focus_prediction_geometry",
            source_unit_id=int(focal_unit_id) if focal_unit_id is not None else None,
            active_code_id=int(code_id) if code_id is not None else None,
            classifier_fit_id=(fit_ids[0] if len(fit_ids) == 1 else None),
        )
        return unit_id, _append_navigation_history(history, unit_id), selection

    @app.callback(
        Output("apply-code-dropdown", "options"),
        Output("apply-code-dropdown", "value"),
        Input("focus-refresh-store", "data"),
        State("apply-code-dropdown", "value"),
    )
    def refresh_apply_codes(
        focus_refresh: int,
        current_code_id: int | None,
    ) -> tuple[list[dict[str, Any]], int | None]:
        del focus_refresh
        code_options = _code_options(project.codes())
        valid_codes = {int(option["value"]) for option in code_options}
        code_id = (
            int(current_code_id)
            if current_code_id is not None and int(current_code_id) in valid_codes
            else (code_options[0]["value"] if code_options else None)
        )
        return code_options, code_id

    @app.callback(
        Output("apply-run-dropdown", "options"),
        Output("apply-run-dropdown", "value"),
        Input("apply-code-dropdown", "value"),
        Input("apply-refresh-store", "data"),
        State("apply-run-dropdown", "value"),
    )
    def refresh_apply_runs(
        code_id: int | None,
        apply_refresh: int,
        current_run_id: int | None,
    ) -> tuple[list[dict[str, Any]], int | None]:
        del apply_refresh
        runs = project.apply_runs(code_id=code_id) if code_id is not None else []
        reviewable_runs = [
            row
            for row in runs
            if str(row["status"]) == "draft"
            or project.can_undo_apply_commit(int(row["apply_run_id"]))
        ]
        run_options = _apply_run_options(reviewable_runs)
        valid_runs = {int(option["value"]) for option in run_options}
        run_id = (
            int(current_run_id)
            if current_run_id is not None and int(current_run_id) in valid_runs
            else (run_options[0]["value"] if run_options else None)
        )
        return run_options, run_id

    @app.callback(
        Output("apply-source-dropdown", "options"),
        Output("apply-source-dropdown", "value"),
        Input("apply-code-dropdown", "value"),
        Input("focus-model-refresh-store", "data"),
        Input("classifier-spec-refresh-store", "data"),
        Input("committee-refresh-store", "data"),
        Input("focus-refresh-store", "data"),
        Input("teaching-example-refresh-store", "data"),
        State("apply-source-dropdown", "value"),
    )
    def refresh_apply_sources(
        code_id: int | None,
        model_refresh: int | None,
        spec_refresh: int | None,
        committee_refresh: int | None,
        focus_refresh: int | None,
        teaching_refresh: int | None,
        current_value: str | None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        del model_refresh, spec_refresh, committee_refresh, focus_refresh, teaching_refresh
        if code_id is None:
            return [], None
        status_rows = project.classifier_status(code_id=int(code_id))["classifiers"]
        options = [
            {
                "label": (
                    f"Classifier: {row['name']} · "
                    f"{str(row['status']).replace('_', ' ')}"
                ),
                "value": f"classifier:{int(row['classifier_spec_id'])}",
            }
            for row in status_rows
            if row["classifier_fit_id"] is not None
        ]
        for committee in project.classifier_committees(int(code_id)):
            committee_status = project.classifier_committee_status(
                code_id=int(code_id), committee_id=int(committee["committee_id"])
            )
            member_counts = committee_status["member_counts"]
            members_ready = int(member_counts["not_trained"]) == 0
            aggregation = str(committee["aggregation"])
            trainable = aggregation == "logistic_stack"
            committee_ready = (
                committee_status["latest_fit"] is not None if trainable else members_ready
            )
            if committee_ready:
                stale = (
                    committee_status["status"] == "stale"
                    if trainable
                    else int(member_counts["stale"]) > 0
                )
                options.append(
                    {
                        "label": (
                            f"Committee: {committee['name']}"
                            + (" · stale" if stale else "")
                        ),
                        "value": f"committee:{int(committee['committee_id'])}",
                    }
                )
        valid = {str(option["value"]) for option in options}
        value = (
            str(current_value)
            if current_value is not None and str(current_value) in valid
            else (str(options[0]["value"]) if options else None)
        )
        return options, value

    @app.callback(
        Output("apply-status", "children", allow_duplicate=True),
        Output("apply-refresh-store", "data", allow_duplicate=True),
        Output("apply-run-dropdown", "options", allow_duplicate=True),
        Output("apply-run-dropdown", "value", allow_duplicate=True),
        Input("apply-generate", "n_clicks"),
        State("apply-code-dropdown", "value"),
        State("apply-source-dropdown", "value"),
        State("apply-threshold", "value"),
        State("apply-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def generate_apply_run(
        n_clicks: int,
        code_id: int | None,
        source_value: str | None,
        threshold: float | None,
        refresh: int | None,
    ) -> tuple[str, int, Any, Any]:
        del n_clicks
        if code_id is None:
            return "Create or select a code first.", int(refresh or 0), no_update, no_update
        if not source_value or ":" not in str(source_value):
            return "Train and select a classifier or committee first.", int(refresh or 0), no_update, no_update
        source_kind, source_id_text = str(source_value).split(":", 1)
        try:
            draft_id = project.create_apply_run(
                code_id=int(code_id),
                source_kind=source_kind,
                source_id=int(source_id_text),
                threshold=float(threshold if threshold is not None else 0.5),
            )
        except (KeyError, ValueError) as error:
            return str(error), int(refresh or 0), no_update, no_update
        draft_options = _apply_run_options(
            [
                row
                for row in project.apply_runs(code_id=int(code_id))
                if str(row["status"]) == "draft"
                or project.can_undo_apply_commit(int(row["apply_run_id"]))
            ]
        )
        return (
            f"Created draft {draft_id}.",
            int(refresh or 0) + 1,
            draft_options,
            int(draft_id),
        )

    @app.callback(
        Output("apply-table", "rowData"),
        Output("apply-run-summary", "children"),
        Input("apply-run-dropdown", "value"),
        Input("apply-proposed-filter", "value"),
        Input("apply-decision-filter", "value"),
        Input("apply-probability-range", "value"),
        Input("apply-refresh-store", "data"),
    )
    def load_apply_table(
        draft_id: int | None,
        proposed_filter: str | None,
        decision_filter: str | None,
        probability_range: list[float] | None,
        refresh: int,
    ) -> tuple[list[dict[str, Any]], Any]:
        del refresh
        if draft_id is None:
            return [], "No proposal draft has been generated for this code."
        lower, upper = _normalized_probability_range(probability_range)
        draft = project.apply_run(int(draft_id))
        proposals = project.apply_proposals(
            int(draft_id),
            proposed_label=proposed_filter or None,
            decision=decision_filter or None,
            probability_min=lower,
            probability_max=upper,
        )
        data = [
            {
                "unit_id": int(row["unit_id"]),
                "key": " › ".join(
                    f"{name}: {value}" for name, value in row["user_key"].items()
                ),
                "text": str(row["text"]),
                "probability": round(float(row["probability"]), 6),
                "proposed_label": str(row["proposed_label"]),
                "decision": APPLY_DECISION_LABELS.get(
                    str(row["decision"]), str(row["decision"])
                ),
            }
            for row in proposals
        ]
        summary = (
            f"Draft {draft['apply_run_id']} · "
            f"{draft['reviewed_count']:,} of {draft['proposal_count']:,} decided · "
            f"threshold {draft['threshold']:.3f} · {draft['aggregation']} aggregation"
        )
        if int(draft.get("unreviewed_count", 0)):
            summary += f" · {int(draft['unreviewed_count']):,} marked Leave unreviewed"
        return data, summary

    @app.callback(
        Output("apply-status", "children", allow_duplicate=True),
        Output("apply-refresh-store", "data", allow_duplicate=True),
        Input("apply-table", "cellValueChanged"),
        State("apply-run-dropdown", "value"),
        State("apply-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def persist_apply_table_edits(
        changed: dict[str, Any] | list[dict[str, Any]] | None,
        draft_id: int | None,
        refresh: int | None,
    ) -> tuple[Any, int]:
        if not changed or draft_id is None:
            return no_update, int(refresh or 0)
        events = changed if isinstance(changed, list) else [changed]
        count = 0
        try:
            for event in events:
                if str(event.get("colId", "")) != "decision":
                    continue
                row = dict(event.get("data") or {})
                decision = _normalize_apply_decision(row.get("decision"))
                project.review_apply_proposal(
                    apply_run_id=int(draft_id),
                    unit_id=int(row["unit_id"]),
                    decision=decision,
                )
                count += 1
        except (KeyError, ValueError) as error:
            return str(error), int(refresh or 0)
        if count == 0:
            return no_update, int(refresh or 0)
        return (
            f"Saved {count:,} review decision{'s' if count != 1 else ''}.",
            int(refresh or 0) + 1,
        )

    apply_action_inputs = [
        Input("apply-bulk-apply", "n_clicks"),
        Input("apply-clear-decisions", "n_clicks"),
        Input("apply-clear-visible", "n_clicks"),
        Input("apply-commit", "n_clicks"),
        Input("apply-commit-visible", "n_clicks"),
        Input("apply-undo-last-commit", "n_clicks"),
    ]

    @app.callback(
        Output("apply-status", "children", allow_duplicate=True),
        Output("apply-refresh-store", "data", allow_duplicate=True),
        Output("apply-table", "selectedRows"),
        *apply_action_inputs,
        State("apply-run-dropdown", "value"),
        State("apply-table", "rowData"),
        State("apply-table", "virtualRowData"),
        State("apply-table", "paginationInfo"),
        State("apply-table", "selectedRows"),
        State("apply-bulk-decision", "value"),
        State("apply-bulk-scope", "value"),
        State("apply-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def apply_action(
        bulk_apply_clicks: int,
        clear_clicks: int,
        clear_visible_clicks: int,
        commit_clicks: int,
        commit_visible_clicks: int,
        undo_clicks: int,
        draft_id: int | None,
        rows: list[dict[str, Any]] | None,
        virtual_rows: list[dict[str, Any]] | None,
        pagination_info: dict[str, Any] | None,
        selected_rows: list[dict[str, Any]] | None,
        bulk_decision: str | None,
        bulk_scope: str | None,
        refresh: int | None,
    ) -> tuple[Any, int, Any]:
        del (
            bulk_apply_clicks,
            clear_clicks,
            clear_visible_clicks,
            commit_clicks,
            commit_visible_clicks,
            undo_clicks,
        )
        if draft_id is None:
            return "Select a draft first.", int(refresh or 0), no_update
        triggered = str(callback_context.triggered_id)
        current_page_rows = _apply_current_page_rows(
            virtual_rows if virtual_rows is not None else (rows or []),
            pagination_info,
        )
        current_page_ids = [int(row["unit_id"]) for row in current_page_rows]
        clear_selection: Any = no_update
        try:
            if triggered == "apply-bulk-apply":
                decision = _normalize_apply_decision(bulk_decision)
                unit_ids = _apply_bulk_scope_unit_ids(
                    scope=str(bulk_scope or "selected"),
                    current_page_rows=current_page_rows,
                    selected_rows=selected_rows or [],
                )
                if bulk_scope == "selected" and not unit_ids:
                    return (
                        "Select at least one proposal before applying a selected-row action.",
                        int(refresh or 0),
                        no_update,
                    )
                count = project.bulk_review_apply_proposals(
                    apply_run_id=int(draft_id),
                    decision=decision,
                    unit_ids=unit_ids,
                )
                scope_label = {
                    "all": "all proposals",
                    "current_page": "proposals on the current page",
                    "selected": "selected proposals",
                }.get(str(bulk_scope), "proposals")
                decision_label = APPLY_DECISION_LABELS.get(decision, decision)
                message = f"Applied {decision_label.lower()} to {count:,} {scope_label}."
                if bulk_scope == "selected":
                    clear_selection = []
            elif triggered == "apply-clear-decisions":
                count = project.bulk_review_apply_proposals(
                    apply_run_id=int(draft_id), decision="pending"
                )
                message = f"Cleared all {count:,} review decisions."
            elif triggered == "apply-clear-visible":
                count = project.bulk_review_apply_proposals(
                    apply_run_id=int(draft_id),
                    decision="pending",
                    unit_ids=current_page_ids,
                )
                message = f"Cleared {count:,} review decisions on the current page."
            elif triggered in {"apply-commit", "apply-commit-visible"}:
                current_page_only = triggered == "apply-commit-visible"
                result = project.commit_apply_run(
                    int(draft_id),
                    unit_ids=current_page_ids if current_page_only else None,
                    scope="current_page" if current_page_only else "all_reviewed",
                )
                scope_label = "current-page reviewed" if current_page_only else "reviewed"
                message = (
                    f"Processed {result['processed']:,} {scope_label} proposals: "
                    f"created {result['committed']:,} assignments"
                )
                if result["removed_unreviewed"]:
                    message += f", left {result['removed_unreviewed']:,} unreviewed"
                if result["skipped_conflicts"]:
                    message += (
                        f", skipped {result['skipped_conflicts']:,} existing assignments"
                    )
                message += f". {result['left_pending']:,} pending proposals remain."
            elif triggered == "apply-undo-last-commit":
                result = project.undo_last_apply_commit(int(draft_id))
                message = (
                    f"Undid the last commit: restored {result['restored_proposals']:,} "
                    f"proposals and retracted {result['retracted_assignments']:,} "
                    "assignments created by that commit."
                )
            else:
                return no_update, int(refresh or 0), no_update
        except (KeyError, ValueError) as error:
            return str(error), int(refresh or 0), no_update
        return message, int(refresh or 0) + 1, clear_selection

    @app.callback(
        Output("apply-undo-last-commit", "disabled"),
        Input("apply-run-dropdown", "value"),
        Input("apply-refresh-store", "data"),
    )
    def update_apply_undo_availability(
        draft_id: int | None, refresh: int | None
    ) -> bool:
        del refresh
        return draft_id is None or not project.can_undo_apply_commit(int(draft_id))

    @app.callback(
        Output("apply-discard-confirm", "displayed"),
        Input("apply-discard", "n_clicks"),
        prevent_initial_call=True,
    )
    def confirm_discard_draft(clicks: int) -> bool:
        return bool(clicks)

    @app.callback(
        Output("apply-status", "children", allow_duplicate=True),
        Output("apply-refresh-store", "data", allow_duplicate=True),
        Input("apply-discard-confirm", "submit_n_clicks"),
        State("apply-run-dropdown", "value"),
        State("apply-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def discard_draft(
        submit_clicks: int | None,
        draft_id: int | None,
        refresh: int | None,
    ) -> tuple[Any, int]:
        if not submit_clicks or draft_id is None:
            return no_update, int(refresh or 0)
        try:
            project.discard_apply_run(int(draft_id))
        except (KeyError, ValueError) as error:
            return str(error), int(refresh or 0)
        return f"Discarded draft {int(draft_id)} without changing annotations.", int(refresh or 0) + 1

    @app.callback(
        Output("session-state-status", "children", allow_duplicate=True),
        Input("explore-code-palette-store", "data"),
        State("active-session-store", "data"),
        State("memo-presentation-location", "search"),
        prevent_initial_call=True,
    )
    def persist_explore_code_palette(
        palette: list[int | None] | None,
        session_id: int,
        presentation_search: str | None,
    ) -> Any:
        presentation_values = parse_qs((presentation_search or "").lstrip("?"))
        if presentation_values.get("presentation", [None])[0] == "memo":
            return no_update
        project.patch_session_state(
            int(session_id),
            {
                "explore_code_palette": _normalize_explore_code_palette(
                    palette, {int(code["code_id"]) for code in project.codes()}
                )
            },
        )
        return "saved"

    @app.callback(
        Output("session-state-status", "children"),
        Input("active-session-store", "data"),
        Input("geometry-dropdown", "value"),
        Input("view-dropdown", "value"),
        Input("focal-unit-store", "data"),
        Input("navigation-history-store", "data"),
        Input("literal-query", "value"),
        Input("literal-options", "value"),
        Input("semantic-query", "value"),
        Input("semantic-range", "data"),
        Input("metadata-filters-store", "data"),
        Input("metadata-filter-operator", "value"),
        Input("explore-uncoded-only", "value"),
        Input("context-level", "value"),
        Input("context-window", "value"),
        Input("page-size", "value"),
        Input("page-number", "value"),
        Input("navigation-new-only", "value"),
        Input("workspace-tabs", "value"),
        Input("focus-code-dropdown", "value"),
        Input("focus-active-classifier-dropdown", "value"),
        Input("focus-committee-dropdown", "value"),
        Input("focus-train-scope", "value"),
        Input("focus-auto-retrain", "value"),
        Input("focus-tune-hyperparameters", "value"),
        Input("focus-context-level", "value"),
        Input("focus-context-window", "value"),
        State("explore-code-palette-store", "data"),
        State("memo-presentation-location", "search"),
    )
    def persist_session_state(
        session_id: int,
        geometry_id: int | None,
        view_id: int | None,
        focal_unit_id: int | None,
        navigation_history: dict[str, Any] | None,
        literal_query: str | None,
        literal_options: list[str] | None,
        semantic_query: str | None,
        semantic_range: list[float] | None,
        metadata_filters: list[dict[str, Any]] | None,
        metadata_filter_operator: str | None,
        explore_uncoded_values: list[str] | None,
        context_level: int,
        context_window: int,
        page_size: Any,
        page_number: Any,
        navigation_new_only: list[str] | None,
        workspace: str,
        focus_code_id: int | None,
        focus_classifier_spec_id: int | None,
        focus_committee_id: int | None,
        focus_train_scope_values: list[str] | None,
        focus_auto_retrain_values: list[str] | None,
        focus_tune_hyperparameter_values: list[str] | None,
        focus_context_level: int,
        focus_context_window: int,
        explore_code_palette: list[int | None] | None,
        presentation_search: str | None,
    ) -> str:
        presentation_values = parse_qs((presentation_search or "").lstrip("?"))
        if presentation_values.get("presentation", [None])[0] == "memo":
            return "Presentation mode leaves Explore session state unchanged."
        options = set(literal_options or [])
        project.update_session_state(
            int(session_id),
            {
                "geometry_id": geometry_id,
                "view_id": view_id,
                "focal_unit_id": focal_unit_id,
                "navigation_history": _normalize_navigation_history(
                    navigation_history,
                    focal_unit_id=focal_unit_id,
                ),
                "literal_query": literal_query or "",
                "case_sensitive": "case" in options,
                "regex": "regex" in options,
                "semantic_query": semantic_query or "",
                "semantic_range": semantic_range,
                "metadata_filters": _normalize_metadata_filters(
                    metadata_filters or [], metadata_profiles
                ),
                "metadata_filter_operator": _normalize_boolean_operator(
                    metadata_filter_operator
                ),
                "explore_uncoded_only": (
                    "uncoded_only" in set(explore_uncoded_values or [])
                ),
                "explore_code_palette": _normalize_explore_code_palette(
                    explore_code_palette,
                    {int(code["code_id"]) for code in project.codes()},
                ),
                "context_level": int(context_level),
                "context_window": max(1, int(context_window or 1)),
                "page_size": _coerce_positive_int(page_size, default=5000),
                "page_number": _coerce_positive_int(page_number, default=1),
                "navigation_new_only": "new_only" in (navigation_new_only or []),
                "workspace": workspace,
                "focus_code_id": focus_code_id,
                "focus_classifier_spec_id": focus_classifier_spec_id,
                "focus_committee_id": focus_committee_id,
                "focus_train_all": "all" in set(focus_train_scope_values or []),
                "focus_auto_retrain": "auto" in set(focus_auto_retrain_values or []),
                "focus_tune_hyperparameters": (
                    "tune" in set(focus_tune_hyperparameter_values or [])
                ),
                "focus_context_level": int(focus_context_level),
                "focus_context_window": max(1, int(focus_context_window or 1)),
            },
        )
        return "saved"

    return app


def _explore_layout(
    *,
    dcc: Any,
    html: Any,
    geometries: list[dict[str, Any]],
    views: list[dict[str, Any]],
    hierarchy: list[dict[str, Any]],
    initial_state: dict[str, Any],
    initial_geometry_id: int | None,
    initial_view_id: int | None,
    initial_semantic_scores: dict[str, list[float]],
    geometry_by_id: dict[int, dict[str, Any]],
    total_units: int,
    sessions: list[dict[str, Any]],
    initial_session_id: int,
    initial_seen_count: int,
    metadata_profiles: dict[str, dict[str, Any]],
    allow_create_view: bool,
) -> Any:
    literal_options = []
    if initial_state.get("case_sensitive"):
        literal_options.append("case")
    if initial_state.get("regex"):
        literal_options.append("regex")
    geometry_name = (
        str(geometry_by_id[initial_geometry_id]["name"])
        if initial_geometry_id is not None
        else ""
    )
    initial_scores = initial_semantic_scores.get(geometry_name)
    if initial_scores:
        semantic_min = float(np.nanmin(initial_scores))
        semantic_max = float(np.nanmax(initial_scores))
        semantic_value = initial_state.get("semantic_range") or [semantic_min, semantic_max]
        semantic_style = {"display": "block"}
    else:
        semantic_min, semantic_max = 0.0, 1.0
        semantic_value = [0.0, 1.0]
        semantic_style = {"display": "none"}
    initial_page_size = max(1, int(initial_state.get("page_size", 5000)))
    initial_page_count = max(1, (total_units + initial_page_size - 1) // initial_page_size)
    initial_page_number = min(
        initial_page_count, max(1, int(initial_state.get("page_number", 1)))
    )
    new_only = ["new_only"] if initial_state.get("navigation_new_only", True) else []
    uncoded_only = (
        ["uncoded_only"] if initial_state.get("explore_uncoded_only", False) else []
    )

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Session", className="control-block-title"),
                            dcc.Dropdown(
                                id="session-dropdown",
                                options=_session_options(sessions),
                                value=initial_session_id,
                                clearable=False,
                                searchable=False,
                            ),
                            html.Div(
                                [
                                    dcc.Input(
                                        id="new-session-title",
                                        type="text",
                                        placeholder="New session name",
                                        className="compact-text-input",
                                    ),
                                    html.Button(
                                        "New session", id="new-session-button", n_clicks=0
                                    ),
                                ],
                                className="session-create-row",
                            ),
                            html.Div(
                                f"Visited {int(initial_seen_count):,} of {int(total_units):,}",
                                id="session-progress",
                                className="session-progress",
                            ),
                        ],
                        className="control-block session-control-block",
                    ),
                    html.Div(
                        [
                            html.H3("Geometry and View", className="control-block-title"),
                            html.Label("Geometry"),
                            dcc.Dropdown(
                                id="geometry-dropdown",
                                options=[
                                    {
                                        "label": record["name"],
                                        "value": int(record["geometry_id"]),
                                    }
                                    for record in geometries
                                ],
                                value=initial_geometry_id,
                                clearable=False,
                            ),
                            html.Label("2D view", className="stacked-control-label"),
                            dcc.Dropdown(
                                id="view-dropdown",
                                options=[
                                    {
                                        "label": view["name"],
                                        "value": int(view["view_id"]),
                                    }
                                    for view in views
                                ],
                                value=initial_view_id,
                                clearable=False,
                                searchable=False,
                            ),
                            html.Button(
                                "Create 2D view",
                                id="projection-open-button",
                                n_clicks=0,
                                className="secondary-action",
                                disabled=not allow_create_view,
                                title=(
                                    None
                                    if allow_create_view
                                    else "Create views in the external provider and register them with GeCo."
                                ),
                            ),
                        ],
                        className="control-block geometry-view-control-block",
                    ),
                    html.Div(
                        [
                            html.H3("Search", className="control-block-title"),
                            html.Div(
                                [
                                    html.Label("Literal search"),
                                    dcc.Input(
                                        id="literal-query",
                                        type="text",
                                        value=initial_state.get("literal_query", ""),
                                        placeholder="Substring or regular expression",
                                        debounce=True,
                                    ),
                                    dcc.Checklist(
                                        id="literal-options",
                                        options=[
                                            {"label": "Case sensitive", "value": "case"},
                                            {"label": "Regex", "value": "regex"},
                                        ],
                                        value=literal_options,
                                        inline=True,
                                    ),
                                    html.Div(
                                        id="literal-search-status",
                                        className="control-status",
                                    ),
                                ],
                                className="search-section",
                            ),
                            html.Div(
                                [
                                    html.Label("Semantic search"),
                                    html.Div(
                                        [
                                            dcc.Input(
                                                id="semantic-query",
                                                type="text",
                                                value=initial_state.get(
                                                    "semantic_query", ""
                                                ),
                                                placeholder="Describe what you want to find",
                                            ),
                                            html.Button(
                                                "Search",
                                                id="semantic-search-button",
                                                n_clicks=0,
                                            ),
                                        ],
                                        className="inline-controls semantic-query-row",
                                    ),
                                    dcc.Loading(
                                        id="semantic-search-loading",
                                        type="circle",
                                        children=html.Div(
                                            [
                                                html.Div(
                                                    id="semantic-search-status",
                                                    className="control-status",
                                                ),
                                                html.Div(
                                                    dcc.RangeSlider(
                                                        id="semantic-range-slider",
                                                        min=semantic_min,
                                                        max=semantic_max,
                                                        step=max(
                                                            (semantic_max - semantic_min)
                                                            / 1000,
                                                            1e-9,
                                                        ),
                                                        value=semantic_value,
                                                        marks=(
                                                            {
                                                                semantic_min: f"{semantic_min:.3f}",
                                                                semantic_max: f"{semantic_max:.3f}",
                                                            }
                                                            if semantic_style.get("display")
                                                            != "none"
                                                            else {}
                                                        ),
                                                        allowCross=False,
                                                        tooltip={
                                                            "placement": "top",
                                                            "always_visible": False,
                                                            "transform": "formatSimilarity",
                                                            "style": {
                                                                "minWidth": "58px",
                                                                "textAlign": "center",
                                                                "fontVariantNumeric": "tabular-nums",
                                                            },
                                                        },
                                                        allow_direct_input=False,
                                                    ),
                                                    id="semantic-range-wrap",
                                                    className="semantic-range-wrap",
                                                    style=semantic_style,
                                                ),
                                                html.Button(
                                                    "Clear semantic search",
                                                    id="semantic-clear-button",
                                                    n_clicks=0,
                                                    className=(
                                                        "secondary-action "
                                                        "semantic-clear-action"
                                                    ),
                                                    style=(
                                                        {"display": "inline-block"}
                                                        if initial_semantic_scores
                                                        else {"display": "none"}
                                                    ),
                                                ),
                                            ]
                                        ),
                                    ),
                                ],
                                className="search-section semantic-search-section",
                            ),
                        ],
                        className="control-block combined-search-block",
                    ),
                    html.Div(
                        [
                            html.H3("Filters", className="control-block-title"),
                            html.Div(
                                id="active-metadata-filter-chips",
                                className="metadata-filter-chip-box",
                            ),
                            dcc.RadioItems(
                                id="metadata-filter-operator",
                                options=[
                                    {"label": "AND", "value": "and"},
                                    {"label": "OR", "value": "or"},
                                    {"label": "XOR", "value": "xor"},
                                ],
                                value=str(
                                    initial_state.get("metadata_filter_operator", "and")
                                ),
                                inline=True,
                                className="metadata-filter-operator",
                            ),
                            dcc.Checklist(
                                id="explore-uncoded-only",
                                options=[
                                    {
                                        "label": "Show only uncoded points",
                                        "value": "uncoded_only",
                                    }
                                ],
                                value=uncoded_only,
                                className="explore-uncoded-filter",
                            ),
                            html.Div(
                                [
                                    html.Button(
                                        "Add",
                                        id="metadata-filter-add-button",
                                        n_clicks=0,
                                        disabled=not bool(metadata_profiles),
                                    ),
                                    html.Button(
                                        "Clear",
                                        id="metadata-filter-clear-button",
                                        n_clicks=0,
                                    ),
                                    html.Button(
                                        "Save",
                                        id="metadata-filter-save-open-button",
                                        n_clicks=0,
                                    ),
                                    html.Button(
                                        "Load",
                                        id="metadata-filter-load-open-button",
                                        n_clicks=0,
                                    ),
                                ],
                                className="metadata-filter-actions",
                            ),
                            html.Div(id="metadata-filter-status", className="control-status"),
                        ],
                        className="control-block metadata-filter-block",
                    ),
                ],
                className="explore-controls",
            ),
            html.Div(
                [
                    html.Button("← Back", id="nav-back", n_clicks=0, disabled=True),
                    html.Button("Forward →", id="nav-forward", n_clicks=0, disabled=True),
                    html.Span(className="navigation-divider"),
                    html.Button("Nearest", id="nav-nearest", n_clicks=0),
                    html.Button("Farthest", id="nav-farthest", n_clicks=0),
                    html.Button(
                        "Most similar",
                        id="nav-most-similar",
                        n_clicks=0,
                        className="semantic-navigation-button",
                        style=(
                            {"display": "inline-block"}
                            if initial_semantic_scores
                            else {"display": "none"}
                        ),
                    ),
                    html.Button(
                        "Least similar",
                        id="nav-least-similar",
                        n_clicks=0,
                        className="semantic-navigation-button",
                        style=(
                            {"display": "inline-block"}
                            if initial_semantic_scores
                            else {"display": "none"}
                        ),
                    ),
                    html.Button("Random", id="nav-random", n_clicks=0),
                    dcc.Checklist(
                        id="navigation-new-only",
                        options=[{"label": "New points only", "value": "new_only"}],
                        value=new_only,
                        inline=True,
                        className="navigation-repeat-control",
                    ),
                    html.Div(id="navigation-status", className="sr-only"),
                ],
                className="navigation-bar",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Graph(
                                id="geometry-map",
                                figure=_empty_figure("Select or compute a geometry view."),
                                config={
                                    "displaylogo": False,
                                    "scrollZoom": True,
                                    "doubleClick": "reset",
                                    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                                },
                                clear_on_unhover=True,
                                className="geometry-map",
                            ),
                            html.Div(
                                [
                                    html.Span(className="visited-outline-swatch"),
                                    html.Span("Visited in this session"),
                                    html.Span(className="focal-outline-swatch"),
                                    html.Span("Current focal unit"),
                                ],
                                className="map-encoding-key",
                            ),
                            html.Div(
                                [
                                    html.Label("Points per page", htmlFor="page-size"),
                                    dcc.Input(
                                        id="page-size",
                                        type="text",
                                        inputMode="numeric",
                                        debounce=True,
                                        value=str(initial_page_size),
                                        className="page-size-input",
                                    ),
                                    html.Button("Previous", id="page-previous", n_clicks=0),
                                    html.Label("Page", htmlFor="page-number"),
                                    dcc.Input(
                                        id="page-number",
                                        type="number",
                                        min=1,
                                        step=1,
                                        debounce=True,
                                        value=initial_page_number,
                                        max=initial_page_count,
                                        className="page-number-input",
                                    ),
                                    html.Button("Next", id="page-next", n_clicks=0),
                                    html.Div(id="page-status", className="page-status"),
                                ],
                                className="pagination-bar",
                            ),
                        ],
                        className="map-pane",
                    ),
                    html.Aside(
                        [
                            html.Div(id="focal-key", className="focal-key inspection-key"),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label("Context level"),
                                            dcc.Dropdown(
                                                id="context-level",
                                                options=[
                                                    {
                                                        "label": level["column_name"],
                                                        "value": int(level["level_index"]),
                                                    }
                                                    for level in hierarchy
                                                ],
                                                value=int(
                                                    initial_state.get(
                                                        "context_level",
                                                        hierarchy[-1]["level_index"],
                                                    )
                                                ),
                                                clearable=False,
                                                searchable=False,
                                            ),
                                        ],
                                        className="context-control context-level-control",
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Window"),
                                            html.Div(
                                                [
                                                    html.Button(
                                                        "−",
                                                        id="context-window-minus",
                                                        n_clicks=0,
                                                    ),
                                                    dcc.Input(
                                                        id="context-window",
                                                        type="text",
                                                        inputMode="numeric",
                                                        value=str(
                                                            max(
                                                                1,
                                                                int(
                                                                    initial_state.get(
                                                                        "context_window", 1
                                                                    )
                                                                ),
                                                            )
                                                        ),
                                                    ),
                                                    html.Button(
                                                        "+",
                                                        id="context-window-plus",
                                                        n_clicks=0,
                                                    ),
                                                ],
                                                className="window-stepper",
                                            ),
                                        ],
                                        className="context-control context-window-control",
                                    ),
                                ],
                                className="context-controls",
                            ),
                            html.Div(
                                [
                                    html.Div(id="context-content", className="context-content"),
                                    html.Details(
                                        [
                                            html.Summary("Metadata"),
                                            html.Div(
                                                id="metadata-content",
                                                className="metadata-content",
                                            ),
                                        ],
                                        open=False,
                                    ),
                                ],
                                className="inspection-scroll",
                            ),
                        ],
                        className="reading-pane inspection-pane",
                    ),
                ],
                className="explore-main",
            ),
            html.Div(
                [
                    html.Section(
                        [
                            html.Div(
                                [
                                    html.H3("Memos"),
                                    html.Div(
                                        [
                                            dcc.Dropdown(
                                                id="memo-dropdown",
                                                options=[],
                                                value=None,
                                                placeholder="Select a memo",
                                                clearable=True,
                                            ),
                                            html.Button(
                                                "New memo", id="memo-new-button", n_clicks=0
                                            ),
                                        ],
                                        className="memo-toolbar",
                                    ),
                                ],
                                className="panel-heading",
                            ),
                            dcc.Input(
                                id="memo-title",
                                type="text",
                                placeholder="Memo title",
                                className="text-input",
                            ),
                            dcc.Textarea(
                                id="memo-body",
                                placeholder=(
                                    "Write in Markdown. Use #hashtags or insert a "
                                    "reference to the focal unit."
                                ),
                                className="memo-editor",
                            ),
                            html.Div(
                                [
                                    html.Button(
                                        "Insert reference to focal unit",
                                        id="memo-insert-reference-button",
                                        n_clicks=0,
                                    ),
                                    html.Button(
                                        "Save memo",
                                        id="memo-save-button",
                                        n_clicks=0,
                                        className="primary-action",
                                    ),
                                    html.Div(id="memo-status", className="control-status"),
                                ],
                                className="memo-actions",
                            ),
                            html.Div(
                                [
                                    html.Div("Preview", className="memo-preview-label"),
                                    html.Div(id="memo-preview", className="memo-preview"),
                                ]
                            ),
                        ],
                        className="analysis-panel memo-panel",
                    ),
                    html.Section(
                        [
                            html.Div(
                                [
                                    html.H3("Coding"),
                                    html.Button(
                                        "New Code",
                                        id="explore-new-code-button",
                                        n_clicks=0,
                                    ),
                                ],
                                className="coding-panel-heading",
                            ),
                            html.Div(
                                id="explore-code-palette",
                                className="explore-code-palette",
                            ),
                            html.Button(
                                "+ Add code",
                                id="explore-add-code-row",
                                n_clicks=0,
                                className="explore-add-code-row",
                            ),
                            html.Div(id="explore-code-status", className="sr-only"),
                        ],
                        className="analysis-panel coding-panel",
                    ),
                ],
                className="explore-analysis-row",
            ),
            _projection_modal(dcc=dcc, html=html),
        ],
        className="explore-workspace",
    )


def _memo_center_layout(*, dcc: Any, html: Any, dash_table: Any) -> Any:
    """Return the versioned project-wide memo workspace."""
    return html.Div(
        [
            html.Aside(
                [
                    html.H2("Memo Center"),
                    html.Label("Search memos"),
                    dcc.Input(
                        id="memo-center-query", type="text",
                        placeholder="Search memo titles or contents", debounce=True,
                        className="text-input",
                    ),
                    dcc.RadioItems(
                        id="memo-center-scope",
                        options=[
                            {"label": "Full memo", "value": "full"},
                            {"label": "Titles only", "value": "title"},
                        ],
                        value="full", inline=True, className="memo-center-scope",
                    ),
                    html.Label("Hashtags", className="memo-center-filter-label"),
                    dcc.Dropdown(
                        id="memo-center-tags", options=[], value=[], multi=True,
                        placeholder="Filter by hashtag",
                    ),
                    dcc.RadioItems(
                        id="memo-center-tag-operator",
                        options=[
                            {"label": "AND", "value": "and"},
                            {"label": "OR", "value": "or"},
                            {"label": "XOR", "value": "xor"},
                        ],
                        value="and", inline=True, className="memo-center-operator",
                    ),
                    html.Div(id="memo-center-status", className="control-status"),
                    dash_table.DataTable(
                        id="memo-center-table",
                        columns=[
                            {"name": "Title", "id": "title"},
                            {"name": "Hashtags", "id": "hashtags"},
                            {"name": "Updated", "id": "updated_at"},
                        ],
                        data=[], row_selectable="single", selected_rows=[], page_size=20,
                        style_cell={"whiteSpace": "normal", "height": "auto", "textAlign": "left"},
                        style_table={"overflowX": "auto"},
                    ),
                ],
                className="memo-center-sidebar",
            ),
            html.Main(
                [
                    html.Div(
                        [
                            html.H2(id="memo-center-title", children="Select a memo"),
                            html.Div(
                                [
                                    html.A(
                                        "Presentation mode",
                                        id="memo-presentation-link",
                                        href=None,
                                        target="_blank",
                                        className="memo-presentation-link disabled",
                                        title="Open this memo with a minimal interactive map",
                                    ),
                                    html.Button("Edit", id="memo-center-edit-button", n_clicks=0, disabled=True),
                                ],
                                className="memo-center-heading-actions",
                            ),
                        ],
                        className="center-heading-row",
                    ),
                    html.Div(
                        [
                            html.Button("←", id="memo-version-previous", n_clicks=0, disabled=True),
                            html.Span("No version selected", id="memo-version-label"),
                            html.Button("→", id="memo-version-next", n_clicks=0, disabled=True),
                        ],
                        className="version-navigation",
                    ),
                    html.Div(
                        id="memo-center-preview",
                        className="memo-center-preview memo-preview",
                    ),
                    html.Div(
                        [
                            html.Label("Title"),
                            dcc.Input(id="memo-center-edit-title", type="text", className="text-input"),
                            html.Label("Markdown", className="memo-center-editor-label"),
                            dcc.Textarea(id="memo-center-edit-body", className="memo-editor memo-center-editor"),
                            html.Div(
                                [
                                    html.Button("Cancel", id="memo-center-cancel-button", n_clicks=0),
                                    html.Button("Save new version", id="memo-center-save-button", n_clicks=0, className="primary-action"),
                                ],
                                className="memo-actions",
                            ),
                            html.Div(id="memo-center-edit-status", className="control-status"),
                            html.Div(id="memo-center-edit-preview", className="memo-preview"),
                        ],
                        id="memo-center-editor-wrap",
                        className="memo-center-editor-wrap",
                        style={"display": "none"},
                    ),
                ],
                className="memo-center-main",
            ),
        ],
        className="memo-center-workspace",
    )


def _memo_presentation_layout(
    *,
    dcc: Any,
    html: Any,
    views: list[dict[str, Any]],
    geometry_by_id: dict[int, dict[str, Any]],
    initial_view_id: int | None,
    total_units: int,
) -> Any:
    """Return the minimal memo-and-map presentation surface."""
    public_views = [
        view for view in views if int(view["geometry_id"]) in geometry_by_id
    ]
    view_options = [
        {
            "label": (
                f"{geometry_by_id[int(view['geometry_id'])]['name']} — {view['name']}"
            ),
            "value": int(view["view_id"]),
        }
        for view in public_views
    ]
    selected_view = (
        int(initial_view_id)
        if initial_view_id is not None
        and any(int(view["view_id"]) == int(initial_view_id) for view in public_views)
        else (int(public_views[0]["view_id"]) if public_views else None)
    )
    initial_page_count = deterministic_page(total_units, page_size=5000).page_count
    return html.Div(
        [
            html.Header(
                [
                    html.Div(
                        [
                            html.Span("🦎", className="memo-presentation-gecko", **{"aria-hidden": "true"}),
                            html.Div(
                                [
                                    html.H1(id="memo-presentation-title", children="Memo presentation"),
                                    html.Div(id="memo-presentation-version", className="memo-presentation-version"),
                                ]
                            ),
                        ],
                        className="memo-presentation-brand",
                    ),
                    html.Div(
                        "Presentation mode · exploratory evidence view",
                        className="memo-presentation-mode-label",
                    ),
                ],
                className="memo-presentation-header",
            ),
            html.Div(
                [
                    html.Section(
                        html.Div(
                            id="memo-presentation-body",
                            className="memo-preview memo-presentation-body",
                        ),
                        className="memo-presentation-memo-pane",
                    ),
                    html.Section(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label("2D view", htmlFor="memo-presentation-view"),
                                            dcc.Dropdown(
                                                id="memo-presentation-view",
                                                options=view_options,
                                                value=selected_view,
                                                clearable=False,
                                                searchable=False,
                                            ),
                                        ],
                                        className="memo-presentation-control",
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Saved filter", htmlFor="memo-presentation-saved-filter"),
                                            dcc.Dropdown(
                                                id="memo-presentation-saved-filter",
                                                options=[{"label": "All observations", "value": "__all__"}],
                                                value="__all__",
                                                clearable=False,
                                            ),
                                        ],
                                        className="memo-presentation-control",
                                    ),
                                ],
                                className="memo-presentation-controls",
                            ),
                            html.Div(
                                [
                                    dcc.Graph(
                                        id="memo-presentation-map",
                                        figure=_empty_figure("Choose a memo and 2D view."),
                                        config={
                                            "displaylogo": False,
                                            "scrollZoom": True,
                                            "doubleClick": "reset",
                                            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                                        },
                                        clear_on_unhover=True,
                                        className="memo-presentation-map",
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Points per page", htmlFor="memo-presentation-page-size"),
                                            dcc.Input(
                                                id="memo-presentation-page-size",
                                                type="text",
                                                inputMode="numeric",
                                                debounce=True,
                                                value="5000",
                                                className="page-size-input",
                                            ),
                                            html.Button("Previous", id="memo-presentation-page-previous", n_clicks=0),
                                            html.Label("Page", htmlFor="memo-presentation-page-number"),
                                            dcc.Input(
                                                id="memo-presentation-page-number",
                                                type="number",
                                                min=1,
                                                step=1,
                                                debounce=True,
                                                value=1,
                                                max=initial_page_count,
                                                className="page-number-input",
                                            ),
                                            html.Button("Next", id="memo-presentation-page-next", n_clicks=0),
                                            html.Div(id="memo-presentation-page-status", className="page-status"),
                                        ],
                                        className="pagination-bar memo-presentation-pagination",
                                    ),
                                ],
                                className="memo-presentation-map-wrap",
                            ),
                            html.Aside(
                                [
                                    html.Div(
                                        id="memo-presentation-focal-key",
                                        className="focal-key inspection-key",
                                        children="No focal observation",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                id="memo-presentation-text",
                                                className="context-content memo-presentation-text",
                                                children="Select a linked observation in the memo or click a point on the map.",
                                            ),
                                            html.Details(
                                                [
                                                    html.Summary("Metadata"),
                                                    html.Div(
                                                        id="memo-presentation-metadata",
                                                        className="metadata-content",
                                                    ),
                                                ],
                                                open=False,
                                            ),
                                        ],
                                        className="inspection-scroll",
                                    ),
                                ],
                                className="reading-pane inspection-pane memo-presentation-reader",
                            ),
                        ],
                        className="memo-presentation-evidence-pane",
                    ),
                ],
                className="memo-presentation-grid",
            ),
        ],
        id="memo-presentation-shell",
        className="memo-presentation-shell",
    )



def _code_center_layout(
    *, dcc: Any, html: Any, dash_table: Any, geometries: list[dict[str, Any]],
    views: list[dict[str, Any]], initial_geometry_id: int | None, initial_view_id: int | None,
    total_units: int,
) -> Any:
    """Return the versioned Code Center with list and geometric views."""
    initial_page_count = deterministic_page(total_units, page_size=5000).page_count
    return html.Div(
        [
            html.Aside(
                [
                    html.H2("Code Center"),
                    html.Label("Search codes"),
                    dcc.Input(id="code-center-query", type="text", debounce=True, className="text-input", placeholder="Search labels or descriptions"),
                    dcc.RadioItems(
                        id="code-center-scope",
                        options=[{"label": "Label and description", "value": "full"}, {"label": "Labels only", "value": "label"}],
                        value="full", inline=True, className="memo-center-scope",
                    ),
                    html.Label("Hashtags", className="memo-center-filter-label"),
                    dcc.Dropdown(id="code-center-tags", options=[], value=[], multi=True, placeholder="Filter by hashtag"),
                    html.Div(id="code-center-status", className="control-status"),
                    dash_table.DataTable(
                        id="code-center-table",
                        columns=[{"name": "Code", "id": "name"}, {"name": "Hashtags", "id": "hashtags"}],
                        data=[], row_selectable="single", selected_rows=[], page_size=20,
                        style_cell={"whiteSpace": "normal", "height": "auto", "textAlign": "left"},
                        style_table={"overflowX": "auto"},
                    ),
                ],
                className="code-center-sidebar",
            ),
            html.Main(
                [
                    html.Div(
                        [
                            html.H2(id="code-center-title", children="Select a code"),
                            html.Div(
                                [
                                    html.Button(
                                        "Teaching Examples",
                                        id="code-center-teaching-button",
                                        n_clicks=0,
                                        disabled=True,
                                    ),
                                    html.Button(
                                        "Edit",
                                        id="code-center-edit-button",
                                        n_clicks=0,
                                        disabled=True,
                                    ),
                                ],
                                className="center-heading-actions",
                            ),
                        ],
                        className="center-heading-row",
                    ),
                    html.Div(
                        [
                            html.Button("←", id="code-version-previous", n_clicks=0, disabled=True),
                            html.Span("No version selected", id="code-version-label"),
                            html.Button("→", id="code-version-next", n_clicks=0, disabled=True),
                        ],
                        className="version-navigation",
                    ),
                    html.Div(id="code-center-description", className="memo-preview code-description-preview"),
                    html.Div(
                        [
                            html.Label("Code label"),
                            dcc.Input(id="code-center-edit-name", type="text", className="text-input code-name-input"),
                            html.Label("Description", className="memo-center-editor-label"),
                            dcc.Textarea(id="code-center-edit-description", className="memo-editor code-center-editor"),
                            html.Div(
                                [
                                    html.Button("Cancel", id="code-center-cancel-button", n_clicks=0),
                                    html.Button("Save new version", id="code-center-save-button", n_clicks=0, className="primary-action"),
                                ], className="memo-actions",
                            ),
                            html.Div(id="code-center-edit-status", className="control-status"),
                            html.Div(id="code-center-edit-preview", className="memo-preview"),
                        ],
                        id="code-center-editor-wrap", style={"display": "none"},
                    ),
                    dcc.Tabs(
                        id="code-center-view-tabs", value="list", className="code-center-content-tabs",
                        children=[
                            dcc.Tab(
                                label="Positive units", value="list",
                                children=dash_table.DataTable(
                                    id="code-center-units-table",
                                    columns=[
                                        {"name": "Type", "id": "kind"},
                                        {"name": "Key", "id": "key"},
                                        {"name": "Text", "id": "text"},
                                    ],
                                    data=[], page_size=25,
                                    style_cell={"whiteSpace": "normal", "height": "auto", "textAlign": "left"},
                                    style_table={"overflowX": "auto"},
                                ),
                            ),
                            dcc.Tab(
                                label="Geometric view", value="geometry",
                                children=html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Div([html.Label("Geometry"), dcc.Dropdown(
                                                    id="code-center-geometry", options=[{"label": g["name"], "value": int(g["geometry_id"])} for g in geometries],
                                                    value=initial_geometry_id, clearable=False,
                                                )]),
                                                html.Div([html.Label("2D view"), dcc.Dropdown(
                                                    id="code-center-view", options=[{"label": v["name"], "value": int(v["view_id"])} for v in views],
                                                    value=initial_view_id, clearable=False,
                                                )]),
                                                html.Div(
                                                    dcc.Checklist(
                                                        id="code-center-include-positives",
                                                        options=[{"label": "Always show all positive units", "value": "include"}],
                                                        value=["include"],
                                                        className="code-center-positive-toggle",
                                                    )
                                                ),
                                            ], className="code-center-geometry-controls",
                                        ),
                                        dcc.Graph(
                                            id="code-center-map",
                                            figure=_code_center_placeholder_figure(),
                                            config={
                                                "displaylogo": False,
                                                "scrollZoom": True,
                                                "doubleClick": "reset",
                                                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                                            },
                                            className="code-center-map",
                                        ),
                                        html.Div(
                                            [
                                                html.Label("Points per page", htmlFor="code-center-page-size"),
                                                dcc.Input(
                                                    id="code-center-page-size",
                                                    type="text",
                                                    inputMode="numeric",
                                                    debounce=True,
                                                    value="5000",
                                                    className="page-size-input",
                                                ),
                                                html.Button("Previous", id="code-center-page-previous", n_clicks=0),
                                                html.Label("Page", htmlFor="code-center-page-number"),
                                                dcc.Input(
                                                    id="code-center-page-number",
                                                    type="number",
                                                    min=1,
                                                    step=1,
                                                    debounce=True,
                                                    value=1,
                                                    max=initial_page_count,
                                                    className="page-number-input",
                                                ),
                                                html.Button("Next", id="code-center-page-next", n_clicks=0),
                                                html.Div(id="code-center-page-status", className="page-status"),
                                            ],
                                            className="pagination-bar code-center-pagination",
                                        ),
                                    ], className="code-center-geometry-pane",
                                ),
                            ),
                        ],
                    ),
                    html.Section(
                        [
                            html.H3("Selected observation"),
                            html.Div(
                                id="code-center-unit-preview",
                                className="memo-preview code-center-unit-preview",
                            ),
                            html.Div(
                                [
                                    html.Button(
                                        "Present",
                                        id="code-center-assignment-positive",
                                        n_clicks=0,
                                        className="judgment-button positive-button",
                                        disabled=True,
                                    ),
                                    html.Button(
                                        "Absent",
                                        id="code-center-assignment-negative",
                                        n_clicks=0,
                                        className="judgment-button negative-button",
                                        disabled=True,
                                    ),
                                    html.Button(
                                        "Unsure",
                                        id="code-center-assignment-unsure",
                                        n_clicks=0,
                                        className="judgment-button unsure-button",
                                        disabled=True,
                                    ),
                                ],
                                className="code-center-assignment-buttons",
                            ),
                            html.Div(
                                id="code-center-assignment-status",
                                className="control-status",
                            ),
                        ],
                        className="code-center-assignment-panel",
                    ),
                ],
                className="code-center-main",
            ),
        ],
        className="code-center-workspace",
    )

def _teaching_example_modal(*, dcc: Any, html: Any, dash_table: Any) -> Any:
    """Return the teaching-example manager for the selected code."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Teaching Examples"),
                            html.Button(
                                "Close", id="teaching-example-close", n_clicks=0
                            ),
                        ],
                        className="modal-heading-row",
                    ),
                    html.P(
                        "Teaching examples are researcher-authored positive or negative "
                        "examples. Active examples receive the same classifier weight as "
                        "coded corpus observations.",
                        className="focus-help",
                    ),
                    html.Label("Example text"),
                    dcc.Textarea(
                        id="teaching-example-text",
                        className="memo-editor teaching-example-text",
                        placeholder="Write an example that resembles the empirical material.",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Label"),
                                    dcc.RadioItems(
                                        id="teaching-example-label",
                                        options=[
                                            {"label": "Present", "value": "positive"},
                                            {"label": "Absent", "value": "negative"},
                                        ],
                                        value="positive",
                                        inline=True,
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label("Optional note"),
                                    dcc.Input(
                                        id="teaching-example-note",
                                        type="text",
                                        className="text-input",
                                    ),
                                ]
                            ),
                        ],
                        className="teaching-example-fields",
                    ),
                    html.Button(
                        "Add active example",
                        id="teaching-example-save",
                        n_clicks=0,
                        className="primary-action",
                    ),
                    dcc.Checklist(
                        id="teaching-example-assess-before-training",
                        options=[
                            {
                                "label": "Assess current classifiers before adding this example to training",
                                "value": "assess",
                            }
                        ],
                        value=[],
                    ),
                    html.Div(id="teaching-example-status", className="control-status"),
                    html.H4("Existing examples"),
                    dash_table.DataTable(
                        id="teaching-example-table",
                        columns=[
                            {"name": "Text", "id": "text"},
                            {"name": "Label", "id": "label"},
                            {"name": "Status", "id": "status"},
                            {"name": "Created", "id": "created"},
                        ],
                        data=[],
                        row_selectable="single",
                        selected_rows=[],
                        page_size=10,
                        style_cell={
                            "whiteSpace": "normal",
                            "height": "auto",
                            "textAlign": "left",
                        },
                        style_table={"overflowX": "auto"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Change status"),
                                    dcc.Dropdown(
                                        id="teaching-example-status-action",
                                        options=[
                                            {"label": "Active", "value": "active"},
                                            {"label": "Inactive", "value": "inactive"},
                                            {"label": "Delete", "value": "deleted"},
                                        ],
                                        value="inactive",
                                        clearable=False,
                                    ),
                                    html.Button(
                                        "Apply status",
                                        id="teaching-example-apply-status",
                                        n_clicks=0,
                                    ),
                                ],
                                className="teaching-example-action-group",
                            ),
                            html.Div(
                                [
                                    html.Label("Change label"),
                                    dcc.Dropdown(
                                        id="teaching-example-label-action",
                                        options=[
                                            {"label": "Present", "value": "positive"},
                                            {"label": "Absent", "value": "negative"},
                                        ],
                                        value="positive",
                                        clearable=False,
                                    ),
                                    html.Button(
                                        "Apply label",
                                        id="teaching-example-apply-label",
                                        n_clicks=0,
                                    ),
                                ],
                                className="teaching-example-action-group",
                            ),
                        ],
                        className="teaching-example-actions",
                    ),
                ],
                className="modal-card teaching-example-modal-card",
            )
        ],
        id="teaching-example-modal",
        className="modal-overlay",
        style={"display": "none"},
    )



def _classifier_spec_modal(*, dcc: Any, html: Any, geometries: list[dict[str, Any]]) -> Any:
    """Return the classifier-specification creation modal."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("New classifier"),
                            html.Button(
                                "×",
                                id="classifier-spec-close",
                                n_clicks=0,
                                className="modal-close-icon",
                                title="Close",
                                **{"aria-label": "Close"},
                            ),
                        ],
                        className="modal-heading-row",
                    ),
                    html.Label("Classifier name"),
                    dcc.Input(
                        id="classifier-spec-name",
                        type="text",
                        className="text-input",
                        placeholder="Optional; GeCo will generate a descriptive name",
                    ),
                    html.Label("Parent geometry"),
                    dcc.Dropdown(
                        id="classifier-spec-geometry",
                        options=[
                            {
                                "label": str(row["name"]),
                                "value": int(row["geometry_id"]),
                            }
                            for row in geometries
                        ],
                        value=(int(geometries[0]["geometry_id"]) if geometries else None),
                        clearable=False,
                    ),
                    html.Label("Algorithm"),
                    dcc.Dropdown(
                        id="classifier-spec-algorithm",
                        options=[
                            {
                                "label": classifier_algorithm_label(algorithm),
                                "value": algorithm,
                            }
                            for algorithm in SUPPORTED_CLASSIFIER_ALGORITHMS
                        ],
                        value="logistic_l2",
                        clearable=False,
                    ),
                    html.Small(
                        "New classifiers start with family defaults. In Develop, optionally "
                        "enable Tune hyperparameters with cross-validation before Train when "
                        "you want a slower hyperparameter search.",
                        className="focus-help",
                    ),
                    html.Div(id="classifier-spec-status", className="control-status"),
                    html.Button(
                        "Create classifier",
                        id="classifier-spec-save",
                        n_clicks=0,
                        className="primary-action",
                    ),
                ],
                className="modal-card classifier-modal-card",
            )
        ],
        id="classifier-spec-modal",
        className="modal-overlay",
        style={"display": "none"},
    )


def _classifier_manager_modal(*, dcc: Any, html: Any, dash_table: Any) -> Any:
    """Return the active-classifier manager."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Manage classifiers"),
                            html.Button(
                                "×",
                                id="classifier-manager-close",
                                n_clicks=0,
                                className="modal-close-icon",
                                title="Close",
                                **{"aria-label": "Close"},
                            ),
                        ],
                        className="modal-heading-row",
                    ),
                    html.P(
                        "Rename a classifier without changing its fitted state, or delete it from active use. Classifier and committee names share one project-wide active predictor namespace. Historical fitted models and drafts retain the name recorded when they were created.",
                        className="focus-help",
                    ),
                    dash_table.DataTable(
                        id="classifier-manager-table",
                        columns=[
                            {"name": "Name", "id": "name"},
                            {"name": "Geometry", "id": "geometry"},
                            {"name": "Algorithm", "id": "algorithm"},
                            {"name": "Selected settings", "id": "settings"},
                        ],
                        data=[],
                        row_selectable="single",
                        selected_rows=[],
                        page_size=12,
                        style_cell={
                            "whiteSpace": "normal",
                            "height": "auto",
                            "textAlign": "left",
                        },
                        style_table={"overflowX": "auto"},
                    ),
                    html.Div(
                        [
                            dcc.Input(
                                id="classifier-manager-rename-name",
                                type="text",
                                className="text-input",
                                placeholder="Classifier name",
                            ),
                            html.Button(
                                "Rename selected classifier",
                                id="classifier-manager-rename",
                                n_clicks=0,
                            ),
                        ],
                        className="classifier-manager-rename-row",
                    ),
                    html.Div(
                        [
                            html.Div(
                                id="classifier-manager-status",
                                className="control-status",
                            ),
                            html.Button(
                                "Delete selected classifier",
                                id="classifier-manager-delete",
                                n_clicks=0,
                                className="danger-action",
                            ),
                        ],
                        className="modal-actions classifier-manager-actions",
                    ),
                ],
                className="modal-card classifier-manager-modal-card",
            )
        ],
        id="classifier-manager-modal",
        className="modal-overlay",
        style={"display": "none"},
    )


def _committee_modal(*, dcc: Any, html: Any, dash_table: Any) -> Any:
    """Return the code-specific classifier-committee manager."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Classifier committees"),
                            html.Button(
                                "×",
                                id="committee-close",
                                n_clicks=0,
                                className="modal-close-icon",
                                title="Close",
                                **{"aria-label": "Close"},
                            ),
                        ],
                        className="modal-heading-row",
                    ),
                    html.P(
                        "A committee is an intentional named set of classifier specifications for the active code. Rename an existing committee without changing its members, aggregation, or learned fitted state; classifier and committee names share one project-wide active predictor namespace.",
                        className="focus-help",
                    ),
                    html.Label("Committee name"),
                    dcc.Input(id="committee-name", type="text", className="text-input"),
                    html.Label("Classifier members"),
                    dcc.Checklist(
                        id="committee-members",
                        options=[],
                        value=[],
                        className="committee-member-list",
                    ),
                    html.Label("Aggregation"),
                    dcc.Dropdown(
                        id="committee-aggregation",
                        options=[
                            {"label": "Mean probability", "value": "mean"},
                            {"label": "Minimum probability (strict AND)", "value": "minimum"},
                            {"label": "Maximum probability (permissive OR)", "value": "maximum"},
                            {"label": "Harmonic mean", "value": "harmonic_mean"},
                            {"label": "Geometric mean", "value": "geometric_mean"},
                            {
                                "label": "Logistic stacking (learned weights)",
                                "value": "logistic_stack",
                            },
                        ],
                        value="mean",
                        clearable=False,
                    ),
                    html.Button(
                        "Save new committee",
                        id="committee-save",
                        n_clicks=0,
                        className="primary-action",
                    ),
                    html.Div(id="committee-status", className="control-status"),
                    html.H4("Existing committees"),
                    dash_table.DataTable(
                        id="committee-table",
                        columns=[
                            {"name": "Name", "id": "name"},
                            {"name": "Aggregation", "id": "aggregation"},
                            {"name": "Members", "id": "members"},
                        ],
                        data=[],
                        row_selectable="single",
                        selected_rows=[],
                        page_size=10,
                        style_cell={
                            "whiteSpace": "normal",
                            "height": "auto",
                            "textAlign": "left",
                        },
                    ),
                    html.Div(
                        [
                            dcc.Input(
                                id="committee-manager-rename-name",
                                type="text",
                                className="text-input",
                                placeholder="Committee name",
                            ),
                            html.Button(
                                "Rename selected committee",
                                id="committee-rename",
                                n_clicks=0,
                            ),
                        ],
                        className="classifier-manager-rename-row",
                    ),
                    html.Button(
                        "Delete selected committee",
                        id="committee-delete",
                        n_clicks=0,
                        className="danger-action",
                    ),
                ],
                className="modal-card committee-modal-card",
            )
        ],
        id="committee-modal",
        className="modal-overlay",
        style={"display": "none"},
    )


def _testing_center_modal(*, dcc: Any, html: Any, dash_table: Any) -> Any:
    """Return the classifier Testing Center foundation."""
    del dcc
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Classifier Testing Center"),
                            html.Button("Close", id="testing-center-close", n_clicks=0),
                        ],
                        className="modal-heading-row",
                    ),
                    html.P(
                        "Diagnostic teaching-example probes are scored before they enter training. These are diagnostics, not formal validation estimates.",
                        className="focus-help",
                    ),
                    html.Div(id="testing-center-summary", className="focus-model-status"),
                    dash_table.DataTable(
                        id="testing-center-table",
                        columns=[
                            {"name": "Classifier", "id": "classifier"},
                            {"name": "Events", "id": "n"},
                            {"name": "Accuracy", "id": "accuracy"},
                            {"name": "Brier", "id": "brier"},
                        ],
                        data=[],
                        page_size=12,
                        style_cell={"textAlign": "left"},
                    ),
                    html.H4("Assessment events"),
                    dash_table.DataTable(
                        id="testing-center-events",
                        columns=[
                            {"name": "Type", "id": "event_type"},
                            {"name": "Classifier", "id": "classifier"},
                            {"name": "Human", "id": "human"},
                            {"name": "Predicted", "id": "predicted"},
                            {"name": "Probability", "id": "probability"},
                        ],
                        data=[],
                        page_size=12,
                        style_cell={"textAlign": "left"},
                    ),
                ],
                className="modal-card testing-center-modal-card",
            )
        ],
        id="testing-center-modal",
        className="modal-overlay",
        style={"display": "none"},
    )

def _code_modal(*, dcc: Any, html: Any) -> Any:
    """Return the shared code-creation modal used by Explore and Develop."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("New Code"),
                            html.Button("×", id="code-modal-cancel-button", n_clicks=0),
                        ],
                        className="modal-header",
                    ),
                    html.Div(
                        [
                            html.Label("Code label"),
                            dcc.Input(
                                id="code-modal-name",
                                type="text",
                                placeholder="Enter a concise code label",
                                className="text-input code-name-input",
                            ),
                        ],
                        className="code-modal-field",
                    ),
                    html.Div(
                        [
                            html.Label("Description (optional)"),
                            dcc.Textarea(
                                id="code-modal-definition",
                                placeholder="Describe what the code captures. Markdown and #hashtags are supported.",
                                className="code-definition-editor code-modal-definition",
                            ),
                        ],
                        className="code-modal-field",
                    ),
                    html.Div(id="code-modal-status", className="control-status modal-status"),
                    html.Div(
                        [
                            html.Button("Cancel", id="code-modal-cancel-footer", n_clicks=0),
                            html.Button(
                                "Create Code",
                                id="code-modal-create-button",
                                n_clicks=0,
                                className="primary-action",
                            ),
                        ],
                        className="modal-actions",
                    ),
                ],
                className="modal-card code-modal-card",
            )
        ],
        id="code-modal",
        className="modal-overlay",
        style={"display": "none"},
    )


def _projection_modal(*, dcc: Any, html: Any) -> Any:
    """Return a native Dash modal for creating a persistent 2D view."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Create 2D view"),
                            html.Button("×", id="projection-cancel-button", n_clicks=0),
                        ],
                        className="modal-header",
                    ),
                    html.P(
                        "The new projection is computed from the selected geometry, "
                        "cached, and added to the 2D-view menu.",
                        className="modal-help",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("View name"),
                                    dcc.Input(
                                        id="projection-name",
                                        type="text",
                                        placeholder="Optional; generated automatically",
                                        className="text-input",
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label("Projection type"),
                                    dcc.Dropdown(
                                        id="projection-method",
                                        options=[
                                            {"label": "UMAP", "value": "umap"},
                                            {"label": "PCA", "value": "pca"},
                                            {
                                                "label": "Truncated SVD",
                                                "value": "truncated_svd",
                                            },
                                        ],
                                        value="umap",
                                        clearable=False,
                                        searchable=False,
                                    ),
                                ]
                            ),
                        ],
                        className="projection-base-fields",
                    ),
                    html.Div(
                        [
                            _number_field(
                                dcc=dcc, html=html, label="Neighbors",
                                component_id="projection-umap-neighbors",
                                value=15,
                                minimum=2,
                                step=1,
                            ),
                            _number_field(
                                dcc=dcc, html=html, label="Minimum distance",
                                component_id="projection-umap-min-dist",
                                value=0.05,
                                minimum=0.0,
                                step=0.01,
                            ),
                            html.Div(
                                [
                                    html.Label("Metric"),
                                    dcc.Dropdown(
                                        id="projection-umap-metric",
                                        options=[
                                            {"label": value, "value": value}
                                            for value in [
                                                "cosine", "euclidean", "manhattan",
                                                "correlation", "chebyshev",
                                            ]
                                        ],
                                        value="cosine",
                                        clearable=False,
                                        searchable=False,
                                    ),
                                ]
                            ),
                            _number_field(
                                dcc=dcc, html=html, label="Spread",
                                component_id="projection-umap-spread",
                                value=2.5,
                                minimum=0.01,
                                step=0.1,
                            ),
                            _number_field(
                                dcc=dcc, html=html, label="Repulsion strength",
                                component_id="projection-umap-repulsion",
                                value=1.5,
                                minimum=0.0,
                                step=0.1,
                            ),
                            _number_field(
                                dcc=dcc, html=html, label="Random state",
                                component_id="projection-umap-random-state", value=0, step=1,
                            ),
                        ],
                        id="projection-umap-fields",
                        className="projection-parameter-grid",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("SVD solver"),
                                    dcc.Dropdown(
                                        id="projection-pca-solver",
                                        options=[
                                            {"label": value, "value": value}
                                            for value in [
                                                "auto",
                                                "full",
                                                "covariance_eigh",
                                                "arpack",
                                                "randomized",
                                            ]
                                        ],
                                        value="auto",
                                        clearable=False,
                                        searchable=False,
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label("Whiten"),
                                    dcc.Checklist(
                                        id="projection-pca-whiten",
                                        options=[{"label": "Scale components", "value": "whiten"}],
                                        value=[],
                                    ),
                                ]
                            ),
                            _number_field(
                                dcc=dcc, html=html, label="Random state",
                                component_id="projection-pca-random-state", value=0, step=1,
                            ),
                        ],
                        id="projection-pca-fields",
                        className="projection-parameter-grid",
                        style={"display": "none"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Algorithm"),
                                    dcc.Dropdown(
                                        id="projection-svd-algorithm",
                                        options=[
                                            {"label": "Randomized", "value": "randomized"},
                                            {"label": "ARPACK", "value": "arpack"},
                                        ],
                                        value="randomized",
                                        clearable=False,
                                        searchable=False,
                                    ),
                                ]
                            ),
                            _number_field(
                                dcc=dcc, html=html, label="Power iterations",
                                component_id="projection-svd-n-iter", value=5, minimum=1, step=1,
                            ),
                            _number_field(
                                dcc=dcc, html=html, label="Random state",
                                component_id="projection-svd-random-state", value=0, step=1,
                            ),
                        ],
                        id="projection-svd-fields",
                        className="projection-parameter-grid",
                        style={"display": "none"},
                    ),
                    html.Div(
                        html.Div(className="projection-progress-bar"),
                        id="projection-progress-wrap",
                        className="projection-progress-track",
                        style={"display": "none"},
                    ),
                    html.Div(
                        id="projection-create-status",
                        className="control-status modal-status",
                    ),
                    html.Div(
                        [
                            html.Button("Cancel", id="projection-cancel-footer", n_clicks=0),
                            html.Button(
                                "Create view",
                                id="projection-create-button",
                                n_clicks=0,
                                className="primary-action",
                            ),
                        ],
                        className="modal-actions",
                    ),
                ],
                className="modal-card",
            )
        ],
        id="projection-modal",
        className="modal-overlay",
        style={"display": "none"},
    )


def _number_field(
    *,
    dcc: Any,
    html: Any,
    label: str,
    component_id: str,
    value: int | float,
    minimum: int | float | None = None,
    step: int | float = 1,
) -> Any:
    """Render a plain numeric text field without browser spinner controls.

    Validation happens only when the projection is submitted, which avoids the
    inconsistent native-number validation users encountered across browsers.
    """
    del minimum, step
    return html.Div(
        [
            html.Label(label),
            dcc.Input(
                id=component_id,
                type="text",
                value=str(value),
                className="projection-number-input",
            ),
        ]
    )



def _focus_layout(
    *,
    dcc: Any,
    html: Any,
    codes: list[dict[str, Any]],
    geometries: list[dict[str, Any]],
    classifier_specs: list[dict[str, Any]],
    hierarchy: list[dict[str, Any]],
    initial_state: dict[str, Any],
) -> Any:
    """Return the code-development workspace built around explicit classifiers."""
    del geometries
    code_ids = {int(code["code_id"]) for code in codes}
    try:
        requested_code_id = int(initial_state.get("focus_code_id"))
    except (TypeError, ValueError):
        requested_code_id = -1
    initial_code_id = requested_code_id if requested_code_id in code_ids else None

    classifier_specs = [
        row
        for row in classifier_specs
        if initial_code_id is not None and int(row["code_id"]) == initial_code_id
    ]
    spec_ids = {int(row["classifier_spec_id"]) for row in classifier_specs}
    try:
        requested_spec_id = int(initial_state.get("focus_classifier_spec_id"))
    except (TypeError, ValueError):
        requested_spec_id = -1
    initial_spec_id = (
        requested_spec_id
        if requested_spec_id in spec_ids
        else (min(spec_ids) if spec_ids else None)
    )
    active_class = "focus-active-area" + (
        " focus-disabled" if initial_code_id is None else ""
    )
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Active code"),
                            html.Div(
                                [
                                    dcc.Dropdown(
                                        id="focus-code-dropdown",
                                        options=_code_options(codes),
                                        value=initial_code_id,
                                        clearable=True,
                                        placeholder="Select a code",
                                    ),
                                    html.Button(
                                        "New Code", id="focus-new-code-button", n_clicks=0
                                    ),
                                ],
                                className="code-selector-row",
                            ),
                            html.Div(
                                id="focus-code-definition", className="code-definition"
                            ),
                            html.Div(id="focus-label-counts", className="focus-counts"),
                            html.Div(
                                id="focus-code-create-status", className="control-status"
                            ),
                        ],
                        className="focus-code-select",
                    ),
                    html.Div(
                        [
                            html.Label("Teaching examples"),
                            html.Div(
                                "Select a code to manage examples.",
                                id="focus-teaching-counts",
                                className="focus-counts",
                            ),
                            html.Button(
                                "Manage teaching examples",
                                id="focus-teaching-button",
                                n_clicks=0,
                                disabled=initial_code_id is None,
                            ),
                            html.Small(
                                "Active examples join corpus assignments at the next manual retrain.",
                                className="focus-help",
                            ),
                        ],
                        className="focus-teaching-select",
                    ),
                ],
                className="focus-topbar",
            ),
            html.Div(
                [
                    html.Div(
                        "Select a code to begin developing it.",
                        id="focus-disabled-message",
                        className="focus-disabled-message",
                    ),
                    html.Div(
                        [
                            html.Aside(
                                [
                                    html.H3("Classifier controls"),
                                    html.Div(
                                        [
                                            html.Label("Active classifier"),
                                            html.Span(
                                                "Not trained",
                                                id="focus-active-classifier-status",
                                                className="classifier-status-badge status-not-trained",
                                            ),
                                        ],
                                        className="active-classifier-heading",
                                    ),
                                    dcc.Dropdown(
                                        id="focus-active-classifier-dropdown",
                                        options=[
                                            {
                                                "label": f"{row['name']} · {row['geometry_name']}",
                                                "value": int(row["classifier_spec_id"]),
                                            }
                                            for row in classifier_specs
                                        ],
                                        value=initial_spec_id,
                                        clearable=False,
                                        placeholder="Create a classifier",
                                    ),
                                    html.Button(
                                        "Train",
                                        id="focus-train-active",
                                        n_clicks=0,
                                        className="primary-action classifier-train-button",
                                    ),
                                    html.Div(
                                        [
                                            dcc.Checklist(
                                                id="focus-train-scope",
                                                options=[
                                                    {
                                                        "label": "Train all classifiers",
                                                        "value": "all",
                                                    }
                                                ],
                                                value=(
                                                    ["all"]
                                                    if bool(
                                                        initial_state.get(
                                                            "focus_train_all", True
                                                        )
                                                    )
                                                    else []
                                                ),
                                                className="classifier-train-scope",
                                            ),
                                            dcc.Checklist(
                                                id="focus-tune-hyperparameters",
                                                options=[
                                                    {
                                                        "label": "Tune hyperparameters with cross-validation",
                                                        "value": "tune",
                                                    }
                                                ],
                                                value=(
                                                    ["tune"]
                                                    if bool(
                                                        initial_state.get(
                                                            "focus_tune_hyperparameters", False
                                                        )
                                                    )
                                                    else []
                                                ),
                                            ),
                                            dcc.Checklist(
                                                id="focus-auto-retrain",
                                                options=[
                                                    {
                                                        "label": "Automatically train before recommendations",
                                                        "value": "auto",
                                                    }
                                                ],
                                                value=(
                                                    ["auto"]
                                                    if bool(
                                                        initial_state.get(
                                                            "focus_auto_retrain", True
                                                        )
                                                    )
                                                    else []
                                                ),
                                            ),
                                        ],
                                        className="classifier-training-options",
                                    ),
                                    html.Small(
                                        "Train uses each classifier's current hyperparameters by default. "
                                        "Check Tune hyperparameters with cross-validation when you want the "
                                        "slower family-specific search before fitting. Automatic training uses "
                                        "the same tuning choice.",
                                        className="focus-help",
                                    ),
                                    html.Div(
                                        [
                                            html.Button(
                                                "New classifier",
                                                id="focus-new-classifier-button",
                                                n_clicks=0,
                                            ),
                                            html.Button(
                                                "Manage classifiers",
                                                id="focus-manage-classifiers",
                                                n_clicks=0,
                                            ),
                                        ],
                                        className="classifier-management-row",
                                    ),
                                    html.Div(
                                        id="focus-active-classifier-detail",
                                        className="focus-model-detail",
                                    ),
                                    html.Div(
                                        id="focus-train-action-status",
                                        className="control-status",
                                    ),
                                    html.H3("Committee"),
                                    dcc.Dropdown(
                                        id="focus-committee-dropdown",
                                        options=[],
                                        value=None,
                                        clearable=True,
                                        placeholder="Select a saved committee",
                                    ),
                                    html.Div(
                                        [
                                            html.Button(
                                                "Manage committees",
                                                id="focus-manage-committees",
                                                n_clicks=0,
                                            ),
                                            html.Button(
                                                "Train committee",
                                                id="focus-train-committee",
                                                n_clicks=0,
                                            ),
                                        ],
                                        className="classifier-action-row",
                                    ),
                                    html.Div(
                                        id="focus-committee-status",
                                        className="focus-model-status",
                                    ),
                                    html.Div(
                                        id="focus-committee-train-status",
                                        className="control-status",
                                    ),
                                    html.Button(
                                        "Testing Center",
                                        id="focus-testing-center-button",
                                        n_clicks=0,
                                    ),
                                ],
                                className="focus-sidebar",
                            ),
                            html.Main(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Label("Recommendation source"),
                                                    dcc.Dropdown(
                                                        id="focus-recommendation-source",
                                                        options=[
                                                            {
                                                                "label": "Active classifier",
                                                                "value": "classifier",
                                                            },
                                                            {
                                                                "label": "Selected committee",
                                                                "value": "committee",
                                                            },
                                                        ],
                                                        value="classifier",
                                                        clearable=False,
                                                    ),
                                                ],
                                                className="recommendation-source-control",
                                            ),
                                            html.Div(
                                                [
                                                    html.Button(
                                                        "Most likely",
                                                        id="focus-most-likely",
                                                        n_clicks=0,
                                                    ),
                                                    html.Button(
                                                        "Least likely",
                                                        id="focus-least-likely",
                                                        n_clicks=0,
                                                    ),
                                                    html.Button(
                                                        "Most uncertain",
                                                        id="focus-most-uncertain",
                                                        n_clicks=0,
                                                    ),
                                                    html.Button(
                                                        "Greatest disagreement",
                                                        id="focus-disagreement",
                                                        n_clicks=0,
                                                        style={"display": "none"},
                                                    ),
                                                    html.Button(
                                                        "Review unsure",
                                                        id="focus-review-unsure",
                                                        n_clicks=0,
                                                    ),
                                                    html.Button(
                                                        "Random",
                                                        id="focus-random",
                                                        n_clicks=0,
                                                    ),
                                                ],
                                                className="focus-recommendation-buttons",
                                            ),
                                        ],
                                        className="focus-recommendation-panel",
                                    ),
                                    html.Div(
                                        id="focus-recommendation-status",
                                        className="focus-status",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Div(
                                                        id="focus-focal-key",
                                                        className="focal-key",
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.Label(
                                                                        "Context level"
                                                                    ),
                                                                    dcc.Dropdown(
                                                                        id="focus-context-level",
                                                                        options=[
                                                                            {
                                                                                "label": level[
                                                                                    "column_name"
                                                                                ],
                                                                                "value": int(
                                                                                    level[
                                                                                        "level_index"
                                                                                    ]
                                                                                ),
                                                                            }
                                                                            for level in hierarchy
                                                                        ],
                                                                        value=int(
                                                                            initial_state.get(
                                                                                "focus_context_level",
                                                                                hierarchy[-1][
                                                                                    "level_index"
                                                                                ],
                                                                            )
                                                                        ),
                                                                        clearable=False,
                                                                        searchable=False,
                                                                    ),
                                                                ],
                                                                className="context-control",
                                                            ),
                                                            html.Div(
                                                                [
                                                                    html.Label("Window"),
                                                                    html.Div(
                                                                        [
                                                                            html.Button(
                                                                                "−",
                                                                                id="focus-context-window-minus",
                                                                                n_clicks=0,
                                                                            ),
                                                                            dcc.Input(
                                                                                id="focus-context-window",
                                                                                type="text",
                                                                                inputMode="numeric",
                                                                                value=str(
                                                                                    max(
                                                                                        1,
                                                                                        int(
                                                                                            initial_state.get(
                                                                                                "focus_context_window",
                                                                                                1,
                                                                                            )
                                                                                        ),
                                                                                    )
                                                                                ),
                                                                            ),
                                                                            html.Button(
                                                                                "+",
                                                                                id="focus-context-window-plus",
                                                                                n_clicks=0,
                                                                            ),
                                                                        ],
                                                                        className="window-stepper",
                                                                    ),
                                                                ],
                                                                className="context-control",
                                                            ),
                                                        ],
                                                        className="context-controls",
                                                    ),
                                                    html.Div(
                                                        id="focus-context-content",
                                                        className="context-content focus-context-content",
                                                    ),
                                                    html.Details(
                                                        [
                                                            html.Summary("Metadata"),
                                                            html.Div(
                                                                id="focus-metadata-content"
                                                            ),
                                                        ]
                                                    ),
                                                ],
                                                className="focus-reading",
                                            ),
                                            html.Aside(
                                                [
                                                    html.H3("Code this observation"),
                                                    html.Div(
                                                        [
                                                            html.Button(
                                                                "Present",
                                                                id="focus-label-positive",
                                                                n_clicks=0,
                                                                className="judgment-button positive-button",
                                                            ),
                                                            html.Button(
                                                                "Absent",
                                                                id="focus-label-negative",
                                                                n_clicks=0,
                                                                className="judgment-button negative-button",
                                                            ),
                                                            html.Button(
                                                                "Unsure",
                                                                id="focus-label-unsure",
                                                                n_clicks=0,
                                                                className="judgment-button unsure-button",
                                                            ),
                                                        ],
                                                        className="focus-label-buttons",
                                                    ),
                                                    html.Div(
                                                        id="focus-current-label",
                                                        className="sr-only",
                                                    ),
                                                    html.Div(
                                                        id="focus-annotation-status",
                                                        className="sr-only",
                                                    ),
                                                    html.Div(
                                                        id="focus-probabilities",
                                                        className="focus-probabilities",
                                                    ),
                                                ],
                                                className="focus-coding-panel",
                                            ),
                                        ],
                                        className="focus-main",
                                    ),
                                    html.Section(
                                        [
                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            html.H3(
                                                                "Prediction geometry"
                                                            ),
                                                            html.P(
                                                                "Choose fitted classifiers. One score is shown as a jittered line, two as direct axes, and three or more through PCA.",
                                                                className="focus-help",
                                                            ),
                                                        ]
                                                    ),
                                                    html.Button(
                                                        "Refresh score view",
                                                        id="focus-prediction-refresh",
                                                        n_clicks=0,
                                                    ),
                                                ],
                                                className="center-heading-row",
                                            ),
                                            html.Details(
                                                [
                                                    html.Summary(
                                                        "Select classifiers for prediction geometry"
                                                    ),
                                                    dcc.Checklist(
                                                        id="focus-prediction-fits",
                                                        options=[],
                                                        value=[],
                                                        className="prediction-fit-checklist",
                                                    ),
                                                ],
                                                className="prediction-classifier-picker",
                                            ),
                                            html.Div(
                                                [
                                                    dcc.Checklist(
                                                        id="focus-prediction-display-options",
                                                        options=[
                                                            {
                                                                "label": "Show all coded units",
                                                                "value": "coded",
                                                            },
                                                            {
                                                                "label": "Show spans",
                                                                "value": "spans",
                                                            },
                                                            {
                                                                "label": "Show teaching examples",
                                                                "value": "teaching",
                                                            },
                                                        ],
                                                        value=["coded"],
                                                        inline=True,
                                                        className="prediction-display-options",
                                                    ),
                                                ],
                                                className="prediction-geometry-options",
                                            ),
                                            html.Div(
                                                id="focus-prediction-status",
                                                className="control-status",
                                            ),
                                            dcc.Graph(
                                                id="focus-prediction-map",
                                                figure=go.Figure(),
                                                config={
                                                    "displaylogo": False,
                                                    "scrollZoom": True,
                                                    "doubleClick": "reset",
                                                    "modeBarButtonsToRemove": [
                                                        "lasso2d",
                                                        "select2d",
                                                    ],
                                                },
                                                className="focus-prediction-map",
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Points per page",
                                                        htmlFor="focus-prediction-page-size",
                                                    ),
                                                    dcc.Input(
                                                        id="focus-prediction-page-size",
                                                        type="text",
                                                        inputMode="numeric",
                                                        debounce=True,
                                                        value="5000",
                                                        className="page-size-input",
                                                    ),
                                                    html.Button(
                                                        "Previous",
                                                        id="focus-prediction-page-previous",
                                                        n_clicks=0,
                                                    ),
                                                    html.Label(
                                                        "Page",
                                                        htmlFor="focus-prediction-page-number",
                                                    ),
                                                    dcc.Input(
                                                        id="focus-prediction-page-number",
                                                        type="number",
                                                        min=1,
                                                        step=1,
                                                        debounce=True,
                                                        value=1,
                                                        max=1,
                                                        className="page-number-input",
                                                    ),
                                                    html.Button(
                                                        "Next",
                                                        id="focus-prediction-page-next",
                                                        n_clicks=0,
                                                    ),
                                                    html.Div(
                                                        id="focus-prediction-page-status",
                                                        className="page-status",
                                                    ),
                                                ],
                                                className="pagination-bar prediction-pagination",
                                            ),
                                        ],
                                        className="prediction-geometry-panel",
                                    ),
                                ],
                                className="focus-content",
                            ),
                        ],
                        className="focus-workspace-grid",
                    ),
                ],
                id="focus-active-area",
                className=active_class,
            ),
        ],
        className="focus-workspace",
    )


def _apply_layout(
    *, dcc: Any, html: Any, dag: Any,
    codes: list[dict[str, Any]], classifier_specs: list[dict[str, Any]],
) -> Any:
    del classifier_specs
    return html.Div(
        [
            html.Div(
                [
                    html.H2("Apply and Review"),
                    html.P("Generate an inspectable proposal draft. Nothing becomes an annotation until you commit reviewed decisions."),
                    html.Label("Code"),
                    dcc.Dropdown(id="apply-code-dropdown", options=_code_options(codes), value=(int(codes[0]["code_id"]) if codes else None), clearable=False),
                    html.Label("Classifier or committee"),
                    dcc.Dropdown(
                        id="apply-source-dropdown",
                        options=[],
                        value=None,
                        clearable=False,
                        placeholder="Train a classifier in Develop first",
                    ),
                    html.Div(
                        "Apply uses the newest explicitly trained fit. Stale fits remain usable and are identified in Develop.",
                        className="focus-help",
                    ),
                    html.Div(
                        [
                            html.Label("Decision threshold"),
                            dcc.Input(id="apply-threshold", type="number", min=0.0, max=1.0, step=0.01, value=0.5),
                            html.Button("Generate proposal draft", id="apply-generate", n_clicks=0, className="primary-action"),
                        ], className="apply-generation-controls",
                    ),
                ], className="apply-sidebar",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div([html.Label("Draft"), dcc.Dropdown(id="apply-run-dropdown", options=[], value=None)], className="apply-draft-select"),
                            html.Button("Discard draft", id="apply-discard", n_clicks=0, className="danger-action"),
                            dcc.ConfirmDialog(id="apply-discard-confirm", message="Discard this proposal draft? No annotations will be changed."),
                        ], className="apply-draft-header",
                    ),
                    html.Div(id="apply-run-summary", className="focus-status"),
                    html.Div(
                        [
                            dcc.Dropdown(id="apply-proposed-filter", options=[{"label": "All proposals", "value": ""}, {"label": "Positive proposals", "value": "positive"}, {"label": "Negative proposals", "value": "negative"}], value="", clearable=False),
                            dcc.Dropdown(id="apply-decision-filter", options=[{"label": "All decisions", "value": ""}, {"label": "Pending", "value": "pending"}, {"label": "Accepted", "value": "accept"}, {"label": "Positive override", "value": "positive"}, {"label": "Negative override", "value": "negative"}, {"label": "Unsure", "value": "unsure"}, {"label": "Leave unreviewed", "value": "unreviewed"}], value="", clearable=False),
                            html.Div(
                                [html.Label("Probability range"), dcc.RangeSlider(
                                    id="apply-probability-range",
                                    min=0.0,
                                    max=1.0,
                                    step=0.000001,
                                    value=[0.0, 1.0],
                                    marks={},
                                    allowCross=False,
                                    allow_direct_input=True,
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": False,
                                        "transform": "formatProbability",
                                        "style": {
                                            "minWidth": "92px",
                                            "textAlign": "center",
                                            "fontVariantNumeric": "tabular-nums",
                                        },
                                    },
                                )],
                                className="apply-probability-filter",
                            ),
                        ], className="apply-filters",
                    ),
                    dag.AgGrid(
                        id="apply-table",
                        columnDefs=[
                            {
                                "headerName": "Unit",
                                "field": "unit_id",
                                "width": 92,
                                "editable": False,
                                "pinned": "left",
                            },
                            {
                                "headerName": "Key",
                                "field": "key",
                                "minWidth": 190,
                                "flex": 0.9,
                                "wrapText": True,
                                "autoHeight": True,
                                "cellStyle": {"lineHeight": "1.3"},
                                "editable": False,
                            },
                            {
                                "headerName": "Text",
                                "field": "text",
                                "minWidth": 360,
                                "flex": 2.2,
                                "wrapText": True,
                                "autoHeight": True,
                                "cellStyle": {"lineHeight": "1.3"},
                                "editable": False,
                            },
                            {
                                "headerName": "Probability",
                                "field": "probability",
                                "width": 128,
                                "editable": False,
                                "valueFormatter": {
                                    "function": "params.value == null ? '' : Number(params.value).toFixed(6)"
                                },
                            },
                            {
                                "headerName": "Proposal",
                                "field": "proposed_label",
                                "width": 118,
                                "editable": False,
                            },
                            {
                                "headerName": "Decision",
                                "field": "decision",
                                "width": 180,
                                "editable": True,
                                "cellEditor": "agSelectCellEditor",
                                "cellEditorParams": {
                                    "values": list(APPLY_DECISION_VALUES),
                                },
                            },
                        ],
                        rowData=[],
                        defaultColDef={
                            "sortable": True,
                            "resizable": True,
                            "suppressHeaderMenuButton": True,
                        },
                        columnSize="responsiveSizeToFit",
                        dashGridOptions={
                            "theme": GECO_AG_GRID_THEME,
                            "animateRows": False,
                            "pagination": True,
                            "paginationPageSize": 25,
                            "paginationPageSizeSelector": [25, 50, 100],
                            "singleClickEdit": True,
                            "stopEditingWhenCellsLoseFocus": True,
                            "rowSelection": {
                                "mode": "multiRow",
                                "checkboxes": True,
                                "headerCheckbox": True,
                                "selectAll": "currentPage",
                            },
                            "selectionColumnDef": {
                                "width": 44,
                                "resizable": False,
                                "sortable": False,
                                "pinned": "left",
                                "lockPosition": "left",
                                "suppressMovable": True,
                                "suppressHeaderMenuButton": True,
                            },
                            "getRowId": {"function": "params.data.unit_id"},
                        },
                        className="apply-review-grid",
                        style={"height": "690px", "width": "100%"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H4("Bulk proposal actions"),
                                    html.Div(
                                        [
                                            html.Label("Decision", htmlFor="apply-bulk-decision"),
                                            dcc.Dropdown(
                                                id="apply-bulk-decision",
                                                options=[
                                                    {"label": "Accept proposal", "value": "accept"},
                                                    {"label": "Assign positive", "value": "positive"},
                                                    {"label": "Assign negative", "value": "negative"},
                                                    {"label": "Assign unsure", "value": "unsure"},
                                                    {"label": "Leave unreviewed", "value": "unreviewed"},
                                                ],
                                                value="accept",
                                                clearable=False,
                                            ),
                                        ],
                                        className="apply-bulk-field",
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Scope", htmlFor="apply-bulk-scope"),
                                            dcc.Dropdown(
                                                id="apply-bulk-scope",
                                                options=[
                                                    {"label": "All proposals", "value": "all"},
                                                    {"label": "Current page", "value": "current_page"},
                                                    {"label": "Selected proposals", "value": "selected"},
                                                ],
                                                value="selected",
                                                clearable=False,
                                            ),
                                        ],
                                        className="apply-bulk-field",
                                    ),
                                    html.Button(
                                        "Apply",
                                        id="apply-bulk-apply",
                                        n_clicks=0,
                                        className="primary-action apply-bulk-button",
                                    ),
                                ],
                                className="apply-action-group apply-bulk-action-group",
                            ),
                            html.Div([html.H4("Clear decisions"), html.Button("Clear all", id="apply-clear-decisions", n_clicks=0), html.Button("Clear current page", id="apply-clear-visible", n_clicks=0)], className="apply-action-group"),
                            html.Div(
                                [
                                    html.H4("Commit"),
                                    html.Button(
                                        "Commit all reviewed",
                                        id="apply-commit",
                                        n_clicks=0,
                                        className="primary-action",
                                    ),
                                    html.Button(
                                        "Commit current page reviewed",
                                        id="apply-commit-visible",
                                        n_clicks=0,
                                    ),
                                    html.Button(
                                        "Undo last commit",
                                        id="apply-undo-last-commit",
                                        n_clicks=0,
                                        className="danger-action",
                                        disabled=True,
                                    ),
                                ],
                                className="apply-action-group",
                            ),
                        ], className="apply-actions",
                    ),
                    html.Div(id="apply-status", className="control-status"),
                ], className="apply-main",
            ),
        ], className="apply-workspace",
    )

def _project_summary_children(
    *,
    html: Any,
    modality: str,
    key_columns: list[str],
    hierarchy_summary: list[dict[str, Any]],
    atomic_count: int,
) -> list[Any]:
    """Build the compact modality and hierarchy summary shown in the header."""
    atomic_key = key_columns[-1]
    children: list[Any] = [
        html.Span([html.Strong("Modality:"), " ", modality]),
        html.Span("·", className="project-summary-separator"),
        html.Span(
            [html.Strong("Atomic key:"), " ", f"{atomic_key} (n={atomic_count:,})"]
        ),
    ]
    parents = hierarchy_summary[:-1]
    if parents:
        label = "Parent key:" if len(parents) == 1 else "Parent keys:"
        parent_children: list[Any] = [html.Strong(label), " "]
        for index, parent in enumerate(parents):
            if index:
                parent_children.append(", ")
            parent_children.append(
                f"{parent['column_name']} (n={int(parent['count']):,})"
            )
        children.extend(
            [
                html.Span("·", className="project-summary-separator"),
                html.Span(parent_children),
            ]
        )
    return children


def _normalized_probability_range(value: list[float] | None) -> tuple[float, float]:
    """Return a validated closed probability interval."""
    if not value or len(value) != 2:
        return 0.0, 1.0
    lower = min(1.0, max(0.0, float(value[0])))
    upper = min(1.0, max(0.0, float(value[1])))
    return (lower, upper) if lower <= upper else (upper, lower)


def _normalize_apply_decision(value: Any) -> str:
    """Normalize an AG Grid display label or stored value to a decision value."""
    rendered = str(value if value is not None else "pending").strip()
    if rendered in APPLY_DECISION_LABELS:
        return rendered
    if rendered in APPLY_DECISION_VALUES:
        return APPLY_DECISION_VALUES[rendered]
    if rendered == "Exclude":
        # Compatibility for a stale pre-0.5.1 browser row. Projects are rebuilt.
        return "unreviewed"
    raise ValueError(f"Unknown review decision {rendered!r}.")


def _apply_current_page_rows(
    rows: list[dict[str, Any]], pagination_info: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Return exactly the rows displayed on the current client-side AG Grid page."""
    info = pagination_info or {}
    current_page = max(0, int(info.get("currentPage", 0) or 0))
    page_size = max(1, int(info.get("pageSize", 25) or 25))
    start = current_page * page_size
    return list(rows[start : start + page_size])


def _apply_bulk_scope_unit_ids(
    *,
    scope: str,
    current_page_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
) -> list[int] | None:
    """Resolve the proposal IDs targeted by one bulk review action."""
    if scope == "all":
        return None
    if scope == "current_page":
        return [int(row["unit_id"]) for row in current_page_rows]
    if scope == "selected":
        return [int(row["unit_id"]) for row in selected_rows]
    raise ValueError(f"Unknown bulk proposal scope {scope!r}.")


def _code_options(codes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"label": code["name"], "value": int(code["code_id"])}
        for code in codes
    ]

def _session_options(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"label": session["title"], "value": int(session["session_id"])}
        for session in sessions
    ]


def _valid_geometry_id(value: Any, geometries: list[dict[str, Any]]) -> int | None:
    valid = {int(record["geometry_id"]) for record in geometries}
    try:
        candidate = int(value)
    except (TypeError, ValueError):
        candidate = -1
    return candidate if candidate in valid else (min(valid) if valid else None)


def _valid_view_id(value: Any, views: list[dict[str, Any]]) -> int | None:
    valid = {int(record["view_id"]) for record in views}
    try:
        candidate = int(value)
    except (TypeError, ValueError):
        candidate = -1
    return candidate if candidate in valid else (min(valid) if valid else None)


def _coerce_positive_int(value: Any, *, default: int) -> int:
    """Parse a user-entered positive integer without browser step constraints."""
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return int(default)
    return parsed if parsed >= 1 else int(default)


def _clicked_unit_id(click_data: dict[str, Any] | None) -> int | None:
    """Extract one unit ID from a Plotly click event."""
    if not click_data or not click_data.get("points"):
        return None
    customdata = click_data["points"][0].get("customdata")
    if isinstance(customdata, (list, tuple, np.ndarray)):
        if len(customdata) == 0:
            return None
        customdata = customdata[0]
    try:
        return int(customdata)
    except (TypeError, ValueError):
        return None


def _projection_parameters(
    *,
    method: str,
    umap_neighbors: int | None,
    umap_min_dist: float | None,
    umap_metric: str | None,
    umap_spread: float | None,
    umap_repulsion: float | None,
    umap_random_state: int | None,
    pca_solver: str | None,
    pca_whiten: list[str] | None,
    pca_random_state: int | None,
    svd_algorithm: str | None,
    svd_n_iter: int | None,
    svd_random_state: int | None,
) -> dict[str, Any]:
    """Validate projection-form fields and return reducer parameters."""
    if method == "umap":
        neighbors = int(umap_neighbors or 15)
        min_dist = float(0.05 if umap_min_dist is None else umap_min_dist)
        spread = float(2.5 if umap_spread is None else umap_spread)
        repulsion = float(1.5 if umap_repulsion is None else umap_repulsion)
        if neighbors < 2:
            raise ValueError("UMAP neighbors must be at least 2.")
        if min_dist < 0:
            raise ValueError("UMAP minimum distance cannot be negative.")
        if spread <= 0:
            raise ValueError("UMAP spread must be positive.")
        if min_dist > spread:
            raise ValueError("UMAP minimum distance cannot exceed spread.")
        if repulsion < 0:
            raise ValueError("UMAP repulsion strength cannot be negative.")
        return {
            "n_neighbors": neighbors,
            "min_dist": min_dist,
            "metric": str(umap_metric or "cosine"),
            "spread": spread,
            "repulsion_strength": repulsion,
            "random_state": int(umap_random_state or 0),
        }
    if method == "pca":
        return {
            "svd_solver": str(pca_solver or "auto"),
            "whiten": "whiten" in (pca_whiten or []),
            "random_state": int(pca_random_state or 0),
        }
    if method in {"truncated_svd", "svd"}:
        n_iter = int(svd_n_iter or 5)
        if n_iter < 1:
            raise ValueError("Truncated-SVD power iterations must be at least 1.")
        return {
            "algorithm": str(svd_algorithm or "randomized"),
            "n_iter": n_iter,
            "random_state": int(svd_random_state or 0),
        }
    raise ValueError(f"Unknown projection method: {method!r}")


_MEMO_REFERENCE_PATTERN = re.compile(r"\[\[unit:(\d+)\]\]")
_MEMO_HASHTAG_PATTERN = re.compile(r"(?<!\w)#([A-Za-z0-9_-]+)")


def _memo_options(memos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"label": str(memo["title"]), "value": int(memo["memo_id"])}
        for memo in memos
    ]


def _extract_memo_metadata(body: str) -> tuple[list[str], list[int]]:
    hashtags = list(dict.fromkeys(_MEMO_HASHTAG_PATTERN.findall(body)))
    references = list(
        dict.fromkeys(int(value) for value in _MEMO_REFERENCE_PATTERN.findall(body))
    )
    return hashtags, references


def _apply_run_options(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "label": (
                f"Draft {run['apply_run_id']} · {run['status']} · "
                f"{run['reviewed_count']}/{run['proposal_count']} decided"
            ),
            "value": int(run["apply_run_id"]),
        }
        for run in runs
    ]



def _geco_index_string() -> str:
    """Return Dash's HTML shell with the GeCo gecko favicon."""
    return """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        <link rel="icon" type="image/svg+xml" href="/assets/geco-favicon.svg">
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""


def _build_metadata_profiles(
    units: list[dict[str, Any]], columns: list[str]
) -> dict[str, dict[str, Any]]:
    """Infer practical UI types for explicitly supplied metadata columns."""
    profiles: dict[str, dict[str, Any]] = {}
    for column in columns:
        values = [unit.get("metadata", {}).get(column) for unit in units]
        nonmissing = [value for value in values if not _is_missing(value)]
        kind = "text"
        options: list[Any] = []
        if nonmissing and all(isinstance(value, bool) for value in nonmissing):
            kind = "boolean"
        elif nonmissing and all(
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, bool)
            for value in nonmissing
        ):
            kind = "numeric"
        elif nonmissing and all(_coerce_datetime(value) is not None for value in nonmissing):
            kind = "datetime"
        else:
            unique = sorted({str(value) for value in nonmissing}, key=str.casefold)
            unique_ratio = len(unique) / max(1, len(nonmissing))
            max_length = max((len(value) for value in unique), default=0)
            if len(unique) <= 100 and (unique_ratio <= 0.35 or max_length <= 80):
                kind = "categorical"
                options = unique
        profiles[column] = {
            "field": column,
            "type": kind,
            "options": options,
            "nonmissing_count": len(nonmissing),
        }
    return profiles


def _normalize_boolean_operator(value: Any) -> str:
    operator = str(value or "and").lower()
    return operator if operator in {"and", "or", "xor"} else "and"


def _normalize_metadata_filters(
    filters: Any,
    profiles: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop malformed filter clauses while preserving stable clause IDs."""
    normalized: list[dict[str, Any]] = []
    if not isinstance(filters, list):
        return normalized
    for raw in filters:
        if not isinstance(raw, dict):
            continue
        field = str(raw.get("field", ""))
        if field not in profiles:
            continue
        profile = profiles[field]
        normalized.append(
            {
                "id": str(raw.get("id") or uuid.uuid4().hex),
                "field": field,
                "type": str(profile["type"]),
                "operator": str(raw.get("operator", "")),
                "value": raw.get("value"),
                "value2": raw.get("value2"),
            }
        )
    return normalized



def _explore_uncoded_mask(
    project: GeometricCoder,
    units: list[dict[str, Any]],
    *,
    enabled: bool,
) -> np.ndarray:
    """Return atomic-unit eligibility for corpus-level coding coverage.

    An atomic point counts as coded only when it currently has at least one
    positive/Present assignment under any code. Negative/Absent and Unsure
    judgments do not satisfy coverage because no substantive code has been
    applied to the point.
    """
    if not enabled:
        return np.ones(len(units), dtype=bool)
    coded_unit_ids = {
        int(annotation["unit_id"])
        for annotation in project.database.current_annotations()
        if annotation.get("unit_id") is not None
        and str(annotation.get("value")) == "positive"
    }
    return np.asarray(
        [int(unit["unit_id"]) not in coded_unit_ids for unit in units],
        dtype=bool,
    )

def _metadata_filtered_page(
    units: list[dict[str, Any]],
    filters: list[dict[str, Any]],
    *,
    operator: str | None,
    page_size: int,
    page_number: int,
    additional_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, Any, np.ndarray, list[str]]:
    """Return one deterministic page drawn only from currently matching units."""
    mask, errors = _evaluate_metadata_filters(units, filters, operator=operator)
    if additional_mask is not None:
        extra = np.asarray(additional_mask, dtype=bool)
        if extra.shape != mask.shape:
            raise ValueError("additional_mask must align with units")
        mask = mask & extra
    matching_positions = np.flatnonzero(mask)
    page = deterministic_page(
        len(matching_positions),
        page_size=page_size,
        page_index=max(0, int(page_number) - 1),
        seed=0,
    )
    positions = matching_positions[page.positions]
    return positions, page, mask, errors


def _metadata_operator_options(kind: str) -> list[dict[str, str]]:
    labels: dict[str, list[tuple[str, str]]] = {
        "numeric": [
            ("Less than", "lt"),
            ("Less than or equal", "lte"),
            ("Greater than", "gt"),
            ("Greater than or equal", "gte"),
            ("Equal to", "eq"),
            ("Not equal to", "neq"),
            ("Between", "between"),
            ("Is missing", "missing"),
            ("Is not missing", "not_missing"),
        ],
        "text": [
            ("Contains", "contains"),
            ("Does not contain", "excludes"),
            ("Equals", "eq"),
            ("Does not equal", "neq"),
            ("Starts with", "starts"),
            ("Ends with", "ends"),
            ("Regular expression", "regex"),
            ("Is missing", "missing"),
            ("Is not missing", "not_missing"),
        ],
        "categorical": [
            ("Is one of", "in"),
            ("Is not one of", "not_in"),
            ("Is missing", "missing"),
            ("Is not missing", "not_missing"),
        ],
        "boolean": [
            ("Is true", "true"),
            ("Is false", "false"),
            ("Is missing", "missing"),
            ("Is not missing", "not_missing"),
        ],
        "datetime": [
            ("Before", "lt"),
            ("On or before", "lte"),
            ("After", "gt"),
            ("On or after", "gte"),
            ("On", "eq"),
            ("Between", "between"),
            ("Is missing", "missing"),
            ("Is not missing", "not_missing"),
        ],
    }
    return [
        {"label": label, "value": value}
        for label, value in labels.get(kind, labels["text"])
    ]


def _metadata_filter_modal(
    *, dcc: Any, html: Any, metadata_profiles: dict[str, dict[str, Any]]
) -> Any:
    fields = [
        {
            "label": f"{field} ({profile['type']})",
            "value": field,
        }
        for field, profile in metadata_profiles.items()
    ]
    first_field = fields[0]["value"] if fields else None
    first_type = (
        str(metadata_profiles[str(first_field)]["type"])
        if first_field is not None
        else "text"
    )
    operators = _metadata_operator_options(first_type)
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Add metadata filter"),
                            html.Button("×", id="metadata-filter-modal-close", n_clicks=0),
                        ],
                        className="modal-header",
                    ),
                    html.P(
                        "Choose a metadata field and a condition. Search filters and "
                        "metadata filters are applied together.",
                        className="modal-help",
                    ),
                    html.Label("Metadata field"),
                    dcc.Dropdown(
                        id="metadata-filter-field",
                        options=fields,
                        value=first_field,
                        clearable=False,
                    ),
                    html.Div(id="metadata-filter-type-help", className="control-status"),
                    html.Label("Condition", className="modal-field-label"),
                    dcc.Dropdown(
                        id="metadata-filter-condition",
                        options=operators,
                        value=(operators[0]["value"] if operators else None),
                        clearable=False,
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Value"),
                                    dcc.Input(
                                        id="metadata-filter-value",
                                        type="text",
                                        className="text-input",
                                    ),
                                ],
                                id="metadata-filter-value-wrap",
                            ),
                            html.Div(
                                [
                                    html.Label("Upper value"),
                                    dcc.Input(
                                        id="metadata-filter-value-second",
                                        type="text",
                                        className="text-input",
                                    ),
                                ],
                                id="metadata-filter-value-second-wrap",
                                style={"display": "none"},
                            ),
                            html.Div(
                                [
                                    html.Label("Values"),
                                    dcc.Dropdown(
                                        id="metadata-filter-category-values",
                                        options=[],
                                        value=[],
                                        multi=True,
                                    ),
                                ],
                                id="metadata-filter-category-wrap",
                                style={"display": "none"},
                            ),
                        ],
                        className="metadata-filter-value-grid",
                    ),
                    html.Div(id="metadata-filter-modal-status", className="modal-status"),
                    html.Div(
                        [
                            html.Button(
                                "Cancel", id="metadata-filter-modal-cancel", n_clicks=0
                            ),
                            html.Button(
                                "Add filter",
                                id="metadata-filter-modal-add",
                                n_clicks=0,
                                className="primary-action",
                            ),
                        ],
                        className="modal-actions",
                    ),
                ],
                className="modal-card filter-modal-card",
            )
        ],
        id="metadata-filter-modal",
        className="modal-overlay",
        style={"display": "none"},
    )


def _save_filter_set_modal(*, dcc: Any, html: Any) -> Any:
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Save filter collection"),
                            html.Button("×", id="save-filter-set-close", n_clicks=0),
                        ],
                        className="modal-header",
                    ),
                    html.Label("Name"),
                    dcc.Input(
                        id="save-filter-set-name",
                        type="text",
                        placeholder="e.g., Public institutions over age 30",
                        className="text-input",
                    ),
                    html.Div(id="save-filter-set-status", className="modal-status"),
                    html.Div(
                        [
                            html.Button("Cancel", id="save-filter-set-cancel", n_clicks=0),
                            html.Button(
                                "Save",
                                id="save-filter-set-confirm",
                                n_clicks=0,
                                className="primary-action",
                            ),
                        ],
                        className="modal-actions",
                    ),
                ],
                className="modal-card filter-modal-card",
            )
        ],
        id="save-filter-set-modal",
        className="modal-overlay",
        style={"display": "none"},
    )


def _load_filter_set_modal(*, dcc: Any, html: Any) -> Any:
    del dcc
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Saved filter collections"),
                            html.Button("×", id="load-filter-set-close", n_clicks=0),
                        ],
                        className="modal-header",
                    ),
                    html.Div(id="saved-filter-set-list", className="saved-filter-set-list"),
                    html.Div(id="load-filter-set-status", className="modal-status"),
                    html.Div(
                        [html.Button("Close", id="load-filter-set-done", n_clicks=0)],
                        className="modal-actions",
                    ),
                ],
                className="modal-card filter-modal-card",
            )
        ],
        id="load-filter-set-modal",
        className="modal-overlay",
        style={"display": "none"},
    )


def _filter_clause_label(clause: dict[str, Any]) -> str:
    field = str(clause.get("field", ""))
    operator = str(clause.get("operator", ""))
    value = clause.get("value")
    value2 = clause.get("value2")
    symbols = {
        "lt": "<",
        "lte": "≤",
        "gt": ">",
        "gte": "≥",
        "eq": "=",
        "neq": "≠",
        "contains": "contains",
        "excludes": "excludes",
        "starts": "starts with",
        "ends": "ends with",
        "regex": "matches",
        "in": "is one of",
        "not_in": "is not one of",
        "true": "is true",
        "false": "is false",
        "missing": "is missing",
        "not_missing": "is not missing",
    }
    if operator == "between":
        return f"{field}: {value} to {value2}"
    if operator in {"missing", "not_missing", "true", "false"}:
        return f"{field}: {symbols[operator]}"
    if isinstance(value, list):
        rendered = ", ".join(str(item) for item in value)
    else:
        rendered = str(value)
    return f"{field}: {symbols.get(operator, operator)} {rendered}".strip()


def _evaluate_metadata_filters(
    units: list[dict[str, Any]],
    filters: list[dict[str, Any]],
    *,
    operator: str | None,
) -> tuple[np.ndarray, list[str]]:
    """Evaluate metadata clauses and combine them with AND/OR/exactly-one XOR."""
    if not filters:
        return np.ones(len(units), dtype=bool), []
    masks: list[np.ndarray] = []
    errors: list[str] = []
    for clause in filters:
        field = str(clause.get("field", ""))
        kind = str(clause.get("type", "text"))
        condition = str(clause.get("operator", ""))
        value = clause.get("value")
        value2 = clause.get("value2")
        try:
            mask = np.asarray(
                [
                    _metadata_value_matches(
                        unit.get("metadata", {}).get(field),
                        kind=kind,
                        condition=condition,
                        value=value,
                        value2=value2,
                    )
                    for unit in units
                ],
                dtype=bool,
            )
        except (TypeError, ValueError, re.error) as error:
            errors.append(f"{field}: {error}")
            mask = np.zeros(len(units), dtype=bool)
        masks.append(mask)
    stacked = np.vstack(masks)
    normalized_operator = _normalize_boolean_operator(operator)
    if normalized_operator == "or":
        combined = np.any(stacked, axis=0)
    elif normalized_operator == "xor":
        combined = np.sum(stacked, axis=0) == 1
    else:
        combined = np.all(stacked, axis=0)
    return combined, errors


def _metadata_value_matches(
    raw: Any,
    *,
    kind: str,
    condition: str,
    value: Any,
    value2: Any,
) -> bool:
    missing = _is_missing(raw)
    if condition == "missing":
        return missing
    if condition == "not_missing":
        return not missing
    if missing:
        return False
    if kind == "boolean":
        return bool(raw) if condition == "true" else not bool(raw)
    if kind == "numeric":
        left = float(raw)
        right = float(value)
        upper = float(value2) if condition == "between" else None
        return _ordered_comparison(left, condition, right, upper)
    if kind == "datetime":
        left_date = _coerce_datetime(raw)
        right_date = _coerce_datetime(value)
        upper_date = _coerce_datetime(value2) if condition == "between" else None
        if left_date is None or right_date is None:
            raise ValueError("enter a valid date")
        return _ordered_comparison(left_date, condition, right_date, upper_date)
    text = str(raw)
    lowered = text.casefold()
    if kind == "categorical":
        selected = value if isinstance(value, list) else [value]
        selected_values = {str(item).casefold() for item in selected}
        matched = lowered in selected_values
        return not matched if condition == "not_in" else matched
    target = str(value or "")
    target_lower = target.casefold()
    if condition == "contains":
        return target_lower in lowered
    if condition == "excludes":
        return target_lower not in lowered
    if condition == "eq":
        return lowered == target_lower
    if condition == "neq":
        return lowered != target_lower
    if condition == "starts":
        return lowered.startswith(target_lower)
    if condition == "ends":
        return lowered.endswith(target_lower)
    if condition == "regex":
        return re.search(target, text, flags=re.IGNORECASE) is not None
    raise ValueError(f"unsupported condition {condition!r}")


def _ordered_comparison(left: Any, condition: str, right: Any, upper: Any) -> bool:
    if condition == "lt":
        return left < right
    if condition == "lte":
        return left <= right
    if condition == "gt":
        return left > right
    if condition == "gte":
        return left >= right
    if condition == "eq":
        return left == right
    if condition == "neq":
        return left != right
    if condition == "between":
        if upper is None:
            raise ValueError("enter both ends of the range")
        lower_bound, upper_bound = sorted((right, upper))
        return lower_bound <= left <= upper_bound
    raise ValueError(f"unsupported condition {condition!r}")


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


def _coerce_datetime(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or not re.match(r"^\d{4}-\d{1,2}-\d{1,2}", text):
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _read_saved_filter_sets(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return payload if isinstance(payload, list) else []


def _write_saved_filter_sets(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_memo_presentation_request(
    search: str | None,
) -> tuple[int | None, int | None]:
    """Parse memo/version identifiers from the presentation URL query string."""
    values = parse_qs((search or "").lstrip("?"))
    try:
        memo_id = int(values.get("memo_id", [""])[0])
    except (TypeError, ValueError):
        memo_id = None
    try:
        version = int(values.get("version", [""])[0])
    except (TypeError, ValueError):
        version = None
    return memo_id, version


def _memo_version_by_number(
    versions: list[dict[str, Any]], requested_version: int | None
) -> dict[str, Any]:
    """Return the requested memo version, falling back to the latest."""
    if not versions:
        raise ValueError("versions must not be empty")
    if requested_version is not None:
        for version in versions:
            if int(version["version_number"]) == int(requested_version):
                return version
    return versions[-1]


def _presentation_unit_id(location_hash: str | None) -> int | None:
    match = re.fullmatch(r"#geco-presentation-unit-(\d+)", location_hash or "")
    return int(match.group(1)) if match is not None else None


def _presentation_filter_definition(
    path: Path,
    saved_filter_id: str | None,
    metadata_profiles: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    """Resolve one saved metadata filter collection for presentation mode."""
    if not saved_filter_id or saved_filter_id == "__all__":
        return [], "and"
    for record in _read_saved_filter_sets(path):
        if str(record.get("id")) == str(saved_filter_id):
            return (
                _normalize_metadata_filters(
                    record.get("filters", []), metadata_profiles
                ),
                _normalize_boolean_operator(record.get("operator")),
            )
    return [], "and"


def _deterministic_page_number_for_position(
    mask: np.ndarray,
    *,
    corpus_position: int,
    page_size: int,
    seed: int = 0,
) -> int | None:
    """Return the 1-based deterministic page containing one eligible corpus row."""
    eligible_positions = np.flatnonzero(np.asarray(mask, dtype=bool))
    local = np.flatnonzero(eligible_positions == int(corpus_position))
    if local.size != 1:
        return None
    permutation = np.random.default_rng(seed).permutation(len(eligible_positions))
    rank = np.flatnonzero(permutation == int(local[0]))
    if rank.size != 1:
        return None
    return int(rank[0]) // max(1, int(page_size)) + 1


def _render_memo_preview(
    dcc: Any,
    html: Any,
    project: GeometricCoder,
    body: str,
    *,
    unit_hash_prefix: str = "geco-unit-",
    link_hashtags: bool = True,
) -> Any:
    """Render Markdown while resolving GeCo hashtags and unit references."""
    if not body:
        return html.P("Memo preview will appear here.", className="memo-preview-empty")

    placeholders: dict[str, str] = {}

    def replace_reference(match: re.Match[str]) -> str:
        unit_id = int(match.group(1))
        try:
            unit = project.unit(unit_id)
            label = " › ".join(
                f"{name}: {value}" for name, value in unit["user_key"].items()
            )
        except KeyError:
            label = f"unit {unit_id}"
        token = f"GECOREFERENCE{len(placeholders)}TOKEN"
        placeholders[token] = (
            f"[{_escape_markdown_label(label)}](#{unit_hash_prefix}{unit_id})"
        )
        return token

    rendered = _MEMO_REFERENCE_PATTERN.sub(replace_reference, body)

    def replace_hashtag(match: re.Match[str]) -> str:
        tag = match.group(1)
        token = f"GECOHASHTAG{len(placeholders)}TOKEN"
        placeholders[token] = (
            f"[#{tag}](#geco-tag-{quote(tag, safe='')})" if link_hashtags else f"#{tag}"
        )
        return token

    rendered = _MEMO_HASHTAG_PATTERN.sub(replace_hashtag, rendered)
    for token, replacement in placeholders.items():
        rendered = rendered.replace(token, replacement)
    return dcc.Markdown(
        rendered,
        dangerously_allow_html=False,
        link_target="_self",
        className="memo-markdown",
    )


def _escape_markdown_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def _normalize_navigation_history(
    history: dict[str, Any] | None,
    *,
    focal_unit_id: int | None = None,
) -> dict[str, Any]:
    data = history or {}
    items: list[int] = []
    for value in data.get("items", []):
        try:
            items.append(int(value))
        except (TypeError, ValueError):
            continue
    if not items and focal_unit_id is not None:
        items = [int(focal_unit_id)]
    try:
        index = int(data.get("index", len(items) - 1))
    except (TypeError, ValueError):
        index = len(items) - 1
    index = min(max(index, -1), len(items) - 1)
    return {"items": items, "index": index}


def _append_navigation_history(
    history: dict[str, Any] | None,
    unit_id: int,
) -> dict[str, Any]:
    normalized = _normalize_navigation_history(history)
    items = list(normalized["items"])
    index = int(normalized["index"])
    if index >= 0 and index < len(items) and items[index] == int(unit_id):
        return normalized
    items = items[: index + 1]
    items.append(int(unit_id))
    return {"items": items, "index": len(items) - 1}


def _move_navigation_history(
    history: dict[str, Any] | None,
    direction: int,
) -> tuple[int, dict[str, Any]] | None:
    normalized = _normalize_navigation_history(history)
    items = list(normalized["items"])
    target = int(normalized["index"]) + int(direction)
    if target < 0 or target >= len(items):
        return None
    updated = {"items": items, "index": target}
    return int(items[target]), updated


def _normalize_explore_code_palette(
    raw: Any,
    valid_code_ids: set[int],
) -> list[int | None]:
    """Return a duplicate-free Explore coding palette with one permanent row."""
    values = raw if isinstance(raw, list) else []
    normalized: list[int | None] = []
    seen: set[int] = set()
    for value in values:
        if value is None:
            normalized.append(None)
            continue
        try:
            code_id = int(value)
        except (TypeError, ValueError):
            normalized.append(None)
            continue
        if code_id not in valid_code_ids or code_id in seen:
            normalized.append(None)
            continue
        normalized.append(code_id)
        seen.add(code_id)
    return normalized or [None]


def _explore_palette_for_row_count(
    raw: Any,
    valid_code_ids: set[int],
    row_count: int | None,
) -> list[int | None]:
    """Return normalized palette values padded/truncated to the visible row count."""
    palette = _normalize_explore_code_palette(raw, valid_code_ids)
    target = max(1, int(row_count or len(palette) or 1))
    if len(palette) < target:
        return [*palette, *([None] * (target - len(palette)))]
    return palette[:target]


def _place_code_in_explore_palette(
    raw: Any,
    code_id: int,
    valid_code_ids: set[int],
) -> list[int | None]:
    """Place a code into the first empty Explore row, appending when needed."""
    palette = _normalize_explore_code_palette(raw, valid_code_ids)
    selected = {int(value) for value in palette if value is not None}
    code_id = int(code_id)
    if code_id in selected:
        return palette
    for index, value in enumerate(palette):
        if value is None:
            updated = list(palette)
            updated[index] = code_id
            return updated
    return [*palette, code_id]


def _judgment_button_classes(
    current: str | None,
    *,
    prefix: str,
) -> tuple[str, str, str]:
    del prefix
    base = "judgment-button"
    return (
        f"{base} positive-button" + (" judgment-selected" if current == "positive" else ""),
        f"{base} negative-button" + (" judgment-selected" if current == "negative" else ""),
        f"{base} unsure-button" + (" judgment-selected" if current == "unsure" else ""),
    )



def _render_readonly_context_units(
    html: Any,
    units: list[dict[str, Any]],
    *,
    focal_unit_id: int,
    key_columns: list[str],
) -> list[Any]:
    """Render hierarchy-aware context without coding/span controls."""
    if not units:
        return []
    parent_columns = key_columns[:-1]
    children: list[Any] = []
    open_values: list[Any] = []
    for unit in units:
        path = [unit["user_key"].get(column) for column in parent_columns]
        shared = 0
        while (
            shared < len(open_values)
            and shared < len(path)
            and open_values[shared] == path[shared]
        ):
            shared += 1
        for level in range(len(open_values) - 1, shared - 1, -1):
            children.append(
                _context_boundary(
                    html,
                    parent_columns[level],
                    open_values[level],
                    boundary="end",
                    level=level,
                )
            )
        open_values = open_values[:shared]
        for level in range(shared, len(path)):
            open_values.append(path[level])
            children.append(
                _context_boundary(
                    html,
                    parent_columns[level],
                    path[level],
                    boundary="start",
                    level=level,
                )
            )
        unit_id = int(unit["unit_id"])
        is_focal = unit_id == int(focal_unit_id)
        children.append(
            html.P(
                [
                    html.Strong(unit["text"], className="focal-text")
                    if is_focal
                    else unit["text"]
                ],
                className=(
                    "context-unit focal-context-unit" if is_focal else "context-unit"
                ),
            )
        )
    for level in range(len(open_values) - 1, -1, -1):
        children.append(
            _context_boundary(
                html,
                parent_columns[level],
                open_values[level],
                boundary="end",
                level=level,
            )
        )
    return children


def _render_context_units(
    html: Any,
    dcc: Any,
    units: list[dict[str, Any]],
    *,
    focal_unit_id: int,
    key_columns: list[str],
    selected_unit_ids: list[int] | None = None,
    checkbox_type: str,
    span_selection_enabled: bool = True,
) -> list[Any]:
    """Render context with a focal-anchored contiguous span selector."""
    if not units:
        return []
    parent_columns = key_columns[:-1]
    selected = set(_normalized_span_selection(units, focal_unit_id, selected_unit_ids))
    children: list[Any] = []
    open_values: list[Any] = []

    for unit in units:
        path = [unit["user_key"].get(column) for column in parent_columns]
        shared = 0
        while (
            shared < len(open_values)
            and shared < len(path)
            and open_values[shared] == path[shared]
        ):
            shared += 1
        for level in range(len(open_values) - 1, shared - 1, -1):
            children.append(
                _context_boundary(
                    html, parent_columns[level], open_values[level],
                    boundary="end", level=level,
                )
            )
        open_values = open_values[:shared]
        for level in range(shared, len(path)):
            open_values.append(path[level])
            children.append(
                _context_boundary(
                    html, parent_columns[level], path[level],
                    boundary="start", level=level,
                )
            )

        unit_id = int(unit["unit_id"])
        is_focal = unit_id == int(focal_unit_id)
        text_child: Any = (
            html.Strong(unit["text"], className="focal-text")
            if is_focal else unit["text"]
        )
        checkbox = dcc.Checklist(
            id={"type": checkbox_type, "unit_id": unit_id},
            options=[
                {
                    "label": "",
                    "value": unit_id,
                    "disabled": is_focal or not span_selection_enabled,
                }
            ],
            value=[unit_id] if unit_id in selected else [],
            className="span-unit-checkbox",
        )
        children.append(
            html.Div(
                [
                    checkbox,
                    html.P(
                        [text_child],
                        className=(
                            "context-unit focal-context-unit"
                            if is_focal else "context-unit"
                        ),
                    ),
                ],
                className=(
                    "context-unit-row span-selected"
                    if unit_id in selected else "context-unit-row"
                ),
            )
        )

    for level in range(len(open_values) - 1, -1, -1):
        children.append(
            _context_boundary(
                html, parent_columns[level], open_values[level],
                boundary="end", level=level,
            )
        )
    return children


def _normalized_span_selection(
    context_units: list[dict[str, Any]],
    focal_unit_id: int,
    selected_unit_ids: list[int] | None,
) -> list[int]:
    """Return one visible contiguous interval anchored on the focal unit."""
    order = [int(unit["unit_id"]) for unit in context_units]
    if int(focal_unit_id) not in order:
        return [int(focal_unit_id)]
    focal_index = order.index(int(focal_unit_id))
    selected = {int(value) for value in (selected_unit_ids or []) if int(value) in order}
    selected.add(int(focal_unit_id))
    indices = [order.index(value) for value in selected]
    start = min(min(indices), focal_index)
    end = max(max(indices), focal_index)
    return order[start : end + 1]


def _update_span_selection(
    context_units: list[dict[str, Any]],
    *,
    focal_unit_id: int,
    current_unit_ids: list[int] | None,
    clicked_unit_id: int,
    checked: bool,
) -> list[int]:
    """Move one endpoint of the focal-anchored discrete interval."""
    order = [int(unit["unit_id"]) for unit in context_units]
    focal = int(focal_unit_id)
    clicked = int(clicked_unit_id)
    current = _normalized_span_selection(context_units, focal, current_unit_ids)
    if clicked not in order or clicked == focal:
        return current
    focal_index = order.index(focal)
    clicked_index = order.index(clicked)
    current_indices = [order.index(value) for value in current]
    start, end = min(current_indices), max(current_indices)
    if checked:
        if clicked_index < focal_index:
            start = clicked_index
        else:
            end = clicked_index
    elif clicked in current:
        if clicked_index < focal_index:
            start = clicked_index + 1
        else:
            end = clicked_index - 1
    return order[start : end + 1]


def _selection_observation_id(
    project: GeometricCoder,
    focal_unit_id: int,
    selected_unit_ids: list[int] | None,
) -> int | None:
    """Resolve the active atomic observation or an already-persisted span."""
    selected = [int(value) for value in (selected_unit_ids or [focal_unit_id])]
    if len(selected) <= 1:
        return project.observation_id_for_unit(int(focal_unit_id))
    span = project.span_observation(selected)
    return int(span["observation_id"]) if span is not None else None


def _context_boundary(
    html: Any,
    column: str,
    value: Any,
    *,
    boundary: str,
    level: int,
) -> Any:
    return html.Div(
        f"— {column}: {value} {boundary} —",
        className=f"context-boundary context-boundary-{boundary}",
        style={"--context-level": str(level)},
    )


def _current_annotation_value(
    project: GeometricCoder, observation_id: int | None, code_id: int
) -> str | None:
    if observation_id is None:
        return None
    annotation = project.current_annotation(
        observation_id=int(observation_id), code_id=int(code_id)
    )
    return str(annotation["value"]) if annotation is not None else None


def _geometry_overlay_payload(
    figure: dict[str, Any] | None,
    *,
    seen_unit_ids: set[int],
    focal_unit_id: int | None,
) -> dict[int, dict[str, list[Any]]] | None:
    """Build lightweight visited/focal overlay arrays from the current base map."""
    if not figure or not isinstance(figure.get("data"), list):
        return None
    data = figure["data"]
    if len(data) <= EXPLORE_FOCAL_TRACE_INDEX:
        return None
    names = tuple(str(data[index].get("name", "")) for index in range(4))
    if names != EXPLORE_TRACE_NAMES:
        return None

    visible: list[tuple[int, Any, Any, Any]] = []
    for trace_index in (EXPLORE_FILTERED_TRACE_INDEX, EXPLORE_ELIGIBLE_TRACE_INDEX):
        trace = data[trace_index]
        xs = list(trace.get("x") or [])
        ys = list(trace.get("y") or [])
        ids = list(trace.get("customdata") or [])
        hover = list(trace.get("hovertext") or [])
        for position, raw_unit_id in enumerate(ids):
            try:
                unit_id = int(raw_unit_id)
            except (TypeError, ValueError):
                continue
            if position >= len(xs) or position >= len(ys):
                continue
            hover_value = hover[position] if position < len(hover) else ""
            visible.append((unit_id, xs[position], ys[position], hover_value))

    seen = {int(value) for value in seen_unit_ids}
    visited_points = [point for point in visible if point[0] in seen]
    focal_points = (
        [point for point in visible if point[0] == int(focal_unit_id)]
        if focal_unit_id is not None
        else []
    )

    def columns(points: list[tuple[int, Any, Any, Any]]) -> dict[str, list[Any]]:
        return {
            "x": [point[1] for point in points],
            "y": [point[2] for point in points],
            "customdata": [point[0] for point in points],
            "hovertext": [point[3] for point in points],
        }

    return {
        EXPLORE_VISITED_TRACE_INDEX: columns(visited_points),
        EXPLORE_FOCAL_TRACE_INDEX: columns(focal_points),
    }


def _apply_relayout_ranges(
    figure: Any, relayout_data: dict[str, Any] | None
) -> None:
    """Carry the current Plotly camera through point-selection redraws."""
    if not relayout_data:
        return
    x_range = relayout_data.get("xaxis.range")
    y_range = relayout_data.get("yaxis.range")
    x0 = relayout_data.get("xaxis.range[0]")
    x1 = relayout_data.get("xaxis.range[1]")
    y0 = relayout_data.get("yaxis.range[0]")
    y1 = relayout_data.get("yaxis.range[1]")
    if isinstance(x_range, (list, tuple)) and len(x_range) == 2:
        x0, x1 = x_range
    if isinstance(y_range, (list, tuple)) and len(y_range) == 2:
        y0, y1 = y_range
    if x0 is not None and x1 is not None:
        figure.update_xaxes(range=[float(x0), float(x1)], autorange=False)
    if y0 is not None and y1 is not None:
        figure.update_yaxes(range=[float(y0), float(y1)], autorange=False)


def _observation_display_key(
    project: GeometricCoder, observation: dict[str, Any]
) -> str:
    kind = str(observation.get("kind", "atomic"))
    if kind == "atomic":
        return " › ".join(
            f"{name}: {value}"
            for name, value in observation.get("user_key", {}).items()
        )
    if kind == "teaching_example":
        return f"Teaching example {int(observation['observation_id']):,}"
    members = observation.get("member_unit_ids")
    if members is None:
        full = project.observation(int(observation["observation_id"]))
        members = full.get("member_unit_ids", [])
    member_ids = [int(value) for value in members]
    if not member_ids:
        return f"Span {int(observation['observation_id']):,}"
    first = project.unit(member_ids[0])
    last = project.unit(member_ids[-1])
    parent_key = list(first["user_key"].items())[:-1]
    parent = " › ".join(f"{name}: {value}" for name, value in parent_key)
    atomic_name = next(reversed(first["user_key"]))
    first_value = first["user_key"][atomic_name]
    last_value = last["user_key"][atomic_name]
    interval = (
        f"{atomic_name}: {first_value}"
        if first_value == last_value
        else f"{atomic_name}: {first_value}–{last_value}"
    )
    return f"{parent} › {interval}" if parent else f"Span {interval}"


def _prediction_geometry_from_cache(
    cache: dict[str, dict[str, Any]],
    refresh_state: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve one server-side prediction-geometry cache entry."""
    if not refresh_state:
        return {}
    key = str(refresh_state.get("cache_key", ""))
    return cache.get(key, {})


def _prediction_geometry_visible_points(
    geometry: dict[str, Any],
    display_options: list[str] | None,
) -> list[dict[str, Any]]:
    """Filter cached prediction points by user-selected observation kinds."""
    options = set(display_options or [])
    allowed = {"atomic"}
    if "spans" in options:
        allowed.add("span")
    if "teaching" in options:
        allowed.add("teaching_example")
    return [
        point
        for point in geometry.get("points", [])
        if str(point.get("kind")) in allowed
    ]


def _build_prediction_geometry_figure(
    *,
    geometry: dict[str, Any],
    points: list[dict[str, Any]],
    page_index: int,
) -> Any:
    """Build the paged classifier-score map with stable type and label encodings."""
    color_by_assignment = {
        None: "#171b18",
        "positive": "#3f8d5d",
        "negative": "#a83f3f",
        "unsure": "#d39a22",
    }
    label_by_assignment = {
        None: "Unreviewed",
        "positive": "Present",
        "negative": "Absent",
        "unsure": "Unsure",
    }
    symbol_by_kind = {
        "atomic": "circle",
        "span": "triangle-up",
        "teaching_example": "star",
    }
    kind_label = {
        "atomic": "Atomic",
        "span": "Span",
        "teaching_example": "Teaching example",
    }
    figure = go.Figure()

    # Assignment-color legend. The real points are grouped by observation kind so
    # changing a judgment does not create Plotly's persistent selected-point fade.
    for assignment in (None, "positive", "negative", "unsure"):
        figure.add_trace(
            go.Scattergl(
                x=[None],
                y=[None],
                mode="markers",
                name=label_by_assignment[assignment],
                hoverinfo="skip",
                marker={
                    "size": 8,
                    "symbol": "circle",
                    "color": color_by_assignment[assignment],
                },
                showlegend=True,
            )
        )

    for kind in ("atomic", "span", "teaching_example"):
        selected = [point for point in points if str(point.get("kind")) == kind]
        if not selected:
            continue
        colors = [color_by_assignment[point.get("assignment")] for point in selected]
        figure.add_trace(
            go.Scattergl(
                x=[float(point["x"]) for point in selected],
                y=[float(point["y"]) for point in selected],
                mode="markers",
                name=kind_label[kind],
                customdata=[
                    [
                        int(point["observation_id"]),
                        (
                            int(point["unit_id"])
                            if point.get("unit_id") is not None
                            else None
                        ),
                        kind,
                    ]
                    for point in selected
                ],
                hovertext=[
                    str(point.get("display_key") or kind_label[kind])
                    for point in selected
                ],
                hovertemplate="%{hovertext}<extra></extra>",
                marker={
                    "size": 8 if kind == "atomic" else 10,
                    "symbol": symbol_by_kind[kind],
                    "color": colors,
                    "opacity": 0.9,
                    "line": {"color": colors, "width": 0.7},
                },
                showlegend=True,
            )
        )
    figure.update_layout(
        template="plotly_white",
        margin={"l": 55, "r": 20, "t": 30, "b": 75},
        hovermode="closest",
        hoverdistance=8,
        clickmode="event",
        dragmode="zoom",
        uirevision=(
            "prediction-geometry:"
            + ",".join(str(value) for value in geometry.get("classifier_fit_ids", []))
            + f":page:{page_index}"
        ),
        xaxis_title=str(geometry["axis_titles"][0]),
        yaxis_title=str(geometry["axis_titles"][1]),
        legend={"orientation": "h", "y": -0.2},
    )
    return figure


def _metadata_table(html: Any, metadata: dict[str, Any]) -> Any:
    if not metadata:
        return html.P("No metadata columns were supplied.")
    rows = []
    for key, value in metadata.items():
        rows.append(html.Tr([html.Th(str(key)), html.Td(str(value))]))
    return html.Table([html.Tbody(rows)], className="metadata-table")


def _code_center_placeholder_figure() -> Any:
    """Return a Patch-safe initial Code Center figure.

    The focal-marker callback partially updates ``data[0]``. Dash 4.x raises
    ``Cannot patch undefined`` if either ``figure`` or that trace is absent
    when the partial update arrives, so the layout must provide the exact
    nested structure that callback patches.
    """
    figure = go.Figure()
    figure.add_trace(
        go.Scattergl(
            x=[],
            y=[],
            mode="markers",
            name="Focal observation",
            customdata=[],
            hovertext=[],
            hovertemplate="%{hovertext}<extra></extra>",
            marker={
                "size": 13,
                "color": "rgba(0, 0, 0, 0)",
                "line": {"color": "#178547", "width": 2.6},
            },
            showlegend=False,
        )
    )
    figure.update_layout(
        template="plotly_white",
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
    )
    return figure


def _empty_figure(message: str) -> Any:
    import plotly.graph_objects as go

    figure = go.Figure()
    figure.add_annotation(text=message, x=0.5, y=0.5, showarrow=False)
    figure.update_layout(
        template="plotly_white",
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
    )
    return figure


def launch_app(
    project: GeometricCoder,
    *,
    host: str,
    port: int,
    debug: bool,
    use_reloader: bool = False,
) -> None:
    """Create and run the local Dash app with a single-process default."""
    app = create_app(project)
    app.run(
        host=host,
        port=port,
        debug=debug,
        use_reloader=use_reloader,
    )
