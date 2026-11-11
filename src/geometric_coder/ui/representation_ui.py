"""Manage Representations: a compact registry table, detail dialog and shared wizard.

TeAL science and durable registration are delegated to representation_manager.
"""
from __future__ import annotations

import re
from typing import Any

from geometric_coder.ui.representation_manager import (
    create_geometry, create_view, filter_representations,
    geometry_parameter_specs, geometry_recipe_options, linked_representation_options,
    set_representation_analytic_enabled, rename_representation,
    representation_inventory, source_filter_options,
    unregister_representation,
)
from geometric_coder.ui.topic_creation import teal_linked_project
from geometric_coder.ui.text_geometry import (
    TOKEN_FLAGS, TOKEN_FIELDS, available_spacy_token_artifacts, create_text_geometry,
)



def suggest_representation_name(
    source_name: str, method: str, dimensions: int | None,
    existing_names: list[str],
) -> str:
    """Offer a SQL-safe name without renaming or modifying existing artifacts."""
    def slug(value: str) -> str:
        clean = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
        clean = clean[:48].rstrip("_") or "representation"
        return f"representation_{clean}" if clean[0].isdigit() else clean

    stem = f"{slug(source_name)}_{slug(method)}"
    if isinstance(dimensions, int) and not isinstance(dimensions, bool) and dimensions > 0:
        stem += f"_{dimensions}d"
    occupied = {str(name).casefold() for name in existing_names}
    candidate, number = stem, 2
    while candidate.casefold() in occupied:
        candidate = f"{stem}_{number}"
        number += 1
    return candidate


def autofill_representation_name(
    current_name: str | None, previous_suggestion: str | None, suggestion: str,
) -> str | None:
    """Only revise a blank or previously generated name, never a manual edit."""
    if not str(current_name or "").strip() or current_name == previous_suggestion:
        return suggestion
    return None


def _positive_dimensions(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = int(value)
        return number if number > 0 and float(value) == number else None
    except (TypeError, ValueError, OverflowError):
        return None


def representation_modal(project: Any, *, html: Any, dcc: Any, dash_table: Any) -> Any:
    linked = teal_linked_project(project) is not None
    return html.Div([
        dcc.Store(id="representation-inventory-store", data=[]),
        dcc.Store(id="representation-selected", data=None),
        dcc.Store(id="representation-wizard-suggested-name", data=None),
        html.Div([
            html.Div([
                html.Div([
                    html.H2("Manage Representations"),
                    html.Button("×", id="representation-manager-close", n_clicks=0,
                                className="modal-close-icon", title="Close"),
                ], className="modal-header"),
                html.P("Select a representation to manage it.", className="modal-help"),
                html.Div([
                    html.Div([
                        html.Label("Filter dimensions"),
                        html.Div([
                            html.Label("Minimum dimensions", htmlFor="representation-filter-min",
                                       className="sr-only"),
                            dcc.Input(id="representation-filter-min", type="text",
                                      placeholder="Min"),
                            html.Span("≤"),
                            html.Span("D", title="Number of dimensions"),
                            html.Span("≤"),
                            html.Label("Maximum dimensions", htmlFor="representation-filter-max",
                                       className="sr-only"),
                            dcc.Input(id="representation-filter-max", type="text",
                                      placeholder="Max"),
                        ], className="representation-dimension-range"),
                    ], className="representation-dimension-filter"),
                    html.Div([
                        html.Label("Filter by source"),
                        dcc.Dropdown(id="representation-filter-source",
                                     clearable=True, placeholder="All sources"),
                    ], className="representation-source-filter"),
                ], className="representation-filters"),
                dash_table.DataTable(
                    id="representation-table",
                    columns=[
                        {"name": "Name", "id": "name"},
                        {"name": "Dimensions", "id": "dimensions", "type": "numeric"},
                        {"name": "New translations", "id": "new_text"},
                        {"name": "Source", "id": "source"},
                    ],
                    data=[], page_size=12, cell_selectable=True,
                    active_cell=None,
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "left", "padding": "10px"},
                    style_header={"fontWeight": "600"},
                    style_data_conditional=[],
                ),
                html.Div(id="representation-list-status", role="status",
                         className="control-status"),
                html.Div([
                    html.Button("Add Representation", id="representation-add-button",
                                n_clicks=0, className="primary-action",
                                disabled=not linked,
                                title=None if linked else
                                "New TeAL transformations require a linked workspace."),
                ], className="modal-actions"),
            ], className="modal-card topic-manager-card representation-manager-card"),
        ], id="representation-manager-modal", className="modal-overlay",
           style={"display": "none"}),
        html.Div([
            html.Div([
                html.Div([
                    html.H2(id="representation-detail-title"),
                    html.Button("×", id="representation-detail-close", n_clicks=0,
                                className="modal-close-icon", title="Close"),
                ], className="modal-header"),
                html.Label("Rename representation"),
                dcc.Input(id="representation-detail-name", type="text",
                          className="text-input"),
                html.Button("Rename", id="representation-detail-rename", n_clicks=0,
                            className="secondary-action"),
                html.Div(id="representation-detail-rename-status", role="status",
                         className="control-status"),
                html.H3("Visual descendants",
                        id="representation-detail-views-heading"),
                html.Div(id="representation-detail-views"),
                html.Div([
                    html.Button("Add View", id="representation-detail-add-view",
                                n_clicks=0, className="secondary-action"),
                    html.Button("Enable for analysis", id="representation-detail-promote",
                                n_clicks=0, className="secondary-action"),
                    html.Button("Register as Topic Model",
                                id="representation-detail-topic", n_clicks=0,
                                className="secondary-action", disabled=not linked),
                    html.Button("Delete", id="representation-detail-delete",
                                n_clicks=0, className="danger-action"),
                    dcc.ConfirmDialog(id="representation-delete-confirm",
                                      message="Delete this representation from GeCo? Its TeAL artifact will be kept."),
                ], className="representation-detail-actions"),
                html.Div(id="representation-detail-promote-status", role="status",
                         className="control-status"),
                html.Div(id="representation-detail-delete-status", role="status",
                         className="control-status"),
                html.P("Delete unregisters the representation in GeCo only. "
                       "Referenced resources cannot be removed; TeAL artifacts are never deleted.",
                       className="modal-help"),
            ], className="modal-card topic-manager-card"),
        ], id="representation-detail-modal", className="modal-overlay",
           style={"display": "none"}),
        html.Div([
            html.Div([
                html.Div([
                    html.H2("Add Representation"),
                    html.Button("×", id="representation-wizard-close", n_clicks=0,
                                className="modal-close-icon", title="Close"),
                ], className="modal-header"),
                html.Label("Representation type"),
                dcc.RadioItems(
                    id="representation-wizard-mode",
                    options=[{"label": "Analytic geometry", "value": "geometry"},
                             {"label": "View (1D or 2D)", "value": "view"}],
                    value="geometry", inline=True,
                ),
                html.Label("Name"),
                dcc.Input(id="representation-wizard-name", type="text",
                          className="text-input", placeholder="Choose a name"),
                html.Div([
                    html.Label("Start from"),
                    dcc.RadioItems(
                        id="representation-wizard-origin",
                        options=[
                            {"label": "Existing representation", "value": "representation"},
                            {"label": "Original text", "value": "text"},
                        ],
                        value="representation", inline=True,
                    ),
                ], id="representation-wizard-origin-wrap"),
                html.Div([
                    html.Label("Source representation"),
                    dcc.Dropdown(id="representation-wizard-source",
                                 options=linked_representation_options(project),
                                 clearable=False, placeholder="Select source geometry"),
                ], id="representation-wizard-source-wrap"),

                html.Div([
                    html.Label("Display dimensions"),
                    dcc.Dropdown(
                        id="representation-wizard-view-dimensions",
                        options=[{"label": "1 dimension (with jitter)", "value": 1},
                                 {"label": "2 dimensions", "value": 2}],
                        value=2, clearable=False,
                    ),
                ], id="representation-wizard-view-dimensions-wrap",
                   style={"display": "none"}),
                html.Label("Representation method"),
                dcc.Dropdown(id="representation-wizard-recipe",
                             options=geometry_recipe_options(),
                             value="umap", clearable=False),
                html.Div(id="representation-wizard-dimension-note",
                         className="modal-help"),
                html.Div(id="representation-wizard-parameters",
                         className="topic-create-parameters"),
                html.Div([
                    html.Label("Token source"),
                    dcc.Dropdown(
                        id="representation-text-token-source",
                        options=[{"label": "Create new tokens", "value": "new"}],
                        value="new", clearable=False,
                    ),
                    html.P(
                        "Compatible saved token artifacts appear here when available. "
                        "You can always create a new one with a different language model.",
                        className="modal-help",
                    ),
                    html.Div([
                        html.Label("spaCy model and language"),
                        dcc.Input(id="representation-text-spacy-model", type="text",
                                  value="", placeholder="e.g. en_core_web_sm",
                                  className="text-input"),
                        html.P([
                            "Choose a model for the language of your text. ",
                            html.A("Browse spaCy language models",
                                   href="https://spacy.io/models",
                                   target="_blank", rel="noopener noreferrer"),
                        ], className="modal-help"),
                        html.Label("When spaCy produces no tokens"),
                        dcc.Dropdown(
                        id="representation-text-empty-doc",
                        options=[
                            {"label": "Raise an error (default)", "value": "raise"},
                            {"label": "Insert a countable dummy token", "value": "dummy"},
                        ], value="raise", clearable=False,
                        ),
                    ], id="representation-new-tokens-wrap"),
                    html.Label("Token feature"),
                    dcc.Dropdown(
                        id="representation-text-token-field",
                        options=[{"label": x.replace("_", " ").capitalize(), "value": x}
                                 for x in TOKEN_FIELDS],
                        value="text", clearable=False,
                    ),
                    html.Div([
                        html.Div([
                            html.Label("Minimum n-gram length"),
                            dcc.Input(id="representation-text-ngram-min", type="number",
                                      min=1, max=5, step=1, value=1),
                        ]),
                        html.Div([
                            html.Label("Maximum n-gram length"),
                            dcc.Input(id="representation-text-ngram-max", type="number",
                                      min=1, max=5, step=1, value=1),
                        ]),
                        html.Div([
                            html.Label("Minimum document frequency"),
                            dcc.Input(id="representation-text-min-df", type="number",
                                      min=1, step=1, value=1),
                        ]),
                        html.Div([
                            html.Label("Maximum features (optional)"),
                            dcc.Input(id="representation-text-max-features", type="number",
                                      min=1, step=1, value=None),
                        ]),
                    ], className="topic-create-parameters"),
                    dcc.Checklist(
                        id="representation-text-binary",
                        options=[{"label": "Binary presence instead of token counts",
                                  "value": "binary"}],
                        value=[],
                    ),
                    html.Details([
                        html.Summary("Token attributes"),
                        html.P("Select which token properties to keep. All attributes are ignored by default.",
                               className="modal-help"),
                        html.Div([
                            html.Div([
                                html.Span("Attribute"),
                                html.Span("True"),
                                html.Span("False"),
                                html.Span("Any"),
                            ], className="representation-flag-header"),
                            *[
                                html.Div([
                                    html.Span(flag.replace("_", " "), className="representation-flag-name"),
                                    dcc.RadioItems(
                                        id={"type": "representation-token-flag", "name": flag},
                                        options=[
                                            {"label": "True", "value": "true"},
                                            {"label": "False", "value": "false"},
                                            {"label": "Any", "value": "any"},
                                        ],
                                        value="any", inline=True,
                                        className="representation-flag-choice",
                                        inputStyle={"margin": "0"},
                                    ),
                                ], className="representation-flag-row")
                                for flag in TOKEN_FLAGS
                            ],
                        ], className="representation-token-grid"),
                    ], className="representation-token-details"),
                ], id="representation-wizard-spacy-wrap", style={"display": "none"}),
                html.Div([
                    html.Label("Hugging Face model ID or model-page URL"),
                    dcc.Input(
                        id="representation-text-model", type="text",
                        placeholder="sentence-transformers/all-MiniLM-L6-v2",
                        className="text-input",
                    ),
                    html.P([
                        "Choose a model that supports sentence embeddings. ",
                        html.A("Browse Hugging Face Sentence Transformer models",
                               href="https://huggingface.co/models?library=sentence-transformers",
                               target="_blank", rel="noopener noreferrer"),
                    ], className="modal-help"),
                    html.Label("Revision (optional)"),
                    dcc.Input(id="representation-text-revision", type="text",
                              placeholder="Commit hash or tag", className="text-input"),
                    dcc.Checklist(
                        id="representation-text-normalize",
                        options=[{"label": "Normalize embeddings", "value": "normalize"}],
                        value=[],
                    ),
                    html.Label("Overlong texts"),
                    dcc.RadioItems(
                        id="representation-text-truncation",
                        options=[
                            {"label": "Stop with an error", "value": "error"},
                            {"label": "Truncate to model context window", "value": "truncate"},
                        ], value="error",
                    ),
                    html.P(
                        "Models may need to download. Runtime, memory and disk use depend "
                        "on the selected model and corpus.",
                        className="modal-help",
                    ),
                ], id="representation-wizard-transformer-wrap",
                   style={"display": "none"}),
                html.Details([
                    html.Summary("Advanced parameters"),
                    html.Div(id="representation-wizard-advanced",
                             className="topic-create-parameters"),
                ], id="representation-wizard-advanced-wrap",
                   style={"display": "none"}),

                html.Div([
                    html.Button("Cancel", id="representation-wizard-cancel",
                                n_clicks=0),
                    html.Button("Create", id="representation-wizard-submit",
                                n_clicks=0, className="primary-action",
                                disabled=not linked),
                ], className="modal-actions"),
                html.Div(id="representation-wizard-status", role="status",
                         className="control-status"),
            ], className="modal-card topic-manager-card representation-wizard-card"),
        ], id="representation-wizard-modal", className="modal-overlay",
           style={"display": "none"}),
    ])


def _parameter_controls(html: Any, dcc: Any, specs: list[dict[str, Any]]) -> list[Any]:
    fields = []
    for spec in specs:
        component_id = {"type": "representation-param", "name": spec["name"]}
        if spec["kind"] == "boolean":
            field = dcc.Dropdown(
                id=component_id,
                options=[{"label": "True", "value": "true"},
                         {"label": "False", "value": "false"}],
                value="true" if spec["default"] else "false", clearable=False,
            )
        elif spec["kind"] == "choice":
            field = dcc.Dropdown(id=component_id, options=spec["options"],
                                 value=spec["default"], clearable=False)
        else:
            field = dcc.Input(
                id=component_id, value=spec["default"],
                type="number" if spec["kind"] in {"integer", "number"} else "text",
                step=1 if spec["kind"] == "integer" else "any",
            )
        fields.append(html.Div([html.Label(spec["label"]), field],
                               className="topic-parameter-field"))
    return fields


def register_representation_callbacks(
    app: Any, project: Any, *, html: Any, dcc: Any, Input: Any,
    Output: Any, State: Any, ALL: Any, callback_context: Any,
    no_update: Any, geometries: list[dict[str, Any]],
    geometry_by_id: dict[int, dict[str, Any]],
) -> None:
    def refresh_geometry_cache() -> None:
        current = project.geometries(public_only=True)
        geometries[:] = current
        geometry_by_id.clear()
        geometry_by_id.update({int(g["geometry_id"]): g for g in current})

    @app.callback(
        Output("representation-manager-modal", "style"),
        Input("representation-manager-open", "n_clicks"),
        Input("representation-manager-close", "n_clicks"),
        prevent_initial_call=True,
    )
    def toggle_manager(_open: int, _close: int) -> dict[str, str]:
        return {"display": "flex"} if callback_context.triggered_id == "representation-manager-open" else {"display": "none"}

    @app.callback(
        Output("representation-inventory-store", "data"),
        Output("representation-filter-source", "options"),
        Output("representation-wizard-source", "options"),
        Input("representation-manager-open", "n_clicks"),
        Input("representation-refresh-store", "data"),
    )
    def refresh_inventory(_open: int, _revision: int) -> tuple[Any, Any, Any]:
        inventory = representation_inventory(project)
        return inventory, source_filter_options(inventory), linked_representation_options(project)

    @app.callback(
        Output("representation-table", "data"),
        Output("representation-list-status", "children"),
        Input("representation-inventory-store", "data"),
        Input("representation-filter-min", "value"),
        Input("representation-filter-max", "value"),
        Input("representation-filter-source", "value"),
    )
    def list_representations(
        inventory: list[dict[str, Any]] | None, minimum: Any, maximum: Any,
        source: str | None,
    ) -> tuple[list[dict[str, Any]], str]:
        def bound(value: Any) -> int | None:
            if value is None or value == "":
                return None
            numeric = float(value)
            if not numeric.is_integer():
                raise ValueError("Dimension filters require whole numbers.")
            return int(numeric)

        try:
            rows = filter_representations(
                inventory or [], min_dimensions=bound(minimum),
                max_dimensions=bound(maximum), source=source,
            )
        except (ValueError, TypeError) as exc:
            return [], str(exc)
        return [
            {
                "key": row["key"], "name": row["name"],
                "dimensions": row["dimensions"],
                "new_text": "✓" if row["new_text"] else "✕",
                "source": row["source"],
            } for row in rows
        ], f"{len(rows)} representations. Click any row to manage it."

    @app.callback(
        Output("representation-detail-modal", "style"),
        Output("representation-selected", "data"),
        Output("representation-table", "active_cell"),
        Input("representation-table", "active_cell"),
        Input("representation-detail-close", "n_clicks"),
        Input("representation-detail-delete-status", "children"),
        Input("representation-detail-promote-status", "children"),
        State("representation-table", "data"),
        prevent_initial_call=True,
    )
    def open_detail(active_cell: Any, _close: int, delete_status: str | None,
                    promote_status: str | None, rows: list[dict[str, Any]] | None) -> tuple[Any, Any, Any]:
        trigger = callback_context.triggered_id
        if trigger == "representation-detail-close":
            return {"display": "none"}, None, None
        if trigger in {"representation-detail-delete-status", "representation-detail-promote-status"}:
            if (trigger == "representation-detail-delete-status" and delete_status == "Deleted.") or (
                trigger == "representation-detail-promote-status" and promote_status == "Updated."
            ):
                return {"display": "none"}, None, None
            return no_update, no_update, no_update
        if not active_cell or not rows:
            return no_update, no_update, no_update
        position = active_cell.get("row")
        if position is None or position >= len(rows):
            return no_update, no_update, None
        return {"display": "flex"}, rows[position]["key"], None

    @app.callback(
        Output("representation-detail-rename-status", "children", allow_duplicate=True),
        Output("representation-detail-promote-status", "children", allow_duplicate=True),
        Output("representation-detail-delete-status", "children", allow_duplicate=True),
        Input("representation-selected", "data"),
        prevent_initial_call=True,
    )
    def clear_detail_notices(_selection: str | None) -> tuple[str, str, str]:
        return "", "", ""

    @app.callback(
        Output("representation-wizard-status", "children", allow_duplicate=True),
        Input("representation-wizard-mode", "value"),
        Input("representation-wizard-recipe", "value"),
        Input("representation-wizard-source", "value"),
        Input("representation-wizard-name", "value"),
        prevent_initial_call=True,
    )
    def clear_wizard_notice(_mode: Any, _recipe: Any, _source: Any, _name: Any) -> str:
        return ""

    @app.callback(
        Output("representation-detail-title", "children"),
        Output("representation-detail-name", "value"),
        Output("representation-detail-views", "children"),
        Output("representation-detail-views-heading", "style"),
        Output("representation-detail-add-view", "style"),
        Output("representation-detail-topic", "style"),
        Output("representation-detail-promote", "style"),
        Output("representation-detail-promote", "children"),
        Input("representation-selected", "data"),
        Input("representation-inventory-store", "data"),
    )
    def populate_detail(
        key: str | None, inventory: list[dict[str, Any]] | None,
    ) -> tuple[Any, ...]:
        matches = [r for r in (inventory or []) if r["key"] == key]
        hidden = {"display": "none"}
        if not matches:
            return "", "", "", hidden, hidden, hidden, hidden, ""
        row = matches[0]
        descendants = []
        try:
            candidates = project.database.eligible_view_representations(
                int(row["representation_id"]), dimensions=(1, 2)
            )
            names = {int(item["representation_id"]): item["name"]
                     for item in inventory or []}
            descendants = [
                html.Div(names.get(int(item["representation_id"]), str(item["name"])))
                for item in candidates
            ]
        except (KeyError, ValueError):
            pass
        can_add_view = (
            teal_linked_project(project) is not None and bool(row["artifact_id"])
        )
        can_toggle = row["dimensions"] in (1, 2) and (
            not row["analytic_enabled"] or row["parent_representation_id"] is not None
        )
        return (
            row["name"], row["name"],
            descendants or ["No compatible 1D or 2D descendants yet."],
            {}, {} if can_add_view else hidden,
            {} if row["analytic_enabled"] else hidden,
            {} if can_toggle else hidden,
            "Disable for analysis" if row["analytic_enabled"] else "Enable for analysis",
        )

    @app.callback(
        Output("representation-detail-rename-status", "children"),
        Output("representation-refresh-store", "data", allow_duplicate=True),
        Input("representation-detail-rename", "n_clicks"),
        State("representation-selected", "data"),
        State("representation-detail-name", "value"),
        State("representation-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def rename(_clicks: int, key: str | None, name: str | None, revision: int | None) -> tuple[Any, Any]:
        if key is None:
            return "Choose a representation.", no_update
        try:
            rename_representation(project, key=key, name=name or "")
        except Exception as exc:
            return f"Rename failed: {exc}", no_update
        refresh_geometry_cache()
        return "Renamed.", int(revision or 0) + 1

    @app.callback(
        Output("representation-detail-promote-status", "children"),
        Output("representation-refresh-store", "data", allow_duplicate=True),
        Output("view-refresh-store", "data", allow_duplicate=True),
        Input("representation-detail-promote", "n_clicks"),
        State("representation-selected", "data"),
        State("representation-inventory-store", "data"),
        State("representation-refresh-store", "data"),
        State("view-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def promote_view(
        _clicks: int, key: str | None,
        inventory: list[dict[str, Any]] | None,
        revision: int | None, view_revision: int | None,
    ) -> tuple[Any, Any, Any]:
        matches = [r for r in inventory or [] if r["key"] == key]
        if not matches:
            return "Choose a representation.", no_update, no_update
        record = matches[0]
        try:
            set_representation_analytic_enabled(
                project, representation_id=int(record["representation_id"]),
                enabled=not bool(record["analytic_enabled"]),
            )
        except Exception as exc:
            return f"Analytic eligibility unchanged: {exc}", no_update, no_update
        refresh_geometry_cache()
        return "Updated.", int(revision or 0) + 1, int(view_revision or 0) + 1

    @app.callback(
        Output("representation-delete-confirm", "displayed"),
        Input("representation-detail-delete", "n_clicks"),
        prevent_initial_call=True,
    )
    def request_representation_delete(clicks: int | None) -> bool:
        return bool(clicks)

    @app.callback(
        Output("representation-detail-delete-status", "children"),
        Output("representation-refresh-store", "data", allow_duplicate=True),
        Input("representation-delete-confirm", "submit_n_clicks"),
        State("representation-selected", "data"),
        State("representation-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def unregister(_clicks: int, key: str | None, revision: int | None) -> tuple[Any, Any]:
        if not _clicks:
            return no_update, no_update
        if key is None:
            return "Choose a representation.", no_update
        try:
            unregister_representation(project, key=key)
        except Exception as exc:
            return f"Cannot delete: {exc}", no_update
        refresh_geometry_cache()
        return "Deleted.", int(revision or 0) + 1

    @app.callback(
        Output("topic-create-modal", "style", allow_duplicate=True),
        Output("topic-create-geometry", "value", allow_duplicate=True),
        Output("topic-create-algorithm", "value", allow_duplicate=True),
        Output("topic-create-name", "value", allow_duplicate=True),
        Input("representation-detail-topic", "n_clicks"),
        State("representation-selected", "data"),
        State("representation-inventory-store", "data"),
        prevent_initial_call=True,
    )
    def register_topic(_clicks: int, key: str | None, inventory: list[dict[str, Any]] | None) -> tuple[Any, ...]:
        rows = [r for r in inventory or []
                if r["key"] == key and r["analytic_enabled"]]
        if not rows:
            return (no_update,) * 4
        row = rows[0]
        record = project.database.get_representation(int(row["representation_id"]))
        if record["legacy_kind"] != "geometry":
            return (no_update,) * 4
        return ({"display": "flex"}, int(record["legacy_id"]), "identity",
                f"{row['name']}_topics")

    @app.callback(
        Output("representation-wizard-modal", "style"),
        Output("representation-wizard-mode", "value"),
        Output("representation-wizard-source", "value"),
        Output("representation-wizard-recipe", "value"),
        Output("representation-wizard-name", "value"),
        Output("representation-wizard-origin", "value"),
        Input("representation-add-button", "n_clicks"),
        Input("representation-detail-add-view", "n_clicks"),
        Input("representation-wizard-close", "n_clicks"),
        Input("representation-wizard-cancel", "n_clicks"),
        Input("representation-wizard-status", "children"),
        State("representation-selected", "data"),
        State("representation-inventory-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_wizard(
        _add: int, _add_view: int, _close: int, _cancel: int, status: str | None,
        selected: str | None, inventory: list[dict[str, Any]] | None,
    ) -> tuple[Any, ...]:
        trigger = callback_context.triggered_id
        if trigger == "representation-wizard-status":
            return ({"display": "none"}, no_update, no_update, no_update, no_update,
                    no_update) if (
                status == "Created."
            ) else (no_update,) * 6
        if trigger in {"representation-wizard-close", "representation-wizard-cancel"}:
            return {"display": "none"}, no_update, no_update, no_update, no_update, no_update
        if trigger == "representation-detail-add-view":
            matches = [r for r in inventory or []
                       if r["key"] == selected and r.get("artifact_id")]
            if not matches:
                return (no_update,) * 6
            return (
                {"display": "flex"}, "view",
                int(matches[0]["representation_id"]), "umap", "", "representation",
            )
        return {"display": "flex"}, "geometry", None, "umap", "", "representation"

    @app.callback(
        Output("representation-wizard-recipe", "options"),
        Output("representation-wizard-dimension-note", "children"),
        Output("representation-wizard-view-dimensions-wrap", "style"),
        Output("representation-wizard-origin-wrap", "style"),
        Output("representation-wizard-source-wrap", "style"),
        Output("representation-wizard-recipe", "value", allow_duplicate=True),
        Input("representation-wizard-mode", "value"),
        Input("representation-wizard-origin", "value"),
        prevent_initial_call=True,
    )
    def wizard_mode(
        mode: str | None, origin: str | None,
    ) -> tuple[Any, str, dict[str, str], dict[str, str], dict[str, str], str]:
        if mode == "view":
            return [
                {"label": "UMAP", "value": "umap"},
                {"label": "SVD", "value": "svd"},
            ], "Views support 1D (jittered) or 2D coordinates.", {}, {"display": "none"}, {}, "umap"
        if origin == "text":
            return (
                [
                    {"label": "Document-term matrix", "value": "spacy_dtm"},
                    {"label": "Sentence Transformer embeddings",
                     "value": "sentence_transformer"},
                ],
                "Build a geometry from the linked original text.",
                {"display": "none"}, {}, {"display": "none"}, "spacy_dtm",
            )
        return (
            geometry_recipe_options(),
            "Replayability is detected automatically from the fitted TeAL pipeline.",
            {"display": "none"}, {}, {}, "umap",
        )

    @app.callback(
        Output("representation-wizard-name", "value", allow_duplicate=True),
        Output("representation-wizard-suggested-name", "data"),
        Input("representation-wizard-modal", "style"),
        Input("representation-wizard-mode", "value"),
        Input("representation-wizard-origin", "value"),
        Input("representation-wizard-source", "value"),
        Input("representation-wizard-recipe", "value"),
        Input("representation-wizard-view-dimensions", "value"),
        Input({"type": "representation-param", "name": ALL}, "value"),
        Input("representation-inventory-store", "data"),
        State({"type": "representation-param", "name": ALL}, "id"),
        State("representation-wizard-name", "value"),
        State("representation-wizard-suggested-name", "data"),
        prevent_initial_call=True,
    )
    def suggest_wizard_name(
        modal_style: dict[str, str] | None, mode: str | None, origin: str | None,
        source_id: int | None, recipe: str | None, view_dimensions: int | None,
        param_values: list[Any], inventory: list[dict[str, Any]] | None,
        param_ids: list[dict[str, str]], current_name: str | None,
        previous_suggestion: str | None,
    ) -> tuple[Any, Any]:
        if not modal_style or modal_style.get("display") == "none" or not recipe:
            return no_update, no_update
        if mode == "view":
            if recipe not in {"umap", "svd"}:
                return no_update, no_update
        elif mode == "geometry":
            allowed = ({"spacy_dtm", "sentence_transformer"} if origin == "text"
                       else {"tfidf", "normalize", "svd", "umap"})
            if recipe not in allowed:
                return no_update, no_update
        else:
            return no_update, no_update

        rows = inventory or []
        if mode == "geometry" and origin == "text":
            source_name, dimensions = "original_text", None
        else:
            source = next(
                (row for row in rows
                 if str(row.get("representation_id")) == str(source_id)), None,
            )
            if source is None:
                return no_update, no_update
            source_name = str(source["name"])
            if mode == "view":
                dimensions = _positive_dimensions(view_dimensions)
            elif recipe in {"svd", "umap"}:
                controls = {
                    item.get("name"): value
                    for item, value in zip(param_ids or [], param_values or [])
                }
                dimensions = _positive_dimensions(controls.get("n_components"))
            else:
                dimensions = _positive_dimensions(source.get("dimensions"))

        existing = [str(row["name"]) for row in rows]
        # Also include registrations hidden from the public inventory.
        existing.extend(str(row["name"]) for row in project.database.list_representations())
        proposed = suggest_representation_name(source_name, recipe, dimensions, existing)
        autofilled = autofill_representation_name(
            current_name, previous_suggestion, proposed,
        )
        return (autofilled if autofilled is not None else no_update), proposed

    @app.callback(
        Output("representation-wizard-parameters", "children"),
        Output("representation-wizard-advanced", "children"),
        Output("representation-wizard-advanced-wrap", "style"),
        Input("representation-wizard-mode", "value"),
        Input("representation-wizard-recipe", "value"),
    )
    def wizard_parameters(mode: str | None, recipe: str | None) -> tuple[Any, Any, Any]:
        if not recipe:
            return [], [], {"display": "none"}
        try:
            common, advanced = geometry_parameter_specs(recipe)
        except (ImportError, ValueError):
            return [], [], {"display": "none"}
        if mode == "view":
            common = [s for s in common if s["name"] != "n_components"]
            advanced = [s for s in advanced if s["name"] != "n_components"]
        return (
            _parameter_controls(html, dcc, common),
            _parameter_controls(html, dcc, advanced),
            {"display": "block"} if advanced else {"display": "none"},
        )

    @app.callback(
        Output("representation-wizard-spacy-wrap", "style"),
        Output("representation-wizard-transformer-wrap", "style"),
        Input("representation-wizard-mode", "value"),
        Input("representation-wizard-origin", "value"),
        Input("representation-wizard-recipe", "value"),
    )
    def text_recipe_panels(
        mode: str | None, origin: str | None, recipe: str | None,
    ) -> tuple[dict[str, str], dict[str, str]]:
        if mode != "geometry" or origin != "text":
            return {"display": "none"}, {"display": "none"}
        return (
            {} if recipe == "spacy_dtm" else {"display": "none"},
            {} if recipe == "sentence_transformer" else {"display": "none"},
        )

    @app.callback(
        Output("representation-text-token-source", "options"),
        Input("representation-wizard-origin", "value"),
        Input("representation-wizard-recipe", "value"),
        Input("representation-wizard-modal", "style"),
    )
    def discover_tokens(origin: str | None, recipe: str | None,
                        modal_style: Any) -> list[dict[str, str]]:
        options = [{"label": "Create new tokens", "value": "new"}]
        if origin == "text" and recipe == "spacy_dtm" and modal_style != {"display": "none"}:
            try:
                options.extend(available_spacy_token_artifacts(project))
            except Exception:
                # Creation does its own validation, so discovery should not
                # prevent the user from creating a fresh token artifact.
                pass
        return options

    @app.callback(
        Output("representation-new-tokens-wrap", "style"),
        Input("representation-text-token-source", "value"),
    )
    def token_source_settings(selection: str | None) -> dict[str, str]:
        return {} if selection in (None, "new") else {"display": "none"}

    @app.callback(
        Output("representation-refresh-store", "data", allow_duplicate=True),
        Output("view-refresh-store", "data", allow_duplicate=True),
        Output("representation-wizard-status", "children"),
        Input("representation-wizard-submit", "n_clicks"),
        State("representation-wizard-mode", "value"),
        State("representation-wizard-name", "value"),
        State("representation-wizard-source", "value"),
        State("representation-wizard-recipe", "value"),
        State("representation-wizard-origin", "value"),
        State("representation-wizard-view-dimensions", "value"),
        State("representation-text-spacy-model", "value"),
        State("representation-text-token-source", "value"),
        State("representation-text-empty-doc", "value"),
        State("representation-text-token-field", "value"),
        State("representation-text-ngram-min", "value"),
        State("representation-text-ngram-max", "value"),
        State("representation-text-min-df", "value"),
        State("representation-text-max-features", "value"),
        State("representation-text-binary", "value"),
        State({"type": "representation-token-flag", "name": ALL}, "id"),
        State({"type": "representation-token-flag", "name": ALL}, "value"),
        State("representation-text-model", "value"),
        State("representation-text-revision", "value"),
        State("representation-text-normalize", "value"),
        State("representation-text-truncation", "value"),
        State({"type": "representation-param", "name": ALL}, "value"),
        State({"type": "representation-param", "name": ALL}, "id"),
        State("representation-refresh-store", "data"),
        State("view-refresh-store", "data"),
        prevent_initial_call=True,
        running=[(Output("representation-wizard-submit", "disabled"), True, False)],
    )
    def create(
        _clicks: int, mode: str | None, name: str | None, source: int | None,
        recipe: str | None, origin: str | None, view_dimensions: int | None,
        spacy_model: str | None, token_source: str | None,
        empty_doc: str | None,
        token_field: str | None,
        ngram_min: int | None, ngram_max: int | None,
        min_df: int | None, max_features: int | None, binary: list[str] | None,
        flag_ids: list[dict[str, str]], flag_values: list[str],
        transformer_model: str | None, model_revision: str | None,
        normalize: list[str] | None, truncation: str | None,
        values: list[Any], ids: list[dict[str, str]],
        revision: int | None,
        view_revision: int | None,
    ) -> tuple[Any, Any, str]:
        if (mode == "view" or origin != "text") and source is None:
            return no_update, no_update, "Choose a source representation."
        params = {item["name"]: value for item, value in zip(ids, values, strict=True)}
        try:
            if mode == "geometry" and origin == "text":
                flags = {
                    item["name"]: value
                    for item, value in zip(flag_ids, flag_values, strict=True)
                }
                created = create_text_geometry(
                    project, name=name or "", method=recipe or "",
                    model=(transformer_model if recipe == "sentence_transformer"
                           else spacy_model),
                    revision=model_revision or None, token_field=token_field or "text",
                    token_flags=flags,
                    existing_tokens=(token_source if token_source not in (None, "new")
                                     else None),
                    empty_doc=empty_doc or "raise",
                    ngram_min=int(ngram_min or 1),
                    ngram_max=int(ngram_max or 1),
                    min_df=int(min_df or 1), max_features=(
                        int(max_features) if max_features is not None else None
                    ),
                    binary="binary" in (binary or []),
                    normalize="normalize" in (normalize or []),
                    truncation=truncation or "error",
                )
                result_id = created.geometry_id
                label = "geometry"
            elif mode == "view":
                result_id = create_view(
                    project, name=name or "", source_representation_id=int(source),
                    method=recipe or "umap", dimensions=int(view_dimensions or 2),
                    parameters=params,
                )
                label = "2D view"
            else:
                created = create_geometry(
                    project, name=name or "", source_representation_id=int(source),
                    recipe=recipe or "umap", parameters=params,
                )
                result_id = created.geometry_id
                label = "geometry"
        except Exception as exc:
            return no_update, no_update, f"Creation failed: {exc}"
        refresh_geometry_cache()
        return (
            int(revision or 0) + 1,
            int(view_revision or 0) + 1,
            "Created.",
        )

    @app.callback(
        Output("geometry-dropdown", "options"),
        Output("topic-create-geometry", "options"),
        Input("representation-refresh-store", "data"),
    )
    def geometry_options(_revision: int) -> tuple[Any, Any]:
        options = [
            {"label": row["name"], "value": int(row["geometry_id"])}
            for row in project.geometries(public_only=True)
        ]
        return options, options
