"""Manage Representations: a compact registry table, detail dialog and shared wizard.

TeAL science and durable registration are delegated to representation_manager.
"""
from __future__ import annotations

from typing import Any

from geometric_coder.ui.representation_manager import (
    create_geometry, create_view, filter_representations,
    geometry_parameter_specs, geometry_recipe_options, linked_geometry_options,
    register_view_as_geometry, rename_representation, representation_inventory, source_filter_options,
    unregister_representation,
)
from geometric_coder.ui.topic_creation import teal_linked_project


def representation_modal(project: Any, *, html: Any, dcc: Any, dash_table: Any) -> Any:
    linked = teal_linked_project(project) is not None
    return html.Div([
        dcc.Store(id="representation-inventory-store", data=[]),
        dcc.Store(id="representation-selected", data=None),
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
                            dcc.Input(id="representation-filter-min", type="number",
                                      min=1, step=1, placeholder="Min"),
                            html.Span("≤"),
                            html.Span("D", title="Number of dimensions"),
                            html.Span("≤"),
                            html.Label("Maximum dimensions", htmlFor="representation-filter-max",
                                       className="sr-only"),
                            dcc.Input(id="representation-filter-max", type="number",
                                      min=1, step=1, placeholder="Max"),
                        ], className="representation-dimension-range"),
                    ], className="representation-dimension-filter"),
                    html.Div([
                        html.Label("Filter by source"),
                        dcc.Dropdown(id="representation-filter-source",
                                     clearable=True, placeholder="All sources"),
                    ], className="representation-source-filter"),
                    html.Div([
                        html.Label("Filter by role"),
                        dcc.Dropdown(id="representation-filter-role", clearable=True,
                            placeholder="All roles", options=[
                                {"label": "Geometry", "value": "geometry"},
                                {"label": "2D view", "value": "view"},
                                {"label": "Geometry / 2D view", "value": "both"},
                            ]),
                    ], className="representation-role-filter"),
                ], className="representation-filters"),
                dash_table.DataTable(
                    id="representation-table",
                    columns=[
                        {"name": "Name", "id": "name"},
                        {"name": "Dimensions", "id": "dimensions", "type": "numeric"},
                        {"name": "New translations", "id": "new_text"},
                        {"name": "Role", "id": "role"},
                    ],
                    data=[], page_size=12, cell_selectable=True,
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "left", "padding": "10px"},
                    style_header={"fontWeight": "600"},
                    style_data_conditional=[{
                        "if": {"state": "active"},
                        "backgroundColor": "#e7f1ea",
                        "border": "1px solid #2f744a",
                    }],
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
                html.H3("2D views using this geometry",
                        id="representation-detail-views-heading"),
                html.Div(id="representation-detail-views"),
                html.Div([
                    html.Button("Add 2D View", id="representation-detail-add-view",
                                n_clicks=0, className="secondary-action"),
                    html.Button("Register as Geometry", id="representation-detail-promote",
                                n_clicks=0, className="secondary-action"),
                    html.Button("Register as Topic Model",
                                id="representation-detail-topic", n_clicks=0,
                                className="secondary-action", disabled=not linked),
                    html.Button("Delete", id="representation-detail-delete",
                                n_clicks=0, className="danger-action"),
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
                             {"label": "2D view", "value": "view"}],
                    value="geometry", inline=True,
                ),
                html.Label("Name"),
                dcc.Input(id="representation-wizard-name", type="text",
                          className="text-input", placeholder="Choose a name"),
                html.Label("Source"),
                dcc.Dropdown(id="representation-wizard-source",
                             options=linked_geometry_options(project),
                             clearable=False, placeholder="Select source geometry"),
                html.Label("Transformation"),
                dcc.Dropdown(id="representation-wizard-recipe",
                             options=geometry_recipe_options(),
                             value="umap", clearable=False),
                html.Div(id="representation-wizard-dimension-note",
                         className="modal-help"),
                html.Div(id="representation-wizard-parameters",
                         className="topic-create-parameters"),
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
            ], className="modal-card topic-manager-card"),
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
        return inventory, source_filter_options(inventory), linked_geometry_options(project)

    @app.callback(
        Output("representation-table", "data"),
        Output("representation-list-status", "children"),
        Input("representation-inventory-store", "data"),
        Input("representation-filter-min", "value"),
        Input("representation-filter-max", "value"),
        Input("representation-filter-source", "value"),
        Input("representation-filter-role", "value"),
    )
    def list_representations(
        inventory: list[dict[str, Any]] | None, minimum: Any, maximum: Any,
        source: str | None, role: str | None,
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
                max_dimensions=bound(maximum), source=source, role=role,
            )
        except (ValueError, TypeError) as exc:
            return [], str(exc)
        return [
            {
                "key": row["key"], "name": row["name"],
                "dimensions": row["dimensions"],
                "new_text": "✓" if row["new_text"] else "✕",
                "role": {"geometry": "Geometry", "view": "2D view",
                         "both": "Geometry / 2D view"}[row["kind"]],
            } for row in rows
        ], f"{len(rows)} representations. Click any row to manage it."

    @app.callback(
        Output("representation-detail-modal", "style"),
        Output("representation-selected", "data"),
        Input("representation-table", "active_cell"),
        Input("representation-detail-close", "n_clicks"),
        Input("representation-detail-delete-status", "children"),
        Input("representation-detail-promote-status", "children"),
        State("representation-table", "data"),
        prevent_initial_call=True,
    )
    def open_detail(active_cell: Any, _close: int, delete_status: str | None,
                    promote_status: str | None, rows: list[dict[str, Any]] | None) -> tuple[Any, Any]:
        trigger = callback_context.triggered_id
        if trigger == "representation-detail-close":
            return {"display": "none"}, None
        if trigger in {"representation-detail-delete-status", "representation-detail-promote-status"}:
            if (trigger == "representation-detail-delete-status" and delete_status == "Deleted.") or (
                trigger == "representation-detail-promote-status" and promote_status == "Registered."
            ):
                return {"display": "none"}, None
            return no_update, no_update
        if not active_cell or not rows:
            return no_update, no_update
        position = active_cell.get("row")
        if position is None or position >= len(rows):
            return no_update, no_update
        return {"display": "flex"}, rows[position]["key"]

    @app.callback(
        Output("representation-detail-title", "children"),
        Output("representation-detail-name", "value"),
        Output("representation-detail-views", "children"),
        Output("representation-detail-views-heading", "style"),
        Output("representation-detail-add-view", "style"),
        Output("representation-detail-topic", "style"),
        Output("representation-detail-promote", "style"),
        Input("representation-selected", "data"),
        Input("representation-inventory-store", "data"),
    )
    def populate_detail(key: str | None, inventory: list[dict[str, Any]] | None) -> tuple[Any, ...]:
        matches = [r for r in (inventory or []) if r["key"] == key]
        if not matches:
            return "", "", "", {"display": "none"}, {"display": "none"}, {"display": "none"}, {"display": "none"}
        row = matches[0]
        if row["kind"] in {"geometry", "both"}:
            views = [
                html.Div(child["name"]) for child in inventory or []
                if child["kind"] == "view"
                and child["geometry_id"] == row["geometry_id"]
            ]
        else:
            views = []
        add_view = (row["kind"] in {"geometry", "both"}
                    and row.get("dimensions") is not None
                    and row["dimensions"] > 2)
        return (
            row["name"], row["name"],
            views or ["No 2D views registered."],
            {} if row["kind"] in {"geometry", "both"} else {"display": "none"},
            {} if add_view else {"display": "none"},
            {} if row["kind"] in {"geometry", "both"} else {"display": "none"},
            {} if row["kind"] == "view" and row.get("storage_kind") == "external"
            else {"display": "none"},
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
    def promote_view(_clicks: int, key: str | None,
                     inventory: list[dict[str, Any]] | None,
                     revision: int | None, view_revision: int | None) -> tuple[Any, Any, Any]:
        rows = [r for r in inventory or [] if r["key"] == key and r["kind"] == "view"]
        if not rows:
            return "Choose a 2D view.", no_update, no_update
        try:
            register_view_as_geometry(project, view_id=int(rows[0]["view_id"]))
        except Exception as exc:
            return f"Registration failed: {exc}", no_update, no_update
        refresh_geometry_cache()
        return "Registered.", int(revision or 0) + 1, int(view_revision or 0) + 1

    @app.callback(
        Output("representation-detail-delete-status", "children"),
        Output("representation-refresh-store", "data", allow_duplicate=True),
        Input("representation-detail-delete", "n_clicks"),
        State("representation-selected", "data"),
        State("representation-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def unregister(_clicks: int, key: str | None, revision: int | None) -> tuple[Any, Any]:
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
        rows = [r for r in inventory or [] if r["key"] == key and r["kind"] in {"geometry", "both"}]
        if not rows:
            return (no_update,) * 4
        row = rows[0]
        return {"display": "flex"}, int(row["geometry_id"]), "identity", f"{row['name']}_topics"

    @app.callback(
        Output("representation-wizard-modal", "style"),
        Output("representation-wizard-mode", "value"),
        Output("representation-wizard-source", "value"),
        Output("representation-wizard-recipe", "value"),
        Output("representation-wizard-name", "value"),
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
            return ({"display": "none"}, no_update, no_update, no_update, no_update) if (
                status == "Created."
            ) else (no_update,) * 5
        if trigger in {"representation-wizard-close", "representation-wizard-cancel"}:
            return {"display": "none"}, no_update, no_update, no_update, no_update
        if trigger == "representation-detail-add-view":
            matches = [r for r in inventory or [] if r["key"] == selected
                       and r["kind"] in {"geometry", "both"} and (r.get("dimensions") or 0) > 2]
            if not matches:
                return (no_update,) * 5
            return {"display": "flex"}, "view", int(matches[0]["geometry_id"]), "umap", ""
        return {"display": "flex"}, "geometry", None, "umap", ""

    @app.callback(
        Output("representation-wizard-recipe", "options"),
        Output("representation-wizard-dimension-note", "children"),
        Input("representation-wizard-mode", "value"),
    )
    def wizard_mode(mode: str | None) -> tuple[Any, str]:
        if mode == "view":
            return [
                {"label": "UMAP", "value": "umap"},
                {"label": "SVD", "value": "svd"},
                {"label": "Identity (existing 2D geometry)", "value": "identity"},
            ], "2D views have exactly two coordinates."
        return geometry_recipe_options(), "Replayability is detected automatically from the fitted TeAL pipeline."

    @app.callback(
        Output("representation-wizard-parameters", "children"),
        Output("representation-wizard-advanced", "children"),
        Output("representation-wizard-advanced-wrap", "style"),
        Input("representation-wizard-mode", "value"),
        Input("representation-wizard-recipe", "value"),
    )
    def wizard_parameters(mode: str | None, recipe: str | None) -> tuple[Any, Any, Any]:
        if mode == "view" and recipe == "identity":
            return [], [], {"display": "none"}
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
        Output("representation-refresh-store", "data", allow_duplicate=True),
        Output("view-refresh-store", "data", allow_duplicate=True),
        Output("representation-wizard-status", "children"),
        Input("representation-wizard-submit", "n_clicks"),
        State("representation-wizard-mode", "value"),
        State("representation-wizard-name", "value"),
        State("representation-wizard-source", "value"),
        State("representation-wizard-recipe", "value"),
        State({"type": "representation-param", "name": ALL}, "value"),
        State({"type": "representation-param", "name": ALL}, "id"),
        State("representation-refresh-store", "data"),
        State("view-refresh-store", "data"),
        prevent_initial_call=True,
        running=[(Output("representation-wizard-submit", "disabled"), True, False)],
    )
    def create(
        _clicks: int, mode: str | None, name: str | None, source: int | None,
        recipe: str | None, values: list[Any], ids: list[dict[str, str]],
        revision: int | None,
        view_revision: int | None,
    ) -> tuple[Any, Any, str]:
        if source is None:
            return no_update, no_update, "Choose a source geometry."
        params = {item["name"]: value for item, value in zip(ids, values, strict=True)}
        try:
            if mode == "view":
                result_id = create_view(
                    project, name=name or "", geometry_id=int(source),
                    method=recipe or "umap", parameters=params,
                )
                label = "2D view"
            else:
                created = create_geometry(
                    project, name=name or "", source_geometry_id=int(source),
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
