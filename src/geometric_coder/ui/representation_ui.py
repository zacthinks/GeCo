"""Dash presentation for the general representation manager.

Scientific transformations and registration live in representation_manager, not
callbacks. The manager does not mutate TeAL artifacts or unregister dependencies.
"""
from __future__ import annotations

from typing import Any

from geometric_coder.ui.representation_manager import (
    create_geometry, create_view, geometry_parameter_specs, geometry_recipe_options,
    linked_geometry_options, representation_inventory,
)
from geometric_coder.ui.topic_creation import teal_linked_project


def representation_modal(project: Any, *, html: Any, dcc: Any) -> Any:
    linked = teal_linked_project(project) is not None
    return html.Div([
        html.Div([
            html.Div([
                html.H2("Manage Representations"),
                html.Button("×", id="representation-manager-close", n_clicks=0,
                            className="modal-close-icon", title="Close"),
            ], className="modal-header"),
            html.P(
                "Geometries are analytic representations. 2D views are display "
                "projections and do not replace the underlying geometry.",
                className="modal-help",
            ),
            html.H3("Registered geometries"),
            html.Div(id="representation-geometry-list"),
            html.H3("Registered 2D views"),
            html.Div(id="representation-view-list"),
            html.H3("Create Geometry"),
            html.P(
                "Apply one TeAL transformation to a registered geometry. "
                "Each saved output can become the source of another step."
            ),
            html.Label("Geometry name"),
            dcc.Input(id="representation-create-name", type="text",
                      className="text-input", placeholder="New representation name"),
            html.Label("Source geometry"),
            dcc.Dropdown(id="representation-create-source",
                         options=linked_geometry_options(project), clearable=False),
            html.Label("Transformation"),
            dcc.Dropdown(id="representation-create-recipe",
                         options=geometry_recipe_options(), value="umap", clearable=False),
            html.Div(id="representation-create-parameters",
                     className="topic-create-parameters"),
            html.Details([
                html.Summary("Advanced parameters"),
                html.Div(id="representation-create-advanced",
                         className="topic-create-parameters"),
            ], id="representation-create-advanced-wrap", style={"display": "none"}),
            dcc.Checklist(
                id="representation-navigation-only",
                options=[{"label": "Navigation only (no semantic query/new-text replay)",
                          "value": "yes"}],
                value=["yes"],
            ),
            html.P(
                "Navigation-only mode accepts saved, keyed fit-only matrices. "
                "Uncheck only if TeAL can replay a complete text-to-geometry pipeline.",
                className="modal-help",
            ),
            html.Div([
                html.Button("Create Geometry", id="representation-create-submit",
                            n_clicks=0, disabled=not linked, className="primary-action"),
            ], className="modal-actions"),
            html.Div(
                "Create Geometry requires a TeAL-linked workspace." if not linked else "",
                id="representation-create-status", role="status", className="control-status",
            ),
            html.H3("Create 2D View"),
            html.P("A view shows two coordinates; it does not replace its parent analytic geometry."),
            html.Label("View name"),
            dcc.Input(id="representation-view-name", type="text", className="text-input"),
            html.Label("Parent geometry"),
            dcc.Dropdown(id="representation-view-source", options=linked_geometry_options(project),
                         clearable=False),
            html.Label("Projection"),
            dcc.Dropdown(id="representation-view-method",
                         options=[{"label": "UMAP", "value": "umap"},
                                  {"label": "SVD", "value": "svd"},
                                  {"label": "Identity (already 2D)", "value": "identity"}],
                         value="umap", clearable=False),
            html.Div(id="representation-view-parameters", className="topic-create-parameters"),
            html.Details([html.Summary("Advanced parameters"),
                          html.Div(id="representation-view-advanced", className="topic-create-parameters")],
                         id="representation-view-advanced-wrap", style={"display": "none"}),
            html.Div([html.Button("Create 2D View", id="representation-view-submit",
                                 n_clicks=0, disabled=not linked, className="primary-action")],
                     className="modal-actions"),
            html.Div(id="representation-view-status", role="status", className="control-status"),
        ], className="modal-card topic-manager-card"),
    ], id="representation-manager-modal", className="modal-overlay",
       style={"display": "none"})


def register_representation_callbacks(
    app: Any, project: Any, *, html: Any, dcc: Any, Input: Any,
    Output: Any, State: Any, ALL: Any, callback_context: Any,
    no_update: Any, geometries: list[dict[str, Any]],
    geometry_by_id: dict[int, dict[str, Any]],
) -> None:
    @app.callback(
        Output("representation-manager-modal", "style"),
        Input("representation-manager-open", "n_clicks"),
        Input("representation-manager-close", "n_clicks"),
        prevent_initial_call=True,
    )
    def toggle(_open: int, _close: int) -> dict[str, str]:
        return {"display": "flex"} if callback_context.triggered_id == "representation-manager-open" else {"display": "none"}

    @app.callback(
        Output("representation-geometry-list", "children"),
        Output("representation-view-list", "children"),
        Output("representation-create-source", "options"),
        Output("representation-view-source", "options"),
        Input("representation-manager-open", "n_clicks"),
        Input("representation-refresh-store", "data"),
    )
    def refresh_inventory(_open: int, _revision: int) -> tuple[Any, Any, list[dict[str, Any]], list[dict[str, Any]]]:
        inventory = representation_inventory(project)
        geometry_items = [
            html.Div([
                html.Strong(
                    f"{record['name']} · "
                    f"{record['dimensions'] if record['dimensions'] is not None else '?'} dimensions"
                ),
                html.P(
                    f"Source: {record['source'] or 'unknown'} · "
                    f"Query: {'yes' if record['queryable'] else 'no'} · "
                    f"New text: {'yes' if record['text_transform'] else 'no'}"
                ),
                html.Small(record["error"] or (record["artifact_id"] or "Locally stored")),
            ], className="control-block") for record in inventory
        ]
        views = project.views()
        names = {g["geometry_id"]: g["name"] for g in inventory}
        view_items = [
            html.Div(
                f"{view['name']} · parent: "
                f"{names.get(int(view['geometry_id']), view['geometry_id'])}"
                f" · {view['method']}"
            ) for view in views
        ]
        return (
            geometry_items or ["No registered geometries."],
            view_items or ["No registered 2D views."],
            linked_geometry_options(project),
            linked_geometry_options(project),
        )

    @app.callback(
        Output("representation-create-parameters", "children"),
        Output("representation-create-advanced", "children"),
        Output("representation-create-advanced-wrap", "style"),
        Input("representation-create-recipe", "value"),
    )
    def recipe_parameters(recipe: str | None) -> tuple[list[Any], list[Any], dict[str, str]]:
        try:
            common, advanced = geometry_parameter_specs(recipe or "umap")
        except ImportError:
            return [], [], {"display": "none"}

        def controls(specs: list[dict[str, Any]]) -> list[Any]:
            result = []
            for spec in specs:
                identifier = {"type": "geometry-hyperparameter", "name": spec["name"]}
                if spec["kind"] == "boolean":
                    field = dcc.Dropdown(
                        id=identifier,
                        options=[{"label": "True", "value": "true"},
                                 {"label": "False", "value": "false"}],
                        value="true" if spec["default"] else "false",
                        clearable=False,
                    )
                elif spec["kind"] == "choice":
                    field = dcc.Dropdown(
                        id=identifier, options=spec["options"],
                        value=spec["default"], clearable=False,
                    )
                else:
                    field = dcc.Input(
                        id=identifier, value=spec["default"],
                        type="number" if spec["kind"] in {"integer", "number"} else "text",
                        step=1 if spec["kind"] == "integer" else "any",
                    )
                result.append(
                    html.Div([html.Label(spec["label"]), field],
                             className="topic-parameter-field")
                )
            return result

        return controls(common), controls(advanced), {
            "display": "block" if advanced else "none"
        }

    @app.callback(
        Output("representation-refresh-store", "data"),
        Output("representation-create-status", "children"),
        Input("representation-create-submit", "n_clicks"),
        State("representation-create-name", "value"),
        State("representation-create-source", "value"),
        State("representation-create-recipe", "value"),
        State({"type": "geometry-hyperparameter", "name": ALL}, "value"),
        State({"type": "geometry-hyperparameter", "name": ALL}, "id"),
        State("representation-navigation-only", "value"),
        State("representation-refresh-store", "data"),
        prevent_initial_call=True,
        running=[(Output("representation-create-submit", "disabled"), True, False)],
    )
    def create(
        _clicks: int, name: str | None, source: int | None, recipe: str | None,
        values: list[Any], ids: list[dict[str, str]], navigation: list[str] | None,
        revision: int | None,
    ) -> tuple[Any, str]:
        if source is None:
            return no_update, "Choose a registered source geometry."
        parameters = {entry["name"]: value for entry, value in zip(ids, values, strict=True)}
        try:
            created = create_geometry(
                project, name=name or "", source_geometry_id=int(source),
                recipe=recipe or "umap", parameters=parameters,
                navigation_only="yes" in (navigation or []),
            )
        except Exception as error:
            return no_update, f"Create Geometry failed: {error}"
        # Existing Explore callbacks retain these shared lookup objects.
        # Update in-place so a newly registered geometry can be selected now.
        current = project.geometries(public_only=True)
        geometries[:] = current
        geometry_by_id.clear()
        geometry_by_id.update({int(g["geometry_id"]): g for g in current})
        return int(revision or 0) + 1, (
            f"Created geometry {(name or '').strip()!r} (ID {created.geometry_id})."
        )

    @app.callback(
        Output("geometry-dropdown", "options"),
        Output("topic-create-geometry", "options"),
        Input("representation-refresh-store", "data"),
    )
    def refresh_geometry_dropdowns(_revision: int) -> tuple[Any, Any]:
        options = [
            {"label": g["name"], "value": int(g["geometry_id"])}
            for g in project.geometries(public_only=True)
        ]
        return options, options

    @app.callback(
        Output("representation-view-parameters", "children"),
        Output("representation-view-advanced", "children"),
        Output("representation-view-advanced-wrap", "style"),
        Input("representation-view-method", "value"),
    )
    def view_parameters(method: str | None) -> tuple[list[Any], list[Any], dict[str, str]]:
        if method == "identity":
            return [], [], {"display": "none"}
        try:
            common, advanced = geometry_parameter_specs(method or "umap")
        except ImportError:
            return [], [], {"display": "none"}
        common = [s for s in common if s["name"] != "n_components"]
        advanced = [s for s in advanced if s["name"] != "n_components"]

        def controls(specs: list[dict[str, Any]]) -> list[Any]:
            result = []
            for spec in specs:
                identifier = {"type": "view-hyperparameter", "name": spec["name"]}
                if spec["kind"] == "boolean":
                    field = dcc.Dropdown(
                        id=identifier,
                        options=[{"label": "True", "value": "true"},
                                 {"label": "False", "value": "false"}],
                        value="true" if spec["default"] else "false", clearable=False,
                    )
                elif spec["kind"] == "choice":
                    field = dcc.Dropdown(id=identifier, options=spec["options"],
                                         value=spec["default"], clearable=False)
                else:
                    field = dcc.Input(
                        id=identifier, value=spec["default"],
                        type="number" if spec["kind"] in {"integer", "number"} else "text",
                        step=1 if spec["kind"] == "integer" else "any",
                    )
                result.append(html.Div([html.Label(spec["label"]), field],
                                       className="topic-parameter-field"))
            return result

        return controls(common), controls(advanced), {
            "display": "block" if advanced else "none"
        }

    @app.callback(
        Output("representation-refresh-store", "data", allow_duplicate=True),
        Output("view-refresh-store", "data", allow_duplicate=True),
        Output("representation-view-status", "children"),
        Input("representation-view-submit", "n_clicks"),
        State("representation-view-name", "value"),
        State("representation-view-source", "value"),
        State("representation-view-method", "value"),
        State({"type": "view-hyperparameter", "name": ALL}, "value"),
        State({"type": "view-hyperparameter", "name": ALL}, "id"),
        State("representation-refresh-store", "data"),
        State("view-refresh-store", "data"),
        prevent_initial_call=True,
        running=[(Output("representation-view-submit", "disabled"), True, False)],
    )
    def create_2d_view(
        _clicks: int, name: str | None, source: int | None, method: str | None,
        values: list[Any], ids: list[dict[str, str]], revision: int | None,
        view_revision: int | None,
    ) -> tuple[Any, Any, str]:
        if source is None:
            return no_update, no_update, "Choose a registered parent geometry."
        parameters = {entry["name"]: value for entry, value in zip(ids, values, strict=True)}
        try:
            view_id = create_view(
                project, name=name or "", geometry_id=int(source),
                method=method or "umap", parameters=parameters,
            )
        except Exception as error:
            return no_update, no_update, f"Create 2D View failed: {error}"
        return (
            int(revision or 0) + 1, int(view_revision or 0) + 1,
            f"Created 2D view {(name or '').strip()!r} (ID {view_id}).",
        )
