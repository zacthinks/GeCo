"""Visualization-first editor for existing executable GeCo analysis charts."""
from __future__ import annotations

from typing import Any

from geometric_coder.analyses import analysis_figure


def variable_options(project: Any) -> list[dict[str, str]]:
    """Stable ID choices, refreshed when constructs change."""
    return [
        {"label": f"Code: {row['name']}", "value": f"code:{row['code_id']}"}
        for row in project.codes()
    ] + [
        {"label": f"Scale: {row['name']}", "value": f"scale:{row['construct_id']}"}
        for row in project.constructs() if row["type"] == "scale"
    ] + [
        {"label": f"Metadata: {field}", "value": f"metadata:{field}"}
        for field in project.metadata.get("metadata_columns", [])
    ]


def _identity(variable: dict[str, Any]) -> str:
    if variable["type"] == "code":
        return f"code:{variable['code_id']}"
    if variable["type"] == "scale":
        return f"scale:{variable['scale_id']}"
    return f"metadata:{variable['field']}"


def _slots(spec: dict[str, Any]) -> dict[str, Any]:
    """Load v1 without conversion or changing the definition on disk."""
    if spec.get("schema_version", 1) == 2:
        return {
            role: ([_identity(v["variable"]) for v in entries]
                   if isinstance(entries, list) else _identity(entries["variable"]))
            for role, entries in spec["roles"].items()
        }
    variables = spec["variables"]
    if spec["kind"] == "bar":
        return {
            "values": [_identity(v) for v in variables if v["type"] == "code"],
            "group": next((_identity(v) for v in variables if v["type"] == "metadata"), None),
        }
    return {
        ("indicators" if spec["kind"] == "heatmap" else "values"):
        [_identity(v) for v in variables]
    }


def _initial_config(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if spec.get("schema_version", 1) == 2:
        return {
            f"{role}:{_identity(item['variable'])}": dict(item["variable"])
            for role, entries in spec["roles"].items()
            for item in (entries if isinstance(entries, list) else [entries])
        }
    role_slots = _slots(spec)
    return {
        f"{role}:{_identity(variable)}": dict(variable)
        for role, values in role_slots.items()
        for variable in spec["variables"]
        if _identity(variable) in (values if isinstance(values, list) else [values])
    }


def _ref(raw: str, settings: dict[str, Any]) -> dict[str, Any]:
    type_, value = raw.split(":", 1)
    if type_ == "code":
        ref = {
            "type": "code", "code_id": int(value),
            "source": settings.get("source", "labels"),
            "value": settings.get("value", "binary"),
        }
        if ref["source"] != "labels":
            ref["predictor_kind"] = settings.get("predictor_kind", "classifier")
            if settings.get("predictor_id") is not None:
                ref["predictor_id"] = int(settings["predictor_id"])
        return ref
    if type_ == "scale":
        return {"type": "scale", "scale_id": int(value), "value": "position"}
    ref = {"type": "metadata", "field": value,
           "value": settings.get("value", "categorical")}
    if settings.get("numeric_strings") and ref["value"] == "numeric":
        ref["numeric_strings"] = True
    return ref


def editor_recipe(kind: str, measure: str, slots: dict[str, Any],
                  configs: dict[str, Any], aggregate_level: str | None,
                  aggregate_method: str, aggregate_threshold: float | None,
                  *, legacy: bool = False) -> dict[str, Any]:
    """Construct one recipe from role identities and role-scoped source choices."""
    raw_roles = {
        "bar": ("values", "group"),
        "heatmap": ("indicators",),
        "distribution": ("values",),
    }.get(kind)
    if raw_roles is None:
        raise ValueError("Unsupported visualization.")
    roles = {}
    for role in raw_roles:
        selection = slots.get(role)
        if role == "group":
            if selection:
                key = f"{role}:{selection}"
                roles[role] = {"variable": _ref(selection, configs.get(key, {}))}
        else:
            roles[role] = [
                {"variable": _ref(value, configs.get(f"{role}:{value}", {}))}
                for value in (selection or [])
            ]
    recipe: dict[str, Any] = {
        "schema_version": 2, "kind": kind, "roles": roles, "measure": measure,
    }
    if aggregate_level and aggregate_level != "none":
        recipe["aggregation"] = {
            "level": aggregate_level, "method": aggregate_method,
            "threshold": 1 if aggregate_threshold is None else aggregate_threshold,
        }
    if legacy:
        # Existing v1 definitions remain v1 when edited; an unsupported
        # representation is rejected rather than changing their meaning.
        # The v2 contract first checks all role settings, including no implicit
        # numeric or temporal interpretation; then faithfully maps to v1.
        return {"kind": kind, "measure": measure,
                "variables": [dict(item["variable"]) for item in
                              roles.get("values", roles.get("indicators", []))]
                + ([dict(roles["group"]["variable"])] if "group" in roles else []),
                **({"aggregation": recipe["aggregation"]} if "aggregation" in recipe else {})}
    return recipe


def analysis_tab(*, dcc: Any, html: Any, project: Any) -> Any:
    return html.Div([
        dcc.Store(id="analysis-refresh", data=0),
        dcc.Store(id="analysis-editing", data=None),
        dcc.Store(id="analysis-role-config", data={}),
        dcc.Store(id="analysis-role-reset", data=0),
        html.Div([
            html.Div([
                html.Label("Saved analyses"),
                dcc.Dropdown(id="analysis-select", options=[], placeholder="Select an analysis"),
            ], style={"flex": "1", "minWidth": "180px"}),
            html.Button("New analysis", id="analysis-new", n_clicks=0),
            html.Button("Edit", id="analysis-edit", n_clicks=0),
            html.Button("Duplicate", id="analysis-duplicate", n_clicks=0),
            html.Button("Delete", id="analysis-delete", n_clicks=0),
        ], className="analysis-actions"),
        html.Div([
            html.P("Delete this analysis? Codes and predictions will not be changed."),
            html.Button("Confirm delete", id="analysis-delete-confirm", n_clicks=0),
            html.Button("Cancel", id="analysis-delete-cancel", n_clicks=0),
        ], id="analysis-delete-dialog", style={"display": "none"}),
        html.Div([
            html.H3(id="analysis-editor-heading", children="New analysis"),
            html.Label("Name"),
            dcc.Input(id="analysis-name", type="text", className="text-input",
                      placeholder="Analysis name", maxLength=160),
            html.Label("Visualization"),
            dcc.Dropdown(id="analysis-kind", options=[
                {"label": "Bar plot — positive Code counts", "value": "bar"},
                {"label": "Association heatmap", "value": "heatmap"},
                {"label": "Code-value distribution", "value": "distribution"},
            ], value="bar", clearable=False),
            html.Div(id="analysis-visualization-help"),
            html.Div([
                html.Label("Values — Codes to count"),
                dcc.Dropdown(id="analysis-role-values", multi=True, options=[], value=[]),
                html.Label("Group — categorical metadata (optional)"),
                dcc.Dropdown(id="analysis-role-group", options=[], value=None),
            ], id="analysis-bar-roles"),
            html.Div([
                html.Label("Values — Codes whose observed values are distributed"),
                dcc.Dropdown(id="analysis-role-distribution", multi=True, options=[], value=[]),
            ], id="analysis-distribution-roles", style={"display": "none"}),
            html.Div([
                html.Label("Indicators — Codes and categorical metadata"),
                dcc.Dropdown(id="analysis-role-indicators", multi=True, options=[], value=[]),
                html.P("A symmetric association matrix, not a Rows × Columns pivot table."),
            ], id="analysis-heatmap-roles", style={"display": "none"}),
            html.Div([
                html.Label("Association statistic"),
                dcc.Dropdown(id="analysis-measure", value="count", clearable=False, options=[
                    {"label": "Positive co-occurrence count", "value": "count"},
                    {"label": "Jaccard similarity", "value": "jaccard"},
                    {"label": "Pearson correlation", "value": "correlation"},
                ]),
            ], id="analysis-measure-options", style={"display": "none"}),
            html.Div(id="analysis-role-settings"),
            html.Label("Aggregate observations (optional)"),
            dcc.Dropdown(id="analysis-aggregate-level", clearable=False,
                         options=[{"label": "No aggregation (atomic units)", "value": "none"}] + [
                             {"label": f"Group by {key} (and preceding keys)", "value": key}
                             for key in project.metadata.get("key_columns", [])[:-1]
                         ], value="none"),
            html.Div([
                html.Label("Aggregate Code values"),
                dcc.Dropdown(id="analysis-aggregate-method", value="any", clearable=False, options=[
                    {"label": "Any observed positive (0 or 1)", "value": "any"},
                    {"label": "Number of positive units per group", "value": "count"},
                    {"label": "Positive count at least threshold", "value": "minimum"},
                    {"label": "Proportion positive among observed units", "value": "mean"},
                    {"label": "Natural log(1 + positive count)", "value": "log_count"},
                ]),
                html.Label("Positive count threshold (threshold method only)"),
                dcc.Input(id="analysis-aggregate-threshold", type="number", min=0,
                          step=1, value=1),
                html.P("Only Codes are reduced. Selected metadata must be constant within a group."),
            ], id="analysis-aggregation-options", style={"display": "none"}),
            html.Div(id="analysis-status", className="analysis-validation", role="alert"),
            html.Div([
                html.Div([html.Button("Preview Changes", id="analysis-preview", n_clicks=0)],
                         className="analysis-preview-actions"),
                html.Div([
                    html.Button("Cancel", id="analysis-cancel", n_clicks=0),
                    html.Button("Create Analysis", id="analysis-save", n_clicks=0,
                                className="primary-action"),
                ], className="analysis-commit-actions"),
            ], className="analysis-editor-actions"),
        ], id="analysis-editor", style={"display": "none"}, className="analysis-editor"),
        dcc.Graph(id="analysis-figure", figure=analysis_figure({
            "kind": "bar", "names": [], "counts": [],
            "denominators": [], "categories": [],
        }), config={"displaylogo": False}),
    ], className="analysis-manager")


def register_analysis_callbacks(app: Any, project: Any) -> None:
    from dash import ALL, Input, Output, State, callback_context, dcc, html, no_update
    from geometric_coder.ui.app import _empty_figure
    from geometric_coder.analyses import compute_analysis

    @app.callback(Output("analysis-select", "options"),
                  Input("analysis-refresh", "data"), Input("workspace-tabs", "value"))
    def options(_revision: int, workspace: str) -> Any:
        if workspace != "analyses":
            return no_update
        return [{"label": row["name"], "value": row["analysis_id"]}
                for row in project.analyses()]

    @app.callback(
        Output("analysis-role-values", "options"),
        Output("analysis-role-distribution", "options"),
        Output("analysis-role-indicators", "options"),
        Output("analysis-role-group", "options"),
        Input("workspace-tabs", "value"),
        Input("explore-code-refresh-store", "data"),
        Input("focus-refresh-store", "data"),
        Input("analysis-editor", "style"),
    )
    def refresh_variables(workspace: str, _code_refresh: Any, _focus_refresh: Any,
                          _editor_style: Any) -> Any:
        if workspace != "analyses":
            return (no_update,) * 4
        choices = variable_options(project)
        codes = [c for c in choices if c["value"].startswith("code:")]
        metadata = [c for c in choices if c["value"].startswith("metadata:")]
        return codes, codes, choices, metadata

    @app.callback(
        Output("analysis-editor", "style"),
        Output("analysis-editing", "data"),
        Output("analysis-editor-heading", "children"),
        Output("analysis-name", "value"),
        Output("analysis-kind", "value"),
        Output("analysis-measure", "value"),
        Output("analysis-role-values", "value"),
        Output("analysis-role-group", "value"),
        Output("analysis-role-distribution", "value"),
        Output("analysis-role-indicators", "value"),
        Output("analysis-role-config", "data"),
        Output("analysis-aggregate-level", "value"),
        Output("analysis-aggregate-method", "value"),
        Output("analysis-aggregate-threshold", "value"),
        Output("analysis-status", "children", allow_duplicate=True),
        Input("analysis-new", "n_clicks"),
        Input("analysis-edit", "n_clicks"),
        Input("analysis-duplicate", "n_clicks"),
        Input("analysis-cancel", "n_clicks"),
        State("analysis-select", "value"),
        prevent_initial_call=True,
    )
    def open_editor(_new: int, _edit: int, _duplicate: int, _cancel: int,
                    selected: int | None) -> tuple[Any, ...]:
        trigger = callback_context.triggered_id
        empty = ({"display": "block"}, {"id": None, "schema_version": 2},
                 "New analysis", "", "bar", "count", [], None, [], [], {},
                 "none", "any", 1, "")
        if trigger == "analysis-cancel":
            return ({"display": "none"}, None, "", "", "bar", "count", [],
                    None, [], [], {}, "none", "any", 1, "")
        if trigger == "analysis-new":
            return empty
        if selected is None:
            return (no_update,) * 15
        record = project.analysis(int(selected))
        spec = record["spec"]
        slots = _slots(spec)
        aggregate = spec.get("aggregation", {})
        return (
            {"display": "block"},
            {"id": int(selected) if trigger == "analysis-edit" else None,
             "schema_version": spec.get("schema_version", 1)},
            "Edit analysis" if trigger == "analysis-edit" else "Duplicate analysis",
            record["name"] if trigger == "analysis-edit" else record["name"] + " (copy)",
            spec["kind"], spec.get("measure", "count"),
            slots.get("values", []) if spec["kind"] == "bar" else [],
            slots.get("group") if spec["kind"] == "bar" else None,
            slots.get("values", []) if spec["kind"] == "distribution" else [],
            slots.get("indicators", []) if spec["kind"] == "heatmap" else [],
            _initial_config(spec), aggregate.get("level", "none"),
            aggregate.get("method", "any"), aggregate.get("threshold", 1), "",
        )

    @app.callback(
        Output("analysis-bar-roles", "style"),
        Output("analysis-distribution-roles", "style"),
        Output("analysis-heatmap-roles", "style"),
        Output("analysis-measure-options", "style"),
        Output("analysis-visualization-help", "children"),
        Input("analysis-kind", "value"),
        Input("analysis-aggregate-level", "value"),
        Input("analysis-aggregate-method", "value"),
    )
    def chart_controls(kind: str, level: str | None, method: str | None) -> tuple[Any, ...]:
        visible = {"display": "block"}
        hidden = {"display": "none"}
        if kind == "distribution":
            help_text = ("Atomic units contribute binary observed Code values (0/1). "
                         "Parent grouping distributes reduced Code values by group."
                         if level and level != "none" else
                         "Counts observed negative (0) and positive (1) Code values; "
                         "not a general numeric histogram.")
        elif kind == "heatmap":
            help_text = ("A symmetric matrix of positive co-occurrence, Jaccard "
                         "similarity, or pairwise-complete correlations. "
                         "Categorical metadata expands into indicators.")
        else:
            help_text = "Counts positive Code assignments, optionally by categorical metadata."
        return (
            visible if kind == "bar" else hidden,
            visible if kind == "distribution" else hidden,
            visible if kind == "heatmap" else hidden,
            visible if kind == "heatmap" else hidden,
            help_text,
        )

    @app.callback(Output("analysis-measure", "value", allow_duplicate=True),
                  Input("analysis-kind", "value"),
                  State("analysis-measure", "value"),
                  prevent_initial_call=True)
    def reset_measure(kind: str, current: str) -> Any:
        return "count" if kind != "heatmap" else (current if current in
            ("count", "jaccard", "correlation") else "count")

    @app.callback(Output("analysis-aggregation-options", "style"),
                  Input("analysis-aggregate-level", "value"))
    def show_aggregation(level: str | None) -> dict[str, str]:
        return {"display": "block"} if level and level != "none" else {"display": "none"}

    @app.callback(Output("analysis-save", "children"),
                  Input("analysis-editing", "data"))
    def primary_button(editing: dict[str, Any] | None) -> str:
        return "Save Changes" if editing and editing.get("id") is not None else "Create Analysis"

    @app.callback(
        Output("analysis-role-settings", "children"),
        Input("analysis-kind", "value"),
        Input("analysis-role-values", "value"),
        Input("analysis-role-group", "value"),
        Input("analysis-role-distribution", "value"),
        Input("analysis-role-indicators", "value"),
        Input("analysis-editing", "data"),
        State("analysis-role-config", "data"),
    )
    def role_settings(kind: str, values: list[str] | None, group: str | None,
                      distribution: list[str] | None, indicators: list[str] | None,
                      editing: Any, configs: dict[str, Any] | None) -> Any:
        if editing is None:
            return []
        if kind == "bar":
            slots = {"values": values or [], "group": [group] if group else []}
        elif kind == "distribution":
            slots = {"values": distribution or []}
        else:
            slots = {"indicators": indicators or []}
        nodes = []
        for role, selected in slots.items():
            for raw in selected:
                family, identity = raw.split(":", 1)
                key = f"{role}:{raw}"
                config = (configs or {}).get(key, {})
                controls = [html.H4(f"{role.title()}: {raw}")]
                if family == "code":
                    code_id = int(identity)
                    sources = [
                        {"label": "Human labels only", "value": "labels"},
                        {"label": "Predictions only", "value": "predictions"},
                        {"label": "Human first, then predictions", "value": "hybrid"},
                    ]
                    controls.extend([
                        html.Label("Source"),
                        dcc.Dropdown(id={"type": "analysis-role-source", "index": key},
                                     options=sources, value=config.get("source", "labels"),
                                     clearable=False),
                        html.Label("Value representation"),
                        dcc.Dropdown(id={"type": "analysis-role-value", "index": key},
                                     options=[
                                         {"label": "Binary (0/1) — supported", "value": "binary"},
                                         {"label": "Probability (Phase 3)", "value": "probability"},
                                     ], value=config.get("value", "binary"), clearable=False),
                        html.Label("Predictor family"),
                        dcc.Dropdown(id={"type": "analysis-role-family", "index": key},
                                     options=[
                                         {"label": "Classifier", "value": "classifier"},
                                         {"label": "Committee", "value": "committee"},
                                     ], value=config.get("predictor_kind", "classifier"),
                                     clearable=False),
                        html.Label("Prediction model (required for prediction/hybrid)"),
                        dcc.Dropdown(id={"type": "analysis-role-predictor", "index": key},
                                     options=[
                                         {"label": f"Classifier: {p['name']}", "value": f"classifier:{p['classifier_spec_id']}"}
                                         for p in project.classifier_specs(code_id)
                                     ] + [
                                         {"label": f"Committee: {p['name']}", "value": f"committee:{p['committee_id']}"}
                                         for p in project.classifier_committees(code_id)
                                     ],
                                     value=(f"{config.get('predictor_kind', 'classifier')}:{config['predictor_id']}"
                                            if config.get("predictor_id") is not None else None)),
                    ])
                elif family == "metadata":
                    controls.extend([
                        html.Label("Interpret metadata as"),
                        dcc.Dropdown(
                            id={"type": "analysis-role-value", "index": key},
                            options=[
                                {"label": "Categorical (supported)", "value": "categorical"},
                                {"label": "Numeric (Phase 3)", "value": "numeric"},
                                {"label": "Temporal / ISO date (Phase 3 grouping)", "value": "temporal"},
                            ], value=config.get("value", "categorical"),
                            clearable=False,
                        ),
                        html.P("Numerical bins and day/week/month/quarter/year grouping "
                               "require Phase 3. Temporal metadata is never silently "
                               "treated as a categorical date string."),
                    ])
                else:
                    status = project.scale_model_status(int(identity))
                    controls.extend([
                        html.P("Predicted continuous position only (Phase 3 chart required)."),
                        html.P(f"Scale fit status: {status['status']}. "
                               "Untrained, stale, or ineligible positions cannot be "
                               "silently replaced by human ratings or categorical bins."),
                    ])
                nodes.append(html.Div(controls, className="analysis-variable-settings"))
        return nodes

    @app.callback(
        Output("analysis-role-config", "data", allow_duplicate=True),
        Input({"type": "analysis-role-source", "index": ALL}, "value"),
        Input({"type": "analysis-role-value", "index": ALL}, "value"),
        Input({"type": "analysis-role-family", "index": ALL}, "value"),
        Input({"type": "analysis-role-predictor", "index": ALL}, "value"),
        State({"type": "analysis-role-source", "index": ALL}, "id"),
        State({"type": "analysis-role-value", "index": ALL}, "id"),
        State({"type": "analysis-role-family", "index": ALL}, "id"),
        State({"type": "analysis-role-predictor", "index": ALL}, "id"),
        State("analysis-role-config", "data"),
        prevent_initial_call=True,
    )
    def preserve_role_configs(
        sources: list[str], values: list[str], families: list[str],
        predictors: list[str | None], source_ids: list[Any], value_ids: list[Any],
        family_ids: list[Any], predictor_ids: list[Any],
        existing: dict[str, Any] | None,
    ) -> Any:
        result = {key: dict(val) for key, val in (existing or {}).items()}
        for component_ids, component_values, field in [
            (source_ids, sources, "source"), (value_ids, values, "value"),
            (family_ids, families, "predictor_kind"),
        ]:
            for item, value in zip(component_ids, component_values):
                result.setdefault(item["index"], {})[field] = value
        for item, value in zip(predictor_ids, predictors):
            config = result.setdefault(item["index"], {})
            if value:
                family, predictor = value.split(":", 1)
                config["predictor_kind"] = family
                config["predictor_id"] = int(predictor)
            else:
                config.pop("predictor_id", None)
        return result if result != (existing or {}) else no_update

    def current_slots(kind: str, values: Any, group: Any,
                      distribution: Any, indicators: Any) -> dict[str, Any]:
        if kind == "bar":
            return {"values": values or [], "group": group}
        if kind == "distribution":
            return {"values": distribution or []}
        return {"indicators": indicators or []}

    def run_recipe(kind: str, measure: str, values: Any, group: Any,
                   distribution: Any, indicators: Any,
                   configs: dict[str, Any] | None, level: str | None,
                   method: str, threshold: float | None,
                   editing: dict[str, Any]) -> dict[str, Any]:
        legacy = editing.get("schema_version", 2) == 1
        recipe = editor_recipe(
            kind, measure, current_slots(kind, values, group, distribution, indicators),
            configs or {}, level, method, threshold, legacy=legacy,
        )
        if legacy:
            # v1 is retained exactly, without non-v1 representation fields.
            for var in recipe["variables"]:
                if var["type"] == "code":
                    if var.get("predictor_kind") == "committee":
                        raise ValueError("Existing v1 analyses cannot store committee sources.")
                    if var["source"] == "labels":
                        var.pop("predictor_kind", None)
                    else:
                        var.pop("predictor_kind", None)
                elif var["type"] == "metadata":
                    if var.get("value", "categorical") != "categorical":
                        raise ValueError("Numeric/temporal metadata requires a Phase 3 chart.")
                    var.pop("value", None)
                    var.pop("numeric_strings", None)
                else:
                    raise ValueError("Scale positions need a Phase 3 chart.")
        return recipe

    state_args = [
        State("analysis-editing", "data"), State("analysis-kind", "value"),
        State("analysis-measure", "value"),
        State("analysis-role-values", "value"),
        State("analysis-role-group", "value"),
        State("analysis-role-distribution", "value"),
        State("analysis-role-indicators", "value"),
        State("analysis-role-config", "data"),
        State("analysis-aggregate-level", "value"),
        State("analysis-aggregate-method", "value"),
        State("analysis-aggregate-threshold", "value"),
    ]

    @app.callback(
        Output("analysis-refresh", "data"),
        Output("analysis-select", "value"),
        Output("analysis-select", "options", allow_duplicate=True),
        Output("analysis-status", "children"),
        Output("analysis-editor", "style", allow_duplicate=True),
        Input("analysis-save", "n_clicks"),
        State("analysis-name", "value"),
        *state_args, State("analysis-refresh", "data"),
        prevent_initial_call=True,
    )
    def save(_click: int, name: str | None, editing: Any, kind: str, measure: str,
             values: Any, group: Any, distribution: Any, indicators: Any,
             configs: Any, level: Any, method: str, threshold: Any,
             revision: int | None) -> tuple[Any, ...]:
        if editing is None:
            return (no_update,) * 5
        try:
            spec = run_recipe(kind, measure, values, group, distribution, indicators,
                              configs, level, method, threshold, editing)
            analysis_id = project.save_analysis(name or "", spec, analysis_id=editing.get("id"))
        except (ValueError, KeyError) as exc:
            return no_update, no_update, no_update, str(exc), no_update
        options = [{"label": row["name"], "value": row["analysis_id"]}
                   for row in project.analyses()]
        return int(revision or 0) + 1, analysis_id, options, "", {"display": "none"}

    @app.callback(
        Output("analysis-figure", "figure", allow_duplicate=True),
        Output("analysis-status", "children", allow_duplicate=True),
        Input("analysis-preview", "n_clicks"),
        *state_args,
        prevent_initial_call=True,
    )
    def preview(_click: int, editing: Any, kind: str, measure: str,
                values: Any, group: Any, distribution: Any, indicators: Any,
                configs: Any, level: Any, method: str, threshold: Any) -> tuple[Any, str]:
        if editing is None:
            return no_update, ""
        try:
            spec = run_recipe(kind, measure, values, group, distribution, indicators,
                              configs, level, method, threshold, editing)
            result = compute_analysis(project, spec)
            fig = analysis_figure(result)
            if result.get("warnings"):
                fig.add_annotation(
                    text=" | ".join(result["warnings"]), xref="paper", yref="paper",
                    x=0, y=1.14, xanchor="left", showarrow=False,
                    font={"color": "#9a6510", "size": 11},
                )
            return fig, ""
        except (ValueError, KeyError) as exc:
            return no_update, f"Preview unavailable: {exc}"

    @app.callback(
        Output("analysis-delete-dialog", "style"),
        Input("analysis-delete", "n_clicks"),
        Input("analysis-delete-cancel", "n_clicks"),
        State("analysis-select", "value"), prevent_initial_call=True,
    )
    def confirm_dialog(_delete: int, _cancel: int, selected: int | None) -> dict[str, str]:
        return {"display": "block"} if (
            callback_context.triggered_id == "analysis-delete" and selected is not None
        ) else {"display": "none"}

    @app.callback(
        Output("analysis-refresh", "data", allow_duplicate=True),
        Output("analysis-select", "value", allow_duplicate=True),
        Output("analysis-delete-dialog", "style", allow_duplicate=True),
        Input("analysis-delete-confirm", "n_clicks"),
        State("analysis-select", "value"), State("analysis-refresh", "data"),
        prevent_initial_call=True,
    )
    def delete(_click: int, selected: int | None, revision: int | None) -> tuple[Any, ...]:
        if selected is None:
            return no_update, no_update, {"display": "none"}
        project.delete_analysis(int(selected))
        return int(revision or 0) + 1, None, {"display": "none"}

    @app.callback(
        Output("analysis-figure", "figure"),
        Input("analysis-select", "value"),
        Input("analysis-refresh", "data"),
        Input("workspace-tabs", "value"),
        Input("analysis-cancel", "n_clicks"),
        Input("explore-annotation-refresh-store", "data"),
        Input("focus-refresh-store", "data"),
        Input("teaching-example-refresh-store", "data"),
    )
    def render(selected: int | None, _refresh: int, workspace: str, _cancel: int,
               _annotations: int, _focus: int, _teaching: int) -> Any:
        if workspace != "analyses":
            return no_update
        if selected is None:
            return _empty_figure("Select or create an analysis.")
        try:
            result = project.run_analysis(int(selected))
            fig = analysis_figure(result)
            if result.get("warnings"):
                fig.add_annotation(
                    text=" | ".join(result["warnings"]), xref="paper", yref="paper",
                    x=0, y=1.14, xanchor="left", showarrow=False,
                    font={"color": "#9a6510", "size": 11},
                )
            return fig
        except (ValueError, KeyError) as exc:
            return _empty_figure(f"Analysis unavailable: {exc}")
