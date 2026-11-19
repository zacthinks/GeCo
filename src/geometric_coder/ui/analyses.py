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
            f"{role}:{_identity(item['variable'])}": {
                **dict(item["variable"]),
                **({"transform": dict(item["transform"])} if "transform" in item else {}),
            }
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


def _reconcile_live_source_choices(
    configs: dict[str, Any] | None,
    sources: Any, source_ids: Any,
    predictors: Any, predictor_ids: Any,
) -> dict[str, Any]:
    """Prefer currently mounted source/model controls over an asynchronous Store."""
    resolved = {key: dict(item) for key, item in (configs or {}).items()}
    for item, source in zip(source_ids or [], sources or []):
        resolved.setdefault(item["index"], {})["source"] = source
    for item, selected in zip(predictor_ids or [], predictors or []):
        config = resolved.setdefault(item["index"], {})
        if selected:
            family, predictor_id = selected.split(":", 1)
            if family not in ("classifier", "committee"):
                raise ValueError("Select a valid classifier or committee.")
            config["predictor_kind"] = family
            config["predictor_id"] = int(predictor_id)
        else:
            # Explicitly cleared selection must not fall back to a cached fit.
            config.pop("predictor_kind", None)
            config.pop("predictor_id", None)
    return resolved


def _preserve_hidden_bin_settings(current: dict[str, Any], previous: Any) -> dict[str, Any]:
    """Keep supported settings absent from the editor when the method is unchanged."""
    if current.get("kind") not in ("numeric_bins", "histogram_bins"):
        # Calendar transformations have a unit, not a binning method.
        return current
    if not isinstance(previous, dict) or any(
        current.get(field) != previous.get(field) for field in ("kind", "method")
    ):
        return current
    hidden = {
        "equal_width": ("min", "max"),
        "fixed_width": ("origin", "min", "max"),
    }.get(current["method"], ())
    return {**current, **{field: previous[field] for field in hidden if field in previous}}


def compatible_aggregation_method(kind: str, measure: str, method: str | None) -> str:
    """Reset incompatible Code reductions without discarding valid settings."""
    selected = method or "any"
    if selected in {"count", "mean", "log_count"} and not (
        kind in {"scatter", "histogram", "summary_bar", "correlation_heatmap", "value_heatmap"}
        or kind == "distribution"
        or (kind == "heatmap" and measure == "correlation")
    ):
        return "any"
    return selected


_AGGREGATION_OPTIONS = [
    {"label": "Any observed positive (0 or 1)", "value": "any"},
    {"label": "Number of positive units per group", "value": "count"},
    {"label": "Positive count at least threshold", "value": "minimum"},
    {"label": "Proportion positive among observed units", "value": "mean"},
    {"label": "Natural log(1 + positive count)", "value": "log_count"},
]


def editor_recipe(kind: str, measure: str, slots: dict[str, Any],
                  configs: dict[str, Any], aggregate_level: str | None,
                  aggregate_method: str, aggregate_threshold: float | None,
                  *, legacy: bool = False,
                  extra_slots: dict[str, Any] | None = None,
                  continuous_options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Construct one recipe from role identities and role-scoped source choices."""
    from geometric_coder.analysis_continuous import CONTINUOUS_KINDS, ROLE_LAYOUT
    if kind == "bar_chart":
        selections = {**slots, **(extra_slots or {})}
        roles = {"values": [
            {"variable": _ref(raw, configs.get(f"values:{raw}", {}))}
            for raw in (selections.get("values") or [])
        ]}
        for role in ("group", "color"):
            raw = selections.get(role)
            if raw:
                conf = configs.get(f"{role}:{raw}", {})
                roles[role] = {"variable": _ref(raw, conf)}
                if conf.get("transform"):
                    roles[role]["transform"] = conf["transform"]
        options = continuous_options or {}
        result = {"schema_version": 2, "kind": kind, "roles": roles,
                  "statistic": options.get("bar_statistic", "count")}
        if aggregate_level and aggregate_level != "none":
            result["aggregation"] = {
                "level": aggregate_level, "method": aggregate_method,
                "threshold": 1 if aggregate_threshold is None else aggregate_threshold,
                "numeric_method": options.get("numeric_method", "mean"),
            }
        return result
    if kind in CONTINUOUS_KINDS:
        if legacy:
            raise ValueError("A legacy analysis cannot be silently converted to a numerical chart; create a new analysis.")
        selections = {**slots, **(extra_slots or {})}
        roles = {}
        for role in ROLE_LAYOUT[kind]:
            selected = selections.get(role)
            if role in ("group", "color") and not selected:
                continue
            if role == "values" and kind == "correlation_heatmap":
                roles[role] = [
                    {"variable": _ref(raw, configs.get(f"{role}:{raw}", {}))}
                    for raw in (selected or [])
                ]
                continue
            if not selected:
                continue
            conf = configs.get(f"{role}:{selected}", {})
            item = {"variable": _ref(selected, conf)}
            if role in ("group", "color", "rows", "columns") and conf.get("transform"):
                item["transform"] = conf["transform"]
            roles[role] = item
        options = continuous_options or {}
        recipe = {"schema_version": 2, "kind": kind, "roles": roles}
        if kind == "histogram":
            recipe["binning"] = options.get("binning", {
                "kind": "histogram_bins", "method": "equal_width", "bins": 20})
        if kind in ("summary_bar", "value_heatmap"):
            recipe["statistic"] = options.get("statistic", "mean")
        if kind == "scatter":
            recipe["max_points"] = options.get("max_points", 2000)
        if aggregate_level and aggregate_level != "none":
            recipe["aggregation"] = {
                "level": aggregate_level, "method": aggregate_method,
                "threshold": 1 if aggregate_threshold is None else aggregate_threshold,
                "numeric_method": options.get("numeric_method", "mean"),
            }
        return recipe
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
            "level": aggregate_level,
            "method": compatible_aggregation_method(kind, measure, aggregate_method),
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



def _source_coverage(result: dict[str, Any], html: Any) -> Any:
    """Visible provenance and denominator context for numerical chart results.

    Missing reasons are always counted over atomic source rows; when a chart
    reduces to parents its top-level observed/complete counts have a different
    grain and must not be subtracted from those atomic missing counts.
    """
    sources = result.get("sources")
    if not sources:
        return []
    coverage = result["coverage"]
    parent_grain = bool(result.get("aggregation"))
    grain = "parent groups" if parent_grain else "atomic observations"
    rows = []
    for index, source in enumerate(sources):
        descriptor = source["descriptor"]
        provenance = source.get("provenance", {})
        if descriptor["type"] == "metadata":
            detail = f"Metadata: {descriptor['field']} ({descriptor['value']})"
        elif descriptor["type"] == "scale":
            detail = (f"Scale #{descriptor['scale_id']} position; "
                      f"version #{provenance.get('scale_version_id', 'unknown')}; "
                      f"fit #{provenance.get('scaler_fit_id') or 'none'}; "
                      f"status: {provenance.get('fit_status', 'unknown')}")
        elif descriptor.get("source") == "labels":
            detail = f"Code #{descriptor['code_id']}: human judgments"
        else:
            kind = descriptor.get("predictor_kind", "classifier")
            fit = (provenance.get("committee_fit_id") if kind == "committee"
                   else provenance.get("classifier_fit_id"))
            members = provenance.get("member_classifier_fit_ids")
            detail = (f"Code #{descriptor['code_id']}: {descriptor['source']} "
                      f"{kind} #{descriptor['predictor_id']}; "
                      f"fit #{fit or 'none'}; "
                      f"status: {provenance.get('fit_status', 'unknown')}")
            if members:
                detail += f"; member fits: {', '.join(str(value) for value in members)}"
        reasons = source.get("missing_reasons", {})
        reason_text = ", ".join(f"{reason}: {count:,}"
                                for reason, count in sorted(reasons.items())) or "none"
        rows.append(html.Tr([
            html.Th(result["names"][index], scope="row"),
            html.Td(detail),
            html.Td(f"{coverage['observed'][index]:,} / {coverage['total']:,} {grain}"),
            html.Td(reason_text),
        ]))
    note = ("Observed and complete counts refer to parent groups; missing-reason "
            "counts refer to atomic observations (not parent groups)."
            if parent_grain else
            "Missing-reason counts refer to atomic observations; missing values are not zero.")
    return html.Div([
        html.P(
            f"{coverage['complete']:,} of {coverage['total']:,} {grain} complete",
            className="analysis-coverage-overview",
        ),
        html.Details([
            html.Summary("Data and sources"),
            html.Div([
                html.P(note),
                html.Table([
                    html.Thead(html.Tr([html.Th("Variable"), html.Th("Source / fit"),
                                        html.Th("Observed"), html.Th("Missing reasons")])),
                    html.Tbody(rows),
                ]),
            ], className="analysis-source-summary"),
        ], className="analysis-provenance-details"),
    ])


def analysis_tab(*, dcc: Any, html: Any, project: Any) -> Any:
    return html.Div([
        dcc.Store(id="analysis-refresh", data=0),
        dcc.Store(id="analysis-editing", data=None),
        dcc.Store(id="analysis-nav-pending", data=None),
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
            html.P("You have unsaved changes. Save them before switching analyses?"),
            html.Div([
                html.Button("Save and switch", id="analysis-nav-save", n_clicks=0),
                html.Button("Discard and switch", id="analysis-nav-discard", n_clicks=0),
                html.Button("Stay here", id="analysis-nav-stay", n_clicks=0),
            ], className="analysis-editor-actions"),
        ], id="analysis-nav-dialog", style={"display": "none"},
           className="analysis-navigation-dialog"),
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
                {"label": "Bar chart", "value": "bar_chart"},
                {"label": "Association heatmap", "value": "heatmap"},
                
                {"label": "Scatterplot", "value": "scatter"},
                {"label": "Continuous histogram", "value": "histogram"},
                
                {"label": "Numerical correlation heatmap", "value": "correlation_heatmap"},
                {"label": "Grouped numerical heatmap", "value": "value_heatmap"},
            ], value="bar_chart", clearable=False),
            html.Div(id="analysis-visualization-help"),
            html.Div([
                html.Label("Values"),
                dcc.Dropdown(id="analysis-role-values", multi=True, options=[], value=[]),
                html.Label("Group (optional)"),
                dcc.Dropdown(id="analysis-role-group", options=[], value=None),
                html.Label("Color (optional)"),
                dcc.Dropdown(id="analysis-bar-color", options=[], value=None),
                html.Label("Statistic"),
                dcc.Dropdown(id="analysis-bar-statistic", value="count", clearable=False,
                    options=[{"label": label, "value": value} for label, value in
                             (("Count", "count"), ("Sum", "sum"), ("Mean", "mean"), ("Median", "median"))]),
            ], id="analysis-bar-roles"),
            html.Div([
                html.Label("Values"),
                dcc.Dropdown(id="analysis-role-distribution", multi=True, options=[], value=[]),
            ], id="analysis-distribution-roles", style={"display": "none"}),
            html.Div([
                html.Label("Indicators"),
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
            html.Div([
                html.Label("X"), dcc.Dropdown(id="analysis-role-x", options=[]),
                html.Label("Y"), dcc.Dropdown(id="analysis-role-y", options=[]),
                html.Label("Color (optional)"),
                dcc.Dropdown(id="analysis-role-color", options=[]),
            ], id="analysis-scatter-roles", style={"display": "none"}),
            html.Div([
                html.Label("Values"),
                dcc.Dropdown(id="analysis-role-continuous-values", options=[], multi=True, value=[]),
                html.Div([
                    html.Label("Group"),
                    dcc.Dropdown(id="analysis-continuous-group", options=[]),
                ], id="analysis-continuous-group-box"),
                html.Div([
                    html.Label("Rows"),
                    dcc.Dropdown(id="analysis-role-rows", options=[]),
                ], id="analysis-continuous-row-box", style={"display":"none"}),
                html.Div([
                    html.Label("Columns"),
                    dcc.Dropdown(id="analysis-role-columns", options=[]),
                ], id="analysis-continuous-column-box", style={"display":"none"}),
            ], id="analysis-continuous-roles", style={"display": "none"}),
            html.Div([
                html.Label("Numerical statistic"), dcc.Dropdown(
                    id="analysis-statistic", options=[{"label":"Mean","value":"mean"},
                    {"label":"Median","value":"median"}], value="mean", clearable=False),
            ], id="analysis-statistic-options", style={"display":"none"}),
            html.Div([
                html.Label("Histogram bin method"),
                dcc.Dropdown(id="analysis-hist-method", options=[
                    {"label":"Equal width","value":"equal_width"},
                    {"label":"Quantiles","value":"quantile"},
                    {"label":"Fixed width","value":"fixed_width"},
                    {"label":"Explicit edges","value":"explicit"}], value="equal_width", clearable=False),
                html.Label("Bin count"), dcc.Input(id="analysis-hist-bins", type="number", min=1, max=100, step=1, value=20),
                html.Label("Fixed bin width"), dcc.Input(id="analysis-hist-width", type="number", min=0, step="any", value=1),
                html.Label("Explicit bin edges (comma-separated, ascending)"),
                dcc.Input(id="analysis-hist-edges", type="text", placeholder="0, 10, 20, 30"),
            ], id="analysis-hist-options", style={"display":"none"}),
            html.Div(id="analysis-role-settings"),
            html.Label("Aggregate observations (optional)"),
            dcc.Dropdown(id="analysis-aggregate-level", clearable=False,
                         options=[{"label": "No aggregation (atomic units)", "value": "none"}] + [
                             {"label": f"Group by {key} (and preceding keys)", "value": key}
                             for key in project.metadata.get("key_columns", [])[:-1]
                         ], value="none"),
            html.Div([
                html.Label("Aggregate Code values"),
                dcc.Dropdown(id="analysis-aggregate-method", value="any",
                             clearable=False, options=_AGGREGATION_OPTIONS),
                html.Label("Positive count threshold (threshold method only)"),
                dcc.Input(id="analysis-aggregate-threshold", type="number", min=0,
                          step=1, value=1),
                html.Label("Parent-grain reduction for numerical values"),
                dcc.Dropdown(id="analysis-parent-numeric-method",
                    options=[{"label":"Mean","value":"mean"},{"label":"Median","value":"median"}],
                    value="mean", clearable=False),
                html.P("Binary Codes use the selected count/proportion reduction; numerical values "
                       "use the numerical reduction. Grouping metadata must be constant in each parent."),

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
        html.Div(id="analysis-provenance"),
    ], className="analysis-manager")


def register_analysis_callbacks(app: Any, project: Any) -> None:
    from dash import ALL, Input, Output, State, callback_context, dcc, html, no_update
    from geometric_coder.ui.app import _empty_figure
    from geometric_coder.analyses import compute_analysis

    @app.callback(
        Output("analysis-kind", "options"),
        Input("analysis-editing", "data"),
    )
    def visualization_options(editing: dict[str, Any] | None) -> list[dict[str, str]]:
        # Present one Bar chart choice for new analyses, without breaking the
        # archived recipe type displayed when editing an existing saved chart.
        options = [
            {"label": "Bar chart", "value": "bar_chart"},
            {"label": "Association heatmap", "value": "heatmap"},
            {"label": "Scatterplot", "value": "scatter"},
            {"label": "Continuous histogram", "value": "histogram"},
            {"label": "Numerical correlation heatmap", "value": "correlation_heatmap"},
            {"label": "Grouped numerical heatmap", "value": "value_heatmap"},
        ]
        old = (editing or {}).get("original_spec") or {}
        kind = old.get("kind")
        if kind in ("bar", "distribution", "summary_bar"):
            options.append({"label": "Bar chart (existing format)", "value": kind})
        return options

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
        Output("analysis-role-x", "options"),
        Output("analysis-role-y", "options"),
        Output("analysis-role-color", "options"),
        Output("analysis-role-continuous-values", "options"),
        Output("analysis-role-rows", "options"),
        Output("analysis-role-columns", "options"),
        Output("analysis-continuous-group", "options"),
        Output("analysis-bar-color", "options"),
        Input("workspace-tabs", "value"),
        Input("explore-code-refresh-store", "data"),
        Input("focus-refresh-store", "data"),
        Input("analysis-editor", "style"),
    )
    def refresh_variables(workspace: str, _code_refresh: Any, _focus_refresh: Any,
                          _editor_style: Any) -> Any:
        if workspace != "analyses":
            return (no_update,) * 12
        choices = variable_options(project)
        codes = [c for c in choices if c["value"].startswith("code:")]
        return choices, codes, choices, choices, choices, choices, choices, choices, choices, choices, choices, choices

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
        Output("analysis-role-x", "value"),
        Output("analysis-role-y", "value"),
        Output("analysis-role-color", "value"),
        Output("analysis-role-continuous-values", "value"),
        Output("analysis-role-rows", "value"),
        Output("analysis-role-columns", "value"),
        Output("analysis-statistic", "value"),
        Output("analysis-hist-method", "value"),
        Output("analysis-hist-bins", "value"),
        Output("analysis-hist-width", "value"),
        Output("analysis-continuous-group", "value"),
        Output("analysis-parent-numeric-method", "value"),
        Output("analysis-hist-edges", "value"),
        Output("analysis-bar-color", "value"),
        Output("analysis-bar-statistic", "value"),
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
                 "New analysis", "", "bar_chart", "count", [], None, [], [], {},
                 "none", "any", 1, "", None, None, None, [], None, None, "mean", "equal_width", 20, 1, None, "mean", "", None, "count")
        if trigger == "analysis-cancel":
            return ({"display": "none"}, None, "", "", "bar_chart", "count", [],
                    None, [], [], {}, "none", "any", 1, "", None, None, None, [], None, None,
                    "mean", "equal_width", 20, 1, None, "mean", "", None, "count")
        if trigger == "analysis-new":
            return empty
        if selected is None:
            return (no_update,) * 30
        record = project.analysis(int(selected))
        spec = record["spec"]
        slots = _slots(spec)
        aggregate = spec.get("aggregation", {})
        return (
            {"display": "block"},
            {"id": int(selected) if trigger == "analysis-edit" else None,
             "schema_version": spec.get("schema_version", 1), "original_spec": spec,
             "original_name": record["name"] if trigger == "analysis-edit" else record["name"] + " (copy)"},
            "Edit analysis" if trigger == "analysis-edit" else "Duplicate analysis",
            record["name"] if trigger == "analysis-edit" else record["name"] + " (copy)",
            spec["kind"], spec.get("measure", "count"),
            slots.get("values", []) if spec["kind"] in ("bar", "bar_chart") else [],
            slots.get("group") if spec["kind"] in ("bar", "bar_chart") else None,
            slots.get("values", []) if spec["kind"] == "distribution" else [],
            slots.get("indicators", []) if spec["kind"] == "heatmap" else [],
            _initial_config(spec), aggregate.get("level", "none"),
            aggregate.get("method", "any"), aggregate.get("threshold", 1), "",
            slots.get("x"), slots.get("y"), slots.get("color"),
            (slots.get("values", []) if isinstance(slots.get("values", []), list) else [slots["values"]]) if spec["kind"] in ("histogram", "summary_bar", "correlation_heatmap", "value_heatmap") else [],
            slots.get("rows"), slots.get("columns"),
            spec.get("statistic", "mean"), spec.get("binning", {}).get("method", "equal_width"),
            spec.get("binning", {}).get("bins", 20), spec.get("binning", {}).get("width", 1),
            slots.get("group") if spec["kind"] in ("histogram", "summary_bar") else None,
            aggregate.get("numeric_method", "mean"),
            ", ".join(str(v) for v in spec.get("binning", {}).get("edges", [])),
            slots.get("color") if spec["kind"] == "bar_chart" else None,
            spec.get("statistic", "count") if spec["kind"] == "bar_chart" else "count",
        )

    @app.callback(
        Output("analysis-bar-roles", "style"),
        Output("analysis-distribution-roles", "style"),
        Output("analysis-heatmap-roles", "style"),
        Output("analysis-measure-options", "style"),
        Output("analysis-visualization-help", "children"),
        Output("analysis-scatter-roles", "style"),
        Output("analysis-continuous-roles", "style"),
        Output("analysis-statistic-options", "style"),
        Output("analysis-hist-options", "style"),
        Output("analysis-continuous-group-box", "style"),
        Output("analysis-continuous-row-box", "style"),
        Output("analysis-continuous-column-box", "style"),
        Input("analysis-kind", "value"),
        Input("analysis-aggregate-level", "value"),
        Input("analysis-aggregate-method", "value"),
    )
    def chart_controls(kind: str, level: str | None, method: str | None) -> tuple[Any, ...]:
        visible = {"display": "block"}
        hidden = {"display": "none"}
        if kind in ("scatter", "histogram", "summary_bar", "correlation_heatmap", "value_heatmap"):
            help_text = ("Continuous and mixed exploratory values; missing sources remain missing. "
                         "Numeric and calendar grouping require explicit role transformations.")
        elif kind == "distribution":
            help_text = ("Atomic units contribute binary observed Code values (0/1). "
                         "Parent grouping distributes reduced Code values by group."
                         if level and level != "none" else
                         "Counts observed negative (0) and positive (1) Code values; "
                         "not a general numeric histogram.")
        elif kind == "heatmap":
            help_text = ("A symmetric matrix of positive co-occurrence, Jaccard "
                         "similarity, or pairwise-complete correlations. "
                         "Categorical metadata expands into indicators.")
        elif kind == "bar_chart":
            help_text = "Choose a statistic, then compare values across an optional group and color."
        else:
            help_text = "Counts positive Code assignments, optionally by categorical metadata."
        return (
            visible if kind in ("bar", "bar_chart") else hidden,
            visible if kind == "distribution" else hidden,
            visible if kind == "heatmap" else hidden,
            visible if kind == "heatmap" else hidden,
            help_text,
            visible if kind == "scatter" else hidden,
            visible if kind in ("histogram", "summary_bar", "correlation_heatmap", "value_heatmap") else hidden,
            visible if kind in ("summary_bar", "value_heatmap") else hidden,
            visible if kind == "histogram" else hidden,
            visible if kind in ("histogram", "summary_bar") else hidden,
            visible if kind == "value_heatmap" else hidden,
            visible if kind == "value_heatmap" else hidden,
        )

    @app.callback(Output("analysis-aggregation-options", "style"),
                  Input("analysis-aggregate-level", "value"))
    def show_aggregation(level: str | None) -> dict[str, str]:
        return {"display": "block"} if level and level != "none" else {"display": "none"}

    @app.callback(
        Output("analysis-aggregate-method", "options"),
        Input("analysis-kind", "value"),
        Input("analysis-measure", "value"),
    )
    def aggregation_method_choices(kind: str, measure: str) -> list[dict[str, str]]:
        """Offer only reductions that the current chart/statistic can execute."""
        return [
            option for option in _AGGREGATION_OPTIONS
            if compatible_aggregation_method(kind, measure, option["value"]) == option["value"]
        ]

    @app.callback(
        Output("analysis-aggregate-method", "value", allow_duplicate=True),
        Input("analysis-kind", "value"),
        Input("analysis-measure", "value"),
        State("analysis-aggregate-method", "value"),
        State("analysis-editing", "data"),
        prevent_initial_call=True,
    )
    def reset_incompatible_aggregation(
        kind: str, measure: str, method: str | None, editing: Any,
    ) -> Any:
        if editing is None:
            return no_update
        compatible = compatible_aggregation_method(kind, measure, method)
        return compatible if compatible != method else no_update

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
        Input("analysis-role-x", "value"),
        Input("analysis-role-y", "value"),
        Input("analysis-role-color", "value"),
        Input("analysis-role-continuous-values", "value"),
        Input("analysis-role-rows", "value"),
        Input("analysis-role-columns", "value"),
        Input("analysis-continuous-group", "value"),
        Input("analysis-bar-color", "value"),
        Input("analysis-editing", "data"),
        State("analysis-role-config", "data"),
    )
    def role_settings(kind: str, values: list[str] | None, group: str | None,
                      distribution: list[str] | None, indicators: list[str] | None,
                      x: str | None, y: str | None, color: str | None,
                      continuous_values: list[str] | None, rows: str | None,
                      columns: str | None, continuous_group: str | None,
                      bar_color: str | None, editing: Any, configs: dict[str, Any] | None) -> Any:
        if editing is None:
            return []
        if kind in ("bar", "bar_chart"):
            slots = {"values": values or [], "group": [group] if group else [], "color": [bar_color] if kind == "bar_chart" and bar_color else []}
        elif kind == "distribution":
            slots = {"values": distribution or []}
        elif kind == "heatmap":
            slots = {"indicators": indicators or []}
        elif kind == "scatter":
            slots = {"x": [x] if x else [], "y": [y] if y else [],
                     "color": [color] if color else []}
        else:
            slots = {"values": continuous_values or [] if kind == "correlation_heatmap" else (continuous_values or [])[:1],
                     "group": [continuous_group] if kind in ("histogram", "summary_bar") and continuous_group else [],
                     "rows": [rows] if kind == "value_heatmap" and rows else [],
                     "columns": [columns] if kind == "value_heatmap" and columns else []}
        nodes = []
        for role, selected in slots.items():
            for raw in selected:
                family, identity = raw.split(":", 1)
                key = f"{role}:{raw}"
                config = (configs or {}).get(key, {})
                display = next((item["label"] for item in variable_options(project) if item["value"] == raw), raw)
                family_name, _, value_name = display.partition(": ")
                readable = f"{role.title()}: {value_name} ({family_name})" if value_name else f"{role.title()}: {display}"
                controls = [html.H4(readable)]
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
                                         {"label": "Binary (0/1)", "value": "binary"},
                                         {"label": "Probability", "value": "probability"},
                                     ], value=config.get("value", "binary"), clearable=False),
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
                                {"label": "Categorical", "value": "categorical"},
                                {"label": "Numerical", "value": "numeric"},
                                {"label": "Date/time", "value": "temporal"},
                            ], value=config.get("value", "categorical"),
                            clearable=False,
                        ),
                        html.P("Temporal values retain date-only, naive, and offset-aware distinctions. "
                               "Aware timestamps group by UTC calendar; ISO weeks start Monday."),

                    ])
                else:
                    status = project.scale_model_status(int(identity))
                    controls.extend([
                        html.P("Predicted continuous position; no human-assigned values."),
                        html.P(f"Scale fit status: {status['status']}. "
                               "Untrained, stale, or ineligible positions cannot be "
                               "silently replaced by human ratings or categorical bins."),
                    ])
                if role in ("group", "color", "rows", "columns") and kind in (
                    "scatter", "histogram", "summary_bar", "value_heatmap", "bar_chart",
                ):
                    transform = config.get("transform", {})
                    controls.extend([
                        html.Label("Grouping transformation"),
                        dcc.Dropdown(id={"type":"analysis-role-transform","index":key},
                                     options=[{"label":"No transformation (categorical / binary)","value":"none"},
                                              {"label":"Calendar day","value":"day"},
                                              {"label":"Calendar week","value":"week"},
                                              {"label":"Calendar month","value":"month"},
                                              {"label":"Calendar quarter","value":"quarter"},
                                              {"label":"Calendar year","value":"year"},
                                              {"label":"Equal-width numeric bins","value":"equal_width"},
                                              {"label":"Quantile numeric bins","value":"quantile"},
                                              {"label":"Fixed-width numeric bins","value":"fixed_width"},
                                              {"label":"Explicit numerical intervals","value":"explicit"}],
                                     value=(transform.get("unit") if transform.get("kind") == "calendar"
                                            else transform.get("method", "none")),clearable=False),
                        html.Label("Numeric bin count"),
                        dcc.Input(id={"type":"analysis-role-bins","index":key},
                                  type="number",min=1,max=100,step=1,value=transform.get("bins",10)),
                        html.Label("Fixed bin width"),
                        dcc.Input(id={"type":"analysis-role-width","index":key},
                                  type="number",min=0,step="any",value=transform.get("width",1)),
                        html.Label("Explicit bin edges (comma-separated)"),
                        dcc.Input(id={"type":"analysis-role-edges","index":key},
                                  type="text",value=", ".join(str(v) for v in transform.get("edges", []))),
                    ])
                nodes.append(html.Div(controls, className="analysis-variable-settings"))
        return nodes

    @app.callback(
        Output("analysis-role-config", "data", allow_duplicate=True),
        Input({"type": "analysis-role-source", "index": ALL}, "value"),
        Input({"type": "analysis-role-value", "index": ALL}, "value"),
        Input({"type": "analysis-role-predictor", "index": ALL}, "value"),
        Input({"type": "analysis-role-transform", "index": ALL}, "value"),
        Input({"type": "analysis-role-bins", "index": ALL}, "value"),
        Input({"type": "analysis-role-width", "index": ALL}, "value"),
        Input({"type": "analysis-role-edges", "index": ALL}, "value"),
        State({"type": "analysis-role-source", "index": ALL}, "id"),
        State({"type": "analysis-role-value", "index": ALL}, "id"),
        State({"type": "analysis-role-predictor", "index": ALL}, "id"),
        State({"type": "analysis-role-transform", "index": ALL}, "id"),
        State({"type": "analysis-role-bins", "index": ALL}, "id"),
        State({"type": "analysis-role-width", "index": ALL}, "id"),
        State({"type": "analysis-role-edges", "index": ALL}, "id"),
        State("analysis-role-config", "data"),
        prevent_initial_call=True,
    )
    def preserve_role_configs(
        sources: list[str], values: list[str],
        predictors: list[str | None], transforms: list[str], bins: list[int],
        widths: list[float], edges: list[str], source_ids: list[Any], value_ids: list[Any],
        predictor_ids: list[Any], transform_ids: list[Any], bin_ids: list[Any],
        width_ids: list[Any], edge_ids: list[Any],
        existing: dict[str, Any] | None,
    ) -> Any:
        result = {key: dict(val) for key, val in (existing or {}).items()}
        for component_ids, component_values, field in [
            (source_ids, sources, "source"), (value_ids, values, "value"),
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
        by_key = {item["index"]: {"method": mode} for item, mode in zip(transform_ids, transforms)}
        for item, value in zip(bin_ids, bins):
            by_key.setdefault(item["index"], {})["bins"] = value
        for item, value in zip(width_ids, widths):
            by_key.setdefault(item["index"], {})["width"] = value
        for item, value in zip(edge_ids, edges):
            by_key.setdefault(item["index"], {})["edges"] = value
        for key, choice in by_key.items():
            mode = choice.get("method", "none")
            if mode in ("day", "week", "month", "quarter", "year"):
                transform = {"kind": "calendar", "unit": mode}
            elif mode in ("equal_width", "quantile"):
                transform = {"kind": "numeric_bins", "method": mode, "bins": choice.get("bins", 10)}
            elif mode == "fixed_width":
                transform = {"kind": "numeric_bins", "method": mode, "width": choice.get("width", 1), "origin": 0}
            elif mode == "explicit":
                raw = (choice.get("edges") or "").strip()
                try:
                    numbers = [float(v.strip()) for v in raw.split(",") if v.strip()]
                except ValueError:
                    # Let the recipe validation issue a targeted invalid-bin message.
                    numbers = []
                transform = {"kind": "numeric_bins", "method": "explicit", "edges": numbers}
            else:
                transform = None
            conf = result.setdefault(key, {})
            if transform is None:
                conf.pop("transform", None)
            else:
                conf["transform"] = _preserve_hidden_bin_settings(transform, conf.get("transform"))
        return result if result != (existing or {}) else no_update

    def current_slots(kind: str, values: Any, group: Any,
                      distribution: Any, indicators: Any,
                      x: Any, y: Any, color: Any, continuous_values: Any,
                      rows: Any, columns: Any, continuous_group: Any, bar_color: Any) -> dict[str, Any]:
        if kind in ("bar", "bar_chart"):
            return {"values": values or [], "group": group, "color": bar_color if kind == "bar_chart" else None}
        if kind == "distribution":
            return {"values": distribution or []}
        if kind == "heatmap":
            return {"indicators": indicators or []}
        if kind == "scatter":
            return {"x": x, "y": y, "color": color}
        if kind == "histogram":
            return {"values": (continuous_values or [None])[0], "group": continuous_group}
        if kind == "summary_bar":
            return {"values": (continuous_values or [None])[0], "group": continuous_group}
        if kind == "value_heatmap":
            return {"values": (continuous_values or [None])[0], "rows": rows, "columns": columns}
        return {"values": continuous_values or []}

    def run_recipe(kind: str, measure: str, values: Any, group: Any,
                   distribution: Any, indicators: Any,
                   configs: dict[str, Any] | None, level: str | None,
                   method: str, threshold: float | None,
                   editing: dict[str, Any], x: Any, y: Any, color: Any,
                   continuous_values: Any, rows: Any, columns: Any,
                   continuous_group: Any, statistic: Any,
                   hist_method: Any, hist_bins: Any, hist_width: Any,
                   parent_numeric_method: Any, hist_edges: Any,
                   bar_color: Any, bar_statistic: Any,
                   live_sources: Any, live_source_ids: Any,
                   live_predictors: Any, live_predictor_ids: Any,
                   live_values: Any, live_value_ids: Any,
                   live_transforms: Any, live_transform_ids: Any,
                   live_bins: Any, live_bin_ids: Any,
                   live_widths: Any, live_width_ids: Any,
                   live_edges: Any, live_edge_ids: Any) -> dict[str, Any]:
        legacy = editing.get("schema_version", 2) == 1
        if kind in ("histogram", "summary_bar", "value_heatmap") and len(continuous_values or []) != 1:
            raise ValueError("Choose exactly one numeric Values variable for this visualization.")
        from geometric_coder.analysis_continuous import CONTINUOUS_KINDS
        if (kind in CONTINUOUS_KINDS or kind == "bar_chart") and legacy:
            raise ValueError("Legacy analyses cannot switch to a numerical chart; create a new analysis.")
        bins = {"kind": "histogram_bins", "method": hist_method}
        if hist_method == "fixed_width":
            bins.update(width=hist_width, origin=0)
        elif hist_method == "explicit":
            try:
                bins["edges"] = [float(v.strip()) for v in (hist_edges or "").split(",") if v.strip()]
            except ValueError:
                raise ValueError("Explicit histogram bin edges must be comma-separated numbers.")
        else:
            bins["bins"] = hist_bins
        original = editing.get("original_spec")
        original = original if isinstance(original, dict) and original.get("kind") == kind else {}
        bins = _preserve_hidden_bin_settings(bins, original.get("binning"))
        original_transforms = {
            f"{role}:{_identity(entry['variable'])}": entry["transform"]
            for role, raw in original.get("roles", {}).items()
            for entry in (raw if isinstance(raw, list) else [raw])
            if "transform" in entry
        }
        # The recipe consumes the current mounted role inputs directly. The
        # Store callback may still be resolving a preceding UI change when
        # Preview/Save is clicked; its snapshot alone is not authoritative.
        resolved = _reconcile_live_source_choices(
            configs, live_sources, live_source_ids, live_predictors, live_predictor_ids,
        )
        for item, value in zip(live_value_ids or [], live_values or []):
            resolved.setdefault(item["index"], {})["value"] = value
        settings_by_key = {
            item["index"]: {"method": mode}
            for item, mode in zip(live_transform_ids or [], live_transforms or [])
        }
        for names, entries, field in (
            (live_bin_ids, live_bins, "bins"),
            (live_width_ids, live_widths, "width"),
            (live_edge_ids, live_edges, "edges"),
        ):
            for item, value in zip(names or [], entries or []):
                settings_by_key.setdefault(item["index"], {})[field] = value
        for key, setting in settings_by_key.items():
            mode = setting.get("method", "none")
            if mode in ("day", "week", "month", "quarter", "year"):
                transform = {"kind": "calendar", "unit": mode}
            elif mode in ("equal_width", "quantile"):
                transform = {"kind": "numeric_bins", "method": mode, "bins": setting.get("bins", 10)}
            elif mode == "fixed_width":
                transform = {"kind": "numeric_bins", "method": mode, "width": setting.get("width", 1), "origin": 0}
            elif mode == "explicit":
                try:
                    edges = [float(v.strip()) for v in (setting.get("edges") or "").split(",") if v.strip()]
                except ValueError:
                    edges = []
                transform = {"kind": "numeric_bins", "method": mode, "edges": edges}
            else:
                transform = None
            if transform is None:
                resolved.setdefault(key, {}).pop("transform", None)
            else:
                config = resolved.setdefault(key, {})
                config["transform"] = _preserve_hidden_bin_settings(
                    transform, original_transforms.get(key, config.get("transform"))
                )
        all_slots = current_slots(kind, values, group, distribution, indicators, x, y,
                                  color, continuous_values, rows, columns, continuous_group, bar_color)
        recipe = editor_recipe(
            kind, measure if kind == "heatmap" else "count",
            all_slots, resolved, level, method, threshold, legacy=legacy,
            extra_slots=all_slots,
            continuous_options={"statistic": statistic or "mean", "bar_statistic": bar_statistic or "count", "binning": bins,
                                "numeric_method": parent_numeric_method or "mean",
                                "max_points": original.get("max_points", 2000)},
        )
        if legacy:
            for var in recipe["variables"]:
                if var["type"] == "code":
                    if var.get("predictor_kind") == "committee":
                        raise ValueError("Existing v1 analyses cannot store committee sources.")
                    var.pop("predictor_kind", None)
                elif var["type"] == "metadata":
                    if var.get("value", "categorical") != "categorical":
                        raise ValueError("Numeric/temporal metadata requires a numerical chart.")
                    var.pop("value", None)
                    var.pop("numeric_strings", None)
                else:
                    raise ValueError("Legacy analyses cannot store Scale positions.")
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
        State("analysis-role-x", "value"),
        State("analysis-role-y", "value"),
        State("analysis-role-color", "value"),
        State("analysis-role-continuous-values", "value"),
        State("analysis-role-rows", "value"),
        State("analysis-role-columns", "value"),
        State("analysis-continuous-group", "value"),
        State("analysis-statistic", "value"),
        State("analysis-hist-method", "value"),
        State("analysis-hist-bins", "value"),
        State("analysis-hist-width", "value"),
        State("analysis-parent-numeric-method", "value"),
        State("analysis-hist-edges", "value"),
        State("analysis-bar-color", "value"),
        State("analysis-bar-statistic", "value"),
        State({"type":"analysis-role-source","index":ALL}, "value"),
        State({"type":"analysis-role-source","index":ALL}, "id"),
        State({"type":"analysis-role-predictor","index":ALL}, "value"),
        State({"type":"analysis-role-predictor","index":ALL}, "id"),
        State({"type":"analysis-role-value","index":ALL}, "value"),
        State({"type":"analysis-role-value","index":ALL}, "id"),
        State({"type":"analysis-role-transform","index":ALL}, "value"),
        State({"type":"analysis-role-transform","index":ALL}, "id"),
        State({"type":"analysis-role-bins","index":ALL}, "value"),
        State({"type":"analysis-role-bins","index":ALL}, "id"),
        State({"type":"analysis-role-width","index":ALL}, "value"),
        State({"type":"analysis-role-width","index":ALL}, "id"),
        State({"type":"analysis-role-edges","index":ALL}, "value"),
        State({"type":"analysis-role-edges","index":ALL}, "id"),
    ]

    @app.callback(
        Output("analysis-refresh", "data"),
        Output("analysis-select", "value"),
        Output("analysis-select", "options", allow_duplicate=True),
        Output("analysis-status", "children"),
        Output("analysis-editor", "style", allow_duplicate=True),
        Output("analysis-editing", "data", allow_duplicate=True),
        Input("analysis-save", "n_clicks"),
        State("analysis-name", "value"),
        *state_args, State("analysis-refresh", "data"),
        prevent_initial_call=True,
    )
    def save(_click: int, name: str | None, editing: Any, kind: str, measure: str,
             values: Any, group: Any, distribution: Any, indicators: Any,
             configs: Any, level: Any, method: str, threshold: Any,
             x: Any, y: Any, color: Any, continuous_values: Any,
             rows: Any, columns: Any, continuous_group: Any, statistic: Any,
             hist_method: Any, hist_bins: Any, hist_width: Any,
             parent_numeric_method: Any, hist_edges: Any,
             bar_color: Any, bar_statistic: Any,
             live_sources: Any, live_source_ids: Any,
             live_predictors: Any, live_predictor_ids: Any,
             live_values: Any, live_value_ids: Any, live_transforms: Any,
             live_transform_ids: Any, live_bins: Any, live_bin_ids: Any,
             live_widths: Any, live_width_ids: Any, live_edges: Any,
             live_edge_ids: Any,
             revision: int | None) -> tuple[Any, ...]:
        if editing is None:
            return (no_update,) * 6
        try:
            spec = run_recipe(kind, measure, values, group, distribution, indicators,
                              configs, level, method, threshold, editing, x, y, color,
                              continuous_values, rows, columns, continuous_group, statistic,
                              hist_method, hist_bins, hist_width, parent_numeric_method,
                              hist_edges, bar_color, bar_statistic, live_sources, live_source_ids, live_predictors,
                              live_predictor_ids, live_values, live_value_ids, live_transforms,
                              live_transform_ids, live_bins, live_bin_ids, live_widths,
                              live_width_ids, live_edges, live_edge_ids)
            analysis_id = project.save_analysis(name or "", spec, analysis_id=editing.get("id"))
        except (ValueError, KeyError) as exc:
            return no_update, no_update, no_update, str(exc), no_update, no_update
        options = [{"label": row["name"], "value": row["analysis_id"]}
                   for row in project.analyses()]
        return int(revision or 0) + 1, analysis_id, options, "", {"display": "none"}, None

    @app.callback(
        Output("analysis-figure", "figure", allow_duplicate=True),
        Output("analysis-provenance", "children", allow_duplicate=True),
        Output("analysis-status", "children", allow_duplicate=True),
        Input("analysis-preview", "n_clicks"),
        *state_args,
        prevent_initial_call=True,
    )
    def preview(_click: int, editing: Any, kind: str, measure: str,
                values: Any, group: Any, distribution: Any, indicators: Any,
                configs: Any, level: Any, method: str, threshold: Any,
                x: Any, y: Any, color: Any, continuous_values: Any,
                rows: Any, columns: Any, continuous_group: Any, statistic: Any,
                hist_method: Any, hist_bins: Any, hist_width: Any,
                parent_numeric_method: Any, hist_edges: Any,
                bar_color: Any, bar_statistic: Any,
                live_sources: Any, live_source_ids: Any,
                live_predictors: Any, live_predictor_ids: Any,
                live_values: Any, live_value_ids: Any, live_transforms: Any,
                live_transform_ids: Any, live_bins: Any, live_bin_ids: Any,
                live_widths: Any, live_width_ids: Any, live_edges: Any,
                live_edge_ids: Any) -> tuple[Any, str]:
        if editing is None:
            return no_update, no_update, ""
        try:
            spec = run_recipe(kind, measure, values, group, distribution, indicators,
                              configs, level, method, threshold, editing, x, y, color,
                              continuous_values, rows, columns, continuous_group, statistic,
                              hist_method, hist_bins, hist_width, parent_numeric_method,
                              hist_edges, bar_color, bar_statistic, live_sources, live_source_ids, live_predictors,
                              live_predictor_ids, live_values, live_value_ids, live_transforms,
                              live_transform_ids, live_bins, live_bin_ids, live_widths,
                              live_width_ids, live_edges, live_edge_ids)
            result = compute_analysis(project, spec)
            fig = analysis_figure(result)
            if result.get("warnings"):
                fig.add_annotation(
                    text=" | ".join(result["warnings"]), xref="paper", yref="paper",
                    x=0, y=1.14, xanchor="left", showarrow=False,
                    font={"color": "#9a6510", "size": 11},
                )
            return fig, _source_coverage(result, html), ""
        except (ValueError, KeyError) as exc:
            return no_update, no_update, f"Preview unavailable: {exc}"

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
        Output("analysis-provenance", "children"),
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
            return no_update, no_update
        if selected is None:
            return _empty_figure("Select or create an analysis."), []
        try:
            result = project.run_analysis(int(selected))
            fig = analysis_figure(result)
            if result.get("warnings"):
                fig.add_annotation(
                    text=" | ".join(result["warnings"]), xref="paper", yref="paper",
                    x=0, y=1.14, xanchor="left", showarrow=False,
                    font={"color": "#9a6510", "size": 11},
                )
            return fig, _source_coverage(result, html)
        except (ValueError, KeyError) as exc:
            return _empty_figure(f"Analysis unavailable: {exc}"), []


    def _draft_recipe(draft: tuple[Any, ...]) -> dict[str, Any]:
        """Use the same live form sources as Preview and Save for dirty checks."""
        return run_recipe(*draft[1:11], draft[0], *draft[11:])

    @app.callback(
        Output("analysis-editor", "style", allow_duplicate=True),
        Output("analysis-editing", "data", allow_duplicate=True),
        Output("analysis-nav-dialog", "style"),
        Output("analysis-nav-pending", "data"),
        Input("analysis-select", "value"),
        State("analysis-editor", "style"),
        State("analysis-name", "value"),
        *state_args,
        prevent_initial_call=True,
    )
    def selection_while_editing(
        selected: int | None, style: dict[str, Any] | None, name: str | None,
        *draft: Any,
    ) -> tuple[Any, ...]:
        editing = draft[0]
        if not editing or not style or style.get("display") == "none":
            return no_update, no_update, no_update, no_update
        old_id = editing.get("id")
        if old_id is not None and selected == old_id:
            return no_update, no_update, no_update, no_update
        original = editing.get("original_spec")
        if original:
            try:
                from geometric_coder.analyses import validate_analysis
                modified = validate_analysis(project, _draft_recipe(draft))
                previous = validate_analysis(project, original)
                dirty = modified != previous or (name or "") != editing.get("original_name", "")
            except (ValueError, KeyError, TypeError):
                # An invalid draft still contains researcher input and must
                # never be discarded without confirmation.
                dirty = True
        else:
            dirty = bool((name or "").strip()) or bool(draft[3]) or bool(draft[5])
        if not dirty:
            return {"display": "none"}, None, {"display": "none"}, None
        return (
            no_update, no_update, {"display": "block"},
            {"from": old_id, "to": selected},
        )

    @app.callback(
        Output("analysis-refresh", "data", allow_duplicate=True),
        Output("analysis-select", "value", allow_duplicate=True),
        Output("analysis-editor", "style", allow_duplicate=True),
        Output("analysis-editing", "data", allow_duplicate=True),
        Output("analysis-nav-dialog", "style", allow_duplicate=True),
        Output("analysis-nav-pending", "data", allow_duplicate=True),
        Output("analysis-status", "children", allow_duplicate=True),
        Input("analysis-nav-save", "n_clicks"),
        Input("analysis-nav-discard", "n_clicks"),
        Input("analysis-nav-stay", "n_clicks"),
        State("analysis-nav-pending", "data"),
        State("analysis-name", "value"),
        *state_args,
        State("analysis-refresh", "data"),
        prevent_initial_call=True,
    )
    def resolve_navigation(
        _save: int, _discard: int, _stay: int,
        pending: dict[str, Any] | None, name: str | None,
        *values: Any,
    ) -> tuple[Any, ...]:
        if pending is None:
            return (no_update,) * 7
        *draft, revision = values
        choice = callback_context.triggered_id
        hidden = {"display": "none"}
        if choice == "analysis-nav-stay":
            return no_update, pending.get("from"), no_update, no_update, hidden, None, ""
        if choice == "analysis-nav-discard":
            return no_update, no_update, hidden, None, hidden, None, ""
        if choice != "analysis-nav-save":
            return (no_update,) * 7
        editing = draft[0]
        try:
            spec = _draft_recipe(tuple(draft))
            project.save_analysis(name or "", spec, analysis_id=editing.get("id"))
        except (ValueError, KeyError, TypeError) as exc:
            # Stay on the draft, with the prompt open and the error visible.
            return no_update, no_update, no_update, no_update, no_update, no_update, str(exc)
        return int(revision or 0) + 1, no_update, hidden, None, hidden, None, ""
