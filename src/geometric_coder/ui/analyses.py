"""Project-wide saved analyses manager and live Plotly display."""
from __future__ import annotations

from typing import Any

from geometric_coder.analyses import analysis_figure


def variable_options(project: Any) -> list[dict[str, str]]:
    """Read available codes at callback time, not at app startup."""
    return [
        {"label": f"Code: {row['name']}", "value": f"code:{row['code_id']}"}
        for row in project.codes()
    ] + [
        {"label": f"Metadata: {field}", "value": f"metadata:{field}"}
        for field in project.metadata.get("metadata_columns", [])
    ]


def analysis_tab(*, dcc: Any, html: Any, project: Any) -> Any:
    return html.Div([
        dcc.Store(id="analysis-refresh", data=0),
        dcc.Store(id="analysis-editing", data=None),
        dcc.Store(id="analysis-source-defaults", data=[]),
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
                {"label": "Bar plot", "value": "bar"},
                {"label": "Heatmap", "value": "heatmap"},
                {"label": "Histogram", "value": "distribution"},
            ], value="bar", clearable=False),
            html.Div(id="analysis-visualization-help"),
            html.Label("Measure"),
            dcc.Dropdown(id="analysis-measure", options=[
                {"label": "Positive counts", "value": "count"},
                {"label": "Jaccard similarity (heatmaps)", "value": "jaccard"},
                {"label": "Pearson correlation (heatmaps)", "value": "correlation"},
            ], value="count", clearable=False),
            html.Label("Variables"),
            dcc.Dropdown(id="analysis-vars", options=[], value=[], multi=True,
                         placeholder="Add codes and, for bar plots, one metadata field"),
            html.Div(id="analysis-code-settings"),
            html.Label("Aggregate observations (optional)"),
            dcc.Dropdown(id="analysis-aggregate-level", clearable=False,
                         options=[{"label": "No aggregation (atomic units)", "value": "none"}] + [
                             {"label": f"Group by {key} (and preceding keys)", "value": key}
                             for key in project.metadata.get("key_columns", [])[:-1]
                         ], value="none"),
            html.Div([
                html.Label("Aggregate code values"),
                dcc.Dropdown(id="analysis-aggregate-method", clearable=False, value="any",
                             options=[
                                 {"label": "Any observed positive (0 or 1)", "value": "any"},
                                 {"label": "Number of positive units per group", "value": "count"},
                                 {"label": "Positive count at least threshold", "value": "minimum"},
                                 {"label": "Proportion positive among observed units", "value": "mean"},
                                 {"label": "Natural log(1 + positive count)", "value": "log_count"},
                             ]),
                html.Label("Positive count threshold (threshold method only)"),
                dcc.Input(id="analysis-aggregate-threshold", type="number", min=0,
                          step=1, value=1),
                html.P("Only codes are reduced. Selected metadata must be constant "
                       "within each group; otherwise the analysis reports an error."),
            ], id="analysis-aggregation-options", style={"display": "none"}),
            html.Div([
                html.P("The displayed chart reflects the saved analysis. Preview changes before saving."),
                html.Button("Preview Changes", id="analysis-preview", n_clicks=0),
                html.Button("Create Analysis", id="analysis-save", n_clicks=0,
                            className="primary-action"),
                html.Button("Cancel", id="analysis-cancel", n_clicks=0),
            ], className="memo-actions"),
        ], id="analysis-editor", style={"display": "none"},
            className="analysis-editor"),
        html.Div(id="analysis-status", className="control-status"),
        html.Div(id="analysis-preview-status", className="control-status"),
        dcc.Graph(id="analysis-figure", figure=analysis_figure({
            "kind": "bar", "names": [], "counts": [],
            "denominators": [], "categories": [],
        }), config={"displaylogo": False}),
    ], className="analysis-manager")


def register_analysis_callbacks(app: Any, project: Any) -> None:
    from dash import ALL, Input, Output, State, callback_context, dcc, html, no_update
    from geometric_coder.ui.app import _empty_figure

    def choice_values(spec: dict[str, Any]) -> list[str]:
        return [
            f"code:{v['code_id']}" if v["type"] == "code"
            else f"metadata:{v['field']}"
            for v in spec["variables"]
        ]

    @app.callback(
        Output("analysis-select", "options"),
        Input("analysis-refresh", "data"),
        Input("workspace-tabs", "value"),
    )
    def options(_revision: int, workspace: str) -> list[dict[str, Any]]:
        if workspace != "analyses":
            return no_update
        return [
            {"label": row["name"], "value": row["analysis_id"]}
            for row in project.analyses()
        ]

    @app.callback(
        Output("analysis-vars", "options"),
        Input("workspace-tabs", "value"),
        Input("explore-code-refresh-store", "data"),
        Input("focus-refresh-store", "data"),
        Input("analysis-editor", "style"),
    )
    def refresh_variables(
        workspace: str, _code_refresh: int | None,
        _focus_refresh: int | None, _editor_style: dict[str, Any] | None,
    ) -> Any:
        if workspace != "analyses":
            return no_update
        return variable_options(project)

    @app.callback(
        Output("analysis-editor", "style"),
        Output("analysis-editing", "data"),
        Output("analysis-editor-heading", "children"),
        Output("analysis-name", "value"),
        Output("analysis-kind", "value"),
        Output("analysis-measure", "value"),
        Output("analysis-vars", "value"),
        Output("analysis-source-defaults", "data"),
        Output("analysis-aggregate-level", "value"),
        Output("analysis-aggregate-method", "value"),
        Output("analysis-aggregate-threshold", "value"),
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
        if trigger == "analysis-cancel":
            return {"display": "none"}, None, "", "", "bar", "count", [], [], "none", "any", 1
        if trigger == "analysis-new":
            return {"display": "block"}, {"id": None}, "New analysis", "", "bar", "count", [], [], "none", "any", 1
        if selected is None:
            return (no_update,) * 11
        record = project.analysis(int(selected))
        spec = record["spec"]
        return (
            {"display": "block"},
            {"id": int(selected) if trigger == "analysis-edit" else None},
            "Edit analysis" if trigger == "analysis-edit" else "Duplicate analysis",
            record["name"] if trigger == "analysis-edit" else record["name"] + " (copy)",
            spec["kind"], spec["measure"], choice_values(spec), spec["variables"],
            spec.get("aggregation", {}).get("level", "none"),
            spec.get("aggregation", {}).get("method", "any"),
            spec.get("aggregation", {}).get("threshold", 1),
        )

    @app.callback(
        Output("analysis-measure", "options"),
        Input("analysis-kind", "value"),
    )
    def measure_choices(kind: str) -> list[dict[str, str]]:
        if kind == "heatmap":
            return [
                {"label": "Positive co-occurrence count", "value": "count"},
                {"label": "Jaccard similarity", "value": "jaccard"},
                {"label": "Pearson correlation", "value": "correlation"},
            ]
        if kind == "distribution":
            return [{"label": "Number of groups at each code value", "value": "count"}]
        return [{"label": "Number of positive units", "value": "count"}]

    @app.callback(
        Output("analysis-visualization-help", "children"),
        Input("analysis-kind", "value"),
        Input("analysis-aggregate-level", "value"),
        Input("analysis-aggregate-method", "value"),
    )
    def explain_visualization(kind: str, level: str | None, method: str | None) -> str:
        if kind != "distribution":
            return ""
        if not level or level == "none":
            return (
                "Without aggregation: this counts positive and negative "
                "atomic observations (0=absent, 1=present). Choose a parent "
                "level below to see values by turn or document."
            )
        descriptions = {
            "any": "whether any observed unit is positive (0 or 1)",
            "minimum": "whether positive count reaches the threshold (0 or 1)",
            "count": "the number of positive units",
            "mean": "the proportion of observed units positive",
            "log_count": "log(1 + positive units)",
        }
        return (
            f"Each {level} contributes one value: "
            f"{descriptions.get(method, 'the selected reduction')}. "
            f"The chart shows how many {level} groups have each value."
        )

    @app.callback(
        Output("analysis-aggregation-options", "style"),
        Input("analysis-aggregate-level", "value"),
    )
    def show_aggregation(level: str | None) -> dict[str, str]:
        return {"display": "block"} if level and level != "none" else {"display": "none"}

    @app.callback(
        Output("analysis-save", "children"),
        Input("analysis-editing", "data"),
    )
    def primary_button(editing: dict[str, Any] | None) -> str:
        return "Save Changes" if editing and editing.get("id") is not None else "Create Analysis"

    @app.callback(
        Output("analysis-code-settings", "children"),
        Input("analysis-vars", "value"),
        Input("analysis-source-defaults", "data"),
    )
    def code_options(values: list[str] | None, defaults: list[dict[str, Any]] | None) -> Any:
        initial = {int(v["code_id"]): v for v in defaults or [] if v["type"] == "code"}
        nodes = []
        for value in values or []:
            if not value.startswith("code:"):
                continue
            code_id = int(value.split(":", 1)[1])
            try:
                name = project.code(code_id)["name"]
            except KeyError:
                continue
            config = initial.get(code_id, {})
            predictors = project.classifier_specs(code_id)
            nodes.append(html.Div([
                html.H4(name),
                html.Label("Source"),
                dcc.Dropdown(
                    id={"type": "analysis-source", "index": code_id},
                    options=[
                        {"label": "Human labels only", "value": "labels"},
                        {"label": "Predictions only", "value": "predictions"},
                        {"label": "Human labels, then predictions", "value": "hybrid"},
                    ],
                    value=config.get("source", "labels"), clearable=False,
                ),
                html.Label("Prediction classifier (required for prediction modes)"),
                dcc.Dropdown(
                    id={"type": "analysis-predictor", "index": code_id},
                    options=[{"label": p["name"], "value": p["classifier_spec_id"]}
                             for p in predictors],
                    value=config.get("predictor_id"), clearable=True,
                ),
            ], className="analysis-variable-settings"))
        return nodes

    @app.callback(
        Output("analysis-source-defaults", "data", allow_duplicate=True),
        Input({"type": "analysis-source", "index": ALL}, "value"),
        Input({"type": "analysis-predictor", "index": ALL}, "value"),
        State({"type": "analysis-source", "index": ALL}, "id"),
        State({"type": "analysis-predictor", "index": ALL}, "id"),
        State("analysis-source-defaults", "data"),
        prevent_initial_call=True,
    )
    def preserve_sources(
        sources: list[str], predictor_values: list[int | None],
        source_ids: list[dict[str, Any]], predictor_ids: list[dict[str, Any]],
        existing: list[dict[str, Any]] | None,
    ) -> Any:
        """Keep unsaved per-code settings when adding another variable."""
        by_source = {int(i["index"]): val for i, val in zip(source_ids, sources)}
        by_predictor = {
            int(i["index"]): val for i, val in zip(predictor_ids, predictor_values)
        }
        result = [dict(v) for v in (existing or [])]
        by_code = {int(v["code_id"]): v for v in result if v["type"] == "code"}
        for code_id, source in by_source.items():
            current = by_code.get(code_id)
            if current is None:
                current = {"type": "code", "code_id": code_id, "value": "binary"}
                result.append(current)
                by_code[code_id] = current
            current["source"] = source
            predictor = by_predictor.get(code_id)
            if predictor is None:
                current.pop("predictor_id", None)
            else:
                current["predictor_id"] = predictor
        return result if result != (existing or []) else no_update

    def editor_spec(
        kind: str, measure: str, values: list[str] | None,
        aggregate_level: str | None, aggregate_method: str,
        aggregate_threshold: int | float | None,
        source_ids: list[dict[str, Any]], sources: list[str],
        predictor_ids: list[dict[str, Any]], predictor_values: list[int | None],
    ) -> dict[str, Any]:
        by_source = {int(k["index"]): v for k, v in zip(source_ids, sources)}
        by_predictor = {int(k["index"]): v for k, v in zip(predictor_ids, predictor_values)}
        variables = []
        for raw in values or []:
            type_, value = raw.split(":", 1)
            if type_ == "metadata":
                variables.append({"type": "metadata", "field": value})
                continue
            cid = int(value)
            source = by_source.get(cid, "labels")
            variable = {"type": "code", "code_id": cid, "source": source,
                        "value": "binary"}
            if source != "labels":
                variable["predictor_id"] = by_predictor.get(cid)
            variables.append(variable)
        spec = {"kind": kind, "measure": measure, "variables": variables}
        if aggregate_level and aggregate_level != "none":
            spec["aggregation"] = {
                "level": aggregate_level,
                "method": aggregate_method,
                "threshold": aggregate_threshold if aggregate_threshold is not None else 1,
            }
        return spec

    @app.callback(
        Output("analysis-refresh", "data"),
        Output("analysis-select", "value"),
        Output("analysis-select", "options", allow_duplicate=True),
        Output("analysis-status", "children"),
        Output("analysis-editor", "style", allow_duplicate=True),
        Input("analysis-save", "n_clicks"),
        State("analysis-editing", "data"),
        State("analysis-name", "value"),
        State("analysis-kind", "value"),
        State("analysis-measure", "value"),
        State("analysis-vars", "value"),
        State("analysis-aggregate-level", "value"),
        State("analysis-aggregate-method", "value"),
        State("analysis-aggregate-threshold", "value"),
        State({"type": "analysis-source", "index": ALL}, "id"),
        State({"type": "analysis-source", "index": ALL}, "value"),
        State({"type": "analysis-predictor", "index": ALL}, "id"),
        State({"type": "analysis-predictor", "index": ALL}, "value"),
        State("analysis-refresh", "data"),
        prevent_initial_call=True,
    )
    def save(_click: int, editing: dict[str, Any] | None,
             name: str | None, kind: str, measure: str,
             values: list[str] | None,
             aggregate_level: str | None, aggregate_method: str,
             aggregate_threshold: int | float | None,
             source_ids: list[dict[str, Any]], sources: list[str],
             predictor_ids: list[dict[str, Any]], predictor_values: list[int | None],
             revision: int | None) -> tuple[Any, ...]:
        if editing is None:
            return (no_update,) * 5
        try:
            spec = editor_spec(
                kind, measure, values, aggregate_level, aggregate_method,
                aggregate_threshold, source_ids, sources,
                predictor_ids, predictor_values,
            )
            analysis_id = project.save_analysis(
                name or "", spec, analysis_id=editing.get("id"),
            )
        except (ValueError, KeyError) as exc:
            return no_update, no_update, no_update, str(exc), no_update
        # Return options and the newly created ID in the same transaction
        # to avoid requiring a second manual dropdown selection.
        options = [{"label": row["name"], "value": row["analysis_id"]}
                   for row in project.analyses()]
        return (int(revision or 0) + 1, analysis_id, options,
                "Analysis created." if editing.get("id") is None else "Changes saved.",
                {"display": "none"})

    @app.callback(
        Output("analysis-figure", "figure", allow_duplicate=True),
        Output("analysis-preview-status", "children"),
        Input("analysis-preview", "n_clicks"),
        State("analysis-editing", "data"),
        State("analysis-kind", "value"),
        State("analysis-measure", "value"),
        State("analysis-vars", "value"),
        State("analysis-aggregate-level", "value"),
        State("analysis-aggregate-method", "value"),
        State("analysis-aggregate-threshold", "value"),
        State({"type": "analysis-source", "index": ALL}, "id"),
        State({"type": "analysis-source", "index": ALL}, "value"),
        State({"type": "analysis-predictor", "index": ALL}, "id"),
        State({"type": "analysis-predictor", "index": ALL}, "value"),
        prevent_initial_call=True,
    )
    def preview(
        _clicks: int, editing: dict[str, Any] | None,
        kind: str, measure: str, values: list[str] | None,
        level: str | None, method: str, threshold: int | float | None,
        source_ids: list[dict[str, Any]], sources: list[str],
        predictor_ids: list[dict[str, Any]], predictor_values: list[int | None],
    ) -> tuple[Any, str]:
        if editing is None:
            return no_update, ""
        try:
            from geometric_coder.analyses import compute_analysis
            spec = editor_spec(
                kind, measure, values, level, method, threshold,
                source_ids, sources, predictor_ids, predictor_values,
            )
            result = compute_analysis(project, spec)
            fig = analysis_figure(result)
            if result.get("warnings"):
                fig.add_annotation(
                    text=" | ".join(result["warnings"]),
                    xref="paper", yref="paper", x=0, y=1.14,
                    xanchor="left", showarrow=False,
                    font={"color": "#9a6510", "size": 11},
                )
            return fig, "Preview of unsaved analysis settings."
        except (ValueError, KeyError) as exc:
            return no_update, f"Preview unavailable: {exc}"

    @app.callback(
        Output("analysis-delete-dialog", "style"),
        Input("analysis-delete", "n_clicks"),
        Input("analysis-delete-cancel", "n_clicks"),
        State("analysis-select", "value"),
        prevent_initial_call=True,
    )
    def confirm_dialog(_delete: int, _cancel: int, selected: int | None) -> dict[str, str]:
        return {"display": "block"} if (
            callback_context.triggered_id == "analysis-delete" and selected is not None
        ) else {"display": "none"}

    @app.callback(
        Output("analysis-refresh", "data", allow_duplicate=True),
        Output("analysis-select", "value", allow_duplicate=True),
        Output("analysis-delete-dialog", "style", allow_duplicate=True),
        Output("analysis-status", "children", allow_duplicate=True),
        Input("analysis-delete-confirm", "n_clicks"),
        State("analysis-select", "value"),
        State("analysis-refresh", "data"),
        prevent_initial_call=True,
    )
    def delete(_click: int, selected: int | None, revision: int | None) -> tuple[Any, ...]:
        if selected is None:
            return no_update, no_update, {"display": "none"}, ""
        project.delete_analysis(int(selected))
        return int(revision or 0) + 1, None, {"display": "none"}, "Analysis deleted."

    @app.callback(
        Output("analysis-figure", "figure"),
        Input("analysis-select", "value"),
        Input("analysis-refresh", "data"),
        Input("workspace-tabs", "value"),
        Input("explore-annotation-refresh-store", "data"),
        Input("focus-refresh-store", "data"),
        Input("teaching-example-refresh-store", "data"),
    )
    def render(selected: int | None, _refresh: int, workspace: str,
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
