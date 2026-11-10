"""Code Center saved analyses manager and live Plotly display."""
from __future__ import annotations

from typing import Any

from geometric_coder.analyses import analysis_figure


def analysis_tab(*, dcc: Any, html: Any, project: Any) -> Any:
    choices = [
        {"label": f"Code: {row['name']}", "value": f"code:{row['code_id']}"}
        for row in project.codes()
    ] + [
        {"label": f"Metadata: {field}", "value": f"metadata:{field}"}
        for field in project.metadata.get("metadata_columns", [])
    ]
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
            ], value="bar", clearable=False),
            html.Label("Measure"),
            dcc.Dropdown(id="analysis-measure", options=[
                {"label": "Positive counts", "value": "count"},
                {"label": "Jaccard similarity (heatmaps)", "value": "jaccard"},
            ], value="count", clearable=False),
            html.Label("Variables"),
            dcc.Dropdown(id="analysis-vars", options=choices, value=[], multi=True,
                         placeholder="Add codes and, for bar plots, one metadata field"),
            html.Div(id="analysis-code-settings"),
            html.Div([
                html.Button("Save analysis", id="analysis-save", n_clicks=0,
                            className="primary-action"),
                html.Button("Cancel", id="analysis-cancel", n_clicks=0),
            ], className="memo-actions"),
        ], id="analysis-editor", style={"display": "none"},
            className="analysis-editor"),
        html.Div(id="analysis-status", className="control-status"),
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
        if workspace != "codes":
            return no_update
        return [
            {"label": row["name"], "value": row["analysis_id"]}
            for row in project.analyses()
        ]

    @app.callback(
        Output("analysis-editor", "style"),
        Output("analysis-editing", "data"),
        Output("analysis-editor-heading", "children"),
        Output("analysis-name", "value"),
        Output("analysis-kind", "value"),
        Output("analysis-measure", "value"),
        Output("analysis-vars", "value"),
        Output("analysis-source-defaults", "data"),
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
            return {"display": "none"}, None, "", "", "bar", "count", [], []
        if trigger == "analysis-new":
            return {"display": "block"}, {"id": None}, "New analysis", "", "bar", "count", [], []
        if selected is None:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
        record = project.analysis(int(selected))
        spec = record["spec"]
        return (
            {"display": "block"},
            {"id": int(selected) if trigger == "analysis-edit" else None},
            "Edit analysis" if trigger == "analysis-edit" else "Duplicate analysis",
            record["name"] if trigger == "analysis-edit" else record["name"] + " (copy)",
            spec["kind"], spec["measure"], choice_values(spec), spec["variables"],
        )

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

    @app.callback(
        Output("analysis-refresh", "data"),
        Output("analysis-select", "value"),
        Output("analysis-status", "children"),
        Output("analysis-editor", "style", allow_duplicate=True),
        Input("analysis-save", "n_clicks"),
        State("analysis-editing", "data"),
        State("analysis-name", "value"),
        State("analysis-kind", "value"),
        State("analysis-measure", "value"),
        State("analysis-vars", "value"),
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
             source_ids: list[dict[str, Any]], sources: list[str],
             predictor_ids: list[dict[str, Any]], predictor_values: list[int | None],
             revision: int | None) -> tuple[Any, ...]:
        if editing is None:
            return no_update, no_update, no_update, no_update
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
        try:
            analysis_id = project.save_analysis(
                name or "", {"kind": kind, "measure": measure, "variables": variables},
                analysis_id=editing.get("id"),
            )
        except (ValueError, KeyError) as exc:
            return no_update, no_update, str(exc), no_update
        return int(revision or 0) + 1, analysis_id, "Analysis saved.", {"display": "none"}

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
        Input("code-center-view-tabs", "value"),
        Input("explore-annotation-refresh-store", "data"),
        Input("focus-refresh-store", "data"),
        Input("teaching-example-refresh-store", "data"),
    )
    def render(selected: int | None, _refresh: int, workspace: str,
               tab: str, _annotations: int, _focus: int, _teaching: int) -> Any:
        if workspace != "codes" or tab != "analysis":
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
