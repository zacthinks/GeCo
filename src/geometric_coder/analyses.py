"""Saved, live exploratory analyses.

Definitions are durable; results are recomputed against current atomic observations.
No models are trained and no result matrices are persisted by this module.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any, Mapping

import plotly.graph_objects as go

_ALLOWED_KINDS = {"bar", "heatmap", "distribution"}
_ALLOWED_SOURCES = {"labels", "predictions", "hybrid"}
_ALLOWED_MEASURES = {"count", "jaccard", "correlation"}
_AGGREGATE_METHODS = {"any", "minimum", "count", "mean", "log_count"}
_ALLOWED_VALUES = {"binary", "probability"}


def validate_analysis(project: Any, spec: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize and validate an analysis definition; reject dangling references."""
    if not isinstance(spec, Mapping):
        raise ValueError("Analysis settings must be an object.")
    kind = str(spec.get("kind", ""))
    if kind not in _ALLOWED_KINDS:
        raise ValueError("Unsupported analysis visualization.")
    measure = str(spec.get("measure", "count"))
    if measure not in _ALLOWED_MEASURES:
        raise ValueError("Unsupported analysis measure.")
    if kind in {"bar", "distribution"} and measure != "count":
        raise ValueError("Bar and distribution plots currently support counts only.")
    aggregate = spec.get("aggregation")
    if aggregate is not None:
        if not isinstance(aggregate, Mapping):
            raise ValueError("Invalid aggregation settings.")
        level = aggregate.get("level")
        keys = list(project.metadata.get("key_columns", []))
        if not isinstance(level, str) or level not in keys[:-1]:
            raise ValueError("Select an existing parent key level for aggregation.")
        method = str(aggregate.get("method", "any"))
        if method not in _AGGREGATE_METHODS:
            raise ValueError("Unsupported code aggregation method.")
        threshold = aggregate.get("threshold", 1)
        if method == "minimum":
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not (0 <= threshold < 1_000_000):
                raise ValueError("Aggregation threshold must be nonnegative.")
        else:
            threshold = 1
        if (kind not in {"distribution"} and not (kind == "heatmap" and measure == "correlation")) and method in {"count", "mean", "log_count"}:
            raise ValueError("Continuous aggregation currently requires a correlation heatmap.")
        aggregate = {"level": level, "method": method, "threshold": threshold}

    raw = spec.get("variables")
    if not isinstance(raw, list) or not 1 <= len(raw) <= 20:
        raise ValueError("Choose between 1 and 20 analysis variables.")
    if kind == "heatmap" and len(raw) < 2:
        raise ValueError("Heatmaps require at least two variables.")
    codes = {int(c["code_id"]) for c in project.codes()}
    metadata = set(project.metadata.get("metadata_columns", []))
    variables: list[dict[str, Any]] = []
    for original in raw:
        if not isinstance(original, Mapping):
            raise ValueError("Invalid analysis variable.")
        vtype = str(original.get("type", ""))
        if vtype == "metadata":
            field = original.get("field")
            if not isinstance(field, str) or field not in metadata:
                raise ValueError("Unknown metadata field.")
            variables.append({"type": "metadata", "field": field})
            continue
        if vtype != "code":
            raise ValueError("Variables must reference a code or metadata.")
        try:
            code_id = int(original["code_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Code variable requires a valid code ID.") from exc
        if code_id not in codes:
            raise ValueError(f"Unknown code ID: {code_id}")
        source = str(original.get("source", "labels"))
        value = str(original.get("value", "binary"))
        if source not in _ALLOWED_SOURCES or value not in _ALLOWED_VALUES:
            raise ValueError("Unsupported source or value representation.")
        if value != "binary":
            raise ValueError("Initial count and similarity analyses require binary values.")
        result: dict[str, Any] = {
            "type": "code", "code_id": code_id,
            "source": source, "value": value,
        }
        if source != "labels":
            try:
                predictor_id = int(original["predictor_id"])
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError("Select a classifier for predicted values.") from exc
            specs = {int(s["classifier_spec_id"]): s
                     for s in project.classifier_specs(code_id)}
            if predictor_id not in specs:
                raise ValueError("Prediction source is no longer available for this code.")
            result["predictor_id"] = predictor_id
        if kind == "heatmap" and value != "binary":
            raise ValueError("Initial heatmaps use binary code values.")
        variables.append(result)
    if kind in {"bar", "distribution"}:
        if not any(v["type"] == "code" for v in variables):
            raise ValueError("Bar and distribution analyses require at least one code.")
        if kind == "distribution" and any(v["type"] == "metadata" for v in variables):
            raise ValueError("Distribution plots currently use code variables only.")
        if sum(v["type"] == "metadata" for v in variables) > 1:
            raise ValueError("Bar analyses support at most one metadata grouping.")
    if kind == "heatmap" and len(variables) < 2:
        raise ValueError("Heatmaps require at least two variables.")
    if len({(v["type"], v.get("code_id", v.get("field"))) for v in variables}) != len(variables):
        raise ValueError("Each analysis variable may appear only once.")
    return {"schema_version": 1, "kind": kind, "measure": measure,
            "variables": variables, **({"aggregation": aggregate} if aggregate is not None else {})}


def list_analyses(project: Any) -> list[dict[str, Any]]:
    with project.database.connect() as conn:
        rows = conn.execute(
            "SELECT analysis_id, name, spec_json, created_at, updated_at "
            "FROM saved_analyses ORDER BY name COLLATE NOCASE, analysis_id"
        ).fetchall()
    return [{**dict(row), "spec": json.loads(row["spec_json"])} for row in rows]


def get_analysis(project: Any, analysis_id: int) -> dict[str, Any]:
    with project.database.connect() as conn:
        row = conn.execute(
            "SELECT analysis_id, name, spec_json, created_at, updated_at "
            "FROM saved_analyses WHERE analysis_id = ?", (int(analysis_id),)
        ).fetchone()
    if row is None:
        raise KeyError(f"Unknown analysis: {analysis_id}")
    return {**dict(row), "spec": json.loads(row["spec_json"])}


def save_analysis(project: Any, name: str, spec: Mapping[str, Any],
                  analysis_id: int | None = None) -> int:
    clean = str(name).strip()
    if not clean or len(clean) > 160:
        raise ValueError("Analysis name must contain 1–160 characters.")
    normalized = validate_analysis(project, spec)
    payload = json.dumps(normalized, sort_keys=True)
    try:
        with project.database.connect() as conn:
            if analysis_id is None:
                cur = conn.execute(
                    "INSERT INTO saved_analyses(name, spec_json, created_at, updated_at) "
                    "VALUES (?, ?, strftime('%Y-%m-%dT%H:%M:%fZ','now'), "
                    "strftime('%Y-%m-%dT%H:%M:%fZ','now'))",
                    (clean, payload),
                )
                return int(cur.lastrowid)
            cur = conn.execute(
                "UPDATE saved_analyses SET name = ?, spec_json = ?, "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now') "
                "WHERE analysis_id = ?", (clean, payload, int(analysis_id))
            )
            if cur.rowcount != 1:
                raise KeyError(f"Unknown analysis: {analysis_id}")
            return int(analysis_id)
    except sqlite3.IntegrityError as exc:
        # Translate only the name uniqueness constraint. Other integrity
        # failures must not be mislabeled as a name collision.
        message = str(exc)
        if "UNIQUE constraint failed: saved_analyses.name" in message:
            raise ValueError(
                f"An analysis named '{clean}' already exists. Choose a different name."
            ) from exc
        raise

def delete_analysis(project: Any, analysis_id: int) -> None:
    with project.database.connect() as conn:
        cur = conn.execute("DELETE FROM saved_analyses WHERE analysis_id = ?",
                           (int(analysis_id),))
        if cur.rowcount != 1:
            raise KeyError(f"Unknown analysis: {analysis_id}")


def _resolve_variables(project: Any, spec: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Pin currently fitted classifier IDs and collect provenance warnings."""
    codes = {int(c["code_id"]): c["name"] for c in project.codes()}
    variables = []
    warnings = []
    for item in spec["variables"]:
        v = dict(item)
        if v["type"] == "metadata":
            v["name"] = v["field"]
            variables.append(v)
            continue
        cid = v["code_id"]
        if cid not in codes:
            raise ValueError(f"Code {cid} has been removed.")
        v["name"] = codes[cid]
        if v["source"] != "labels":
            status = project.classifier_status(
                code_id=cid, classifier_spec_ids=[v["predictor_id"]]
            )["classifiers"]
            if not status or status[0].get("classifier_fit_id") is None:
                raise ValueError(f"No trained prediction source for {v['name']}.")
            v["fit_id"] = int(status[0]["classifier_fit_id"])
            if status[0]["status"] != "current":
                warnings.append(f"{v['name']}: selected classifier is stale.")
            if project.database.has_missing_atomic_classifier_predictions(v["fit_id"]):
                warnings.append(f"{v['name']}: predictions do not cover all atomic observations.")
        variables.append(v)
    return variables, warnings


def _code_join(variable: dict[str, Any], index: int) -> tuple[str, str, list[Any]]:
    """SQL join/value expression for a code, with parameterized IDs.

    Unreviewed and Unsure are NULL. In hybrid mode, Unsure stays NULL rather
    than quietly deferring to a machine prediction.
    """
    a, p = f"a{index}", f"p{index}"
    joins = [f"LEFT JOIN current_annotations {a} "
             f"ON {a}.observation_id = u.observation_id AND {a}.code_id = ?"]
    args: list[Any] = [variable["code_id"]]
    human = f"CASE {a}.value WHEN 'positive' THEN 1.0 WHEN 'negative' THEN 0.0 ELSE NULL END"
    if variable["source"] == "labels":
        return " ".join(joins), human, args
    joins.append(f"LEFT JOIN classifier_predictions {p} "
                 f"ON {p}.observation_id = u.observation_id AND {p}.classifier_fit_id = ?")
    args.append(variable["fit_id"])
    prediction = (f"{p}.probability" if variable["value"] == "probability"
                  else f"CASE {p}.predicted_label WHEN 'positive' THEN 1.0 "
                       "WHEN 'negative' THEN 0.0 ELSE NULL END")
    if variable["source"] == "predictions":
        return " ".join(joins), prediction, args
    if variable["value"] == "probability":
        human = f"CASE {a}.value WHEN 'positive' THEN 1.0 WHEN 'negative' THEN 0.0 ELSE NULL END"
    return " ".join(joins), (
        f"CASE WHEN {a}.observation_id IS NOT NULL THEN {human} "
        f"ELSE {prediction} END"
    ), args


def compute_analysis(project: Any, spec: Mapping[str, Any]) -> dict[str, Any]:
    """Execute a live, bounded aggregation; all sample counts remain explicit."""
    normalized = validate_analysis(project, spec)
    variables, warnings = _resolve_variables(project, normalized)
    # All Jaccard heatmaps use the same pairwise-complete calculations,
    # regardless of whether categorical metadata are also selected.
    if (
        normalized.get("aggregation") is not None
        or normalized["kind"] == "distribution"
        or normalized["measure"] in {"correlation", "jaccard"}
        or (normalized["kind"] == "heatmap" and any(v["type"] == "metadata" for v in variables))
    ):
        from geometric_coder.analysis_matrix import compute_expanded_analysis
        return compute_expanded_analysis(project, normalized, variables, warnings)
    codes = [v for v in variables if v["type"] == "code"]
    grouping = next((v for v in variables if v["type"] == "metadata"), None)
    joins, expressions, params = [], [], []
    for i, var in enumerate(codes):
        join, expr, values = _code_join(var, i)
        joins.append(join)
        expressions.append(expr)
        params.extend(values)
    from_clause = " FROM units u " + " ".join(joins)
    if normalized["kind"] == "heatmap":
        # Aggregates are evaluated in SQL, not fetched as observation arrays.
        # For each unordered pair: N both-positive, A positive, B positive.
        cols = []
        for i in range(len(codes)):
            cols.append(f"SUM(CASE WHEN ({expressions[i]}) = 1 THEN 1 ELSE 0 END) AS c{i}")
        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                cols.append(f"SUM(CASE WHEN ({expressions[i]}) = 1 AND "
                            f"({expressions[j]}) = 1 THEN 1 ELSE 0 END) AS p{i}_{j}")
        with project.database.connect() as conn:
            row = conn.execute("SELECT " + ", ".join(cols) + from_clause,
                               params).fetchone()
        n = len(codes)
        values = [[0.0] * n for _ in range(n)]
        for i in range(n):
            count = int(row[f"c{i}"] or 0)
            values[i][i] = float(count)
            for j in range(i + 1, n):
                both = int(row[f"p{i}_{j}"] or 0)
                values[i][j] = values[j][i] = float(both)
        return {"kind": "heatmap", "names": [v["name"] for v in codes],
                "values": values, "measure": normalized["measure"], "warnings": warnings}
    # Bar charts: counts of positive values, optionally grouped by one
    # categorical metadata field. SQL returns only aggregated group rows.
    if grouping:
        path = '$."' + grouping["field"].replace("\\", "\\\\").replace('"', '\\"') + '"'
        group_expr = "CAST(json_extract(u.metadata_json, ?) AS TEXT)"
        selects = [f"{group_expr} AS category"]
        query_params = [path, *params]
        group_by = " GROUP BY category ORDER BY category LIMIT 101"
    else:
        selects = ["'All observations' AS category"]
        query_params = params
        group_by = ""
    for i, expr in enumerate(expressions):
        selects.append(f"SUM(CASE WHEN ({expr}) = 1 THEN 1 ELSE 0 END) AS c{i}")
        selects.append(f"SUM(CASE WHEN ({expr}) IS NOT NULL THEN 1 ELSE 0 END) AS n{i}")
    with project.database.connect() as conn:
        rows = conn.execute("SELECT " + ", ".join(selects) + from_clause + group_by,
                            query_params).fetchall()
    if len(rows) > 100:
        raise ValueError("Too many metadata categories for a bar plot (maximum 100).")
    return {
        "kind": "bar", "names": [v["name"] for v in codes],
        "categories": [str(row["category"] or "(Missing)") for row in rows],
        "counts": [[int(row[f"c{i}"] or 0) for row in rows] for i in range(len(codes))],
        "denominators": [[int(row[f"n{i}"] or 0) for row in rows] for i in range(len(codes))],
        "warnings": warnings,
    }


def analysis_figure(result: Mapping[str, Any]) -> go.Figure:
    """Convert the compact aggregate into an interactive Plotly figure."""
    fig = go.Figure()
    if result["kind"] == "distribution":
        # Explain both the observational grain and the reduction. Without a
        # parent aggregation this is a binary frequency plot, not a histogram
        # of an unnamed 'aggregated code value'.
        aggregation = result.get("aggregation")
        if aggregation:
            parent = str(aggregation["level"])
            child = str(result.get("atomic_level", "atomic"))
            method = aggregation["method"]
            y_label = f"Number of {parent} groups"
            if method == "count":
                x_label = f"Number of positive {child} observations per {parent}"
            elif method == "mean":
                x_label = f"Proportion of observed {child} units positive per {parent}"
            elif method == "log_count":
                x_label = f"log(1 + positive {child} observations per {parent})"
            elif method == "minimum":
                threshold = aggregation.get("threshold", 1)
                x_label = f"At least {threshold:g} positive {child} units per {parent} (0=no, 1=yes)"
            else:
                x_label = f"Any positive {child} unit per {parent} (0=no, 1=yes)"
            title = f"Distribution across {parent} groups"
        else:
            x_label = "Code judgment at atomic level (0=absent, 1=present)"
            y_label = "Number of atomic observations"
            title = "Code judgments across atomic observations"
        for name, hist in zip(result["names"], result["histograms"]):
            observed = sum(hist["counts"])
            fig.add_trace(go.Bar(
                name=name, x=hist["values"], y=hist["counts"],
                hovertemplate=(
                    "%{x}: %{y} observations<extra>%{fullData.name}</extra>"
                    if not aggregation
                    else "%{x}: %{y} groups<extra>%{fullData.name}</extra>"
                ),
            ))
            if len(result["names"]) == 1:
                title += f" — {name} ({observed:,} with observed values)"
        fig.update_layout(
            barmode="group", title=title,
            xaxis_title=x_label, yaxis_title=y_label,
        )
    elif result["kind"] == "heatmap":
        correlation = result.get("measure") == "correlation"
        samples = result.get("samples")
        fig.add_trace(go.Heatmap(
            x=result["names"], y=result["names"], z=result["values"],
            customdata=samples if samples is not None else None,
            colorscale="RdBu" if correlation else "Greens",
            reversescale=correlation,
            zmin=-1 if correlation else None,
            zmax=1 if correlation else None,
            hovertemplate=(
                "%{y} × %{x}: %{z:.3f} (n=%{customdata})<extra></extra>"
                if samples is not None
                else "%{y} × %{x}: %{z:.3g}<extra></extra>"
            ),
        ))
    else:
        for name, counts, denominators in zip(
            result["names"], result["counts"], result["denominators"]
        ):
            fig.add_trace(go.Bar(
                name=name, x=result["categories"], y=counts,
                customdata=denominators,
                hovertemplate="%{x}: %{y} positive (of %{customdata} observed)<extra>%{fullData.name}</extra>",
            ))
        fig.update_layout(barmode="group", yaxis_title="Positive observations")
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=50, r=20, t=70 if result["kind"] == "distribution" else 25,
                    b=90 if result["kind"] == "distribution" else 60),
    )
    if result["kind"] == "distribution":
        fig.update_xaxes(automargin=True)
        fig.update_yaxes(automargin=True)
    return fig
