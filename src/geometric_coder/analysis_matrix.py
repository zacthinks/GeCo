"""SQL-backed parent-level reductions, categorical indicators and matrix summaries.

Only aggregated rows cross the database boundary. Missing code judgments are
NULL throughout; one-hot indicators remain NULL when their source is missing.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

from geometric_coder.analyses import _code_join

_MAX_CATEGORIES = 24
_MAX_DIMENSIONS = 24


def _literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _path(field: str) -> str:
    # SQLite JSON path for a declared key / metadata column.
    return '$."' + field.replace('\\', '\\\\').replace('"', '\\"') + '"'


def _dataset_sql(
    project: Any,
    spec: Mapping[str, Any],
    variables: list[dict[str, Any]],
) -> tuple[str, list[Any], list[str], list[str]]:
    """Construct source-unit and optional keyed parent-aggregation CTEs."""
    codes = [v for v in variables if v["type"] == "code"]
    metadata = [v for v in variables if v["type"] == "metadata"]
    columns = [f"json_extract(u.metadata_json, {_literal(_path(v['field']))}) AS m{i}"
               for i, v in enumerate(metadata)]
    joins = []
    params = []
    for i, code in enumerate(codes):
        join, expr, args = _code_join(code, i)
        columns.append(f"({expr}) AS c{i}")
        joins.append(join)
        params.extend(args)
    aggregate = spec.get("aggregation")
    group_columns: list[str] = []
    if aggregate is not None:
        keys = list(project.metadata["key_columns"])
        prefix = keys[:keys.index(aggregate["level"]) + 1]
        group_columns = [f"k{i}" for i in range(len(prefix))]
        columns.extend(
            f"json_extract(u.user_key_json, {_literal(_path(key))}) AS k{i}"
            for i, key in enumerate(prefix)
        )
    source = "WITH base AS (SELECT " + ", ".join(columns) + " FROM units u " + " ".join(joins) + ")"
    if aggregate is None:
        return source + ", derived AS (SELECT * FROM base)", params, [
            f"c{i}" for i in range(len(codes))
        ], [f"m{i}" for i in range(len(metadata))]

    method = aggregate["method"]
    reductions = []
    for i in range(len(codes)):
        column = f"c{i}"
        count_pos = f"SUM(CASE WHEN {column} = 1 THEN 1 ELSE 0 END)"
        if method == "any":
            result = f"CASE WHEN COUNT({column}) = 0 THEN NULL WHEN {count_pos} > 0 THEN 1.0 ELSE 0.0 END"
        elif method == "minimum":
            result = (f"CASE WHEN COUNT({column}) = 0 THEN NULL "
                      f"WHEN {count_pos} >= {float(aggregate['threshold'])} THEN 1.0 ELSE 0.0 END")
        elif method == "count":
            result = f"CASE WHEN COUNT({column}) = 0 THEN NULL ELSE {count_pos} END"
        elif method == "mean":
            result = f"AVG({column})"
        elif method == "log_count":
            result = f"CASE WHEN COUNT({column}) = 0 THEN NULL ELSE geco_log1p({count_pos}) END"
        else:
            raise ValueError("Unsupported code reduction.")
        reductions.append(f"{result} AS {column}")
    selections = [*group_columns, *[f"MIN(m{i}) AS m{i}" for i in range(len(metadata))], *reductions]
    group_by = ", ".join(group_columns)
    # Distinct NULL and non-NULL values are different; unlike COUNT(DISTINCT)
    # JSON_QUOTE preserves NULL as the JSON string 'null'.
    conflict_sql = " OR ".join(
        f"MIN(json_quote(m{i})) != MAX(json_quote(m{i}))" for i in range(len(metadata))
    )
    if conflict_sql:
        # The guard executes before the actual analysis query; a conflicting
        # metadata field cannot quietly inherit an arbitrary representative.
        guarded = source + f", conflicts AS (SELECT 1 FROM base GROUP BY {group_by} HAVING {conflict_sql} LIMIT 1)"
    else:
        guarded = source
    derived = f", derived AS (SELECT {', '.join(selections)} FROM base GROUP BY {group_by})"
    return guarded + derived, params, [
        f"c{i}" for i in range(len(codes))
    ], [f"m{i}" for i in range(len(metadata))]


def compute_expanded_analysis(
    project: Any, spec: Mapping[str, Any],
    variables: list[dict[str, Any]], warnings: list[str],
) -> dict[str, Any]:
    codes = [v for v in variables if v["type"] == "code"]
    metadata = [v for v in variables if v["type"] == "metadata"]
    cte, params, code_columns, metadata_columns = _dataset_sql(project, spec, variables)
    aggregation = spec.get("aggregation")
    with project.database.connect() as conn:
        conn.create_function("geco_log1p", 1, lambda value: math.log1p(value))
        if aggregation is not None and metadata_columns:
            mismatch = conn.execute(
                cte + " SELECT 1 FROM conflicts LIMIT 1", params
            ).fetchone()
            if mismatch is not None:
                raise ValueError(
                    "Selected metadata vary within the chosen aggregation level. "
                    "Choose a finer parent key or remove the conflicting metadata."
                )

        if spec["kind"] == "distribution":
            histograms = []
            for column in code_columns:
                # Bin continuous results to two decimals and aggregate on the
                # database side, rather than sending individual group values.
                bucket = f"ROUND({column}, 2)"
                rows = conn.execute(
                    cte + f" SELECT {bucket} AS value, COUNT(*) AS amount"
                    f" FROM derived WHERE {column} IS NOT NULL"
                    f" GROUP BY value ORDER BY value LIMIT 101", params
                ).fetchall()
                if len(rows) > 100:
                    raise ValueError(
                        "More than 100 distinct code-value bins; narrow the analysis."
                    )
                histograms.append({
                    "values": [float(row["value"]) for row in rows],
                    "counts": [int(row["amount"]) for row in rows],
                })
            return {
                "kind": "distribution",
                "names": [v["name"] for v in codes],
                "histograms": histograms,
                "aggregation": dict(aggregation) if aggregation is not None else None,
                "atomic_level": str(project.metadata.get("key_columns", ["observation"])[-1]),
                "warnings": warnings,
            }

        if spec["kind"] == "bar":
            if not codes:
                raise ValueError("Bar analyses require at least one code.")
            grouping = metadata_columns[0] if metadata_columns else None
            selects = [
                (f"CAST({grouping} AS TEXT) AS category" if grouping
                 else "'All observations' AS category")
            ]
            for i, column in enumerate(code_columns):
                selects.extend([
                    f"SUM(CASE WHEN {column} = 1 THEN 1 ELSE 0 END) AS c{i}",
                    f"COUNT({column}) AS n{i}",
                ])
            group_by = f" GROUP BY {grouping} ORDER BY category" if grouping else ""
            rows = conn.execute(
                cte + " SELECT " + ", ".join(selects)
                + " FROM derived" + group_by + " LIMIT 101", params
            ).fetchall()
            if len(rows) > 100:
                raise ValueError("Metadata has more than 100 categories.")
            return {
                "kind": "bar", "names": [v["name"] for v in codes],
                "categories": [str(row["category"]) if row["category"] is not None else "(Missing)" for row in rows],
                "counts": [[int(row[f"c{i}"] or 0) for row in rows] for i in range(len(codes))],
                "denominators": [[int(row[f"n{i}"] or 0) for row in rows] for i in range(len(codes))],
                "warnings": warnings,
            }

        # Expand categorical metadata *after* aggregating codes and validating
        # metadata consistency, so every category indicator has the right grain.
        names = [v["name"] for v in codes]
        expressions = list(code_columns)
        for i, meta in enumerate(metadata):
            column = metadata_columns[i]
            categories = conn.execute(
                cte + f" SELECT DISTINCT CAST({column} AS TEXT) AS category "
                f"FROM derived WHERE {column} IS NOT NULL ORDER BY category LIMIT 25",
                params,
            ).fetchall()
            if len(categories) > _MAX_CATEGORIES:
                raise ValueError(
                    f"Metadata {meta['name']} has too many categories for a heatmap "
                    f"(maximum {_MAX_CATEGORIES})."
                )
            for row in categories:
                category = str(row["category"])
                names.append(f"{meta['name']} = {category}")
                expressions.append(
                    f"CASE WHEN {column} IS NULL THEN NULL "
                    f"WHEN CAST({column} AS TEXT) = {_literal(category)} "
                    "THEN 1.0 ELSE 0.0 END"
                )
        if len(expressions) < 2:
            raise ValueError("Heatmaps require at least two encoded variables.")
        if len(expressions) > _MAX_DIMENSIONS:
            raise ValueError(
                f"Too many encoded variables for a heatmap (maximum {_MAX_DIMENSIONS})."
            )

        # Pairwise-complete moments avoid materializing large row matrices in
        # Python and preserve unknown label states as genuinely missing.
        fields = []
        for i, x in enumerate(expressions):
            for j in range(i, len(expressions)):
                y = expressions[j]
                valid = f"({x}) IS NOT NULL AND ({y}) IS NOT NULL"
                prefix = f"p{i}_{j}"
                fields.extend([
                    f"SUM(CASE WHEN {valid} THEN 1 ELSE 0 END) AS {prefix}_n",
                    f"SUM(CASE WHEN {valid} THEN ({x}) ELSE 0 END) AS {prefix}_sx",
                    f"SUM(CASE WHEN {valid} THEN ({y}) ELSE 0 END) AS {prefix}_sy",
                    f"SUM(CASE WHEN {valid} THEN ({x}) * ({x}) ELSE 0 END) AS {prefix}_xx",
                    f"SUM(CASE WHEN {valid} THEN ({y}) * ({y}) ELSE 0 END) AS {prefix}_yy",
                    f"SUM(CASE WHEN {valid} THEN ({x}) * ({y}) ELSE 0 END) AS {prefix}_xy",
                ])
        row = conn.execute(
            cte + " SELECT " + ", ".join(fields) + " FROM derived", params
        ).fetchone()

    size = len(expressions)
    values: list[list[float | None]] = [[None] * size for _ in range(size)]
    counts: list[list[int]] = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i, size):
            key = f"p{i}_{j}"
            n = int(row[key + "_n"] or 0)
            sx = float(row[key + "_sx"] or 0)
            sy = float(row[key + "_sy"] or 0)
            xy = float(row[key + "_xy"] or 0)
            counts[i][j] = counts[j][i] = n
            if spec["measure"] == "count":
                value = xy
            elif spec["measure"] == "jaccard":
                # Jaccard is based only on units observed for *both*
                # variables. No joint coverage is different from observing
                # both codes as absent throughout a nonempty complete set.
                union = sx + sy - xy
                value = (xy / union if union else 0.0) if n else None
            else:
                xx = float(row[key + "_xx"] or 0)
                yy = float(row[key + "_yy"] or 0)
                vx = n * xx - sx * sx
                vy = n * yy - sy * sy
                # Pearson r is undefined for insufficient pairwise coverage or
                # any zero-variance variable.
                value = (max(-1.0, min(1.0, (n * xy - sx * sy) / math.sqrt(vx * vy)))
                         if n >= 2 and vx > 1e-12 and vy > 1e-12 else None)
            values[i][j] = values[j][i] = value
    return {
        "kind": "heatmap", "names": names, "values": values, "samples": counts,
        "measure": spec["measure"], "warnings": warnings,
    }
