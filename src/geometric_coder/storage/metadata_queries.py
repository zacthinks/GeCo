"""Parameterized SQLite predicates for the GeCo metadata filter UI.

Only metadata and annotation state are filtered here. GeCo retains authority
over code annotations; external TeAL files never need to be fully read to
select a metadata-filtered display page.
"""

from __future__ import annotations

from datetime import date, datetime
import re
from typing import Any, Sequence


def _date_value(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or not re.match(r"^\d{4}-\d{1,2}-\d{1,2}", text):
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return None


def _ordered(kind: str, condition: str, value: Any, value2: Any) -> tuple[str, list[Any]]:
    if kind == "numeric":
        first = float(value)
        second = float(value2) if condition == "between" else None
    else:
        first = _date_value(value)
        second = _date_value(value2) if condition == "between" else None
        if first is None:
            raise ValueError("enter a valid date")
        if condition == "between" and second is None:
            raise ValueError("enter both ends of the range")
    if condition == "between":
        if second is None:
            raise ValueError("enter both ends of the range")
        low, high = sorted((first, second))
        return "BETWEEN ? AND ?", [low, high]
    op = {"lt": "<", "lte": "<=", "gt": ">", "gte": ">=", "eq": "=", "neq": "!="}.get(condition)
    if op is None:
        raise ValueError(f"unsupported condition {condition!r}")
    return f"{op} ?", [first]


def compile_filter_sql(
    filters: Sequence[dict[str, Any]],
    *,
    operator: str | None,
    uncoded_only: bool = False,
) -> tuple[str, list[Any]]:
    """Return a bound SQL WHERE predicate without interpolating user values.

    SQLite evaluates the JSON metadata in-place; only the selected page's
    document text and metadata are decoded into Python.
    """
    clauses: list[str] = []
    params: list[Any] = []
    for clause in filters:
        field = str(clause.get("field", ""))
        kind = str(clause.get("type", "text"))
        condition = str(clause.get("operator", ""))
        value = clause.get("value")
        value2 = clause.get("value2")
        path = '$."' + field.replace("\\", "\\\\").replace('"', '\\"') + '"'
        expr = "json_extract(u.metadata_json, ?)"
        field_params: list[Any] = [path]
        if condition in {"missing", "not_missing"}:
            predicate = f"{expr} IS {'NOT ' if condition == 'not_missing' else ''}NULL"
        elif kind == "boolean":
            if condition not in {"true", "false"}:
                raise ValueError(f"unsupported condition {condition!r}")
            # Stored JSON true and false are SQLite 1/0; mirror bool(raw).
            truth = f"(CASE WHEN typeof({expr}) = 'text' THEN length({expr}) > 0 ELSE CAST({expr} AS REAL) != 0 END)"
            predicate = f"({expr} IS NOT NULL AND {truth} = {1 if condition == 'true' else 0})"
            field_params = [path, path, path, path]
        elif kind == "categorical":
            if condition not in {"in", "not_in"}:
                raise ValueError(f"unsupported condition {condition!r}")
            selected = value if isinstance(value, list) else [value]
            choices = [str(item).casefold() for item in selected]
            if choices:
                placeholders = ", ".join("?" for _ in choices)
                cmp = "NOT IN" if condition == "not_in" else "IN"
                predicate = f"({expr} IS NOT NULL AND GECO_FOLD({expr}) {cmp} ({placeholders}))"
                field_params = [path, path, *choices]
            else:
                predicate = f"({expr} IS NOT NULL)" if condition == "not_in" else "0"
        elif kind == "text":
            if condition not in {"contains", "excludes", "eq", "neq", "starts", "ends", "regex"}:
                raise ValueError(f"unsupported condition {condition!r}")
            target = str(value or "")
            if condition == "regex":
                re.compile(target, flags=re.IGNORECASE)
                comparison = f"GECO_REGEX(?, CAST({expr} AS TEXT)) = 1"
                field_params = [path, target, path]
            else:
                # GECO_FOLD performs Unicode casefold like the old Python
                # matcher (SQLite LOWER is ASCII-only).
                folded = target.casefold()
                lhs = f"GECO_FOLD({expr})"
                if condition == "contains":
                    comparison = f"instr({lhs}, ?) > 0"
                elif condition == "excludes":
                    comparison = f"instr({lhs}, ?) = 0"
                elif condition == "starts":
                    comparison = f"substr({lhs}, 1, length(?)) = ?"
                elif condition == "ends":
                    comparison = f"substr({lhs}, -length(?)) = ?"
                else:
                    comparison = f"{lhs} {'!=' if condition == 'neq' else '='} ?"
                field_params = [path, path, folded, folded] if condition in {"starts", "ends"} else [path, path, folded]
            predicate = f"({expr} IS NOT NULL AND {comparison})"
        elif kind in {"numeric", "datetime"}:
            op, vals = _ordered(kind, condition, value, value2)
            lhs = f"CAST({expr} AS REAL)" if kind == "numeric" else f"GECO_DATE({expr})"
            predicate = f"({expr} IS NOT NULL AND {lhs} {op})"
            field_params = [path, path, *vals]
        else:
            raise ValueError(f"unsupported metadata kind {kind!r}")
        clauses.append(f"({predicate})")
        params.extend(field_params)

    normalized = str(operator or "and").lower()
    if normalized not in {"and", "or", "xor"}:
        normalized = "and"
    if not clauses:
        result = "1"
    elif normalized == "xor":
        result = "(" + " + ".join(f"(CASE WHEN {sql} THEN 1 ELSE 0 END)" for sql in clauses) + ") = 1"
    else:
        result = (" AND " if normalized == "and" else " OR ").join(clauses)

    if uncoded_only:
        result = f"({result}) AND u.observation_id NOT IN (" + (
            "SELECT DISTINCT ca.observation_id FROM current_annotations ca "
            "WHERE ca.value = 'positive'"
        ) + ")"
    return result, params


def register_filter_functions(connection: Any) -> None:
    connection.create_function("GECO_FOLD", 1, lambda raw: None if raw is None else str(raw).casefold())
    connection.create_function(
        "GECO_REGEX", 2,
        lambda pattern, raw: int(re.search(str(pattern), str(raw), flags=re.IGNORECASE) is not None),
    )
    connection.create_function("GECO_DATE", 1, _date_value)
