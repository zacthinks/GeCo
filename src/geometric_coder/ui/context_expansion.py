"""Bounded, incremental text context for researcher-facing readers.

Keep this separate from the unlimited public GeometricCoder.context contract.
"""
from __future__ import annotations

from typing import Any

INITIAL_CONTEXT_UNITS = 200
EXPANDED_CONTEXT_MAX = 5000


def context_key(*parts: Any) -> list[Any]:
    return [str(part) for part in parts]


def current_limit(state: Any, key: list[Any]) -> int:
    if not isinstance(state, dict) or state.get("key") != key:
        return INITIAL_CONTEXT_UNITS
    try:
        return max(INITIAL_CONTEXT_UNITS, min(
            EXPANDED_CONTEXT_MAX, int(state["limit"]),
        ))
    except (KeyError, TypeError, ValueError):
        return INITIAL_CONTEXT_UNITS


def advance_limit(state: Any, key: list[Any], *, expand: bool) -> dict[str, Any]:
    limit = current_limit(state, key)
    if expand:
        limit = min(EXPANDED_CONTEXT_MAX, limit + INITIAL_CONTEXT_UNITS)
    return {"key": key, "limit": limit}


def read_context(
    project: Any, focal_unit_id: int, *, level: int, window: int,
    limit: int,
) -> tuple[list[dict[str, Any]], bool]:
    """One bounded lookahead reveals truncation; preserve closest-N source order."""
    limit = max(INITIAL_CONTEXT_UNITS, min(EXPANDED_CONTEXT_MAX, int(limit)))
    rows = project.context(
        int(focal_unit_id), level=int(level), window=int(window),
        max_units=limit + 1,
    )
    if len(rows) <= limit:
        return rows, False
    focal = next(row for row in rows if int(row["unit_id"]) == int(focal_unit_id))
    position = int(focal["row_position"])
    selected = sorted(
        rows, key=lambda row: (
            abs(int(row["row_position"]) - position),
            int(row["row_position"]),
        ),
    )[:limit]
    return sorted(selected, key=lambda row: int(row["row_position"])), True


def truncation_notice(html: Any, rows: int, truncated: bool, limit: int) -> Any:
    if not truncated:
        return None
    guidance = (
        "Select Load 200 more to extend the visible context and selectable span."
        if limit < EXPANDED_CONTEXT_MAX else
        "Reached the 5,000-unit display limit; move the focal observation to read farther."
    )
    return html.P(
        f"Showing {rows:,} nearby units; more context is hidden. {guidance}",
        className=("context-truncated" if limit < EXPANDED_CONTEXT_MAX
                   else "context-truncated context-at-limit"),
        role="status",
    )
