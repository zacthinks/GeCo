"""Executable schema-v2 role recipes for the existing exploratory charts.

This intentionally is not a general pivot-table engine. Role recipes adapt to
legacy SQL aggregations wherever possible; only committee-sourced binary charts
use bounded server-side batches from the typed resolver. No analysis runs train
models, mutate evidence, or ship per-observation values to the browser.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import math
from typing import Any

from geometric_coder.analysis_variables import (
    MAX_RESOLUTION_OBSERVATIONS, resolve_analytical_variables, validate_variable,
)

# The names express the *actual* mathematical operation, not future chart types.
ROLE_CONTRACT = {
    "bar": {"values": ("code",), "group": ("metadata",)},
    "distribution": {"values": ("code",)},
    "heatmap": {"indicators": ("code", "metadata")},
}


def _entry(project: Any, item: Any, allowed: tuple[str, ...], role: str) -> dict[str, Any]:
    if not isinstance(item, Mapping) or set(item) != {"variable"}:
        raise ValueError(f"{role}: provide a variable; transformations are not implemented in Phase 2.")
    variable = validate_variable(project, item["variable"])
    if variable["type"] not in allowed:
        raise ValueError(f"{role}: this variable family requires a different visualization (Phase 3).")
    if variable["type"] == "scale":
        raise ValueError("Scale positions require a continuous visualization (Phase 3); no automatic bins.")
    if variable["type"] == "metadata" and variable["value"] != "categorical":
        raise ValueError(
            f"{role}: {variable['value']} metadata needs an explicit supported chart "
            "transformation (Phase 3); dates are not treated as categories."
        )
    if variable["type"] == "code" and variable["value"] != "binary":
        raise ValueError("Code probabilities require a numerical visualization (Phase 3); no thresholding.")
    return {"variable": variable}


def validate_role_analysis(project: Any, spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a complete, executable v2 chart with role-scoped source values."""
    from geometric_coder.analyses import validate_analysis

    if spec.get("schema_version") != 2:
        raise ValueError("Role-based analyses require schema_version 2.")
    kind = spec.get("kind")
    if kind not in ROLE_CONTRACT:
        raise ValueError("This visualization is not implemented.")
    if not isinstance(spec.get("roles"), Mapping):
        raise ValueError("Assign variables to the visualization's roles.")
    roles = spec["roles"]
    if set(roles) - set(ROLE_CONTRACT[kind]):
        raise ValueError("This visualization does not support the supplied roles.")
    normalized: dict[str, Any] = {}
    for role, allowed in ROLE_CONTRACT[kind].items():
        raw = roles.get(role, None)
        if role == "group":
            if raw is not None:
                normalized[role] = _entry(project, raw, allowed, role)
        else:
            if not isinstance(raw, list) or not raw:
                raise ValueError(f"Choose at least one {role} variable.")
            normalized[role] = [_entry(project, item, allowed, role) for item in raw]
    entries = normalized.get("values", normalized.get("indicators", []))
    if kind == "heatmap" and len(entries) < 2:
        raise ValueError("Association matrices require at least two indicators.")
    variables = [item["variable"] for item in entries]
    if "group" in normalized:
        variables.append(normalized["group"]["variable"])
    if len(variables) > 20:
        raise ValueError("Choose at most 20 variables.")
    identities = [(v["type"], v.get("code_id", v.get("field"))) for v in variables]
    if len(set(identities)) != len(identities):
        raise ValueError("A variable may only appear once in these existing chart computations.")
    measure = spec.get("measure", "count")
    aggregate = spec.get("aggregation")
    has_committee = any(
        v["type"] == "code" and v.get("predictor_kind") == "committee"
        for v in variables
    )
    if has_committee and aggregate is not None:
        raise ValueError("Committee predictions with parent-level aggregation need Phase 3.")
    if has_committee and kind == "heatmap" and any(v["type"] == "metadata" for v in variables):
        raise ValueError("Committee/metadata association matrices need Phase 3.")
    # Reuse the entire legacy compatibility and aggregation policy, including
    # heatmap measure and group checks, without altering schema-v1 definitions.
    legacy_vars = []
    for variable in variables:
        v = dict(variable)
        if v["type"] == "code":
            v.pop("predictor_kind", None)
            if v.get("source") != "labels" and variable.get("predictor_kind") == "committee":
                # Only the compatibility check uses a human placeholder; the
                # executable v2 spec retains the selected committee identity.
                v = {"type": "code", "code_id": v["code_id"],
                     "source": "labels", "value": "binary"}
        else:
            v.pop("numeric_strings", None)
            v.pop("value", None)
        legacy_vars.append(v)
    validate_analysis(project, {
        "kind": kind, "measure": measure, "variables": legacy_vars,
        **({"aggregation": aggregate} if aggregate is not None else {}),
    })
    result = {"schema_version": 2, "kind": kind, "measure": measure,
              "roles": normalized}
    if aggregate is not None:
        # Preserve the canonical v1 method/threshold normalization.
        result["aggregation"] = validate_analysis(project, {
            "kind": kind, "measure": measure, "variables": legacy_vars,
            "aggregation": aggregate,
        })["aggregation"]
    return result



def ensure_role_sources_ready(project: Any, spec: Mapping[str, Any]) -> None:
    """Gate v2 saves on retained prediction fits, without training or scoring.

    This is a save-time check, not a validation rule for stored live recipes:
    a model can later be removed or become stale, and the original recipe
    must remain readable and report its current execution state. A stale
    retained fit remains usable with the existing warning semantics.
    """
    roles = spec["roles"]
    entries = roles.get("values", roles.get("indicators", []))
    checked: set[tuple[int, str, int]] = set()
    for item in entries:
        variable = item["variable"]
        if variable["type"] != "code" or variable["source"] == "labels":
            continue
        code_id = variable["code_id"]
        kind = variable.get("predictor_kind", "classifier")
        predictor_id = variable["predictor_id"]
        key = (code_id, kind, predictor_id)
        if key in checked:
            continue
        checked.add(key)
        name = project.code(code_id)["name"]
        if kind == "classifier":
            statuses = project.classifier_status(
                code_id=code_id, classifier_spec_ids=[predictor_id]
            )["classifiers"]
            ready = bool(statuses and statuses[0].get("classifier_fit_id") is not None)
        else:
            state = project.classifier_committee_status(
                code_id=code_id, committee_id=predictor_id
            )
            # Fixed committees require retained fits for all members; learned
            # committees require their own trained snapshot. Both report
            # "not_trained" when no executable fitted prediction source exists.
            ready = state["status"] != "not_trained"
        if not ready:
            raise ValueError(
                f"{name}: selected {kind} has no trained prediction fit. "
                "Train the selected predictor before saving this analysis."
            )


def ensure_role_chart_executable(project: Any, spec: Mapping[str, Any]) -> None:
    """Check data-dependent chart bounds before persisting a new v2 recipe.

    The schema cannot know metadata cardinality, expanded heatmap dimensions,
    parent-level bin counts, or metadata consistency. The existing chart engine
    already enforces all of these with bounded SQL aggregates (or bounded
    committee resolver pages), and returns only compact statistics.
    Avoid recomputing ordinary ungrouped Code-only charts, which have no
    data-dependent dimension or bin limits.
    """
    roles = spec["roles"]
    entries = roles.get("values", roles.get("indicators", []))
    variables = [item["variable"] for item in entries]
    if "group" in roles:
        variables.append(roles["group"]["variable"])
    needs_data_check = (
        any(variable["type"] == "metadata" for variable in variables)
        or (spec["kind"] == "distribution" and "aggregation" in spec)
    )
    if needs_data_check:
        # This read-only computation is not stored and never fits predictors.
        compute_role_analysis(project, spec)


def to_legacy_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a validated *non-committee* v2 recipe without semantic loss."""
    variables = [dict(item["variable"])
                 for item in spec["roles"].get("values", spec["roles"].get("indicators", []))]
    if "group" in spec["roles"]:
        variables.append(dict(spec["roles"]["group"]["variable"]))
    legacy = []
    for variable in variables:
        if variable.get("predictor_kind") == "committee":
            raise ValueError("Committee sources use the bounded role computation.")
        variable.pop("predictor_kind", None)
        if variable["type"] == "metadata":
            variable.pop("numeric_strings", None)
            variable.pop("value", None)
        legacy.append(variable)
    return {"kind": spec["kind"], "measure": spec["measure"], "variables": legacy,
            **({"aggregation": spec["aggregation"]} if "aggregation" in spec else {})}


def _pages(project: Any, descriptors: list[dict[str, Any]]):
    """Read only IDs and bounded typed columns, never unit texts or full arrays."""
    offset = 0
    while True:
        with project.database.connect() as conn:
            rows = conn.execute(
                "SELECT observation_id FROM units ORDER BY row_position LIMIT ? OFFSET ?",
                (MAX_RESOLUTION_OBSERVATIONS, offset),
            ).fetchall()
        if not rows:
            break
        ids = [int(row[0]) for row in rows]
        yield resolve_analytical_variables(project, descriptors, ids)
        offset += len(ids)


def _committee_analysis(project: Any, spec: Mapping[str, Any]) -> dict[str, Any]:
    """Atomic binary committee execution, accumulating only compact statistics."""
    kind = spec["kind"]
    entries = spec["roles"].get("values", spec["roles"].get("indicators", []))
    descriptors = [item["variable"] for item in entries]
    group = spec["roles"].get("group")
    if group:
        descriptors.append(group["variable"])
    code_count = len(entries)
    names = [
        str(project.code(v["code_id"])["name"]) if v["type"] == "code" else v["field"]
        for v in descriptors[:code_count]
    ]
    warnings: set[str] = set()
    if kind == "bar":
        groups: dict[str, list[list[int]]] = defaultdict(
            lambda: [[0, 0] for _ in range(code_count)]
        )
    elif kind == "distribution":
        frequencies: list[dict[float, int]] = [defaultdict(int) for _ in range(code_count)]
    else:
        # n, sum(x), sum(y), sum(x^2), sum(y^2), sum(xy), union-positive
        matrix = [[[0.0] * 7 for _ in range(code_count)] for _ in range(code_count)]
    for page in _pages(project, descriptors):
        warnings.update(page["warnings"])
        columns = page["columns"]
        for i in range(page["row_count"]):
            values = [col["values"][i] for col in columns[:code_count]]
            if kind == "bar":
                category = (
                    "(Missing)" if group and columns[-1]["values"][i] is None
                    else str(columns[-1]["values"][i]) if group
                    else "All observations"
                )
                if category not in groups and len(groups) >= 100:
                    raise ValueError("Metadata has more than 100 categories.")
                counts = groups[category]
                for j, value in enumerate(values):
                    if value is not None:
                        counts[j][1] += 1
                        counts[j][0] += int(value == 1)
            elif kind == "distribution":
                for j, value in enumerate(values):
                    if value is not None:
                        frequencies[j][float(value)] += 1
                        if len(frequencies[j]) > 100:
                            raise ValueError("More than 100 distinct code-value bins.")
            else:
                for a, x in enumerate(values):
                    if x is None:
                        continue
                    for b, y in enumerate(values):
                        if y is None:
                            continue
                        cell = matrix[a][b]
                        cell[0] += 1
                        cell[1] += x
                        cell[2] += y
                        cell[3] += x*x
                        cell[4] += y*y
                        cell[5] += x*y
                        cell[6] += float(x == 1 or y == 1)
    if kind == "bar":
        if len(groups) > 100:
            raise ValueError("Metadata has more than 100 categories.")
        categories = sorted(groups)
        return {"kind": "bar", "names": names, "categories": categories,
                "counts": [[groups[g][j][0] for g in categories] for j in range(code_count)],
                "denominators": [[groups[g][j][1] for g in categories] for j in range(code_count)],
                "warnings": sorted(warnings)}
    if kind == "distribution":
        return {"kind": "distribution", "names": names,
                "histograms": [
                    {"values": sorted(f), "counts": [f[v] for v in sorted(f)]}
                    for f in frequencies
                ], "aggregation": None,
                "atomic_level": str(project.metadata.get("key_columns", ["observation"])[-1]),
                "warnings": sorted(warnings)}
    n = code_count
    values_out: list[list[float | None]] = [[None] * n for _ in range(n)]
    samples = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            count, sx, sy, xx, yy, xy, union = matrix[i][j]
            samples[i][j] = int(count)
            if spec["measure"] == "count":
                value = xy
            elif spec["measure"] == "jaccard":
                value = (xy / union if union else 0.0) if count else None
            else:
                vx, vy = count*xx-sx*sx, count*yy-sy*sy
                value = ((count*xy-sx*sy) / math.sqrt(vx*vy)
                         if count >= 2 and vx > 1e-12 and vy > 1e-12 else None)
                if value is not None:
                    value = max(-1.0, min(1.0, value))
            values_out[i][j] = value
    return {"kind": "heatmap", "names": names, "values": values_out,
            "samples": samples, "measure": spec["measure"], "warnings": sorted(warnings)}


def compute_role_analysis(project: Any, spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate before any data read; adapt stable legacy SQL whenever possible."""
    from geometric_coder.analyses import compute_analysis
    normalized = validate_role_analysis(project, spec)
    refs = [item["variable"] for item in
            normalized["roles"].get("values", normalized["roles"].get("indicators", []))]
    if any(v.get("predictor_kind") == "committee" for v in refs):
        return _committee_analysis(project, normalized)
    return compute_analysis(project, to_legacy_spec(normalized))
