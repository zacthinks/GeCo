"""One researcher-facing bar chart with typed values and optional group/series.

Existing v1/v2 bar, distribution, and numerical summary recipes remain supported
by their original engines. Only new schema-v2 bar_chart recipes use this module.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import math
from statistics import median
from typing import Any

from geometric_coder.analysis_continuous import (
    MAX_CATEGORIES, _bin, _bin_labels, _categorical_key,
    _category_labels, _edges, _label, _numeric, _parent_rows, _rank,
    _reservoir, _transform, calendar_period,
)
from geometric_coder.analysis_variables import validate_variable


MAX_MEDIAN_SAMPLE = 5000
MAX_BAR_CELLS = 2500
STATS = {"count", "sum", "mean", "median"}


def _variable_identity(variable: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Canonical source/representation identity, not a display name or slot index."""
    return tuple(sorted(variable.items()))


def _series_labels(
    descriptors: list[dict[str, Any]], names: list[str], count: int,
) -> dict[tuple[tuple[str, Any], ...], str]:
    """Disambiguate equal-looking names *after* aggregation keys are determined."""
    identities = [_variable_identity(variable) for variable in descriptors[:count]]
    duplicates = Counter(names[:count])
    labels: dict[tuple[tuple[str, Any], ...], str] = {}
    used: set[str] = set()
    for identifier, variable, name in zip(
        identities, descriptors[:count], names[:count], strict=True
    ):
        if identifier in labels:
            continue
        label = name
        if duplicates[name] > 1:
            family = variable["type"].title()
            if variable["type"] == "code" and variable.get("source") != "labels":
                family += " " + variable["source"]
            label = f"{name} ({family})"
        original = label
        suffix = 2
        while label in used:
            label = f"{original} [{suffix}]"
            suffix += 1
        labels[identifier] = label
        used.add(label)
    return labels


def validate_bar_chart(project: Any, spec: Mapping[str, Any]) -> dict[str, Any]:
    if spec.get("schema_version") != 2 or spec.get("kind") != "bar_chart":
        raise ValueError("Expected a schema-v2 Bar chart.")
    roles = spec.get("roles")
    if not isinstance(roles, Mapping) or set(roles) - {"values", "group", "color"}:
        raise ValueError("Bar chart roles must be Values, Group, or Color.")
    raw_values = roles.get("values")
    if not isinstance(raw_values, list) or not 1 <= len(raw_values) <= 20:
        raise ValueError("Choose between 1 and 20 Values variables.")
    stat = spec.get("statistic", "count")
    if stat not in STATS:
        raise ValueError("Choose Count, Sum, Mean, or Median.")
    normalized = {"values": []}
    seen_values: set[tuple[tuple[str, Any], ...]] = set()
    for item in raw_values:
        if not isinstance(item, Mapping) or set(item) != {"variable"}:
            raise ValueError("Bar Values must contain a variable without a transformation.")
        variable = validate_variable(project, item["variable"])
        identity = _variable_identity(variable)
        if identity in seen_values:
            raise ValueError(
                "Each Bar chart Values variable may be selected only once. "
                "The same variable may still appear in another role."
            )
        seen_values.add(identity)
        if stat != "count" and not _numeric(variable):
            raise ValueError(f"{stat.title()} requires numerical Values (including binary Codes).")
        normalized["values"].append({"variable": variable})
    for role in ("group", "color"):
        item = roles.get(role)
        if item is None:
            continue
        if not isinstance(item, Mapping) or set(item) - {"variable", "transform"} or "variable" not in item:
            raise ValueError(f"{role.title()}: choose a grouping variable.")
        variable = validate_variable(project, item["variable"])
        transform = _transform(item.get("transform"), variable, role)
        normalized[role] = {"variable": variable}
        if transform is not None:
            normalized[role]["transform"] = transform
    descriptor_count = len(normalized["values"]) + sum(
        role in normalized for role in ("group", "color")
    )
    if descriptor_count > 20:
        raise ValueError(
            f"Bar chart selects {descriptor_count} analytical variables across Values, "
            "Group, and Color; the maximum is 20 in total."
        )
    result = {"schema_version": 2, "kind": "bar_chart", "roles": normalized, "statistic": stat}
    if "aggregation" in spec:
        agg = spec["aggregation"]
        if not isinstance(agg, Mapping):
            raise ValueError("Observation aggregation must be an object.")
        keys = project.metadata.get("key_columns", [])[:-1]
        if agg.get("level") not in keys or set(agg) - {"level", "method", "threshold", "numeric_method"}:
            raise ValueError("Select an existing parent level for observation aggregation.")
        method = agg.get("method", "any")
        if method not in ("any", "count", "minimum", "mean", "log_count"):
            raise ValueError("Unknown binary Code parent reduction.")
        numeric_method = agg.get("numeric_method", "mean")
        if numeric_method not in ("mean", "median", "sum"):
            raise ValueError("Choose Mean, Median, or Sum for numerical parent aggregation.")
        threshold = agg.get("threshold", 1)
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold < 0:
            raise ValueError("Invalid Code threshold.")
        result["aggregation"] = {
            "level": agg["level"], "method": method, "threshold": float(threshold),
            "numeric_method": numeric_method,
        }
    if set(spec) - {"schema_version", "kind", "roles", "statistic", "aggregation"}:
        raise ValueError("Unrecognized Bar chart settings.")
    return result


def compute_bar_chart(project: Any, raw_spec: Mapping[str, Any]) -> dict[str, Any]:
    """Bounded server-side frequencies and numerical aggregates by stable observation ID."""
    spec = validate_bar_chart(project, raw_spec)
    roles = spec["roles"]
    entries = [*roles["values"]]
    for role in ("group", "color"):
        if role in roles:
            entries.append(roles[role])
    descriptors = [item["variable"] for item in entries]
    names = [_label(v, project) for v in descriptors]
    n_values = len(roles["values"])
    value_names = _series_labels(descriptors, names, n_values)
    stat = spec["statistic"]
    # Categorical Count Values need only contribute an observed parent value;
    # explicit grouping roles retain their ordinary parent consistency rules.
    categorical_count_values = frozenset(
        i for i in range(n_values)
        if stat == "count" and not _numeric(descriptors[i])
    )
    role_indices = {role: n_values + i for i, role in enumerate(
        role for role in ("group", "color") if role in roles
    )}
    warnings: set[str] = set()
    observed = [0] * len(descriptors)
    coverage = Counter()
    source_info: dict[str, Any] = {"provenance": None, "missing_reasons": []}
    bins: dict[str, list[float]] = {}
    transform_info: dict[str, Any] = {}
    samples: dict[str, list[tuple[int, float]]] = {}
    limits = {role: [math.inf, -math.inf, 0] for role in role_indices
              if roles[role].get("transform", {}).get("kind") == "numeric_bins"}
    timezones = {role: set() for role in role_indices
                 if roles[role].get("transform", {}).get("kind") == "calendar"}

    for identity, values, issues in _parent_rows(
        project, descriptors, spec, source_info=source_info,
        categorical_count_values=categorical_count_values,
    ):
        warnings.update(issues)
        coverage["total"] += 1
        coverage["complete"] += int(all(v is not None for v in values))
        for index, value in enumerate(values):
            observed[index] += int(value is not None)
        for role, bound in limits.items():
            value = values[role_indices[role]]
            if value is None:
                continue
            number = float(value)
            bound[0], bound[1] = min(bound[0], number), max(bound[1], number)
            bound[2] += 1
            if roles[role]["transform"]["method"] == "quantile":
                _reservoir(samples.setdefault(role, []), MAX_MEDIAN_SAMPLE,
                           _rank((identity, role)), number)
        for role, zones in timezones.items():
            value = values[role_indices[role]]
            if value is not None and value["kind"] == "datetime":
                zones.add(value["timezone"])
    if any(len(kinds) > 1 for kinds in timezones.values()):
        raise ValueError("Mixed naive and offset-aware timestamps cannot share calendar grouping.")
    for role, (lo, hi, count) in limits.items():
        transform = roles[role]["transform"]
        if not count:
            edges = _edges([], transform)
        elif transform["method"] == "quantile":
            edges = _edges([value for _, value in samples[role]], transform)
            if edges and hi > lo:
                edges[0] = min(edges[0], lo)
                edges[-1] = max(edges[-1], hi)
            if count > MAX_MEDIAN_SAMPLE:
                warnings.add(f"{role}: numeric bins use a deterministic sample of at most 5,000 observations.")
        else:
            edges = _edges([lo, hi], transform)
        bins[role] = edges
        transform_info[role] = {"edges": edges, "excluded_range": 0}
    labels = {role: _bin_labels(edges) for role, edges in bins.items()}

    def category(role: str, value: Any) -> tuple[str, Any] | None:
        if value is None:
            return None
        transform = roles[role].get("transform")
        if transform and transform["kind"] == "numeric_bins":
            index = _bin(float(value), bins[role])
            if index is None:
                transform_info[role]["excluded_range"] += 1
                return None
            return ("bin", index)
        if transform and transform["kind"] == "calendar":
            return ("calendar", calendar_period(value, transform["unit"]))
        variable = roles[role]["variable"]
        # Parent reductions can turn binary Code judgments into numeric
        # positive counts, proportions, or log-counts. Only atomic labels and
        # binary-valued parent reductions deserve Present/Absent labels.
        binary_reduction = (
            not spec.get("aggregation")
            or spec["aggregation"]["method"] in ("any", "minimum")
        )
        if (variable["type"] == "code" and variable["value"] == "binary"
                and binary_reduction and value in (0, 1)):
            return ("code_label", "Present" if value == 1 else "Absent")
        return _categorical_key(value)

    # (x category, series category) -> exact observed count, sum, bounded median sample.
    cells: dict[tuple[tuple[str, Any], tuple[str, Any]], list[Any]] = {}
    for identity, values, issues in _parent_rows(
        project, descriptors, spec, source_info=source_info, count_missing=False,
        categorical_count_values=categorical_count_values,
    ):
        warnings.update(issues)
        group_key = category("group", values[role_indices["group"]]) if "group" in roles else None
        color_key = category("color", values[role_indices["color"]]) if "color" in roles else None
        if ("group" in roles and group_key is None) or ("color" in roles and color_key is None):
            continue
        for index in range(n_values):
            value = values[index]
            if value is None:
                continue
            value_identity = _variable_identity(descriptors[index])
            # Values identify the measured variable, not an implicit grouping.
            # Count means observed/nonmissing Values within the explicit
            # Group and Color partitions; no categorical Value splitting.
            x = group_key if group_key is not None else ("series", value_identity)
            series = ("series", value_identity)
            if color_key is not None:
                series = ("colored", series, color_key)
            key = (x, series)
            if key not in cells:
                if len(cells) >= MAX_BAR_CELLS:
                    raise ValueError("Bar chart exceeds 2,500 category/series cells.")
                cells[key] = [0, 0.0, []]
            cell = cells[key]
            cell[0] += 1
            if stat in ("sum", "mean", "median"):
                cell[1] += float(value)
                if stat == "median":
                    _reservoir(cell[2], MAX_MEDIAN_SAMPLE, _rank((identity, index)), float(value))
    if stat == "median" and any(cell[0] > MAX_MEDIAN_SAMPLE for cell in cells.values()):
        warnings.add("Medians use deterministic samples of up to 5,000 observed values per bar.")
    x_keys = sorted({x for x, _ in cells}, key=(
        (lambda key: key[1]) if "group" in bins else (lambda key: str(key))
    ))
    s_keys = sorted({s for _, s in cells}, key=str)
    if len(x_keys) > MAX_CATEGORIES or len(s_keys) > MAX_CATEGORIES:
        raise ValueError("Bar chart exceeds 100 categories or series.")
    # Series keys retain canonical variable identities. Format those identities
    # only after aggregation, so same-named Codes and metadata never merge.
    if "group" not in roles:
        x_labels = {key: value_names[key[1]] for key in x_keys}
    else:
        x_labels = _category_labels(x_keys, "group", labels)
    color_keys = sorted({s[2] for s in s_keys if s[0] == "colored"}, key=str)
    color_labels = (_category_labels(color_keys, "color", labels) if color_keys else {})
    def series_name(key: tuple[str, Any]) -> str:
        if key[0] == "colored":
            return series_name(key[1]) + " / " + color_labels[key[2]]
        return value_names[key[1]]

    result_series = []
    for series in s_keys:
        vals: list[float | int | None] = []
        denominators = []
        for x in x_keys:
            cell = cells.get((x, series))
            if cell is None:
                # Count describes observed frequency and may show an empty
                # combination as zero. Sum needs at least one observed Value:
                # no evidence is missing (None), not an observed sum of zero.
                vals.append(0 if stat == "count" else None)
                denominators.append(0)
                continue
            n, total, sample = cell
            denominators.append(n)
            vals.append(n if stat == "count" else total if stat == "sum" else
                        total / n if stat == "mean" else median(value for _, value in sample))
        result_series.append({"name": series_name(series), "values": vals,
                              "denominators": denominators})
    return {
        "kind": "bar_chart", "statistic": stat,
        "categories": [x_labels[key] for key in x_keys],
        "series": result_series, "names": names,
        "xaxis_title": (names[n_values] if "group" in roles else "Values"),
        "coverage": {"total": coverage["total"], "complete": coverage["complete"], "observed": observed},
        "sources": [
            {"descriptor": variable,
             "provenance": source_info["provenance"][i] if source_info["provenance"] else {},
             "missing_reasons": dict(source_info["missing_reasons"][i])
             if source_info["missing_reasons"] else {}}
            for i, variable in enumerate(descriptors)
        ],
        "warnings": sorted(warnings),
        "transformations": transform_info,
        **({"aggregation": spec["aggregation"]} if "aggregation" in spec else {}),
    }
