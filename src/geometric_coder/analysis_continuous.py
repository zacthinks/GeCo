"""Continuous and mixed exploratory chart computations for schema-v2 analyses.

All results contain compact aggregates or a bounded deterministic scatter sample.
Source values are resolved by atomic observation ID, never by DataFrame position.
No fitting, evidence writes, or browser-side raw-corpus transfers occur here.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Mapping
from datetime import datetime, date, timezone, timedelta
import heapq
import math
from statistics import median
from typing import Any

from geometric_coder.analysis_variables import validate_variable
from geometric_coder.analysis_roles import _pages

NUMERIC_KINDS = {"numeric", "position", "probability", "binary"}
CONTINUOUS_KINDS = {
    "scatter", "histogram", "summary_bar", "correlation_heatmap", "value_heatmap",
}
ROLE_LAYOUT = {
    "scatter": ("x", "y", "color"),
    "histogram": ("values", "group"),
    "summary_bar": ("values", "group"),
    "correlation_heatmap": ("values",),
    "value_heatmap": ("values", "rows", "columns"),
}
GROUP_ROLES = {"group", "color", "rows", "columns"}
MAX_CATEGORIES = 100
MAX_CORRELATION_VARIABLES = 12


def _numeric(v: Mapping[str, Any]) -> bool:
    return v["type"] == "scale" or (v["type"] == "metadata" and v["value"] == "numeric") or (
        v["type"] == "code" and v["value"] in {"binary", "probability"}
    )


def _transform(raw: Any, variable: Mapping[str, Any], role: str) -> dict[str, Any] | None:
    if role not in GROUP_ROLES:
        if raw is not None:
            raise ValueError(f"{role}: binning belongs to categorical roles or a histogram.")
        return None
    if variable["type"] == "metadata" and variable["value"] == "categorical":
        if raw is not None:
            raise ValueError(f"{role}: categorical metadata does not need a transformation.")
        return None
    if variable["type"] == "code" and variable["value"] == "binary":
        if raw is not None:
            raise ValueError(f"{role}: a binary Code is already categorical.")
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(f"{role}: explicitly select numeric binning or a calendar unit.")
    if variable["type"] == "metadata" and variable["value"] == "temporal":
        if set(raw) != {"kind", "unit"} or raw["kind"] != "calendar" or raw["unit"] not in (
            "day", "week", "month", "quarter", "year"
        ):
            raise ValueError(f"{role}: select calendar day/week/month/quarter/year.")
        return {"kind": "calendar", "unit": raw["unit"]}
    if not _numeric(variable):
        raise ValueError(f"{role}: unsupported categorical grouping source.")
    if raw.get("kind") != "numeric_bins":
        raise ValueError(f"{role}: numerical grouping requires explicit bins.")
    return _bin_spec(raw, role, kind="numeric_bins")


def _bin_spec(raw: Any, role: str, *, kind: str = "numeric_bins") -> dict[str, Any]:
    if not isinstance(raw, Mapping) or raw.get("kind") != kind:
        raise ValueError(f"{role}: provide numerical binning settings.")
    method = raw.get("method")
    allowed = {
        "equal_width": {"kind", "method", "bins", "min", "max"},
        "quantile": {"kind", "method", "bins"},
        "fixed_width": {"kind", "method", "width", "origin", "min", "max"},
        "explicit": {"kind", "method", "edges"},
    }
    if method not in allowed or set(raw) - allowed[method]:
        raise ValueError(f"{role}: unsupported bin method or settings.")
    out = {"kind": kind, "method": method}
    if method in ("equal_width", "quantile"):
        bins = raw.get("bins", 20 if kind == "histogram_bins" else 10)
        if isinstance(bins, bool) or not isinstance(bins, int) or not 1 <= bins <= 100:
            raise ValueError(f"{role}: bins must be an integer from 1 to 100.")
        out["bins"] = bins
    if method == "fixed_width":
        width = raw.get("width")
        if isinstance(width, bool) or not isinstance(width, (int, float)) or not math.isfinite(width) or width <= 0:
            raise ValueError(f"{role}: fixed bin width must be positive and finite.")
        out["width"] = float(width)
        origin = raw.get("origin", 0)
        if isinstance(origin, bool) or not isinstance(origin, (int, float)) or not math.isfinite(origin):
            raise ValueError(f"{role}: bin origin must be finite.")
        out["origin"] = float(origin)
    if method == "explicit":
        edges = raw.get("edges")
        if not isinstance(edges, list) or not 2 <= len(edges) <= 101 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in edges
        ) or any(b <= a for a, b in zip(edges, edges[1:])):
            raise ValueError(f"{role}: provide 2–101 strictly increasing finite edges.")
        out["edges"] = [float(x) for x in edges]
    for field in ("min", "max"):
        if field in raw:
            value = raw[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{role}: {field} must be finite.")
            out[field] = float(value)
    if "min" in out and "max" in out and out["min"] >= out["max"]:
        raise ValueError(f"{role}: minimum must be smaller than maximum.")
    return out


def _entry(project: Any, raw: Any, role: str, *, grouping: bool = False) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) - {"variable", "transform"} or "variable" not in raw:
        raise ValueError(f"{role}: provide a variable and optional transformation.")
    v = validate_variable(project, raw["variable"])
    if grouping:
        transform = _transform(raw.get("transform"), v, role)
        out = {"variable": v}
        if transform is not None:
            out["transform"] = transform
        return out
    if not _numeric(v):
        raise ValueError(f"{role}: choose a numeric Code, Scale, or metadata representation.")
    if "transform" in raw:
        raise ValueError(f"{role}: numeric values must remain unbinned.")
    return {"variable": v}


def validate_continuous_analysis(project: Any, spec: Mapping[str, Any]) -> dict[str, Any]:
    kind = spec.get("kind")
    if spec.get("schema_version") != 2 or kind not in CONTINUOUS_KINDS:
        raise ValueError("Unknown schema-v2 continuous visualization.")
    roles = spec.get("roles")
    if not isinstance(roles, Mapping) or set(roles) - set(ROLE_LAYOUT[kind]):
        raise ValueError("Assign only roles supported by this visualization.")
    clean = {}
    for role in ROLE_LAYOUT[kind]:
        raw = roles.get(role)
        if role == "values" and kind == "correlation_heatmap":
            if not isinstance(raw, list) or not 2 <= len(raw) <= MAX_CORRELATION_VARIABLES:
                raise ValueError("Correlation heatmap needs 2–12 numeric Values.")
            clean[role] = [_entry(project, item, role) for item in raw]
        elif role in GROUP_ROLES:
            if raw is None:
                if role in ("group", "color"):
                    if kind == "summary_bar":
                        raise ValueError("Group is required for a numerical summary bar.")
                    continue
                raise ValueError(f"{role.title()} is required.")
            clean[role] = _entry(project, raw, role, grouping=True)
        else:
            if raw is None:
                raise ValueError(f"{role.title()} is required.")
            clean[role] = _entry(project, raw, role)
    descriptors = [item["variable"] for v in clean.values()
                   for item in (v if isinstance(v, list) else [v])]
    if len(descriptors) > 20:
        raise ValueError("Too many analytical variables.")
    # Reusing a source in separate roles is meaningful (e.g. x=y); columns
    # are positional and their independent settings are preserved.
    result = {"schema_version": 2, "kind": kind, "roles": clean}
    if kind == "histogram":
        result["binning"] = _bin_spec(
            spec.get("binning", {"kind": "histogram_bins", "method": "equal_width", "bins": 20}),
            "Histogram", kind="histogram_bins",
        )
    elif "binning" in spec:
        raise ValueError("Binning settings apply only to a histogram.")
    if kind in ("summary_bar", "value_heatmap"):
        statistic = spec.get("statistic", "mean")
        if statistic not in ("mean", "median"):
            raise ValueError("Numerical summaries support mean or median.")
        result["statistic"] = statistic
    elif "statistic" in spec:
        raise ValueError("This chart does not use a summary statistic.")
    if kind == "scatter":
        limit = spec.get("max_points", 2000)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 5000:
            raise ValueError("Scatter sample must contain 1–5000 points.")
        result["max_points"] = limit
    elif "max_points" in spec:
        raise ValueError("Only scatterplots use point sampling.")
    if "aggregation" in spec:
        aggregate = spec["aggregation"]
        if not isinstance(aggregate, Mapping):
            raise ValueError("Aggregation must be an object.")
        keys = project.metadata.get("key_columns", [])[:-1]
        if aggregate.get("level") not in keys or set(aggregate) - {
            "level", "method", "threshold", "numeric_method"
        }:
            raise ValueError("Choose a valid parent-level aggregation key.")
        method = aggregate.get("method", "any")
        if method not in ("any", "count", "minimum", "mean", "log_count"):
            raise ValueError("Unknown binary Code parent reduction.")
        threshold = aggregate.get("threshold", 1)
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not math.isfinite(threshold) or threshold < 0:
            raise ValueError("Invalid Code threshold.")
        numeric_method = aggregate.get("numeric_method", "mean")
        if numeric_method not in ("mean", "median"):
            raise ValueError("Numeric parent reduction must be mean or median.")
        result["aggregation"] = {"level": aggregate["level"], "method": method,
                                 "threshold": float(threshold), "numeric_method": numeric_method}
    extra = set(spec) - {"schema_version", "kind", "roles", "binning", "statistic", "max_points", "aggregation"}
    if extra:
        raise ValueError(f"Unknown continuous chart settings: {', '.join(sorted(extra))}.")
    return result


def _label(variable: Mapping[str, Any], project: Any) -> str:
    if variable["type"] == "code":
        return str(project.code(variable["code_id"])["name"])
    if variable["type"] == "scale":
        return str(project.scale(variable["scale_id"])["name"]) + " (position)"
    return variable["field"]


def _edges(numbers: list[float], spec: Mapping[str, Any]) -> list[float]:
    method = spec["method"]
    if method == "explicit":
        return list(spec["edges"])
    if not numbers:
        return []
    lo, hi = min(numbers), max(numbers)
    lo = spec.get("min", lo)
    hi = spec.get("max", hi)
    if hi < lo:
        return []
    if method == "fixed_width":
        width, origin = spec["width"], spec["origin"]
        first = math.floor((lo - origin) / width)
        last = math.floor((hi - origin) / width) + 1
        if last - first > 100:
            raise ValueError("Fixed-width bins exceed 100 intervals; increase width or restrict the range.")
        return [origin + (first + i) * width for i in range(last - first + 1)]
    bins = spec["bins"]
    if method == "quantile":
        ordered = sorted(numbers)
        points = [ordered[min(len(ordered) - 1, int(i * (len(ordered) - 1) / bins))]
                  for i in range(bins + 1)]
        # Tied quantiles collapse rather than creating zero-width bins.
        dedup = list(dict.fromkeys(points))
        return dedup if len(dedup) >= 2 else [ordered[0] - 0.5, ordered[0] + 0.5]
    if hi == lo:
        delta = max(abs(lo) * 0.01, 0.5)
        lo -= delta
        hi += delta
    step = (hi - lo) / bins
    return [lo + step * i for i in range(bins)] + [hi]


def _bin(value: float | None, edges: list[float]) -> int | None:
    if value is None or len(edges) < 2 or value < edges[0] or value > edges[-1]:
        return None
    return min(len(edges) - 2, bisect_right(edges, value) - 1)


def _bin_labels(edges: list[float]) -> list[str]:
    return [f"[{a:.6g}, {b:.6g}{']' if i == len(edges) - 2 else ')'}"
            for i, (a, b) in enumerate(zip(edges, edges[1:]))]


def calendar_period(value: Mapping[str, str], unit: str) -> str:
    """Date-only and naive timestamps retain their written calendar day.

    Offset-aware timestamps are converted to UTC for period grouping. Weeks
    start on Monday and use ISO week-years; months are YEAR-month, not pooled.
    """
    if value["kind"] == "date":
        day = date.fromisoformat(value["iso"])
    else:
        dt = datetime.fromisoformat(value["iso"])
        if value["timezone"] == "offset":
            dt = dt.astimezone(timezone.utc)
        day = dt.date()
    if unit == "day":
        return day.isoformat()
    if unit == "week":
        year, week, _ = day.isocalendar()
        return f"{year:04d}-W{week:02d}"
    if unit == "month":
        return f"{day.year:04d}-{day.month:02d}"
    if unit == "quarter":
        return f"{day.year:04d}-Q{(day.month - 1) // 3 + 1}"
    return f"{day.year:04d}"


def _categories(rows: list[tuple[Any, ...]], index: int, spec: Mapping[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    """Return transformed categories and their coverage, without modifying sources."""
    entry = list(spec["roles"].values())[index]  # caller passes positional role index
    transform = entry.get("transform")
    raw = [r[index] for r in rows]
    if transform and transform["kind"] == "numeric_bins":
        edges = _edges([float(x) for x in raw if x is not None], transform)
        labels = _bin_labels(edges)
        out = [None if _bin(x, edges) is None else labels[_bin(x, edges)] for x in raw]
        return out, {"edges": edges, "excluded_range": sum(x is not None and _bin(x, edges) is None for x in raw)}
    if transform and transform["kind"] == "calendar":
        zones = {x["timezone"] for x in raw if x is not None and x["kind"] == "datetime"}
        if len(zones) > 1:
            raise ValueError("Temporal metadata mixes naive and offset-aware timestamps; choose a consistent source before calendar grouping.")
        return [calendar_period(x, transform["unit"]) if x is not None else None for x in raw], {}
    return [str(v) if v is not None else None for v in raw], {}


def _parent_rows(project: Any, descriptors: list[dict[str, Any]], spec: Mapping[str, Any]):
    """Yield aligned group rows at the requested parent grain.

    Numeric sources use mean/median, binary Codes use selected observed-only
    reductions; constant categorical/temporal metadata is preserved. A parent
    with no observed values remains missing. SQL reads only key JSON.
    """
    aggregation = spec.get("aggregation")
    if not aggregation:
        for page in _pages(project, descriptors):
            for i, observation_id in enumerate(page["observation_ids"]):
                yield (observation_id, tuple(col["values"][i] for col in page["columns"]), page["warnings"])
        return
    keys = project.metadata["key_columns"]
    prefix = keys[:keys.index(aggregation["level"]) + 1]
    import json
    parents: dict[tuple[Any, ...], list[list[Any]]] = {}
    warnings: set[str] = set()
    for page in _pages(project, descriptors):
        warnings.update(page["warnings"])
        ids = page["observation_ids"]
        with project.database.connect() as conn:
            placeholders = ",".join("?" for _ in ids)
            fetched = conn.execute(
                f"SELECT observation_id, user_key_json FROM units WHERE observation_id IN ({placeholders})",
                ids,
            ).fetchall()
        parent_for_id = {int(row[0]): tuple(json.loads(row[1]).get(k) for k in prefix) for row in fetched}
        for i, observation_id in enumerate(ids):
            parent = parent_for_id[observation_id]
            state = parents.setdefault(parent, [[] for _ in descriptors])
            for j, col in enumerate(page["columns"]):
                value = col["values"][i]
                if value is not None:
                    state[j].append(value)
    for parent in sorted(parents, key=lambda p: tuple(str(v) for v in p)):
        state = parents[parent]
        reduced = []
        for values, variable in zip(state, descriptors):
            if not values:
                reduced.append(None)
            elif variable["type"] == "code" and variable["value"] == "binary":
                positives = sum(x == 1 for x in values)
                method = aggregation["method"]
                reduced.append({
                    "any": lambda: float(positives > 0),
                    "minimum": lambda: float(positives >= aggregation["threshold"]),
                    "count": lambda: float(positives),
                    "mean": lambda: float(positives / len(values)),
                    "log_count": lambda: math.log1p(positives),
                }[method]())
            elif _numeric(variable):
                reduced.append(median(values) if aggregation["numeric_method"] == "median"
                               else sum(values) / len(values))
            else:
                if any(value != values[0] for value in values[1:]):
                    raise ValueError("Selected grouping metadata varies within a parent group.")
                reduced.append(values[0])
        yield (parent, tuple(reduced), sorted(warnings))


def compute_continuous_analysis(project: Any, spec: Mapping[str, Any]) -> dict[str, Any]:
    spec = validate_continuous_analysis(project, spec)
    kind = spec["kind"]
    roles = spec["roles"]
    role_names = list(roles)
    entries = [item for role in role_names
               for item in (roles[role] if isinstance(roles[role], list) else [roles[role]])]
    descriptors = [item["variable"] for item in entries]
    names = [_label(v, project) for v in descriptors]
    rows = []
    warnings: set[str] = set()
    coverage = Counter()
    for identity, values, issues in _parent_rows(project, descriptors, spec):
        warnings.update(issues)
        coverage["total"] += 1
        coverage["complete"] += int(all(v is not None for v in values))
        for j, value in enumerate(values):
            coverage[f"observed_{j}"] += int(value is not None)
        rows.append((identity, *values))
    data = [row[1:] for row in rows]
    base = {
        "kind": kind, "warnings": sorted(warnings),
        "coverage": {"total": coverage["total"], "complete": coverage["complete"],
                     "observed": [coverage[f"observed_{i}"] for i in range(len(descriptors))]},
        "names": names,
    }
    group_indices = {role: sum(len(roles[prev]) if isinstance(roles[prev], list) else 1
                               for prev in role_names[:role_names.index(role)])
                     for role in role_names if role in GROUP_ROLES}
    transformed = {}
    transformations = {}
    for role, i in group_indices.items():
        entry = roles[role]
        transform = entry.get("transform")
        raw = [row[i] for row in data]
        if transform and transform["kind"] == "numeric_bins":
            edges = _edges([float(x) for x in raw if x is not None], transform)
            labels = _bin_labels(edges)
            mapped = [labels[j] if (j := _bin(v, edges)) is not None else None for v in raw]
            transformations[role] = {"edges": edges,
                                     "excluded_range": sum(v is not None and _bin(v, edges) is None for v in raw)}
        elif transform and transform["kind"] == "calendar":
            zones = {x["timezone"] for x in raw if x is not None and x["kind"] == "datetime"}
            if len(zones) > 1:
                raise ValueError("Mixed naive and offset-aware timestamps cannot be grouped together without an explicit timezone policy.")
            mapped = [calendar_period(v, transform["unit"]) if v is not None else None for v in raw]
        else:
            mapped = [str(v) if v is not None else None for v in raw]
        transformed[role] = mapped
    base["transformations"] = transformations

    if kind == "scatter":
        max_points = spec["max_points"]
        heap = []
        eligible = 0
        for idx, row in enumerate(data):
            x, y = row[:2]
            if x is None or y is None:
                continue
            category = transformed.get("color", [None] * len(data))[idx]
            if "color" in roles and category is None:
                continue
            eligible += 1
            # Stable hash of the atomic ID avoids first-N or order bias.
            import hashlib
            rank = int.from_bytes(hashlib.blake2b(repr(rows[idx][0]).encode(), digest_size=8).digest(), "big")
            item = (-rank, idx, float(x), float(y), category)
            if len(heap) < max_points:
                heapq.heappush(heap, item)
            elif rank < -heap[0][0]:
                heapq.heapreplace(heap, item)
        ordered = sorted(heap, key=lambda entry: -entry[0])
        base.update(points=[{"id": rows[j][0], "x": x, "y": y, "color": cat}
                            for _, j, x, y, cat in ordered],
                    eligible=eligible, sampled=len(ordered),
                    sampling="stable hash of observation identity; no extrapolation")
        return base

    if kind == "histogram":
        values = [r[0] for r in data]
        edges = _edges([float(x) for x in values if x is not None], spec["binning"])
        labels = _bin_labels(edges)
        groups = transformed.get("group", ["All observations"] * len(data))
        category_order = sorted({g for g in groups if g is not None})
        if len(category_order) > MAX_CATEGORIES:
            raise ValueError("More than 100 histogram groups.")
        counts = {g: [0] * len(labels) for g in category_order}
        excluded = 0
        missing_group = 0
        for value, group in zip(values, groups):
            if value is None:
                continue
            if group is None:
                missing_group += 1
                continue
            j = _bin(value, edges)
            if j is None:
                excluded += 1
            else:
                counts[group][j] += 1
        base.update(edges=edges, bins=labels, groups=category_order, counts=counts,
                    excluded_range=excluded, missing_group=missing_group,
                    denominators={group: sum(counts[group]) for group in category_order})
        return base

    if kind in ("summary_bar", "value_heatmap"):
        groups = transformed["group"] if kind == "summary_bar" else transformed["rows"]
        columns = ([None] * len(data) if kind == "summary_bar" else transformed["columns"])
        buckets: dict[tuple[str, str | None], list[float]] = defaultdict(list)
        for row, group, col in zip(data, groups, columns):
            if group is not None and (kind == "summary_bar" or col is not None) and row[0] is not None:
                buckets[(group, col)].append(float(row[0]))
        labels = sorted({key[0] for key in buckets})
        cols = sorted({key[1] for key in buckets if key[1] is not None})
        if len(labels) > MAX_CATEGORIES or len(cols) > MAX_CATEGORIES or len(labels) * max(1, len(cols)) > 2500:
            raise ValueError("Numerical summary has too many category cells.")
        statistic = spec["statistic"]
        summary = lambda vals: (median(vals) if statistic == "median" else sum(vals) / len(vals)) if vals else None
        if kind == "summary_bar":
            base.update(categories=labels, values=[summary(buckets[(g, None)]) for g in labels],
                        denominators=[len(buckets[(g, None)]) for g in labels],
                        statistic=statistic)
        else:
            base.update(rows=labels, columns=cols,
                        values=[[summary(buckets.get((g, c), [])) for c in cols] for g in labels],
                        samples=[[len(buckets.get((g, c), [])) for c in cols] for g in labels],
                        statistic=statistic)
        return base

    # Pairwise-complete Pearson correlation. A constant variable has no r.
    size = len(descriptors)
    samples = [[0] * size for _ in range(size)]
    matrix = [[None] * size for _ in range(size)]
    for i in range(size):
        for j in range(i, size):
            paired = [(float(row[i]), float(row[j])) for row in data
                      if row[i] is not None and row[j] is not None]
            n = len(paired)
            samples[i][j] = samples[j][i] = n
            if n >= 2:
                sx = sum(x for x, _ in paired)
                sy = sum(y for _, y in paired)
                xx = sum(x * x for x, _ in paired)
                yy = sum(y * y for _, y in paired)
                xy = sum(x * y for x, y in paired)
                vx, vy = n * xx - sx * sx, n * yy - sy * sy
                if vx > 1e-12 and vy > 1e-12:
                    matrix[i][j] = matrix[j][i] = max(-1., min(1., (n * xy - sx * sy) / math.sqrt(vx * vy)))
    base.update(values=matrix, samples=samples, measure="correlation")
    return base
