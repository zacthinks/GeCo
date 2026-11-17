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
    """Yield atomic rows or bounded-memory parent-grain reductions by stable keys.

    No raw observations are sent to the browser. Parent states retain only
    numerical moments, counts, a bounded median sample, and grouping labels.
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
    # Each source stores [n, sum, positives, first categorical value, reservoir].
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
            if parent not in parents and len(parents) >= 100000:
                raise ValueError("More than 100,000 parent groups; choose a coarser observation level.")
            state = parents.setdefault(parent, [[0, 0., 0, None, []] for _ in descriptors])
            for j, col in enumerate(page["columns"]):
                value = col["values"][i]
                if value is None:
                    continue
                cell = state[j]
                cell[0] += 1
                variable = descriptors[j]
                if variable["type"] == "code" and variable["value"] == "binary":
                    cell[2] += int(value == 1)
                elif _numeric(variable):
                    cell[1] += float(value)
                    if aggregation["numeric_method"] == "median":
                        _reservoir(cell[4], 5000, _rank(observation_id), float(value))
                else:
                    if cell[0] > 1 and value != cell[3]:
                        raise ValueError(
                            "Selected grouping metadata varies within a parent group."
                        )
                    cell[3] = value
    for parent in sorted(parents, key=lambda p: tuple(str(v) for v in p)):
        state = parents[parent]
        reduced = []
        for cell, variable in zip(state, descriptors):
            count, total, positives, first, sample = cell
            if not count:
                reduced.append(None)
            elif variable["type"] == "code" and variable["value"] == "binary":
                method = aggregation["method"]
                if method == "any":
                    reduced.append(float(positives > 0))
                elif method == "minimum":
                    reduced.append(float(positives >= aggregation["threshold"]))
                elif method == "count":
                    reduced.append(float(positives))
                elif method == "mean":
                    reduced.append(positives / count)
                else:
                    reduced.append(math.log1p(positives))
            elif _numeric(variable):
                if aggregation["numeric_method"] == "median":
                    reduced.append(median(item[1] for item in sample))
                    if count > 5000:
                        warnings.add("Parent-level median uses bounded deterministic 5,000-point samples.")
                else:
                    reduced.append(total / count)
            else:
                reduced.append(first)
        yield (parent, tuple(reduced), sorted(warnings))


def _rank(identity: Any) -> int:
    import hashlib
    return int.from_bytes(
        hashlib.blake2b(repr(identity).encode("utf-8"), digest_size=8).digest(), "big"
    )


def _reservoir(heap: list[tuple[int, Any]], limit: int, rank: int, value: Any) -> None:
    item = (-rank, value)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item[0] > heap[0][0]:
        heapq.heapreplace(heap, item)


def _group_positions(roles: Mapping[str, Any]) -> dict[str, int]:
    positions = {}
    i = 0
    for role, value in roles.items():
        if role in GROUP_ROLES:
            positions[role] = i
        i += len(value) if isinstance(value, list) else 1
    return positions


def compute_continuous_analysis(project: Any, spec: Mapping[str, Any]) -> dict[str, Any]:
    """Compute bounded scatter samples, grouped summaries, and pairwise moments.

    Histograms scan the corpus twice: once for exact range and a deterministic
    bounded quantile-edge sample, then for exact cell frequencies. Median is
    exact for cells with at most 5,000 observations and uses a deterministic
    bounded observation sample for larger cells (reported as approximate).
    """
    spec = validate_continuous_analysis(project, spec)
    kind, roles = spec["kind"], spec["roles"]
    entries = [item for value in roles.values()
               for item in (value if isinstance(value, list) else [value])]
    descriptors = [item["variable"] for item in entries]
    names = [_label(v, project) for v in descriptors]
    position = _group_positions(roles)
    warnings: set[str] = set()
    coverage = Counter()
    observed = [0] * len(descriptors)
    MAX_SAMPLE = 5000

    # A first, bounded scan is needed to determine numerical bin edges before
    # accumulating exact histogram counts and transformed grouped summaries.
    bin_specs = {role: item["transform"]
                 for role, item in roles.items()
                 if role in GROUP_ROLES and item.get("transform", {}).get("kind") == "numeric_bins"}
    if kind == "histogram":
        bin_specs["__histogram__"] = spec["binning"]
    limits = {}
    samples = {role: [] for role, settings in bin_specs.items()
               if settings["method"] == "quantile"}
    for role in bin_specs:
        limits[role] = [math.inf, -math.inf, 0]
    timezone_kinds = {role: set() for role, item in roles.items()
                      if role in GROUP_ROLES and item.get("transform", {}).get("kind") == "calendar"}
    for identity, values, issues in _parent_rows(project, descriptors, spec):
        warnings.update(issues)
        coverage["total"] += 1
        coverage["complete"] += int(all(v is not None for v in values))
        for i, value in enumerate(values):
            observed[i] += int(value is not None)
        for role, settings in bin_specs.items():
            index = 0 if role == "__histogram__" else position[role]
            value = values[index]
            if value is None:
                continue
            value = float(value)
            bounds = limits[role]
            bounds[0] = min(bounds[0], value)
            bounds[1] = max(bounds[1], value)
            bounds[2] += 1
            if role in samples:
                _reservoir(samples[role], MAX_SAMPLE, _rank((identity, role)), value)
        for role in timezone_kinds:
            value = values[position[role]]
            if value is not None and value["kind"] == "datetime":
                timezone_kinds[role].add(value["timezone"])
    if any(len(zones) > 1 for zones in timezone_kinds.values()):
        raise ValueError(
            "Mixed naive and offset-aware timestamps cannot share calendar grouping; "
            "use a consistent source or separate the analyses."
        )

    edges = {}
    transform_info = {}
    for role, settings in bin_specs.items():
        bounds = limits[role]
        if not bounds[2]:
            edges[role] = _edges([], settings)
        elif settings["method"] == "quantile":
            values = [item[1] for item in samples[role]]
            edges[role] = _edges(values, settings)
            # Quantile ranks may be sampled, but counts must never discard
            # corpus extrema merely because the sample missed them.
            if edges[role]:
                edges[role][0] = min(edges[role][0], bounds[0])
                edges[role][-1] = max(edges[role][-1], bounds[1])
            if bounds[2] > MAX_SAMPLE:
                warnings.add(
                    f"{role}: quantile boundaries estimated from a deterministic "
                    f"{MAX_SAMPLE:,}-observation sample of {bounds[2]:,} observed values."
                )
        else:
            # Equal-width/fixed-width need only exact extrema, not raw rows.
            edges[role] = _edges([bounds[0], bounds[1]], settings)
        transform_info[role] = {"edges": edges[role], "excluded_range": 0}

    def category(role: str, value: Any) -> str | None:
        if value is None:
            return None
        item = roles[role]
        transform = item.get("transform")
        if transform and transform["kind"] == "numeric_bins":
            index = _bin(float(value), edges[role])
            if index is None:
                transform_info[role]["excluded_range"] += 1
                return None
            return _bin_labels(edges[role])[index]
        if transform and transform["kind"] == "calendar":
            return calendar_period(value, transform["unit"])
        return str(value)

    result = {
        "kind": kind, "warnings": sorted(warnings),
        "coverage": {"total": coverage["total"], "complete": coverage["complete"],
                     "observed": observed},
        "names": names, "transformations": transform_info,
    }
    if kind == "scatter":
        heap: list[tuple[int, Any]] = []
        eligible = 0
        for identity, values, _ in _parent_rows(project, descriptors, spec):
            x, y = values[:2]
            color = category("color", values[position["color"]]) if "color" in roles else None
            if x is None or y is None or ("color" in roles and color is None):
                continue
            eligible += 1
            _reservoir(heap, spec["max_points"], _rank(identity),
                       (identity, float(x), float(y), color))
        result.update(
            points=[{"id": id_, "x": x, "y": y, "color": color}
                    for _, (id_, x, y, color) in sorted(heap, reverse=True)],
            eligible=eligible, sampled=len(heap),
            sampling="deterministic identity-hash sample; raw corpus not sent to browser",
        )
    elif kind == "histogram":
        hist_edges = edges["__histogram__"]
        bins = _bin_labels(hist_edges)
        counts: dict[str, list[int]] = {}
        excluded = missing_group = 0
        for _, values, _ in _parent_rows(project, descriptors, spec):
            value = values[0]
            if value is None:
                continue
            group = category("group", values[position["group"]]) if "group" in roles else "All observations"
            if group is None:
                missing_group += 1
                continue
            index = _bin(float(value), hist_edges)
            if index is None:
                excluded += 1
                continue
            if group not in counts:
                if len(counts) >= MAX_CATEGORIES:
                    raise ValueError("More than 100 histogram groups.")
                counts[group] = [0] * len(bins)
            counts[group][index] += 1
        result.update(edges=hist_edges, bins=bins, groups=sorted(counts),
                      counts=counts, excluded_range=excluded,
                      missing_group=missing_group,
                      denominators={g: sum(counts[g]) for g in counts})
    elif kind in ("summary_bar", "value_heatmap"):
        # Sums and counts are exact at every corpus size. Cell medians use
        # bounded stable samples only if more than 5,000 observations contribute.
        sums: dict[tuple[str, str | None], float] = defaultdict(float)
        counts: dict[tuple[str, str | None], int] = defaultdict(int)
        medians: dict[tuple[str, str | None], list[tuple[int, float]]] = defaultdict(list)
        for identity, values, _ in _parent_rows(project, descriptors, spec):
            if values[0] is None:
                continue
            row = category("group", values[position["group"]]) if kind == "summary_bar" else category("rows", values[position["rows"]])
            col = None if kind == "summary_bar" else category("columns", values[position["columns"]])
            if row is None or (kind == "value_heatmap" and col is None):
                continue
            key = row, col
            if key not in counts and len(counts) >= 2500:
                raise ValueError("Numerical summary has more than 2,500 cells.")
            counts[key] += 1
            sums[key] += float(values[0])
            if spec["statistic"] == "median":
                _reservoir(medians[key], MAX_SAMPLE, _rank((identity, row, col)), float(values[0]))
        if spec["statistic"] == "median" and any(n > MAX_SAMPLE for n in counts.values()):
            warnings.add("Median uses a deterministic sample of at most 5,000 values per group where needed.")
        def value(key: tuple[str, str | None]) -> float | None:
            if not counts.get(key):
                return None
            if spec["statistic"] == "mean":
                return sums[key] / counts[key]
            return median(item[1] for item in medians[key])
        row_labels = sorted({key[0] for key in counts})
        if len(row_labels) > MAX_CATEGORIES:
            raise ValueError("More than 100 summary categories.")
        if kind == "summary_bar":
            result.update(categories=row_labels,
                          values=[value((row, None)) for row in row_labels],
                          denominators=[counts[(row, None)] for row in row_labels],
                          statistic=spec["statistic"])
        else:
            column_labels = sorted({key[1] for key in counts if key[1] is not None})
            if len(column_labels) > MAX_CATEGORIES:
                raise ValueError("More than 100 summary columns.")
            result.update(rows=row_labels, columns=column_labels,
                          values=[[value((row, col)) for col in column_labels]
                                  for row in row_labels],
                          samples=[[counts.get((row, col), 0) for col in column_labels]
                                   for row in row_labels],
                          statistic=spec["statistic"])
    else:
        size = len(descriptors)
        # Stable online covariance avoids subtracting two very large sums.
        n = [[0] * size for _ in range(size)]
        means_x = [[0.0] * size for _ in range(size)]
        means_y = [[0.0] * size for _ in range(size)]
        ss_x = [[0.0] * size for _ in range(size)]
        ss_y = [[0.0] * size for _ in range(size)]
        cross = [[0.0] * size for _ in range(size)]
        for _, values, _ in _parent_rows(project, descriptors, spec):
            for i, x in enumerate(values):
                if x is None:
                    continue
                for j in range(i, size):
                    y = values[j]
                    if y is None:
                        continue
                    x, y = float(x), float(y)
                    k = n[i][j] + 1
                    dx = x - means_x[i][j]
                    dy = y - means_y[i][j]
                    means_x[i][j] += dx / k
                    means_y[i][j] += dy / k
                    ss_x[i][j] += dx * (x - means_x[i][j])
                    ss_y[i][j] += dy * (y - means_y[i][j])
                    cross[i][j] += dx * (y - means_y[i][j])
                    n[i][j] = k
        matrix = [[None] * size for _ in range(size)]
        pair_counts = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(i, size):
                pair_counts[i][j] = pair_counts[j][i] = n[i][j]
                if n[i][j] >= 2 and ss_x[i][j] > 0 and ss_y[i][j] > 0:
                    r = max(-1., min(1., cross[i][j] / math.sqrt(ss_x[i][j] * ss_y[i][j])))
                    matrix[i][j] = matrix[j][i] = r
        result.update(values=matrix, samples=pair_counts, measure="correlation")
    result["warnings"] = sorted(warnings)
    return result
