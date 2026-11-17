# Typed analytical variables (internal Phase 1 contract)

The `geometric_coder.analysis_variables` module resolves source values for
future Analyses chart roles without changing the saved Analysis v1 engine.

## Source descriptor

`validate_variable(project, reference)` returns a normalized descriptor:

- Code: `{"type":"code","code_id":12,"source":"labels","value":"binary"}`.
  The source is `labels`, `predictions`, or `hybrid`; the representation
  is `binary` or `probability`. Predicted and hybrid sources require
  `predictor_id`, a code-owned classifier *specification* ID. The retained
  classifier fit is resolved and reported at each call. A committee can
  instead be selected by `"predictor_kind":"committee"` and its stable
  `predictor_id` (committee ID); the default kind is `"classifier"` for
  compatibility. Fixed committees use the selected members' retained fit
  probabilities with the existing aggregation rule (and a 0.5 threshold for
  binary predictions). Learned committees read their retained committee-fit
  predictions. Both return fit/member provenance, current/stale status,
  and explicit missing coverage; incomplete member coverage is never replaced
  by a partial aggregation. Neither committee type is trained or prepared
  merely by analytical resolution. Human-only labels cannot be described as
  model probabilities.
- Scale: `{"type":"scale","scale_id":16,"value":"position"}`.
  Positions are continuous model predictions from the Scale's retained
  Davidson fit, never direct human numerical judgments. The selected
  `scale_id`, current definition version, eligibility revision, retained
  fit ID and freshness status are returned as provenance.
- Metadata: `{"type":"metadata","field":"year","value":"numeric"}`, or
  `"categorical"` (default). Numeric conversion accepts original finite JSON
  numbers, not strings, identifiers, booleans or nonfinite values. An
  explicit `"numeric_strings":true` opts into parsing numeric text.

Source selection does **not** specify chart roles or operations. A future
schema-v2 spec can assign independent references to roles, for example:

```json
{
  "schema_version": 2,
  "kind": "scatter",
  "roles": {
    "x": {"variable": {"type": "scale", "scale_id": 16, "value": "position"}},
    "y": {"variable": {"type": "metadata", "field": "year", "value": "numeric"}},
    "color": {
      "variable": {"type": "metadata", "field": "year", "value": "numeric"},
      "transform": {"kind": "bin", "method": "fixed_width", "width": 10}
    }
  }
}
```

**Illustrative only.** Schema v2 is not currently accepted or persisted by
`save_analysis`; its kinds, roles, transformations, bins and validations
belong to Phases 2 and 3. Role transforms are distinct from source descriptors.

## Bounded resolution and alignment

```python
resolve_analytical_variables(
    project, references, observation_ids, allow_stale_scales=False
)
```

The caller supplies up to 20 descriptors and up to 2,048 **atomic
observation IDs** per request. Observation IDs are not unit IDs or corpus row
positions. Unknown/non-atomic IDs fail explicitly. An empty ID selection
returns empty aligned columns. Input order and duplicates are retained,
including for TeAL-backed geometry scoring. The caller is responsible for
paging, SQL-side aggregations, or deterministic chart sampling before sending
results to the browser. The resolver does not load all units, texts, or
prediction rows; its SQL selection is batched and only requested metadata
is decoded from selected unit records.

Return shape:

```python
{
    "observation_ids": [stable_atomic_id, ...],
    "row_count": int,
    "columns": [
        {
            "descriptor": {...},
            "name": str,
            "kind": "binary" | "numeric" | "categorical",
            "values": [float | str | bool | int | None, ...],
            "missing_reasons": [str | None, ...],
            "coverage": {
                "observed": int, "missing": int,
                "by_reason": {reason: count, ...}
            },
            "provenance": {...},
            "warnings": [str, ...]
        }, ...
    ],
    "warnings": [str, ...]
}
```

Columns are positional; identical source descriptors may appear in multiple
roles with different transforms. Missing values remain `None`, never 0.
An annotation of `unsure` is explicitly missing and blocks the prediction
fallback in hybrid mode; an unreviewed observation may use the selected fit.
Hybrid probability uses 1.0/0.0 for reviewed Present/Absent values and the
retained predictor probability only for unreviewed values. Classifier
predictions with no trained fit remain missing, with fit ID/status reported.

Scale scoring filters the requested IDs through the **current** eligibility
policy. Untrained, unsupported and stale fits yield unavailable values with
reasons, unless `allow_stale_scales=True` explicitly permits a retained
stale fit. Changed geometry still blocks stale scoring in the model API.
No substitute fit is selected if the chosen configuration has no fit.
All resolutions are read-only and never train classifiers/scalers, persist
predictions, or modify coding evidence.

## Existing saved analyses

The current `schema_version: 1` format, SQL aggregation paths, distributions,
bar/heatmaps, missingness, parent-level reductions, and live recomputation
remain unchanged. Phase 1 deliberately does not route existing charts through
the new resolver, which is an additive boundary for later charts.
