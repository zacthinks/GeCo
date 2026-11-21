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
- Metadata: `{"type":"metadata","field":"year","value":"numeric"}`,
  `"categorical"` (default), or `"temporal"`. Numeric conversion accepts
  original finite JSON numbers, not strings, identifiers, booleans or nonfinite
  values. An explicit `"numeric_strings":true` opts into parsing numeric text,
  but cannot be combined with temporal interpretation. Temporal interpretation
  is chosen explicitly; categorical metadata continue to preserve their original
  strings, including date-looking strings.

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

## Temporal metadata interpretation

A temporal descriptor is `{"type":"metadata","field":"date","value":"temporal"}`.
The resolver accepts **only unambiguous ISO-style calendar dates and datetimes**
from string-valued source metadata:

- Date: `YYYY-MM-DD`, validated as a Gregorian calendar date (including
  leap-year rules). It resolves to a JSON-safe tagged value:
  `{"kind":"date","iso":"2026-11-17"}`.
- Timestamp: `YYYY-MM-DDTHH:MM[:SS[.fraction]][Z|±HH:MM]`. An ASCII
  space or lowercase `t` is also accepted in place of `T` to accommodate
  GeCo's existing pandas `Timestamp` serialization. `fraction` is one to
  nine decimal digits and is preserved without truncating nanoseconds.
  Seconds omitted from input are canonicalized to `:00`. A UTC `Z`/ `z`
  suffix becomes `+00:00`. For example,
  `2026-11-17 08:30:15.123456789+05:30` yields:
  `{"kind":"datetime","iso":"2026-11-17T08:30:15.123456789+05:30","timezone":"offset"}`.
  A timestamp without a zone suffix yields `"timezone":"naive"`, preserving
  its wall-clock meaning without interpreting it as UTC or local machine time.

**Dates and instants are not interchangeable.** A date-only value never
becomes midnight UTC, and naive timestamps are never silently localized. Offset
timestamps retain the supplied numeric offset rather than being normalized to
UTC or assigned an invented named zone. Mixed date/timestamp and mixed
naive/offset-aware timestamp columns are **accepted** as explicitly tagged,
per-row values, with warnings to future chart consumers to choose a consistent
calendar/timezone operation before grouping or plotting. No calendar binning
or temporal grouping is implemented here.

`None` and blank strings yield `missing_metadata`. Impossible dates,
malformed timestamps, unsupported types, and arbitrary locale/word-based formats
yield `invalid_temporal`. Local day/month formats such as `01/02/2026`, and
RFC-3339 `-00:00` (which denotes an unknown offset), yield
`ambiguous_temporal`. They are returned as `None` with a counted missing
reason; **no locale interpretation or timezone guessing** occurs.

GeCo's existing import storage converts Python `date` and pandas
`Timestamp` values to text via `storage.database._json_value`;
the temporal resolver interprets the stored JSON strings and does not mutate
project metadata or change import serialization.

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
            "kind": "binary" | "numeric" | "categorical" | "temporal",
            "values": [float | str | bool | int | dict | None, ...],
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


## Final acceptance polish: binary metadata and opt-in Unsure fallback

Numerical chart roles accept binary Codes, model probabilities, fitted Scale positions,
numeric metadata, and explicit binary metadata (0/1 or true/false). The
`{"type":"metadata","field":"flag","value":"binary"}` descriptor maps observed
true/1 to 1.0 and false/0 to 0.0. It does not assign a numerical ordering
to arbitrary category labels; invalid binary observations are reported as
missing with `invalid_binary`. The existing `value:"numeric"` contract remains
strict about boolean metadata. Multi-category variables need an explicit,
chart-appropriate representation rather than implicit integer labels.

Code source `hybrid` retains conservative semantics: Present/Absent human
judgments override model values; Unsure remains missing; unreviewed units
may use the selected retained predictor. The new opt-in `hybrid_unsure`
uses the same human Present/Absent overrides but allows the selected
predictor (classifier or committee) to supply values for both Unsure and
unreviewed observations, without changing the human assignment events.
Missing/stale/untrained predictor coverage is still reported. Binary and
probability representations use the same source-choice semantics.

The editor presents the histogram grouping series as **Color**, but
existing `histogram.roles.group` recipe keys are preserved on disk.
Legacy saved definitions are not silently migrated.


## Unified Bar Count semantics

New `schema_version:2` / `kind:"bar_chart"` recipes use **Count** to
count observed, nonmissing Values in each explicitly selected Group/Color
partition. The Values' own categorical levels are **not** an implicit axis
or series split. Without a Group, a single Value generates a single
aggregate bar; without Color, it generates a single series. For example,
a binary human Code with 17 Present, 15 Absent, 2 Unsure and 6 unreviewed
judgments has Count = **32**, but Sum = **17**. To display Present/Absent
separately, explicitly assign that Code to Group or Color.

For Code Values, Count is not the total size of the input universe when
there are Unsure/unreviewed judgments: those remain missing, not zero or
negative. Existing saved legacy `bar` definitions keep their prior
positive-count meaning. The researcher-approved change applies to unified
`bar_chart` Count recipes without changing their persisted role schema.

## Parent Code aggregation selector

The editor offers a single inclusive binary reduction, **At least N positive
judgments**, with **Minimum positive count** defaulting to 1. In each parent,
the reduced value is 1 when the number of observed positives is greater than
or equal to N, otherwise 0. Parents with no observed binary Code judgments
remain missing. N=1 is exactly the historical **Any observed positive**
operation; saved recipes containing `method:"any"` remain executable and
reopen in the editor as the equivalent `method:"minimum", threshold:1`
selection. New edits use `minimum` without rewriting stored recipes merely
when opening them. Other numerical reductions (positive count, proportion,
log-count) retain distinct meaning; mean/median/sum of numerical sources uses
the separate **Aggregation function** control, shown only when a numerical
role is selected. Sum ignores missing observations and leaves an entirely
missing parent unobserved rather than treating it as zero. This parent-level
function is applied before the chart's own statistic (such as Mean across
parents), which remains an independent choice.
