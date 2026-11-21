# Continuous and mixed exploratory Analyses (schema v2)

Phase 3 adds five *explicitly numerical* visualizations, alongside the unchanged
schema-v1 and Phase-2 categorical chart paths.

| Chart | Roles | Statistic |
|---|---|---|
| `bar_chart` | One or more Values, optional Group and Color | Count for observed categorical/binary frequencies or observed numeric values; Sum, Mean, Median for numerical/Code 0/1 values. Group/Color can be categorical, explicitly binned numeric, or calendar period. |
| `scatter` | X numeric, Y numeric, optional categorical Color | Paired observed positions; deterministic bounded sample |
| `histogram` | one numeric Values, optional categorical Group | Exact observed frequencies within explicit bins |
| `summary_bar` | one numeric Values, categorical Group | Mean or median per category, observed-only denominator |
| `correlation_heatmap` | 2–12 numeric Values | Pairwise-complete Pearson correlations and sample counts |
| `value_heatmap` | one numeric Values, categorical Rows and Columns | Grouped means/medians and cell-level sample counts |

Code binaries may be numerical 0/1 measurements; Code probabilities require an
explicit trained classifier or committee source (no probability is invented for a
human-only Code). Scales expose predicted scalar positions only and never human
numeric judgments. Metadata must be explicitly interpreted as categorical,
numeric, or temporal. All source references are checked using the existing typed
resolver and retain stable atomic observation IDs. Missing/untrained/stale
prediction coverage remains missing and generates warnings; there is no model
training or evidence mutation in the Analyses workspace.

## Role transformations

Numerical grouping roles require an explicit `numeric_bins` transformation.
Methods: equal-width with bin count (1–100), quantile with bin count,
fixed-width with positive finite width and origin, or explicit strictly increasing
finite edges. Histogram frequencies use the same methods with `histogram_bins`.
Intervals are left-closed/right-open except the final interval, which includes
the right endpoint. Values outside explicit boundaries are excluded and reported.
Tied quantile boundaries are collapsed rather than creating empty zero-width
intervals. The median and quantile-edge computations are exact until the
bounded sample thresholds described below; large-sample quantiles report
approximation warnings.

Temporal grouping roles require `{"kind":"calendar","unit":"day|week|month|quarter|year"}`.
Dates retain the recorded date; naive timestamps retain their recorded wall-clock
calendar date; explicit-offset timestamps are converted to UTC for calendar
bucketing. Mixed naive/offset-aware timestamps in one grouping are rejected.
Weeks start Monday and use ISO week years, so late December may belong to the
next year's first ISO week. Month groups are `YYYY-MM`, not January pooled
across years. Quarter groups are `YYYY-Qn`; period labels sort chronologically.
The source metadata itself is never changed.

Role transformations belong to each role entry, separate from its underlying
source/value descriptor. A saved histogram with explicit edges may look like:

```json
{
  "schema_version": 2,
  "kind": "histogram",
  "roles": {
    "values": {"variable": {"type":"metadata", "field":"rating", "value":"numeric",
                            "numeric_strings":false}},
    "group": {"variable": {"type":"metadata", "field":"date", "value":"temporal",
                           "numeric_strings":false},
              "transform":{"kind":"calendar","unit":"month"}}
  },
  "binning": {"kind":"histogram_bins","method":"equal_width","bins":20}
}
```

## Observation grain and numerical semantics

Without `aggregation`, each atomic unit contributes once. Parent-grain analyses
first group by the declared prefix of key columns; binary Codes then use observed
positive count, any, threshold, proportion, or log-count according to the
explicit method, while numeric Codes/Scale/metadata use the selected `numeric_method`: mean, median, sum, minimum, maximum, observed count, population standard deviation, percentile, range, or proportion within bounds. Percentile uses a specified value from 0 to 100. Proportion within bounds accepts optional inclusive/exclusive lower and upper limits, requires at least one limit, and divides the number meeting the limits by the observed nonmissing count. A parent with zero observed values remains
missing; categorical/temporal values must be constant within a parent, including missingness (a parent cannot mix missing and nonmissing categories).
Correlation always uses the pairwise-complete set for *each variable pair*,
with separate `samples` counts. Constant or insufficiently observed variables
produce undefined (`null`) correlations; no significance tests are performed.

Categorical group keys preserve source types: numerical 1, text "1", and boolean true are distinct, with unique labels when displayed. Coverage reports total units at the selected grain, fully complete rows, and
per-source observed counts. Numerical results also expose a positional `sources`
array with each variable descriptor, the resolved predictor/Scale fit identity,
fit status and source provenance, and an atomic missing-reason tally. Missing
reasons are collected from the first bounded corpus scan only, not doubled by
the chart's subsequent aggregation pass. In parent-grain charts, these source
missing-reason counts refer to atomic observations, whereas the top-level
coverage denominator refers to parents; they must not be interchanged. For scatterplots, `eligible` and `sampled` report
the complete-pair population and plotted sample. Histogram outputs provide
per-group denominators, values excluded from bin ranges, and missing group counts.
Numerical summaries provide each group's actually observed denominator. No
chart silently substitutes zeros for missing values.

## Performance, approximation, and limits

Raw observation-level points are never transferred to Dash except for at most
5,000 stable identity-hash scatter points. Histograms use two bounded scans:
one for extrema and deterministic sampled quantile boundaries, then one for
full-population exact bin counts. Summaries stream mean/count aggregates;
medians use exact group values up to 5,000, then bounded deterministic
5,000-observation samples with visible warnings. Parent-level medians and percentiles follow the same threshold, with deterministic 5,000-value samples and explicit warnings above that limit. Other numerical parent reductions use bounded exact accumulators. Analysis status/provenance is cached only within a single
paged resolver scan (never globally). Parent-group states are bounded to
100,000 distinct groups, numerical summary cells to 2,500, and category
cardinalities to 100 (including scatterplot Color categories, checked over
all eligible observations rather than only the plotted sample). Invalid or
infeasible new v2 recipes are rejected before save. Saved recipes are recomputed against current project data each time.

Existing v1 saved recipes, original Code count charts, and legacy categorical
association matrices still execute on their original paths. This is an
exploratory visualization feature, not a measurement-validity claim or
statistical inference workflow.


## Unified Bar chart editor and saved-analysis compatibility

The researcher-facing new-chart picker has a single **Bar chart** choice. Values,
optional Group, optional Color, and Statistic are separate roles/controls. A
binary Code can be displayed as two observed-value categories (Absent/Present)
with Count, or as a Present count with Sum. Numerical metadata and fitted Scale
positions can be summarized with Sum, Mean or Median. Categorical metadata can
be counted without creating a Code. A second grouping produces named colored
series, not a separate visualization type.

The new `bar_chart` recipe is schema-v2 and uses bounded, observation-ID-aligned
aggregation. The existing saved `bar`, `distribution`, and `summary_bar` v1/v2
recipes remain executable without migrations or changes to their calculations.
Existing-format saved charts retain their original controls on reopening rather
than being silently converted to the new statistical contract. This limitation
is visible in the editor and is subject to hands-on researcher acceptance.

When switching a selected saved analysis during editing, a pristine draft closes
automatically. Modified drafts require Save and switch, Discard and switch, or
Stay here; Save always writes to the original editing ID. Numerical charts show
a compact complete/total coverage count with detailed sources, fit identity,
missing reasons, and denominators under a disclosure. This does not constitute
inferential validation or automatic model training.
