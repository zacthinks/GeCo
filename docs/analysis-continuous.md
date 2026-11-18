# Continuous and mixed exploratory Analyses (schema v2)

Phase 3 adds five *explicitly numerical* visualizations, alongside the unchanged
schema-v1 and Phase-2 categorical chart paths.

| Chart | Roles | Statistic |
|---|---|---|
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
explicit method, while numeric Codes/Scale/metadata are reduced to a mean or
median using `numeric_method`. A parent with zero observed values remains
missing; categorical/temporal values must be constant within a parent.
Correlation always uses the pairwise-complete set for *each variable pair*,
with separate `samples` counts. Constant or insufficiently observed variables
produce undefined (`null`) correlations; no significance tests are performed.

Coverage reports total units at the selected grain, fully complete rows, and
per-source observed counts. For scatterplots, `eligible` and `sampled` report
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
5,000-observation samples with visible warnings. Parent-level medians follow
the same threshold. Analysis status/provenance is cached only within a single
paged resolver scan (never globally). Parent-group states are bounded to
100,000 distinct groups, numerical summary cells to 2,500, and category
cardinalities to 100 (including scatterplot Color categories, checked over
all eligible observations rather than only the plotted sample). Invalid or
infeasible new v2 recipes are rejected before save. Saved recipes are recomputed against current project data each time.

Existing v1 saved recipes, original Code count charts, and legacy categorical
association matrices still execute on their original paths. This is an
exploratory visualization feature, not a measurement-validity claim or
statistical inference workflow.
