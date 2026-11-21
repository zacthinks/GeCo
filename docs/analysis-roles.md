# Visualization-first Analyses roles (schema v2)

**Current for proposed GeCo v0.14.0.** Analyses is a descriptive, chart-first
workspace. Researchers select a chart, then assign typed variables to its
independently configured roles. A role is not interchangeable with an
arbitrary pivot-table axis: executable combinations are validated before save.

| Visualization | Main roles | Computation |
|---|---|---|
| Unified Bar (`bar_chart`) | One or more Values, optional Group and Color | Observed Count, or Sum/Mean/Median where the chosen Value supports it |
| Code-value distribution / categorical comparisons | Binary Code Values, optional grouping | Observed frequencies and denominators |
| Association heatmap | Categorical/Code indicators | Co-occurrence, Jaccard, or pairwise-complete correlation where supported |
| Scatter (`scatter`) | Numeric X and Y, optional categorical Color | Bounded deterministic sample of paired observed values |
| Histogram (`histogram`) | One numeric Value, optional Group | Full observed frequency counts within explicit bins |
| Numerical summary bar (`summary_bar`) | One numeric Value and categorical Group | Group mean or median and observed denominator |
| Correlation heatmap (`correlation_heatmap`) | Two to twelve numeric Values | Pairwise-complete Pearson correlations and pair-specific sample counts |
| Grouped-value heatmap (`value_heatmap`) | Numeric Value and categorical Rows and Columns | Grouped means/medians and cell counts |

Variable *source* and its *interpretation* are independently specified:
a Code uses human labels, fitted predictions or a documented hybrid source,
with a binary or supported probability representation; a Scale supplies
**fitted signed model positions**, never direct human numerical labels; metadata
is explicitly categorical, numeric or temporal. Selecting a numerical Code
probability requires an actual trained classifier/committee. A variable's
stable source ID and predictor or Scale-fit provenance are retained with the
analysis; stale, unavailable, or missing predictions are not fabricated.

## Role-specific transformations

Grouping roles can explicitly bin numeric metadata (equal-width, quantiles,
fixed-width with anchor, or declared bin edges) and can group temporal metadata
by day, ISO week, month, quarter or year. Histogram Values use their own
explicit binning. Transformations are chart-local: they do not rewrite source
metadata, construct definitions or saved evidence. Numeric parent aggregation
has explicit methods and denominators; inconsistent parent-level categorical
metadata are rejected.

For exact supported role shapes, numerical semantics, cell/cardinality caps,
missingness treatment, deterministic samples and approximation warnings,
see [continuous and mixed Analyses](analysis-continuous.md) and
[typed analytical variables](analysis-variables.md).

## Saving, recomputation and compatibility

Saved role-based recipes use `schema_version: 2` and name stable source
identities and independently configured role entries. The legacy
`schema_version: 1` format remains supported and continues to use its
historical calculation semantics when reopened or edited; no implicit
conversion or invented new chart role is applied to a saved legacy recipe.
Both schemas save *definitions*, not immutable numeric results: opening and
previewing recomputes from current eligible source data and retained model
state without training models or committing judgments.

Save-time checks validate chart-role combinations, source availability,
retained classifier/committee fit readiness where needed, category limits,
parent aggregation and selected data feasibility. A previously saved recipe
may later show unavailable or stale sources if model state changes.
No computation silently thresholds probabilities or substitutes zeros for
unobserved and missing values. Source-level missingness, complete-case
counts, and fitted-model provenance are exposed as available. Correlations
are exploratory descriptions, not significance tests.

Example schema-v2 source reference:

```json
{
  "schema_version": 2,
  "kind": "histogram",
  "roles": {
    "values": {
      "variable": {
        "type": "metadata",
        "field": "rating",
        "value": "numeric"
      }
    }
  },
  "binning": {
    "kind": "histogram_bins",
    "method": "equal_width",
    "bins": 20
  }
}
```

The editor and engine use bounded stable-identity resolution, with per-page
metadata decoding reused when several selected roles need the same row's
metadata. Plotly receives reduced chart results, not the complete raw corpus;
scatterplots show a bounded, reproducibly sampled set of source identities.
Performance evidence is limited to separately benchmarked backend operations,
not a guarantee of instantaneous browser rendering on all corpus sizes.

Relevant tests include `tests/test_analysis_roles.py`,
`tests/test_analysis_continuous.py`,
`tests/test_analysis_variables.py`, `tests/test_analysis_bars.py`,
`tests/test_analysis_acceptance.py`, and the browser interaction tests.
