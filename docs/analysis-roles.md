# Visualization-first analysis roles (schema v2)

The Analyses editor now starts with a chart, followed by its actual executable roles:

| Visualization | Roles | Current computation |
|---|---|---|
| Bar | Values (one or more binary Codes); optional Group (categorical metadata) | Positive counts and observed denominators |
| Code-value distribution | Values (one or more binary Codes) | Frequencies of observed 0/1 values, or explicit parent-level Code reductions |
| Association heatmap | Indicators (two or more binary Codes and/or categorical metadata) | Symmetric co-occurrence, Jaccard or pairwise-complete Pearson correlation |

This is **not** a generic pivot chart. There are no arbitrary X/Y axes, general
histograms, numerical aggregation heatmaps, or time-period grouping in Phase 2.
The editor exposes Scale fitted positions and metadata numeric/temporal source
interpretations but rejects unsupported chart assignments before saving or
executing. Code probabilities are likewise not silently thresholded.
The existing metadata categorical interpretation is explicit; choosing temporal
does not automatically turn a date into a category.

Schema v2 recipes are executable, versioned, and role-scoped:

```json
{
  "schema_version": 2,
  "kind": "bar",
  "measure": "count",
  "roles": {
    "values": [
      {"variable": {"type": "code", "code_id": 2, "source": "labels", "value": "binary"}}
    ],
    "group": {
      "variable": {"type": "metadata", "field": "category", "value": "categorical"}
    }
  }
}
```

Every role has an independent variable descriptor. There is no implicit
transformation. A selection key is the pair of role name and stable variable
identity; changing a different role cannot attach a classifier ID to the wrong
Code. The legacy SQL chart engine remains authoritative for ordinary classifier
and human Code sources and their parent-level aggregation. Committee-based
binary analyses use the Phase 1 typed resolver in batches of at most 2048
atomic IDs, reducing results server-side to compact chart aggregates. These
committee recipes currently support unaggregated bar and Code distribution;
Code-only association matrices also work. Parent-level committee aggregation
and committee/metadata association are deferred. Analysis never trains
models or changes research evidence.

A variable can be selected in distinct roles and retain independent UI
settings, but the existing chart computations reject using the same variable
twice in one executable chart. Duplicate roles/representations in numerical
visualizations are a Phase 3 concern.

Old `schema_version: 1` definitions remain unchanged in storage, and their
execution retains all original missingness, source precedence, grouping and
heatmap statistics. Editing one preserves schema v1 and its identity; a user
cannot upgrade an old recipe to an unsupported v2 representation accidentally.
New analyses are stored as v2. Both remain live definitions: chart results are
recomputed from current data and retained model state on preview/open.

Source/value representation is independent of chart transformations:
`code.source` controls human/prediction/hybrid provenance;
`code.value` distinguishes binary from probability;
`metadata.value` distinguishes categorical/numeric/temporal; and a Scale
always has `value: "position"`. The role contract currently rejects role
transformation keys rather than pretending to execute unsupported bins.
Calendar period controls and explicit timezone selection should only be added
when calendar-period computation exists in Phase 3.

## Validation and performance

`geometric_coder.analyses.validate_analysis` dispatches v2 recipes to
`analysis_roles.validate_role_analysis`, which enforces role membership,
variable type, measure, uniqueness and supported computation. v1 dispatch
stays unchanged. `compute_analysis` uses the corresponding executable path.
No full-corpus values or texts go to Dash/Plotly: the browser only receives
the reduced chart result. The server-side committee path pages IDs and uses
the bounded typed resolver; the legacy engine remains SQL-aggregated.

Regression tests: `tests/test_saved_analyses.py`,
`tests/test_analysis_roles.py`, and browser interaction tests.

## Save-time source readiness

Before saving a new v2 chart (or updating one), GeCo also runs the existing
read-only chart feasibility checks whenever metadata expansion or parent-level
Code-value binning can make a valid role specification unrenderable. This
rejects grouped bars with over 100 categories, categorical heatmaps exceeding
24 expanded indicators or 24 categories per metadata field, inconsistent
parent-level metadata, and aggregated distributions exceeding 100 bins. The
chart engine returns only compact aggregates, which are discarded by this
preflight; no models are trained. Legacy v1 saves and the live recomputation
of existing recipes are unchanged.

New schema-v2 analyses are checked for a retained trained predictor **at save time**. A classifier must have a saved fit; fixed committees must have fitted members and learned committees must have their own saved fit. Selecting an unfitted source produces an actionable validation error without creating a saved analysis, training a predictor, or mutating evidence. This check does not alter v1 validation or execution. A previously saved v2 definition remains live: model status may change after saving, at which point ordinary execution reports unavailable or stale sources according to existing behavior.

The browser regression scenarios reuse a single project, fitted classifier, committee and app. CI runs the browser tests once in a Playwright-enabled step, while redundant same-branch push/PR test jobs are canceled when a newer head arrives.
