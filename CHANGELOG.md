# Changelog

Notable user-facing changes to GeCo are recorded here. GeCo follows pre-1.0 versioning: minor versions may include changes to public APIs or project schemas.

## [Unreleased]

_No changes recorded yet._

## [0.11.0] - release pending

### Added

- Project-wide **Analyses** tab for creating, editing, duplicating, deleting, and revisiting saved exploratory analyses (PR #44).
- Bar plots, histograms, and categorical/code heatmaps with positive co-occurrence counts, Jaccard similarity, and exploratory Pearson correlations.
- Per-code selection of human judgments, fitted classifier predictions, or human-first prediction fallback.
- Optional parent-key aggregation of codes, including any-positive, positive count, thresholds, observed-positive proportion, and log-transformed counts; metadata conflicts within aggregate groups are reported.
- Preview Changes for unsaved analysis configurations, with meaningful histogram axes and code-source coverage.

### Changed

- Project schema advances to version 18 to persist analysis configurations; existing version 17 projects migrate forward on open.
- Available code variables refresh without restarting the interface, newly created analyses become selected automatically, and analysis-editor actions avoid persistent success messages.

### Notes

- Analyses are descriptive and exploratory, not inferential population estimates. Grouped metadata must be constant within each parent grouping.
- Back up workspaces before upgrading: previous GeCo versions may not reopen schema 18 projects.

## [0.10.0] - release pending

### Added

- Project-wide, automatically saved semantic-search query strings in Explore, with Search/Enter submission and Load/delete controls (PR #26).
- Lazy similarity scoring for the active geometry, with reusable per-query/per-geometry results persisted across sessions (PR #26).
- Cache validation against geometry and observation alignment, including handling for stale or unsupported results (PR #26).

### Changed

- Semantic searches use query-string identity rather than manually named geometry-specific entries; draft text and the active submitted query have separate states (PR #26).
- Project schema advances to version 17, with forward migrations from schema 15 through the interim schema 16 search storage. The Python `semantic_search(query, geometry_names=...)` API still supports explicit multi-geometry searches (PR #26).

### Notes

- Switching geometries can require a first computation for each geometry; switching views within the same geometry reuses the existing result.
- Back up research workspaces before upgrading. Forward migrations do not guarantee that older GeCo versions can reopen upgraded workspaces.

## [0.9.0] - 2026-10-31

### Added
- **Manage Representations** for creating, inspecting, renaming, and managing numerical representations and their source relationships in TeAL-linked workspaces.
- Reusable TF-IDF, normalization, SVD, and UMAP transformation steps, including multidimensional analytic representations and one-/two-dimensional visualizations.
- Ability to make an existing display-only representation available for analysis without refitting it.
- Adjustable, deterministic jitter for one-dimensional Explore plots.
- New-text capability reporting based on the availability of a replayable transformation pipeline.

### Changed
- Representation handling now uses one numerical registry for analysis and visualization, with compatibility handling for existing projects.
- Improved navigation and representation selection in Explore.

### Maintenance
- Added automated GitHub Releases with version verification, tests, distribution builds, and clean-install checks.
