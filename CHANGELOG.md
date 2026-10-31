# Changelog

Notable user-facing changes to GeCo are recorded here. GeCo follows pre-1.0 versioning: minor versions may include changes to public APIs or project schemas.

## [Unreleased]

_No changes recorded yet._

## [0.9.0] - release pending

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
