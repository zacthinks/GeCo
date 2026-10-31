# GeCo architecture

This technical companion to the [README](../README.md) is for contributors and developers integrating with GeCo. GeCo is primarily a **researcher-facing interface** for exploring text, developing codes, and reviewing coding proposals. Researchers should not need to understand the internals to use it.

This describes GeCo 0.9.0 and project schema 15. Internal implementation modules are not a stable plugin SDK.

## Project model and persistence

`GeometricCoder` is the main Python project object. GeCo's local Dash interface uses the same persistent project state. SQLite stores structured research records; numerical arrays and fitted models are stored in registered project-local sidecar artifacts. Projects can have local fitted representations or use external numerical providers, particularly [Text Analysis Lab (TeAL)](https://github.com/zacthinks/text-analysis-lab).

Opening a project verifies its schema version. Supported forward migrations run on open, including schema 13 to 14 and 14 to 15; unsupported versions fail explicitly rather than being silently interpreted.

## Observations, keys, and coding

Imported rows are **atomic observations** with source keys and canonical corpus order. GeCo also supports contiguous **spans** and researcher-authored **teaching examples**. The internal `unit_id` identifies atomic rows, while `observation_id` identifies both atomic and derived observations. Integrations should preserve original source keys for alignment.

Human judgments distinguish Present, Absent, Unsure, and **unreviewed** (no judgment). Coding changes are recorded as events. Model predictions and proposed assignments stay separate from human judgments until a researcher explicitly commits decisions.

## Representations and views

GeCo maintains a **unified numerical representation registry**. Representations have stable identities and may have parent/source ancestry. Analytic and visualization capabilities are separate:

- Analytic representations provide features for distances, classifiers, and supported searches.
- One- and two-dimensional representations can be visualized. A view-only representation can be enabled for analysis without refitting; a two-dimensional analytic representation may display itself.
- New-text queries and derived-observation transformations require compatible frozen replayable pipelines. Enabling analysis does not by itself grant new-text capability.

In TeAL-linked workspaces, Manage Representations supports transformations such as TF-IDF, normalization, SVD, and UMAP. TeAL owns the immutable numerical artifacts and transformation lineage; GeCo manages local names, registrations, analytic/display eligibility, and research interactions. Removing a GeCo registration does not delete the TeAL artifact.

The project-level numerical access boundaries include `geometry_matrix(...)` and `view_coordinates(...)`. Jitter for one-dimensional displays affects only rendering, not analytic coordinates.

## Explore, Develop, and Apply & Review

**Explore** manages navigation, text and context reading, sessions, filters, search state, and linked memos. A document counts as visited when it becomes focal, not merely because it appeared on a map.

**Develop** connects researcher-defined codes to their own classifiers and committees. Training maintains current fitted model state and prediction caches while preserving the separation between human judgments and machine suggestions.

**Apply & Review** stores machine-generated proposals in drafts. Only committed researcher decisions become human assignment events; predictions are never silently accepted.

## Integration boundaries

GeCo can operate on its own or use externally backed numerical data. External providers align their results with the project's original source keys; GeCo owns codes, observations, assignments, sessions, and other mutable exploratory state. The UI does not need to interpret an external provider's private storage layout.

Existing integration entry points include project creation/opening, representation access, and selected exports such as `export_codes(...)`. For current signatures, consult the [source package](../src/geometric_coder/) and tests rather than accessing GeCo's SQLite tables or private sidecar files.

## Future extensions

GeCo is intended to stay a focused **research interface** rather than accumulate every specialized analytical feature. A future plugin system could allow trusted, separately installed Python libraries to contribute actions, small configuration workflows, keyed derived results, or exports. For example, an interview-analysis library such as LIPE might supply protocol-specific analyses without requiring GeCo's core application to implement those algorithms.

The intended division of responsibility is GeCo for the research interface and mutable coding state, an appropriate analytic engine such as TeAL for immutable derived artifacts, and domain-specific extensions for specialist methods or outputs.

**A general plugin API does not exist yet.** Registration, compatibility, permissions, execution, and UI integration remain design work. Installed Python plugins would run trusted local code, not sandboxed scripts.
