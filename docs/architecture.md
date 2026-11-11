# GeCo architecture

This technical companion to the [README](../README.md) is for contributors and developers integrating with GeCo. GeCo is primarily a **researcher-facing interface** for exploring text, developing codes, and reviewing coding proposals. Researchers should not need to understand the internals to use it.

This describes GeCo 0.11.0 and project schema 18. Internal implementation modules are not a stable plugin SDK.

## Project model and persistence

`GeometricCoder` is the main Python project object. GeCo's local Dash interface uses the same persistent project state. SQLite stores structured research records; numerical arrays and fitted models are stored in registered project-local sidecar artifacts. Projects can have local fitted representations or use external numerical providers, particularly [Text Analysis Lab (TeAL)](https://github.com/zacthinks/text-analysis-lab).

Opening a project verifies its schema version. Supported forward migrations run on open, including schema 13 to 14, 14 to 15, 15 to 16, 16 to 17, and 17 to 18; unsupported versions fail explicitly rather than being silently interpreted.

## Observations, keys, and coding

Imported rows are **atomic observations** with source keys and canonical corpus order. GeCo also supports contiguous **spans** and researcher-authored **teaching examples**. The internal `unit_id` identifies atomic rows, while `observation_id` identifies both atomic and derived observations. Integrations should preserve original source keys for alignment.

Human judgments distinguish Present, Absent, Unsure, and **unreviewed** (no judgment). Coding changes are recorded as events. Model predictions and proposed assignments stay separate from human judgments until a researcher explicitly commits decisions.

## Geometries and Views

GeCo maintains a unified registry of **numerical representations of text**. A representation can serve as a **geometry**, a **view**, or both. A **geometry** is used to examine relationships among observations, including distances, neighborhoods, and patterns. A **view** is a lower-dimensional representation of a geometry, typically in one or two dimensions, used for visualization.

Geometries provide numerical features for distance-based navigation, classifiers, and supported searches. A two-dimensional geometry can directly serve as its own view; a visualization-only view can be enabled as a geometry without refitting the underlying numerical artifact. These roles are independent of new-text capability: transforming new queries or derived observations still requires a compatible frozen, replayable pipeline.

New-text capability is **geometry-specific**, not a project-wide all-or-nothing flag. A workspace may create a span or teaching example when at least one registered geometry supports transforming new text. Its vector artifacts are stored only for compatible geometries; unsupported geometries retain their existing atomic representations. Each classifier uses its own geometry's atomic observations plus only those derived observations for which that geometry has a vector. Evidence snapshots, prediction coverage, training, and freshness must obey the same eligibility set. A learned committee can use only the observations jointly represented by all member classifiers; an unavailable derived score is not synthesized. Derived observations remain visible as coding objects even if they cannot be plotted in a particular view.

In TeAL-linked workspaces, Manage Representations supports transformations such as TF-IDF, normalization, SVD, and UMAP. TeAL owns immutable numerical artifacts and their transformation lineage; GeCo manages local names, registrations, and use as geometries and views. Removing a GeCo registration does not delete the underlying TeAL artifact.

The project-level accessors `geometry_matrix(...)` and `view_coordinates(...)` remain compatibility boundaries. One-dimensional jitter changes displayed coordinates only, not analytic distances.

## Explore, Develop, and Apply & Review

**Explore** manages navigation, text and context reading, sessions, filters, search state, and linked memos. A document counts as visited when it becomes focal, not merely because it appeared on a map. Submitted semantic-search queries are saved project-wide, with similarity results cached per query and eligible geometry. Switching views reuses valid cached results; switching geometries computes scores only when needed. Saved queries can be loaded or deleted, and the Python API retains explicit multi-geometry searches.

**Develop** connects researcher-defined codes to their own classifiers and committees. Training maintains current fitted model state and prediction caches while preserving the separation between human judgments and machine suggestions.

Classifiers include linear and RBF-kernel SVMs. A fitted classifier stores its predicted labels, estimated positive-class probabilities when available, and its **native decision scores** independently. SVM probability estimation uses cross-validated sigmoid calibration when feasible; very small class counts use a monotone sigmoid fallback that is explicitly not a validated calibration estimate. An SVM with explicitly disabled probability estimation retains native margin scores only. Other families may expose native probabilities without separate calibration. The Develop prediction-geometry API and UI can select either probability estimates or native scores, rejecting unavailable score types rather than mixing incompatible scales.

GeCo's optional training-time CV tunes classifier hyperparameters using shared stratified folds, then refits the selected configuration on all eligible evidence. Continuous tuning parameters use bounded, reproducible logarithmic candidate searches: logistic L1/L2, linear SVM and Complement Naive Bayes use one-dimensional searches; RBF SVM jointly explores regularization and gamma; logistic elastic net jointly explores regularization and L1 ratio. A model's current saved configuration is included as an anchor. Small genuinely discrete search spaces (decision-tree choices and k-nearest-neighbor settings) remain exhaustively evaluated. These bounded searches improve granularity over the former fixed powers-of-ten grids but cannot guarantee a global optimum. Calibration folds fit only within the training portion of any outer hyperparameter-validation fold. Scores from adaptively sampled human labels are exploratory model outputs and should not be interpreted as demonstrated population-level calibration.

**Apply & Review** stores machine-generated proposals in drafts. Only committed researcher decisions become human assignment events; predictions are never silently accepted.

Apply & Review now reads sorted, filtered proposal pages directly from SQLite (25, 50 or 100 rows per browser payload), with counts computed from the same predicate. Its page snapshot binds draft, filters, probability range, order, page size, refresh generation and displayed unit IDs; mutations verify those against the current database before proceeding. **Current page** operates only on the verified page, **selected** requires a selection on that page, and **all** still means the full active draft, not merely loaded rows. Individual edits, pending and deliberately unreviewed proposals, commit/undo, conflict provenance and optional post-commit training retain their previous semantics. Sort columns are constrained by an SQL whitelist and have a stable unit-ID tie-breaker.

Apply classifier/committee source status is not computed while Apply is hidden; switching to Apply refreshes the current source list. After a committed Apply batch changes training evidence, the Develop classifier/committee badges reconcile on returning to Develop: unaffected fits remain Current and affected retained fits become Stale until retrained. Provisional Apply decisions do not change training freshness. Memo Presentation likewise defers map and page work until its presentation URL is active, resolves the current public view catalog, and accesses local display coordinates by page rows where possible. The rendered memo map is not an Explore visit and does not record navigation.

## Codes, Analyses, and Memos

**Codes** and **Memos** are project-wide research objects rather than exclusive workflow stages. **Analyses** are also project-wide: each named analysis saves a versioned configuration, not computed results. Opening an analysis recalculates results from current observations, human judgments, and selected fitted classifier predictions. Analyses may combine codes and categorical metadata in bar plots, heatmaps, and histograms.

The optional aggregation layer reduces code values from atomic observations to a selected parent key level. Metadata must be constant within each resulting parent group; conflicts fail explicitly. Pairwise correlations use complete cases and are exploratory descriptions, not inferential estimates. Saved-analysis definitions are persisted in schema 18.

## Integration boundaries

GeCo can operate on its own or use externally backed numerical data. External providers align their results with the project's original source keys; GeCo owns codes, observations, assignments, sessions, and other mutable exploratory state. The UI does not need to interpret an external provider's private storage layout.

Existing integration entry points include project creation/opening, representation access, and selected exports such as `export_codes(...)`. For current signatures, consult the [source package](../src/geometric_coder/) and tests rather than accessing GeCo's SQLite tables or private sidecar files.

## Future extensions

GeCo is intended to stay a focused **research interface** rather than accumulate every specialized analytical feature. A future plugin system could allow trusted, separately installed Python libraries to contribute actions, small configuration workflows, keyed derived results, or exports. For example, an interview-analysis library such as LIPE might supply protocol-specific analyses without requiring GeCo's core application to implement those algorithms.

The intended division of responsibility is GeCo for the research interface and mutable coding state, an appropriate analytic engine such as TeAL for immutable derived artifacts, and domain-specific extensions for specialist methods or outputs.

**A general plugin API does not exist yet.** Registration, compatibility, permissions, execution, and UI integration remain design work. Installed Python plugins would run trusted local code, not sandboxed scripts.
