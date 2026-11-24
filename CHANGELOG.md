# Changelog

Notable user-facing changes to GeCo are recorded here. GeCo follows pre-1.0 versioning: minor versions may include changes to public APIs or project schemas.

## [Unreleased]

No changes recorded after this release candidate.

## [0.15.0] - Unreleased (preparing)

### Added

- **Automatic local port selection** (GeCo #94): ordinary and Focus launches use an OS-allocated available port by default. An explicitly requested positive port remains strict and fails if occupied.
- **Launch-scoped execution device** (GeCo #95): `GeometricCoder.launch(device=...)` and `geco launch --device` accept `auto`, `cpu`, `cuda`, `cuda:N`, and `mps`. Auto prefers an available CUDA device, then Apple MPS, otherwise CPU. Explicitly unavailable accelerators fail rather than silently falling back.

### Changed

- **Standard embedding support** (GeCo #96): Sentence Transformers are now a normal dependency of `geometric-coder`. The former `embeddings` and `all` extras are removed; model weights are still fetched on demand.
- **Session device visibility** (GeCo #95–#96): Settings shows a concise, read-only CPU or GPU backend status for the resolved launch session. No device preference is persisted and there is no live backend toggle.
- **Development installation** (GeCo #96): workflows use plain `uv sync` rather than removed extras.

### Compatibility and limits

- GPU selection applies only where GeCo/TeAL support accelerated execution; CPU-only stages remain on CPU. Frozen scientific device constraints remain authoritative, and an incompatible geometry fails when that geometry is used, without blocking unrelated compatible geometry operations.
- This is a minor pre-1.0 release. GeCo's project schema, saved scientific models, and TeAL provider contracts are not rewritten by device selection or port binding.

### Release notes

- Preparation only: verify exact-head CI, acceptance tests, a clean wheel install, build artifacts and the GitHub Release before publishing the `v0.15.0` tag.

## [0.14.1] - 2026-11-21

### Fixed

- **Context span selection and parent boundaries** (GeCo #92): keep one focal-anchored, contiguous interval even with redundant checkbox events. Only canonical consecutive source rows within the focal unit's immediate parent can be selected together; interleaved groups cannot create a span the persistence validator rejects.
- **Context reader navigation** (GeCo #92): double-click a passage to navigate directly to that observation in Explore or Develop, with existing visit provenance and Back history. Enter and Space provide keyboard navigation. Interactive checkboxes retain their separate single-click semantics; the passage text remains the accessible name for screen readers.
- **Scale prediction browser-test stability** (GeCo #92): extend only the post-refresh second-point inspection wait from 15 to 30 seconds, retaining correct-point checks and adding diagnostic output for persistent failures.

### Notes

- This is a **patch release** on top of v0.14.0, with no new project schema, numerical representation compatibility contract, model fitting behavior, or TeAL-linked API requirement.
- The v0.14.1 tag predates the automatic-port and session-device changes included in v0.15.0.

## [0.14.0] - 2026-11-21

### Added

- **Constructs and continuous Scales** (GeCo #55–#57, #68): define either binary Codes or continuous Scales in Develop; record ordered pairwise judgments (A higher, parity, B higher, Unsure), optionally bootstrap comparisons from an existing Code with an explicit Present > Absent assumption, train and retain a linear Davidson scaler, and inspect reference-centered signed prediction positions and conditional posterior uncertainty. Random and bounded Bayesian expected-information-gain pair recommendations support iterative comparison review. These are exploratory model-based scores—not human numerical ratings, probabilities, or validated population measures.
- **Visualization-first Analyses** (#81 and its integration phases): choose a chart, then assign typed sources (human or predicted Codes, fitted Scale positions, categorical/numeric/temporal metadata) to independent roles. New continuous and mixed charts include scatterplots, histograms, numerical grouped summaries, pairwise-complete correlation heatmaps, grouped-value heatmaps, and a unified Bar editor with configurable Count/Sum/Mean/Median where valid. Source missingness, model freshness, coverage counts, aggregation choices, and deterministic sampling/approximation warnings are explicit.
- **Scale modeling and analysis references**: new contributor notes for Davidson fitting, active pair acquisition, Scale comparison UI, typed analytical variables, and numerical chart interpretation. Saved schema-v1 Analyses remain executable alongside newer role-based recipes.
- **Performance diagnostics**: opt-in 1k/50k/300k tiered benchmarks with reproducible fixtures, paired comparisons and explicit allocation/resource limitations (#85).

### Changed

- **Larger-workspace responsiveness** (#83, #84, #86, #89, #90): bounded SQL context retrieval with visible 200-unit starting window and progressive expansion (up to 5,000 displayed units), compact Develop classifier plot payloads, optional batched TeAL topic artifact reads with legacy provider fallback, adaptive small-group context SQL, and reuse of decoded metadata within a bounded Analyses resolver page. Correctness/output parity and scoped benchmarks were checked; these are not universal browser-rendering speedup claims.
- **Representation and startup behavior** (#71, #75, #82): newly registered analytic geometries refresh in the interface; creating another geometry remains allowed when spans or teaching examples exist even if that geometry cannot represent them. Standard launch now prewarms common UMAP 2D cosine kernels as well as applicable semantic-search libraries. Opt out using `geco launch --no-warmup` or `GeometricCoder.launch(warmup=False)`; it may increase initial startup time.
- **Project schema advances to version 21**, adding typed Construct and Scale comparison/fit/acquisition persistence through supported forward migrations from existing schema 18 projects. Existing Code and Analyses v1 research evidence is retained. Earlier GeCo versions may not reopen migrated workspaces; back up research projects before upgrading.

### Notes

- **GeCo v0.13.0 already provided classifier feature-column normalization**, which is distinct from the new continuous **Scale** Construct functionality. The former transforms classifier input features; the latter models pairwise researcher comparisons.
- Continuous Scale predictions and learned acquisition are conditional on the chosen representation and model assumptions, not confirmatory measurement or population inference. Analyses remain exploratory and never train models or silently commit evidence.
- TeAL-linked topic batching uses the optional `read_artifacts` provider capability added in TeAL #79; older providers remain supported with sequential reads. Test against a compatible TeAL version when enabling the new integration.
- The initial context display cap is expandable, but very long spans may require deliberate navigation. Contiguous span interval corrections and double-click-to-focal behavior were subsequently introduced in v0.14.1. Multiple active units of analysis within one GeCo workspace were not included in v0.14.0 and remain outside the current product scope.
- **Published November 21, 2026:** v0.14.0 is available as a tagged GitHub Release with wheel and source distribution.

## [0.13.0] - 2026-11-12

### Added

- **Classifier feature-column scaling** in Develop (PR #52). For regularized logistic regression and linear SVM classifiers, select Off (existing behavior), Fixed (full scaling), or Tune to cross-validate the feature-scale exponent γ in {0, 0.5, 1} alongside existing classifier hyperparameters. The scaler uses training-column RMS by default, supports sparse matrices, and is fitted separately in each validation fold.
- **Add metadata features** in Manage Representations for TeAL-linked numerical geometries (PR #53). Append selected source metadata using raw numeric values, z-score, min-max, categorical one-hot, fixed-width bins with an anchor, quantile bins, equal-width bins, or explicit bin edges. TeAL 0.12.0 owns and freezes the FeatureMatrix recipe; GeCo registers the result as an ordinary analytic geometry, available for views, exploration, classifiers, and cross-geometry committees.
- **Out-of-fold logistic stacking** for learned classifier committees (PR #54). The logistic meta-model trains on held-out member probabilities from up to five shared stratified folds rather than predictions made on each member's own training examples. Each member retains its own geometry, fitted model configuration, scaling, and probability calibration.

### Changed

- Metadata-augmented geometries preserve TeAL source keys, lazy numerical storage, and the original representation as the parent. They are navigation/analysis-only: arbitrary new-text transformation and semantic query are not offered when corresponding metadata is unavailable.
- Learned committees require at least **two jointly represented Present and two Absent examples** to construct valid out-of-fold training predictions. Infeasible training now reports an error rather than quietly falling back to in-sample stacking.
- Previously fitted **in-sample learned committees become Stale** and must be retrained to use the new OOF procedure. Existing historical prediction records and Apply review provenance are retained. Superseded heavy stacker estimator files are retired after successful replacement.
- The frozen classifier/committee predictor export continues to route distinct member geometries independently. Scaled member estimators use portable fitted normalization state for TeAL integration.

### Notes

- This release does **not** change the GeCo project schema (version 18). Feature-column scaling is opt-in; existing classifiers remain unchanged by default.
- Metadata augmentation requires a **TeAL-linked workspace with TeAL 0.12.0 or newer** and metadata queryable from the selected source artifact. Fields copied only into GeCo's local SQLite database are not silently included.
- Out-of-fold stacking removes direct in-sample meta-feature leakage, but previously chosen member hyperparameters may still reflect full-development-data tuning. Neither cross-validation nor estimated probabilities establish confirmatory accuracy. Use an independent audit or appropriate downstream inferential procedure for population claims.
- Trainable committees may need retraining after upgrading. Back up research workspaces before upgrading during pre-1.0 development.

## [0.12.0] - 2026-11-11

### Added

- **Original text** as a source in Manage Representations for TeAL-linked projects: create a spaCy token-based document-term matrix or Sentence Transformer embeddings directly from the linked text source (PR #51).
- Reuse compatible previously saved spaCy token artifacts, including token artifacts produced alongside sentence outputs by a single TeAL spaCy operation. TeAL retains ownership of artifact provenance, frozen operators, and native query-text replay.
- Token-feature selection (text, lemma, or normalized form), Boolean token-attribute filters, n-gram lengths, minimum document frequency, optional feature limits, binary counts, and explicit handling of empty spaCy documents.
- User-facing links to spaCy model catalogs and Hugging Face Sentence Transformer models, without exposing long local model paths in saved-token dropdown labels.
- Automatic, editable representation-name suggestions based on source, transformation and output dimensions (when known), using lowercase underscores and numeric suffixes to avoid collisions.

### Changed

- Cosine becomes the default metric in the UMAP representation wizard and in compatible K-medoids and DBSCAN topic-model controls, without changing TeAL operator defaults or metric-constrained clustering algorithms.
- The linked GeCo integration workflow now tests against merged TeAL `main`, rather than an interim upstream commit.

### Fixed

- DTM creation with no countable values in the selected spaCy token field now reports the selected field and attribute filters, with a specific suggestion for missing lemmas from spaCy models lacking a lemmatizer. The requested feature is never silently replaced.
- Saved-token dropdown labels remain compact while preserving their stable TeAL artifact IDs as selection values.

### Notes

- New text geometries require a compatible TeAL-linked project; source keys and full observation coverage are validated before registration. The new sibling spaCy sentence/token discovery and replay contract requires TeAL v0.11.3 or newer.
- The frozen spaCy operator used for new sentence text may produce context-sensitive annotations that differ from the original full-document parse. TeAL handles the narrowly verified looser-lineage reconstruction without modifying recorded corpus provenance.
## [0.11.0] - 2026-11-11

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

## [0.10.0] - 2026-11-03

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
