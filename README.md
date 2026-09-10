# Geometric Coder (GeCo)

**GeCo** (pronounced “Gecko”) is an early-stage Python package for explicit
**geometric coding** and, especially, **geometric qualitative coding**.

Instead of treating rich observations only as a list to be coded one-by-one,
GeCo represents atomic observations as points in one or more researcher-chosen
spaces. Researchers can compare geometries, navigate neighborhoods and
outliers, search literally or semantically, code observations in context,
write linked memos, and use code-specific classifiers to support focused
review.

GeCo is intentionally an **exploratory** environment. Its classifiers, scales,
proposals, and geometric views support qualitative inquiry and hypothesis
generation; GeCo does not present confirmatory p-values, inferential standard
errors, audit-sample correction, or claims that machine-assisted measurements
are population-valid. Confirmatory workflows should freeze and consume GeCo
artifacts through an external system such as Text Analysis Lab.

> Status: pre-alpha. GeCo 0.8.15 implements generalized observations,
> observation-based assignments, contiguous span coding, teaching examples,
> code-owned classifiers and committees sharing one project-wide active predictor-name namespace, with one retained
> fitted state per classifier, seven geometry-based classifier families,
> opt-in training-integrated cross-validation and full-evidence refitting with
> single-read geometry assembly for fast iterative retraining, fixed and trainable
> classifier committees with fit-preserving rename, optimized learned logistic
> stacking recommendations, optional automatic training before recommendations,
> neutral frozen predictor export for single classifiers and committees,
> finite Focus Coding tasks with protocol-aware completion,
> TeAL-style externally backed numerical representations, a
> prediction-geometry view in Develop, auditable/undoable Apply and Review bulk
> commits, and the five functional development workspaces.

## Installation

GeCo uses [uv](https://docs.astral.sh/uv/) for dependency and environment
management. The ordinary visual interface and built-in UMAP projection support
are standard dependencies, so a normal install can launch GeCo without a UI or
projection extra.

```bash
uv sync --dev
```

For development/acceptance testing with every optional model backend:

```bash
uv sync --all-extras --dev
```

Lemmatized count geometries also require a spaCy English model:

```bash
uv run python -m spacy download en_core_web_sm
```

SentenceTransformer models are downloaded to the normal user-level Hugging Face
cache when their geometries are first computed, so another project can reuse them.
The planned desktop Model Manager will expose account creation/sign-in, installed
spaCy and Hugging Face models, shared cache location and size, offline readiness,
and optional PyTorch/CPU/CUDA/MPS diagnostics without storing credentials in a
GeCo project.


### Install the English spaCy model

The bundled lemmatized lexical recipes require spaCy's English pipeline:

```powershell
uv run python -m spacy download en_core_web_sm
```

GeCo checks this requirement before creating a project, so a missing model now fails immediately rather than after corpus import. Surface-form geometries can instead use `CountGeometry(lemmatize=False)`.

## Quick start

```python
import pandas as pd

from geometric_coder import GeometricCoder
from geometric_coder.recipes import lemma_tfidf, semantic_english

rows = pd.DataFrame(
    {
        "document_id": ["d1", "d1", "d2"],
        "paragraph_id": [1, 1, 1],
        "sentence_id": [1, 2, 1],
        "text": [
            "The meeting opened quietly.",
            "Several participants became visibly angry.",
            "The second document discusses institutional trust.",
        ],
        "speaker": ["A", "B", "C"],
    }
)

coder = GeometricCoder.create(
    project_dir="demo.geco",
    data=rows,
    keys=["document_id", "paragraph_id", "sentence_id"],
    text="text",
    modality="text",
    metadata=["speaker"],
    geometries={
        "lexical": lemma_tfidf(),
        "semantic": semantic_english(),
    },
)

# The same project object will power the notebook API and local web UI.
# coder.launch()
```

When launched from IPython/Jupyter, GeCo keeps Dash's embedded notebook view and also displays a clickable **local browser URL** for opening the same running session in a full browser window. Outside IPython, that URL is printed as plain text. If GeCo binds to `host="0.0.0.0"`, the displayed local-browser link uses `127.0.0.1` instead of the wildcard address.

The supplied key columns may contain a single key (for example, `tweet_id`) or
an arbitrary hierarchy from broadest to most specific. Input row order is
canonical. Each row becomes an atomic observation; broader key levels recover
context. Every codable or displayable analytic object also has a general
`observation_id`, allowing assignments and geometry access to work consistently
for atomic rows, persistent spans, and teaching examples.

## Incremental ingestion

Projects whose corpora grow over time can append already-preprocessed atomic rows
without rebuilding the project:

```python
new_rows = pd.DataFrame(...)
result = coder.ingest(new_rows)
print(result["added"])
```

Incremental ingestion is deliberately strict in the current API. The incoming
DataFrame must contain exactly the same key, text, and metadata columns as the
original project, and its keys must not duplicate existing atomic keys. New rows
are transformed through the project's frozen geometry pipelines and persisted
transform-capable 2D views; existing representations are not refitted. If a
project contains a legacy/custom view that cannot transform new rows, ingestion
raises before changing project state.

Existing classifier fits are preserved, but become stale until their prediction
cache has been extended to the newly ingested observations. Pressing **Train**
with unchanged labels reuses the fitted estimator and scores only the missing
observations rather than fitting the same model again.

The exit-ticket example exposes this workflow directly for daily batches:

```powershell
uv run python examples/exit_ticket_project.py --csv todays_tickets.csv --append --launch
```

For now, `--append` assumes the supplied CSV contains only newly collected
tickets. It intentionally does not attempt to reconcile or deduplicate cumulative
re-exports.

By default, all non-key, non-text columns are preserved as metadata. Pass an
explicit `metadata=[...]` list to control exactly which fields appear in the
reading panel. Text-source columns used to construct the atomic text can
therefore remain in the DataFrame without being exposed in the interface.

## Geometry recipes and dependencies

Recipes may return one geometry or a nested mapping of geometries. This allows
one recipe to expose both a prerequisite geometry and a derived geometry:

```python
from geometric_coder.recipes import lemma_tfidf_svd

geometries = {
    "apple": lemma_tfidf_svd(n_components=100),
}
```

The bundle is normalized into public names such as:

```text
apple_tfidf
apple_svd_100
```

The SVD geometry retains an object-level dependency on the TF-IDF geometry;
dependencies are never inferred from names.

## Project storage

A GeCo project uses one SQLite database as its source of truth and sidecar
files for large artifacts:

```text
demo.geco/
├── project.sqlite3
└── artifacts/
    ├── geometries/
    ├── views/
    ├── observations/
    └── models/
```

SQLite stores the corpus, normalized key hierarchy, generalized observation
identities, ordered span membership, teaching-example status, geometry and view
registries, sessions, navigation events, stable code and memo identities,
append-only code-description and memo versions, assignment events, code-owned
classifier definitions, lightweight fit-history metadata, the current retained
classifier predictions, committees, evaluation events, and Apply drafts. For
ordinary projects, full corpus matrices and per-derived-observation vectors remain
in efficient sidecar formats. Externally backed projects may instead persist opaque
JSON references and obtain geometry matrices or 2D coordinates from a runtime
provider without copying those numerical artifacts into GeCo.

During private pre-1.0 development, GeCo intentionally carries **no backward-
compatibility or migration architecture**. Schema/API changes optimize for the
clean target design. A workspace whose schema does not match the running build
must be recreated from its source data and external-resource registrations. Do not
add compatibility shims or one-step migrators for developmental schemas.

## Interface

GeCo has five distinct workspaces. The header summarizes the modality, atomic
unit count, and each parent hierarchy level with its group count.

- **Explore**: the current implementation includes a click-responsive Plotly
  map whose zoom is preserved when focal units change, geometry and view
  selection, literal/regex filtering, semantic-search coloring, an explicit
  semantic-search clear action, interval filtering, type-aware metadata filters
  that subset the plotted and navigable corpus with AND/OR/XOR composition,
  reusable saved filter collections, deterministic pagination,
  and page-scoped nearest/farthest/random navigation. During an active
  semantic search, **Most similar** and **Least similar** appear and navigate
  within the selected geometry, current page, and active filters. A **Show only
  new points** option controls whether automated
  navigation excludes units visited in the current session. Contextual reading,
  selected metadata, and session state are persisted. Explore also includes a
  project-wide memo editor with highlighted hashtags and clickable evidence
  references, browser-style Back and Forward history, plus a coding panel for
  creating codes and assigning Present, Absent, or Unsure judgments without
  leaving the map. Context rows include a focal-anchored contiguous span
  selector: expanding an endpoint fills intervening rows, and a multi-row span
  is persisted only when first assigned.
  Explore interaction performance is partitioned deliberately: focal/visited navigation
  updates use small Plotly overlay patches rather than rebuilding the geometry map;
  multi-code palette structure is rebuilt only when rows/codes change; hidden workspaces
  short-circuit annotation refreshes; and ordinary coding refreshes the map only when
  the **Show only uncoded points** filter is active.
- **Develop**: a classifier-first, code-specific active-learning workspace.
  Every classifier belongs to exactly one code. Active classifier and committee names
  share one project-wide predictor namespace. Users select or create a classifier for the active code.
  GeCo currently supports L2, L1, and elastic-net logistic regression,
  Complement Naive Bayes, a linear-kernel SVM with probability estimation, a
  decision tree, and k-nearest neighbors. The selector carries a Current, Stale,
  or Not trained badge, while geometry, family, and selected hyperparameters
  appear as ordinary descriptive detail rather than internal fit identifiers.
  **Train** uses the classifier's currently stored hyperparameters by default,
  keeping ordinary active-learning refreshes fast. An opt-in **Tune hyperparameters
  with cross-validation** checkbox runs the compact family-specific stratified CV
  search before fitting; explicit tuning runs even when the retained fit is already
  current. Train all classifiers and Automatically train before recommendations
  remain stacked beneath Train and enabled by default for new Develop state, while
  hyperparameter tuning defaults off. Train all scopes both manual and automatic
  refreshes, and automatic training follows the same tuning checkbox. Each classifier
  retains one current fitted estimator plus prediction cache; with tuning off,
  already-current fits are reused immediately.
  Replacing a fit retires the prior heavy state while lightweight historical
  metadata and workflow snapshots preserve provenance needed by old Apply drafts
  and evaluation events. A source selector applies Most likely, Least likely,
  and Most uncertain either to the active classifier or a selected committee;
  Greatest disagreement appears only for committees, while Random and review of
  unsure judgments remain generally available. Committees support mean, median,
  minimum, maximum, harmonic-mean, geometric-mean aggregation, and learned logistic
  stacking. Manage committees can rename an active committee without changing its
  stable ID, members, aggregation, or learned fit; active committee names share the
  same project-wide predictor namespace as classifiers. Current learned-committee recommendations use the persisted committee score
  vector directly and fetch member scores only for the selected observation; Greatest
  disagreement batches member score vectors once. The coding panel displays member
  model probabilities and the resulting committee probability separately. Training may
  include atomic observations, persistent spans, and active teaching examples.
  The Teaching Examples panel supports status and label changes plus optional
  assessment against existing fits before a new example joins later training. A
  Testing Center summarizes diagnostic events. A refreshable prediction geometry
  plots one classifier as a jittered
  score line, two classifiers as direct axes, and three or more through PCA.
  The view uses deterministic pagination, can retain every coded observation,
  and can optionally include spans and teaching examples. Assignment colors are
  black/green/red/yellow for unreviewed/Present/Absent/Unsure; circles, triangles,
  and stars distinguish atomic observations, spans, and teaching examples.
  Clicking an atomic observation or span returns to the contextual coding view
  without fading unrelated points; hover labels show only the identifying key.
- **Apply and Review**: a proposal-review workflow built around inspectable
  drafts and a themed Dash AG Grid review table. Users select a code and either
  one of that code's classifiers or a named committee. The generated draft
  snapshots the exact predictions and fit-time provenance used, so replacing a
  classifier's current fitted state later does not alter the draft's meaning.
  Users can filter by proposal, review decision,
  or probability range; edit individual decisions with automatic persistence;
  run organized bulk actions; commit all or only visible reviewed proposals;
  and discard drafts with confirmation. Pending means no decision has been
  made. Leave unreviewed records a deliberate draft-level nondecision but
  remains visible until commit; commit removes it without creating an
  assignment. Existing annotations are never overwritten.
- **Codes**: a Code Center for searching labels or descriptions, filtering by
  hashtags, browsing and editing append-only description versions, duplicating
  access to the shared teaching-example manager, listing the current positive
  extension across observation kinds, revising the selected corpus observation
  among Present, Absent, and Unsure, and inspecting positives in registered
  geometries and transform-capable 2D views. Every positive observation is
  black; circles, triangles, and stars distinguish atomic observations, spans,
  and teaching examples. The focal observation retains GeCo's green outline.
- **Memos**: a project-wide Memo Center for searching titles or full memo text,
  filtering by multiple hashtags with AND/OR/XOR logic, browsing immutable memo
  versions, editing by appending a new version, and following memo references
  back to the focal unit in Explore. Clicking a memo hashtag opens the Memo
  Center with that hashtag already selected. Memo Center also offers
  **Presentation mode** in a separate window/tab: the selected memo version is
  rendered beside a minimal map with a 2D-view selector, saved-filter selector,
  deterministic pagination, and a read-only text/context panel. Presentation
  links and map clicks do not create session visits. A memo-linked observation
  that falls outside the selected saved filter is overlaid without changing the
  filter so its evidence can still be read.

Named sessions preserve visited units and interface state. Automated
recommendations exclude units already focalized in the current session;
manual clicks and memo hyperlinks may revisit them. Explore pagination and all
automated navigation commands remain within the active page. Operations that take long enough to create ambiguity show a delayed,
viewport-wide translucent gray “GeCo is working” overlay. A right-facing gecko
walks across the card, turns around, and returns; reduced-motion mode keeps it
still. First-time span transformation, classifier fitting, view creation, and
other nontrivial callbacks use the same feedback pattern.

### Generalized observations from Python

```python
code_id = coder.create_code("Joint claim")

# Atomic rows already have observation IDs.
atomic_observation_id = coder.observation_id_for_unit(7)
coder.annotate(atomic_observation_id, code_id, "positive")

# A contiguous span is created lazily on its first assignment.
span_observation_id, event_id = coder.annotate_span(
    [7, 8, 9],
    code_id,
    "positive",
)

# Teaching examples are code-specific observations and start Active.
teaching_observation_id = coder.create_teaching_example(
    code_id=code_id,
    text="An authored example of the same joint claim.",
    label="positive",
    note="Clarifies the intended boundary.",
)
coder.set_teaching_example_label(teaching_observation_id, "negative")
coder.set_teaching_example_status(teaching_observation_id, "inactive")
```

A reviewed assignment can change among `positive`, `negative`, and `unsure`,
but it cannot be returned to `unreviewed`. No assignment event means the
observation-code pair has never been reviewed.

### Classifiers and Apply from Python

```python
code_id = coder.create_code("Anger")
coder.annotate(1, code_id, "positive")
coder.annotate(20, code_id, "negative")

# Each code owns its classifiers. Project defaults are code-specific, and active
# classifier names are unique project-wide.
specs = coder.classifier_specs(code_id=code_id)
active_spec_id = specs[0]["classifier_spec_id"]

# One code may own multiple independently configured classifiers, including
# multiple families over the same geometry. Current families are:
# logistic_l2, logistic_l1, logistic_elasticnet, complement_nb, linear_svm,
# decision_tree, and knn.
second_spec_id = coder.create_classifier_spec(
    code_id=code_id,
    name="Anger semantic L1",
    geometry_id=specs[0]["geometry_id"],
    algorithm="logistic_l1",
)

# Ordinary training is fast: it uses each classifier's currently stored
# hyperparameters. Opt into the slower CV search when you deliberately want to
# retune them. Each classifier still retains only one live fitted state.
fits = coder.train_classifiers(
    code_id=code_id,
    classifier_spec_ids=[active_spec_id, second_spec_id],
    tune=False,
)

tuned_fits = coder.train_classifiers(
    code_id=code_id,
    classifier_spec_ids=[active_spec_id, second_spec_id],
    tune=True,
)

committee_id = coder.create_classifier_committee(
    code_id=code_id,
    name="Two-logistic committee",
    classifier_spec_ids=[active_spec_id, second_spec_id],
    aggregation="mean",
)

apply_run_id = coder.create_apply_run(
    code_id=code_id,
    source_kind="committee",
    source_id=committee_id,
    threshold=0.5,
)

# Review can be individual or bulk. No annotation changes yet.
coder.bulk_review_apply_proposals(
    apply_run_id=apply_run_id,
    decision="accept",
)
coder.review_apply_proposal(
    apply_run_id=apply_run_id,
    unit_id=7,
    decision="unsure",
)
# A deliberate draft-level nondecision remains visible until it is committed.
coder.review_apply_proposal(
    apply_run_id=apply_run_id,
    unit_id=8,
    decision="unreviewed",
)

# Commit appends reviewed decisions as assignment events.
result = coder.commit_apply_run(apply_run_id)
```

Apply drafts include only units that were unreviewed for the selected code when
the draft was generated. Within a draft, `pending` means no review decision yet,
whereas `unreviewed` means deliberately make no assignment. The row stays in the
active draft until commit, which then removes it without creating an assignment.
If another workflow labels a proposed unit before commit,
GeCo preserves the newer annotation and reports a skipped conflict.

## AERA 2026 example

Place the AERA CSV files anywhere beneath `examples/aera_2026/`. The example
loader recursively reads every CSV, checks that their schemas match, drops
`source_url`, joins `title` and `abstract` with a blank line, and splits each
record with spaCy's lightweight rule-based sentencizer. GeCo receives one row
per sentence with the compound key `(record_id, sentence_id)`. `record_id`
follows canonical file-and-row order, while a selected set of AERA identifiers and program fields remains available
as repeated sentence-level metadata. Title, abstract, and the derived record
text are deliberately omitted from the metadata panel because they are already
represented in the reading view. This provides a real hierarchical corpus for
testing GeCo's context interface.

```bash
uv run python examples/aera_2026_project.py --overwrite --launch
```

For a faster development run, sample records before sentence expansion:

```bash
uv run python examples/aera_2026_project.py \
  --sample-records 500 \
  --sample-seed 17 \
  --overwrite \
  --launch
```

The example eagerly creates three public geometries: lemmatized TF-IDF, its
100-dimensional SVD/LSA derivative, and MiniLM embeddings. MPNet is omitted from
the default development example so first-time setup and rebuilding are faster.
It reopens a current-schema project without refitting. The disposable AERA
example may rebuild an outdated example project from source data. During pre-1.0
development, any schema-mismatched workspace should likewise be recreated rather
than migrated. Use `--overwrite` to force a rebuild even when the schema is current.
The first run therefore downloads the selected embedding models and computes
all matrices and default views before opening the interface. The resulting
project is cached at `examples/aera_2026.geco/`. Launch it again without
recomputing anything using:

```bash
uv run geco launch examples/aera_2026.geco
```

This opens the frozen project in place without recomputing representations or views.

Pre-1.0 schema mismatches are deliberate hard boundaries. Recreate the workspace
from source rather than migrating it or adding runtime compatibility logic.

For externally backed numerical representations (including the intended TeAL
integration seam), see [docs/teal-integration.md](docs/teal-integration.md).

### Focus Coding

Focus Coding is a finite human-labeling mode for a fixed set of atomic observations
and required codes. It uses the same GeCo project, assignments, code versions, and
resumable session state, but presents a stripped-down coding surface with progress,
protocol-aware Unsure handling, unresolved-only review, and a validated Done action.

```python
focus_session_id = coder.configure_focus(
    codes=[qual_code_id, quant_code_id],
    allow_unsure=False,
    # Optional external ordering/selection by stable user key:
    user_keys=[{"document_id": "d3"}, {"document_id": "d1"}],
)

coder.launch_focus_coder(session_id=focus_session_id)
```

If `user_keys` is omitted, the complete atomic project snapshot is used in canonical
import order. External systems such as TeAL should use stable user keys rather than
GeCo-local unit IDs. The requested text and metadata are already part of the local
GeCo snapshot, so Focus Coding requires no live numerical provider and exposes no
geometry, classifier, committee, or machine-prediction controls. When
`allow_unsure=False`, the Unsure action is absent and any pre-existing Unsure judgment
counts as unresolved until changed to Present or Absent.

Clicking **Done** validates every required document-by-code judgment. Incomplete tasks
enter unresolved-review mode; complete tasks persist a closed Focus status that can be
explicitly reopened for editing. See
[docs/teal-focus-coding.md](docs/teal-focus-coding.md) for the TeAL creation contract.

### Public external-resource inspection

Restartable TeAL or notebook workflows can inspect and reuse registered resources
without reaching into GeCo internals or interpreting opaque `external_ref` values:

```python
for geometry in coder.geometries():
    print(geometry["geometry_id"], geometry["name"], geometry["storage_kind"], geometry["supports_query"], geometry["supports_text_transform"])

for view in coder.views():
    print(view["view_id"], view["name"], view["storage_kind"])

if not coder.has_geometry("minilm"):
    coder.register_external_geometry(
        name="minilm",
        external_ref={"artifact": "..."},
        supports_query=True,
        supports_text_transform=True,
    )

# Equivalent idempotent notebook style:
coder.register_external_geometry(
    name="minilm",
    external_ref={"artifact": "..."},
    supports_query=True,
    supports_text_transform=True,
    if_exists="reuse",
)

query_vector = coder.transform_query("minilm", "qualitative interview methods")
```

`if_exists="reuse"` only reuses an existing external resource when its persisted
identity and declared capabilities match. Conflicting registrations still fail.

External providers may be invoked from Dash callback threads. Provider implementations
must therefore avoid reusing thread-affine database connections across calls; open and
close any such state inside the calling thread.

### Frozen predictor export

GeCo 0.8.14 exposes one neutral prediction handoff for both individual classifiers
and classifier committees. External systems should discover stable predictor references
and export the selected fitted procedure without asking whether it is internally a
classifier or committee:

```python
refs = coder.predictors(code_id)
predictor = coder.export_predictor(refs[0])

print(predictor.sources)
print(predictor.manifest())
```

Active classifiers and committees share one project-wide predictor-name namespace, so
every live predictor has an unambiguous name regardless of its internal GeCo kind.
Archived `_deleted...` names are outside that active namespace, and historical fit-time
names remain unchanged for provenance.

The frozen predictor declares an ordered list of unique required geometries. Batch
inference always accepts an equally ordered list of matrices, so sparse and dense
representations remain separate rather than being concatenated and split again:

```python
inputs = [coder.geometry_matrix(source.geometry_id) for source in predictor.sources]
probabilities = predictor.predict_proba(inputs)
```

If two committee members use the same geometry, that source appears only once and the
frozen member routing sends it to both estimators. Learned logistic-stacking committees
carry the exact fitted member estimators, member order, and fitted stacker. Export never
retrains. A stale retained procedure is rejected by default and may be frozen only with
`allow_stale=True`; an old learned stacker is never silently paired with a newer member
fit. The neutral export contains no TeAL dependency; the TeAL bridge can translate this
contract into its own durable `GeCoPredictor` operator.

During GeCo/TeAL integration work, routinely launch the three-record external-resource
debug smoke with Dash development checks enabled:

```bash
uv run python scripts/debug_external_ui_smoke.py
```

The expected result is an Explore map containing exactly three points immediately after
launch. The smoke uses `debug=True` with the reloader disabled so Dash's validation and
dev tooling remain active without duplicating the temporary provider lifecycle.


Use custom paths when needed:

```bash
uv run python examples/aera_2026_project.py \
  --data-dir path/to/aera_2026 \
  --project-dir path/to/aera_2026.geco \
  --overwrite \
  --launch
```

## Development

```bash
uv sync --dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
```

The packaging acceptance smoke deliberately uses a new virtual environment and
installs only the built wheel plus its declared standard dependencies:

```bash
uv run python scripts/run_clean_install_smoke.py
```

This verifies package import, tiny-project creation/reopen, the core Dash/Plotly/
AG Grid/UMAP imports, and `create_app(...)` construction without relying on the
fully provisioned repository environment.

See [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/architecture.md](docs/architecture.md).

## License

MIT. See [LICENSE](LICENSE).

## Development UI smoke checks

Use a real locally computed project—not only an external toy provider—when checking
the ordinary browser surface:

```bash
python scripts/debug_local_ui_smoke.py
```

The real AERA acceptance example likewise prepares or reopens the project first and
then launches it directly through `GeometricCoder.launch(...)`. It deliberately does
not use the temporary `BrowserProgressPage` handoff; that helper is a separate utility
lifecycle and is not part of the ordinary launch acceptance path.

GeCo no longer carries the temporary Dash `<4.2` diagnostic pin. The normal package
constraint is again `dash>=4.0`; renderer compatibility should be evaluated through
the real-project smoke rather than by silently substituting an external toy case.

The clean-install smoke also creates a local TF-IDF geometry and persisted 2D view,
then requests `/`, `/_dash-layout`, and `/_dash-dependencies` through Dash's Flask
server before separately exercising the external-provider seam.

