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

> Status: pre-alpha. GeCo 0.8.1 implements generalized observations,
> observation-based assignments, contiguous span coding, teaching examples,
> code-owned classifiers with project-wide unique active names and one retained
> fitted state per classifier, seven geometry-based classifier families,
> training-integrated cross-validation and full-evidence refitting, fixed and
> trainable classifier committees, optional automatic training before
> recommendations, TeAL-style externally backed numerical representations, a
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

During private pre-1.0 development, GeCo does not maintain runtime backward
compatibility. Once a project contains substantive analytic work, each
schema-changing checkpoint should instead provide a narrowly scoped standalone
script for the immediately previous schema. These migrations are **never run
automatically by GeCo**. Run the matching script explicitly, inspect the backup,
and then open the migrated project normally. Disposable example/test projects may
still be rebuilt from source when no analytic work needs preservation.

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
- **Develop**: a classifier-first, code-specific active-learning workspace.
  Every classifier belongs to exactly one code, and active classifier names are
  unique project-wide. Users select or create a classifier for the active code.
  GeCo currently supports L2, L1, and elastic-net logistic regression,
  Complement Naive Bayes, a linear-kernel SVM with probability estimation, a
  decision tree, and k-nearest neighbors. The selector carries a Current, Stale,
  or Not trained badge, while geometry, family, and selected hyperparameters
  appear as ordinary descriptive detail rather than internal fit identifiers.
  **Train** owns model selection: once at least two Present and two Absent
  examples exist, GeCo performs compact family-specific stratified
  cross-validation and then refits the selected configuration on all current
  evidence; earlier than that it trains immediately from the current/default
  settings. Train all classifiers and Automatically train before recommendations
  are stacked beneath Train and enabled by default for new Develop state. Train all
  scopes both manual and automatic refreshes, so early active-learning loops can
  stay smooth; either can be disabled later for larger manual batches. Each classifier retains one current fitted estimator plus
  prediction cache, and already-current fits are reused without rerunning CV.
  Replacing a fit retires the prior heavy state while lightweight historical
  metadata and workflow snapshots preserve provenance needed by old Apply drafts
  and evaluation events. A source selector applies Most likely, Least likely,
  and Most uncertain either to the active classifier or a selected committee;
  Greatest disagreement appears only for committees, while Random and review of
  unsure judgments remain generally available. Committees support mean, median,
  minimum, maximum, harmonic-mean, and geometric-mean aggregation. Training may
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

# Train owns model selection. If there are at least two Present and two Absent
# examples, GeCo performs family-specific stratified CV, selects hyperparameters,
# and then refits on all current evidence. Earlier than that it trains from the
# current/default settings. Each classifier still retains only one live fitted state.
fits = coder.train_classifiers(
    code_id=code_id,
    classifier_spec_ids=[active_spec_id, second_spec_id],
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
example may rebuild an outdated example project from source data, but GeCo itself
never migrates substantive projects automatically. Use `--overwrite` to force a
rebuild even when the schema is current.
The first run therefore downloads the selected embedding models and computes
all matrices and default views before opening the interface. The resulting
project is cached at `examples/aera_2026.geco/`. Launch it again without
recomputing anything using:

```bash
uv run geco launch examples/aera_2026.geco
```

This opens the frozen project in place without recomputing representations or views.

When a substantive project is exactly one schema version behind, run the matching
standalone migration explicitly. For example:

```bash
uv run python scripts/migrate_schema_11_to_12.py path/to/project.geco
```

The script validates the source schema, creates a timestamped pre-migration SQLite
backup, performs the one-step migration, and runs integrity checks. `geco launch`
and `GeometricCoder.open(...)` deliberately reject mismatched schemas; they never
migrate projects automatically.

For externally backed numerical representations (including the intended TeAL
integration seam), see [docs/teal-integration.md](docs/teal-integration.md).

### Public external-resource inspection

Restartable TeAL or notebook workflows can inspect and reuse registered resources
without reaching into GeCo internals or interpreting opaque `external_ref` values:

```python
for geometry in coder.geometries():
    print(geometry["geometry_id"], geometry["name"], geometry["storage_kind"], geometry["supports_query"])

for view in coder.views():
    print(view["view_id"], view["name"], view["storage_kind"])

if not coder.has_geometry("minilm"):
    coder.register_external_geometry(
        name="minilm",
        external_ref={"artifact": "..."},
        supports_query=True,
    )

# Equivalent idempotent notebook style:
coder.register_external_geometry(
    name="minilm",
    external_ref={"artifact": "..."},
    supports_query=True,
    if_exists="reuse",
)

query_vector = coder.transform_query("minilm", "qualitative interview methods")
```

`if_exists="reuse"` only reuses an existing external resource when its persisted
identity and declared capabilities match. Conflicting registrations still fail.


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
