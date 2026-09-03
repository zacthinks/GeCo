# Geometric Coder (GeCo)

**Geometric Coder (GeCo)** is research software for **geometric coding**: exploring and coding qualitative data while moving through one or more numerical representations of the corpus.

GeCo combines close reading with geometric navigation. Researchers can inspect neighborhoods and outliers, compare representations, search literally or semantically, code observations in context, write evidence-linked memos, and train code-specific classifiers to help direct attention during iterative analysis.

> **Status:** GeCo is pre-1.0 research software. The public API and project schema may still evolve.

## Why GeCo?

Many text-analysis workflows separate qualitative interpretation from numerical representation. GeCo puts them in the same exploratory workspace. A corpus remains readable as text and context, while embeddings, lexical spaces, projections, metadata filters, and model-guided recommendations provide additional ways to navigate it.

GeCo is deliberately **exploratory**. Its classifiers, geometric views, and recommendations are tools for inquiry and hypothesis development; they are not presented as confirmatory statistical inference or as automatically valid measurements for downstream population claims.

## Highlights

- **Plural geometries.** Work with lexical, reduced lexical, embedding, or custom numerical representations without treating any one space as canonical.
- **Context-aware coding.** Assign Present, Absent, or Unsure judgments while reading an observation in its hierarchy-aware context.
- **Generalized observations.** Code atomic rows, contiguous spans, and researcher-authored teaching examples.
- **Persistent exploration.** Save sessions, filters, navigation state, and linked Markdown memos.
- **Code-specific classifiers.** Each classifier belongs to one code, has a project-wide unique active name, and can be explicitly retrained as coding develops.
- **Model-guided review.** Use uncertainty, disagreement, prediction geometry, classifier committees, and Apply/Review workflows without turning model proposals into human judgments automatically.
- **External numerical resources.** A generic provider interface lets another system supply matrices and 2D views without copying those artifacts into the GeCo project. GeCo 0.7.0 includes the seam used by Text Analysis Lab (TeAL).
- **Local-first project storage.** Structured project state lives in SQLite with sidecar files for large numerical artifacts and models.

## Requirements

- Python 3.11–3.13
- [`uv`](https://docs.astral.sh/uv/) is recommended for environment and package management

The full UI and representation stack is installed from the repository with:

```bash
uv sync --all-extras --dev
```

For the lightweight core and development tools only:

```bash
uv sync --dev
```

Lemmatized count geometries require spaCy's English model:

```bash
uv run python -m spacy download en_core_web_sm
```

Sentence-transformer geometries use the normal user-level Hugging Face cache.

## Quick start

```python
import pandas as pd

from geometric_coder import GeometricCoder
from geometric_coder.recipes import lemma_tfidf, semantic_english

rows = pd.DataFrame(
    {
        "document_id": ["d1", "d1", "d2"],
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
    keys=["document_id", "sentence_id"],
    text="text",
    metadata=["speaker"],
    modality="text",
    geometries={
        "lexical": lemma_tfidf(),
        "semantic": semantic_english(),
    },
)

coder.launch()
```

The key columns may define a single identifier or a hierarchy from broadest to most specific. Input row order is canonical. Each input row becomes an atomic observation; broader key levels are used to reconstruct context.

To reopen an existing standalone project from the command line:

```bash
uv run geco launch demo.geco
```

## Interface

GeCo is organized around five workspaces:

- **Explore** — navigate 2D views, neighborhoods, searches, filters, context, coding controls, and linked memos.
- **Develop** — refine codes with teaching examples, code-specific classifiers, active-learning recommendations, committees, and prediction geometry.
- **Apply and Review** — review persisted machine proposals and commit only explicit human decisions.
- **Codes** — inspect and manage the evolving code system and coded observations.
- **Memos** — write versioned Markdown memos with clickable references back to corpus evidence, including a presentation mode for moving between an analytic narrative and the underlying observations.

## Project storage

A standalone GeCo project has a simple local layout:

```text
demo.geco/
├── project.sqlite3
└── artifacts/
    ├── geometries/
    ├── views/
    ├── observations/
    └── models/
```

SQLite is the source of truth for structured project state. Large matrices, fitted transforms, derived-observation vectors, and model artifacts are stored as registered sidecars.

GeCo 0.7.0 uses **project schema 12**. Runtime opening requires the exact schema version; GeCo never migrates a project automatically. A substantive schema-11 project can be upgraded explicitly with:

```bash
uv run python scripts/migrate_schema_11_to_12.py path/to/project.geco
```

The migrator creates a timestamped SQLite backup and checks database integrity before advancing the schema.

## External representations and TeAL

GeCo can also use externally backed geometries and 2D views. The external system owns the numerical artifacts; GeCo stores only opaque JSON references and asks a runtime provider for rows in GeCo's canonical key order.

```python
geco = GeometricCoder.create_external(
    project_dir="analysis.geco",
    data=document_frame,
    keys=["row_id"],
    text="text",
    metadata=["year"],
    external_provider=provider,
)

geometry_id = geco.register_external_geometry(
    name="tfidf",
    external_ref={"artifact_id": "..."},
    supports_query=True,
)

geco.register_external_view(
    geometry_id=geometry_id,
    name="umap",
    external_ref={"artifact_id": "..."},
)
```

## Design principles

GeCo's implementation follows a few strong rules:

1. Human judgments and model predictions remain distinct.
2. Original user keys and canonical row order are preserved.
3. Geometry dependencies are explicit rather than inferred from names.
4. Structured state is local and inspectable.
5. Expensive model fitting is explicit rather than silently triggered by ordinary UI interaction.
6. Provenance is retained where it matters, while superseded heavy classifier artifacts are not accumulated indefinitely.
7. External integrations go through narrow interfaces rather than embedding another application's storage or catalog logic inside GeCo.

For a deeper technical overview, see [`docs/architecture.md`](docs/architecture.md).

## Development

```bash
uv sync --all-extras --dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv build
```

## Citation

Publication forthcoming.

## License

GeCo is released under the [MIT License](LICENSE).