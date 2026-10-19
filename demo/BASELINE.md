# GeCo Wikipedia / DBpedia authoring baseline

This is **not yet the frozen public demo**. It is an editable starting project
you can explore, code, annotate, memo, and retrain before publication.

## Provenance

The earlier `InteractiveTopicModel` notebook
(`zacthinks/InteractiveTopicModel/examples/dbpedia_vignette.ipynb`)
uses the Hugging Face DBpedia-14 train data, categories
EducationalInstitution, Artist, Athlete, Building, Plant, seed 42,
and 100 articles per category.

This authoring workflow follows that five-category selection and obtains
the *current English Wikipedia lead paragraph* for each selected article
title using Wikipedia's API. DBpedia-14 `content` itself is **not**
guaranteed to be exactly a Wikipedia first paragraph, so it is not used as
the sentence text. Source URLs and revision identifiers are retained.
It uses a larger candidate pool to replace missing/short articles.

Text is segmented into sentences; stable nested keys are
`document_id` → `sentence_id`. Each row retains article title,
DBpedia category, source URL, and Wikipedia revision ID.
A sample of 500 articles is expected to yield thousands of sentence rows;
the exact count is determined only after collection.

Wikipedia text attribution and ShareAlike obligations must be reviewed
before republishing bundled text. Keep the article URLs, page revisions,
and any applicable license/attribution notice with the project.

## Run locally (Python 3.13)

Install GeCo and `datasets` in a suitable development environment.
The script uses spaCy's lightweight sentencizer; no language model is
required for splitting text.

```sh
uv sync --all-extras --dev
uv pip install datasets
uv run python -m demo.authoring collect
uv run python -m demo.authoring build
uv run python -c "from geometric_coder import GeometricCoder; GeometricCoder.open('demo_workbench/authoring.geco').launch()"
```

The collect step makes public HTTP requests to English Wikipedia and
downloads DBpedia-14 from Hugging Face. It caches each article's response
in `demo_workbench/sentences.articles.jsonl`; it does not silently use
synthetic sentences if requests fail.

The resulting GeCo project is yours to play with. Add codes, memos,
labels, and fitted classifiers as desired. The builder does **not** supply
a fixed initial set of codes or memos; we should not dictate those before
you try it.

## Freeze only when you approve the baseline

**Close the live GeCo UI first**, so no process can write sidecar artifacts
while the snapshot is taken:

```sh
uv run python -m demo.authoring freeze
```

This copies the *entire* project folder, not only SQLite. A freeze manifest
records SHA-256 digests for the database and artifacts. The file
`demo_data/starter.geco` is the proposed source for each disposable
hosted visitor project.

The current Dockerfiles **still construct the previous small synthetic
fixture** at build time; do not publish it as the DBpedia demo.
Before deployment, change the image assembly to **include the approved
frozen baseline** instead of running the old `demo.build_starter`.
The baseline may be too large for ordinary Git; use an appropriate versioned
artifact distribution rather than committing huge model files directly.

## Acceptance checks before deployment

- Correct category distribution, unique article keys, sentence order and provenance.
- Real TF-IDF and semantic geometries, saved 2D views, and context navigation.
- Codes, memos, classifier fits and predictions survive reopening the snapshot.
- Editing a visitor clone cannot change the pristine baseline.
- End Session and idle cleanup delete only the visitor clone.
