# 🦎 Geometric Coder (GeCo)

**Explore your corpus. Develop your codes. Apply and review them at scale.**

GeCo (pronounced *gecko*) is an open-source, local workspace for exploratory text analysis and qualitative coding. It brings together close reading and researcher-defined codes with computational tools for navigating a corpus, learning from human judgments, and reviewing coding proposals efficiently.

A central idea is **geometry**: a computational representation of relationships among observations. One geometry may bring together texts that use similar words; another may connect texts that express similar ideas. GeCo lets researchers explore these different perspectives while remaining connected to the original text and its context.

GeCo's main workflow has three interconnected stages: **Explore, Develop, and Apply & Review**. The stages are not a one-way pipeline: researchers can move among them as their codes evolve.

## The GeCo workflow

### 1. Explore — discover patterns, follow connections, and start coding

Explore your corpus through interactive maps of different **geometries**—for example, representations based on word use or semantic meaning. Move among nearby, distant, or randomly selected observations; filter by metadata; and save useful combinations of filters. Select any observation to read its text in context, examine surrounding passages, and start coding.

Search in different ways depending on the question. **Literal search** finds words or phrases in the original text (including regular-expression searches). **Semantic search** finds observations similar in meaning to a query, using the selected geometry. Because similarity depends on the geometry, switching representations can facilitate the retrieval of different documents.

GeCo also supports **topic-model exploration**. Inspect topic memberships, color observations by their dominant topic or their association with a particular topic, and give topics meaningful labels as you interpret them. In projects integrated with **Text Analysis Lab (TeAL)**, you can also fit topic models from existing representations—including latent Dirichlet allocation (LDA) and a broad range of clustering-based approaches—and explore their results in GeCo.

As you discover patterns and exceptions, develop candidate codes, assign judgments to individual observations or contiguous spans, and write memos linked to the original evidence. Explore is a place to make interpretive discoveries, not just to browse a visualization.

### 2. Develop — teach models what your codes mean

Develop your codes by combining human coding decisions with **active learning**. For each code, GeCo can train classifiers in different geometries—for example, a lexical representation based on word use and a semantic representation based on meaning. These are computational ways of recognizing the code, learned from examples you have judged yourself.

Instead of labeling cases at random, you can ask GeCo to find informative observations: ones a classifier considers likely to match a code, ones it is uncertain about, or ones on which multiple classifiers disagree. Review those observations, refine your coding decisions, and train again. You can compare classifiers across geometries or combine them in a committee.

The goal is not for the computer to define your code. **You define and refine the code; GeCo helps you develop computational representations that can recognize it across the corpus.**

### 3. Apply & Review — code broadly, review efficiently

Once a code-specific classifier or committee is useful, use it to generate coding proposals for observations you have not yet reviewed. GeCo's **Apply & Review** workspace puts those proposals in a review table, where you can inspect suggested labels and probabilities, filter the results, accept or revise individual decisions, and take bulk actions when appropriate.

You retain control over what becomes a coding decision. You can change a proposal to **Present**, **Absent**, or **Unsure**, or deliberately leave it unreviewed. Machine predictions are kept separate from human assignments until you commit reviewed decisions; existing annotations are not silently overwritten.

### Move back and forth

This is an **iterative workflow**, not a sequence you complete only once. Reviewing a broad set of proposals may reveal a surprising case or a weakness in a code's definition. You can return to Explore to read it in context, revise the code and its examples, go back to Develop to improve the classifiers, and then apply and review again.

Dedicated **Codes** and **Memos** workspaces help you manage code definitions, coding evidence, and interpretive notes throughout the process.

## GeCo, TeAL, and GeCo Desktop

GeCo can be used independently or integrated with [Text Analysis Lab (TeAL)](https://github.com/zacthinks/text-analysis-lab). TeAL supports corpus preparation, computational representations, and downstream analysis; GeCo adds interactive exploration, qualitative coding, and classifier development within that broader research workflow.

We are also developing [🦎 GeCo Desktop](https://github.com/zacthinks/geco-desktop), a desktop application designed to let researchers create and use GeCo projects **without writing Python or managing Python environments**. GeCo Desktop is still under active development and is not yet a finished, distributable application.

## Getting started

GeCo currently supports **text projects** and requires **Python 3.11 or newer** when used directly. It runs a local browser-based interface; the initial project is created from Python. The Python-free setup described above is the goal of GeCo Desktop, not a prerequisite for using GeCo today.

### 1. Install

Install the current source version from GitHub:

```bash
python -m pip install "git+https://github.com/zacthinks/GeCo.git"
```

This includes GeCo's local interface and the built-in lexical geometry tools. To use sentence-transformer embeddings, install the optional dependencies instead:

```bash
python -m pip install "geometric-coder[embeddings] @ git+https://github.com/zacthinks/GeCo.git"
```

The chosen embedding model is downloaded the first time it is used.

### 2. Create a project

Suppose you have a CSV file, `responses.csv`, with an `id` column, a `text` column, and optionally other columns (which GeCo can retain as metadata). Save the following as `create_project.py`:

```python
import pandas as pd

from geometric_coder import GeometricCoder
from geometric_coder.recipes import raw_tfidf

responses = pd.read_csv("responses.csv")

project = GeometricCoder.create(
    project_dir="responses.geco",
    data=responses,
    keys=["id"],
    text="text",
    geometries={"lexical": raw_tfidf()},
)
```

Run it once:

```bash
python create_project.py
```

This creates a persistent `responses.geco/` project directory, including its initial representation and 2D view. The `id` values must uniquely identify the imported rows. For longer documents, you can supply multiple hierarchical keys instead.

### 3. Open GeCo

```bash
geco launch responses.geco
```

Open the local URL shown in the terminal. You can also launch directly from a Python or Jupyter session with `server = project.launch()` and close it later with `server.close()`.

### 4. Try another perspective

GeCo is designed for comparing representations, not committing to a single one. To create a new project with both lexical and semantic geometries, use:

```python
from geometric_coder.recipes import raw_tfidf, semantic_fast

geometries = {
    "lexical": raw_tfidf(),
    "semantic": semantic_fast(),
}
```

Pass this mapping as `geometries=geometries` when creating the project. The semantic model's weights are downloaded on first use. A project's initially computed geometries are stored for subsequent sessions.

## A few core concepts

| Concept | Meaning in GeCo |
| --- | --- |
| **Observation** | A codable unit of text: an imported row, a contiguous span, or a researcher-authored teaching example. |
| **Geometry** | A computational representation that defines relationships among observations. |
| **View** | A 2D projection of a geometry for interactive exploration. |
| **Topic model** | An exploratory representation of recurring themes or groupings in a corpus, with memberships that can be inspected alongside the original text. |
| **Code** | A researcher-defined concept with an evolving description and explicit Present, Absent, or Unsure assignments. |
| **Classifier** | A code-specific model trained from human judgments in a chosen geometry; it supports active learning and generates proposals for review. |
| **Memo** | A versioned interpretive note that can link back to supporting observations. |

## Research scope and project status

GeCo is an **exploratory** environment. Its maps, classifiers, and proposals support reading, code development, interpretation, and hypothesis generation. They are **not**, on their own, population estimates or a substitute for an inferential research design. Projects that require confirmatory inference should use an appropriate downstream analysis workflow.

**Current status:** GeCo is under active pre-1.0 development (package version 0.9.0). Interfaces, APIs, and project schemas may change; compatibility with older project directories is not guaranteed. The current interface supports text, not image coding.

## Development

To contribute or run the tests from a checkout:

```bash
git clone https://github.com/zacthinks/GeCo.git
cd GeCo
uv sync --dev
uv run pytest
uv run ruff check .
```

## License

GeCo is released under the [MIT License](LICENSE).
