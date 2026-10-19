"""Reproducible DBpedia-to-Wikipedia sentence corpus and GeCo baseline workflow.

This is an *authoring* utility, not run by visitors or on Cloud Run requests.
Collect Wikipedia lead paragraphs, create a real editable GeCo workspace, then
freeze its complete project directory after editing codes/memos/classifiers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

CLASSES = (
    "EducationalInstitution",
    "Artist",
    "Athlete",
    "Building",
    "Plant",
)
COLUMNS = (
    "document_id", "sentence_id", "text", "title", "dbpedia_class",
    "source_url", "revision_id",
)
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "GeCoDemoCorpus/0.1 (https://github.com/zacthinks/GeCo)"
WORKBENCH = Path("demo_workbench")


def first_paragraph(extract: str) -> str:
    """The first nonblank paragraph, not the entire pre-heading introduction."""
    paragraphs = re.split(r"\n\s*\n", extract.replace("\r\n", "\n").strip())
    return next((p.strip().replace("\n", " ") for p in paragraphs if p.strip()), "")


def sentenceize(paragraph: str) -> list[str]:
    """Sentence segmentation without downloading a language model."""
    import spacy

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    return [sent.text.strip() for sent in nlp(paragraph).sents if sent.text.strip()]


def fetch_wikipedia(title: str, *, pause: float = 0.5) -> dict:
    """Retrieve article lead and revision metadata; follow Wikimedia throttling."""
    params = urlencode({
        "action": "query", "format": "json", "formatversion": "2",
        "redirects": "1", "prop": "extracts|revisions",
        "rvprop": "ids", "explaintext": "1", "exintro": "1", "titles": title,
    })
    request = Request(
        WIKIPEDIA_API + "?" + params,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    for attempt in range(6):
        try:
            with urlopen(request, timeout=35) as response:
                data = json.load(response)
            page = data["query"]["pages"][0]
            if page.get("missing") or not page.get("extract"):
                return {"title": title, "error": "missing or empty"}
            return {
                "title": title,
                "resolved_title": page["title"],
                "page_id": page.get("pageid"),
                "revision_id": (page.get("revisions") or [{}])[0].get("revid"),
                "source_url": "https://en.wikipedia.org/wiki/" +
                __import__("urllib.parse", fromlist=["quote"]).quote(
                    page["title"].replace(" ", "_"), safe="()_"
                ),
                "paragraph": first_paragraph(page["extract"]),
            }
        except HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 5:
                raise
            delay = min(60.0, float(error.headers.get("Retry-After", 2 ** attempt)))
        except (URLError, TimeoutError):
            if attempt == 5:
                raise
            delay = min(60.0, 2 ** attempt)
        time.sleep(delay)
    raise RuntimeError(f"Unable to retrieve {title}")


def _read_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    cache = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        cache[record["title"]] = record
    return cache


def collect(args: argparse.Namespace) -> None:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit("Install authoring dependency: uv pip install datasets") from error
    import numpy as np

    # This matches the source and five categories used in ITM's DBpedia vignette.
    dataset = load_dataset("fancyzhx/dbpedia_14", split="train")
    names = dataset.features["label"].names
    labels = dataset["label"]
    rng = np.random.default_rng(args.seed)
    selected: dict[str, list[int]] = {}
    for category in CLASSES:
        label_id = names.index(category)
        candidates = [index for index, label in enumerate(labels) if label == label_id]
        # First 100 per class (when eligible) follow the old seed-42 permutation.
        selected[category] = [
            candidates[i] for i in rng.permutation(len(candidates))[: args.candidates]
        ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache_path = args.output.with_suffix(".articles.jsonl")
    cache = _read_cache(cache_path)
    result = []
    used_page_ids: set[int] = set()
    with cache_path.open("a", encoding="utf-8") as cache_file:
        for category in CLASSES:
            accepted = 0
            for index in selected[category]:
                if accepted == args.articles_per_class:
                    break
                title = str(dataset[int(index)]["title"]).strip()
                if not title:
                    continue
                if title not in cache:
                    article = fetch_wikipedia(title)
                    cache[title] = article
                    cache_file.write(json.dumps(article, ensure_ascii=False) + "\n")
                    cache_file.flush()
                    time.sleep(args.pause)
                article = cache[title]
                paragraph = article.get("paragraph", "")
                sentences = sentenceize(paragraph) if paragraph else []
                if len(sentences) < 2 or len(paragraph) < 100:
                    continue
                page_id = article.get("page_id")
                if page_id in used_page_ids:
                    continue
                used_page_ids.add(page_id)
                accepted += 1
                document_id = len(used_page_ids)
                for sentence_id, sentence in enumerate(sentences, start=1):
                    result.append({
                        "document_id": document_id, "sentence_id": sentence_id,
                        "text": sentence,
                        "title": article["resolved_title"],
                        "dbpedia_class": category,
                        "source_url": article["source_url"],
                        "revision_id": article.get("revision_id") or "",
                    })
            if accepted < args.articles_per_class:
                raise RuntimeError(
                    f"Only {accepted}/{args.articles_per_class} eligible "
                    f"Wikipedia articles for {category}; raise --candidates "
                    "and retry using the existing cache."
                )
            print(f"{category}: {accepted} articles")

    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(result)
    manifest = {
        "source": "fancyzhx/dbpedia_14 train, article titles",
        "wikipedia_source": "English Wikipedia TextExtracts lead, first paragraph only",
        "categories": CLASSES,
        "seed": args.seed,
        "articles_per_class": args.articles_per_class,
        "articles": len(used_page_ids),
        "sentences": len(result),
        "license_note": "Wikipedia text is CC BY-SA; retain article URLs and revisions.",
        "columns": COLUMNS,
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(result):,} sentences across {len(used_page_ids)} articles to {args.output}")


def build(args: argparse.Namespace) -> None:
    import pandas as pd
    from geometric_coder import GeometricCoder
    from geometric_coder.recipes import raw_tfidf, semantic_fast

    frame = pd.read_csv(args.csv).fillna("")
    expected = set(COLUMNS)
    if not expected.issubset(frame.columns):
        raise ValueError(f"Missing CSV columns: {sorted(expected - set(frame.columns))}")
    if frame.duplicated(subset=["document_id", "sentence_id"]).any():
        raise ValueError("Duplicate sentence keys")
    if frame["document_id"].nunique() < 5 or len(frame) < 10:
        raise ValueError("The corpus is too small")
    frame = frame.sort_values(["document_id", "sentence_id"], kind="stable")
    geometries = {"lexical": raw_tfidf()}
    if not args.no_semantic:
        geometries["semantic"] = semantic_fast()
    project = GeometricCoder.create(
        project_dir=args.output, data=frame,
        keys=["document_id", "sentence_id"], text="text",
        modality="text", geometries=geometries,
        metadata=["title", "dbpedia_class", "source_url", "revision_id"],
        overwrite=args.overwrite,
    )
    print(f"Created editable project: {project.project_dir}")
    print("Open it using GeometricCoder.open(project_dir).launch();")
    print("Create your codes, memos and classifier fits; then close GeCo before freezing.")


def freeze(args: argparse.Namespace) -> None:
    """Freeze a consistent SQLite backup *and* every numerical/model sidecar."""
    from geometric_coder import GeometricCoder

    source = args.project.resolve()
    destination = args.output.resolve()
    if source == destination or not (source / "project.sqlite3").is_file():
        raise ValueError("Source must be an existing, different GeCo project directory")
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"Already exists: {destination}; use --overwrite")
    project = GeometricCoder.open(source)
    print(
        f"Freezing {len(project.units())} atomic units, {len(project.codes())} "
        f"codes, and {len(project.memos())} memos"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="geco-freeze-", dir=destination.parent) as tmp:
        staged = Path(tmp) / "project.geco"
        shutil.copytree(
            source, staged,
            ignore=shutil.ignore_patterns(
                "project.sqlite3", "project.sqlite3-wal", "project.sqlite3-shm"
            ),
        )
        # SQLite backup protects against copying a partially written database.
        with sqlite3.connect(f"file:{source / 'project.sqlite3'}?mode=ro", uri=True) as src:
            with sqlite3.connect(staged / "project.sqlite3") as dst:
                src.backup(dst)
                if dst.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise RuntimeError("Frozen SQLite database failed integrity check")
        reopened = GeometricCoder.open(staged)
        if len(reopened.units()) != len(project.units()):
            raise RuntimeError("Frozen project's observation count differs")
        hashes = {}
        for path in sorted(staged.rglob("*")):
            if path.is_file():
                hashes[str(path.relative_to(staged))] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
        (staged / "demo_freeze_manifest.json").write_text(
            json.dumps({
                "schema": "geco-demo-freeze-v1",
                "atomic_units": len(reopened.units()),
                "codes": len(reopened.codes()),
                "memos": len(reopened.memos()),
                "sha256": hashes,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(staged), str(destination))
    print(f"Frozen baseline: {destination}")
    print("Keep the complete project directory (database + artifacts); do not publish DB alone.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect_cmd = commands.add_parser("collect", help="Download DBpedia article leads")
    collect_cmd.add_argument("--output", type=Path, default=WORKBENCH / "sentences.csv")
    collect_cmd.add_argument("--seed", type=int, default=42)
    collect_cmd.add_argument("--articles-per-class", type=int, default=100)
    collect_cmd.add_argument("--candidates", type=int, default=250)
    collect_cmd.add_argument("--pause", type=float, default=0.5)
    collect_cmd.set_defaults(run=collect)
    build_cmd = commands.add_parser("build", help="Create editable project from sentences")
    build_cmd.add_argument("--csv", type=Path, default=WORKBENCH / "sentences.csv")
    build_cmd.add_argument("--output", type=Path, default=WORKBENCH / "authoring.geco")
    build_cmd.add_argument("--no-semantic", action="store_true")
    build_cmd.add_argument("--overwrite", action="store_true")
    build_cmd.set_defaults(run=build)
    freeze_cmd = commands.add_parser("freeze", help="Copy full authored project as baseline")
    freeze_cmd.add_argument("--project", type=Path, default=WORKBENCH / "authoring.geco")
    freeze_cmd.add_argument("--output", type=Path, default=Path("demo_data/starter.geco"))
    freeze_cmd.add_argument("--overwrite", action="store_true")
    freeze_cmd.set_defaults(run=freeze)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
