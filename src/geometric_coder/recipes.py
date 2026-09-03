"""Inspectable convenience recipes for common GeCo geometries."""

from __future__ import annotations

from geometric_coder.geometry import CountGeometry, SVDGeometry, SentenceTransformerGeometry


def raw_tfidf() -> CountGeometry:
    """Lowercased surface-form unigram TF-IDF."""
    return CountGeometry(
        lemmatize=False,
        lowercase=True,
        weighting="tfidf",
        ngram_range=(1, 1),
    )


def lemma_tfidf(*, ngram_range: tuple[int, int] = (1, 1)) -> CountGeometry:
    """Lowercased, lemmatized TF-IDF using spaCy."""
    return CountGeometry(
        lemmatize=True,
        lowercase=True,
        remove_punctuation=True,
        weighting="tfidf",
        ngram_range=ngram_range,
    )


def lemma_binary(*, ngram_range: tuple[int, int] = (1, 1)) -> CountGeometry:
    """Lowercased, lemmatized binary lexical occurrence."""
    return CountGeometry(
        lemmatize=True,
        lowercase=True,
        remove_punctuation=True,
        weighting="binary",
        binary=True,
        ngram_range=ngram_range,
    )


def lemma_tfidf_svd(*, n_components: int = 100) -> dict[str, object]:
    """Expose both a lemmatized TF-IDF geometry and an LSA/SVD derivative."""
    tfidf = lemma_tfidf()
    svd = SVDGeometry(source=tfidf, n_components=n_components)
    return {
        "tfidf": tfidf,
        f"svd_{n_components}": svd,
    }


def semantic_fast() -> SentenceTransformerGeometry:
    """A common lightweight English SentenceTransformer model."""
    return SentenceTransformerGeometry("sentence-transformers/all-MiniLM-L6-v2")


def semantic_english() -> SentenceTransformerGeometry:
    """A common higher-quality general-purpose English model."""
    return SentenceTransformerGeometry("sentence-transformers/all-mpnet-base-v2")


def semantic_multilingual() -> SentenceTransformerGeometry:
    """A common multilingual SentenceTransformer model."""
    return SentenceTransformerGeometry(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
