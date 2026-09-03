"""Count-, binary-, and TF-IDF-based text geometries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import spacy
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from geometric_coder.exceptions import ConfigurationError
from geometric_coder.geometry.base import Geometry, Matrix, ViewSpec

Weighting = Literal["count", "binary", "tfidf"]


@dataclass(frozen=True, slots=True)
class CountOptions:
    """Conventional lexical-vectorizer options exposed by GeCo."""

    spacy_model: str = "en_core_web_sm"
    lowercase: bool = True
    lemmatize: bool = True
    remove_stopwords: bool = False
    remove_punctuation: bool = True
    remove_numbers: bool = False
    alpha_only: bool = False
    min_token_length: int = 1
    ngram_range: tuple[int, int] = (1, 1)
    weighting: Weighting = "tfidf"
    binary: bool = False
    norm: str | None = "l2"
    use_idf: bool = True
    sublinear_tf: bool = False
    min_df: int | float = 1
    max_df: int | float = 1.0
    max_features: int | None = None


class CountGeometry(Geometry):
    """A spaCy-tokenized lexical geometry."""

    default_view = ViewSpec("truncated_svd", {"n_components": 2, "random_state": 0})

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.options = CountOptions(**kwargs)
        self._vectorizer: CountVectorizer | TfidfVectorizer | None = None
        self._nlp: Any | None = None

    @property
    def supports_query(self) -> bool:
        return self._vectorizer is not None

    def validate_runtime(self) -> None:
        """Load the configured spaCy pipeline before expensive project work.

        Surface-form geometries can always fall back to ``spacy.blank("en")``.
        Lemmatized geometries require a trained pipeline and fail immediately
        with an actionable installation command when it is unavailable.
        """
        self._load_spacy()

    def _load_spacy(self) -> Any:
        if self._nlp is not None:
            return self._nlp
        try:
            nlp = spacy.load(
                self.options.spacy_model,
                disable=["parser", "ner", "textcat"],
            )
        except OSError:
            if self.options.lemmatize:
                raise ConfigurationError(
                    f"spaCy model {self.options.spacy_model!r} is required for "
                    "lemmatization but is not installed. Install it before creating "
                    "this project with:\n\n"
                    f"    uv run python -m spacy download {self.options.spacy_model}\n\n"
                    "Alternatively, construct the CountGeometry with "
                    "lemmatize=False to use a blank English tokenizer."
                ) from None
            nlp = spacy.blank("en")
        self._nlp = nlp
        return nlp

    def _analyze(self, text: str) -> list[str]:
        nlp = self._load_spacy()
        tokens: list[str] = []
        document = nlp(text) if self.options.lemmatize else nlp.make_doc(text)
        for token in document:
            if self.options.remove_punctuation and token.is_punct:
                continue
            if self.options.remove_numbers and token.like_num:
                continue
            if self.options.remove_stopwords and token.is_stop:
                continue
            if self.options.alpha_only and not token.is_alpha:
                continue
            value = token.lemma_ if self.options.lemmatize else token.text
            if self.options.lowercase:
                value = value.lower()
            value = value.strip()
            if len(value) < self.options.min_token_length:
                continue
            if value:
                tokens.append(value)
        return tokens

    def _build_vectorizer(self) -> CountVectorizer | TfidfVectorizer:
        common: dict[str, Any] = {
            "analyzer": "word",
            "tokenizer": self._analyze,
            "preprocessor": None,
            "token_pattern": None,
            "lowercase": False,
            "ngram_range": self.options.ngram_range,
            "min_df": self.options.min_df,
            "max_df": self.options.max_df,
            "max_features": self.options.max_features,
        }
        if self.options.weighting == "tfidf":
            return TfidfVectorizer(
                **common,
                norm=self.options.norm,
                use_idf=self.options.use_idf,
                sublinear_tf=self.options.sublinear_tf,
                binary=self.options.binary,
            )
        return CountVectorizer(
            **common,
            binary=self.options.binary or self.options.weighting == "binary",
        )

    def _fit_transform(self, texts: list[str]) -> Matrix:
        self._vectorizer = self._build_vectorizer()
        matrix = self._vectorizer.fit_transform(texts)
        return sparse.csr_matrix(matrix)

    def transform_texts(self, texts: list[str]) -> Matrix:
        if self._vectorizer is None:
            raise RuntimeError("Count geometry must be fitted before transformation.")
        return sparse.csr_matrix(self._vectorizer.transform(texts))

    def configuration(self) -> dict[str, Any]:
        return {"type": "count", **asdict(self.options)}

    def artifact_state(self) -> dict[str, Any]:
        return {"vectorizer": self._vectorizer}

    def load_artifact_state(self, state: dict[str, Any]) -> None:
        vectorizer = state.get("vectorizer")
        if not isinstance(vectorizer, (CountVectorizer, TfidfVectorizer)):
            raise TypeError("Invalid count-geometry artifact state.")
        self._vectorizer = vectorizer
