"""Create TeAL-backed geometries directly from the linked source text.

GeCo owns the user-facing recipe and geometry registration, never a second
tokenizer, vocabulary, embedding cache, or artifact format.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
import re
from typing import Any, Mapping
from urllib.parse import urlsplit

from geometric_coder.exceptions import ConfigurationError
from geometric_coder.ui.topic_creation import teal_linked_project

# Boolean attributes emitted by TeAL's SpacyTranslator, not arbitrary Python
# expressions. Each can be ignored, required true, or required false.
TOKEN_FLAGS = (
    "is_alpha", "is_ascii", "is_digit", "is_lower", "is_upper", "is_title",
    "is_punct", "is_left_punct", "is_right_punct", "is_space", "is_bracket",
    "is_quote", "is_currency", "is_stop", "is_oov", "like_url", "like_num",
    "like_email", "is_sent_start", "is_sent_end",
)
TOKEN_FIELDS = ("text", "lemma", "norm")
_HF_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class TextGeometryResult:
    geometry_id: int
    artifact_id: str
    replayable: bool


def parse_sentence_transformer_model(value: str) -> str:
    """Accept a Hugging Face repository ID or its page URL, never arbitrary URLs."""
    model = str(value).strip()
    if model.startswith(("https://", "http://")):
        url = urlsplit(model)
        if (url.scheme != "https" or url.netloc.lower() not in
                {"huggingface.co", "www.huggingface.co"} or url.query or url.fragment
                or url.username or url.password):
            raise ValueError("Enter a Hugging Face model page URL or owner/model ID.")
        model = url.path.strip("/")
    if not _HF_ID.fullmatch(model) or ".." in model or model.startswith("."):
        raise ValueError("Enter a valid Hugging Face model ID (owner/model) or model URL.")
    return model


def normalize_token_flags(flags: Mapping[str, str] | None) -> dict[str, bool]:
    """User selections are ignore / true / false, never free-form predicates."""
    result: dict[str, bool] = {}
    for key, setting in (flags or {}).items():
        if key not in TOKEN_FLAGS:
            raise ValueError(f"Unsupported spaCy token attribute: {key!r}.")
        if setting in ("", "any", None):
            continue
        if setting not in ("true", "false"):
            raise ValueError(f"Invalid filter for {key!r}: choose Any, True, or False.")
        result[key] = setting == "true"
    return result


def _keep_tokens(frame: Any, required: Mapping[str, bool]) -> Any:
    """Pure deterministic mask saved as an ordinary TeAL subset callable asset."""
    mask = frame.index == frame.index
    for column, expected in required.items():
        mask = mask & frame[column].eq(expected).fillna(False)
    return mask


def _validate_name(coder: Any, name: str) -> str:
    clean = str(name).strip()
    if not clean:
        raise ValueError("Enter a geometry name.")
    if coder.has_geometry(clean) or coder.has_view(clean):
        raise ConfigurationError(f"Representation {clean!r} is already registered.")
    if coder.database.list_observations(kinds=["span", "teaching_example"]):
        raise ConfigurationError(
            "Creating new geometries requires a workspace without existing spans "
            "or teaching examples until derived-observation backfill is available."
        )
    if teal_linked_project(coder) is None:
        raise ConfigurationError(
            "Creating geometries from text requires a TeAL-linked workspace."
        )
    return clean


def create_text_geometry(
    coder: Any, *, name: str, method: str,
    model: str | None = None,
    revision: str | None = None,
    token_field: str = "text",
    token_flags: Mapping[str, str] | None = None,
    ngram_min: int = 1,
    ngram_max: int = 1,
    min_df: int = 1,
    max_features: int | None = None,
    binary: bool = False,
    normalize: bool = False,
    truncation: str = "error",
) -> TextGeometryResult:
    """Build one complete numerical TeAL artifact and link it as a GeCo geometry.

    TeAL handles text tokenization, model caching, translation and persistence.
    The linking bridge checks key coverage before registration.
    """
    clean = _validate_name(coder, name)
    if method not in {"spacy_dtm", "sentence_transformer"}:
        raise ValueError("Choose spaCy document-term matrix or Sentence Transformer.")
    path = teal_linked_project(coder)
    assert path is not None
    if method == "spacy_dtm":
        if token_field not in TOKEN_FIELDS:
            raise ValueError("Choose token text, lemma or normalized form.")
        required = normalize_token_flags(token_flags)
        if (isinstance(ngram_min, bool) or isinstance(ngram_max, bool)
                or not isinstance(ngram_min, int) or not isinstance(ngram_max, int)
                or not 1 <= ngram_min <= ngram_max <= 5):
            raise ValueError("N-grams must satisfy 1 <= minimum <= maximum <= 5.")
        if isinstance(min_df, bool) or not isinstance(min_df, int) or min_df < 1:
            raise ValueError("Minimum document frequency must be a positive integer.")
        if max_features is not None and (
            isinstance(max_features, bool) or not isinstance(max_features, int)
            or max_features < 1
        ):
            raise ValueError("Maximum features must be a positive integer.")
    else:
        if not model:
            raise ValueError("Enter a Sentence Transformer model ID or Hugging Face URL.")
        model = parse_sentence_transformer_model(model)
        if revision is not None and (not revision.strip() or len(revision) > 160):
            raise ValueError("Model revision must be a nonempty revision string.")
        if truncation not in {"error", "truncate"}:
            raise ValueError("Truncation must be error or truncate.")

    from text_analysis_lab.core.project import Project
    from text_analysis_lab.core.errors import PipelineError

    with Project.open(path) as teal:
        teal.catalog.validate_artifact_alias(clean)
        matches = [
            row for row in teal.geco.list()
            if row.get("mode", "explore") == "explore"
            and Path(str(row["workspace_path"])).resolve() == coder.project_dir.resolve()
        ]
        if len(matches) != 1:
            raise ConfigurationError("Could not uniquely identify the linked TeAL workspace.")
        with teal.geco.open(str(matches[0]["name"])) as linked:
            source = teal.get_artifact(str(linked.manifest["text_artifact_id"]))
            if getattr(source.artifact_type, "value", None) != "table":
                raise ConfigurationError("The linked text source must be a TeAL table.")
            text_field = str(linked.manifest.get("text_field", "text"))
            if text_field not in source.get_data_columns():
                raise ConfigurationError("The linked source text field is missing.")
            if method == "sentence_transformer":
                from text_analysis_lab.translators import SentenceTransformerEncoder
                output = teal.translate(
                    SentenceTransformerEncoder(
                        model=str(model), revision=revision or None,
                        text_field=text_field, normalize=normalize,
                        truncation=truncation, trust_remote_code=False,
                    ),
                    source, workers=1,
                    memo=f"GeCo source-text Sentence Transformer geometry {clean!r}.",
                )["output"]
            else:
                from text_analysis_lab.translators import (
                    ArtifactCountVectorizer, SpacyTranslator,
                )
                parsed = teal.translate(
                    SpacyTranslator(
                        model=model or "en_core_web_sm",
                        text_field=text_field,
                        download_model=True,
                    ),
                    source, workers=1, batch_size=256,
                    memo=f"GeCo spaCy tokens for geometry {clean!r}.",
                )
                tokens = parsed["tokens"]
                if required:
                    tokens = teal.subset(
                        tokens, partial(_keep_tokens, required=required),
                        data_columns=list(required),
                        form="table", iter_batches=True, workers=1,
                        memo=f"GeCo token attributes for geometry {clean!r}.",
                    )["output"]
                output = teal.translate(
                    ArtifactCountVectorizer(
                        field=token_field,
                        group_by=tuple(linked.documents.primary_key),
                        ngram_range=(ngram_min, ngram_max),
                        min_df=min_df, max_features=max_features, binary=binary,
                    ),
                    tokens, workers=1,
                    memo=f"GeCo spaCy document-term matrix {clean!r}.",
                )["output"]
            if getattr(output.artifact_type, "value", None) not in {
                "dense_matrix", "sparse_matrix"
            }:
                raise ConfigurationError("The text recipe did not produce a numerical matrix.")
            # Key coverage is an absolute prerequisite. A token filter can remove
            # all tokens in a document, and TeAL's vectorizer may omit that group.
            if list(output.primary_key) != list(linked.documents.primary_key):
                raise ConfigurationError(
                    "The output key hierarchy does not match the linked observations."
                )
            alias_added = False
            try:
                if clean not in output.aliases:
                    output.add_alias(clean)
                    alias_added = True
                try:
                    teal.pipeline(start=source, end=output)
                except PipelineError:
                    replayable = False
                else:
                    replayable = True
                if replayable:
                    geometry_id = linked.add_geometry(clean, output)
                else:
                    # TeAL's provider validates complete ordered coverage
                    # before we create a navigation-only registration.
                    from text_analysis_lab.integrations.geco import _external_ref
                    keys = linked._manager._document_keys(linked.documents)
                    linked.external_provider.geometry_matrix(_external_ref(output), keys)
                    geometry_id = coder.register_external_geometry(
                        name=clean, external_ref={"artifact_id": str(output.artifact_id)},
                        supports_query=False, supports_text_transform=False,
                    )
            except Exception:
                if alias_added:
                    output.remove_alias(clean)
                raise
            return TextGeometryResult(
                geometry_id=int(geometry_id),
                artifact_id=str(output.artifact_id),
                replayable=bool(replayable),
            )
