"""Create TeAL-backed geometries directly from the linked source text.

GeCo owns the user-facing recipe and geometry registration, never a second
tokenizer, vocabulary, embedding cache, or artifact format.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping
from urllib.parse import urlsplit

from geometric_coder.exceptions import ConfigurationError
from geometric_coder.ui.topic_creation import teal_linked_project

# Boolean attributes emitted by TeAL's SpacyTranslator, not arbitrary Python
# expressions. Each can be ignored, required true, or required false.
TOKEN_FLAGS = (
    "is_placeholder", "is_alpha", "is_ascii", "is_digit", "is_lower", "is_upper", "is_title",
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


def _distinct_spacy_keys(names: list[str]) -> tuple[str, str]:
    """Choose sentence/token extension keys that do not collide with input data."""
    used = set(names)
    result = []
    for preferred in ("sentence_id", "token_id"):
        name = preferred
        if name in used:
            name = "spacy_" + preferred
        suffix = 2
        while name in used:
            name = f"spacy_{preferred}_{suffix}"
            suffix += 1
        used.add(name)
        result.append(name)
    return result[0], result[1]


def _resolve_hf_model_commit(model: str, revision: str | None) -> str:
    """Resolve a precise Hub commit for a reproducible frozen model."""
    if revision and re.fullmatch(r"[a-fA-F0-9]{40}", revision):
        return revision.lower()
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo_id=model, revision=revision or "main")
        commit = str(getattr(info, "sha", "") or "")
        if not re.fullmatch(r"[a-fA-F0-9]{40}", commit):
            raise ValueError("Hugging Face did not return a valid commit SHA.")
        return commit.lower()
    except Exception as exc:
        raise ConfigurationError(
            f"Could not resolve a reproducible revision for {model!r}. "
            "Check the model ID, revision, and network connection."
        ) from exc


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


def _linked_text_source(teal: Any, coder: Any) -> tuple[Any, Any]:
    """Locate the TeAL link and its declared text source, never a shape-based guess."""
    matches = [
        row for row in teal.geco.list()
        if row.get("mode", "explore") == "explore"
        and Path(str(row["workspace_path"])).resolve() == coder.project_dir.resolve()
    ]
    if len(matches) != 1:
        raise ConfigurationError("Could not uniquely identify the linked TeAL workspace.")
    linked = teal.geco.open(str(matches[0]["name"]))
    source = teal.get_artifact(str(linked.manifest["text_artifact_id"]))
    return linked, source


def _compatible_spacy_tokens(
    teal: Any, source: Any, *, text_field: str,
) -> list[dict[str, str]]:
    """Discover directly sourced, frozen, replayable spaCy token artifacts.

    We deliberately don't accept lookalike tables or sibling outputs from a
    document-to-sentences operation: those cannot be replayed from the linked
    sentence text artifact without changing the original tokenization context.
    """
    from text_analysis_lab.translators import SpacyTranslator

    candidates: list[dict[str, str]] = []
    source_key = tuple(source.primary_key)
    for row in teal.list_artifacts(
        artifact_type="table", source_artifact_id=str(source.artifact_id),
    ):
        artifact_id = str(row["artifact_id"])
        try:
            candidate = teal.get_artifact(artifact_id)
            operation = teal.operation_for_artifact(candidate)
            if operation is None or str(operation.get("status")) != "complete":
                continue
            outputs = teal.operation_outputs(str(operation["operation_id"]))
            if not any(
                str(item.get("artifact_id")) == artifact_id
                and str(item.get("output_label")) == "tokens"
                for item in outputs
            ):
                continue
            translator = teal.get_operator(str(operation["operator_id"]))
            if not isinstance(translator, SpacyTranslator):
                continue
            if translator.text_field != text_field:
                continue
            key = tuple(candidate.primary_key)
            if len(key) != len(source_key) + 2 or key[:len(source_key)] != source_key:
                continue
            teal.pipeline(start=source, end=candidate)
            aliases = row.get("aliases") or []
            display_name = str(aliases[0]) if aliases else artifact_id[:12]
            candidates.append({
                "label": f"{display_name} — {translator.model}",
                "value": artifact_id,
            })
        except Exception:
            # An incomplete/deleted or nonreplayable artifact is not offered.
            continue
    return sorted(candidates, key=lambda item: item["label"].casefold())


def available_spacy_token_artifacts(coder: Any) -> list[dict[str, str]]:
    """Options for the wizard, populated on opening from TeAL provenance."""
    path = teal_linked_project(coder)
    if path is None:
        return []
    from text_analysis_lab.core.project import Project
    with Project.open(path) as teal:
        linked, source = _linked_text_source(teal, coder)
        try:
            return _compatible_spacy_tokens(
                teal, source, text_field=str(linked.manifest.get("text_field", "text")),
            )
        finally:
            linked.close()


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
    existing_tokens: str | None = None,
    empty_doc: str = "raise",
    ngram_min: int = 1,
    ngram_max: int = 1,
    min_df: int = 1,
    max_features: int | None = None,
    binary: bool = False,
    normalize: bool = False,
    truncation: str = "error",
) -> TextGeometryResult:
    """Build a TeAL numerical artifact and link it as a GeCo geometry.

    TeAL handles tokenization, model caching, translation and persistence.
    spaCy can emit no token rows for empty inputs; GeCo checks full
    observation coverage before registering a representation.
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
        if not existing_tokens:
            if not model or not str(model).strip():
                raise ValueError(
                    "Select a spaCy model to create new tokens, or choose an existing token artifact."
                )
            if empty_doc not in {"raise", "dummy"}:
                raise ValueError("Choose to raise on empty documents or insert a dummy token.")
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
    with Project.open(path) as teal:
        teal.catalog.validate_artifact_alias(clean)
        linked, source = _linked_text_source(teal, coder)
        with linked:
            if getattr(source.artifact_type, "value", None) != "table":
                raise ConfigurationError("The linked text source must be a TeAL table.")
            text_field = str(linked.manifest.get("text_field", "text"))
            if text_field not in source.get_data_columns():
                raise ConfigurationError("The linked source text field is missing.")
            if method == "sentence_transformer":
                from text_analysis_lab.translators import SentenceTransformerEncoder
                resolved_commit = _resolve_hf_model_commit(str(model), revision)
                output = teal.translate(
                    SentenceTransformerEncoder(
                        model=str(model), revision=resolved_commit,
                        resolved_revision=resolved_commit,
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
                if existing_tokens:
                    valid = {
                        item["value"]
                        for item in _compatible_spacy_tokens(
                            teal, source, text_field=text_field,
                        )
                    }
                    if existing_tokens not in valid:
                        raise ConfigurationError(
                            "That token artifact is no longer available or cannot "
                            "be replayed from this GeCo's linked text source. "
                            "Select a compatible artifact or create new tokens."
                        )
                    tokens = teal.get_artifact(existing_tokens)
                else:
                    sentence_key, token_key = _distinct_spacy_keys(
                        list(source.primary_key) + list(source.get_data_columns())
                    )
                    parsed = teal.translate(
                        SpacyTranslator(
                            model=str(model), text_field=text_field,
                            sentence_key=sentence_key, token_key=token_key,
                            empty_doc=empty_doc, download_model=True,
                            save_model=Path(str(model)).expanduser().is_dir(),
                        ),
                        source, workers=1, batch_size=256,
                        memo=f"GeCo spaCy tokens for geometry {clean!r}.",
                    )
                    tokens = parsed["tokens"]
                output = teal.translate(
                    ArtifactCountVectorizer(
                        field=token_field,
                        group_by=tuple(linked.documents.primary_key),
                        metadata_filters=required,
                        ngram_range=(ngram_min, ngram_max),
                        min_df=min_df, max_features=max_features, binary=binary,
                    ),
                    tokens, workers=1,
                    batch_by=linked.documents.primary_key[-1], batch_size=256,
                    memo=f"GeCo spaCy document-term matrix {clean!r}.",
                )["output"]
            if getattr(output.artifact_type, "value", None) not in {
                "dense_matrix", "sparse_matrix"
            }:
                raise ConfigurationError("The text recipe did not produce a numerical matrix.")
            # TeAL's tokens-only count vectorizer intentionally omits any
            # observation with no original token rows. GeCo presently requires
            # complete matrix coverage for its existing observation universe.
            if list(output.primary_key) != list(linked.documents.primary_key):
                raise ConfigurationError(
                    "The output key hierarchy does not match the linked observations."
                )
            if output.n_rows != linked.documents.n_rows:
                raise ConfigurationError(
                    "The spaCy document-term matrix does not cover every GeCo "
                    "observation. spaCy may emit no tokens for empty documents. "
                    "Filter such observations before creating the linked GeCo "
                    "workspace, then retry."
                )
            alias_added = False
            try:
                if clean not in output.aliases:
                    output.add_alias(clean)
                    alias_added = True
                from text_analysis_lab.core.errors import PipelineError
                try:
                    pipeline = teal.pipeline(start=source, end=output)
                    # A recovered graph is not enough: execute a sample input
                    # through the frozen operators before exposing query search.
                    replay = pipeline.translate(["geco replay probe"], project=teal)["output"]
                    shape = replay["values"].shape
                    if shape != (1, len(output.get_data_columns())):
                        raise ValueError(
                            f"Frozen geometry replay returned {shape}, expected "
                            f"(1, {len(output.get_data_columns())})."
                        )
                except (PipelineError, ValueError) as exc:
                    raise ConfigurationError(
                        f"Geometry {clean!r} cannot replay original text through "
                        "its frozen TeAL pipeline."
                    ) from exc
                geometry_id = linked.add_geometry(clean, output)
            except Exception:
                if alias_added:
                    output.remove_alias(clean)
                raise
            return TextGeometryResult(
                geometry_id=int(geometry_id),
                artifact_id=str(output.artifact_id),
                replayable=True,
            )
