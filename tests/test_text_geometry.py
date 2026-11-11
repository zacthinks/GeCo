"""Original-text geometry input, filtering, and UI contract tests."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from geometric_coder.exceptions import ConfigurationError
from geometric_coder.ui import text_geometry as tg


def test_sentence_transformer_model_accepts_hf_id_or_page_only():
    assert tg.parse_sentence_transformer_model(
        "sentence-transformers/all-MiniLM-L6-v2"
    ) == "sentence-transformers/all-MiniLM-L6-v2"
    assert tg.parse_sentence_transformer_model(
        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"
    ) == "sentence-transformers/all-MiniLM-L6-v2"
    for bad in (
        "https://evil.example/sentence-transformers/all-MiniLM-L6-v2",
        "http://huggingface.co/owner/model", "https://huggingface.co@evil.example/a/b",
        "https://huggingface.co/owner/model/tree/main",
        "https://huggingface.co/owner/model?download=1", "https://example.com/payload",
        "owner/../model", "owner", "C:/local/model",
    ):
        with pytest.raises(ValueError):
            tg.parse_sentence_transformer_model(bad)


def test_token_flag_predicate_has_explicit_true_false_and_any():
    flags = tg.normalize_token_flags({
        "is_alpha": "true", "is_stop": "false",
        "like_url": "any", "like_num": "true",
    })
    assert flags == {"is_alpha": True, "is_stop": False, "like_num": True}
    with pytest.raises(ValueError, match="Unsupported"):
        tg.normalize_token_flags({"unknown": "true"})
    with pytest.raises(ValueError, match="Invalid filter"):
        tg.normalize_token_flags({"is_alpha": "maybe"})


def _coder():
    return SimpleNamespace(
        has_geometry=lambda _: False,
        has_view=lambda _: False,
        database=SimpleNamespace(list_observations=lambda **_: []),
    )


def test_bad_source_text_requests_fail_before_teal_project_open(monkeypatch):
    monkeypatch.setattr(tg, "teal_linked_project", lambda _: Path("/tmp/teal"))
    with pytest.raises(ValueError, match="geometry name"):
        tg.create_text_geometry(_coder(), name=" ", method="spacy_dtm")
    with pytest.raises(ValueError, match="Select a spaCy model"):
        tg.create_text_geometry(_coder(), name="DTM", method="spacy_dtm")
    with pytest.raises(ValueError, match="N-grams"):
        tg.create_text_geometry(
            _coder(), name="DTM", method="spacy_dtm",
            model="en_core_web_sm", ngram_min=3, ngram_max=2,
        )
    with pytest.raises(ValueError, match="positive integer"):
        tg.create_text_geometry(
            _coder(), name="DTM", method="spacy_dtm",
            model="en_core_web_sm", min_df=0,
        )
    with pytest.raises(ValueError, match="Hugging Face"):
        tg.create_text_geometry(
            _coder(), name="Embed", method="sentence_transformer",
            model="https://somewhere.example/model",
        )
    with pytest.raises(ValueError, match="model ID"):
        tg.create_text_geometry(_coder(), name="Embed", method="sentence_transformer")
    with pytest.raises(ValueError, match="Truncation"):
        tg.create_text_geometry(
            _coder(), name="Embed", method="sentence_transformer",
            model="owner/model", truncation="other",
        )


def test_spans_and_unlinked_workspaces_fail_preflight(monkeypatch):
    coder = _coder()
    coder.database.list_observations = lambda **_: [{"kind": "span"}]
    monkeypatch.setattr(tg, "teal_linked_project", lambda _: Path("/tmp/teal"))
    with pytest.raises(ConfigurationError, match="backfill"):
        tg.create_text_geometry(coder, name="DTM", method="spacy_dtm")
    coder = _coder()
    monkeypatch.setattr(tg, "teal_linked_project", lambda _: None)
    with pytest.raises(ConfigurationError, match="TeAL-linked"):
        tg.create_text_geometry(coder, name="DTM", method="spacy_dtm")


def test_wizard_exposes_text_source_and_spacy_token_checklists():
    from geometric_coder.ui import representation_ui as ui

    source = Path(ui.__file__).read_text(encoding="utf-8")
    assert 'id="representation-wizard-origin"' in source
    assert '"value": "text"' in source
    assert '"value": "spacy_dtm"' in source
    assert '"value": "sentence_transformer"' in source
    assert 'id="representation-text-model"' in source
    assert '"type": "representation-token-flag"' in source
    assert "create_text_geometry(" in source
    assert "metadata_filters=required" in Path(tg.__file__).read_text(encoding="utf-8")
    assert 'batch_by=linked.documents.primary_key[-1]' in Path(tg.__file__).read_text(encoding="utf-8")
    assert 'tokens, workers=1,' in Path(tg.__file__).read_text(encoding="utf-8")
    assert '{"source": tokens, "target": source}' not in Path(tg.__file__).read_text(encoding="utf-8")
    assert "teal.subset(" not in Path(tg.__file__).read_text(encoding="utf-8")
    assert 'Output("representation-wizard-recipe", "value", allow_duplicate=True)' in source


def test_distinct_spacy_keys_avoid_sentence_level_collisions():
    assert tg._distinct_spacy_keys(["document_id", "sentence_id"]) == (
        "spacy_sentence_id", "token_id",
    )
    assert tg._distinct_spacy_keys([
        "document_id", "sentence_id", "token_id",
        "spacy_sentence_id", "spacy_token_id",
    ]) == ("spacy_sentence_id_2", "spacy_token_id_2")


def test_hf_revision_is_pinned_from_hub_or_existing_commit(monkeypatch):
    import sys
    from types import ModuleType
    fake = ModuleType("huggingface_hub")
    calls = []
    class HfApi:
        def model_info(self, *, repo_id, revision):
            calls.append((repo_id, revision))
            return SimpleNamespace(sha="a" * 40)
    fake.HfApi = HfApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    assert tg._resolve_hf_model_commit("owner/model", "v2") == "a" * 40
    assert calls == [("owner/model", "v2")]
    assert tg._resolve_hf_model_commit("owner/model", "b" * 40) == "b" * 40
    assert calls == [("owner/model", "v2")]


def test_ui_exposes_saved_token_choice_and_compact_radio_attribute_grid():
    from geometric_coder.ui import representation_ui as ui
    source = Path(ui.__file__).read_text(encoding="utf-8")
    assert 'id="representation-text-token-source"' in source
    assert "available_spacy_token_artifacts(project)" in source
    assert 'id="representation-new-tokens-wrap"' in source
    assert 'className="representation-token-grid"' in source
    assert 'className="representation-flag-choice"' in source
    assert "dcc.RadioItems(" in source
    assert "https://spacy.io/models" in source
    assert "Remote custom code is disabled." not in source
