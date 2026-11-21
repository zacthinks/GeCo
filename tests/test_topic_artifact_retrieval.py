"""R2: optional keyed membership, ArgMax and description retrieval."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from geometric_coder import GeometricCoder
from geometric_coder.exceptions import ConfigurationError
from geometric_coder.topic_artifacts import ArtifactRows, validate_topic_model_rows


class Provider:
    def __init__(self, *, as_sparse=False, invalid=""):
        self.as_sparse = as_sparse
        self.invalid = invalid
        self.calls = []

    def read_artifact(self, artifact_id, requested_keys):
        self.calls.append((artifact_id, requested_keys))
        if artifact_id == "membership":
            positions = [i for i, key in enumerate(requested_keys) if key["turn"] != 2]
            values = np.array([
                [0, 1] if requested_keys[i]["turn"] == 0 else
                [1, 0] if requested_keys[i]["turn"] == 1 else [0, 0]
                for i in positions
            ], dtype=float).reshape(-1, 2)
            if self.invalid == "bad_positions":
                positions = [0, 0]
            return ArtifactRows("matrix", sparse.csr_matrix(values) if self.as_sparse else values,
                                ["topic_0", "topic_1"], positions)
        if artifact_id == "argmax":
            positions = [i for i, key in enumerate(requested_keys) if key["turn"] != 2]
            if self.invalid == "coverage":
                positions = positions[:-1]
            data = []
            for i in positions:
                turn = requested_keys[i]["turn"]
                label = "topic_1" if turn == 0 else "topic_0" if turn == 1 else None
                if self.invalid == "unknown":
                    label = "topic_99"
                data.append((label, 0 if turn == 3 else 1))
            return ArtifactRows("table",
                                pd.DataFrame(data, columns=["max_feature", "max_value"]),
                                ["max_feature", "max_value"], positions)
        if artifact_id == "description":
            row_ids = [1, 0] if self.invalid == "description" else [0, 1]
            return ArtifactRows("matrix", np.array([[0.9, 0.1], [0.2, 0.8]]),
                                ["apple", "banana"], row_ids=row_ids)
        raise RuntimeError("Unknown artifact")


def make_project(tmp_path: Path, provider):
    project = GeometricCoder.create_external(
        project_dir=tmp_path / "topic.geco",
        data=pd.DataFrame({"interview": ["A"] * 4, "turn": range(4),
                           "text": ["a", "b", "c", "d"]}),
        keys=["interview", "turn"], text="text", metadata=[],
        external_provider=provider,
    )
    project.register_topic_model(name="T", membership_artifact_id="membership",
                                 dominant_artifact_id="argmax",
                                 description_artifact_id="description",
                                 topic_labels={"topic_0": "Apple"})
    return project


@pytest.mark.parametrize("as_sparse", [False, True])
def test_matrix_argmax_and_explicit_coverage(tmp_path, as_sparse):
    provider = Provider(as_sparse=as_sparse)
    project = make_project(tmp_path, provider)
    rows = project.topic_model_rows("T")
    assert rows.topic_ids == ("topic_0", "topic_1")
    assert rows.matched_positions == (0, 1, 3)
    assert rows.memberships.shape == (3, 2)
    assert sparse.issparse(rows.memberships) is as_sparse
    assert rows.dominant["max_feature"].tolist()[:2] == ["topic_1", "topic_0"]
    assert pd.isna(rows.dominant["max_feature"].iloc[2])
    assert rows.topic_labels == {"topic_0": "Apple"}
    assert rows.descriptions is None


def test_subsets_and_description(tmp_path):
    provider = Provider()
    project = make_project(tmp_path, provider)
    rows = project.topic_model_rows(1, positions=[3, 2, 1], include_description=True)
    assert rows.matched_positions == (0, 2)
    assert pd.isna(rows.dominant["max_feature"].iloc[0])
    assert rows.descriptions is not None
    assert rows.descriptions.row_ids == [0, 1]
    assert provider.calls[-1] == ("description", None)


@pytest.mark.parametrize("invalid,phrase", [
    ("bad_positions", "duplicate key positions|duplicate key"),
    ("coverage", "coverage do not match"),
    ("unknown", "unknown topic component"),
    ("description", "topic IDs do not match"),
])
def test_bad_provider_results(tmp_path, invalid, phrase):
    project = make_project(tmp_path, Provider(invalid=invalid))
    with pytest.raises(ConfigurationError, match=phrase):
        project.topic_model_rows("T", include_description=invalid == "description")


def test_missing_capability_and_invalid_requests(tmp_path):
    project = make_project(tmp_path, object())
    with pytest.raises(ConfigurationError, match="does not support"):
        project.topic_model_rows("T")
    project.external_provider = Provider()
    for positions in ([0, 0], [-1], [4], [True]):
        with pytest.raises(ConfigurationError, match="requested positions"):
            project.topic_model_rows("T", positions=positions)


def test_packet_alignment_with_out_of_order_rows():
    model = {"topic_model_id": 1, "topic_labels": {}}
    membership = ArtifactRows("matrix", np.array([[0., 1.], [1., 0.]]),
                              ["topic_0", "topic_1"], [2, 0])
    argmax = ArtifactRows("table", pd.DataFrame({
        "max_feature": ["topic_0", "topic_1"], "max_value": [1., 1.]}),
        ["max_feature", "max_value"], [0, 2])
    result = validate_topic_model_rows(model, membership, argmax, requested_count=3)
    assert result.matched_positions == (0, 2)
    assert result.memberships.tolist() == [[1., 0.], [0., 1.]]


def test_invalid_scores():
    model = {"topic_model_id": 1, "topic_labels": {}}
    membership = ArtifactRows("matrix", np.array([[1., 0.]]),
                              ["topic_0", "topic_1"], [0])
    argmax = ArtifactRows("table", pd.DataFrame({
        "max_feature": [None], "max_value": [1.]}),
        ["max_feature", "max_value"], [0])
    with pytest.raises(ConfigurationError, match="Unassigned"):
        validate_topic_model_rows(model, membership, argmax, requested_count=1)


def test_bounded_topic_page_never_reads_full_corpus(tmp_path, monkeypatch):
    """Page and selector requests must query only their requested stable keys."""
    provider = Provider()
    project = make_project(tmp_path, provider)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Topic Model page attempted full-corpus unit loading")

    monkeypatch.setattr(project, "_ordered_user_keys", forbidden)
    monkeypatch.setattr(project.database, "list_units", forbidden)
    expected = [{"interview": "A", "turn": 3}, {"interview": "A", "turn": 1}]
    rows = project.topic_model_rows("T", positions=[3, 1])
    assert rows.matched_positions == (0, 1)
    assert provider.calls == [
        ("membership", expected),
        ("argmax", expected),
    ]
    provider.calls.clear()
    options = project.topic_model_rows("T", positions=[0])
    assert options.topic_ids == ("topic_0", "topic_1")
    assert provider.calls == [
        ("membership", [{"interview": "A", "turn": 0}]),
        ("argmax", [{"interview": "A", "turn": 0}]),
    ]


def test_empty_topic_page_avoids_full_universe(tmp_path, monkeypatch):
    project = make_project(tmp_path, Provider())
    monkeypatch.setattr(
        project, "_ordered_user_keys",
        lambda: (_ for _ in ()).throw(AssertionError("eager corpus load")),
    )
    rows = project.topic_model_rows("T", positions=[])
    assert rows.matched_positions == ()
    assert rows.memberships.shape == (0, 2)


@pytest.mark.parametrize("as_sparse", [False, True])
@pytest.mark.parametrize("include_description", [False, True])
@pytest.mark.parametrize("positions", [[3, 2, 1], [0, 1, 3], []])
def test_batched_topic_provider_matches_legacy_results(
    tmp_path, as_sparse, include_description, positions
):
    """Both capabilities preserve partial coverage, order, labels and matrix format."""
    legacy = Provider(as_sparse=as_sparse)

    class BatchedProvider(Provider):
        def __init__(self):
            super().__init__(as_sparse=as_sparse)
            self.batches = []

        def read_artifacts(self, requests):
            self.batches.append(list(requests))
            # Invoke inherited implementation without exposing individual
            # provider calls as an available public operation.
            return [Provider.read_artifact(self, artifact_id, keys)
                    for artifact_id, keys in requests]

        def read_artifact(self, artifact_id, requested_keys):
            raise AssertionError("Single-artifact path used despite batch capability")

    batch = BatchedProvider()
    expected = make_project(tmp_path / "legacy", legacy).topic_model_rows(
        "T", positions=positions, include_description=include_description
    )
    actual = make_project(tmp_path / "batched", batch).topic_model_rows(
        "T", positions=positions, include_description=include_description
    )
    assert len(batch.batches) == 1
    keys = [{"interview": "A", "turn": p} for p in positions]
    assert batch.batches == [[
        ("membership", keys),
        ("argmax", keys),
        *([("description", None)] if include_description else []),
    ]]
    assert len(legacy.calls) == (3 if include_description else 2)
    assert batch.calls == legacy.calls
    assert actual.topic_model_id == expected.topic_model_id
    assert actual.topic_ids == expected.topic_ids
    assert actual.matched_positions == expected.matched_positions
    assert actual.topic_labels == expected.topic_labels
    assert sparse.issparse(actual.memberships) is as_sparse
    if as_sparse:
        np.testing.assert_array_equal(actual.memberships.toarray(), expected.memberships.toarray())
    else:
        np.testing.assert_array_equal(actual.memberships, expected.memberships)
    pd.testing.assert_frame_equal(actual.dominant, expected.dominant, check_exact=True)
    if include_description:
        assert actual.descriptions is not None and expected.descriptions is not None
        assert actual.descriptions.row_ids == expected.descriptions.row_ids
        np.testing.assert_array_equal(actual.descriptions.values, expected.descriptions.values)
    else:
        assert actual.descriptions is None


def test_batch_capability_is_optional_and_errors_do_not_trigger_fallback(tmp_path):
    class NonCallableBatch(Provider):
        read_artifacts = None

    old = NonCallableBatch()
    rows = make_project(tmp_path / "old", old).topic_model_rows("T")
    assert rows.topic_ids == ("topic_0", "topic_1")
    assert [call[0] for call in old.calls] == ["membership", "argmax"]

    class BatchOnly(Provider):
        read_artifact = None

        def __init__(self):
            super().__init__()
            self.batches = []

        def read_artifacts(self, requests):
            self.batches.append(requests)
            return [Provider.read_artifact(self, *request) for request in requests]

    batch_only = BatchOnly()
    rows = make_project(tmp_path / "batch_only", batch_only).topic_model_rows("T")
    assert rows.matched_positions == (0, 1, 3)
    assert len(batch_only.batches) == 1

    class BrokenBatch(Provider):
        def read_artifacts(self, requests):
            raise RuntimeError("real provider failure")

    broken = BrokenBatch()
    with pytest.raises(RuntimeError, match="real provider failure"):
        make_project(tmp_path / "broken", broken).topic_model_rows("T")
    assert broken.calls == []


@pytest.mark.parametrize("count", [0, 1, 4])
def test_batch_packet_count_must_match_requested_artifacts(tmp_path, count):
    class IncorrectBatch(Provider):
        def read_artifacts(self, requests):
            return [Provider.read_artifact(self, *request)
                    for request in requests[:2]][:count] + (
                        [Provider.read_artifact(self, *requests[0])] * max(0, count - 2)
                    )

    project = make_project(tmp_path, IncorrectBatch())
    with pytest.raises(ConfigurationError, match="wrong number of artifacts"):
        project.topic_model_rows("T", include_description=True)


@pytest.mark.parametrize("invalid,phrase", [
    ("bad_positions", "duplicate key positions|duplicate key"),
    ("coverage", "coverage do not match"),
    ("unknown", "unknown topic component"),
    ("description", "topic IDs do not match"),
])
def test_batch_results_preserve_strict_integrity_errors(tmp_path, invalid, phrase):
    class InvalidBatch(Provider):
        def read_artifacts(self, requests):
            return [Provider.read_artifact(self, *request) for request in requests]

    provider = InvalidBatch(invalid=invalid)
    project = make_project(tmp_path, provider)
    with pytest.raises(ConfigurationError, match=phrase):
        project.topic_model_rows("T", include_description=invalid == "description")


def test_batched_topic_page_retrieval_does_not_eagerly_load_corpus(tmp_path, monkeypatch):
    class BatchedProvider(Provider):
        def __init__(self):
            super().__init__()
            self.batches = []

        def read_artifacts(self, requests):
            self.batches.append(requests)
            return [Provider.read_artifact(self, *request) for request in requests]

    provider = BatchedProvider()
    project = make_project(tmp_path, provider)
    monkeypatch.setattr(project, "_ordered_user_keys", lambda: (_ for _ in ()).throw(
        AssertionError("full-corpus key materialization")
    ))
    monkeypatch.setattr(project.database, "list_units", lambda: (_ for _ in ()).throw(
        AssertionError("full-corpus unit materialization")
    ))
    result = project.topic_model_rows("T", positions=[3, 1], include_description=True)
    assert result.matched_positions == (0, 1)
    assert provider.batches == [[
        ("membership", [{"interview": "A", "turn": 3},
                        {"interview": "A", "turn": 1}]),
        ("argmax", [{"interview": "A", "turn": 3},
                    {"interview": "A", "turn": 1}]),
        ("description", None),
    ]]
