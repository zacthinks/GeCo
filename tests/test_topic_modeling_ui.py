"""V1 Topic Modeling view-model tests: no Dash server or TeAL fitting required."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from geometric_coder.exceptions import ConfigurationError
from geometric_coder.topic_artifacts import ArtifactRows, TopicModelRows
from geometric_coder.ui.topic_modeling import topic_inspection, topic_name


def _rows(*, coverage=True, unassigned=False, sparse_membership=False, descriptors=True):
    weights = np.array([[0.0, 0.0] if unassigned else [0.3, 0.7]])
    if not coverage:
        weights = weights[:0]
    dominant = pd.DataFrame(
        {
            "max_feature": [None if unassigned else "topic_1"] if coverage else [],
            "max_value": [0.0 if unassigned else 0.7] if coverage else [],
        }
    )
    desc = (
        ArtifactRows(
            "matrix", np.array([[0.8, 0.2], [0.1, 0.9]]),
            ["apple", "banana"], row_ids=[0, 1]
        )
        if descriptors else None
    )
    return TopicModelRows(
        topic_model_id=1,
        topic_ids=("topic_0", "topic_1"),
        matched_positions=(0,) if coverage else (),
        memberships=sparse.csr_matrix(weights) if sparse_membership else weights,
        dominant=dominant,
        descriptions=desc,
        topic_labels={"topic_1": "Research label"},
    )


@pytest.mark.parametrize("sparse_membership", [False, True])
def test_focal_membership_keeps_original_topic_ids(sparse_membership):
    data = topic_inspection(_rows(sparse_membership=sparse_membership))
    assert data["coverage"] == "assigned"
    assert data["dominant"] == "topic_1"
    assert data["max_value"] == pytest.approx(0.7)
    assert data["topics"][1] == {
        "id": "topic_1", "name": "Research label (topic_1)", "weight": 0.7
    }
    assert data["selected_component"] == "topic_0"
    assert data["descriptors"][0]["name"] == "apple"


def test_unassigned_differs_from_outside_coverage():
    unassigned = topic_inspection(_rows(unassigned=True))
    outside = topic_inspection(_rows(coverage=False))
    assert unassigned["coverage"] == "unassigned"
    assert unassigned["topics"][0]["weight"] == 0
    assert unassigned["dominant"] is None
    assert outside["coverage"] == "outside"
    assert outside["topics"][0]["weight"] is None


def test_descriptions_follow_original_component_axis_not_local_label():
    data = topic_inspection(_rows(), selected_component="topic_1")
    assert data["descriptors"][0] == {"name": "banana", "weight": 0.9}
    assert topic_name("topic_1", {"topic_1": "Renamed"}) == "Renamed (topic_1)"
    assert topic_name("topic_0", {}) == "topic_0"


def test_no_descriptor_has_no_invented_topic_words():
    assert topic_inspection(_rows(descriptors=False))["descriptors"] is None


def test_inspection_requires_single_focal_key():
    rows = _rows()
    rows = TopicModelRows(
        topic_model_id=rows.topic_model_id, topic_ids=rows.topic_ids,
        matched_positions=(1,), memberships=rows.memberships,
        dominant=rows.dominant, descriptions=rows.descriptions,
        topic_labels=rows.topic_labels,
    )
    with pytest.raises(ConfigurationError, match="one requested observation"):
        topic_inspection(rows)


def test_standard_ui_registers_topic_model_workspace_and_focal_link():
    from pathlib import Path

    app = (Path(__file__).resolve().parents[1] /
           "src/geometric_coder/ui/app.py").read_text(encoding="utf-8")
    assert 'label="Topic Modeling"' not in app
    assert "register_topic_details_callbacks(" in app
    assert 'id="topic-memberships"' in app
    assert 'id="topic-descriptors"' in app
    assert 'id="topic-model-new-name"' in app
    assert 'id="topic-component-new-label"' in app
    assert 'dcc.Store(id="topic-model-refresh-store", data=0)' in app
