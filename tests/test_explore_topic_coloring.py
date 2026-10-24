"""Explore topic color contracts independent of web server and TeAL installation."""
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from geometric_coder.topic_artifacts import TopicModelRows
from geometric_coder.ui.topic_explore import (
    UNASSIGNED_COLOR, UNCOVERED_COLOR, component_color,
    normalize_color_mode, topic_point_colors,
)
from geometric_coder.ui.figures import (
    EXPLORE_TRACE_NAMES, build_geometry_figure,
)


def packet(*, sparse_values=False):
    matrix = np.array([[0.2, 0.8], [0.0, 0.0], [0.9, 0.1]])
    return TopicModelRows(
        topic_model_id=1,
        topic_ids=("topic_0", "topic_1"),
        matched_positions=(0, 2, 3),
        memberships=sparse.csr_matrix(matrix) if sparse_values else matrix,
        dominant=pd.DataFrame({
            "max_feature": ["topic_1", None, "topic_0"],
            "max_value": [0.8, 0.0, 0.9],
        }),
        descriptions=None,
        topic_labels={"topic_1": "Interpretation"},
    )


@pytest.mark.parametrize("sparse_values", [False, True])
def test_dominant_uses_argmax_and_distinguishes_missing_and_noise(sparse_values):
    result = topic_point_colors(
        packet(sparse_values=sparse_values),
        mode="dominant", selected_topic=None, requested_count=4,
    )
    assert result.colors == (
        component_color("topic_1"), UNCOVERED_COLOR,
        UNASSIGNED_COLOR, component_color("topic_0"),
    )
    assert "Interpretation (topic_1)" in [label for _, label in result.legend]


@pytest.mark.parametrize("sparse_values", [False, True])
def test_selected_topic_values_and_missing_noise(sparse_values):
    result = topic_point_colors(
        packet(sparse_values=sparse_values),
        mode="intensity", selected_topic="topic_1", requested_count=4,
    )
    assert result.colors[1] == UNCOVERED_COLOR
    assert result.colors[2] == UNASSIGNED_COLOR
    assert result.colors[0] != result.colors[3]
    assert len(result.colors) == 4


def test_invalid_selection_and_session_mode():
    assert normalize_color_mode("unknown") == "plain"
    with pytest.raises(ValueError, match="valid topic"):
        topic_point_colors(packet(), mode="intensity", selected_topic="missing", requested_count=4)


def test_map_overlay_contract_and_topic_semantic_exclusivity():
    units = [{"unit_id": i, "user_key": {"row": i}} for i in range(1, 5)]
    kwargs = dict(
        coordinates=np.column_stack([np.arange(4), np.arange(4)]),
        units=units, eligible=[True, False, True, True],
        seen_unit_ids={1, 3}, focal_unit_id=4,
    )
    colors = ["#ff0000", "#aabbcc", "#00ff00", "#0000ff"]
    figure = build_geometry_figure(**kwargs, topic_colors=colors)
    assert tuple(trace.name for trace in figure.data) == EXPLORE_TRACE_NAMES
    assert figure.data[1].marker.color == (colors[0], colors[2], colors[3])
    assert list(figure.data[2].customdata) == [1, 3]
    assert list(figure.data[3].customdata) == [4]
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_geometry_figure(**kwargs, topic_colors=colors, semantic_scores=[0.1] * 4)


def test_explore_only_no_new_tab():
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / "src/geometric_coder/ui/app.py").read_text()
    assert 'id="topic-color-mode"' in source
    assert 'label="Topic Modeling"' not in source
    assert '"topic_color_mode": normalize_color_mode(topic_color_mode)' in source

def test_layout_helper_resolves_at_module_scope():
    from geometric_coder.ui import app as app_module
    assert app_module.normalize_color_mode("dominant") == "dominant"
    assert callable(app_module._explore_layout)
