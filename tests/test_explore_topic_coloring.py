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


class _MissingTeALProvider:
    def topic_model_rows(self, model_id, positions):
        raise RuntimeError("Topic artifact 'missing-id' is not available.")


def test_missing_provider_artifact_degrades_to_uncolored_map():
    from geometric_coder.ui.topic_explore import load_topic_page_colors
    palette, message = load_topic_page_colors(
        _MissingTeALProvider(), model_id=1, positions=[0, 2],
        mode="dominant", selected_topic=None,
    )
    assert palette is None
    assert "missing-id" in message
    assert "unavailable" in message
    # Geometry remains independent and renderable despite missing topics.
    fig = build_geometry_figure(
        coordinates=np.asarray([[0.0, 0.0], [1.0, 1.0]]),
        units=[{"unit_id": 1, "user_key": {"id": 1}},
               {"unit_id": 2, "user_key": {"id": 2}}],
        eligible=[True, True], topic_colors=palette,
    )
    assert len(fig.data) == 4


def test_topic_legend_display_includes_alias_and_color():
    from dash import html
    from geometric_coder.ui.topic_explore import topic_legend_content
    data = topic_point_colors(
        packet(), mode="dominant", selected_topic=None, requested_count=4,
    )
    content = topic_legend_content(html, data).to_plotly_json()
    import json
    serialized = json.dumps(content)
    assert "Interpretation (topic_1)" in serialized
    assert component_color("topic_1") in serialized
    assert "Outside Topic Model coverage" in serialized
    assert "Covered, unassigned/noise" in serialized


def test_legacy_semantic_search_session_preserves_coloring():
    assert normalize_color_mode(None, has_semantic_query=True) == "semantic"
    assert normalize_color_mode(None, has_semantic_query=False) == "plain"
    assert normalize_color_mode("plain", has_semantic_query=True) == "plain"
    assert normalize_color_mode("dominant", has_semantic_query=True) == "dominant"


def test_map_callback_has_explore_legend_output_and_provider_boundary():
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / "src/geometric_coder/ui/app.py").read_text()
    assert 'Output("topic-color-legend", "children")' in source
    assert 'id="topic-color-legend"' in source
    assert "load_topic_page_colors(" in source
    assert 'except (ConfigurationError, RuntimeError, ValueError, KeyError)' in source


def test_topic_explore_source_compiles():
    """Explicit syntax guard for the dynamically rendered legend module."""
    from pathlib import Path
    source_path = Path(__file__).resolve().parents[1] / "src/geometric_coder/ui/topic_explore.py"
    compile(source_path.read_text(encoding="utf-8"), str(source_path), "exec")
