"""Saved Analysis regression tests: persistence, refreshing, and atomic joins."""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from geometric_coder import GeometricCoder
from geometric_coder.analyses import analysis_figure


@pytest.fixture
def project(tmp_path):
    workspace = GeometricCoder.create(
        project_dir=tmp_path / "analyses.geco",
        data=pd.DataFrame({
            "id": [101, 102, 103, 104],
            "text": ["alpha", "beta", "gamma", "delta"],
            "group": ["A", "A", "B", "B"],
        }),
        keys=["id"], text="text", metadata=["group"], geometries=None,
    )
    first = workspace.create_code("Equity")
    second = workspace.create_code("Access")
    observations = [u["observation_id"] for u in workspace.database.list_units()]
    workspace.annotate(observations[0], first, "positive")
    workspace.annotate(observations[1], first, "negative")
    workspace.annotate(observations[2], first, "unsure")
    workspace.annotate(observations[0], second, "positive")
    workspace.annotate(observations[1], second, "positive")
    workspace.annotate(observations[3], second, "positive")
    return workspace, first, second, observations


def _code(code_id, source="labels"):
    return {"type": "code", "code_id": code_id, "source": source, "value": "binary"}


def test_registry_roundtrip_edit_rename_refresh_and_delete(project):
    workspace, equity, access, observations = project
    spec = {"kind": "heatmap", "measure": "count",
            "variables": [_code(equity), _code(access)]}
    analysis_id = workspace.save_analysis("Code matrix", spec)
    assert workspace.analyses()[0]["analysis_id"] == analysis_id
    assert workspace.run_analysis(analysis_id)["values"] == [[1.0, 1.0], [1.0, 3.0]]
    assert workspace.save_analysis("Renamed", spec, analysis_id=analysis_id) == analysis_id
    assert [x["name"] for x in workspace.analyses()] == ["Renamed"]
    reopened = GeometricCoder.open(workspace.project_dir)
    assert reopened.analysis(analysis_id)["name"] == "Renamed"
    # A saved recipe is live, not a cached numerical snapshot.
    reopened.annotate(observations[1], equity, "positive")
    assert reopened.run_analysis(analysis_id)["values"] == [[2.0, 2.0], [2.0, 3.0]]
    reopened.delete_analysis(analysis_id)
    assert reopened.analyses() == []
    assert reopened.current_annotation(observation_id=observations[1], code_id=equity)["value"] == "positive"


def test_jaccard_and_metadata_grouping(project):
    workspace, equity, access, _observations = project
    spec = {"kind": "heatmap", "measure": "jaccard",
            "variables": [_code(equity), _code(access)]}
    matrix = workspace.run_analysis(workspace.save_analysis("Jaccard", spec))
    assert matrix["values"][0][1] == pytest.approx(1 / 3)
    bar = {"kind": "bar", "variables": [
        _code(access), {"type": "metadata", "field": "group"}
    ]}
    result = workspace.run_analysis(workspace.save_analysis("By group", bar))
    assert result["categories"] == ["A", "B"]
    assert result["counts"] == [[2, 1]]
    assert result["denominators"] == [[2, 1]]
    assert len(analysis_figure(result).data) == 1


def test_invalid_analysis_never_mutates_registry(project):
    workspace, equity, access, _ = project
    bad = [
        {"kind": "heatmap", "variables": [_code(equity)]},
        {"kind": "heatmap", "variables": [_code(equity), _code(equity)]},
        {"kind": "bar", "variables": [{"type": "metadata", "field": "group"}]},
        {"kind": "bar", "variables": [_code(99999)]},
        {"kind": "bar", "variables": [_code(equity, "predictions")]},
        {"kind": "heatmap", "variables": [_code(equity), {"type": "metadata", "field": "not_a_field"}]},
    ]
    for spec in bad:
        with pytest.raises(ValueError):
            workspace.save_analysis("Invalid", spec)
    assert workspace.analyses() == []
    with pytest.raises(KeyError):
        workspace.delete_analysis(7899)


def test_schema_17_migration_installs_analysis_registry(project):
    workspace, equity, access, _ = project
    db_path = workspace.project_dir / workspace.DATABASE_FILENAME
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TABLE saved_analyses")
        conn.execute("UPDATE project SET schema_version=17")
    reopened = GeometricCoder.open(workspace.project_dir)
    assert reopened.database.project_metadata()["schema_version"] == 18
    assert reopened.analyses() == []
    analysis_id = reopened.save_analysis(
        "Migrated analysis", {"kind": "bar", "variables": [_code(equity)]}
    )
    assert reopened.run_analysis(analysis_id)["counts"] == [[1]]


def test_prediction_spec_requires_explicit_valid_classifier(project):
    workspace, equity, _access, _observations = project
    with pytest.raises(ValueError, match="Select a classifier"):
        workspace.save_analysis("Needs predictor", {
            "kind": "bar", "variables": [_code(equity, "hybrid")]
        })


def test_predictions_and_human_first_fallback_are_distinct(tmp_path):
    from geometric_coder import CountGeometry
    workspace = GeometricCoder.create(
        project_dir=tmp_path / "predictions.geco",
        data=pd.DataFrame({
            "id": list(range(6)),
            "text": [
                "excellent helpful", "great excellent", "useful clear",
                "bad confusing", "terrible unclear", "weak bad",
            ],
        }),
        keys=["id"], text="text",
        geometries={"lexical": CountGeometry(lemmatize=False, weighting="tfidf")},
    )
    code = workspace.create_code("sentiment")
    workspace.ensure_default_classifier_specs(code_id=code)
    predictor = int(workspace.classifier_specs(code_id=code)[0]["classifier_spec_id"])
    observations = [row["observation_id"] for row in workspace.database.list_units()]
    workspace.annotate(observations[0], code, "positive")
    workspace.annotate(observations[3], code, "negative")
    workspace.train_classifier(code_id=code, classifier_spec_id=predictor)
    fit_id = workspace.database.latest_classifier_fit(
        code_id=code, classifier_spec_id=predictor
    )["classifier_fit_id"]
    predictions = {
        int(row["observation_id"]): row["predicted_label"]
        for row in workspace.database.classifier_predictions(fit_id, atomic_only=True)
    }
    src = lambda mode: {
        "kind": "bar",
        "variables": [{
            "type": "code", "code_id": code, "source": mode,
            "predictor_id": predictor, "value": "binary",
        }],
    }
    predicted = workspace.run_analysis(workspace.save_analysis("Model", src("predictions")))
    assert predicted["counts"] == [[sum(v == "positive" for v in predictions.values())]]
    hybrid_id = workspace.save_analysis("Fallback", src("hybrid"))
    expected = 1 + sum(
        predictions[obs] == "positive"
        for obs in observations[1:3] + observations[4:]
    )
    assert workspace.run_analysis(hybrid_id)["counts"] == [[expected]]
    # Unsure is an explicit human judgment, not a missing label to fill.
    workspace.annotate(observations[1], code, "unsure")
    after = workspace.run_analysis(hybrid_id)
    expected -= int(predictions[observations[1]] == "positive")
    assert after["counts"] == [[expected]]
    assert after["denominators"] == [[len(observations) - 1]]
    # Unsure is not new binary training evidence, so a stale-model warning is not required.
    assert after["warnings"] == []


def test_analysis_variable_choices_refresh_after_new_and_renamed_codes(project):
    from geometric_coder.ui.analyses import variable_options

    workspace, equity, access, _ = project
    initial = variable_options(workspace)
    assert {"label": "Code: Equity", "value": f"code:{equity}"} in initial
    assert {"label": "Metadata: group", "value": "metadata:group"} in initial
    added = workspace.create_code("New theme")
    updated = variable_options(workspace)
    assert {"label": "Code: New theme", "value": f"code:{added}"} in updated
    workspace.update_code(added, name="Renamed theme", description="")
    refreshed = variable_options(workspace)
    assert {"label": "Code: Renamed theme", "value": f"code:{added}"} in refreshed
    assert {"label": "Code: New theme", "value": f"code:{added}"} not in refreshed


def test_analyses_is_top_level_and_code_center_has_no_analysis_tab():
    from pathlib import Path
    from geometric_coder.ui import app as app_module
    from geometric_coder.ui import analyses as analyses_module

    source = Path(app_module.__file__).read_text(encoding="utf-8")
    start = source.index("def _code_center_layout(")
    code_center = source[start:]
    assert 'label="Analyses"' in source
    assert 'value="analyses"' in source
    assert 'label="Analysis", value="analysis"' not in code_center
    assert 'children=analysis_tab(dcc=dcc, html=html)' in source
    callback_source = Path(analyses_module.__file__).read_text(encoding="utf-8")
    assert 'Output("analysis-vars", "options")' in callback_source
    assert 'Input("explore-code-refresh-store", "data")' in callback_source
    assert 'Input("focus-refresh-store", "data")' in callback_source
    assert 'if workspace != "analyses":' in callback_source
    assert 'Input("code-center-view-tabs", "value")' not in callback_source


@pytest.fixture
def turn_project(tmp_path):
    workspace = GeometricCoder.create(
        project_dir=tmp_path / "turns.geco",
        data=pd.DataFrame({
            "interview": [1, 1, 1, 1, 2, 2],
            "turn": [1, 1, 2, 2, 1, 1],
            "sentence": [1, 2, 1, 2, 1, 2],
            "text": ["a", "b", "c", "d", "e", "f"],
            "speaker": ["A", "A", "B", "B", "A", "A"],
            "topic": ["X", "X", "Y", "Y", "X", "X"],
        }),
        keys=["interview", "turn", "sentence"], text="text",
        metadata=["speaker", "topic"], geometries=None,
    )
    code_id = workspace.create_code("Question")
    obs = [u["observation_id"] for u in workspace.database.list_units()]
    for index, label in enumerate(["positive", "negative", "positive",
                                   "positive", "negative", "unsure"]):
        workspace.annotate(obs[index], code_id, label)
    return workspace, code_id


def test_aggregation_to_turns_count_threshold_and_metadata(turn_project):
    workspace, code_id = turn_project
    values = [_code(code_id), {"type": "metadata", "field": "topic"}]
    options = {"kind": "bar", "measure": "count", "variables": values}
    spec = {**options, "aggregation": {"level": "turn", "method": "any"}}
    result = workspace.run_analysis(workspace.save_analysis("Any at turn", spec))
    assert result["categories"] == ["X", "Y"]
    assert result["counts"] == [[1, 1]]
    assert result["denominators"] == [[2, 1]]
    spec["aggregation"] = {"level": "turn", "method": "minimum", "threshold": 2}
    result = workspace.run_analysis(workspace.save_analysis("At least two", spec))
    assert result["counts"] == [[0, 1]]
    assert result["denominators"] == [[2, 1]]


def test_aggregation_rejects_metadata_varying_within_parent(turn_project):
    workspace, code_id = turn_project
    spec = {
        "kind": "bar",
        "variables": [_code(code_id), {"type": "metadata", "field": "speaker"}],
        "aggregation": {"level": "interview", "method": "any"},
    }
    analysis_id = workspace.save_analysis("Invalid grouped speaker", spec)
    with pytest.raises(ValueError, match="metadata vary"):
        workspace.run_analysis(analysis_id)


def test_categorical_one_hot_and_pairwise_correlation(turn_project):
    workspace, code_id = turn_project
    spec = {
        "kind": "heatmap", "measure": "correlation",
        "variables": [_code(code_id), {"type": "metadata", "field": "topic"}],
        "aggregation": {"level": "turn", "method": "any"},
    }
    result = workspace.run_analysis(workspace.save_analysis("Code and topic", spec))
    assert result["names"] == ["Question", "topic = X", "topic = Y"]
    assert result["values"][0][1] == pytest.approx(-0.5)
    assert result["values"][0][2] == pytest.approx(0.5)
    assert result["samples"][0][1] == 3
    assert result["values"][1][2] == pytest.approx(-1)
    fig = analysis_figure(result)
    assert fig.data[0].zmin == -1
    assert fig.data[0].zmax == 1


def test_parent_aggregation_mean_log_and_missing_binary_values(turn_project):
    import math
    workspace, code_id = turn_project
    base = {"kind": "heatmap", "measure": "correlation",
            "variables": [_code(code_id), {"type": "metadata", "field": "topic"}]}
    for method in ("mean", "log_count"):
        config = {**base, "aggregation": {"level": "turn", "method": method}}
        result = workspace.run_analysis(workspace.save_analysis(f"Using {method}", config))
        assert result["samples"][0][1] == 3
        assert result["values"][0][1] is not None
    bad = {**base, "measure": "count",
           "aggregation": {"level": "turn", "method": "mean"}}
    with pytest.raises(ValueError, match="Continuous aggregation"):
        workspace.save_analysis("Bad average count", bad)


def test_correlation_undefined_for_constant_label(turn_project):
    workspace, code_id = turn_project
    spec = {
        "kind": "heatmap", "measure": "correlation",
        "variables": [_code(code_id), {"type": "metadata", "field": "speaker"}],
        "aggregation": {"level": "turn", "method": "minimum", "threshold": 9},
    }
    result = workspace.run_analysis(workspace.save_analysis("Constant column", spec))
    assert result["values"][0][0] is None
    assert result["values"][0][1] is None


def test_metadata_only_heatmap_encodes_categories(turn_project):
    workspace, _code_id = turn_project
    spec = {
        "kind": "heatmap", "measure": "count",
        "variables": [{"type": "metadata", "field": "speaker"},
                      {"type": "metadata", "field": "topic"}],
    }
    result = workspace.run_analysis(workspace.save_analysis("Metadata counts", spec))
    assert "speaker = A" in result["names"]
    assert "topic = X" in result["names"]
    i = result["names"].index("speaker = A")
    j = result["names"].index("topic = X")
    assert result["values"][i][j] == 4
