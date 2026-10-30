from __future__ import annotations

from pathlib import Path


def test_projection_numeric_inputs_do_not_use_invalid_decimal_input_mode() -> None:
    app_source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "geometric_coder"
        / "ui"
        / "app.py"
    ).read_text(encoding="utf-8")
    assert 'inputMode="decimal"' not in app_source


def test_initial_geometry_and_view_selection_falls_back_to_registered_resources() -> None:
    from geometric_coder.ui.app import _valid_geometry_id, _valid_view_id

    geometries = [{"geometry_id": 7}, {"geometry_id": 11}]
    views = [{"view_id": 23}, {"view_id": 31}]
    assert _valid_geometry_id(None, geometries) == 7
    assert _valid_view_id(None, views) == 23
    # Legacy default remains selected even when a canonical-only view has a
    # negative adapter ID; explicit saved canonical selections remain valid.
    mixed = [{"view_id": 1}, {"view_id": -1}]
    assert _valid_view_id(None, mixed) == 1
    assert _valid_view_id(999, mixed) == 1
    assert _valid_view_id(-1, mixed) == -1
    assert _valid_view_id(None, [{"view_id": -42}]) == -42


def test_code_center_patch_target_is_initialized_before_partial_updates() -> None:
    from geometric_coder.ui.app import _code_center_placeholder_figure

    figure = _code_center_placeholder_figure().to_plotly_json()
    assert len(figure["data"]) == 1
    focal = figure["data"][0]
    assert focal["x"] == []
    assert focal["y"] == []
    assert focal["customdata"] == []
    assert focal["hovertext"] == []

    app_source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "geometric_coder"
        / "ui"
        / "app.py"
    ).read_text(encoding="utf-8")
    assert 'State("code-center-map", "figure")' in app_source
    assert 'if not current_figure or not current_figure.get("data"):' in app_source


def test_public_debug_launch_disables_auto_reloader_by_default(monkeypatch) -> None:
    from geometric_coder import GeometricCoder
    import geometric_coder.ui as ui

    captured: dict[str, object] = {}

    def fake_launch_app(project, *, host, port, debug, use_reloader=False):
        captured.update(
            project=project,
            host=host,
            port=port,
            debug=debug,
            use_reloader=use_reloader,
        )

    monkeypatch.setattr(ui, "launch_app", fake_launch_app)
    project = object.__new__(GeometricCoder)
    project.launch(debug=True)

    assert captured["debug"] is True
    assert captured["use_reloader"] is False


def test_public_launch_allows_explicit_reloader_opt_in(monkeypatch) -> None:
    from geometric_coder import GeometricCoder
    import geometric_coder.ui as ui

    captured: dict[str, object] = {}

    def fake_launch_app(project, *, host, port, debug, use_reloader=False):
        captured["use_reloader"] = use_reloader

    monkeypatch.setattr(ui, "launch_app", fake_launch_app)
    project = object.__new__(GeometricCoder)
    project.launch(debug=True, use_reloader=True)
    assert captured["use_reloader"] is True


def test_package_version_matches_project_metadata() -> None:
    import tomllib

    import geometric_coder

    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert geometric_coder.__version__ == project["project"]["version"]


def test_dash_dependency_is_not_speculatively_pinned_to_pre_42() -> None:
    import tomllib

    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    dash_requirements = [
        requirement
        for requirement in project["project"]["dependencies"]
        if requirement.startswith("dash") and not requirement.startswith("dash-ag-grid")
    ]
    assert dash_requirements == ["dash>=4.0"]


def test_classifier_manager_exposes_identity_preserving_rename_action() -> None:
    app_source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "geometric_coder"
        / "ui"
        / "app.py"
    ).read_text(encoding="utf-8")
    assert 'id="classifier-manager-rename-name"' in app_source
    assert 'id="classifier-manager-rename"' in app_source
    assert '"Rename selected classifier"' in app_source
    assert "project.rename_classifier_spec(" in app_source


def test_develop_exposes_opt_in_cross_validation_tuning() -> None:
    app_source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "geometric_coder"
        / "ui"
        / "app.py"
    ).read_text(encoding="utf-8")
    assert 'id="focus-tune-hyperparameters"' in app_source
    assert '"label": "Tune hyperparameters with cross-validation"' in app_source
    assert '"focus_tune_hyperparameters", False' in app_source
    assert 'tune=tune_hyperparameters' in app_source
    assert 'retune_current=tune_hyperparameters' in app_source
    # Cross-validation remains opt-in at the explicit training boundary, not
    # on recommendation clicks (which must only use frozen predictions).
    assert '"tune": "tune" in set(tune_values or [])' in app_source
    assert 'tune=bool(next_request["tune"])' in app_source
    assert 'auto_retrain=False' in app_source


def test_explore_multi_code_palette_normalizes_and_places_codes() -> None:
    from geometric_coder.ui.app import (
        _normalize_explore_code_palette,
        _place_code_in_explore_palette,
    )

    valid = {1, 2, 3, 4}
    assert _normalize_explore_code_palette(None, valid) == [None]
    assert _normalize_explore_code_palette([], valid) == [None]
    assert _normalize_explore_code_palette([1, 2, 1, 99, None], valid) == [1, 2, None, None, None]
    assert _place_code_in_explore_palette([1, None], 2, valid) == [1, 2]
    assert _place_code_in_explore_palette([1, 2], 3, valid) == [1, 2, 3]
    assert _place_code_in_explore_palette([1, 2], 2, valid) == [1, 2]


def test_explore_exposes_multi_code_coding_palette() -> None:
    app_source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "geometric_coder"
        / "ui"
        / "app.py"
    ).read_text(encoding="utf-8")
    assert 'id="explore-code-palette-store"' in app_source
    assert 'id="explore-code-palette"' in app_source
    assert 'id="explore-add-code-row"' in app_source
    assert '"+ Add code"' in app_source
    assert '{"type": "explore-code-dropdown", "index": index}' in app_source
    assert '{"type": "explore-code-remove", "index": index}' in app_source
    assert '"explore_code_palette"' in app_source
    assert 'max-height: 360px' in (
        Path(__file__).resolve().parents[1]
        / "src"
        / "geometric_coder"
        / "ui"
        / "assets"
        / "geco.css"
    ).read_text(encoding="utf-8")


def test_explore_palette_row_count_preserves_visible_rows_without_duplicates() -> None:
    from geometric_coder.ui.app import _explore_palette_for_row_count

    valid = {1, 2, 3}
    assert _explore_palette_for_row_count([1], valid, 3) == [1, None, None]
    assert _explore_palette_for_row_count([1, 2, 1], valid, 3) == [1, 2, None]
    assert _explore_palette_for_row_count([1, 2, 3], valid, 2) == [1, 2]
    assert _explore_palette_for_row_count(None, valid, None) == [None]


def test_explore_map_uses_stable_navigation_overlay_traces() -> None:
    import numpy as np

    from geometric_coder.ui.app import _geometry_overlay_payload
    from geometric_coder.ui.figures import (
        EXPLORE_FOCAL_TRACE_INDEX,
        EXPLORE_TRACE_NAMES,
        EXPLORE_VISITED_TRACE_INDEX,
        build_geometry_figure,
    )

    units = [
        {"unit_id": 1, "user_key": {"row_id": 1}},
        {"unit_id": 2, "user_key": {"row_id": 2}},
        {"unit_id": 3, "user_key": {"row_id": 3}},
    ]
    figure = build_geometry_figure(
        coordinates=np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
        units=units,
        eligible=[True, False, True],
        seen_unit_ids={1},
        focal_unit_id=2,
    )
    payload_json = figure.to_plotly_json()
    assert tuple(trace["name"] for trace in payload_json["data"]) == EXPLORE_TRACE_NAMES
    payload = _geometry_overlay_payload(
        payload_json, seen_unit_ids={1, 3}, focal_unit_id=3
    )
    assert payload is not None
    assert payload[EXPLORE_VISITED_TRACE_INDEX]["customdata"] == [1, 3]
    assert payload[EXPLORE_FOCAL_TRACE_INDEX]["customdata"] == [3]


def test_explore_ui_callbacks_are_partitioned_for_lightweight_interactions() -> None:
    root = Path(__file__).resolve().parents[1]
    app_source = (root / "src" / "geometric_coder" / "ui" / "app.py").read_text(
        encoding="utf-8"
    )
    css_source = (root / "src" / "geometric_coder" / "ui" / "assets" / "geco.css").read_text(
        encoding="utf-8"
    )

    assert "update_title=None" in app_source
    assert 'Output("explore-focal-unit-store", "data")' in app_source
    assert 'Output("focus-focal-unit-store", "data")' in app_source
    assert 'Input("explore-map-filter-refresh-store", "data")' in app_source
    assert "def update_map_navigation_overlays(" in app_source
    assert "patch[\"data\"][int(trace_index)][property_name] = property_value" in app_source

    # Dynamic pattern buttons may fire when they mount. Only positive click counts
    # are user actions, so mounting a row cannot remove it or annotate anything.
    assert app_source.count("int(triggered_value or 0) < 1") >= 2

    # Palette structure should not be rebuilt merely because focal/span/annotation
    # state changes; only row-count or code-registry changes rebuild its DOM.
    start = app_source.index("def render_explore_code_palette(")
    decorator_start = app_source.rfind("@app.callback", 0, start)
    palette_block = app_source[decorator_start:start]
    assert 'Input("explore-code-row-count-store", "data")' in palette_block
    assert 'Input("explore-code-refresh-store", "data")' in palette_block
    assert 'Input("explore-focal-unit-store", "data")' not in palette_block
    assert 'Input("explore-span-selection-store", "data")' not in palette_block
    assert 'Input("explore-annotation-refresh-store", "data")' not in palette_block

    # Code Center can observe annotation refreshes, but must short-circuit while
    # hidden so Explore coding does not launch background map/table work.
    assert app_source.count('if workspace != "codes":') >= 3

    assert ".coding-panel-heading" in css_source
    assert "margin-bottom: 0.85rem" in css_source


def test_committee_manager_rename_and_probability_explanation_contract() -> None:
    app_source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "geometric_coder"
        / "ui"
        / "app.py"
    ).read_text(encoding="utf-8")
    assert 'id="committee-manager-rename-name"' in app_source
    assert 'id="committee-rename"' in app_source
    assert '"Committee probability"' in app_source
    assert 'committee_probability = data.get("committee_probability")' in app_source


def test_local_browser_url_normalizes_wildcard_bind_host() -> None:
    from geometric_coder.ui.app import _local_browser_url

    assert _local_browser_url("0.0.0.0", 8050) == "http://127.0.0.1:8050/"
    assert _local_browser_url("::", 8051) == "http://127.0.0.1:8051/"
    assert _local_browser_url("127.0.0.1", 8123) == "http://127.0.0.1:8123/"
    assert _local_browser_url("localhost", 8124) == "http://localhost:8124/"


def test_launch_app_returns_closeable_server_and_shows_bound_url(monkeypatch) -> None:
    import geometric_coder.ui.app as app_module
    import geometric_coder.ui.server as server_module

    events = []

    class FakeServer:
        port = 8125

        def __init__(self, app, *, host, port):
            events.append(("bound", host, port))

        def start(self):
            events.append(("started",))
            return self

        def close(self):
            events.append(("closed",))

    monkeypatch.setattr(app_module, "create_app", lambda *args, **kwargs: object())
    monkeypatch.setattr(server_module, "GeCoServer", FakeServer)
    monkeypatch.setattr(
        app_module,
        "_display_local_browser_url",
        lambda host, port: events.append(("url", host, port)),
    )
    handle = app_module.launch_app(
        object(), host="0.0.0.0", port=8125, debug=False,
    )
    assert handle is not None
    assert events == [
        ("bound", "0.0.0.0", 8125),
        ("started",),
        ("url", "0.0.0.0", 8125),
    ]


def test_geco_server_closes_without_killing_notebook_and_endpoint_is_available():
    from unittest.mock import Mock, patch

    from flask import Flask

    from geometric_coder.ui.server import GeCoServer

    flask_app = Flask(__name__)
    wrapper = Mock(server=flask_app)
    with patch("geometric_coder.ui.server.make_server") as make:
        http = make.return_value
        http.server_port = 8125
        server = GeCoServer(wrapper, host="127.0.0.1", port=8125)
        server.start()
        response = flask_app.test_client().post("/_geco/close")
        assert response.status_code == 200
        assert response.json == {"status": "closing"}
        server.close()
        server.close()
        assert server.closed
        http.shutdown.assert_called_once()
        http.server_close.assert_called_once()


def test_geco_close_copy_in_both_workspaces():
    root = Path(__file__).resolve().parents[1] / "src/geometric_coder/ui"
    for filename in ("app.py", "focus_coding.py"):
        source = (root / filename).read_text(encoding="utf-8")
        assert "Close GeCo" in source
        assert "Geometric Coder: Scalable, computationally augmented exploration and coding" in source
        assert "End session" not in source



def test_focus_mobile_css_keeps_touch_targets_and_footer_available() -> None:
    from pathlib import Path

    css_source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "geometric_coder"
        / "ui"
        / "assets"
        / "geco.css"
    ).read_text(encoding="utf-8")
    assert "@media (max-width: 650px)" in css_source
    assert ".focus-coding-document-pane { max-height: 48vh; overflow-y: auto; }" in css_source
    assert ".focus-coding-judgment { min-height: 44px; font-size: 1rem; }" in css_source
    assert "position: sticky;" in css_source
    assert "safe-area-inset-bottom" in css_source


def test_committee_manager_clears_hidden_cross_code_member_selection() -> None:
    app_source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "geometric_coder"
        / "ui"
        / "app.py"
    ).read_text(encoding="utf-8")
    assert 'Output("committee-members", "value")' in app_source
    assert 'valid_member_ids = {int(row["classifier_spec_id"]) for row in specs}' in app_source
    assert 'if int(value) in valid_member_ids' in app_source
    assert 'triggered == "focus-manage-committees"' in app_source


def test_auto_training_ui_describes_cross_workspace_triggers() -> None:
    app_source = (
        Path(__file__).resolve().parents[1]
        / "src" / "geometric_coder" / "ui" / "app.py"
    ).read_text(encoding="utf-8")
    assert '"label": "Automatically train selected models after saved coding changes"' in app_source
    assert "With automatic training off, only explicit Train actions fit models;" in app_source
    assert "saved changes can leave frozen predictions stale but usable." in app_source
    assert "saved judgments in Develop, Explore, or Code Center, teaching-example changes," in app_source
    assert "and Apply commits or undos that change assignments can request training only " in app_source
    assert "when the affected code matches the code selected in Develop." in app_source
    assert "decisions, proposal generation, zero-assignment commits, and recommendation " in app_source
    assert "or navigation clicks never start training." in app_source
    assert "The scope is the active classifier " in app_source
    assert "or selected committee members, unless Train All is explicitly chosen." in app_source
    assert "Apply reviews are not yet connected." not in app_source


def test_teaching_examples_help_describes_save_and_training_timing() -> None:
    app_source = (
        Path(__file__).resolve().parents[1]
        / "src" / "geometric_coder" / "ui" / "app.py"
    ).read_text(encoding="utf-8")
    assert "Saving a teaching example updates the evidence, not the fitted model." in app_source
    assert "For the code selected in Develop, active examples can trigger training of " in app_source
    assert "the selected models after a change when automatic training is enabled." in app_source
    assert "Otherwise, use Train to include active examples in the next manual fit." in app_source
    assert "Active examples join corpus assignments at the next manual retrain." not in app_source


def test_develop_code_summary_does_not_refresh_on_focal_navigation() -> None:
    """Reading a new recommendation must not rescan code-wide labels."""
    app_source = (
        Path(__file__).resolve().parents[1]
        / "src" / "geometric_coder" / "ui" / "app.py"
    ).read_text(encoding="utf-8")
    start = app_source.index('    def update_focus_code_summary(')
    decorator = app_source.rfind("    @app.callback(", 0, start)
    callback = app_source[decorator:start]
    assert 'Input("focus-code-dropdown", "value")' in callback
    assert 'Input("focus-refresh-store", "data")' in callback
    assert 'Input("teaching-example-refresh-store", "data")' in callback
    assert 'Input("workspace-tabs", "value")' in callback
    assert 'Input("focus-focal-unit-store", "data")' not in callback
    assert 'Input("apply-refresh-store", "data")' not in callback
    assert 'Input("explore-annotation-refresh-store", "data")' not in callback
    function_body = app_source[start:app_source.index("    @app.callback(", start)]
    assert "project.current_annotations(int(code_id))" in function_body
    assert "project.training_label_counts(int(code_id))" in function_body
    assert 'if workspace != "focus":' in function_body
    assert "return no_update, no_update, no_update" in function_body


def test_training_and_recommendation_callbacks_preserve_explicit_boundaries() -> None:
    """Protect the core Dash UI contract across coding sources."""
    app_source = (
        Path(__file__).resolve().parents[1]
        / "src" / "geometric_coder" / "ui" / "app.py"
    ).read_text(encoding="utf-8")
    assert 'auto_retrain=False,' in app_source
    recommend = app_source[app_source.index("    def focus_recommend("):]
    recommend = recommend[:recommend.index("    @app.callback(")]
    assert "_recommend_without_fitting(" in recommend
    assert "train_classifiers_after_evidence_change" not in recommend

    training = app_source[app_source.index("    def train_after_develop_judgment("):]
    training = training[:training.index("    @app.callback(")]
    assert 'messages.get("develop", no_update)' in training
    assert 'messages.get("explore", no_update)' in training
    assert 'messages.get("code_center", no_update)' in training
    assert 'messages.get("teaching_example", no_update)' in training
    assert 'messages.get("apply", no_update)' in training
    assert 'if not request:' in training
    assert 'if not completed:' in training
    assert "training_inbox.drain(fit)" in training



def test_code_center_uses_transitive_unified_display_candidates() -> None:
    from geometric_coder.ui.app import _display_view_record, _display_view_records

    class FakeProject:
        def views(self, geometry_id, *, include_descendants=False):
            assert geometry_id == 7
            assert include_descendants is True
            return [
                {"view_id": 23, "name": "Legacy 2D view"},
                {"view_id": -42, "name": "Canonical 1D descendant"},
            ]

    project = FakeProject()
    records = _display_view_records(project, 7)
    assert [item["view_id"] for item in records] == [23, -42]
    assert _display_view_record(project, geometry_id=7, view_id=-42) == records[1]
    assert _display_view_record(project, geometry_id=7, view_id=-99) is None

    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "geometric_coder" / "ui" / "app.py"
    ).read_text(encoding="utf-8")
    assert 'Input("representation-refresh-store", "data")' in source
    assert '_display_view_records(project, int(geometry_id))' in source
    assert '_display_view_record(' in source


def test_presentation_views_include_canonical_descendants_without_duplicate_ids() -> None:
    from geometric_coder.ui.app import (
        _presentation_view_options, _presentation_view_records,
    )

    class FakeProject:
        def views(self, geometry_id, *, include_descendants=False):
            assert include_descendants
            return {
                7: [
                    {"view_id": 23, "name": "Legacy"},
                    {"view_id": -42, "name": "1D child"},
                ],
                11: [{"view_id": -42, "name": "1D child"}],
            }[geometry_id]

    geometries = [
        {"geometry_id": 7, "name": "Source A"},
        {"geometry_id": 11, "name": "Source B"},
    ]
    records = _presentation_view_records(FakeProject(), geometries)
    assert {row["view_id"] for row in records} == {23, -42}
    assert len(records) == 2
    options = _presentation_view_options(records, geometries)
    assert {option["value"] for option in options} == {23, -42}
    assert next(option for option in options if option["value"] == -42)["label"] == "Source A — 1D child"


def test_explorer_map_resolves_canonical_display_without_legacy_view_row():
    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "geometric_coder" / "ui" / "app.py"
    ).read_text(encoding="utf-8")
    section = source.split("    def update_map(", 1)[1].split(
        "    def update_map_navigation_overlays(", 1
    )[0]
    assert "_display_view_record(" in section
    assert "project.database.get_view(int(view_id))" not in section
    assert "project.view_coordinates(int(view_id))" in section
