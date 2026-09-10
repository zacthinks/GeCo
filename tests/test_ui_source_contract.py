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


def test_real_aera_acceptance_example_uses_direct_public_launch_path() -> None:
    root = Path(__file__).resolve().parents[1]
    example_source = (root / "examples" / "aera_2026_project.py").read_text(encoding="utf-8")
    assert "BrowserProgressPage(" not in example_source
    assert "ProgressGroup(" not in example_source
    assert "project.launch(" in example_source
    assert "use_reloader=False" in example_source


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
    assert 'tune_hyperparameters="tune" in set(tune_hyperparameter_values or [])' in app_source


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


def test_launch_app_displays_url_before_running(monkeypatch) -> None:
    import geometric_coder.ui.app as app_module

    events: list[tuple[str, object]] = []

    class FakeApp:
        def run(self, **kwargs):
            events.append(("run", kwargs))

    monkeypatch.setattr(app_module, "create_app", lambda *args, **kwargs: FakeApp())
    monkeypatch.setattr(
        app_module,
        "_display_local_browser_url",
        lambda host, port: events.append(("url", (host, port))),
    )

    app_module.launch_app(
        object(),
        host="0.0.0.0",
        port=8125,
        debug=False,
        use_reloader=False,
    )

    assert events[0] == ("url", ("0.0.0.0", 8125))
    assert events[1][0] == "run"
    assert events[1][1]["host"] == "0.0.0.0"
    assert events[1][1]["port"] == 8125


def test_jupyter_browser_link_opens_new_tab(monkeypatch) -> None:
    import IPython
    import IPython.display
    from geometric_coder.ui.app import _display_local_browser_url

    rendered: list[str] = []

    class FakeHTML:
        def __init__(self, data: str):
            self.data = data

    monkeypatch.setattr(IPython, "get_ipython", lambda: object())
    monkeypatch.setattr(IPython.display, "HTML", FakeHTML)
    monkeypatch.setattr(IPython.display, "display", lambda value: rendered.append(value.data))

    url = _display_local_browser_url("0.0.0.0", 8050)

    assert url == "http://127.0.0.1:8050/"
    assert len(rendered) == 1
    assert "Open in browser" in rendered[0]
    assert "target='_blank'" in rendered[0]
    assert "http://127.0.0.1:8050/" in rendered[0]


def test_non_ipython_launch_url_is_plain_text(monkeypatch, capsys) -> None:
    import IPython
    from geometric_coder.ui.app import _display_local_browser_url

    monkeypatch.setattr(IPython, "get_ipython", lambda: None)
    _display_local_browser_url("127.0.0.1", 8126)

    assert "GeCo is running at http://127.0.0.1:8126/ (local browser URL)" in capsys.readouterr().out


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
