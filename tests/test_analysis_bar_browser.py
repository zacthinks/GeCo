"""Browser checks for the unified bar editor and unsaved selection navigation."""
from __future__ import annotations

import shutil
import threading

import pandas as pd
import pytest
from werkzeug.serving import make_server

from geometric_coder import GeometricCoder


def test_bar_editor_and_switch_guard(tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    chrome = next((path for exe in ("google-chrome", "chromium", "chromium-browser")
                   if (path := shutil.which(exe)) is not None), None)
    if chrome is None:
        pytest.skip("Chrome/Chromium unavailable")
    from geometric_coder.ui.app import create_app

    project = GeometricCoder.create(
        project_dir=tmp_path / "bar-browser.geco",
        data=pd.DataFrame({
            "id": [1, 2, 3, 4],
            "text": ["nice product", "poor product", "useful tool", "unclear tool"],
            "condition": ["A", "A", "B", "B"],
            "rating": [0.0, 2.0, 4.0, None],
            "score": [1.0, 2.0, 3.0, 4.0],
        }),
        keys=["id"], text="text", metadata=["condition", "rating", "score"],
        geometries=None,
    )
    code = project.create_code("Positive experience")
    units = project.database.list_units()
    for unit, label in zip(units, ["positive", "negative", "positive", "unsure"]):
        project.annotate(unit["observation_id"], code, label)
    other = project.save_analysis(
        "Other saved analysis", {
            "kind": "bar", "measure": "count",
            "variables": [{"type": "code", "code_id": code, "source": "labels", "value": "binary"}],
        },
    )
    app = create_app(project)
    server = make_server("127.0.0.1", 0, app.server, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with playwright.sync_playwright() as runner:
            browser = runner.chromium.launch(
                executable_path=chrome, headless=True, args=["--no-sandbox"],
            )
            try:
                page = browser.new_page(viewport={"width": 1400, "height": 1000})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(f"http://127.0.0.1:{server.server_port}/")
                page.get_by_text("Analyses", exact=True).first.click()

                def choose(locator, label):
                    locator.click()
                    page.get_by_text(label, exact=True).filter(visible=True).last.click()

                page.locator("#analysis-new").click()
                page.locator("#analysis-name").fill("Unified bars")
                choose(page.locator("#analysis-role-values"), "Code: Positive experience")
                choose(page.locator("#analysis-role-group"), "Metadata: condition")
                choose(page.locator("#analysis-bar-statistic"), "Sum")
                page.get_by_text("Values: Positive experience (Code)").wait_for()
                page.get_by_text("Group: condition (Metadata)").wait_for()
                page.locator("#analysis-preview").click()
                page.wait_for_function(
                    "() => document.querySelector('#analysis-figure .js-plotly-plot')"
                    "?.data?.[0]?.y?.join(',') === '1,1'", timeout=20000,
                )
                assert page.locator(".analysis-provenance-details").count() == 1
                page.locator("#analysis-save").click()
                page.wait_for_function(
                    "() => getComputedStyle(document.querySelector('#analysis-editor'))"
                    ".display === 'none'", timeout=15000,
                )
                saved = next(row for row in project.analyses()
                             if row["name"] == "Unified bars")
                assert saved["spec"]["kind"] == "bar_chart"

                # Existing-format Bars expose only supported controls. No
                # Color/Statistic edit can silently disappear during a save
                # or bypass the navigation guard.
                choose(page.locator("#analysis-select"), "Other saved analysis")
                page.locator("#analysis-edit").click()
                page.wait_for_function(
                    "() => document.querySelector('#analysis-kind')"
                    "?.textContent?.includes('existing format')",
                    timeout=15000,
                )
                page.locator("#analysis-bar-color-box").wait_for(state="hidden")
                page.locator("#analysis-bar-statistic-box").wait_for(state="hidden")
                assert page.locator("#analysis-bar-color").is_hidden()
                assert page.locator("#analysis-bar-statistic").is_hidden()
                # A supported legacy Group change must be tracked and saved,
                # preserving legacy count semantics and the original ID.
                choose(page.locator("#analysis-role-group"), "Metadata: condition")
                choose(page.locator("#analysis-select"), "Unified bars")
                page.locator("#analysis-nav-dialog").wait_for(state="visible")
                page.locator("#analysis-nav-save").click()
                page.locator("#analysis-editor").wait_for(state="hidden")
                legacy = project.analysis(other)["spec"]
                assert legacy["kind"] == "bar"
                assert legacy["measure"] == "count"
                assert "statistic" not in legacy
                assert "color" not in legacy.get("roles", {})
                assert any(variable.get("field") == "condition"
                           for variable in legacy.get("variables", []))

                # The unchanged edit must close immediately on analysis switch.
                choose(page.locator("#analysis-select"), "Unified bars")
                page.locator("#analysis-edit").click()
                page.get_by_text("Values: Positive experience (Code)").wait_for()
                choose(page.locator("#analysis-select"), "Other saved analysis")
                page.wait_for_function(
                    "() => getComputedStyle(document.querySelector('#analysis-editor'))"
                    ".display === 'none'", timeout=15000,
                )
                assert page.locator("#analysis-nav-dialog").is_hidden()

                # Dirty edits offer an explicit discard decision.
                choose(page.locator("#analysis-select"), "Unified bars")
                page.locator("#analysis-edit").click()
                page.wait_for_function(
                    "() => document.querySelector('#analysis-name')?.value === 'Unified bars' && "
                    "getComputedStyle(document.querySelector('#analysis-editor')).display !== 'none'",
                    timeout=15000,
                )
                page.locator("#analysis-name").fill("Unsaved change")
                assert page.locator("#analysis-name").input_value() == "Unsaved change"
                choose(page.locator("#analysis-select"), "Other saved analysis")
                page.locator("#analysis-nav-dialog").wait_for(state="visible")
                page.locator("#analysis-nav-discard").click()
                page.locator("#analysis-editor").wait_for(state="hidden")
                assert project.analysis(saved["analysis_id"])["name"] == "Unified bars"

                # Save and switch must update the original ID, not the newly selected chart.
                choose(page.locator("#analysis-select"), "Unified bars")
                page.locator("#analysis-edit").click()
                page.wait_for_function(
                    "() => document.querySelector('#analysis-name')?.value === 'Unified bars' && "
                    "getComputedStyle(document.querySelector('#analysis-editor')).display !== 'none'",
                    timeout=15000,
                )
                page.locator("#analysis-name").fill("Unified bars edited")
                assert page.locator("#analysis-name").input_value() == "Unified bars edited"
                choose(page.locator("#analysis-select"), "Other saved analysis")
                page.locator("#analysis-nav-dialog").wait_for(state="visible")
                page.locator("#analysis-nav-save").click()
                page.locator("#analysis-editor").wait_for(state="hidden")
                assert project.analysis(saved["analysis_id"])["name"] == "Unified bars edited"
                assert project.analysis(other)["name"] == "Other saved analysis"

                # Stay must restore the original selected analysis and keep the
                # modified editor open for continued editing.
                choose(page.locator("#analysis-select"), "Unified bars edited")
                page.locator("#analysis-edit").click()
                page.wait_for_function(
                    "() => document.querySelector('#analysis-name')?.value === 'Unified bars edited' && "
                    "getComputedStyle(document.querySelector('#analysis-editor')).display !== 'none'",
                    timeout=15000,
                )
                page.locator("#analysis-name").fill("Stay here draft")
                assert page.locator("#analysis-name").input_value() == "Stay here draft"
                choose(page.locator("#analysis-select"), "Other saved analysis")
                page.locator("#analysis-nav-dialog").wait_for(state="visible")
                # The navigation decision remains open until a researcher
                # explicitly resolves it; do not depend on the private DOM of
                # Dash's dropdown implementation for disabled-state checks.
                page.locator("#analysis-nav-stay").click()
                page.locator("#analysis-nav-dialog").wait_for(state="hidden")
                assert page.locator("#analysis-editor").is_visible()
                assert page.locator("#analysis-name").input_value() == "Stay here draft"
                page.wait_for_function(
                    "(id) => document.querySelector('#analysis-select')"
                    "?.textContent?.includes('Unified bars edited')",
                    timeout=15000,
                )
                choose(page.locator("#analysis-select"), "Other saved analysis")
                page.locator("#analysis-nav-dialog").wait_for(state="visible")
                page.locator("#analysis-nav-discard").click()
                page.locator("#analysis-editor").wait_for(state="hidden")

                # A modified *unnamed* new Scatterplot draft must not be
                # discarded just because the bar Values selectors are empty.
                page.locator("#analysis-new").click()
                page.wait_for_function(
                    "() => document.querySelector('#analysis-name')?.value === '' && "
                    "getComputedStyle(document.querySelector('#analysis-editor')).display !== 'none'",
                    timeout=15000,
                )
                choose(page.locator("#analysis-kind"), "Scatterplot")
                choose(page.locator("#analysis-role-x"), "Metadata: rating")
                choose(page.locator("#analysis-role-y"), "Metadata: score")
                choose(page.locator("#analysis-select"), "Unified bars edited")
                page.locator("#analysis-nav-dialog").wait_for(state="visible")
                # Invalid Save keeps the unsaved draft and the warning visible.
                page.locator("#analysis-nav-save").click()
                page.locator("#analysis-status").get_by_text(
                    "Analysis name must contain", exact=False
                ).wait_for(state="visible")
                assert page.locator("#analysis-editor").is_visible()
                assert page.locator("#analysis-nav-dialog").is_visible()
                page.locator("#analysis-nav-discard").click()
                page.locator("#analysis-editor").wait_for(state="hidden")
                assert not any(row["name"] == "" for row in project.analyses())

                # A duplicate with its default name unchanged can have
                # modified roles, which must also trigger the prompt.
                choose(page.locator("#analysis-select"), "Unified bars edited")
                page.locator("#analysis-duplicate").click()
                page.locator("#analysis-name").wait_for(state="visible")
                assert page.locator("#analysis-name").input_value() == "Unified bars edited (copy)"
                choose(page.locator("#analysis-role-group"), "Metadata: score")
                choose(page.locator("#analysis-select"), "Other saved analysis")
                page.locator("#analysis-nav-dialog").wait_for(state="visible")
                page.locator("#analysis-nav-discard").click()
                page.locator("#analysis-editor").wait_for(state="hidden")
                assert not any(row["name"] == "Unified bars edited (copy)"
                               for row in project.analyses())

                assert not errors, errors
            finally:
                browser.close()
    finally:
        server.shutdown()
        thread.join(timeout=10)
