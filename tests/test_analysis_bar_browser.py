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
            "site": ["North", "South", "North", "South"],
            "cohort": ["X", "X", "Y", "Y"],
            "rating": [0.0, 2.0, 4.0, None],
            "score": [1.0, 2.0, 3.0, 4.0],
        }),
        keys=["id"], text="text", metadata=["condition", "site", "cohort", "rating", "score"],
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
                requests = []
                page.on("request", lambda request: requests.append(
                    request.post_data_json
                ) if request.method == "POST" and request.url.endswith(
                    "/_dash-update-component"
                ) else None)
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

                # Categorical Count with both Group and Color must compute,
                # preview, and persist its complete typed recipe.
                page.locator("#analysis-new").click()
                page.wait_for_function(
                    "() => document.querySelector('#analysis-name')?.value === ''",
                    timeout=15000,
                )
                page.locator("#analysis-name").fill("Three-role categorical Count")
                choose(page.locator("#analysis-role-values"), "Metadata: condition")
                choose(page.locator("#analysis-role-group"), "Metadata: site")
                choose(page.locator("#analysis-bar-color"), "Metadata: cohort")
                page.get_by_text("Values: condition (Metadata)", exact=True).wait_for()
                page.get_by_text("Group: site (Metadata)", exact=True).wait_for()
                page.get_by_text("Color: cohort (Metadata)", exact=True).wait_for()
                page.locator("#analysis-preview").click()
                page.wait_for_function(
                    "() => { const plot = document.querySelector('#analysis-figure .js-plotly-plot'); "
                    "return plot?.data?.length === 2 && "
                    "plot.data.every(trace => trace.y?.join(',') === '1,1'); }",
                    timeout=20000,
                )
                page.locator("#analysis-save").click()
                page.locator("#analysis-editor").wait_for(state="hidden")
                categorical_saved = next(
                    row for row in project.analyses()
                    if row["name"] == "Three-role categorical Count"
                )
                assert categorical_saved["spec"]["kind"] == "bar_chart"
                assert categorical_saved["spec"]["statistic"] == "count"
                assert set(categorical_saved["spec"]["roles"]) == {
                    "values", "group", "color"
                }
                assert len(project.run_analysis(categorical_saved["analysis_id"])["series"]) == 2

                # Clearing an optional role must persist through Save and reopen.
                choose(page.locator("#analysis-select"), "Three-role categorical Count")
                page.locator("#analysis-edit").click()
                page.get_by_text("Color: cohort (Metadata)", exact=True).wait_for()
                page.locator("#analysis-bar-color [aria-label='Clear selection']").click()
                page.get_by_text("Color: cohort (Metadata)", exact=True).wait_for(state="detached")
                page.locator("#analysis-preview").click()
                page.wait_for_function(
                    "() => { const plot = document.querySelector('#analysis-figure .js-plotly-plot'); "
                    "return plot?.data?.length === 1 && plot.data[0]?.y?.join(',') === '2,2'; }",
                    timeout=20000,
                )
                page.locator("#analysis-save").click()
                page.locator("#analysis-editor").wait_for(state="hidden")
                cleared = project.analysis(categorical_saved["analysis_id"])["spec"]
                assert "color" not in cleared["roles"]
                choose(page.locator("#analysis-select"), "Three-role categorical Count")
                page.locator("#analysis-edit").click()
                page.locator("#analysis-editor").wait_for(state="visible")
                page.get_by_text("Group: site (Metadata)", exact=True).wait_for(
                    state="visible", timeout=15000,
                )
                # A visible role summary may still be from the previous editor.
                # Finish the edit response before allowing Cancel to race it.
                page.wait_for_load_state("networkidle")
                assert not page.get_by_text("Color: cohort (Metadata)", exact=True).count()
                page.locator("#analysis-cancel").click()
                page.locator("#analysis-editor").wait_for(state="hidden")
                # Drain the Cancel response, not only its initial DOM update,
                # before selecting an unrelated saved analysis.
                page.wait_for_load_state("networkidle")

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
                # Wait for the restored legacy role configuration before
                # selecting a new Group; otherwise an outstanding editor-open
                # response can overwrite the newly selected dropdown value.
                page.get_by_text(
                    "Values: Positive experience (Code)", exact=True
                ).wait_for(state="visible", timeout=15000)
                # The backend restores the saved legacy recipe in an async
                # multi-output callback. Await that response before changing
                # a role, rather than using a transient blank Group as a signal.
                page.wait_for_load_state("networkidle")
                assert page.locator("#analysis-bar-color").is_hidden()
                assert page.locator("#analysis-bar-statistic").is_hidden()
                # The saved legacy chart has no Group. Wait until the previous
                # chart's stale Site selection has been cleared, not merely
                # until the new Values heading is visible. Otherwise an
                # outstanding editor-restore response can overwrite a click.
                page.wait_for_function(
                    "() => { const group = document.querySelector('#analysis-role-group'); "
                    "return group && !group.textContent.includes('Metadata: site') "
                    "&& !group.textContent.includes('Metadata: condition') "
                    "&& !document.querySelector('#analysis-role-settings')"
                    "?.textContent.includes('Group: site (Metadata)'); }",
                    timeout=15000,
                )
                # A supported legacy Group change must be tracked and saved,
                # preserving legacy count semantics and the original ID.
                choose(page.locator("#analysis-role-group"), "Metadata: condition")
                # Wait for the role-settings callback to acknowledge the
                # selection before triggering another asynchronous callback.
                try:
                    page.get_by_text("Group: condition (Metadata)", exact=True).wait_for(
                        state="visible", timeout=6000,
                    )
                except playwright.Error as exc:
                    raise AssertionError(
                        "Legacy Group selection did not render; "
                        "dropdown=" + repr(page.locator("#analysis-role-group").inner_text())
                        + "; roles=" + repr(page.locator("#analysis-role-settings").inner_text())
                        + "; editor=" + repr(page.locator("#analysis-editor").inner_text()[:300])
                        + "; recent group requests=" + repr([
                            (x.get("output"), next((
                                y.get("value") for y in x.get("inputs", [])
                                if y.get("id") == "analysis-role-group"
                            ), None), next((
                                y.get("value") for y in x.get("state", [])
                                if y.get("id") == "analysis-select"
                            ), None)) for x in requests[-15:]
                        ])
                    ) from exc
                choose(page.locator("#analysis-select"), "Unified bars")
                page.locator("#analysis-nav-dialog").wait_for(state="visible")
                page.locator("#analysis-nav-save").click()
                page.locator("#analysis-editor").wait_for(state="hidden")
                legacy = project.analysis(other)["spec"]
                assert legacy["kind"] == "bar"
                assert legacy["measure"] == "count"
                assert "statistic" not in legacy
                assert "color" not in legacy.get("roles", {})
                assert legacy.get("schema_version") == 1, legacy
                assert any(variable.get("field") == "condition"
                           for variable in legacy.get("variables", [])), legacy

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
