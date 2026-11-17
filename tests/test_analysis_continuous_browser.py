"""Playwright exercise for numerical and calendar-grouped saved Analyses."""
from __future__ import annotations

import shutil
import threading

import pandas as pd
import pytest
from werkzeug.serving import make_server

from geometric_coder import GeometricCoder


def test_continuous_editor_scatter_histogram_and_restore(tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    chrome = next((path for executable in ("google-chrome", "chromium", "chromium-browser")
                   if (path := shutil.which(executable)) is not None), None)
    if chrome is None:
        pytest.skip("Chrome/Chromium unavailable")

    from geometric_coder.ui.app import create_app

    project = GeometricCoder.create(
        project_dir=tmp_path / "numerical-browser.geco",
        data=pd.DataFrame({
            "id": [1, 2, 3, 4],
            "text": ["first", "second", "third", "fourth"],
            "score": [1.0, 2.0, 3.0, 4.0],
            "rating": [4.0, 3.0, 2.0, 1.0],
            "date": ["2025-12-31", "2026-01-01", "2026-02-03", "2026-02-05"],
        }), keys=["id"], text="text",
        metadata=["score", "rating", "date"], geometries=None,
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
                page = browser.new_page(viewport={"width": 1500, "height": 1100})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(f"http://127.0.0.1:{server.server_port}/")
                page.get_by_text("Analyses", exact=True).first.click()

                def choose(dropdown, label):
                    dropdown.wait_for(state="visible", timeout=15000)
                    dropdown.click()
                    page.get_by_text(label, exact=True).filter(visible=True).last.click()

                page.locator("#analysis-new").click()
                page.locator("#analysis-name").fill("Numerical browser")
                choose(page.locator("#analysis-kind"), "Scatterplot — two numerical axes")
                choose(page.locator("#analysis-role-x"), "Metadata: score")
                choose(page.locator("#analysis-role-y"), "Metadata: rating")
                choose(page.locator('button[id*="analysis-role-value"][id*="x:metadata:score"]'),
                       "Numeric (continuous charts)")
                choose(page.locator('button[id*="analysis-role-value"][id*="y:metadata:rating"]'),
                       "Numeric (continuous charts)")
                page.locator("#analysis-preview").click()
                page.wait_for_function(
                    "() => document.querySelector('#analysis-figure .js-plotly-plot')"
                    "?.data?.[0]?.type === 'scattergl' || "
                    "(document.querySelector('#analysis-status')?.textContent || '').trim().length > 0",
                    timeout=20000,
                )
                assert not page.locator("#analysis-status").inner_text().strip(), (
                    "Scatter preview validation: " + page.locator("#analysis-status").inner_text()
                )
                assert page.evaluate(
                    "() => document.querySelector('#analysis-figure .js-plotly-plot')?.data?.[0]?.type"
                ) == "scattergl"
                page.locator("#analysis-save").click()
                page.wait_for_function(
                    "() => getComputedStyle(document.querySelector('#analysis-editor'))"
                    ".display === 'none'", timeout=15000,
                )
                record = next(row for row in project.analyses()
                              if row["name"] == "Numerical browser")
                assert record["spec"]["kind"] == "scatter"
                assert project.run_analysis(record["analysis_id"])["eligible"] == 4

                choose(page.locator("#analysis-select"), "Numerical browser")
                page.locator("#analysis-edit").click()
                page.locator("#analysis-role-y").wait_for(state="visible")
                assert project.analysis(record["analysis_id"])["spec"]["roles"]["y"]["variable"]["field"] == "rating"
                page.locator("#analysis-cancel").click()

                page.locator("#analysis-new").click()
                page.locator("#analysis-name").fill("Calendar histogram")
                choose(page.locator("#analysis-kind"), "Continuous histogram")
                choose(page.locator("#analysis-role-continuous-values"), "Metadata: score")
                choose(page.locator("#analysis-continuous-group"), "Metadata: date")
                choose(page.locator('button[id*="analysis-role-value"][id*="values:metadata:score"]'),
                       "Numeric (continuous charts)")
                choose(page.locator('button[id*="analysis-role-value"][id*="group:metadata:date"]'),
                       "Temporal / ISO date (calendar grouping)")
                choose(page.locator('button[id*="analysis-role-transform"][id*="group:metadata:date"]'),
                       "Calendar month")
                page.locator("#analysis-preview").click()
                page.wait_for_function(
                    "() => document.querySelector('#analysis-figure .js-plotly-plot')"
                    "?.data?.length > 0", timeout=20000,
                )
                page.locator("#analysis-save").click()
                page.wait_for_function(
                    "() => getComputedStyle(document.querySelector('#analysis-editor'))"
                    ".display === 'none'", timeout=15000,
                )
                saved = next(row for row in project.analyses()
                             if row["name"] == "Calendar histogram")
                assert saved["spec"]["roles"]["group"]["transform"] == {
                    "kind": "calendar", "unit": "month",
                }
                assert not errors, errors
            finally:
                browser.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)
