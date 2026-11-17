"""Browser smoke test for visualization-first saved analyses.

Requires Playwright and a system Chrome/Chromium; CI provides both.
"""
from __future__ import annotations

import shutil
import threading

import pandas as pd
import pytest
from werkzeug.serving import make_server

from geometric_coder import GeometricCoder


def test_analysis_role_editor_preview_save_edit_and_incompatible_chart(tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    chrome = next((path for executable in ("google-chrome", "chromium", "chromium-browser")
                   if (path := shutil.which(executable)) is not None), None)
    if chrome is None:
        pytest.skip("Chrome/Chromium unavailable")

    from geometric_coder.ui.app import create_app

    project = GeometricCoder.create(
        project_dir=tmp_path / "role-browser.geco",
        data=pd.DataFrame({
            "id": [1, 2, 3], "text": ["a", "b", "c"],
            "group": ["A", "B", "A"],
            "date": ["2026-01-02", "2026-02-01", "2026-03-09"],
        }),
        keys=["id"], text="text", metadata=["group", "date"], geometries=None,
    )
    code = project.create_code("Theme")
    ids = [row["observation_id"] for row in project.database.list_units()]
    project.annotate(ids[0], code, "positive")
    project.annotate(ids[1], code, "negative")
    project.annotate(ids[2], code, "positive")

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
                page.locator("#analysis-new").click()
                page.locator("#analysis-name").fill("Roles browser")
                page.locator("#analysis-role-values").click()
                page.get_by_text("Code: Theme", exact=True).last.click()
                page.locator("#analysis-preview").click()
                page.wait_for_function(
                    "() => document.querySelector('#analysis-figure .js-plotly-plot')"
                    "?.data?.[0]?.y?.[0] === 2", timeout=20000,
                )
                page.locator("#analysis-save").click()
                page.wait_for_function(
                    "() => getComputedStyle(document.querySelector('#analysis-editor'))"
                    ".display === 'none'", timeout=15000,
                )
                assert project.analyses()[0]["spec"]["schema_version"] == 2
                page.locator("#analysis-edit").click()
                page.wait_for_function(
                    "() => getComputedStyle(document.querySelector('#analysis-editor'))"
                    ".display !== 'none'", timeout=15000,
                )
                assert page.locator("#analysis-name").input_value() == "Roles browser"
                page.locator("#analysis-kind").click()
                page.get_by_text("Association heatmap", exact=True).last.click()
                page.locator("#analysis-preview").click()
                page.locator("#analysis-status").get_by_text(
                    "Preview unavailable:", exact=False,
                ).wait_for(timeout=15000)
                # The rejected preview must not modify the saved recipe.
                assert project.analyses()[0]["spec"]["kind"] == "bar"
                page.locator("#analysis-cancel").click()
                assert not errors
            finally:
                browser.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)
