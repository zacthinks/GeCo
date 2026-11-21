"""Browser regression for conditional numerical parent aggregation controls."""
from __future__ import annotations

import shutil
import threading

import pandas as pd
import pytest
from werkzeug.serving import make_server

from geometric_coder import GeometricCoder


def test_numerical_parent_controls_restore_and_save(tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    chrome = next((path for exe in ("google-chrome", "chromium", "chromium-browser")
                   if (path := shutil.which(exe)) is not None), None)
    if chrome is None:
        pytest.skip("Chrome/Chromium unavailable")
    from geometric_coder.ui.app import create_app

    project = GeometricCoder.create(
        project_dir=tmp_path / "numeric-aggregation-browser.geco",
        data=pd.DataFrame({
            "parent": [1, 1, 2, 2],
            "turn": [1, 2, 1, 2],
            "text": ["one", "two", "three", "four"],
            "score": [1., 4., 2., 6.],
            "group": ["A"] * 4,
        }),
        keys=["parent", "turn"], text="text", metadata=["score", "group"],
        geometries=None,
    )
    roles = {
        "values": {"variable": {"type": "metadata", "field": "score", "value": "numeric"}},
        "group": {"variable": {"type": "metadata", "field": "group", "value": "categorical"}},
    }
    percentile_id = project.save_analysis("Parent percentile", {
        "schema_version": 2, "kind": "summary_bar", "roles": roles,
        "aggregation": {"level": "parent", "numeric_method": "percentile",
                        "numeric_percentile": 75},
    })
    bounds_id = project.save_analysis("Parent bounds", {
        "schema_version": 2, "kind": "summary_bar", "roles": roles,
        "aggregation": {"level": "parent", "numeric_method": "proportion_within_bounds",
                        "numeric_bounds": {"lower": 2, "lower_inclusive": True,
                                           "upper": 6, "upper_inclusive": True}},
    })
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

                def select_and_edit(name):
                    page.goto(f"http://127.0.0.1:{server.server_port}/")
                    page.get_by_text("Analyses", exact=True).first.click()
                    dropdown = page.locator("#analysis-select")
                    dropdown.click()
                    page.get_by_text(name, exact=True).filter(visible=True).last.click()
                    page.locator("#analysis-edit").click()
                    page.locator("#analysis-parent-numeric-box").wait_for(state="visible")

                select_and_edit("Parent percentile")
                page.locator("#analysis-parent-percentile-box").wait_for(state="visible")
                p = page.locator("#analysis-parent-percentile")
                assert p.input_value() == "75"
                p.fill("25")
                p.press("Tab")
                page.wait_for_load_state("networkidle")
                page.locator("#analysis-save").click()
                page.locator("#analysis-editor").wait_for(state="hidden")
                aggregate = project.analysis(percentile_id)["spec"]["aggregation"]
                assert aggregate["numeric_method"] == "percentile"
                assert aggregate["numeric_percentile"] == 25

                select_and_edit("Parent bounds")
                page.locator("#analysis-parent-bounds-box").wait_for(state="visible")
                lower = page.locator("#analysis-parent-lower-bound")
                upper = page.locator("#analysis-parent-upper-bound")
                assert lower.input_value() == "2"
                assert upper.input_value() == "6"
                upper.fill("4")
                upper.press("Tab")
                page.locator("#analysis-parent-upper-inclusive input").uncheck()
                page.wait_for_load_state("networkidle")
                page.locator("#analysis-save").click()
                page.locator("#analysis-editor").wait_for(state="hidden")
                aggregate = project.analysis(bounds_id)["spec"]["aggregation"]
                assert aggregate["numeric_method"] == "proportion_within_bounds"
                assert aggregate["numeric_bounds"]["lower"] == 2
                assert aggregate["numeric_bounds"]["upper"] == 4
                assert aggregate["numeric_bounds"]["lower_inclusive"] is True
                assert aggregate["numeric_bounds"]["upper_inclusive"] is False
            finally:
                browser.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)
