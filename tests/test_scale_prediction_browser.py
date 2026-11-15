"""Live browser regression for Scale prediction-point inspection.

Run with: uv run --with playwright pytest tests/test_scale_prediction_browser.py
Requires an installed Chrome/Chromium executable; runs in the normal CI job.
"""
from __future__ import annotations

import shutil
import threading

import pandas as pd
import pytest
from werkzeug.serving import make_server

from geometric_coder import GeometricCoder


def test_plotly_mouse_click_preview_survives_refresh_and_context(tmp_path, monkeypatch):
    playwright = pytest.importorskip("playwright.sync_api")
    chrome = next(
        (path for executable in ("google-chrome", "chromium", "chromium-browser")
         if (path := shutil.which(executable)) is not None),
        None,
    )
    if chrome is None:
        pytest.skip("Chrome/Chromium unavailable for live browser interaction")

    from geometric_coder.ui.app import create_app

    count = 130  # Two prediction pages: a real page transition can invalidate selection.
    project = GeometricCoder.create(
        project_dir=tmp_path / "browser-preview.geco",
        data=pd.DataFrame({
            "key": list(range(count)),
            "text": [f"Browser observation {i}" for i in range(count)],
        }),
        keys=["key"], text="text", geometries=None,
    )
    scale = project.create_scale("Browser Scale", "Continuous")
    session_id = int(project.sessions()[0]["session_id"])
    project.patch_session_state(session_id, {
        "workspace": "focus", "focus_construct_id": scale, "focus_code_id": None,
    })
    fit = {"id": 13}
    monkeypatch.setattr(
        project, "scale_model_status",
        lambda scale_id: {"status": "current", "fit": {"scaler_fit_id": fit["id"]}},
    )
    monkeypatch.setattr(
        project, "scale_scores",
        lambda scale_id, ids, **kwargs: [
            {
                "observation_id": identity, "score": float(index),
                "epistemic_sd": 0.2 + index * 0.03,
                "scaler_fit_id": fit["id"],
            }
            for index, identity in enumerate(ids)
        ],
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
                page = browser.new_page(viewport={"width": 1440, "height": 1100})
                page.goto(f"http://127.0.0.1:{server.server_port}/")
                page.wait_for_function("""() => {
                    const gd = document.querySelector('#scale-prediction-map .js-plotly-plot');
                    return gd && gd.data && gd.data[0] &&
                        gd.data[0].customdata && gd.data[0].customdata.length > 20 &&
                        gd._fullLayout && gd._fullLayout.xaxis;
                }""", timeout=30000)
                page.locator("#scale-prediction-map .js-plotly-plot").scroll_into_view_if_needed()
                # Click a plotted WebGL point using actual browser mouse input,
                # not a direct Python callback invocation.
                xy = page.evaluate("""() => {
                    const gd = document.querySelector('#scale-prediction-map .js-plotly-plot');
                    const index = 20;
                    const ax = gd._fullLayout.xaxis;
                    const ay = gd._fullLayout.yaxis;
                    const rect = gd.getBoundingClientRect();
                    return {
                        x: rect.left + ax._offset + ax.l2p(gd.data[0].x[index]),
                        y: rect.top + ay._offset + ay.l2p(gd.data[0].y[index])
                    };
                }""")
                page.mouse.click(xy["x"], xy["y"])
                preview = page.locator("#scale-prediction-preview")
                page.wait_for_function(
                    "() => document.querySelector('#scale-prediction-preview')"
                    "?.innerText.includes('Selected observation')",
                    timeout=15000,
                )
                original = preview.inner_text()
                assert "Browser observation" in original
                assert project.scale_comparisons(scale) == []

                page.locator("#scale-prediction-refresh").click()
                page.wait_for_timeout(600)
                assert "Selected observation" in preview.inner_text()

                window = page.locator("#scale-context-window")
                window.fill("2")
                window.press("Tab")
                page.wait_for_timeout(600)
                assert "Selected observation" in preview.inner_text()
                assert project.scale_comparisons(scale) == []

                # The next page does not contain the inspected point.
                number = page.locator("#scale-prediction-page")
                number.fill("2")
                number.press("Tab")
                page.wait_for_function(
                    "() => document.querySelector('#scale-prediction-preview')"
                    "?.innerText.includes('Select a plotted observation')",
                    timeout=15000,
                )
                assert project.scale_comparisons(scale) == []
            finally:
                browser.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)
