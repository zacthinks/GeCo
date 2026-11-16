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
                # The app starts on Explore even when a Develop selection was
                # persisted in the session; open the real researcher workspace.
                page.get_by_text("Develop", exact=True).first.click()
                page.wait_for_function(
                    "() => getComputedStyle(document.querySelector('#focus-scale-state')).display !== 'none'",
                    timeout=15000,
                )
                assert not page.locator("#focus-active-area").is_visible()
                assert not page.locator("#focus-disabled-message").is_visible()
                assert page.locator("#focus-scale-top-summary").is_visible()
                assert page.locator("#focus-scale-count-summary button").is_disabled()

                # Switch away after ordinary UI activity, then back. The
                # active session must not replay its older workspace setting.
                for tab in ("Explore", "Develop", "Apply and Review", "Develop"):
                    page.get_by_text(tab, exact=True).first.click()
                    page.wait_for_function("""wanted => {
                        const selected = document.querySelector(
                            '#workspace-tabs .workflow-tab-selected');
                        return selected && selected.innerText.trim() === wanted;
                    }""", arg=tab, timeout=15000)
                    if tab == "Explore":
                        # A real control edit causes session-state persistence.
                        page.locator("#page-size").fill("250")
                        page.locator("#page-size").press("Tab")
                    elif tab == "Apply and Review":
                        page.locator("#apply-threshold").fill("0.61")
                        page.locator("#apply-threshold").press("Tab")
                page.wait_for_timeout(500)
                assert page.locator("#focus-scale-state").is_visible()
                assert not page.locator("#focus-disabled-message").is_visible()
                assert page.locator("#scale-prediction-page-size").input_value() == "1000"
                page.locator("#scale-prediction-page-size").fill("64")
                page.locator("#scale-prediction-page-size").press("Tab")
                try:
                    page.wait_for_function("""() => {
                    const gd = document.querySelector('#scale-prediction-map .js-plotly-plot');
                    return gd && gd.data && gd.data[0] &&
                        gd.data[0].customdata && gd.data[0].customdata.length === 64 &&
                        gd._fullLayout && gd._fullLayout.xaxis &&
                        document.querySelector('#scale-prediction-status')
                            ?.innerText.includes('64 of 130');
                }""", timeout=30000)
                    page.wait_for_timeout(350)
                except Exception:
                    print("Browser debug:", page.evaluate("""() => ({
                        workspace: document.querySelector('#workspace-tabs')?.innerText?.slice(0,400),
                        scalePlot: document.querySelector('#scale-prediction-map')?.innerText?.slice(0,300),
                        status: document.querySelector('#scale-prediction-status')?.innerText,
                        selectedConstruct: document.querySelector('#focus-construct-dropdown')?.innerText,
                        graphs: [...document.querySelectorAll('.js-plotly-plot')].length
                    })"""))
                    page.screenshot(path="scale-prediction-browser-failure.png", full_page=True)
                    raise
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
                assert "Browser observation 20" in original
                assert project.scale_comparisons(scale) == []

                manual = page.locator(".scale-manual-pairs")
                manual.locator("summary").click()
                manual.get_by_role("button", name="Choose as First").click()
                page.wait_for_function(
                    "() => document.querySelector('#scale-selection-summary')"
                    "?.innerText.includes('First selection: key: 20')",
                    timeout=15000,
                )

                page.evaluate("""() => {
                    const gd = document.querySelector('#scale-prediction-map .js-plotly-plot');
                    window.__oldScaleToken = gd.data[0].customdata[20][2];
                }""")
                page.locator("#scale-prediction-refresh").click()
                page.wait_for_function("""() => {
                    const gd = document.querySelector('#scale-prediction-map .js-plotly-plot');
                    return gd && gd.data && gd.data[0] &&
                           gd.data[0].customdata &&
                           gd.data[0].customdata[20][2] !== window.__oldScaleToken;
                }""")
                assert "Selected observation" in preview.inner_text()
                # Click a different point immediately after refreshing; the
                # FIRST click must inspect that point, not stale clickData.
                # Opening Manual Pairs can change scroll position, so restore
                # the visible Plotly viewport before calculating mouse coords.
                page.locator("#scale-prediction-map .js-plotly-plot").scroll_into_view_if_needed()
                second = page.evaluate("""() => {
                    const gd = document.querySelector('#scale-prediction-map .js-plotly-plot');
                    const i = 40;
                    const ax = gd._fullLayout.xaxis, ay = gd._fullLayout.yaxis;
                    const rect = gd.getBoundingClientRect();
                    return {
                        x: rect.left + ax._offset + ax.l2p(gd.data[0].x[i]),
                        y: rect.top + ay._offset + ay.l2p(gd.data[0].y[i])
                    };
                }""")
                page.mouse.click(second["x"], second["y"])
                page.wait_for_function(
                    "() => document.querySelector('#scale-prediction-preview')"
                    "?.innerText.includes('Browser observation 40')",
                    timeout=15000,
                )
                assert "Browser observation 40" in preview.inner_text()
                manual.get_by_role("button", name="Choose as Second").click()
                manual.get_by_role("button", name="Show Selected Pair").click()
                page.wait_for_function(
                    "() => document.querySelector('#scale-reading-first')"
                    "?.innerText.includes('Browser observation 20') || "
                    "document.querySelector('#scale-reading-second')"
                    "?.innerText.includes('Browser observation 20')",
                    timeout=15000,
                )
                readings = (
                    page.locator("#scale-reading-first").inner_text()
                    + page.locator("#scale-reading-second").inner_text()
                )
                assert "Browser observation 20" in readings
                assert "Browser observation 40" in readings
                assert project.scale_comparisons(scale) == []

                window = page.locator("#scale-context-window")
                window.fill("2")
                window.press("Tab")
                page.wait_for_timeout(600)
                assert "Selected observation" in preview.inner_text()
                assert project.scale_comparisons(scale) == []
                page.screenshot(path="scale-prediction-browser-success.png", full_page=True)

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
