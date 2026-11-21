"""Browser smoke test for visualization-first saved analyses.

Requires Playwright and a system Chrome/Chromium; CI provides both.
"""
from __future__ import annotations

import shutil
import threading

import pandas as pd
import pytest
from werkzeug.serving import make_server

from geometric_coder import GeometricCoder, CountGeometry


def test_analysis_role_editor_preview_save_edit_and_incompatible_chart(tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    chrome = next((path for executable in ("google-chrome", "chromium", "chromium-browser")
                   if (path := shutil.which(executable)) is not None), None)
    if chrome is None:
        pytest.skip("Chrome/Chromium unavailable")

    from geometric_coder.ui.app import create_app

    # One initialized project, one trained classifier, and one committee for
    # every browser scenario; do not refit or reload an app per assertion.
    project = GeometricCoder.create(
        project_dir=tmp_path / "role-browser.geco",
        data=pd.DataFrame({
            "interview": [1, 1, 1, 2, 2, 2],
            "id": [1, 2, 3, 1, 2, 3],
            "text": [
                "excellent useful", "great helpful", "nice excellent",
                "bad confusing", "terrible weak", "poor unclear",
            ],
            "group": ["A", "A", "A", "B", "B", "B"],
            "date": [
                "2026-01-02", "2026-02-01", "2026-03-09",
                "2026-04-01", "2026-05-03", "2026-06-04",
            ],
        }),
        keys=["interview", "id"], text="text",
        metadata=["group", "date"],
        geometries={"lexical": CountGeometry(lemmatize=False, weighting="tfidf")},
    )
    code = project.create_code("Theme")
    ids = [row["observation_id"] for row in project.database.list_units()]
    project.annotate(ids[0], code, "positive")
    project.annotate(ids[1], code, "positive")
    project.annotate(ids[3], code, "negative")
    project.ensure_default_classifier_specs(code_id=code)
    classifier = int(project.classifier_specs(code_id=code)[0]["classifier_spec_id"])
    project.train_classifier(code_id=code, classifier_spec_id=classifier)
    committee = project.create_classifier_committee(
        code_id=code, name="Theme committee",
        classifier_spec_ids=[classifier], aggregation="mean",
    )
    legacy_id = project.save_analysis("Legacy grouped", {
        "kind": "bar", "measure": "count",
        "variables": [{"type": "code", "code_id": code,
                       "source": "labels", "value": "binary"},
                      {"type": "metadata", "field": "group"}],
        "aggregation": {"level": "interview", "method": "minimum", "threshold": 2},
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
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(f"http://127.0.0.1:{server.server_port}/")
                page.get_by_text("Analyses", exact=True).first.click()

                def choose(dropdown, label):
                    dropdown.wait_for(state="visible", timeout=15000)
                    dropdown.click()
                    # Other visualization roles stay mounted but hidden. Restrict
                    # the option match to visible text in the open dropdown.
                    page.get_by_text(label, exact=True).filter(visible=True).last.click()

                def selected_model():
                    # Read the visible dropdown, not an implementation-specific
                    # React-Select class that varies across Dash versions.
                    return page.locator(
                        '[id*="analysis-role-predictor"]'
                    ).first.inner_text()

                page.locator("#analysis-new").click()
                page.locator("#analysis-name").fill("Roles browser")
                choose(page.locator("#analysis-role-values"), "Code: Theme")
                choose(page.locator("#analysis-bar-statistic"), "Sum")
                page.locator("#analysis-preview").click()
                page.wait_for_function(
                    "() => document.querySelector('#analysis-figure .js-plotly-plot')"
                    "?.data?.[0]?.y?.[0] === 2", timeout=20000,
                )

                # Real dynamic controls: pick a classifier and confirm that
                # adding another role does not reset the trained model choice.
                choose(page.locator('[id*="analysis-role-source"]').first,
                       "Predictions only")
                choose(page.locator('[id*="analysis-role-predictor"]').first,
                       next(label for label in [
                           f"Classifier: {row['name']}"
                           for row in project.classifier_specs(code)
                           if int(row["classifier_spec_id"]) == classifier
                       ]))
                choose(page.locator("#analysis-role-group"), "Metadata: group")
                page.locator('[id*="analysis-role-source"]').first.wait_for()
                assert selected_model().startswith("Classifier:")
                page.locator("#analysis-preview").click()
                page.locator("#analysis-status").get_by_text(
                    "Preview unavailable:", exact=False
                ).wait_for(state="hidden", timeout=15000)
                page.locator("#analysis-save").click()
                page.wait_for_function(
                    "() => getComputedStyle(document.querySelector('#analysis-editor'))"
                    ".display === 'none'", timeout=15000,
                )
                created = next(row for row in project.analyses()
                               if row["name"] == "Roles browser")
                descriptor = created["spec"]["roles"]["values"][0]["variable"]
                assert descriptor["source"] == "predictions"
                assert descriptor["predictor_kind"] == "classifier"
                assert descriptor["predictor_id"] == classifier

                # Switch to committee on the same fitted model. The selection
                # must survive changing an unrelated Group role.
                page.locator("#analysis-edit").click()
                page.wait_for_function(
                    "() => getComputedStyle(document.querySelector('#analysis-editor'))"
                    ".display !== 'none'", timeout=15000,
                )
                choose(page.locator('[id*="analysis-role-predictor"]').first,
                       "Committee: Theme committee")
                choose(page.locator("#analysis-role-group"), "Metadata: date")
                assert selected_model() == "Committee: Theme committee"
                page.locator("#analysis-save").click()
                page.wait_for_function(
                    "() => getComputedStyle(document.querySelector('#analysis-editor'))"
                    ".display === 'none'", timeout=15000,
                )
                changed = project.analysis(created["analysis_id"])["spec"]
                assert changed["roles"]["values"][0]["variable"]["predictor_kind"] == "committee"
                assert changed["roles"]["values"][0]["variable"]["predictor_id"] == committee

                # Save immediately after changing the source and model. These
                # mounted controls must win over a lagging role-config Store.
                page.locator("#analysis-edit").click()
                choose(page.locator('button[id*="analysis-role-source"]').first,
                       "Human labels only")
                page.locator("#analysis-save").click()
                page.wait_for_function(
                    "() => getComputedStyle(document.querySelector('#analysis-editor'))"
                    ".display === 'none'", timeout=15000,
                )
                assert project.analysis(created["analysis_id"])["spec"]["roles"]["values"][0][
                    "variable"]["source"] == "labels"

                page.locator("#analysis-edit").click()
                choose(page.locator('button[id*="analysis-role-source"]').first,
                       "Predictions only")
                choose(page.locator('button[id*="analysis-role-predictor"]').first,
                       next(f"Classifier: {row['name']}"
                            for row in project.classifier_specs(code)
                            if int(row["classifier_spec_id"]) == classifier))
                page.locator("#analysis-save").click()
                page.wait_for_function(
                    "() => getComputedStyle(document.querySelector('#analysis-editor'))"
                    ".display === 'none'", timeout=15000,
                )
                restored_predictor = project.analysis(created["analysis_id"])["spec"][
                    "roles"]["values"][0]["variable"]
                assert restored_predictor["source"] == "predictions"
                assert restored_predictor["predictor_kind"] == "classifier"
                assert restored_predictor["predictor_id"] == classifier

                # v1 edit and duplicate restore aggregation without changing
                # either the existing identity or its output.
                before = project.run_analysis(legacy_id)
                choose(page.locator("#analysis-select"), "Legacy grouped")
                page.locator("#analysis-edit").click()
                page.wait_for_function(
                    "() => document.querySelector('#analysis-name')"
                    "?.value === 'Legacy grouped'", timeout=15000,
                )
                assert page.locator("#analysis-aggregate-level").inner_text().find(
                    "interview"
                ) >= 0
                assert page.locator("#analysis-aggregate-threshold").input_value() == "2"
                page.locator("#analysis-save").click()
                page.wait_for_function(
                    "() => getComputedStyle(document.querySelector('#analysis-editor'))"
                    ".display === 'none'", timeout=15000,
                )
                assert project.analysis(legacy_id)["spec"]["schema_version"] == 1
                assert project.run_analysis(legacy_id) == before
                page.locator("#analysis-duplicate").click()
                page.wait_for_function(
                    "() => document.querySelector('#analysis-name')"
                    "?.value.includes('(copy)')", timeout=15000,
                )
                page.locator("#analysis-save").click()
                page.wait_for_function(
                    "() => getComputedStyle(document.querySelector('#analysis-editor'))"
                    ".display === 'none'", timeout=15000,
                )
                duplicate = next(row for row in project.analyses()
                                 if row["name"] == "Legacy grouped (copy)")
                assert duplicate["analysis_id"] != legacy_id
                assert project.run_analysis(duplicate["analysis_id"]) == before

                # Unified bars support binary Code sums and parent-level
                # positive counts, while keeping the source and denominator explicit.
                page.locator("#analysis-new").click()
                page.locator("#analysis-name").fill("Switching aggregation")
                choose(page.locator("#analysis-role-values"), "Code: Theme")
                choose(page.locator("#analysis-role-group"), "Metadata: group")
                choose(page.locator("#analysis-bar-statistic"), "Sum")
                choose(page.locator("#analysis-aggregate-level"), "Group by interview (and preceding keys)")
                choose(page.locator("#analysis-aggregate-method"), "Number of positive units per group")
                page.locator("#analysis-preview").click()
                page.locator("#analysis-status").get_by_text(
                    "Preview unavailable:", exact=False
                ).wait_for(state="hidden", timeout=15000)
                page.locator("#analysis-save").click()
                page.wait_for_function(
                    "() => getComputedStyle(document.querySelector('#analysis-editor'))"
                    ".display === 'none'", timeout=15000,
                )
                switched = next(row for row in project.analyses()
                                if row["name"] == "Switching aggregation")
                assert switched["spec"]["kind"] == "bar_chart"
                assert switched["spec"]["aggregation"]["method"] == "count"

                # Numerical parent reduction and grouped mean remain available
                # in the same new Bar chart workflow.
                page.locator("#analysis-new").click()
                page.locator("#analysis-name").fill("Numerical parent count")
                choose(page.locator("#analysis-role-values"), "Code: Theme")
                choose(page.locator("#analysis-role-group"), "Metadata: group")
                choose(page.locator("#analysis-bar-statistic"), "Mean")
                choose(page.locator("#analysis-aggregate-level"),
                       "Group by interview (and preceding keys)")
                choose(page.locator("#analysis-aggregate-method"),
                       "Number of positive units per group")
                page.locator("#analysis-save").click()
                page.wait_for_function(
                    "() => getComputedStyle(document.querySelector('#analysis-editor'))"
                    ".display === 'none'", timeout=15000,
                )
                parent_count = next(row for row in project.analyses()
                                    if row["name"] == "Numerical parent count")
                assert parent_count["spec"]["aggregation"]["method"] == "count"
                assert project.run_analysis(parent_count["analysis_id"])["series"][0]["values"] == [2.0, 0.0]

                # Temporal grouping now exposes a calendar-only selector and
                # explicitly defaults to month. The recipe must persist that
                # choice rather than rejecting an unspecified calendar unit.
                page.locator("#analysis-new").click()
                page.locator("#analysis-name").fill("Temporal month")
                choose(page.locator("#analysis-role-values"), "Code: Theme")
                choose(page.locator("#analysis-role-group"), "Metadata: date")
                date_interpretation = page.locator(
                    'button[id*="analysis-role-value"][id*="group:metadata:date"]'
                )
                date_interpretation.wait_for(state="visible", timeout=15000)
                choose(date_interpretation, "Date/time")
                calendar = page.locator(
                    'button[id*="analysis-role-transform"][id*="group:metadata:date"]'
                )
                calendar.wait_for(state="visible", timeout=15000)
                page.wait_for_function(
                    "() => Array.from(document.querySelectorAll('button'))"
                    ".some(el => el.id.includes('analysis-role-transform') && "
                    "el.id.includes('group:metadata:date') && "
                    "el.textContent.includes('Calendar month'))",
                    timeout=15000,
                )
                page.locator("#analysis-save").click()
                page.locator("#analysis-editor").wait_for(state="hidden")
                temporal = next(row for row in project.analyses()
                                if row["name"] == "Temporal month")
                assert temporal["spec"]["roles"]["group"]["transform"] == {
                    "kind": "calendar", "unit": "month",
                }
                assert not errors
            finally:
                browser.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_parent_aggregation_controls_for_legacy_distribution_and_heatmap(tmp_path):
    """Editing saved parent charts must expose Code reductions and thresholds."""
    playwright = pytest.importorskip("playwright.sync_api")
    chrome = next((path for exe in ("google-chrome", "chromium", "chromium-browser")
                   if (path := shutil.which(exe)) is not None), None)
    if chrome is None:
        pytest.skip("Chrome/Chromium unavailable")
    from geometric_coder.ui.app import create_app

    project = GeometricCoder.create(
        project_dir=tmp_path / "legacy-parent-ui.geco",
        data=pd.DataFrame({
            "interview": [1, 1, 2, 2],
            "turn": [1, 2, 1, 2],
            "text": ["good", "bad", "great", "poor"],
        }),
        keys=["interview", "turn"], text="text", metadata=[], geometries=None,
    )
    code = project.create_code("Theme")
    other = project.create_code("Other")
    for unit, label in zip(project.database.list_units(),
                           ["positive", "negative", "positive", "negative"], strict=True):
        project.annotate(unit["observation_id"], code, label)
    for unit, label in zip(project.database.list_units(),
                           ["negative", "positive", "negative", "positive"], strict=True):
        project.annotate(unit["observation_id"], other, label)
    references = [
        {"type": "code", "code_id": identifier, "source": "labels", "value": "binary"}
        for identifier in (code, other)
    ]
    distribution_id = project.save_analysis("Parent distribution", {
        "kind": "distribution", "measure": "count", "variables": references[:1],
        "aggregation": {"level": "interview", "method": "minimum", "threshold": 2},
    })
    heatmap_id = project.save_analysis("Parent heatmap", {
        "kind": "heatmap", "measure": "jaccard", "variables": references,
        "aggregation": {"level": "interview", "method": "minimum", "threshold": 2},
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
                page.goto(f"http://127.0.0.1:{server.server_port}/")
                page.get_by_text("Analyses", exact=True).first.click()

                def choose(selector, label):
                    dropdown = page.locator(selector)
                    dropdown.click()
                    page.get_by_text(label, exact=True).filter(visible=True).last.click()

                for name, chart_id in (
                    ("Parent distribution", distribution_id),
                    ("Parent heatmap", heatmap_id),
                ):
                    choose("#analysis-select", name)
                    page.locator("#analysis-edit").click()
                    page.wait_for_function(
                        "(expected) => document.querySelector('#analysis-name')"
                        "?.value === expected && document.querySelector('#analysis-aggregate-level')"
                        "?.textContent.includes('interview')",
                        arg=name, timeout=15000,
                    )
                    page.locator("#analysis-code-aggregate-box").wait_for(state="visible")
                    page.locator("#analysis-threshold-box").wait_for(state="visible")
                    assert page.locator("#analysis-aggregate-threshold").input_value() == "2"
                    assert page.locator("#analysis-threshold-box").get_by_text(
                        "Minimum positive count", exact=True
                    ).is_visible()
                    # Only one inclusive positive-threshold operation remains.
                    dropdown = page.locator("#analysis-aggregate-method")
                    dropdown.click()
                    page.get_by_text("At least N positive judgments", exact=True).filter(
                        visible=True
                    ).last.wait_for()
                    assert page.get_by_text(
                        "Any observed positive (0 or 1)", exact=True
                    ).filter(visible=True).count() == 0
                    page.keyboard.press("Escape")
                    # Dash renders the number input as a controlled React
                    # component. Commit the edited value and finish its
                    # callback cycle before Save samples all form States.
                    threshold = page.locator("#analysis-aggregate-threshold")
                    threshold.fill("1")
                    threshold.press("Tab")
                    page.wait_for_function(
                        "() => document.querySelector('#analysis-aggregate-threshold')"
                        "?.value === '1'",
                        timeout=15000,
                    )
                    page.wait_for_load_state("networkidle")
                    assert threshold.input_value() == "1"
                    page.locator("#analysis-save").click()
                    page.locator("#analysis-editor").wait_for(state="hidden")
                    saved = project.analysis(chart_id)["spec"]["aggregation"]
                    assert saved["method"] == "minimum"
                    assert saved["threshold"] == 1
            finally:
                browser.close()
    finally:
        server.shutdown()
        thread.join(timeout=10)
