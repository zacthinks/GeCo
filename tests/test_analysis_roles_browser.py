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
                    dropdown.click()
                    page.get_by_text(label, exact=True).last.click()

                def selected_model():
                    # Read the visible dropdown, not an implementation-specific
                    # React-Select class that varies across Dash versions.
                    return page.locator(
                        '[id*="analysis-role-predictor"]'
                    ).first.inner_text()

                page.locator("#analysis-new").click()
                page.locator("#analysis-name").fill("Roles browser")
                choose(page.locator("#analysis-role-values"), "Code: Theme")
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

                # A distribution supports parent-level counts. Switching to a
                # bar must reset that incompatible reduction, not reject a
                # chart that is otherwise valid. A correlation heatmap may
                # retain the count reduction; Jaccard may not.
                page.locator("#analysis-new").click()
                page.locator("#analysis-name").fill("Switching aggregation")
                choose(page.locator("#analysis-kind"), "Code-value distribution")
                choose(page.locator("#analysis-role-distribution"), "Code: Theme")
                choose(page.locator("#analysis-aggregate-level"), "Group by interview (and preceding keys)")
                choose(page.locator("#analysis-aggregate-method"), "Number of positive units per group")
                choose(page.locator("#analysis-kind"), "Bar plot — positive Code counts")
                page.wait_for_function(
                    "() => document.querySelector('#analysis-aggregate-method')"
                    "?.innerText?.includes('Any observed positive')", timeout=15000,
                )
                choose(page.locator("#analysis-role-values"), "Code: Theme")
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
                assert switched["spec"]["aggregation"]["method"] == "any"

                # An explicitly temporal role is visible but cannot execute
                # calendar bucketing in this phase, even if the raw data look
                # like valid dates.
                page.locator("#analysis-new").click()
                page.locator("#analysis-name").fill("Temporal unavailable")
                choose(page.locator("#analysis-role-values"), "Code: Theme")
                choose(page.locator("#analysis-role-group"), "Metadata: date")
                # The metadata settings arrive through a separate Dash callback.
                # Do not select the last generic role-value dropdown: before
                # that callback finishes, it can still be the Code's control.
                date_interpretation = page.locator(
                    'button[id*="analysis-role-value"][id*="group:metadata:date"]'
                )
                date_interpretation.wait_for(state="visible", timeout=15000)
                choose(date_interpretation, "Temporal / ISO date (Phase 3 grouping)")
                page.locator("#analysis-preview").click()
                page.locator("#analysis-status").get_by_text(
                    "Phase 3", exact=False
                ).wait_for(timeout=15000)
                page.locator("#analysis-save").click()
                page.locator("#analysis-status").get_by_text(
                    "Phase 3", exact=False
                ).wait_for(timeout=15000)
                assert not any(row["name"] == "Temporal unavailable"
                               for row in project.analyses())
                page.locator("#analysis-cancel").click()
                assert not errors
            finally:
                browser.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)
