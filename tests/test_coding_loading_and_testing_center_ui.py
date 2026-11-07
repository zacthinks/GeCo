"""Source contract for slow human coding feedback and the beta control."""

from pathlib import Path

import geometric_coder.ui.app as app_module


def test_human_coding_outputs_trigger_existing_working_indicator() -> None:
    source = Path(app_module.__file__).read_text(encoding="utf-8")
    loading_targets = source.split('id="global-working-indicator"', 1)[1].split(
        "target_components={", 1
    )[1].split("}", 1)[0]
    for target in ("explore-code-status", "focus-annotation-status"):
        assert f'"{target}": "children"' in loading_targets


def test_testing_center_beta_button_uses_full_sidebar_width() -> None:
    source = Path(app_module.__file__).read_text(encoding="utf-8")
    button = source.split('"Testing Center (beta)"', 1)[1].split("),", 1)[0]
    assert 'id="focus-testing-center-button"' in button
    assert 'style={"width": "100%"}' in button
