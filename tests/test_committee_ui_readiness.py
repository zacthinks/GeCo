"""Committee UI contracts: distinct creation, badges and CV defaults."""

from pathlib import Path

from geometric_coder.ui.app import _committee_modal, _new_committee_modal
from dash import dcc, html, dash_table


def _all_ids(component):
    ids = []
    def visit(node):
        if isinstance(node, (list, tuple)):
            for item in node:
                visit(item)
        elif hasattr(node, "to_plotly_json"):
            if getattr(node, "id", None) is not None:
                ids.append(node.id)
            visit(getattr(node, "children", None))
    visit(component)
    return ids


def test_create_and_manager_forms_are_distinct_without_duplicate_ids():
    create = _new_committee_modal(dcc=dcc, html=html)
    manage = _committee_modal(dcc=dcc, html=html, dash_table=dash_table)
    created_ids = set(_all_ids(create))
    manager_ids = set(_all_ids(manage))
    assert {"committee-name", "committee-members", "committee-aggregation",
            "committee-save", "committee-create-close"}.issubset(created_ids)
    assert {"committee-table", "committee-rename", "committee-delete",
            "committee-close"}.issubset(manager_ids)
    assert created_ids.isdisjoint(manager_ids)


def test_develop_committee_actions_and_tuning_default_source_contract():
    source = (
        Path(__file__).resolve().parents[1] / "src" / "geometric_coder" /
        "ui" / "app.py"
    ).read_text(encoding="utf-8")
    assert 'id="focus-new-committee"' in source
    assert 'id="focus-manage-committees"' in source
    assert 'id="focus-train-committee"' in source
    assert 'id="focus-committee-badge"' in source
    assert 'id="focus-prepare-committee"' not in source
    assert '"focus_tune_hyperparameters", True' in source
    assert 'state.get("focus_tune_hyperparameters", True)' in source
    assert '"tune" in set(focus_tune_hyperparameter_values or [])' in source
