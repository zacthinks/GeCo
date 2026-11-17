"""Phase 2B Scale evidence adapter and durable fit lifecycle, with numerical test double."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from geometric_coder import GeometricCoder
from geometric_coder.scale_training import build_scale_training_data


class RecordingScaler:
    """Test double for Team A's numerical contract; never performs Davidson fitting."""

    def __init__(self, **kwargs):
        self.config = kwargs
        self.success_ = True
        self.message_ = "mock converged"
        self.n_iterations_ = 2
        self.optimality_ = 0.0

    def fit(self, X, pairs, outcomes, *, reference_X=None):
        self.pairs_seen = np.asarray(pairs).copy()
        self.outcomes_seen = np.asarray(outcomes).copy()
        self.rows_seen = X.shape[0]
        self.reference_rows_seen = reference_X.shape[0]
        self.posterior_ = None
        return self

    def build_posterior(self):
        self.posterior_ = "mock posterior"
        return self

    def predict_scores(self, X):
        first_column = X[:, 0].toarray() if sparse.issparse(X) else X[:, 0]
        return np.asarray(first_column).ravel()

    def posterior_position_variance(self, X):
        return np.full(X.shape[0], 0.25)


class FailingScaler(RecordingScaler):
    def fit(self, X, pairs, outcomes, *, reference_X=None):
        raise RuntimeError("simulated optimizer failure")


def _setup(tmp_path: Path, *, sparse_geometry=False):
    project = GeometricCoder.create(
        project_dir=tmp_path / "scale-fit.geco",
        data=pd.DataFrame({
            "key": [11, 22, 33, 44, 55],
            "text": ["first", "second", "third", "fourth", "fifth"],
        }),
        keys=["key"], text="text", metadata=[], geometries=None,
    )
    matrix = np.asarray([
        [5., 1.], [4., 0.], [3., 1.], [2., 0.], [1., 1.]
    ])
    name = "features_sparse" if sparse_geometry else "features_dense"
    path, is_sparse = project.artifacts.save_geometry_matrix(
        name, sparse.csr_matrix(matrix) if sparse_geometry else matrix,
    )
    geom_id = project.database.register_geometry(
        name=name, class_name="SyntheticGeometry", modality="text", public=True,
        config={}, dependency_names=[], matrix_path=path, matrix_sparse=is_sparse,
        state_path=None, supports_query=False, supports_text_transform=False,
    )
    rep = project.database.representation_for_legacy("geometry", geom_id)
    scale_id = project.create_scale("Quality")
    ids = [project.database.observation_for_unit(i) for i in range(1, 6)]
    return project, scale_id, int(rep["representation_id"]), geom_id, ids


@pytest.mark.parametrize("sparse_geometry", [False, True])
def test_human_pair_orientation_repeats_parity_and_skip(tmp_path, sparse_geometry):
    project, scale, rep, geom, (a, b, c, d, _) = _setup(
        tmp_path, sparse_geometry=sparse_geometry
    )
    e1 = project.record_scale_comparison(scale, a, b, "first_higher")
    e2 = project.record_scale_comparison(scale, b, a, "second_higher")
    e3 = project.record_scale_comparison(scale, b, c, "parity")
    project.record_scale_comparison(scale, a, c, "unsure")
    project.record_scale_comparison(scale, c, d, "second_higher", origin="model_apply")
    data = build_scale_training_data(project, scale, geom)
    assert data.comparison_event_ids == (e1, e2, e3)
    assert data.outcomes.tolist() == [0, 0, 1]
    assert data.pairs.tolist() == [[0, 1], [0, 1], [1, 2]]
    assert data.observation_ids == (a, b, c)
    assert data.features.shape == (3, 2)
    assert sparse.issparse(data.features) == sparse_geometry
    project.configure_scale_model(scale, rep)
    result = project.train_scale(scale, estimator_factory=RecordingScaler)
    assert result["status"] == "current"
    loaded = project.artifacts.load_model(project.project_dir, result["fit"]["artifact_path"])
    assert loaded.outcomes_seen.tolist() == [0, 0, 1]
    assert loaded.pairs_seen.tolist() == [[0, 1], [0, 1], [1, 2]]
    assert project.scale_scores(scale, [a, c], include_uncertainty=True) == [
        {"observation_id": a, "score": 5.0, "epistemic_sd": 0.5,
         "scaler_fit_id": result["fit"]["scaler_fit_id"], "fit_status": "current"},
        {"observation_id": c, "score": 3.0, "epistemic_sd": 0.5,
         "scaler_fit_id": result["fit"]["scaler_fit_id"], "fit_status": "current"},
    ]


def test_current_fit_reused_and_staleness_after_corrections(tmp_path):
    project, scale, rep, geom, (a, b, c, d, _) = _setup(tmp_path)
    first = project.record_scale_comparison(scale, a, b, "first_higher")
    project.record_scale_comparison(scale, b, c, "second_higher")
    project.configure_scale_model(scale, rep)
    trained = project.train_scale(scale, estimator_factory=RecordingScaler)
    fit_id = trained["fit"]["scaler_fit_id"]

    def forbidden(**kwargs):
        raise AssertionError("unchanged fit unexpectedly retrained")
    reused = project.train_scale(scale, estimator_factory=forbidden)
    assert reused["reused"] and reused["fit"]["scaler_fit_id"] == fit_id

    # The number of effective comparisons stays two but its semantics change.
    project.record_scale_comparison(
        scale, b, a, "first_higher", supersedes_event_id=first,
    )
    assert project.scale_model_status(scale)["status"] == "stale"
    assert project.scale_model_status(scale)["fit"]["scaler_fit_id"] == fit_id
    with pytest.raises(Exception, match="stale"):
        project.scale_scores(scale, [a])
    assert project.scale_scores(scale, [a], allow_stale=True)[0]["fit_status"] == "stale"
    refit = project.train_scale(scale, estimator_factory=RecordingScaler)
    assert refit["fit"]["scaler_fit_id"] != fit_id
    assert refit["status"] == "current"
    reopened = GeometricCoder.open(project.project_dir)
    assert reopened.scale_model_status(scale)["status"] == "current"
    assert reopened.scale_scores(scale, [c])[0]["score"] == 3.0

    project.update_scale(scale, "Quality revised", "Different operationalization")
    assert project.scale_model_status(scale)["status"] == "stale"
    assert project.train_scale(scale, estimator_factory=RecordingScaler)["status"] == "insufficient_evidence"


def test_failed_fit_retains_previous_and_eligibility_changes(tmp_path):
    project, scale, rep, geom, (a, b, c, d, _) = _setup(tmp_path)
    project.record_scale_comparison(scale, a, b, "first_higher")
    project.record_scale_comparison(scale, c, d, "second_higher")
    project.configure_scale_model(scale, rep)
    initial = project.train_scale(scale, estimator_factory=RecordingScaler)
    previous_id = initial["fit"]["scaler_fit_id"]
    project.record_scale_comparison(scale, a, d, "parity")
    assert project.scale_model_status(scale)["status"] == "stale"
    with pytest.raises(RuntimeError, match="simulated optimizer failure"):
        project.train_scale(scale, estimator_factory=FailingScaler)
    assert project.scale_model_status(scale)["fit"]["scaler_fit_id"] == previous_id
    project.update_scale_eligibility(scale, {"kind": "all"})
    # No-op eligibility update does not invalidate a snapshot.
    assert project.scale_model_status(scale)["status"] == "stale"
    with project.database.connect() as con:
        assert con.execute(
            "SELECT COUNT(*) FROM scaler_fits WHERE scale_id=?", (scale,)
        ).fetchone()[0] == 1


def test_unsupported_view_and_metadata_only_status(tmp_path, monkeypatch):
    project, scale, rep, geom, (a, b, c, d, _) = _setup(tmp_path)
    project.record_scale_comparison(scale, a, b, "first_higher")
    project.record_scale_comparison(scale, c, d, "second_higher")
    project.configure_scale_model(scale, rep)
    project.train_scale(scale, estimator_factory=RecordingScaler)
    monkeypatch.setattr(project, "geometry_matrix", lambda *a, **k: 1/0)
    monkeypatch.setattr(project, "geometry_page_rows", lambda *a, **k: 1/0)
    assert project.scale_model_status(scale)["status"] == "current"


def test_schema_19_to_20_upgrade_preserves_comparison_history(tmp_path):
    project, scale, rep, geom, (a, b, c, d, _) = _setup(tmp_path)
    event = project.record_scale_comparison(scale, a, b, "parity")
    path = project.project_dir / project.DATABASE_FILENAME
    with sqlite3.connect(path) as con:
        con.execute("DROP TABLE scaler_fits")
        con.execute("DROP TABLE scaler_specs")
        con.execute("UPDATE project SET schema_version = 19")
    reopened = GeometricCoder.open(project.project_dir)
    assert reopened.database.project_metadata()["schema_version"] == 21
    assert reopened.scale_comparisons(scale)[0]["comparison_event_id"] == event
    assert reopened.scale_model_status(scale)["status"] == "not_configured"


def test_external_geometry_uses_requested_key_order(tmp_path):
    """Provider rows must follow requested stable keys, not numeric observation IDs."""

    class KeyedProvider:
        def geometry_matrix(self, external_ref, keys):
            assert external_ref == {"geometry": "keyed"}
            return np.asarray([
                [float(item["key"]), 2.0] for item in keys
            ])

    project = GeometricCoder.create_external(
        project_dir=tmp_path / "linked.geco",
        data=pd.DataFrame({
            "key": [101, 102, 103, 104],
            "text": ["a", "b", "c", "d"],
        }),
        keys=["key"], text="text", metadata=[], external_provider=KeyedProvider(),
    )
    geom = project.database.register_geometry(
        name="external_features", class_name="ExternalGeometry", modality="text",
        public=True, config={}, dependency_names=[], matrix_path="",
        matrix_sparse=False, state_path=None, supports_query=False,
        supports_text_transform=False, storage_kind="external",
        external_ref={"geometry": "keyed"},
    )
    representation = project.database.representation_for_legacy("geometry", geom)
    scale_id = project.create_scale("Strongness")
    ids = [project.database.observation_for_unit(i) for i in range(1, 5)]
    project.record_scale_comparison(scale_id, ids[3], ids[0], "first_higher")
    project.record_scale_comparison(scale_id, ids[2], ids[1], "parity")
    data = build_scale_training_data(project, scale_id, geom)
    assert data.features[:, 0].tolist() == [101., 102., 103., 104.]
    assert data.pairs.tolist() == [[0, 3], [1, 2]]
    assert data.outcomes.tolist() == [2, 1]
    project.configure_scale_model(scale_id, representation["representation_id"])
    fit = project.train_scale(scale_id, estimator_factory=RecordingScaler)
    assert fit["status"] == "current"
    assert project.scale_scores(scale_id, [ids[2], ids[0]])[0]["score"] == 103.


def test_migration_failure_rolls_back_all_new_tables(tmp_path, monkeypatch):
    from geometric_coder.storage import database as storage_database

    project, _, _, _, _ = _setup(tmp_path)
    db_path = project.project_dir / project.DATABASE_FILENAME
    with sqlite3.connect(db_path) as connection:
        connection.execute("DROP TABLE scaler_fits")
        connection.execute("DROP TABLE scaler_specs")
        connection.execute("UPDATE project SET schema_version = 19")
    original = storage_database._MIGRATIONS[19]

    def fail_after_table_creation(connection):
        original(connection)
        raise RuntimeError("forced schema-20 migration failure")

    monkeypatch.setitem(storage_database._MIGRATIONS, 19, fail_after_table_creation)
    with pytest.raises(RuntimeError, match="forced schema-20"):
        GeometricCoder.open(project.project_dir)
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT schema_version FROM project").fetchone()[0] == 19
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE name='scaler_specs'"
        ).fetchone() is None
    monkeypatch.setitem(storage_database._MIGRATIONS, 19, original)
    assert GeometricCoder.open(project.project_dir).database.project_metadata()["schema_version"] == 21


def test_no_comparisons_and_parity_only_stay_untrained(tmp_path):
    project, scale, rep, geom, (a, b, c, d, _) = _setup(tmp_path)
    project.configure_scale_model(scale, rep)
    assert project.train_scale(scale, estimator_factory=RecordingScaler)["status"] == "insufficient_evidence"
    project.record_scale_comparison(scale, a, b, "unsure")
    assert project.train_scale(scale, estimator_factory=RecordingScaler)["status"] == "insufficient_evidence"
    project.record_scale_comparison(scale, a, b, "parity")
    project.record_scale_comparison(scale, c, d, "parity")
    assert project.train_scale(scale, estimator_factory=RecordingScaler)["status"] == "insufficient_evidence"
    assert project.scale_model_status(scale)["status"] == "not_trained"


def test_real_davidson_end_to_end_after_phase2a_merge(tmp_path):
    """Skip until Phase 2A's production module is available on this branch."""
    pytest.importorskip("geometric_coder.davidson_scaler")
    project, scale, rep, geom, (a, b, c, d, e) = _setup(tmp_path)
    project.configure_scale_model(scale, rep, hyperparameters={
        "score_prior_sd": 1.0, "feature_normalization": "none"
    })
    for first, second in [
        (a, d), (a, e), (b, d), (b, e), (c, d), (c, e)
    ]:
        project.record_scale_comparison(scale, first, second, "first_higher")
    first_fit = project.train_scale(scale)
    assert first_fit["status"] == "current"
    scored = project.scale_scores(scale, [a, e])
    assert scored[0]["score"] > scored[1]["score"]
    project.record_scale_comparison(scale, a, c, "first_higher")
    assert project.scale_model_status(scale)["status"] == "stale"
    second_fit = project.train_scale(scale)
    assert second_fit["status"] == "current"
    assert second_fit["fit"]["scaler_fit_id"] != first_fit["fit"]["scaler_fit_id"]
    reopened = GeometricCoder.open(project.project_dir)
    assert reopened.scale_model_status(scale)["status"] == "current"
    reopened_scores = reopened.scale_scores(scale, [a, e], include_uncertainty=True)
    assert reopened_scores[0]["score"] > reopened_scores[1]["score"]
    assert reopened_scores[0]["scaler_fit_id"] == second_fit["fit"]["scaler_fit_id"]


def test_eligibility_and_code_dependency_revision_stale_even_without_count_change(tmp_path):
    project, scale, rep, geom, (a, b, c, d, e) = _setup(tmp_path)
    project.record_scale_comparison(scale, a, b, "first_higher")
    project.record_scale_comparison(scale, c, d, "second_higher")
    project.configure_scale_model(scale, rep)
    project.train_scale(scale, estimator_factory=RecordingScaler)
    assert project.scale_model_status(scale)["status"] == "current"

    gate = project.create_code("Eligible")
    for obs in [a, b, c, d, e]:
        project.annotate(obs, gate, "positive", origin="human_direct")
    project.update_scale_eligibility(scale, {
        "kind": "filtered",
        "code_filters": [{"code_id": gate, "value": "positive", "source": "human"}],
    })
    assert project.scale_model_status(scale)["status"] == "stale"
    project.train_scale(scale, estimator_factory=RecordingScaler)
    assert project.scale_model_status(scale)["status"] == "current"
    before = project.scale_model_status(scale)["comparison_count"]

    # Code definition revisions affect filter semantics even when membership is same.
    project.update_code(gate, name="Eligible", description="New meaning")
    assert project.scale_model_status(scale)["comparison_count"] == before
    assert project.scale_model_status(scale)["status"] == "stale"


def test_changing_geometry_invalidates_frozen_fit(tmp_path):
    project, scale, rep, geom, (a, b, c, d, _) = _setup(tmp_path)
    project.record_scale_comparison(scale, a, b, "first_higher")
    project.record_scale_comparison(scale, c, d, "second_higher")
    project.configure_scale_model(scale, rep)
    project.train_scale(scale, estimator_factory=RecordingScaler)
    assert project.scale_model_status(scale)["status"] == "current"
    path, is_sparse = project.artifacts.save_geometry_matrix(
        "alternate", np.ones((5, 2), dtype=float)
    )
    other_geom = project.database.register_geometry(
        name="alternate", class_name="SyntheticGeometry", modality="text",
        public=True, config={}, dependency_names=[], matrix_path=path,
        matrix_sparse=is_sparse, state_path=None, supports_query=False,
        supports_text_transform=False,
    )
    other_rep = project.database.representation_for_legacy("geometry", other_geom)
    project.configure_scale_model(scale, other_rep["representation_id"])
    assert project.scale_model_status(scale)["status"] == "stale"



def test_scale_training_preferences_follow_selected_session(tmp_path):
    """Switching sessions restores both Scale checkboxes, including defaults."""
    import inspect

    pytest.importorskip("dash")
    from geometric_coder.ui.app import create_app

    project, scale, _, _, _ = _setup(tmp_path)
    first = project.create_session("Auto-train")
    second = project.create_session("Manual")
    project.patch_session_state(first, {
        "focus_construct_id": scale,
        "scale_auto_train": True,
        "scale_tune_hyperparameters": True,
    })
    project.patch_session_state(second, {
        "focus_construct_id": scale,
        "scale_auto_train": False,
        "scale_tune_hyperparameters": False,
    })
    app = create_app(project)
    entry = next(
        item for item in app.callback_map.values()
        if "callback" in item
        and inspect.unwrap(item["callback"]).__name__ == "restore_session"
    )
    restore = inspect.unwrap(entry["callback"])
    names = [f"{output.component_id}.{output.component_property}"
             for output in entry["output"]]
    for session, expected in [(first, ["auto"]), (second, []), (first, ["auto"])]:
        values = dict(zip(names, restore(session), strict=True))
        assert values["active-session-store.data"] == session
        assert values["scale-auto-train.value"] == expected
        assert values["scale-tune-hyperparameters.value"] == (
            ["tune"] if expected else []
        )

    third = project.create_session("Defaults")
    values = dict(zip(names, restore(third), strict=True))
    assert values["scale-auto-train.value"] == []
    assert values["scale-tune-hyperparameters.value"] == []


def test_develop_auto_training_runs_only_after_saved_evidence(tmp_path, monkeypatch):
    import inspect
    pytest.importorskip("dash")
    from geometric_coder.ui.app import create_app

    project, scale, rep, geom, (a, b, c, d, _) = _setup(tmp_path)
    event = project.record_scale_comparison(scale, a, b, "first_higher")
    app = create_app(project)
    auto = next(
        inspect.unwrap(item["callback"])
        for item in app.callback_map.values()
        if "callback" in item
        and inspect.unwrap(item["callback"]).__name__ == "train_after_saved_scale_comparison"
    )
    calls = []
    def trained(scale_id):
        calls.append(scale_id)
        return {"status": "current", "reused": False, "fit": {"scaler_fit_id": 42}}
    monkeypatch.setattr(project, "train_scale", trained)
    task = {"status": "saved", "event_id": event, "scale_id": scale}
    saved = f"Saved event {event}: A is higher."
    assert auto("New pair ready.", task, scale, ["auto"], rep) == ""
    assert "manual" in auto(saved, task, scale, [], rep)
    assert calls == []
    assert "fit 42" in auto(saved, task, scale, ["auto"], rep)
    assert calls == [scale]
    assert auto(saved, {"status": "ready", "event_id": event, "scale_id": scale},
                scale, ["auto"], rep) == ""
    assert calls == [scale]


def test_code_to_scale_bootstrap_is_nine_ordinary_training_rows(tmp_path):
    project, _, rep, geom, (a, b, c, d, e) = _setup(tmp_path)
    code = project.create_code("Binary indicator")
    for obs in [a, b]:
        project.annotate(obs, code, "positive", origin="human_direct")
    for obs in [c, d, e]:
        project.annotate(obs, code, "negative", origin="human_direct")
    scale = project.create_scale_from_code(
        "Bootstrapped ordinal", "Present higher than Absent", code,
        expected_preview=project.code_scale_bootstrap_preview(code),
        submission_key="b" * 32,
    )["scale_id"]
    training = build_scale_training_data(project, scale, geom)
    assert training.n_comparisons == 6
    assert training.outcomes.tolist() == [0] * 6
    assert len(set(training.comparison_event_ids)) == 6
    project.configure_scale_model(scale, rep)
    fit = project.train_scale(scale, estimator_factory=RecordingScaler)
    assert fit["status"] == "current"
    assert fit["fit"]["diagnostics"]["comparison_count"] == 6


def test_reference_sample_respects_scale_code_eligibility(tmp_path):
    project, scale, rep, geom, (a, b, c, d, e) = _setup(tmp_path)
    gate = project.create_code("Eligible for reference")
    for obs in (a, b, c):
        project.annotate(obs, gate, "positive", origin="human_direct")
    project.update_scale_eligibility(scale, {
        "kind": "filtered",
        "code_filters": [{"code_id": gate, "value": "positive", "source": "human"}],
    })
    project.record_scale_comparison(scale, a, b, "first_higher")
    project.record_scale_comparison(scale, b, c, "parity")
    project.configure_scale_model(scale, rep)
    result = project.train_scale(scale, estimator_factory=RecordingScaler)
    assert result["status"] == "current"
    assert result["fit"]["diagnostics"]["reference_population_count"] == 3
    assert result["fit"]["diagnostics"]["reference_observation_count"] == 3
    loaded = project.artifacts.load_model(
        project.project_dir, result["fit"]["artifact_path"]
    )
    assert loaded.reference_rows_seen == 3


def test_live_scaler_artifact_replaced_without_losing_fit_history(tmp_path):
    project, scale, rep, geom, (a, b, c, d, e) = _setup(tmp_path)
    project.record_scale_comparison(scale, a, b, "first_higher")
    project.record_scale_comparison(scale, c, d, "second_higher")
    project.configure_scale_model(scale, rep)
    first = project.train_scale(scale, estimator_factory=RecordingScaler)["fit"]
    old_artifact = project.project_dir / first["artifact_path"]
    assert old_artifact.is_file()
    project.record_scale_comparison(scale, a, c, "parity")
    with pytest.raises(RuntimeError, match="simulated optimizer failure"):
        project.train_scale(scale, estimator_factory=FailingScaler)
    assert old_artifact.is_file()
    second = project.train_scale(scale, estimator_factory=RecordingScaler)["fit"]
    assert second["scaler_fit_id"] != first["scaler_fit_id"]
    assert not old_artifact.exists()
    assert (project.project_dir / second["artifact_path"]).is_file()
    with project.database.connect() as connection:
        rows = connection.execute(
            "SELECT artifact_path FROM scaler_fits WHERE scale_id = ? "
            "ORDER BY scaler_fit_id", (scale,),
        ).fetchall()
    assert len(rows) == 2
    assert [row["artifact_path"] for row in rows].count("") == 1


class DeterministicCVScaler(RecordingScaler):
    cv_fit_records = []

    def fit(self, X, pairs, outcomes, *, reference_X=None):
        self.__class__.cv_fit_records.append(
            (np.asarray(pairs).copy(), np.asarray(outcomes).copy())
        )
        return super().fit(X, pairs, outcomes, reference_X=reference_X)

    def predict_comparisons(self, X, pairs):
        return np.tile(np.asarray([[0.55, 0.20, 0.25]]), (len(pairs), 1))


def test_tuning_uses_grouped_comparison_folds_and_persists_selection(tmp_path):
    project, scale, rep, geom, (a, b, c, d, e) = _setup(tmp_path)
    first = project.record_scale_comparison(scale, a, b, "first_higher")
    project.record_scale_comparison(scale, b, a, "second_higher")
    project.record_scale_comparison(scale, b, c, "parity")
    project.record_scale_comparison(scale, c, d, "first_higher")
    project.record_scale_comparison(scale, d, e, "second_higher")
    project.configure_scale_model(scale, rep, hyperparameters={
        "score_prior_sd": 1.0, "feature_normalization": "none",
    })
    DeterministicCVScaler.cv_fit_records.clear()
    result = project.train_scale(
        scale, estimator_factory=DeterministicCVScaler,
        tune=True, retune_current=True, folds=2,
    )
    assert result["training_selection"]["tuned"]
    assert result["training_selection"]["folds"] == 2
    assert project.database.get_scaler_spec(scale)["hyperparameters"] == {
        "score_prior_sd": 0.25, "feature_normalization": "none",
    }
    # Every candidate's training fold includes either both A/B repeats or neither.
    cv_records = DeterministicCVScaler.cv_fit_records[:-1]
    assert cv_records
    assert all(
        sum(tuple(pair) == (0, 1) for pair in pairs) in (0, 2)
        for pairs, _outcomes in cv_records
    )
    reused = project.train_scale(scale, estimator_factory=DeterministicCVScaler)
    assert reused["reused"]


def test_frozen_scale_acquisition_storage_survives_reopen_and_refit(tmp_path):
    project, scale, rep, geom, (a, b, c, d, e) = _setup(tmp_path)
    project.record_scale_comparison(scale, a, b, "first_higher")
    project.record_scale_comparison(scale, b, c, "parity")
    project.configure_scale_model(scale, rep)
    fit = project.train_scale(scale, estimator_factory=RecordingScaler)["fit"]
    metadata = {"random_seed": 7, "requested_budget": 2,
                "evaluated_candidate_count": 2}
    ranking = [(a, d, .19), (c, e, .13)]
    result = project.database.store_scale_acquisition(
        scale_id=scale,
        fit_id=fit["scaler_fit_id"],
        scale_version_id=fit["scale_version_id"],
        spec_id=fit["scaler_spec_id"],
        fingerprint="fixture_acquisition_v1",
        provenance=metadata,
        diagnostics={"method": "joint_laplace_gh_7x7_v1"},
        strategy="eig", reason=None, ranked_pairs=ranking,
    )
    run = project.database.scale_acquisition_run(scale)
    assert run is not None and run["run_id"] == result["run_id"]
    assert run["provenance"] == metadata
    page = project.database.scale_acquisition_page(run["run_id"])
    assert [(row["observation_low_id"], row["observation_high_id"],
             row["acquisition_score"]) for row in page] == ranking

    reopened = GeometricCoder.open(project.project_dir)
    assert reopened.database.scale_acquisition_page(run["run_id"]) == page
    with pytest.raises(ValueError, match="Duplicate acquisition pair"):
        reopened.database.store_scale_acquisition(
            scale_id=scale, fit_id=fit["scaler_fit_id"],
            scale_version_id=fit["scale_version_id"],
            spec_id=fit["scaler_spec_id"], fingerprint="dup",
            provenance={}, diagnostics={}, strategy="eig", reason=None,
            ranked_pairs=[ranking[0], ranking[0]],
        )
    assert reopened.database.scale_acquisition_page(run["run_id"]) == page


def test_phase3a_phase3b_fit_ranking_frozen_and_reopen(tmp_path):
    """A real Davidson fit, shared sampler, durable queue and human evidence lifecycle."""
    from geometric_coder.scale_recommendations import (
        next_scale_recommendation, scale_acquisition_status,
    )

    project, scale, rep, _, (a, b, c, d, e) = _setup(tmp_path)
    project.configure_scale_model(scale, rep)
    events = [
        (a, b, "first_higher"), (b, c, "first_higher"),
        (a, c, "first_higher"), (c, d, "parity"),
        (a, d, "first_higher"), (b, e, "second_higher"),
    ]
    for first, second, outcome in events:
        project.record_scale_comparison(scale, first, second, outcome)

    result = project.train_scale(scale)
    assert result["status"] == "current"
    rank = scale_acquisition_status(project, scale)
    assert rank["strategy"] == "eig"
    assert rank["scaler_fit_id"] == result["fit"]["scaler_fit_id"]
    assert rank["diagnostics"]["evaluated_candidate_count"] == 4
    page = project.database.scale_acquisition_page(rank["run_id"])
    assert len(page) == 4
    assert all(row["acquisition_score"] >= 0 for row in page)
    assert [row["acquisition_score"] for row in page] == sorted(
        (row["acquisition_score"] for row in page), reverse=True
    )

    first_choice = next_scale_recommendation(project, scale)
    assert first_choice["strategy"] == "eig"
    assert tuple(first_choice["pair"]) == (
        page[0]["observation_low_id"], page[0]["observation_high_id"]
    )

    # Saving a judgment with automatic training off must only stale the fit
    # and consume a cached recommendation; it must not refresh the sorted scores.
    project.record_scale_comparison(
        scale, *first_choice["pair"], "first_higher"
    )
    assert project.scale_model_status(scale)["status"] == "stale"
    frozen = scale_acquisition_status(project, scale)
    assert frozen["frozen"] and frozen["run_id"] == rank["run_id"]
    second_choice = next_scale_recommendation(project, scale)
    assert second_choice["strategy"] == "eig"
    assert second_choice["pair"] != first_choice["pair"]
    assert project.database.scale_acquisition_page(rank["run_id"]) == page

    reopened = GeometricCoder.open(project.project_dir)
    assert scale_acquisition_status(reopened, scale)["run_id"] == rank["run_id"]
    assert next_scale_recommendation(reopened, scale)["pair"] == second_choice["pair"]

    project.train_scale(scale)
    new_run = scale_acquisition_status(project, scale)
    assert new_run["run_id"] != rank["run_id"]
    assert new_run["scaler_fit_id"] != rank["scaler_fit_id"]


def test_phase3a_phase3b_random_bootstrap_has_finite_budget(tmp_path):
    from geometric_coder.scale_recommendations import (
        next_scale_recommendation, refresh_scale_acquisition,
    )

    project, scale, rep, _, (a, b, c, d, e) = _setup(tmp_path)
    random_pair = next_scale_recommendation(project, scale, seed=5)
    assert random_pair["strategy"] == "random"
    assert random_pair["pair"][0] < random_pair["pair"][1]
    assert random_pair["run_id"] is None
    assert project.scale_training_comparisons(scale) == []

    project.configure_scale_model(scale, rep)
    project.record_scale_comparison(scale, a, b, "first_higher")
    project.record_scale_comparison(scale, b, c, "parity")
    trained = project.train_scale(scale)
    assert trained["status"] == "current"
    run = refresh_scale_acquisition(project, scale, budget=3, seed=8)
    assert run["strategy"] == "random"
    assert run["count"] == 3
    assert run["provenance"]["evaluated_candidate_count"] == 3
    assert all(
        row["acquisition_score"] is None
        for row in project.database.scale_acquisition_page(run["run_id"])
    )


def test_publication_guard_rejects_interleaved_evidence_without_replacing_run(
    tmp_path, monkeypatch,
):
    """An intervening saved judgment invalidates the candidate snapshot at commit."""
    from geometric_coder.scale_recommendations import refresh_scale_acquisition

    project, scale, rep, _, (a, b, c, d, e) = _setup(tmp_path)
    project.configure_scale_model(scale, rep)
    for first, second, outcome in (
        (a, b, "first_higher"), (b, c, "first_higher"),
        (a, c, "first_higher"), (c, d, "parity"),
        (a, d, "first_higher"), (b, e, "second_higher"),
    ):
        project.record_scale_comparison(scale, first, second, outcome)
    trained = project.train_scale(scale)
    original = project.database.store_scale_acquisition
    prior = project.database.scale_acquisition_run(scale)
    assert prior is not None
    prior_page = project.database.scale_acquisition_page(prior["run_id"])

    def interleave(*args, **kwargs):
        project.record_scale_comparison(scale, d, e, "first_higher")
        return original(*args, **kwargs)

    monkeypatch.setattr(project.database, "store_scale_acquisition", interleave)
    with pytest.raises(RuntimeError, match="changed during acquisition"):
        refresh_scale_acquisition(project, scale, seed=11)
    current = project.database.scale_acquisition_run(scale)
    assert current == prior
    assert project.database.scale_acquisition_page(prior["run_id"]) == prior_page


def test_publication_guard_runs_inside_locked_transaction(tmp_path):
    """A failing freshness guard rolls back instead of deleting the old queue."""
    project, scale, rep, _, (a, b, c, d, e) = _setup(tmp_path)
    project.configure_scale_model(scale, rep)
    project.record_scale_comparison(scale, a, b, "first_higher")
    project.record_scale_comparison(scale, b, c, "parity")
    fit = project.train_scale(scale, estimator_factory=RecordingScaler)["fit"]
    kwargs = dict(
        scale_id=scale, fit_id=fit["scaler_fit_id"],
        scale_version_id=fit["scale_version_id"],
        spec_id=fit["scaler_spec_id"], fingerprint="first",
        provenance={}, diagnostics={}, strategy="random",
        reason="fixture", ranked_pairs=[(a, d, None)],
    )
    first = project.database.store_scale_acquisition(**kwargs)
    with pytest.raises(RuntimeError, match="intervening"):
        project.database.store_scale_acquisition(
            **{**kwargs, "fingerprint": "second"},
            publication_guard=lambda: (_ for _ in ()).throw(
                RuntimeError("intervening change")
            ),
        )
    assert project.database.scale_acquisition_run(scale)["run_id"] == first["run_id"]


def test_sparse_acquisition_loads_local_csr_once_across_batches(
    tmp_path, monkeypatch,
):
    """Multiple candidate batches do not decompress the same local CSR repeatedly."""
    from geometric_coder.scale_recommendations import refresh_scale_acquisition

    project, scale, rep, geometry, (a, b, c, d, e) = _setup(
        tmp_path, sparse_geometry=True
    )
    project.configure_scale_model(scale, rep)
    for first, second, outcome in (
        (a, b, "first_higher"), (b, c, "first_higher"),
        (a, c, "first_higher"), (c, d, "parity"),
        (a, d, "first_higher"), (b, e, "second_higher"),
    ):
        project.record_scale_comparison(scale, first, second, outcome)
    project.train_scale(scale)
    calls = []
    original = project.geometry_matrix

    def counting_load(geometry_arg):
        calls.append(geometry_arg)
        return original(geometry_arg)

    monkeypatch.setattr(project, "geometry_matrix", counting_load)
    # One batch per task to make repeated load attempts observable.
    import geometric_coder.candidate_sampling as sampling
    actual = sampling.prepare_candidate_run

    def one_at_a_time(*args, **kwargs):
        return actual(*args, **{**kwargs, "batch_size": 1})

    monkeypatch.setattr(sampling, "prepare_candidate_run", one_at_a_time)
    acquisition = refresh_scale_acquisition(project, scale, seed=17)
    assert acquisition["strategy"] == "eig"
    assert acquisition["count"] > 1
    assert calls == [geometry]
