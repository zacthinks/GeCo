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
    assert reopened.database.project_metadata()["schema_version"] == 20
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
    assert GeometricCoder.open(project.project_dir).database.project_metadata()["schema_version"] == 20


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
