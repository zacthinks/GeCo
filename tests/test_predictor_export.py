from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from geometric_coder import CountGeometry, GeCoPredictorRef, GeometricCoder, SVDGeometry


def _data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": list(range(8)),
            "text": [
                "excellent clear helpful",
                "excellent useful strong",
                "helpful clear useful",
                "bad confusing weak",
                "terrible unclear weak",
                "bad difficult confusing",
                "excellent clear",
                "bad weak",
            ],
        }
    )


def _project(tmp_path: Path, *, multi_geometry: bool = False) -> GeometricCoder:
    lexical = CountGeometry(lemmatize=False, weighting="tfidf")
    geometries = {"lexical": lexical}
    if multi_geometry:
        geometries["svd"] = SVDGeometry(lexical, n_components=2)
    return GeometricCoder.create(
        project_dir=tmp_path / "predictor.geco",
        data=_data(),
        keys=["row_id"],
        text="text",
        metadata=[],
        geometries=geometries,
    )


def _label_training_examples(project: GeometricCoder, code_id: int) -> None:
    observations = project.database.list_observations()
    for position in (0, 1, 2):
        project.annotate(int(observations[position]["observation_id"]), code_id, "positive")
    for position in (3, 4, 5):
        project.annotate(int(observations[position]["observation_id"]), code_id, "negative")


def _classifier_ref(project: GeometricCoder, classifier_spec_id: int) -> GeCoPredictorRef:
    return next(
        ref
        for ref in project.predictors()
        if ref.kind == "classifier" and int(ref.id) == int(classifier_spec_id)
    )


def _committee_ref(project: GeometricCoder, committee_id: int) -> GeCoPredictorRef:
    return next(
        ref
        for ref in project.predictors()
        if ref.kind == "committee" and int(ref.id) == int(committee_id)
    )


def test_predictor_discovery_unifies_classifiers_and_committees(tmp_path: Path) -> None:
    project = _project(tmp_path)
    code_id = project.create_code("quality")
    classifier_id = int(project.classifier_specs(code_id=code_id)[0]["classifier_spec_id"])
    committee_id = project.create_classifier_committee(
        code_id=code_id,
        name="quality committee",
        classifier_spec_ids=[classifier_id],
        aggregation="mean",
    )

    refs = project.predictors(code_id)
    assert [(ref.kind, ref.id, ref.name) for ref in refs] == [
        ("classifier", classifier_id, project.database.get_classifier_spec(classifier_id)["name"]),
        ("committee", committee_id, "quality committee"),
    ]



def test_classifier_and_committee_names_share_one_project_wide_active_namespace(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    code_a = project.create_code("quality")
    code_b = project.create_code("relevance")
    classifier_a = project.classifier_specs(code_id=code_a)[0]
    classifier_b = project.classifier_specs(code_id=code_b)[0]
    classifier_a_id = int(classifier_a["classifier_spec_id"])
    classifier_b_id = int(classifier_b["classifier_spec_id"])

    with pytest.raises(
        ValueError, match=r"active predictor named .* already exists \(classifier\)"
    ):
        project.create_classifier_committee(
            code_id=code_b,
            name=str(classifier_a["name"]),
            classifier_spec_ids=[classifier_b_id],
            aggregation="mean",
        )

    committee_a = project.create_classifier_committee(
        code_id=code_a,
        name="shared predictor name",
        classifier_spec_ids=[classifier_a_id],
        aggregation="mean",
    )
    with pytest.raises(
        ValueError, match=r"active predictor named 'shared predictor name' already exists \(committee\)"
    ):
        project.rename_classifier_spec(classifier_b_id, name="shared predictor name")

    with pytest.raises(
        ValueError, match=r"active predictor named 'shared predictor name' already exists \(committee\)"
    ):
        project.create_classifier_spec(
            code_id=code_b,
            name="shared predictor name",
            geometry_id=int(classifier_b["geometry_id"]),
            algorithm="decision_tree",
        )

    with pytest.raises(
        ValueError, match=r"active predictor named .* already exists \(classifier\)"
    ):
        project.rename_classifier_committee(committee_a, name=str(classifier_a["name"]))

    with pytest.raises(
        ValueError, match=r"active predictor named 'shared predictor name' already exists \(committee\)"
    ):
        project.create_classifier_committee(
            code_id=code_b,
            name="shared predictor name",
            classifier_spec_ids=[classifier_b_id],
            aggregation="mean",
        )


def test_deleting_predictor_frees_name_for_other_predictor_kind(tmp_path: Path) -> None:
    project = _project(tmp_path)
    code_id = project.create_code("quality")
    classifier = project.classifier_specs(code_id=code_id)[0]
    classifier_id = int(classifier["classifier_spec_id"])

    committee_id = project.create_classifier_committee(
        code_id=code_id,
        name="reusable predictor",
        classifier_spec_ids=[classifier_id],
        aggregation="mean",
    )
    project.delete_classifier_committee(committee_id)
    project.rename_classifier_spec(classifier_id, name="reusable predictor")
    assert project.database.get_classifier_spec(classifier_id)["name"] == "reusable predictor"


def test_predictor_discovery_rejects_legacy_active_name_collision(tmp_path: Path) -> None:
    project = _project(tmp_path)
    code_id = project.create_code("quality")
    classifier = project.classifier_specs(code_id=code_id)[0]
    classifier_id = int(classifier["classifier_spec_id"])
    now = project.database.get_classifier_spec(classifier_id)["created_at"]

    # Simulate a pre-0.8.14 workspace whose code-local committee constraint allowed
    # an active committee to reuse an active classifier name. Normal APIs no longer
    # permit this state.
    with project.database.connect() as connection:
        connection.execute(
            """
            INSERT INTO classifier_committees(
                code_id, name, aggregation, status, created_at, updated_at
            ) VALUES (?, ?, 'mean', 'active', ?, ?)
            """,
            (code_id, str(classifier["name"]), now, now),
        )

    with pytest.raises(ValueError, match="Active predictor names must be unique project-wide"):
        project.predictors()

def test_single_classifier_export_reproduces_retained_predictions(tmp_path: Path) -> None:
    project = _project(tmp_path)
    code_id = project.create_code("quality")
    _label_training_examples(project, code_id)
    classifier = project.classifier_specs(code_id=code_id)[0]
    classifier_id = int(classifier["classifier_spec_id"])
    fit = project.train_classifier(code_id=code_id, classifier_spec_id=classifier_id)

    exported = project.export_predictor(_classifier_ref(project, classifier_id))
    assert exported.ref.kind == "classifier"
    assert exported.stale_at_export is False
    assert exported.output_fields == ("prediction", "probability")
    assert len(exported.sources) == 1
    assert len(exported.members) == 1
    assert exported.members[0].source_index == 0
    assert exported.members[0].positive_class == 1

    matrix = project.geometry_matrix(int(fit["geometry_id"]))
    actual = exported.predict_proba([matrix])
    expected_rows = project.database.classifier_predictions(
        int(fit["classifier_fit_id"]), atomic_only=True
    )
    expected = np.asarray([row["probability"] for row in expected_rows], dtype=float)
    np.testing.assert_allclose(actual, expected)
    json.dumps(exported.manifest())


def test_export_rejects_stale_by_default_and_allows_explicit_freeze(tmp_path: Path) -> None:
    project = _project(tmp_path)
    code_id = project.create_code("quality")
    _label_training_examples(project, code_id)
    classifier_id = int(project.classifier_specs(code_id=code_id)[0]["classifier_spec_id"])
    project.train_classifier(code_id=code_id, classifier_spec_id=classifier_id)

    observations = project.database.list_observations()
    project.annotate(int(observations[6]["observation_id"]), code_id, "positive")
    ref = _classifier_ref(project, classifier_id)

    with pytest.raises(ValueError, match="stale retained fit"):
        project.export_predictor(ref)
    exported = project.export_predictor(ref, allow_stale=True)
    assert exported.stale_at_export is True
    assert exported.provenance["classifier_fit_id"] == exported.members[0].classifier_fit_id


def test_fixed_committee_deduplicates_shared_geometry_source(tmp_path: Path) -> None:
    project = _project(tmp_path)
    code_id = project.create_code("quality")
    _label_training_examples(project, code_id)
    geometry_id = int(project.database.list_geometries(public_only=True)[0]["geometry_id"])
    first_id = int(project.classifier_specs(code_id=code_id)[0]["classifier_spec_id"])
    second_id = project.create_classifier_spec(
        code_id=code_id,
        name="quality tree",
        geometry_id=geometry_id,
        algorithm="decision_tree",
    )
    project.train_classifiers(code_id=code_id, classifier_spec_ids=[first_id, second_id])
    committee_id = project.create_classifier_committee(
        code_id=code_id,
        name="shared geometry mean",
        classifier_spec_ids=[first_id, second_id],
        aggregation="mean",
    )

    exported = project.export_predictor(_committee_ref(project, committee_id))
    assert len(exported.sources) == 1
    assert [member.source_index for member in exported.members] == [0, 0]

    matrix = project.geometry_matrix(geometry_id)
    member_probabilities = [
        project.database.classifier_predictions(member.classifier_fit_id, atomic_only=True)
        for member in exported.members
    ]
    expected = np.mean(
        np.column_stack(
            [
                np.asarray([row["probability"] for row in rows], dtype=float)
                for rows in member_probabilities
            ]
        ),
        axis=1,
    )
    np.testing.assert_allclose(exported.predict_proba([matrix]), expected)


def test_learned_committee_export_uses_separate_sparse_dense_sources_and_matches_geco(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path, multi_geometry=True)
    code_id = project.create_code("quality")
    _label_training_examples(project, code_id)
    specs = {str(row["geometry_name"]): row for row in project.classifier_specs(code_id=code_id)}
    lexical_id = int(specs["lexical"]["classifier_spec_id"])
    svd_id = int(specs["svd"]["classifier_spec_id"])
    project.train_classifiers(code_id=code_id, classifier_spec_ids=[lexical_id, svd_id])
    committee_id = project.create_classifier_committee(
        code_id=code_id,
        name="mixed learned",
        classifier_spec_ids=[lexical_id, svd_id],
        aggregation="logistic_stack",
    )
    committee_fit = project.train_classifier_committee(
        code_id=code_id, committee_id=committee_id
    )

    exported = project.export_predictor(_committee_ref(project, committee_id))
    assert exported.aggregation == "logistic_stack"
    assert exported.stacker is not None
    assert exported.stacker_fit_id == int(committee_fit["committee_fit_id"])
    assert len(exported.sources) == 2
    assert [member.source_index for member in exported.members] == [0, 1]

    inputs = [project.geometry_matrix(source.geometry_id) for source in exported.sources]
    assert sparse.issparse(inputs[0])
    assert not sparse.issparse(inputs[1])
    actual = exported.predict_proba(inputs)
    expected_rows = project.database.classifier_committee_predictions(
        int(committee_fit["committee_fit_id"])
    )
    expected = np.asarray(
        [row["probability"] for row in expected_rows if row["unit_id"] is not None],
        dtype=float,
    )
    np.testing.assert_allclose(actual, expected)


def test_learned_committee_stale_export_never_substitutes_new_member_fit(tmp_path: Path) -> None:
    project = _project(tmp_path)
    code_id = project.create_code("quality")
    _label_training_examples(project, code_id)
    geometry_id = int(project.database.list_geometries(public_only=True)[0]["geometry_id"])
    first_id = int(project.classifier_specs(code_id=code_id)[0]["classifier_spec_id"])
    second_id = project.create_classifier_spec(
        code_id=code_id,
        name="tree member",
        geometry_id=geometry_id,
        algorithm="decision_tree",
    )
    project.train_classifiers(code_id=code_id, classifier_spec_ids=[first_id, second_id])
    committee_id = project.create_classifier_committee(
        code_id=code_id,
        name="learned",
        classifier_spec_ids=[first_id, second_id],
        aggregation="logistic_stack",
    )
    project.train_classifier_committee(code_id=code_id, committee_id=committee_id)

    # New evidence followed by member retraining retires the exact estimator fit that
    # the old stacker consumed.  Export must fail rather than pairing that old stacker
    # with the newer member estimator and silently changing the prediction procedure.
    observations = project.database.list_observations()
    project.annotate(int(observations[6]["observation_id"]), code_id, "positive")
    project.train_classifier(code_id=code_id, classifier_spec_id=first_id)

    ref = _committee_ref(project, committee_id)
    with pytest.raises(ValueError, match="exact frozen predictor cannot be exported"):
        project.export_predictor(ref, allow_stale=True)


def test_export_validates_source_count_row_alignment_and_feature_width(tmp_path: Path) -> None:
    project = _project(tmp_path, multi_geometry=True)
    code_id = project.create_code("quality")
    _label_training_examples(project, code_id)
    specs = project.classifier_specs(code_id=code_id)
    project.train_classifiers(
        code_id=code_id,
        classifier_spec_ids=[int(row["classifier_spec_id"]) for row in specs],
    )
    committee_id = project.create_classifier_committee(
        code_id=code_id,
        name="multi",
        classifier_spec_ids=[int(row["classifier_spec_id"]) for row in specs],
        aggregation="mean",
    )
    exported = project.export_predictor(_committee_ref(project, committee_id))
    inputs = [project.geometry_matrix(source.geometry_id) for source in exported.sources]

    with pytest.raises(ValueError, match="requires 2 source matrix"):
        exported.predict_proba(inputs[:1])
    with pytest.raises(ValueError, match="same number of rows"):
        exported.predict_proba([inputs[0][:-1], inputs[1]])

    wrong_width = np.zeros((inputs[1].shape[0], inputs[1].shape[1] + 1), dtype=float)
    with pytest.raises(ValueError, match="requires .* features"):
        exported.predict_proba([inputs[0], wrong_width])
