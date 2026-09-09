"""SQLite schema and data import for GeCo projects."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

SCHEMA_VERSION = 13

FIXED_COMMITTEE_AGGREGATIONS = {
    "mean",
    "median",  # retained for existing projects; not emphasized in the UI
    "minimum",
    "maximum",
    "harmonic_mean",
    "geometric_mean",
}
TRAINABLE_COMMITTEE_AGGREGATIONS = {"logistic_stack"}
COMMITTEE_AGGREGATIONS = FIXED_COMMITTEE_AGGREGATIONS | TRAINABLE_COMMITTEE_AGGREGATIONS

_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE project (
    project_id INTEGER PRIMARY KEY CHECK (project_id = 1),
    schema_version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    modality TEXT NOT NULL,
    text_column TEXT NOT NULL,
    key_columns_json TEXT NOT NULL,
    metadata_columns_json TEXT NOT NULL,
    external_backed INTEGER NOT NULL DEFAULT 0 CHECK (external_backed IN (0, 1))
);

CREATE TABLE hierarchy_levels (
    level_index INTEGER PRIMARY KEY,
    column_name TEXT NOT NULL UNIQUE
);

CREATE TABLE groups (
    group_id INTEGER PRIMARY KEY,
    level_index INTEGER NOT NULL REFERENCES hierarchy_levels(level_index),
    parent_group_id INTEGER REFERENCES groups(group_id),
    position_within_parent INTEGER NOT NULL,
    user_value_json TEXT NOT NULL
);
CREATE INDEX groups_level_parent_idx
ON groups(level_index, parent_group_id, position_within_parent);

CREATE TABLE observations (
    observation_id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('atomic', 'span', 'teaching_example')),
    text TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE units (
    unit_id INTEGER PRIMARY KEY,
    observation_id INTEGER NOT NULL UNIQUE REFERENCES observations(observation_id),
    row_position INTEGER NOT NULL UNIQUE,
    text TEXT NOT NULL,
    user_key_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL
);

CREATE TABLE unit_groups (
    unit_id INTEGER NOT NULL REFERENCES units(unit_id) ON DELETE CASCADE,
    level_index INTEGER NOT NULL REFERENCES hierarchy_levels(level_index),
    group_id INTEGER NOT NULL REFERENCES groups(group_id),
    PRIMARY KEY (unit_id, level_index)
);

CREATE TABLE span_members (
    span_observation_id INTEGER NOT NULL REFERENCES observations(observation_id) ON DELETE CASCADE,
    unit_id INTEGER NOT NULL REFERENCES units(unit_id),
    member_order INTEGER NOT NULL,
    PRIMARY KEY (span_observation_id, member_order),
    UNIQUE (span_observation_id, unit_id)
);
CREATE INDEX span_members_unit_idx ON span_members(unit_id);

CREATE TABLE codes (
    code_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    definition TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE code_versions (
    code_version_id INTEGER PRIMARY KEY,
    code_id INTEGER NOT NULL REFERENCES codes(code_id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL,
    name TEXT NOT NULL,
    description_markdown TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    UNIQUE(code_id, version_number)
);

CREATE TABLE code_version_hashtags (
    code_version_id INTEGER NOT NULL REFERENCES code_versions(code_version_id) ON DELETE CASCADE,
    hashtag TEXT NOT NULL,
    PRIMARY KEY (code_version_id, hashtag)
);

CREATE TABLE teaching_examples (
    observation_id INTEGER PRIMARY KEY REFERENCES observations(observation_id) ON DELETE CASCADE,
    code_id INTEGER NOT NULL REFERENCES codes(code_id),
    label TEXT NOT NULL CHECK (label IN ('positive', 'negative')),
    status TEXT NOT NULL CHECK (status IN ('active', 'inactive', 'deleted')),
    note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX teaching_examples_code_status_idx
ON teaching_examples(code_id, status, observation_id);

CREATE TABLE geometries (
    geometry_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    class_name TEXT NOT NULL,
    modality TEXT NOT NULL,
    public INTEGER NOT NULL CHECK (public IN (0, 1)),
    config_json TEXT NOT NULL,
    dependency_names_json TEXT NOT NULL,
    matrix_path TEXT NOT NULL,
    matrix_sparse INTEGER NOT NULL CHECK (matrix_sparse IN (0, 1)),
    state_path TEXT,
    supports_query INTEGER NOT NULL CHECK (supports_query IN (0, 1)),
    supports_text_transform INTEGER NOT NULL CHECK (supports_text_transform IN (0, 1)),
    storage_kind TEXT NOT NULL DEFAULT 'local' CHECK (storage_kind IN ('local', 'external')),
    external_ref_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE observation_geometry_vectors (
    observation_id INTEGER NOT NULL REFERENCES observations(observation_id) ON DELETE CASCADE,
    geometry_id INTEGER NOT NULL REFERENCES geometries(geometry_id) ON DELETE CASCADE,
    artifact_path TEXT NOT NULL,
    matrix_sparse INTEGER NOT NULL CHECK (matrix_sparse IN (0, 1)),
    created_at TEXT NOT NULL,
    PRIMARY KEY (observation_id, geometry_id)
);

CREATE TABLE views (
    view_id INTEGER PRIMARY KEY,
    geometry_id INTEGER NOT NULL REFERENCES geometries(geometry_id),
    name TEXT NOT NULL UNIQUE,
    method TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    state_path TEXT,
    supports_transform INTEGER NOT NULL DEFAULT 0 CHECK (supports_transform IN (0, 1)),
    storage_kind TEXT NOT NULL DEFAULT 'local' CHECK (storage_kind IN ('local', 'external')),
    external_ref_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE sessions (
    session_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_active_at TEXT NOT NULL,
    state_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE navigation_events (
    event_id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(session_id),
    event_index INTEGER NOT NULL,
    unit_id INTEGER NOT NULL REFERENCES units(unit_id),
    visited_at TEXT NOT NULL,
    navigation_method TEXT NOT NULL,
    source_unit_id INTEGER REFERENCES units(unit_id),
    geometry_id INTEGER REFERENCES geometries(geometry_id),
    active_code_id INTEGER,
    classifier_fit_id INTEGER REFERENCES classifier_fits(classifier_fit_id),
    UNIQUE(session_id, event_index)
);
CREATE INDEX navigation_session_unit_idx ON navigation_events(session_id, unit_id);

CREATE TABLE annotation_events (
    annotation_event_id INTEGER PRIMARY KEY,
    observation_id INTEGER NOT NULL REFERENCES observations(observation_id),
    code_id INTEGER NOT NULL REFERENCES codes(code_id),
    value TEXT NOT NULL CHECK (value IN ('positive', 'negative', 'unsure')),
    origin TEXT NOT NULL,
    created_at TEXT NOT NULL,
    classifier_fit_id INTEGER REFERENCES classifier_fits(classifier_fit_id)
);
CREATE INDEX annotation_observation_code_idx
ON annotation_events(observation_id, code_id, annotation_event_id);

CREATE TABLE memos (
    memo_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    body_markdown TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE memo_hashtags (
    memo_id INTEGER NOT NULL REFERENCES memos(memo_id) ON DELETE CASCADE,
    hashtag TEXT NOT NULL,
    PRIMARY KEY (memo_id, hashtag)
);
CREATE TABLE memo_unit_references (
    reference_id INTEGER PRIMARY KEY,
    memo_id INTEGER NOT NULL REFERENCES memos(memo_id) ON DELETE CASCADE,
    unit_id INTEGER NOT NULL REFERENCES units(unit_id),
    token TEXT NOT NULL
);
CREATE TABLE memo_versions (
    memo_version_id INTEGER PRIMARY KEY,
    memo_id INTEGER NOT NULL REFERENCES memos(memo_id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL,
    title TEXT NOT NULL,
    body_markdown TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(memo_id, version_number)
);
CREATE TABLE memo_version_hashtags (
    memo_version_id INTEGER NOT NULL REFERENCES memo_versions(memo_version_id) ON DELETE CASCADE,
    hashtag TEXT NOT NULL,
    PRIMARY KEY (memo_version_id, hashtag)
);
CREATE TABLE memo_version_unit_references (
    reference_id INTEGER PRIMARY KEY,
    memo_version_id INTEGER NOT NULL REFERENCES memo_versions(memo_version_id) ON DELETE CASCADE,
    unit_id INTEGER NOT NULL REFERENCES units(unit_id),
    token TEXT NOT NULL
);

CREATE TABLE classifier_specs (
    classifier_spec_id INTEGER PRIMARY KEY,
    code_id INTEGER NOT NULL REFERENCES codes(code_id) ON DELETE CASCADE,
    name TEXT NOT NULL UNIQUE,
    geometry_id INTEGER NOT NULL REFERENCES geometries(geometry_id),
    algorithm TEXT NOT NULL,
    hyperparameters_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'deleted')),
    created_at TEXT NOT NULL
);
CREATE INDEX classifier_specs_geometry_idx
ON classifier_specs(geometry_id, classifier_spec_id);
CREATE INDEX classifier_specs_code_idx
ON classifier_specs(code_id, classifier_spec_id);

CREATE TABLE classifier_fits (
    classifier_fit_id INTEGER PRIMARY KEY,
    classifier_spec_id INTEGER NOT NULL REFERENCES classifier_specs(classifier_spec_id),
    code_id INTEGER NOT NULL REFERENCES codes(code_id),
    classifier_name TEXT NOT NULL,
    geometry_id INTEGER NOT NULL REFERENCES geometries(geometry_id),
    algorithm TEXT NOT NULL,
    hyperparameters_json TEXT NOT NULL,
    training_snapshot_json TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    score_kind TEXT NOT NULL CHECK (score_kind IN ('probability', 'decision_score')),
    created_at TEXT NOT NULL
);
CREATE INDEX classifier_fits_spec_code_idx
ON classifier_fits(classifier_spec_id, code_id, classifier_fit_id);

CREATE TABLE classifier_predictions (
    classifier_fit_id INTEGER NOT NULL REFERENCES classifier_fits(classifier_fit_id)
        ON DELETE CASCADE,
    observation_id INTEGER NOT NULL REFERENCES observations(observation_id),
    predicted_label TEXT NOT NULL CHECK (predicted_label IN ('positive', 'negative')),
    probability REAL,
    decision_score REAL,
    uncertainty REAL NOT NULL,
    PRIMARY KEY (classifier_fit_id, observation_id)
);
CREATE INDEX classifier_predictions_observation_idx
ON classifier_predictions(observation_id, classifier_fit_id);

CREATE TABLE classifier_committees (
    committee_id INTEGER PRIMARY KEY,
    code_id INTEGER NOT NULL REFERENCES codes(code_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    aggregation TEXT NOT NULL CHECK (aggregation IN ('mean', 'median', 'minimum', 'maximum', 'harmonic_mean', 'geometric_mean', 'logistic_stack')),
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'deleted')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(code_id, name)
);

CREATE TABLE classifier_committee_members (
    committee_id INTEGER NOT NULL REFERENCES classifier_committees(committee_id)
        ON DELETE CASCADE,
    classifier_spec_id INTEGER NOT NULL REFERENCES classifier_specs(classifier_spec_id),
    member_order INTEGER NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0 CHECK (weight > 0.0),
    PRIMARY KEY (committee_id, classifier_spec_id),
    UNIQUE(committee_id, member_order)
);

CREATE TABLE classifier_committee_fits (
    committee_fit_id INTEGER PRIMARY KEY,
    committee_id INTEGER NOT NULL REFERENCES classifier_committees(committee_id),
    code_id INTEGER NOT NULL REFERENCES codes(code_id),
    aggregation TEXT NOT NULL CHECK (aggregation IN ('logistic_stack')),
    member_fit_ids_json TEXT NOT NULL,
    training_snapshot_json TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX classifier_committee_fits_committee_code_idx
ON classifier_committee_fits(committee_id, code_id, committee_fit_id);

CREATE TABLE classifier_committee_predictions (
    committee_fit_id INTEGER NOT NULL REFERENCES classifier_committee_fits(committee_fit_id)
        ON DELETE CASCADE,
    observation_id INTEGER NOT NULL REFERENCES observations(observation_id),
    predicted_label TEXT NOT NULL CHECK (predicted_label IN ('positive', 'negative')),
    probability REAL NOT NULL CHECK (probability >= 0.0 AND probability <= 1.0),
    uncertainty REAL NOT NULL,
    PRIMARY KEY (committee_fit_id, observation_id)
);
CREATE INDEX classifier_committee_predictions_observation_idx
ON classifier_committee_predictions(observation_id, committee_fit_id);

CREATE TABLE classifier_evaluation_events (
    evaluation_event_id INTEGER PRIMARY KEY,
    event_type TEXT NOT NULL CHECK (
        event_type IN ('teaching_example_pretraining', 'random_probe')
    ),
    code_id INTEGER NOT NULL REFERENCES codes(code_id),
    classifier_fit_id INTEGER NOT NULL REFERENCES classifier_fits(classifier_fit_id),
    observation_id INTEGER REFERENCES observations(observation_id),
    human_label TEXT NOT NULL CHECK (human_label IN ('positive', 'negative')),
    predicted_label TEXT NOT NULL CHECK (predicted_label IN ('positive', 'negative')),
    probability REAL,
    decision_score REAL,
    added_to_training INTEGER NOT NULL CHECK (added_to_training IN (0, 1)),
    created_at TEXT NOT NULL
);
CREATE INDEX classifier_evaluation_code_idx
ON classifier_evaluation_events(code_id, evaluation_event_id);

CREATE TABLE classifier_tuning_runs (
    tuning_run_id INTEGER PRIMARY KEY,
    code_id INTEGER NOT NULL REFERENCES codes(code_id),
    classifier_spec_id INTEGER NOT NULL REFERENCES classifier_specs(classifier_spec_id),
    metric TEXT NOT NULL CHECK (metric IN ('log_loss', 'brier', 'accuracy')),
    folds INTEGER NOT NULL CHECK (folds >= 2),
    candidate_lambdas_json TEXT NOT NULL,
    results_json TEXT NOT NULL,
    selected_lambda REAL NOT NULL CHECK (selected_lambda > 0.0),
    created_at TEXT NOT NULL
);
CREATE INDEX classifier_tuning_code_spec_idx
ON classifier_tuning_runs(code_id, classifier_spec_id, tuning_run_id);

CREATE TABLE apply_runs (
    apply_run_id INTEGER PRIMARY KEY,
    code_id INTEGER NOT NULL REFERENCES codes(code_id),
    aggregation TEXT NOT NULL CHECK (aggregation IN ('mean', 'median', 'minimum', 'maximum', 'harmonic_mean', 'geometric_mean', 'logistic_stack')),
    source_kind TEXT NOT NULL CHECK (source_kind IN ('classifier', 'committee')),
    source_id INTEGER NOT NULL,
    committee_fit_id INTEGER REFERENCES classifier_committee_fits(committee_fit_id),
    threshold REAL NOT NULL CHECK (threshold >= 0.0 AND threshold <= 1.0),
    eligibility TEXT NOT NULL CHECK (eligibility IN ('unreviewed')),
    status TEXT NOT NULL CHECK (status IN ('draft', 'committed', 'discarded')),
    created_at TEXT NOT NULL,
    committed_at TEXT,
    discarded_at TEXT
);
CREATE TABLE apply_run_fits (
    apply_run_id INTEGER NOT NULL REFERENCES apply_runs(apply_run_id) ON DELETE CASCADE,
    classifier_fit_id INTEGER NOT NULL REFERENCES classifier_fits(classifier_fit_id),
    PRIMARY KEY (apply_run_id, classifier_fit_id)
);
CREATE TABLE apply_commit_operations (
    commit_operation_id INTEGER PRIMARY KEY,
    apply_run_id INTEGER NOT NULL REFERENCES apply_runs(apply_run_id) ON DELETE CASCADE,
    scope TEXT NOT NULL CHECK (scope IN ('all_reviewed', 'current_page', 'explicit')),
    created_at TEXT NOT NULL,
    undone_at TEXT
);
CREATE INDEX apply_commit_operations_run_idx
ON apply_commit_operations(apply_run_id, commit_operation_id);

CREATE TABLE annotation_event_retractions (
    annotation_event_id INTEGER PRIMARY KEY REFERENCES annotation_events(annotation_event_id),
    commit_operation_id INTEGER NOT NULL REFERENCES apply_commit_operations(commit_operation_id),
    retracted_at TEXT NOT NULL
);

CREATE TABLE apply_proposals (
    apply_run_id INTEGER NOT NULL REFERENCES apply_runs(apply_run_id) ON DELETE CASCADE,
    unit_id INTEGER NOT NULL REFERENCES units(unit_id),
    probability REAL NOT NULL CHECK (probability >= 0.0 AND probability <= 1.0),
    proposed_label TEXT NOT NULL CHECK (proposed_label IN ('positive', 'negative')),
    decision TEXT NOT NULL DEFAULT 'pending'
        CHECK (decision IN ('pending', 'accept', 'positive', 'negative', 'unsure', 'unreviewed')),
    review_mode TEXT CHECK (review_mode IN ('individual', 'bulk')),
    reviewed_at TEXT,
    committed_annotation_event_id INTEGER REFERENCES annotation_events(annotation_event_id),
    commit_operation_id INTEGER REFERENCES apply_commit_operations(commit_operation_id),
    removed_at TEXT,
    PRIMARY KEY (apply_run_id, unit_id)
);
CREATE INDEX apply_proposals_run_label_idx
ON apply_proposals(apply_run_id, proposed_label, decision);

CREATE VIEW current_annotations AS
SELECT latest.annotation_event_id,
       latest.observation_id,
       u.unit_id,
       latest.code_id,
       latest.value,
       latest.origin,
       latest.created_at,
       latest.classifier_fit_id
FROM (
    SELECT ae.*,
           ROW_NUMBER() OVER (
               PARTITION BY ae.observation_id, ae.code_id
               ORDER BY ae.annotation_event_id DESC
           ) AS row_number
    FROM annotation_events AS ae
    LEFT JOIN annotation_event_retractions AS aer
      ON aer.annotation_event_id = ae.annotation_event_id
    WHERE aer.annotation_event_id IS NULL
) AS latest
LEFT JOIN units AS u ON u.observation_id = latest.observation_id
WHERE latest.row_number = 1;
"""

def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat()


class ProjectDatabase:
    """Thin SQLite access layer for one GeCo project."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Open, transact with, and explicitly close a SQLite connection.

        ``sqlite3.Connection`` commits or rolls back when used as a context
        manager, but it does *not* close itself. Explicit closure matters on
        Windows, where an open handle prevents failed project directories from
        being removed.
        """
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
        except BaseException:
            connection.rollback()
            raise
        else:
            connection.commit()
        finally:
            connection.close()

    def initialize(
        self,
        *,
        modality: str,
        text_column: str,
        key_columns: Sequence[str],
        metadata_columns: Sequence[str],
        external_backed: bool = False,
    ) -> None:
        """Create the schema and project metadata."""
        with self.connect() as connection:
            connection.executescript(_SCHEMA)
            connection.execute(
                """
                INSERT INTO project (
                    project_id, schema_version, created_at, modality,
                    text_column, key_columns_json, metadata_columns_json, external_backed
                ) VALUES (1, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    SCHEMA_VERSION,
                    utc_now(),
                    modality,
                    text_column,
                    json.dumps(list(key_columns)),
                    json.dumps(list(metadata_columns)),
                    int(bool(external_backed)),
                ),
            )
            connection.executemany(
                "INSERT INTO hierarchy_levels(level_index, column_name) VALUES (?, ?)",
                enumerate(key_columns),
            )
            now = utc_now()
            connection.execute(
                """
                INSERT INTO sessions(title, created_at, last_active_at, state_json)
                VALUES (?, ?, ?, '{}')
                """,
                ("Default session", now, now),
            )

    def import_dataframe(
        self,
        data: pd.DataFrame,
        *,
        key_columns: Sequence[str],
        text_column: str,
        metadata_columns: Sequence[str],
        external_backed: bool = False,
    ) -> None:
        """Import rows and construct first-seen integer group indices."""
        group_id_counter = 0
        group_lookup: dict[tuple[int, int | None, str], int] = {}
        child_positions: dict[tuple[int, int | None], int] = {}

        with self.connect() as connection:
            for row_position, (_, row) in enumerate(data.iterrows()):
                unit_id = row_position + 1
                user_key = {column: _json_value(row[column]) for column in key_columns}
                metadata = {column: _json_value(row[column]) for column in metadata_columns}
                text_value = str(row[text_column])
                connection.execute(
                    "INSERT INTO observations(observation_id, kind, text, created_at) "
                    "VALUES (?, 'atomic', ?, ?)",
                    (unit_id, text_value, utc_now()),
                )
                connection.execute(
                    """
                    INSERT INTO units(
                        unit_id, observation_id, row_position, text, user_key_json, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        unit_id,
                        unit_id,
                        row_position,
                        text_value,
                        json.dumps(user_key, ensure_ascii=False),
                        json.dumps(metadata, ensure_ascii=False),
                    ),
                )

                parent_group_id: int | None = None
                for level_index, column in enumerate(key_columns):
                    value_json = json.dumps(_json_value(row[column]), ensure_ascii=False)
                    lookup_key = (level_index, parent_group_id, value_json)
                    group_id = group_lookup.get(lookup_key)
                    if group_id is None:
                        group_id_counter += 1
                        group_id = group_id_counter
                        group_lookup[lookup_key] = group_id
                        position_key = (level_index, parent_group_id)
                        position = child_positions.get(position_key, 0)
                        child_positions[position_key] = position + 1
                        connection.execute(
                            """
                            INSERT INTO groups(
                                group_id, level_index, parent_group_id,
                                position_within_parent, user_value_json
                            ) VALUES (?, ?, ?, ?, ?)
                            """,
                            (
                                group_id,
                                level_index,
                                parent_group_id,
                                position,
                                value_json,
                            ),
                        )
                    connection.execute(
                        """
                        INSERT INTO unit_groups(unit_id, level_index, group_id)
                        VALUES (?, ?, ?)
                        """,
                        (unit_id, level_index, group_id),
                    )
                    parent_group_id = group_id

    def append_dataframe(
        self,
        data: pd.DataFrame,
        *,
        key_columns: Sequence[str],
        text_column: str,
        metadata_columns: Sequence[str],
        geometry_artifacts: Mapping[int, tuple[str, bool]],
        view_artifacts: Mapping[int, str],
    ) -> list[dict[str, int]]:
        """Append atomic rows and atomically switch matrix/view artifact pointers.

        The numerical artifacts are written before this transaction begins. If the
        relational append fails, SQLite rolls back and the project continues to
        reference its previous artifacts; newly written files are merely orphaned.
        """
        if data.empty:
            return []

        with self.connect() as connection:
            existing_key_rows = connection.execute(
                "SELECT user_key_json FROM units"
            ).fetchall()
            existing_keys = {str(row["user_key_json"]) for row in existing_key_rows}

            incoming_keys: list[str] = []
            incoming_key_set: set[str] = set()
            for _, row in data.iterrows():
                user_key = {column: _json_value(row[column]) for column in key_columns}
                serialized = json.dumps(user_key, ensure_ascii=False)
                if serialized in existing_keys:
                    raise ValueError(
                        "Incremental ingestion would duplicate an existing key: "
                        f"{user_key}"
                    )
                if serialized in incoming_key_set:
                    raise ValueError(
                        "Incremental ingestion contains duplicate keys: "
                        f"{user_key}"
                    )
                incoming_keys.append(serialized)
                incoming_key_set.add(serialized)

            counters = connection.execute(
                """
                SELECT
                    COALESCE((SELECT MAX(unit_id) FROM units), 0) AS max_unit_id,
                    COALESCE((SELECT MAX(observation_id) FROM observations), 0)
                        AS max_observation_id,
                    COALESCE((SELECT MAX(row_position) FROM units), -1) AS max_row_position,
                    COALESCE((SELECT MAX(group_id) FROM groups), 0) AS max_group_id
                """
            ).fetchone()
            if counters is None:
                raise RuntimeError("Could not initialize append counters.")
            max_unit_id = int(counters["max_unit_id"])
            max_observation_id = int(counters["max_observation_id"])
            max_row_position = int(counters["max_row_position"])
            group_id_counter = int(counters["max_group_id"])

            group_rows = connection.execute(
                """
                SELECT group_id, level_index, parent_group_id,
                       position_within_parent, user_value_json
                FROM groups
                ORDER BY group_id
                """
            ).fetchall()
            group_lookup: dict[tuple[int, int | None, str], int] = {}
            child_positions: dict[tuple[int, int | None], int] = {}
            for group in group_rows:
                level_index = int(group["level_index"])
                parent_group_id = (
                    int(group["parent_group_id"])
                    if group["parent_group_id"] is not None
                    else None
                )
                group_lookup[
                    (level_index, parent_group_id, str(group["user_value_json"]))
                ] = int(group["group_id"])
                position_key = (level_index, parent_group_id)
                child_positions[position_key] = max(
                    child_positions.get(position_key, 0),
                    int(group["position_within_parent"]) + 1,
                )

            appended: list[dict[str, int]] = []
            for offset, (_, row) in enumerate(data.iterrows(), start=1):
                unit_id = max_unit_id + offset
                observation_id = max_observation_id + offset
                row_position = max_row_position + offset
                user_key_json = incoming_keys[offset - 1]
                metadata = {
                    column: _json_value(row[column]) for column in metadata_columns
                }
                text_value = str(row[text_column])
                connection.execute(
                    "INSERT INTO observations(observation_id, kind, text, created_at) "
                    "VALUES (?, 'atomic', ?, ?)",
                    (observation_id, text_value, utc_now()),
                )
                connection.execute(
                    """
                    INSERT INTO units(
                        unit_id, observation_id, row_position, text,
                        user_key_json, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        unit_id,
                        observation_id,
                        row_position,
                        text_value,
                        user_key_json,
                        json.dumps(metadata, ensure_ascii=False),
                    ),
                )

                parent_group_id: int | None = None
                for level_index, column in enumerate(key_columns):
                    value_json = json.dumps(_json_value(row[column]), ensure_ascii=False)
                    lookup_key = (level_index, parent_group_id, value_json)
                    group_id = group_lookup.get(lookup_key)
                    if group_id is None:
                        group_id_counter += 1
                        group_id = group_id_counter
                        group_lookup[lookup_key] = group_id
                        position_key = (level_index, parent_group_id)
                        position = child_positions.get(position_key, 0)
                        child_positions[position_key] = position + 1
                        connection.execute(
                            """
                            INSERT INTO groups(
                                group_id, level_index, parent_group_id,
                                position_within_parent, user_value_json
                            ) VALUES (?, ?, ?, ?, ?)
                            """,
                            (
                                group_id,
                                level_index,
                                parent_group_id,
                                position,
                                value_json,
                            ),
                        )
                    connection.execute(
                        """
                        INSERT INTO unit_groups(unit_id, level_index, group_id)
                        VALUES (?, ?, ?)
                        """,
                        (unit_id, level_index, group_id),
                    )
                    parent_group_id = group_id

                appended.append(
                    {
                        "unit_id": unit_id,
                        "observation_id": observation_id,
                        "row_position": row_position,
                    }
                )

            for geometry_id, (artifact_path, matrix_sparse) in geometry_artifacts.items():
                cursor = connection.execute(
                    """
                    UPDATE geometries
                    SET matrix_path = ?, matrix_sparse = ?
                    WHERE geometry_id = ?
                    """,
                    (str(artifact_path), int(bool(matrix_sparse)), int(geometry_id)),
                )
                if cursor.rowcount != 1:
                    raise KeyError(f"Unknown geometry_id: {geometry_id}")

            for view_id, artifact_path in view_artifacts.items():
                cursor = connection.execute(
                    "UPDATE views SET artifact_path = ? WHERE view_id = ?",
                    (str(artifact_path), int(view_id)),
                )
                if cursor.rowcount != 1:
                    raise KeyError(f"Unknown view_id: {view_id}")

            return appended

    def register_geometry(
        self,
        *,
        name: str,
        class_name: str,
        modality: str,
        public: bool,
        config: dict[str, Any],
        dependency_names: Iterable[str],
        matrix_path: str,
        matrix_sparse: bool,
        state_path: str | None,
        supports_query: bool,
        supports_text_transform: bool,
        storage_kind: str = "local",
        external_ref: Any | None = None,
    ) -> int:
        """Register a local or externally backed geometry and return its ID."""
        if storage_kind not in {"local", "external"}:
            raise ValueError("storage_kind must be 'local' or 'external'")
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO geometries(
                    name, class_name, modality, public, config_json,
                    dependency_names_json, matrix_path, matrix_sparse,
                    state_path, supports_query, supports_text_transform, storage_kind, external_ref_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    class_name,
                    modality,
                    int(public),
                    json.dumps(config, sort_keys=True),
                    json.dumps(list(dependency_names)),
                    matrix_path,
                    int(matrix_sparse),
                    state_path,
                    int(supports_query),
                    int(supports_text_transform),
                    storage_kind,
                    (json.dumps(external_ref) if external_ref is not None else None),
                    utc_now(),
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return a geometry ID.")
            return int(cursor.lastrowid)

    def register_view(
        self,
        *,
        geometry_id: int,
        name: str,
        method: str,
        parameters: dict[str, Any],
        artifact_path: str,
        state_path: str | None = None,
        supports_transform: bool = False,
        storage_kind: str = "local",
        external_ref: Any | None = None,
    ) -> int:
        """Register a local or externally backed two-dimensional view."""
        if storage_kind not in {"local", "external"}:
            raise ValueError("storage_kind must be 'local' or 'external'")
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO views(
                    geometry_id, name, method, parameters_json,
                    artifact_path, state_path, supports_transform, storage_kind,
                    external_ref_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    geometry_id,
                    name,
                    method,
                    json.dumps(parameters, sort_keys=True),
                    artifact_path,
                    state_path,
                    int(supports_transform),
                    storage_kind,
                    (json.dumps(external_ref) if external_ref is not None else None),
                    utc_now(),
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return a view ID.")
            return int(cursor.lastrowid)


    def list_geometries(self, *, public_only: bool = False) -> list[dict[str, Any]]:
        """List persisted geometries in computation order."""
        query = "SELECT * FROM geometries"
        parameters: tuple[Any, ...] = ()
        if public_only:
            query += " WHERE public = ?"
            parameters = (1,)
        query += " ORDER BY geometry_id"
        with self.connect() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
        return [_geometry_row(row) for row in rows]

    def get_geometry(self, geometry_id: int) -> dict[str, Any]:
        """Return one persisted geometry registry row."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM geometries WHERE geometry_id = ?",
                (geometry_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown geometry_id: {geometry_id}")
        return _geometry_row(row)

    def get_geometry_by_name(self, name: str) -> dict[str, Any]:
        """Return one persisted geometry by public or hidden registry name."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM geometries WHERE name = ?",
                (name,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown geometry name: {name!r}")
        return _geometry_row(row)

    def list_views(self, *, geometry_id: int | None = None) -> list[dict[str, Any]]:
        """List persisted two-dimensional views."""
        query = "SELECT * FROM views"
        parameters: tuple[Any, ...] = ()
        if geometry_id is not None:
            query += " WHERE geometry_id = ?"
            parameters = (geometry_id,)
        query += " ORDER BY view_id"
        with self.connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [_view_row(row) for row in rows]

    def get_view(self, view_id: int) -> dict[str, Any]:
        """Return one persisted view."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM views WHERE view_id = ?",
                (view_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown view_id: {view_id}")
        return _view_row(row)

    def hierarchy_levels(self) -> list[dict[str, Any]]:
        """Return hierarchy levels from broadest to most specific."""
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM hierarchy_levels ORDER BY level_index"
            ).fetchall()
        return [
            {
                "level_index": int(row["level_index"]),
                "column_name": str(row["column_name"]),
            }
            for row in rows
        ]

    def hierarchy_level_counts(self) -> list[dict[str, Any]]:
        """Return hierarchy levels and the number of full-prefix groups at each level."""
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT hl.level_index, hl.column_name, COUNT(g.group_id) AS group_count
                FROM hierarchy_levels AS hl
                LEFT JOIN groups AS g ON g.level_index = hl.level_index
                GROUP BY hl.level_index, hl.column_name
                ORDER BY hl.level_index
                """
            ).fetchall()
        return [
            {
                "level_index": int(row["level_index"]),
                "column_name": str(row["column_name"]),
                "count": int(row["group_count"]),
            }
            for row in rows
        ]

    def list_units(self) -> list[dict[str, Any]]:
        """Return all atomic units in canonical row order."""
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM units ORDER BY row_position"
            ).fetchall()
        return [_unit_row(row) for row in rows]

    def get_unit(self, unit_id: int) -> dict[str, Any]:
        """Return one atomic unit."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM units WHERE unit_id = ?", (unit_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown unit_id: {unit_id}")
        return _unit_row(row)

    def context_units(
        self,
        *,
        unit_id: int,
        level_index: int,
        window: int = 1,
    ) -> list[dict[str, Any]]:
        """Return a total window of groups around the focal group.

        ``window`` includes the focal group. Odd windows are balanced around the
        focal group. For even windows, the extra group is placed before the focal
        group, so a window of 6 yields 3 groups above, the focal group, and 2 below.
        """
        if window < 1:
            raise ValueError("window must be at least 1")
        with self.connect() as connection:
            focal = connection.execute(
                """
                SELECT g.group_id, g.parent_group_id, g.position_within_parent
                FROM unit_groups AS ug
                JOIN groups AS g ON g.group_id = ug.group_id
                WHERE ug.unit_id = ? AND ug.level_index = ?
                """,
                (unit_id, level_index),
            ).fetchone()
            if focal is None:
                raise KeyError(
                    f"No hierarchy level {level_index} for unit_id {unit_id}."
                )
            if focal["parent_group_id"] is None:
                sibling_rows = connection.execute(
                    """
                    SELECT group_id, position_within_parent FROM groups
                    WHERE level_index = ? AND parent_group_id IS NULL
                    ORDER BY position_within_parent
                    """,
                    (level_index,),
                ).fetchall()
            else:
                sibling_rows = connection.execute(
                    """
                    SELECT group_id, position_within_parent FROM groups
                    WHERE level_index = ? AND parent_group_id = ?
                    ORDER BY position_within_parent
                    """,
                    (level_index, focal["parent_group_id"]),
                ).fetchall()

            sibling_count = len(sibling_rows)
            desired = min(int(window), sibling_count)
            focal_position = int(focal["position_within_parent"])
            surrounding = desired - 1
            preferred_before = (surrounding + 1) // 2
            lower = focal_position - preferred_before
            lower = min(max(0, lower), max(0, sibling_count - desired))
            upper = lower + desired - 1

            if focal["parent_group_id"] is None:
                group_rows = connection.execute(
                    """
                    SELECT group_id FROM groups
                    WHERE level_index = ?
                      AND parent_group_id IS NULL
                      AND position_within_parent BETWEEN ? AND ?
                    ORDER BY position_within_parent
                    """,
                    (level_index, lower, upper),
                ).fetchall()
            else:
                group_rows = connection.execute(
                    """
                    SELECT group_id FROM groups
                    WHERE level_index = ?
                      AND parent_group_id = ?
                      AND position_within_parent BETWEEN ? AND ?
                    ORDER BY position_within_parent
                    """,
                    (level_index, focal["parent_group_id"], lower, upper),
                ).fetchall()
            group_ids = [int(row["group_id"]) for row in group_rows]
            if not group_ids:
                return []
            placeholders = ",".join("?" for _ in group_ids)
            rows = connection.execute(
                f"""
                SELECT DISTINCT u.*
                FROM units AS u
                JOIN unit_groups AS ug ON ug.unit_id = u.unit_id
                WHERE ug.level_index = ?
                  AND ug.group_id IN ({placeholders})
                ORDER BY u.row_position
                """,
                (level_index, *group_ids),
            ).fetchall()
        return [_unit_row(row) for row in rows]

    def get_observation(self, observation_id: int) -> dict[str, Any]:
        """Return one atomic, span, or teaching-example observation."""
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT o.observation_id, o.kind, o.text, o.created_at,
                       u.unit_id, u.row_position, u.user_key_json, u.metadata_json,
                       te.code_id AS teaching_code_id, te.label AS teaching_label,
                       te.status AS teaching_status, te.note AS teaching_note
                FROM observations o
                LEFT JOIN units u ON u.observation_id = o.observation_id
                LEFT JOIN teaching_examples te ON te.observation_id = o.observation_id
                WHERE o.observation_id = ?
                """,
                (observation_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown observation_id: {observation_id}")
            result = _observation_row(row)
            if result["kind"] == "span":
                result["member_unit_ids"] = [
                    int(item["unit_id"])
                    for item in connection.execute(
                        "SELECT unit_id FROM span_members WHERE span_observation_id = ? "
                        "ORDER BY member_order",
                        (observation_id,),
                    ).fetchall()
                ]
            return result

    def list_observations(
        self,
        *,
        kinds: Sequence[str] | None = None,
        include_deleted_teaching_examples: bool = False,
    ) -> list[dict[str, Any]]:
        """List observations in stable analytic order."""
        clauses: list[str] = []
        parameters: list[Any] = []
        if kinds:
            normalized = [str(kind) for kind in kinds]
            placeholders = ",".join("?" for _ in normalized)
            clauses.append(f"o.kind IN ({placeholders})")
            parameters.extend(normalized)
        if not include_deleted_teaching_examples:
            clauses.append("(o.kind != 'teaching_example' OR te.status != 'deleted')")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT o.observation_id, o.kind, o.text, o.created_at,
                       u.unit_id, u.row_position, u.user_key_json, u.metadata_json,
                       te.code_id AS teaching_code_id, te.label AS teaching_label,
                       te.status AS teaching_status, te.note AS teaching_note
                FROM observations AS o
                LEFT JOIN units AS u ON u.observation_id = o.observation_id
                LEFT JOIN teaching_examples AS te ON te.observation_id = o.observation_id
                {where}
                ORDER BY CASE o.kind
                           WHEN 'atomic' THEN 0 WHEN 'span' THEN 1 ELSE 2 END,
                         COALESCE(u.row_position, o.observation_id), o.observation_id
                """,
                parameters,
            ).fetchall()
        return [_observation_row(row) for row in rows]

    def observation_for_unit(self, unit_id: int) -> int:
        """Return the atomic observation identity for one imported unit."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT observation_id FROM units WHERE unit_id = ?", (unit_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown unit_id: {unit_id}")
        return int(row["observation_id"])

    def find_span(self, member_unit_ids: Sequence[int]) -> dict[str, Any] | None:
        """Find a persistent span by its exact ordered member signature."""
        members = [int(value) for value in member_unit_ids]
        if len(members) < 2:
            return None
        with self.connect() as connection:
            observation_id = _find_span_id(connection, members)
        return self.get_observation(observation_id) if observation_id is not None else None

    def validate_span_members(self, member_unit_ids: Sequence[int]) -> None:
        """Validate canonical order, contiguity, and immediate-parent membership."""
        members = [int(value) for value in member_unit_ids]
        if len(members) < 2:
            raise ValueError("A span must contain at least two atomic units.")
        with self.connect() as connection:
            _validate_contiguous_span(connection, members)

    def create_span_with_annotation(
        self,
        *,
        member_unit_ids: Sequence[int],
        text: str,
        code_id: int,
        value: str,
        origin: str,
        geometry_vectors: Mapping[int, tuple[str, bool]],
        classifier_fit_id: int | None = None,
    ) -> tuple[int, int]:
        """Create one span, its vectors, and first assignment in one transaction."""
        members = [int(value) for value in member_unit_ids]
        if len(members) < 2:
            raise ValueError("A span must contain at least two atomic units.")
        if value not in {"positive", "negative", "unsure"}:
            raise ValueError("value must be 'positive', 'negative', or 'unsure'")
        with self.connect() as connection:
            existing = _find_span_id(connection, members)
            if existing is not None:
                cursor = connection.execute(
                    """INSERT INTO annotation_events(
                        observation_id, code_id, value, origin, created_at, classifier_fit_id
                    ) VALUES (?, ?, ?, ?, ?, ?)""",
                    (existing, code_id, value, origin, utc_now(), classifier_fit_id),
                )
                return existing, int(cursor.lastrowid)
            _validate_contiguous_span(connection, members)
            cursor = connection.execute(
                "INSERT INTO observations(kind, text, created_at) VALUES ('span', ?, ?)",
                (text, utc_now()),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return a span observation ID.")
            observation_id = int(cursor.lastrowid)
            connection.executemany(
                "INSERT INTO span_members(span_observation_id, unit_id, member_order) "
                "VALUES (?, ?, ?)",
                ((observation_id, unit_id, order) for order, unit_id in enumerate(members)),
            )
            connection.executemany(
                """INSERT INTO observation_geometry_vectors(
                    observation_id, geometry_id, artifact_path, matrix_sparse, created_at
                ) VALUES (?, ?, ?, ?, ?)""",
                (
                    (observation_id, int(geometry_id), path, int(is_sparse), utc_now())
                    for geometry_id, (path, is_sparse) in geometry_vectors.items()
                ),
            )
            event = connection.execute(
                """INSERT INTO annotation_events(
                    observation_id, code_id, value, origin, created_at, classifier_fit_id
                ) VALUES (?, ?, ?, ?, ?, ?)""",
                (observation_id, code_id, value, origin, utc_now(), classifier_fit_id),
            )
            if event.lastrowid is None:
                raise RuntimeError("SQLite did not return an annotation event ID.")
            return observation_id, int(event.lastrowid)

    def create_teaching_example(
        self,
        *,
        code_id: int,
        text: str,
        label: str,
        note: str,
        geometry_vectors: Mapping[int, tuple[str, bool]],
    ) -> int:
        """Create an active teaching-example observation and assignment."""
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Teaching-example text may not be empty.")
        if label not in {"positive", "negative"}:
            raise ValueError("Teaching-example label must be positive or negative.")
        now = utc_now()
        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO observations(kind, text, created_at) VALUES ('teaching_example', ?, ?)",
                (clean_text, now),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return a teaching-example ID.")
            observation_id = int(cursor.lastrowid)
            connection.execute(
                """INSERT INTO teaching_examples(
                    observation_id, code_id, label, status, note, created_at, updated_at
                ) VALUES (?, ?, ?, 'active', ?, ?, ?)""",
                (observation_id, code_id, label, note, now, now),
            )
            connection.executemany(
                """INSERT INTO observation_geometry_vectors(
                    observation_id, geometry_id, artifact_path, matrix_sparse, created_at
                ) VALUES (?, ?, ?, ?, ?)""",
                (
                    (observation_id, int(geometry_id), path, int(is_sparse), now)
                    for geometry_id, (path, is_sparse) in geometry_vectors.items()
                ),
            )
            connection.execute(
                """INSERT INTO annotation_events(
                    observation_id, code_id, value, origin, created_at
                ) VALUES (?, ?, ?, 'teaching_example', ?)""",
                (observation_id, code_id, label, now),
            )
            return observation_id

    def list_teaching_examples(
        self, code_id: int, *, include_deleted: bool = False
    ) -> list[dict[str, Any]]:
        """List teaching examples for a code."""
        clause = "" if include_deleted else " AND te.status != 'deleted'"
        with self.connect() as connection:
            rows = connection.execute(
                f"""SELECT o.observation_id, o.kind, o.text, o.created_at,
                           te.code_id AS teaching_code_id, te.label AS teaching_label,
                           te.status AS teaching_status, te.note AS teaching_note
                    FROM teaching_examples te
                    JOIN observations o ON o.observation_id = te.observation_id
                    WHERE te.code_id = ?{clause}
                    ORDER BY o.observation_id""",
                (code_id,),
            ).fetchall()
        return [_observation_row(row) for row in rows]

    def set_teaching_example_status(self, observation_id: int, status: str) -> None:
        """Activate, deactivate, or soft-delete a teaching example."""
        if status not in {"active", "inactive", "deleted"}:
            raise ValueError("Unknown teaching-example status.")
        with self.connect() as connection:
            cursor = connection.execute(
                "UPDATE teaching_examples SET status = ?, updated_at = ? WHERE observation_id = ?",
                (status, utc_now(), observation_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown teaching-example observation_id: {observation_id}")

    def set_teaching_example_label(self, observation_id: int, label: str) -> int:
        """Relabel a teaching example and append a matching assignment event."""
        if label not in {"positive", "negative"}:
            raise ValueError("Teaching-example label must be positive or negative.")
        now = utc_now()
        with self.connect() as connection:
            row = connection.execute(
                "SELECT code_id, label FROM teaching_examples WHERE observation_id = ?",
                (observation_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown teaching-example observation_id: {observation_id}")
            if str(row["label"]) == label:
                current = connection.execute(
                    "SELECT annotation_event_id FROM current_annotations "
                    "WHERE observation_id = ? AND code_id = ?",
                    (observation_id, int(row["code_id"])),
                ).fetchone()
                return int(current["annotation_event_id"]) if current is not None else 0
            connection.execute(
                "UPDATE teaching_examples SET label = ?, updated_at = ? "
                "WHERE observation_id = ?",
                (label, now, observation_id),
            )
            event = connection.execute(
                """INSERT INTO annotation_events(
                    observation_id, code_id, value, origin, created_at
                ) VALUES (?, ?, ?, 'teaching_example_relabel', ?)""",
                (observation_id, int(row["code_id"]), label, now),
            )
            if event.lastrowid is None:
                raise RuntimeError("SQLite did not return an annotation event ID.")
            return int(event.lastrowid)

    def observation_vector_record(
        self, observation_id: int, geometry_id: int
    ) -> dict[str, Any] | None:
        """Return a derived observation's persisted vector record."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM observation_geometry_vectors "
                "WHERE observation_id = ? AND geometry_id = ?",
                (observation_id, geometry_id),
            ).fetchone()
        if row is None:
            return None
        return {
            "observation_id": int(row["observation_id"]),
            "geometry_id": int(row["geometry_id"]),
            "artifact_path": str(row["artifact_path"]),
            "matrix_sparse": bool(row["matrix_sparse"]),
            "created_at": str(row["created_at"]),
        }

    def create_session(self, title: str, state: dict[str, Any] | None = None) -> int:
        """Create a persistent named session."""
        clean_title = title.strip()
        if not clean_title:
            raise ValueError("Session title may not be empty.")
        now = utc_now()
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO sessions(title, created_at, last_active_at, state_json)
                VALUES (?, ?, ?, ?)
                """,
                (clean_title, now, now, json.dumps(state or {})),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return a session ID.")
            return int(cursor.lastrowid)

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all resumable sessions."""
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM sessions ORDER BY session_id"
            ).fetchall()
        return [
            {
                "session_id": int(row["session_id"]),
                "title": row["title"],
                "created_at": row["created_at"],
                "last_active_at": row["last_active_at"],
                "state": json.loads(row["state_json"]),
            }
            for row in rows
        ]

    def get_session(self, session_id: int) -> dict[str, Any]:
        """Return one resumable session."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown session_id: {session_id}")
        return {
            "session_id": int(row["session_id"]),
            "title": str(row["title"]),
            "created_at": str(row["created_at"]),
            "last_active_at": str(row["last_active_at"]),
            "state": json.loads(row["state_json"]),
        }

    def update_session_state(self, session_id: int, state: dict[str, Any]) -> None:
        """Replace resumable interface state for a session."""
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE sessions
                SET state_json = ?, last_active_at = ?
                WHERE session_id = ?
                """,
                (json.dumps(state), utc_now(), session_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown session_id: {session_id}")

    def patch_session_state(self, session_id: int, patch: dict[str, Any]) -> None:
        """Merge a small interface-state patch without replacing unrelated state."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT state_json FROM sessions WHERE session_id = ?",
                (int(session_id),),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown session_id: {session_id}")
            state = json.loads(row["state_json"])
            state.update(dict(patch))
            connection.execute(
                """
                UPDATE sessions
                SET state_json = ?, last_active_at = ?
                WHERE session_id = ?
                """,
                (json.dumps(state), utc_now(), int(session_id)),
            )

    def record_visit(
        self,
        *,
        session_id: int,
        unit_id: int,
        navigation_method: str,
        source_unit_id: int | None = None,
        geometry_id: int | None = None,
        active_code_id: int | None = None,
        classifier_fit_id: int | None = None,
    ) -> int:
        """Append one focal-unit navigation event."""
        with self.connect() as connection:
            next_index = connection.execute(
                """
                SELECT COALESCE(MAX(event_index), 0) + 1
                FROM navigation_events
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()[0]
            cursor = connection.execute(
                """
                INSERT INTO navigation_events(
                    session_id, event_index, unit_id, visited_at,
                    navigation_method, source_unit_id, geometry_id,
                    active_code_id, classifier_fit_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    next_index,
                    unit_id,
                    utc_now(),
                    navigation_method,
                    source_unit_id,
                    geometry_id,
                    active_code_id,
                    classifier_fit_id,
                ),
            )
            connection.execute(
                "UPDATE sessions SET last_active_at = ? WHERE session_id = ?",
                (utc_now(), session_id),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return a navigation event ID.")
            return int(cursor.lastrowid)

    def seen_unit_ids(self, session_id: int) -> set[int]:
        """Return units focalized at least once in a session."""
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT unit_id
                FROM navigation_events
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchall()
        return {int(row["unit_id"]) for row in rows}

    def create_code(self, name: str, definition: str = "") -> int:
        """Create a code and its first append-only description version."""
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Code name may not be empty.")
        now = utc_now()
        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO codes(name, definition, created_at) VALUES (?, ?, ?)",
                (clean_name, definition, now),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return a code ID.")
            code_id = int(cursor.lastrowid)
            version_cursor = connection.execute(
                """
                INSERT INTO code_versions(
                    code_id, version_number, name, description_markdown, created_at
                ) VALUES (?, 1, ?, ?, ?)
                """,
                (code_id, clean_name, definition, now),
            )
            if version_cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return a code version ID.")
            version_id = int(version_cursor.lastrowid)
            connection.executemany(
                "INSERT INTO code_version_hashtags(code_version_id, hashtag) VALUES (?, ?)",
                ((version_id, tag) for tag in _extract_hashtags(definition)),
            )
            return code_id

    def update_code(self, code_id: int, *, name: str, description: str) -> int:
        """Append a new code label/description version without rewriting history."""
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Code name may not be empty.")
        with self.connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM codes WHERE code_id = ?", (code_id,)
            ).fetchone()
            if exists is None:
                raise KeyError(f"Unknown code_id: {code_id}")
            version_number = int(connection.execute(
                "SELECT COALESCE(MAX(version_number), 0) + 1 FROM code_versions WHERE code_id = ?",
                (code_id,),
            ).fetchone()[0])
            cursor = connection.execute(
                """
                INSERT INTO code_versions(
                    code_id, version_number, name, description_markdown, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (code_id, version_number, clean_name, description, utc_now()),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return a code version ID.")
            version_id = int(cursor.lastrowid)
            connection.executemany(
                "INSERT INTO code_version_hashtags(code_version_id, hashtag) VALUES (?, ?)",
                ((version_id, tag) for tag in _extract_hashtags(description)),
            )
            # Keep the legacy columns as a latest-version cache while the
            # append-only code_versions table remains authoritative.
            connection.execute(
                "UPDATE codes SET name = ?, definition = ? WHERE code_id = ?",
                (clean_name, description, code_id),
            )
            return version_id

    def list_codes(self) -> list[dict[str, Any]]:
        """List codes with their latest version in creation order."""
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT c.code_id, c.created_at, cv.code_version_id, cv.version_number,
                       cv.name, cv.description_markdown, cv.created_at AS version_created_at
                FROM codes c
                JOIN code_versions cv ON cv.code_version_id = (
                    SELECT cv2.code_version_id FROM code_versions cv2
                    WHERE cv2.code_id = c.code_id
                    ORDER BY cv2.version_number DESC LIMIT 1
                )
                ORDER BY c.code_id
                """
            ).fetchall()
        return [_code_version_row(row) for row in rows]

    def get_code(self, code_id: int, version_number: int | None = None) -> dict[str, Any]:
        """Return one code version; latest when no version is requested."""
        query = """
            SELECT c.code_id, c.created_at, cv.code_version_id, cv.version_number,
                   cv.name, cv.description_markdown, cv.created_at AS version_created_at
            FROM codes c JOIN code_versions cv ON cv.code_id = c.code_id
            WHERE c.code_id = ?
        """
        parameters: tuple[Any, ...] = (code_id,)
        if version_number is not None:
            query += " AND cv.version_number = ?"
            parameters = (code_id, int(version_number))
        query += " ORDER BY cv.version_number DESC LIMIT 1"
        with self.connect() as connection:
            row = connection.execute(query, parameters).fetchone()
            if row is None:
                raise KeyError(f"Unknown code/version: {code_id}/{version_number}")
            result = _code_version_row(row)
            result["hashtags"] = [
                str(item["hashtag"]) for item in connection.execute(
                    "SELECT hashtag FROM code_version_hashtags WHERE code_version_id = ? ORDER BY hashtag",
                    (int(row["code_version_id"]),),
                ).fetchall()
            ]
        return result

    def code_versions(self, code_id: int) -> list[dict[str, Any]]:
        """Return every immutable version for one code."""
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT c.code_id, c.created_at, cv.code_version_id, cv.version_number,
                       cv.name, cv.description_markdown, cv.created_at AS version_created_at
                FROM codes c JOIN code_versions cv ON cv.code_id = c.code_id
                WHERE c.code_id = ? ORDER BY cv.version_number
                """,
                (code_id,),
            ).fetchall()
        if not rows:
            raise KeyError(f"Unknown code_id: {code_id}")
        return [_code_version_row(row) for row in rows]

    def code_hashtags(self) -> list[str]:
        """Return hashtags attached to latest code descriptions."""
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT cvh.hashtag
                FROM codes c
                JOIN code_versions cv ON cv.code_version_id = (
                    SELECT cv2.code_version_id FROM code_versions cv2
                    WHERE cv2.code_id = c.code_id ORDER BY cv2.version_number DESC LIMIT 1
                )
                JOIN code_version_hashtags cvh ON cvh.code_version_id = cv.code_version_id
                ORDER BY cvh.hashtag COLLATE NOCASE
                """
            ).fetchall()
        return [str(row["hashtag"]) for row in rows]

    def search_codes(
        self, *, query: str = "", scope: str = "full", hashtags: Iterable[str] = ()
    ) -> list[dict[str, Any]]:
        """Search current code labels/descriptions and hashtags."""
        if scope not in {"full", "label"}:
            raise ValueError("scope must be 'full' or 'label'")
        needle = query.strip().casefold()
        selected = {str(tag).strip().lstrip("#").casefold() for tag in hashtags if str(tag).strip().lstrip("#")}
        results: list[dict[str, Any]] = []
        for summary in self.list_codes():
            code = self.get_code(int(summary["code_id"]))
            haystack = str(code["name"])
            if scope == "full":
                haystack += "\n" + str(code["description"])
            if needle and needle not in haystack.casefold():
                continue
            if selected and not selected.issubset({str(tag).casefold() for tag in code["hashtags"]}):
                continue
            results.append(code)
        return results

    def positive_observations_for_code(self, code_id: int) -> list[dict[str, Any]]:
        """Return all observations currently labeled positive for one code."""
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT o.observation_id, o.kind, o.text, o.created_at,
                       u.unit_id, u.row_position, u.user_key_json, u.metadata_json,
                       te.code_id AS teaching_code_id,
                       te.label AS teaching_label,
                       te.status AS teaching_status, te.note AS teaching_note
                FROM current_annotations ca
                JOIN observations o ON o.observation_id = ca.observation_id
                LEFT JOIN units u ON u.observation_id = o.observation_id
                LEFT JOIN teaching_examples te ON te.observation_id = o.observation_id
                WHERE ca.code_id = ? AND ca.value = 'positive'
                  AND (o.kind != 'teaching_example' OR te.status != 'deleted')
                ORDER BY CASE o.kind WHEN 'atomic' THEN 0 WHEN 'span' THEN 1 ELSE 2 END,
                         COALESCE(u.row_position, o.observation_id), o.observation_id
                """,
                (code_id,),
            ).fetchall()
        return [_observation_row(row) for row in rows]

    def positive_units_for_code(self, code_id: int) -> list[dict[str, Any]]:
        """Return atomic observations currently labeled positive for one code."""
        return [
            row for row in self.positive_observations_for_code(code_id)
            if row["kind"] == "atomic"
        ]

    def annotate(
        self,
        *,
        observation_id: int,
        code_id: int,
        value: str,
        origin: str = "human_direct",
        classifier_fit_id: int | None = None,
    ) -> int:
        """Append a positive, negative, or unsure assignment event."""
        if value not in {"positive", "negative", "unsure"}:
            raise ValueError("value must be 'positive', 'negative', or 'unsure'")
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO annotation_events(
                    observation_id, code_id, value, origin, created_at, classifier_fit_id
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (observation_id, code_id, value, origin, utc_now(), classifier_fit_id),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return an annotation event ID.")
            return int(cursor.lastrowid)

    def current_annotation(
        self, *, observation_id: int, code_id: int
    ) -> dict[str, Any] | None:
        """Return the current assignment for one observation-code pair."""
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM current_annotations
                WHERE observation_id = ? AND code_id = ?
                """,
                (int(observation_id), int(code_id)),
            ).fetchone()
        if row is None:
            return None
        return {
            "annotation_event_id": int(row["annotation_event_id"]),
            "observation_id": int(row["observation_id"]),
            "unit_id": int(row["unit_id"]) if row["unit_id"] is not None else None,
            "code_id": int(row["code_id"]),
            "value": str(row["value"]),
            "origin": str(row["origin"]),
            "created_at": str(row["created_at"]),
            "classifier_fit_id": (
                int(row["classifier_fit_id"])
                if row["classifier_fit_id"] is not None
                else None
            ),
        }

    def current_annotations_for_observation(
        self, observation_id: int
    ) -> list[dict[str, Any]]:
        """Return all current code assignments for one observation in one query."""
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM current_annotations
                WHERE observation_id = ?
                ORDER BY code_id
                """,
                (int(observation_id),),
            ).fetchall()
        return [
            {
                "annotation_event_id": int(row["annotation_event_id"]),
                "observation_id": int(row["observation_id"]),
                "unit_id": int(row["unit_id"]) if row["unit_id"] is not None else None,
                "code_id": int(row["code_id"]),
                "value": str(row["value"]),
                "origin": str(row["origin"]),
                "created_at": str(row["created_at"]),
                "classifier_fit_id": (
                    int(row["classifier_fit_id"])
                    if row["classifier_fit_id"] is not None
                    else None
                ),
            }
            for row in rows
        ]

    def current_annotations(self, code_id: int | None = None) -> list[dict[str, Any]]:
        """Return current assignment states, optionally for one code."""
        query = "SELECT * FROM current_annotations"
        parameters: tuple[Any, ...] = ()
        if code_id is not None:
            query += " WHERE code_id = ?"
            parameters = (code_id,)
        query += " ORDER BY observation_id, code_id"
        with self.connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [
            {
                "annotation_event_id": int(row["annotation_event_id"]),
                "observation_id": int(row["observation_id"]),
                "unit_id": int(row["unit_id"]) if row["unit_id"] is not None else None,
                "code_id": int(row["code_id"]),
                "value": str(row["value"]),
                "origin": str(row["origin"]),
                "created_at": str(row["created_at"]),
                "classifier_fit_id": (
                    int(row["classifier_fit_id"]) if row["classifier_fit_id"] is not None else None
                ),
            }
            for row in rows
        ]

    def annotation_counts(self, code_id: int) -> dict[str, int]:
        """Count current assignment states for one code across all observations."""
        counts = {"positive": 0, "negative": 0, "unsure": 0}
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT value, COUNT(*) AS count
                FROM current_annotations
                WHERE code_id = ?
                GROUP BY value
                """,
                (code_id,),
            ).fetchall()
        for row in rows:
            counts[str(row["value"])] = int(row["count"])
        return counts

    def create_classifier_spec(
        self,
        *,
        code_id: int,
        name: str,
        geometry_id: int,
        algorithm: str,
        hyperparameters: Mapping[str, Any],
    ) -> int:
        """Create one code-owned classifier in the project-wide active predictor namespace."""
        clean_name = str(name).strip()
        if not clean_name:
            raise ValueError("Classifier name may not be empty.")
        with self.connect() as connection:
            _free_classifier_name(connection, clean_name)
            cursor = connection.execute(
                """
                INSERT INTO classifier_specs(
                    code_id, name, geometry_id, algorithm, hyperparameters_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    int(code_id),
                    clean_name,
                    int(geometry_id),
                    str(algorithm),
                    json.dumps(dict(hyperparameters), sort_keys=True),
                    utc_now(),
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return a classifier ID.")
            return int(cursor.lastrowid)

    def list_classifier_specs(
        self,
        *,
        code_id: int | None = None,
        geometry_id: int | None = None,
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        """List code-owned classifiers."""
        query = """
            SELECT cs.*, g.name AS geometry_name, c.name AS code_name
            FROM classifier_specs AS cs
            JOIN geometries AS g ON g.geometry_id = cs.geometry_id
            JOIN codes AS c ON c.code_id = cs.code_id
        """
        clauses: list[str] = []
        parameters: list[Any] = []
        if code_id is not None:
            clauses.append("cs.code_id = ?")
            parameters.append(int(code_id))
        if geometry_id is not None:
            clauses.append("cs.geometry_id = ?")
            parameters.append(int(geometry_id))
        if not include_deleted:
            clauses.append("cs.status = 'active'")
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY cs.classifier_spec_id"
        with self.connect() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
        return [_classifier_spec_row(row) for row in rows]

    def get_classifier_spec(self, classifier_spec_id: int) -> dict[str, Any]:
        """Return one code-owned classifier."""
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT cs.*, g.name AS geometry_name, c.name AS code_name
                FROM classifier_specs AS cs
                JOIN geometries AS g ON g.geometry_id = cs.geometry_id
                JOIN codes AS c ON c.code_id = cs.code_id
                WHERE cs.classifier_spec_id = ?
                """,
                (int(classifier_spec_id),),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown classifier_spec_id: {classifier_spec_id}")
        return _classifier_spec_row(row)

    def get_classifier_spec_by_name(
        self, name: str, *, include_deleted: bool = False
    ) -> dict[str, Any]:
        """Return one classifier by exact project-wide name."""
        query = """
            SELECT cs.*, g.name AS geometry_name, c.name AS code_name
            FROM classifier_specs AS cs
            JOIN geometries AS g ON g.geometry_id = cs.geometry_id
            JOIN codes AS c ON c.code_id = cs.code_id
            WHERE cs.name = ?
        """
        parameters: list[Any] = [str(name)]
        if not include_deleted:
            query += " AND cs.status = 'active'"
        with self.connect() as connection:
            row = connection.execute(query, parameters).fetchone()
        if row is None:
            raise KeyError(f"Unknown active classifier name: {name!r}")
        return _classifier_spec_row(row)

    def rename_classifier_spec(self, classifier_spec_id: int, *, name: str) -> None:
        """Rename one active classifier without changing its fitted/configured state."""
        clean_name = str(name).strip()
        if not clean_name:
            raise ValueError("Classifier name may not be empty.")
        with self.connect() as connection:
            current = connection.execute(
                "SELECT status FROM classifier_specs WHERE classifier_spec_id = ?",
                (int(classifier_spec_id),),
            ).fetchone()
            if current is None:
                raise KeyError(f"Unknown classifier_spec_id: {classifier_spec_id}")
            if str(current["status"]) != "active":
                raise ValueError("Deleted classifiers cannot be renamed.")
            _free_classifier_name(
                connection, clean_name, exclude_classifier_spec_id=int(classifier_spec_id)
            )
            connection.execute(
                "UPDATE classifier_specs SET name = ? WHERE classifier_spec_id = ?",
                (clean_name, int(classifier_spec_id)),
            )

    def update_classifier_spec(
        self,
        classifier_spec_id: int,
        *,
        name: str,
        hyperparameters: Mapping[str, Any],
    ) -> None:
        """Update one classifier for its next explicit training run."""
        clean_name = str(name).strip()
        if not clean_name:
            raise ValueError("Classifier name may not be empty.")
        with self.connect() as connection:
            current = connection.execute(
                "SELECT status FROM classifier_specs WHERE classifier_spec_id = ?",
                (int(classifier_spec_id),),
            ).fetchone()
            if current is None:
                raise KeyError(f"Unknown classifier_spec_id: {classifier_spec_id}")
            if str(current["status"]) != "active":
                raise ValueError("Deleted classifiers cannot be edited.")
            _free_classifier_name(
                connection, clean_name, exclude_classifier_spec_id=int(classifier_spec_id)
            )
            connection.execute(
                """
                UPDATE classifier_specs
                SET name = ?, hyperparameters_json = ?
                WHERE classifier_spec_id = ?
                """,
                (
                    clean_name,
                    json.dumps(dict(hyperparameters), sort_keys=True),
                    int(classifier_spec_id),
                ),
            )

    def delete_classifier_spec(self, classifier_spec_id: int) -> str:
        """Soft-delete one classifier, archiving its name so the original is reusable."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT name, status FROM classifier_specs WHERE classifier_spec_id = ?",
                (int(classifier_spec_id),),
            ).fetchone()
            if row is None or str(row["status"]) != "active":
                raise KeyError(f"Unknown active classifier_spec_id: {classifier_spec_id}")
            archived_name = _next_classifier_deleted_name(
                connection, str(row["name"]), exclude_classifier_spec_id=int(classifier_spec_id)
            )
            connection.execute(
                """
                UPDATE classifier_specs
                SET name = ?, status = 'deleted'
                WHERE classifier_spec_id = ?
                """,
                (archived_name, int(classifier_spec_id)),
            )
            connection.execute(
                "DELETE FROM classifier_committee_members WHERE classifier_spec_id = ?",
                (int(classifier_spec_id),),
            )
            connection.execute(
                """
                UPDATE classifier_committees
                SET status = 'deleted', updated_at = ?
                WHERE status = 'active'
                  AND NOT EXISTS (
                    SELECT 1
                    FROM classifier_committee_members AS members
                    WHERE members.committee_id = classifier_committees.committee_id
                )
                """,
                (utc_now(),),
            )
            return archived_name

    def find_classifier_fit(
        self,
        *,
        code_id: int,
        classifier_spec_id: int,
        geometry_id: int,
        algorithm: str,
        hyperparameters: Mapping[str, Any],
        training_snapshot: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Find an immutable fit matching both specification and training state."""
        snapshot_json = json.dumps(training_snapshot, sort_keys=True)
        hyperparameters_json = json.dumps(dict(hyperparameters), sort_keys=True)
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT cf.*, g.name AS geometry_name
                FROM classifier_fits AS cf
                JOIN geometries AS g ON g.geometry_id = cf.geometry_id
                WHERE cf.code_id = ? AND cf.classifier_spec_id = ?
                  AND cf.geometry_id = ? AND cf.algorithm = ?
                  AND cf.hyperparameters_json = ?
                  AND cf.training_snapshot_json = ?
                  AND cf.artifact_path != ''
                ORDER BY cf.classifier_fit_id DESC
                LIMIT 1
                """,
                (
                    int(code_id), int(classifier_spec_id), int(geometry_id),
                    str(algorithm), hyperparameters_json, snapshot_json,
                ),
            ).fetchone()
        return _classifier_fit_row(row) if row is not None else None

    def latest_classifier_fit(
        self, *, code_id: int, classifier_spec_id: int
    ) -> dict[str, Any] | None:
        """Return the newest fit for a code/specification pair."""
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT cf.*, g.name AS geometry_name
                FROM classifier_fits AS cf
                JOIN geometries AS g ON g.geometry_id = cf.geometry_id
                WHERE cf.code_id = ? AND cf.classifier_spec_id = ?
                  AND cf.artifact_path != ''
                ORDER BY cf.classifier_fit_id DESC
                LIMIT 1
                """,
                (int(code_id), int(classifier_spec_id)),
            ).fetchone()
        return _classifier_fit_row(row) if row is not None else None

    def list_classifier_fits(
        self,
        *,
        code_id: int | None = None,
        classifier_spec_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """List immutable classifier fits, newest first."""
        clauses: list[str] = []
        parameters: list[Any] = []
        if code_id is not None:
            clauses.append("cf.code_id = ?")
            parameters.append(int(code_id))
        if classifier_spec_id is not None:
            clauses.append("cf.classifier_spec_id = ?")
            parameters.append(int(classifier_spec_id))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT cf.*, g.name AS geometry_name
                FROM classifier_fits AS cf
                JOIN geometries AS g ON g.geometry_id = cf.geometry_id
                {where}
                ORDER BY cf.classifier_fit_id DESC
                """,
                parameters,
            ).fetchall()
        return [_classifier_fit_row(row) for row in rows]

    def register_classifier_fit(
        self,
        *,
        code_id: int,
        classifier_spec_id: int,
        classifier_name: str,
        geometry_id: int,
        algorithm: str,
        hyperparameters: Mapping[str, Any],
        training_snapshot: dict[str, Any],
        artifact_path: str,
        score_kind: str,
        predictions: Sequence[Mapping[str, Any]],
    ) -> int:
        """Persist an immutable classifier fit and observation-level predictions."""
        if score_kind not in {"probability", "decision_score"}:
            raise ValueError("Unknown classifier score kind.")
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO classifier_fits(
                    classifier_spec_id, code_id, classifier_name, geometry_id,
                    algorithm, hyperparameters_json, training_snapshot_json,
                    artifact_path, score_kind, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(classifier_spec_id),
                    int(code_id),
                    str(classifier_name),
                    int(geometry_id),
                    str(algorithm),
                    json.dumps(dict(hyperparameters), sort_keys=True),
                    json.dumps(training_snapshot, sort_keys=True),
                    str(artifact_path),
                    str(score_kind),
                    utc_now(),
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return a classifier fit ID.")
            classifier_fit_id = int(cursor.lastrowid)
            connection.executemany(
                """
                INSERT INTO classifier_predictions(
                    classifier_fit_id, observation_id, predicted_label,
                    probability, decision_score, uncertainty
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        classifier_fit_id,
                        int(row["observation_id"]),
                        str(row["predicted_label"]),
                        (
                            float(row["probability"])
                            if row.get("probability") is not None
                            else None
                        ),
                        (
                            float(row["decision_score"])
                            if row.get("decision_score") is not None
                            else None
                        ),
                        float(row["uncertainty"]),
                    )
                    for row in predictions
                ),
            )
            return classifier_fit_id

    def get_classifier_fit(self, classifier_fit_id: int) -> dict[str, Any]:
        """Return one persisted classifier fit."""
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT cf.*, g.name AS geometry_name
                FROM classifier_fits AS cf
                JOIN geometries AS g ON g.geometry_id = cf.geometry_id
                WHERE cf.classifier_fit_id = ?
                """,
                (int(classifier_fit_id),),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown classifier_fit_id: {classifier_fit_id}")
        return _classifier_fit_row(row)

    def classifier_predictions(
        self,
        classifier_fit_id: int,
        *,
        atomic_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Return fit predictions in stable observation order."""
        atomic_clause = "AND o.kind = 'atomic'" if atomic_only else ""
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT cp.*, o.kind, o.text, u.unit_id, u.row_position
                FROM classifier_predictions AS cp
                JOIN observations AS o ON o.observation_id = cp.observation_id
                LEFT JOIN units AS u ON u.observation_id = cp.observation_id
                WHERE cp.classifier_fit_id = ? {atomic_clause}
                ORDER BY CASE WHEN u.row_position IS NULL THEN 1 ELSE 0 END,
                         u.row_position, cp.observation_id
                """,
                (int(classifier_fit_id),),
            ).fetchall()
        return [_classifier_prediction_row(row) for row in rows]

    def classifier_prediction_count(self, classifier_fit_id: int) -> int:
        """Return the number of persisted predictions for one classifier fit."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS n FROM classifier_predictions WHERE classifier_fit_id = ?",
                (int(classifier_fit_id),),
            ).fetchone()
        return int(row["n"]) if row is not None else 0

    def missing_classifier_prediction_observation_ids(
        self, classifier_fit_id: int
    ) -> list[int]:
        """Return active observations lacking a prediction for one classifier fit."""
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT o.observation_id
                FROM observations AS o
                LEFT JOIN teaching_examples AS te ON te.observation_id = o.observation_id
                LEFT JOIN classifier_predictions AS cp
                  ON cp.observation_id = o.observation_id
                 AND cp.classifier_fit_id = ?
                WHERE (o.kind != 'teaching_example' OR te.status != 'deleted')
                  AND cp.observation_id IS NULL
                ORDER BY o.observation_id
                """,
                (int(classifier_fit_id),),
            ).fetchall()
        return [int(row["observation_id"]) for row in rows]

    def classifier_score_maps(
        self, classifier_fit_ids: Sequence[int]
    ) -> dict[int, dict[int, float]]:
        """Return lean observation->score maps for several fits in one SQLite query."""
        fit_ids = list(dict.fromkeys(int(value) for value in classifier_fit_ids))
        if not fit_ids:
            return {}
        placeholders = ",".join("?" for _ in fit_ids)
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT classifier_fit_id, observation_id, probability, decision_score
                FROM classifier_predictions
                WHERE classifier_fit_id IN ({placeholders})
                ORDER BY observation_id, classifier_fit_id
                """,
                fit_ids,
            ).fetchall()
        result: dict[int, dict[int, float]] = {fit_id: {} for fit_id in fit_ids}
        for row in rows:
            score = row["probability"]
            if score is None:
                score = row["decision_score"]
            if score is not None:
                result[int(row["classifier_fit_id"])][int(row["observation_id"])] = float(score)
        return result

    def classifier_atomic_score_vectors(
        self, classifier_fit_ids: Sequence[int]
    ) -> dict[int, dict[int, float]]:
        """Return lean unit->score maps for several fits in one SQLite query."""
        fit_ids = list(dict.fromkeys(int(value) for value in classifier_fit_ids))
        if not fit_ids:
            return {}
        placeholders = ",".join("?" for _ in fit_ids)
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT cp.classifier_fit_id, u.unit_id, cp.probability, cp.decision_score
                FROM classifier_predictions AS cp
                JOIN units AS u ON u.observation_id = cp.observation_id
                WHERE cp.classifier_fit_id IN ({placeholders})
                ORDER BY u.row_position, cp.classifier_fit_id
                """,
                fit_ids,
            ).fetchall()
        result: dict[int, dict[int, float]] = {fit_id: {} for fit_id in fit_ids}
        for row in rows:
            score = row["probability"]
            if score is None:
                score = row["decision_score"]
            if score is not None:
                result[int(row["classifier_fit_id"])][int(row["unit_id"])] = float(score)
        return result

    def classifier_scores_for_unit(
        self, classifier_fit_ids: Sequence[int], unit_id: int
    ) -> dict[int, float]:
        """Return probability-like scores for several classifier fits at one unit."""
        fit_ids = list(dict.fromkeys(int(value) for value in classifier_fit_ids))
        if not fit_ids:
            return {}
        placeholders = ",".join("?" for _ in fit_ids)
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT cp.classifier_fit_id, cp.probability, cp.decision_score
                FROM classifier_predictions AS cp
                JOIN units AS u ON u.observation_id = cp.observation_id
                WHERE cp.classifier_fit_id IN ({placeholders}) AND u.unit_id = ?
                """,
                [*fit_ids, int(unit_id)],
            ).fetchall()
        result: dict[int, float] = {}
        for row in rows:
            score = row["probability"]
            if score is None:
                score = row["decision_score"]
            if score is not None:
                result[int(row["classifier_fit_id"])] = float(score)
        return result

    def classifier_score_for_unit(
        self, classifier_fit_id: int, unit_id: int
    ) -> float | None:
        """Return one classifier probability-like score for one atomic unit."""
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT cp.probability, cp.decision_score
                FROM classifier_predictions AS cp
                JOIN units AS u ON u.observation_id = cp.observation_id
                WHERE cp.classifier_fit_id = ? AND u.unit_id = ?
                """,
                (int(classifier_fit_id), int(unit_id)),
            ).fetchone()
        if row is None:
            return None
        score = row["probability"]
        if score is None:
            score = row["decision_score"]
        return float(score) if score is not None else None

    def append_classifier_predictions(
        self,
        classifier_fit_id: int,
        predictions: Sequence[Mapping[str, Any]],
    ) -> None:
        """Append scores for newly ingested observations to an existing fit."""
        if not predictions:
            return
        self.get_classifier_fit(int(classifier_fit_id))
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO classifier_predictions(
                    classifier_fit_id, observation_id, predicted_label,
                    probability, decision_score, uncertainty
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        int(classifier_fit_id),
                        int(row["observation_id"]),
                        str(row["predicted_label"]),
                        (
                            float(row["probability"])
                            if row.get("probability") is not None
                            else None
                        ),
                        (
                            float(row["decision_score"])
                            if row.get("decision_score") is not None
                            else None
                        ),
                        float(row["uncertainty"]),
                    )
                    for row in predictions
                ),
            )

    def retire_classifier_fit_states(
        self,
        classifier_spec_id: int,
        *,
        keep_classifier_fit_id: int | None = None,
    ) -> list[str]:
        """Discard superseded estimator/prediction state while retaining fit provenance."""
        clauses = ["classifier_spec_id = ?", "artifact_path != ''"]
        parameters: list[Any] = [int(classifier_spec_id)]
        if keep_classifier_fit_id is not None:
            clauses.append("classifier_fit_id != ?")
            parameters.append(int(keep_classifier_fit_id))
        where = " AND ".join(clauses)
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT classifier_fit_id, artifact_path FROM classifier_fits WHERE {where}",
                parameters,
            ).fetchall()
            fit_ids = [int(row["classifier_fit_id"]) for row in rows]
            if fit_ids:
                placeholders = ",".join("?" for _ in fit_ids)
                connection.execute(
                    f"DELETE FROM classifier_predictions WHERE classifier_fit_id IN ({placeholders})",
                    fit_ids,
                )
                connection.execute(
                    f"UPDATE classifier_fits SET artifact_path = '' WHERE classifier_fit_id IN ({placeholders})",
                    fit_ids,
                )
        return [str(row["artifact_path"]) for row in rows if str(row["artifact_path"])]

    def create_classifier_committee(
        self,
        *,
        code_id: int,
        name: str,
        classifier_spec_ids: Sequence[int],
        aggregation: str = "mean",
    ) -> int:
        """Create a named classifier committee in the shared active predictor namespace."""
        clean_name = str(name).strip()
        members = list(dict.fromkeys(int(value) for value in classifier_spec_ids))
        if not clean_name:
            raise ValueError("Committee name may not be empty.")
        if not members:
            raise ValueError("A committee requires at least one classifier.")
        if aggregation not in COMMITTEE_AGGREGATIONS:
            raise ValueError("Unknown classifier-committee aggregation.")
        now = utc_now()
        with self.connect() as connection:
            _free_committee_name(connection, int(code_id), clean_name)
            cursor = connection.execute(
                """
                INSERT INTO classifier_committees(
                    code_id, name, aggregation, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (int(code_id), clean_name, aggregation, now, now),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return a committee ID.")
            committee_id = int(cursor.lastrowid)
            connection.executemany(
                """
                INSERT INTO classifier_committee_members(
                    committee_id, classifier_spec_id, member_order, weight
                ) VALUES (?, ?, ?, 1.0)
                """,
                (
                    (committee_id, classifier_spec_id, position)
                    for position, classifier_spec_id in enumerate(members)
                ),
            )
            return committee_id

    def update_classifier_committee(
        self,
        committee_id: int,
        *,
        name: str,
        classifier_spec_ids: Sequence[int],
        aggregation: str,
    ) -> None:
        """Replace a committee's name, aggregation, and ordered membership."""
        clean_name = str(name).strip()
        members = list(dict.fromkeys(int(value) for value in classifier_spec_ids))
        if not clean_name:
            raise ValueError("Committee name may not be empty.")
        if not members:
            raise ValueError("A committee requires at least one classifier.")
        if aggregation not in COMMITTEE_AGGREGATIONS:
            raise ValueError("Unknown classifier-committee aggregation.")
        with self.connect() as connection:
            current = connection.execute(
                "SELECT code_id, status FROM classifier_committees WHERE committee_id = ?",
                (int(committee_id),),
            ).fetchone()
            if current is None:
                raise KeyError(f"Unknown committee_id: {committee_id}")
            if str(current["status"]) != "active":
                raise ValueError("Deleted committees cannot be edited.")
            _free_committee_name(
                connection, int(current["code_id"]), clean_name,
                exclude_committee_id=int(committee_id),
            )
            cursor = connection.execute(
                """
                UPDATE classifier_committees
                SET name = ?, aggregation = ?, updated_at = ?
                WHERE committee_id = ?
                """,
                (clean_name, aggregation, utc_now(), int(committee_id)),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown committee_id: {committee_id}")
            connection.execute(
                "DELETE FROM classifier_committee_members WHERE committee_id = ?",
                (int(committee_id),),
            )
            connection.executemany(
                """
                INSERT INTO classifier_committee_members(
                    committee_id, classifier_spec_id, member_order, weight
                ) VALUES (?, ?, ?, 1.0)
                """,
                (
                    (int(committee_id), classifier_spec_id, position)
                    for position, classifier_spec_id in enumerate(members)
                ),
            )

    def rename_classifier_committee(self, committee_id: int, *, name: str) -> None:
        """Rename one active committee without changing its definition or fitted state."""
        clean_name = str(name).strip()
        if not clean_name:
            raise ValueError("Committee name may not be empty.")
        with self.connect() as connection:
            current = connection.execute(
                "SELECT code_id, status FROM classifier_committees WHERE committee_id = ?",
                (int(committee_id),),
            ).fetchone()
            if current is None:
                raise KeyError(f"Unknown committee_id: {committee_id}")
            if str(current["status"]) != "active":
                raise ValueError("Deleted committees cannot be renamed.")
            _free_committee_name(
                connection, int(current["code_id"]), clean_name,
                exclude_committee_id=int(committee_id),
            )
            # Deliberately do not touch updated_at: that timestamp is part of the
            # learned-committee training snapshot and a cosmetic rename must not
            # invalidate a fitted stacker.
            connection.execute(
                "UPDATE classifier_committees SET name = ? WHERE committee_id = ?",
                (clean_name, int(committee_id)),
            )

    def list_classifier_committees(
        self, *, code_id: int | None = None, include_deleted: bool = False
    ) -> list[dict[str, Any]]:
        """List committees and their ordered classifier members."""
        clauses: list[str] = []
        parameters: list[Any] = []
        if code_id is not None:
            clauses.append("code_id = ?")
            parameters.append(int(code_id))
        if not include_deleted:
            clauses.append("status = 'active'")
        query = "SELECT * FROM classifier_committees"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY committee_id"
        with self.connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
            results = []
            for row in rows:
                item = _classifier_committee_row(row)
                item["members"] = [
                    {
                        "classifier_spec_id": int(member["classifier_spec_id"]),
                        "classifier_name": str(member["classifier_name"]),
                        "geometry_id": int(member["geometry_id"]),
                        "geometry_name": str(member["geometry_name"]),
                        "member_order": int(member["member_order"]),
                        "weight": float(member["weight"]),
                    }
                    for member in connection.execute(
                        """
                        SELECT ccm.*, cs.name AS classifier_name, cs.geometry_id,
                               g.name AS geometry_name
                        FROM classifier_committee_members AS ccm
                        JOIN classifier_specs AS cs
                          ON cs.classifier_spec_id = ccm.classifier_spec_id
                        JOIN geometries AS g ON g.geometry_id = cs.geometry_id
                        WHERE ccm.committee_id = ?
                        ORDER BY ccm.member_order
                        """,
                        (int(row["committee_id"]),),
                    ).fetchall()
                ]
                results.append(item)
        return results

    def get_classifier_committee(self, committee_id: int) -> dict[str, Any]:
        """Return one named classifier committee."""
        committees = [
            row
            for row in self.list_classifier_committees(include_deleted=True)
            if int(row["committee_id"]) == int(committee_id)
        ]
        if not committees:
            raise KeyError(f"Unknown committee_id: {committee_id}")
        return committees[0]

    def delete_classifier_committee(self, committee_id: int) -> None:
        """Soft-delete one committee and archive its name for active reuse."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT code_id, name, status FROM classifier_committees WHERE committee_id = ?",
                (int(committee_id),),
            ).fetchone()
            if row is None or str(row["status"]) != "active":
                raise KeyError(f"Unknown active committee_id: {committee_id}")
            archived_name = _next_committee_deleted_name(
                connection, int(row["code_id"]), str(row["name"]),
                exclude_committee_id=int(committee_id),
            )
            connection.execute(
                """
                UPDATE classifier_committees
                SET name = ?, status = 'deleted', updated_at = ?
                WHERE committee_id = ?
                """,
                (archived_name, utc_now(), int(committee_id)),
            )

    def register_classifier_committee_fit(
        self,
        *,
        committee_id: int,
        code_id: int,
        aggregation: str,
        member_fit_ids: Sequence[int],
        training_snapshot: Mapping[str, Any],
        artifact_path: str,
        predictions: Sequence[Mapping[str, Any]],
    ) -> int:
        """Persist one trainable committee fit and its observation scores."""
        if aggregation not in TRAINABLE_COMMITTEE_AGGREGATIONS:
            raise ValueError("Only trainable committee aggregations create fits.")
        fit_ids = [int(value) for value in member_fit_ids]
        if not fit_ids:
            raise ValueError("A committee fit requires member classifier fits.")
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO classifier_committee_fits(
                    committee_id, code_id, aggregation, member_fit_ids_json,
                    training_snapshot_json, artifact_path, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(committee_id), int(code_id), aggregation,
                    json.dumps(fit_ids),
                    json.dumps(dict(training_snapshot), sort_keys=True),
                    str(artifact_path), utc_now(),
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return a committee fit ID.")
            committee_fit_id = int(cursor.lastrowid)
            connection.executemany(
                """
                INSERT INTO classifier_committee_predictions(
                    committee_fit_id, observation_id, predicted_label,
                    probability, uncertainty
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    (
                        committee_fit_id, int(row["observation_id"]),
                        str(row["predicted_label"]), float(row["probability"]),
                        float(row["uncertainty"]),
                    )
                    for row in predictions
                ),
            )
            return committee_fit_id

    def latest_classifier_committee_fit(
        self, *, committee_id: int, code_id: int
    ) -> dict[str, Any] | None:
        """Return the newest persisted fit for one trainable committee."""
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM classifier_committee_fits
                WHERE committee_id = ? AND code_id = ?
                ORDER BY committee_fit_id DESC LIMIT 1
                """,
                (int(committee_id), int(code_id)),
            ).fetchone()
        return _classifier_committee_fit_row(row) if row is not None else None

    def get_classifier_committee_fit(self, committee_fit_id: int) -> dict[str, Any]:
        """Return one immutable trainable committee fit."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM classifier_committee_fits WHERE committee_fit_id = ?",
                (int(committee_fit_id),),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown committee_fit_id: {committee_fit_id}")
        return _classifier_committee_fit_row(row)

    def classifier_committee_predictions(
        self, committee_fit_id: int
    ) -> list[dict[str, Any]]:
        """Return persisted scores for one trainable committee fit."""
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT ccp.*, o.kind, o.text, u.unit_id, u.row_position
                FROM classifier_committee_predictions AS ccp
                JOIN observations AS o ON o.observation_id = ccp.observation_id
                LEFT JOIN units AS u ON u.observation_id = ccp.observation_id
                WHERE ccp.committee_fit_id = ?
                ORDER BY CASE WHEN u.row_position IS NULL THEN 1 ELSE 0 END,
                         u.row_position, ccp.observation_id
                """,
                (int(committee_fit_id),),
            ).fetchall()
        return [
            {
                "committee_fit_id": int(row["committee_fit_id"]),
                "observation_id": int(row["observation_id"]),
                "kind": str(row["kind"]),
                "text": str(row["text"]),
                "unit_id": int(row["unit_id"]) if row["unit_id"] is not None else None,
                "row_position": int(row["row_position"]) if row["row_position"] is not None else None,
                "predicted_label": str(row["predicted_label"]),
                "probability": float(row["probability"]),
                "uncertainty": float(row["uncertainty"]),
            }
            for row in rows
        ]

    def missing_classifier_committee_prediction_observation_ids(
        self, committee_fit_id: int
    ) -> list[int]:
        """Return active observations lacking a learned-committee prediction."""
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT o.observation_id
                FROM observations AS o
                LEFT JOIN teaching_examples AS te ON te.observation_id = o.observation_id
                LEFT JOIN classifier_committee_predictions AS ccp
                  ON ccp.observation_id = o.observation_id
                 AND ccp.committee_fit_id = ?
                WHERE (o.kind != 'teaching_example' OR te.status != 'deleted')
                  AND ccp.observation_id IS NULL
                ORDER BY o.observation_id
                """,
                (int(committee_fit_id),),
            ).fetchall()
        return [int(row["observation_id"]) for row in rows]

    def classifier_committee_atomic_probabilities(
        self, committee_fit_id: int
    ) -> dict[int, float]:
        """Return a lean unit->probability map for one learned committee fit."""
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT u.unit_id, ccp.probability
                FROM classifier_committee_predictions AS ccp
                JOIN units AS u ON u.observation_id = ccp.observation_id
                WHERE ccp.committee_fit_id = ?
                ORDER BY u.row_position
                """,
                (int(committee_fit_id),),
            ).fetchall()
        return {int(row["unit_id"]): float(row["probability"]) for row in rows}

    def classifier_committee_probability_for_unit(
        self, committee_fit_id: int, unit_id: int
    ) -> float | None:
        """Return one learned committee probability for one atomic unit."""
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT ccp.probability
                FROM classifier_committee_predictions AS ccp
                JOIN units AS u ON u.observation_id = ccp.observation_id
                WHERE ccp.committee_fit_id = ? AND u.unit_id = ?
                """,
                (int(committee_fit_id), int(unit_id)),
            ).fetchone()
        return float(row["probability"]) if row is not None else None

    def append_classifier_committee_predictions(
        self,
        committee_fit_id: int,
        predictions: Sequence[Mapping[str, Any]],
    ) -> None:
        """Append scores for newly ingested observations to a trained committee."""
        if not predictions:
            return
        self.get_classifier_committee_fit(int(committee_fit_id))
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO classifier_committee_predictions(
                    committee_fit_id, observation_id, predicted_label,
                    probability, uncertainty
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    (
                        int(committee_fit_id),
                        int(row["observation_id"]),
                        str(row["predicted_label"]),
                        float(row["probability"]),
                        float(row["uncertainty"]),
                    )
                    for row in predictions
                ),
            )

    def register_classifier_evaluation(
        self,
        *,
        event_type: str,
        code_id: int,
        classifier_fit_id: int,
        observation_id: int | None,
        human_label: str,
        predicted_label: str,
        probability: float | None,
        decision_score: float | None,
        added_to_training: bool,
    ) -> int:
        """Persist one diagnostic classifier assessment event."""
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO classifier_evaluation_events(
                    event_type, code_id, classifier_fit_id, observation_id,
                    human_label, predicted_label, probability, decision_score,
                    added_to_training, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(event_type),
                    int(code_id),
                    int(classifier_fit_id),
                    int(observation_id) if observation_id is not None else None,
                    str(human_label),
                    str(predicted_label),
                    float(probability) if probability is not None else None,
                    float(decision_score) if decision_score is not None else None,
                    int(bool(added_to_training)),
                    utc_now(),
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return an evaluation event ID.")
            return int(cursor.lastrowid)

    def list_classifier_evaluations(
        self, *, code_id: int | None = None
    ) -> list[dict[str, Any]]:
        """List diagnostic classifier assessment events."""
        query = """
            SELECT cee.*, cf.classifier_name, g.name AS geometry_name
            FROM classifier_evaluation_events AS cee
            JOIN classifier_fits AS cf
              ON cf.classifier_fit_id = cee.classifier_fit_id
            JOIN geometries AS g ON g.geometry_id = cf.geometry_id
        """
        parameters: tuple[Any, ...] = ()
        if code_id is not None:
            query += " WHERE cee.code_id = ?"
            parameters = (int(code_id),)
        query += " ORDER BY cee.evaluation_event_id DESC"
        with self.connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [_classifier_evaluation_row(row) for row in rows]

    def register_classifier_tuning_run(
        self,
        *,
        code_id: int,
        classifier_spec_id: int,
        metric: str,
        folds: int,
        candidate_lambdas: Sequence[float],
        results: Sequence[Mapping[str, Any]],
        selected_lambda: float,
    ) -> int:
        """Persist one heuristic cross-validation search for classifier regularization."""
        if metric not in {"log_loss", "brier", "accuracy"}:
            raise ValueError("Unknown classifier tuning metric.")
        if int(folds) < 2:
            raise ValueError("Classifier tuning requires at least two folds.")
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO classifier_tuning_runs(
                    code_id, classifier_spec_id, metric, folds,
                    candidate_lambdas_json, results_json, selected_lambda, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(code_id),
                    int(classifier_spec_id),
                    str(metric),
                    int(folds),
                    json.dumps([float(value) for value in candidate_lambdas]),
                    json.dumps([dict(row) for row in results], sort_keys=True),
                    float(selected_lambda),
                    utc_now(),
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return a classifier tuning run ID.")
            return int(cursor.lastrowid)

    def list_classifier_tuning_runs(
        self,
        *,
        code_id: int | None = None,
        classifier_spec_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """List persisted heuristic classifier tuning runs."""
        clauses: list[str] = []
        parameters: list[Any] = []
        if code_id is not None:
            clauses.append("ctr.code_id = ?")
            parameters.append(int(code_id))
        if classifier_spec_id is not None:
            clauses.append("ctr.classifier_spec_id = ?")
            parameters.append(int(classifier_spec_id))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT ctr.*, cs.name AS classifier_name, g.name AS geometry_name
                FROM classifier_tuning_runs AS ctr
                JOIN classifier_specs AS cs
                  ON cs.classifier_spec_id = ctr.classifier_spec_id
                JOIN geometries AS g ON g.geometry_id = cs.geometry_id
                {where}
                ORDER BY ctr.tuning_run_id DESC
                """,
                parameters,
            ).fetchall()
        return [
            {
                "tuning_run_id": int(row["tuning_run_id"]),
                "code_id": int(row["code_id"]),
                "classifier_spec_id": int(row["classifier_spec_id"]),
                "classifier_name": str(row["classifier_name"]),
                "geometry_name": str(row["geometry_name"]),
                "metric": str(row["metric"]),
                "folds": int(row["folds"]),
                "candidate_lambdas": json.loads(str(row["candidate_lambdas_json"])),
                "results": json.loads(str(row["results_json"])),
                "selected_lambda": float(row["selected_lambda"]),
                "created_at": str(row["created_at"]),
            }
            for row in rows
        ]

    def register_apply_run(
        self,
        *,
        code_id: int,
        aggregation: str,
        threshold: float,
        source_kind: str,
        source_id: int,
        classifier_fit_ids: Sequence[int],
        committee_fit_id: int | None = None,
        proposals: Sequence[Mapping[str, Any]],
    ) -> int:
        """Persist one immutable draft proposal run and its exact component fits."""
        if aggregation not in COMMITTEE_AGGREGATIONS:
            raise ValueError("Unknown classifier-committee aggregation.")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must lie in the closed interval [0, 1]")
        if source_kind not in {"classifier", "committee"}:
            raise ValueError("source_kind must be 'classifier' or 'committee'")
        if not classifier_fit_ids:
            raise ValueError("At least one classifier fit is required.")
        if not proposals:
            raise ValueError("No eligible units remain for Apply proposals.")
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO apply_runs(
                    code_id, aggregation, source_kind, source_id, committee_fit_id, threshold,
                    eligibility, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'unreviewed', 'draft', ?)
                """,
                (
                    int(code_id),
                    aggregation,
                    source_kind,
                    int(source_id),
                    int(committee_fit_id) if committee_fit_id is not None else None,
                    float(threshold),
                    utc_now(),
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return an Apply run ID.")
            apply_run_id = int(cursor.lastrowid)
            connection.executemany(
                """
                INSERT INTO apply_run_fits(apply_run_id, classifier_fit_id)
                VALUES (?, ?)
                """,
                (
                    (apply_run_id, int(classifier_fit_id))
                    for classifier_fit_id in classifier_fit_ids
                ),
            )
            connection.executemany(
                """
                INSERT INTO apply_proposals(
                    apply_run_id, unit_id, probability, proposed_label
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    (
                        apply_run_id,
                        int(proposal["unit_id"]),
                        float(proposal["probability"]),
                        str(proposal["proposed_label"]),
                    )
                    for proposal in proposals
                ),
            )
            return apply_run_id

    def list_apply_runs(
        self,
        *,
        code_id: int | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """List Apply runs, newest first."""
        clauses: list[str] = []
        parameters: list[Any] = []
        if code_id is not None:
            clauses.append("ar.code_id = ?")
            parameters.append(int(code_id))
        if status is not None:
            clauses.append("ar.status = ?")
            parameters.append(str(status))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT ar.*, cv.name AS code_name,
                       SUM(CASE WHEN ap.removed_at IS NULL
                                      AND ap.committed_annotation_event_id IS NULL
                                THEN 1 ELSE 0 END) AS proposal_count,
                       SUM(CASE WHEN ap.removed_at IS NULL
                                      AND ap.committed_annotation_event_id IS NULL
                                      AND ap.decision != 'pending'
                                THEN 1 ELSE 0 END) AS reviewed_count,
                       SUM(CASE WHEN ap.removed_at IS NULL
                                      AND ap.committed_annotation_event_id IS NULL
                                      AND ap.decision = 'unreviewed'
                                THEN 1 ELSE 0 END) AS unreviewed_count
                FROM apply_runs AS ar
                JOIN codes AS c ON c.code_id = ar.code_id
                JOIN code_versions AS cv ON cv.code_version_id = (
                    SELECT cv2.code_version_id FROM code_versions AS cv2
                    WHERE cv2.code_id = c.code_id
                    ORDER BY cv2.version_number DESC LIMIT 1
                )
                LEFT JOIN apply_proposals AS ap ON ap.apply_run_id = ar.apply_run_id
                {where}
                GROUP BY ar.apply_run_id
                ORDER BY ar.apply_run_id DESC
                """,
                parameters,
            ).fetchall()
        return [_apply_run_row(row) for row in rows]

    def get_apply_run(self, apply_run_id: int) -> dict[str, Any]:
        """Return one Apply run with its component model IDs."""
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT ar.*, cv.name AS code_name,
                       SUM(CASE WHEN ap.removed_at IS NULL
                                      AND ap.committed_annotation_event_id IS NULL
                                THEN 1 ELSE 0 END) AS proposal_count,
                       SUM(CASE WHEN ap.removed_at IS NULL
                                      AND ap.committed_annotation_event_id IS NULL
                                      AND ap.decision != 'pending'
                                THEN 1 ELSE 0 END) AS reviewed_count,
                       SUM(CASE WHEN ap.removed_at IS NULL
                                      AND ap.committed_annotation_event_id IS NULL
                                      AND ap.decision = 'unreviewed'
                                THEN 1 ELSE 0 END) AS unreviewed_count
                FROM apply_runs AS ar
                JOIN codes AS c ON c.code_id = ar.code_id
                JOIN code_versions AS cv ON cv.code_version_id = (
                    SELECT cv2.code_version_id FROM code_versions AS cv2
                    WHERE cv2.code_id = c.code_id
                    ORDER BY cv2.version_number DESC LIMIT 1
                )
                LEFT JOIN apply_proposals AS ap ON ap.apply_run_id = ar.apply_run_id
                WHERE ar.apply_run_id = ?
                GROUP BY ar.apply_run_id
                """,
                (apply_run_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown apply_run_id: {apply_run_id}")
            fit_rows = connection.execute(
                """
                SELECT arf.classifier_fit_id, cf.geometry_id, cf.classifier_name
                FROM apply_run_fits AS arf
                JOIN classifier_fits AS cf
                  ON cf.classifier_fit_id = arf.classifier_fit_id
                WHERE arf.apply_run_id = ?
                ORDER BY arf.classifier_fit_id
                """,
                (apply_run_id,),
            ).fetchall()
        result = _apply_run_row(row)
        result["classifier_fit_ids"] = [
            int(item["classifier_fit_id"]) for item in fit_rows
        ]
        result["geometry_ids"] = [int(item["geometry_id"]) for item in fit_rows]
        result["classifier_names"] = [str(item["classifier_name"]) for item in fit_rows]
        return result

    def list_apply_proposals(
        self,
        apply_run_id: int,
        *,
        proposed_label: str | None = None,
        decision: str | None = None,
        include_unreviewed: bool = True,
        include_inactive: bool = False,
        probability_min: float | None = None,
        probability_max: float | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return proposal rows joined to their atomic text and metadata.

        Deliberately unreviewed proposals remain visible until a commit removes
        them from the active draft. Inactive rows remain available for audit.
        """
        clauses = ["ap.apply_run_id = ?"]
        parameters: list[Any] = [int(apply_run_id)]
        if not include_inactive:
            clauses.extend([
                "ap.removed_at IS NULL",
                "ap.committed_annotation_event_id IS NULL",
            ])
        if proposed_label is not None:
            clauses.append("ap.proposed_label = ?")
            parameters.append(str(proposed_label))
        if decision is not None:
            clauses.append("ap.decision = ?")
            parameters.append(str(decision))
        elif not include_unreviewed:
            clauses.append("ap.decision != 'unreviewed'")
        if probability_min is not None:
            if not 0.0 <= float(probability_min) <= 1.0:
                raise ValueError("probability_min must lie in [0, 1]")
            clauses.append("ap.probability >= ?")
            parameters.append(float(probability_min))
        if probability_max is not None:
            if not 0.0 <= float(probability_max) <= 1.0:
                raise ValueError("probability_max must lie in [0, 1]")
            clauses.append("ap.probability <= ?")
            parameters.append(float(probability_max))
        if (
            probability_min is not None
            and probability_max is not None
            and float(probability_min) > float(probability_max)
        ):
            raise ValueError("probability_min may not exceed probability_max")
        limit_sql = ""
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be positive")
            limit_sql = " LIMIT ? OFFSET ?"
            parameters.extend([int(limit), max(0, int(offset))])
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT ap.*, u.row_position, u.text, u.user_key_json, u.metadata_json
                FROM apply_proposals AS ap
                JOIN units AS u ON u.unit_id = ap.unit_id
                WHERE {' AND '.join(clauses)}
                ORDER BY ap.probability DESC, ap.unit_id
                {limit_sql}
                """,
                parameters,
            ).fetchall()
        return [
            {
                "apply_run_id": int(row["apply_run_id"]),
                "unit_id": int(row["unit_id"]),
                "row_position": int(row["row_position"]),
                "text": str(row["text"]),
                "user_key": json.loads(row["user_key_json"]),
                "metadata": json.loads(row["metadata_json"]),
                "probability": float(row["probability"]),
                "proposed_label": str(row["proposed_label"]),
                "decision": str(row["decision"]),
                "review_mode": row["review_mode"],
                "reviewed_at": row["reviewed_at"],
                "committed_annotation_event_id": (
                    int(row["committed_annotation_event_id"])
                    if row["committed_annotation_event_id"] is not None
                    else None
                ),
                "commit_operation_id": (
                    int(row["commit_operation_id"])
                    if row["commit_operation_id"] is not None
                    else None
                ),
                "removed_at": row["removed_at"],
            }
            for row in rows
        ]

    def set_apply_decision(
        self,
        *,
        apply_run_id: int,
        unit_id: int,
        decision: str,
        review_mode: str = "individual",
    ) -> None:
        """Set or clear the review decision for one draft proposal."""
        _validate_apply_decision(decision, review_mode)
        with self.connect() as connection:
            _require_draft_apply_run(connection, apply_run_id)
            cursor = connection.execute(
                """
                UPDATE apply_proposals
                SET decision = ?, review_mode = ?, reviewed_at = ?
                WHERE apply_run_id = ? AND unit_id = ?
                  AND removed_at IS NULL
                  AND committed_annotation_event_id IS NULL
                """,
                (
                    decision,
                    None if decision == "pending" else review_mode,
                    None if decision == "pending" else utc_now(),
                    apply_run_id,
                    unit_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(
                    f"Unknown Apply proposal ({apply_run_id}, {unit_id})."
                )

    def bulk_set_apply_decision(
        self,
        *,
        apply_run_id: int,
        decision: str,
        proposed_label: str | None = None,
        pending_only: bool = False,
        unit_ids: Iterable[int] | None = None,
    ) -> int:
        """Apply one bulk review decision to matching draft proposals."""
        _validate_apply_decision(decision, "bulk")
        clauses = [
            "apply_run_id = ?",
            "removed_at IS NULL",
            "committed_annotation_event_id IS NULL",
        ]
        parameters: list[Any] = [int(apply_run_id)]
        if proposed_label is not None:
            if proposed_label not in {"positive", "negative"}:
                raise ValueError("proposed_label must be 'positive' or 'negative'")
            clauses.append("proposed_label = ?")
            parameters.append(proposed_label)
        if pending_only:
            clauses.append("decision = 'pending'")
        selected_ids = None if unit_ids is None else sorted({int(value) for value in unit_ids})
        if selected_ids is not None:
            if not selected_ids:
                return 0
            placeholders = ",".join("?" for _ in selected_ids)
            clauses.append(f"unit_id IN ({placeholders})")
            parameters.extend(selected_ids)
        with self.connect() as connection:
            _require_draft_apply_run(connection, apply_run_id)
            cursor = connection.execute(
                f"""
                UPDATE apply_proposals
                SET decision = ?, review_mode = ?, reviewed_at = ?
                WHERE {' AND '.join(clauses)}
                """,
                [
                    decision,
                    None if decision == "pending" else "bulk",
                    None if decision == "pending" else utc_now(),
                    *parameters,
                ],
            )
            return int(cursor.rowcount)

    def commit_apply_run(
        self,
        apply_run_id: int,
        *,
        unit_ids: Iterable[int] | None = None,
        scope: str = "all_reviewed",
        finalize: bool = True,
    ) -> dict[str, int]:
        """Commit every selected non-pending proposal in one set-based transaction.

        Label decisions append assignment events. ``unreviewed`` decisions create no
        assignment. Every processed proposal leaves the active draft, while pending
        proposals remain available for later review. The run closes automatically only
        when no active proposals remain. ``finalize`` is retained for API compatibility
        but no longer closes a draft that still contains pending proposals.
        """
        del finalize
        if scope not in {"all_reviewed", "current_page", "explicit"}:
            raise ValueError("Unknown Apply commit scope.")
        selected_ids = (
            None
            if unit_ids is None
            else sorted({int(value) for value in unit_ids})
        )
        if selected_ids == []:
            return {
                "apply_run_id": int(apply_run_id),
                "processed": 0,
                "committed": 0,
                "skipped_conflicts": 0,
                "left_pending": 0,
                "remaining_reviewed": 0,
                "removed_unreviewed": 0,
                "commit_operation_id": 0,
            }

        with self.connect() as connection:
            run = _require_draft_apply_run(connection, apply_run_id)
            connection.execute(
                "CREATE TEMP TABLE IF NOT EXISTS selected_apply_commit_units "
                "(unit_id INTEGER PRIMARY KEY)"
            )
            connection.execute("DELETE FROM selected_apply_commit_units")
            if selected_ids is not None:
                connection.executemany(
                    "INSERT INTO selected_apply_commit_units(unit_id) VALUES (?)",
                    ((unit_id,) for unit_id in selected_ids),
                )
            scope_clause = (
                ""
                if selected_ids is None
                else "AND ap.unit_id IN (SELECT unit_id FROM selected_apply_commit_units)"
            )

            connection.execute("DROP TABLE IF EXISTS apply_commit_candidates")
            connection.execute(
                """
                CREATE TEMP TABLE apply_commit_candidates (
                    unit_id INTEGER PRIMARY KEY,
                    observation_id INTEGER NOT NULL,
                    value TEXT,
                    origin TEXT,
                    has_conflict INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                f"""
                INSERT INTO apply_commit_candidates(
                    unit_id, observation_id, value, origin, has_conflict
                )
                SELECT
                    ap.unit_id,
                    u.observation_id,
                    CASE
                        WHEN ap.decision = 'accept' THEN ap.proposed_label
                        WHEN ap.decision IN ('positive', 'negative', 'unsure')
                            THEN ap.decision
                        ELSE NULL
                    END AS value,
                    CASE
                        WHEN ap.decision = 'accept' AND ap.review_mode = 'bulk'
                            THEN 'model_committed'
                        WHEN ap.decision = 'accept'
                            THEN 'human_confirmed_model'
                        WHEN ap.decision IN ('positive', 'negative')
                             AND ap.decision = ap.proposed_label
                            THEN 'human_confirmed_model'
                        WHEN ap.decision IN ('positive', 'negative', 'unsure')
                            THEN 'human_overrode_model'
                        ELSE NULL
                    END AS origin,
                    CASE WHEN ca.annotation_event_id IS NULL THEN 0 ELSE 1 END
                FROM apply_proposals AS ap
                JOIN units AS u ON u.unit_id = ap.unit_id
                LEFT JOIN current_annotations AS ca
                  ON ca.unit_id = ap.unit_id
                 AND ca.code_id = ?
                WHERE ap.apply_run_id = ?
                  AND ap.removed_at IS NULL
                  AND ap.committed_annotation_event_id IS NULL
                  AND ap.decision != 'pending'
                  {scope_clause}
                """,
                (int(run["code_id"]), int(apply_run_id)),
            )

            counts = connection.execute(
                """
                SELECT
                    COUNT(*) AS processed,
                    SUM(CASE WHEN value IS NOT NULL AND has_conflict = 0 THEN 1 ELSE 0 END)
                        AS committed,
                    SUM(CASE WHEN value IS NOT NULL AND has_conflict = 1 THEN 1 ELSE 0 END)
                        AS skipped_conflicts,
                    SUM(CASE WHEN value IS NULL THEN 1 ELSE 0 END)
                        AS removed_unreviewed
                FROM apply_commit_candidates
                """
            ).fetchone()
            processed = int(counts["processed"] or 0)
            if processed == 0:
                remaining = connection.execute(
                    "SELECT COUNT(*) FROM apply_proposals WHERE apply_run_id = ? "
                    "AND removed_at IS NULL AND committed_annotation_event_id IS NULL "
                    "AND decision = 'pending'",
                    (int(apply_run_id),),
                ).fetchone()
                return {
                    "apply_run_id": int(apply_run_id),
                    "processed": 0,
                    "committed": 0,
                    "skipped_conflicts": 0,
                    "left_pending": int(remaining[0] or 0),
                    "remaining_reviewed": 0,
                    "removed_unreviewed": 0,
                    "commit_operation_id": 0,
                }
            operation_cursor = connection.execute(
                "INSERT INTO apply_commit_operations(apply_run_id, scope, created_at) "
                "VALUES (?, ?, ?)",
                (int(apply_run_id), str(scope), utc_now()),
            )
            if operation_cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return an Apply commit-operation ID.")
            commit_operation_id = int(operation_cursor.lastrowid)

            fit_rows = connection.execute(
                "SELECT classifier_fit_id FROM apply_run_fits WHERE apply_run_id = ?",
                (apply_run_id,),
            ).fetchall()
            classifier_fit_id = (
                int(fit_rows[0]["classifier_fit_id"]) if len(fit_rows) == 1 else None
            )
            committed_at = utc_now()
            connection.execute(
                """
                INSERT INTO annotation_events(
                    observation_id, code_id, value, origin, created_at, classifier_fit_id
                )
                SELECT observation_id, ?, value, origin, ?, ?
                FROM apply_commit_candidates
                WHERE value IS NOT NULL AND has_conflict = 0
                ORDER BY unit_id
                """,
                (int(run["code_id"]), committed_at, classifier_fit_id),
            )
            connection.execute(
                """
                UPDATE apply_proposals AS ap
                SET committed_annotation_event_id = (
                        SELECT ae.annotation_event_id
                        FROM annotation_events AS ae
                        JOIN units AS u ON u.observation_id = ae.observation_id
                        WHERE ae.code_id = ?
                          AND ae.created_at = ?
                          AND u.unit_id = ap.unit_id
                        ORDER BY ae.annotation_event_id DESC
                        LIMIT 1
                    ),
                    commit_operation_id = ?,
                    removed_at = ?
                WHERE ap.apply_run_id = ?
                  AND ap.unit_id IN (SELECT unit_id FROM apply_commit_candidates)
                  AND ap.removed_at IS NULL
                  AND ap.committed_annotation_event_id IS NULL
                """,
                (
                    int(run["code_id"]),
                    committed_at,
                    commit_operation_id,
                    committed_at,
                    int(apply_run_id),
                ),
            )

            remaining = connection.execute(
                """
                SELECT
                    COUNT(*) AS active_count,
                    SUM(CASE WHEN decision = 'pending' THEN 1 ELSE 0 END)
                        AS pending_count,
                    SUM(CASE WHEN decision != 'pending' THEN 1 ELSE 0 END)
                        AS reviewed_count
                FROM apply_proposals
                WHERE apply_run_id = ?
                  AND removed_at IS NULL
                  AND committed_annotation_event_id IS NULL
                """,
                (int(apply_run_id),),
            ).fetchone()
            active_count = int(remaining["active_count"] or 0)
            left_pending = int(remaining["pending_count"] or 0)
            remaining_reviewed = int(remaining["reviewed_count"] or 0)
            if active_count == 0:
                connection.execute(
                    """
                    UPDATE apply_runs
                    SET status = 'committed', committed_at = ?
                    WHERE apply_run_id = ?
                    """,
                    (committed_at, int(apply_run_id)),
                )

            return {
                "apply_run_id": int(apply_run_id),
                "processed": processed,
                "committed": int(counts["committed"] or 0),
                "skipped_conflicts": int(counts["skipped_conflicts"] or 0),
                "left_pending": left_pending,
                "remaining_reviewed": remaining_reviewed,
                "removed_unreviewed": int(counts["removed_unreviewed"] or 0),
                "commit_operation_id": commit_operation_id,
            }

    def latest_apply_commit_operation(self, apply_run_id: int) -> dict[str, Any] | None:
        """Return the newest non-undone commit operation for one Apply run."""
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM apply_commit_operations
                WHERE apply_run_id = ? AND undone_at IS NULL
                ORDER BY commit_operation_id DESC LIMIT 1
                """,
                (int(apply_run_id),),
            ).fetchone()
        if row is None:
            return None
        return {
            "commit_operation_id": int(row["commit_operation_id"]),
            "apply_run_id": int(row["apply_run_id"]),
            "scope": str(row["scope"]),
            "created_at": str(row["created_at"]),
            "undone_at": row["undone_at"],
        }

    def undo_last_apply_commit(self, apply_run_id: int) -> dict[str, int]:
        """Undo the newest commit without deleting provenance events.

        Assignment events created by that operation are retracted through a separate
        audit table. Later manual assignment events therefore remain current. Proposal
        rows from the operation return to the draft with their review decisions intact.
        """
        with self.connect() as connection:
            operation = connection.execute(
                """
                SELECT * FROM apply_commit_operations
                WHERE apply_run_id = ? AND undone_at IS NULL
                ORDER BY commit_operation_id DESC LIMIT 1
                """,
                (int(apply_run_id),),
            ).fetchone()
            if operation is None:
                raise ValueError("This draft has no commit operation to undo.")
            operation_id = int(operation["commit_operation_id"])
            event_rows = connection.execute(
                """
                SELECT committed_annotation_event_id
                FROM apply_proposals
                WHERE apply_run_id = ? AND commit_operation_id = ?
                  AND committed_annotation_event_id IS NOT NULL
                """,
                (int(apply_run_id), operation_id),
            ).fetchall()
            retracted_at = utc_now()
            connection.executemany(
                """
                INSERT OR IGNORE INTO annotation_event_retractions(
                    annotation_event_id, commit_operation_id, retracted_at
                ) VALUES (?, ?, ?)
                """,
                (
                    (int(row["committed_annotation_event_id"]), operation_id, retracted_at)
                    for row in event_rows
                ),
            )
            restored = connection.execute(
                """
                UPDATE apply_proposals
                SET committed_annotation_event_id = NULL,
                    commit_operation_id = NULL,
                    removed_at = NULL
                WHERE apply_run_id = ? AND commit_operation_id = ?
                """,
                (int(apply_run_id), operation_id),
            ).rowcount
            connection.execute(
                "UPDATE apply_commit_operations SET undone_at = ? WHERE commit_operation_id = ?",
                (retracted_at, operation_id),
            )
            connection.execute(
                "UPDATE apply_runs SET status = 'draft', committed_at = NULL "
                "WHERE apply_run_id = ?",
                (int(apply_run_id),),
            )
            return {
                "apply_run_id": int(apply_run_id),
                "commit_operation_id": operation_id,
                "restored_proposals": int(restored or 0),
                "retracted_assignments": len(event_rows),
            }

    def discard_apply_run(self, apply_run_id: int) -> None:
        """Close a draft without changing any annotations."""
        with self.connect() as connection:
            _require_draft_apply_run(connection, apply_run_id)
            connection.execute(
                """
                UPDATE apply_runs
                SET status = 'discarded', discarded_at = ?
                WHERE apply_run_id = ?
                """,
                (utc_now(), apply_run_id),
            )

    def create_memo(
        self,
        *,
        title: str,
        body_markdown: str,
        hashtags: Iterable[str] = (),
        unit_references: Iterable[int] = (),
    ) -> int:
        """Create a memo and its first immutable version."""
        now = utc_now()
        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO memos(title, body_markdown, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (title.strip(), body_markdown, now, now),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return a memo ID.")
            memo_id = int(cursor.lastrowid)
            self._insert_memo_version(
                connection, memo_id=memo_id, version_number=1, title=title,
                body_markdown=body_markdown, hashtags=hashtags, unit_references=unit_references,
                created_at=now,
            )
            self._sync_memo_snapshot(
                connection, memo_id=memo_id, title=title, body_markdown=body_markdown,
                hashtags=hashtags, unit_references=unit_references,
            )
            return memo_id

    def _insert_memo_version(
        self, connection: sqlite3.Connection, *, memo_id: int, version_number: int,
        title: str, body_markdown: str, hashtags: Iterable[str],
        unit_references: Iterable[int], created_at: str | None = None,
    ) -> int:
        cursor = connection.execute(
            """INSERT INTO memo_versions(
                memo_id, version_number, title, body_markdown, created_at
            ) VALUES (?, ?, ?, ?, ?)""",
            (memo_id, version_number, title.strip(), body_markdown, created_at or utc_now()),
        )
        if cursor.lastrowid is None:
            raise RuntimeError("SQLite did not return a memo version ID.")
        version_id = int(cursor.lastrowid)
        normalized_hashtags = sorted({tag.strip().lstrip("#") for tag in hashtags if tag.strip().lstrip("#")})
        normalized_references = list(dict.fromkeys(int(unit_id) for unit_id in unit_references))
        connection.executemany(
            "INSERT INTO memo_version_hashtags(memo_version_id, hashtag) VALUES (?, ?)",
            ((version_id, tag) for tag in normalized_hashtags),
        )
        connection.executemany(
            "INSERT INTO memo_version_unit_references(memo_version_id, unit_id, token) VALUES (?, ?, ?)",
            ((version_id, unit_id, f"[[unit:{unit_id}]]") for unit_id in normalized_references),
        )
        return version_id

    def _sync_memo_snapshot(
        self, connection: sqlite3.Connection, *, memo_id: int, title: str,
        body_markdown: str, hashtags: Iterable[str], unit_references: Iterable[int],
    ) -> None:
        """Maintain legacy latest-version tables without losing version history."""
        normalized_hashtags = sorted(
            {tag.strip().lstrip("#") for tag in hashtags if tag.strip().lstrip("#")}
        )
        normalized_references = list(
            dict.fromkeys(int(unit_id) for unit_id in unit_references)
        )
        connection.execute(
            "UPDATE memos SET title = ?, body_markdown = ?, updated_at = ? WHERE memo_id = ?",
            (title.strip(), body_markdown, utc_now(), memo_id),
        )
        connection.execute("DELETE FROM memo_hashtags WHERE memo_id = ?", (memo_id,))
        connection.executemany(
            "INSERT INTO memo_hashtags(memo_id, hashtag) VALUES (?, ?)",
            ((memo_id, tag) for tag in normalized_hashtags),
        )
        connection.execute("DELETE FROM memo_unit_references WHERE memo_id = ?", (memo_id,))
        connection.executemany(
            "INSERT INTO memo_unit_references(memo_id, unit_id, token) VALUES (?, ?, ?)",
            ((memo_id, unit_id, f"[[unit:{unit_id}]]") for unit_id in normalized_references),
        )

    def list_memos(self) -> list[dict[str, Any]]:
        """Return memos represented by their latest immutable version."""
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT m.memo_id, m.created_at, mv.memo_version_id, mv.version_number,
                       mv.title, mv.body_markdown, mv.created_at AS updated_at
                FROM memos m JOIN memo_versions mv ON mv.memo_version_id = (
                    SELECT mv2.memo_version_id FROM memo_versions mv2
                    WHERE mv2.memo_id = m.memo_id ORDER BY mv2.version_number DESC LIMIT 1
                )
                ORDER BY mv.created_at DESC, m.memo_id DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def memo_hashtags(self) -> list[str]:
        """Return hashtags from the latest version of every memo."""
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT mvh.hashtag FROM memos m
                JOIN memo_versions mv ON mv.memo_version_id = (
                    SELECT mv2.memo_version_id FROM memo_versions mv2
                    WHERE mv2.memo_id = m.memo_id ORDER BY mv2.version_number DESC LIMIT 1
                )
                JOIN memo_version_hashtags mvh ON mvh.memo_version_id = mv.memo_version_id
                ORDER BY mvh.hashtag COLLATE NOCASE
                """
            ).fetchall()
        return [str(row["hashtag"]) for row in rows]

    def search_memos(
        self, *, query: str = "", scope: str = "full", hashtags: Iterable[str] = (),
        hashtag_operator: str = "and",
    ) -> list[dict[str, Any]]:
        """Search latest memo versions by text and composable hashtag filters."""
        if scope not in {"full", "title"}:
            raise ValueError("scope must be 'full' or 'title'")
        if hashtag_operator not in {"and", "or", "xor"}:
            raise ValueError("hashtag_operator must be 'and', 'or', or 'xor'")
        needle = query.strip().casefold()
        selected = {str(tag).strip().lstrip("#").casefold() for tag in hashtags if str(tag).strip().lstrip("#")}
        results: list[dict[str, Any]] = []
        for summary in self.list_memos():
            memo = self.get_memo(int(summary["memo_id"]))
            haystack = str(memo["title"])
            if scope == "full":
                haystack += "\n" + str(memo["body_markdown"])
            if needle and needle not in haystack.casefold():
                continue
            memo_tags = {str(tag).casefold() for tag in memo["hashtags"]}
            if selected:
                overlap = len(selected & memo_tags)
                if hashtag_operator == "and" and not selected.issubset(memo_tags): continue
                if hashtag_operator == "or" and overlap == 0: continue
                if hashtag_operator == "xor" and overlap != 1: continue
            results.append(memo)
        return results

    def get_memo(self, memo_id: int, version_number: int | None = None) -> dict[str, Any]:
        """Return one memo version; latest when no version is requested."""
        query = """
            SELECT m.memo_id, m.created_at, mv.memo_version_id, mv.version_number,
                   mv.title, mv.body_markdown, mv.created_at AS updated_at
            FROM memos m JOIN memo_versions mv ON mv.memo_id = m.memo_id
            WHERE m.memo_id = ?
        """
        parameters: tuple[Any, ...] = (memo_id,)
        if version_number is not None:
            query += " AND mv.version_number = ?"
            parameters = (memo_id, int(version_number))
        query += " ORDER BY mv.version_number DESC LIMIT 1"
        with self.connect() as connection:
            row = connection.execute(query, parameters).fetchone()
            if row is None:
                raise KeyError(f"Unknown memo/version: {memo_id}/{version_number}")
            hashtags = [str(item["hashtag"]) for item in connection.execute(
                "SELECT hashtag FROM memo_version_hashtags WHERE memo_version_id = ? ORDER BY hashtag",
                (int(row["memo_version_id"]),),
            ).fetchall()]
            references = [{
                "reference_id": int(item["reference_id"]), "unit_id": int(item["unit_id"]),
                "token": str(item["token"]),
            } for item in connection.execute(
                "SELECT reference_id, unit_id, token FROM memo_version_unit_references WHERE memo_version_id = ? ORDER BY reference_id",
                (int(row["memo_version_id"]),),
            ).fetchall()]
        result = dict(row)
        result["hashtags"] = hashtags
        result["unit_references"] = references
        return result

    def memo_versions(self, memo_id: int) -> list[dict[str, Any]]:
        """Return every immutable version of one memo."""
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT version_number FROM memo_versions WHERE memo_id = ? ORDER BY version_number",
                (memo_id,),
            ).fetchall()
        if not rows:
            raise KeyError(f"Unknown memo_id: {memo_id}")
        return [self.get_memo(memo_id, int(row["version_number"])) for row in rows]

    def update_memo(
        self, memo_id: int, *, title: str, body_markdown: str,
        hashtags: Iterable[str] = (), unit_references: Iterable[int] = (),
    ) -> int:
        """Append a new memo version without overwriting earlier versions."""
        with self.connect() as connection:
            exists = connection.execute("SELECT 1 FROM memos WHERE memo_id = ?", (memo_id,)).fetchone()
            if exists is None:
                raise KeyError(f"Unknown memo_id: {memo_id}")
            version_number = int(connection.execute(
                "SELECT COALESCE(MAX(version_number), 0) + 1 FROM memo_versions WHERE memo_id = ?",
                (memo_id,),
            ).fetchone()[0])
            version_id = self._insert_memo_version(
                connection, memo_id=memo_id, version_number=version_number, title=title,
                body_markdown=body_markdown, hashtags=hashtags, unit_references=unit_references,
            )
            self._sync_memo_snapshot(
                connection, memo_id=memo_id, title=title, body_markdown=body_markdown,
                hashtags=hashtags, unit_references=unit_references,
            )
            return version_id

    def project_metadata(self) -> dict[str, Any]:
        """Read the single project metadata record."""
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM project WHERE project_id = 1").fetchone()
        if row is None:
            raise RuntimeError("Project metadata is missing.")
        return {
            "schema_version": row["schema_version"],
            "created_at": row["created_at"],
            "modality": row["modality"],
            "text_column": row["text_column"],
            "key_columns": json.loads(row["key_columns_json"]),
            "metadata_columns": json.loads(row["metadata_columns_json"]),
            "external_backed": (
                bool(row["external_backed"])
                if "external_backed" in row.keys()
                else False
            ),
        }



def _next_committee_deleted_name(
    connection: sqlite3.Connection,
    code_id: int,
    original_name: str,
    *,
    exclude_committee_id: int | None = None,
) -> str:
    """Return the first available archival committee name within one code."""
    index = 0
    while True:
        candidate = (
            f"{original_name}_deleted" if index == 0 else f"{original_name}_deleted_{index}"
        )
        query = "SELECT committee_id FROM classifier_committees WHERE code_id = ? AND name = ?"
        params: list[Any] = [int(code_id), candidate]
        if exclude_committee_id is not None:
            query += " AND committee_id != ?"
            params.append(int(exclude_committee_id))
        if connection.execute(query, params).fetchone() is None:
            return candidate
        index += 1


def _bump_archived_committee_name(
    connection: sqlite3.Connection,
    code_id: int,
    archived_name: str,
    *,
    exclude_committee_id: int | None = None,
) -> str:
    """Move a deleted committee-name collision farther into its archival namespace."""
    import re

    match = re.match(r"^(.*_deleted)(?:_(\d+))?$", archived_name)
    if match is None:
        return _next_committee_deleted_name(
            connection, code_id, archived_name, exclude_committee_id=exclude_committee_id
        )
    prefix = str(match.group(1))
    index = int(match.group(2) or 0) + 1
    while True:
        candidate = f"{prefix}_{index}"
        query = "SELECT committee_id FROM classifier_committees WHERE code_id = ? AND name = ?"
        params: list[Any] = [int(code_id), candidate]
        if exclude_committee_id is not None:
            query += " AND committee_id != ?"
            params.append(int(exclude_committee_id))
        if connection.execute(query, params).fetchone() is None:
            return candidate
        index += 1


def _assert_active_predictor_name_available(
    connection: sqlite3.Connection,
    requested_name: str,
    *,
    exclude_classifier_spec_id: int | None = None,
    exclude_committee_id: int | None = None,
) -> None:
    """Require one project-wide active namespace across classifiers and committees."""
    classifier_query = (
        "SELECT classifier_spec_id FROM classifier_specs "
        "WHERE status = 'active' AND name = ?"
    )
    classifier_params: list[Any] = [requested_name]
    if exclude_classifier_spec_id is not None:
        classifier_query += " AND classifier_spec_id != ?"
        classifier_params.append(int(exclude_classifier_spec_id))
    classifier = connection.execute(classifier_query, classifier_params).fetchone()
    if classifier is not None:
        raise ValueError(
            f"An active predictor named {requested_name!r} already exists (classifier)."
        )

    committee_query = (
        "SELECT committee_id FROM classifier_committees "
        "WHERE status = 'active' AND name = ?"
    )
    committee_params: list[Any] = [requested_name]
    if exclude_committee_id is not None:
        committee_query += " AND committee_id != ?"
        committee_params.append(int(exclude_committee_id))
    committee = connection.execute(committee_query, committee_params).fetchone()
    if committee is not None:
        raise ValueError(
            f"An active predictor named {requested_name!r} already exists (committee)."
        )


def _free_committee_name(
    connection: sqlite3.Connection,
    code_id: int,
    requested_name: str,
    *,
    exclude_committee_id: int | None = None,
) -> None:
    """Free a committee name while enforcing the shared active predictor namespace."""
    _assert_active_predictor_name_available(
        connection,
        requested_name,
        exclude_committee_id=exclude_committee_id,
    )

    # The table's physical uniqueness constraint is code-local and includes deleted
    # rows, so a same-code archival collision must be moved out of the way even
    # though archival names do not participate in the active predictor namespace.
    query = "SELECT committee_id, status FROM classifier_committees WHERE code_id = ? AND name = ?"
    params: list[Any] = [int(code_id), requested_name]
    if exclude_committee_id is not None:
        query += " AND committee_id != ?"
        params.append(int(exclude_committee_id))
    row = connection.execute(query, params).fetchone()
    if row is None:
        return
    if str(row["status"]) == "active":
        raise RuntimeError(
            "Active committee name collision escaped shared predictor validation."
        )
    archived_id = int(row["committee_id"])
    replacement = _bump_archived_committee_name(
        connection, int(code_id), requested_name, exclude_committee_id=archived_id
    )
    connection.execute(
        "UPDATE classifier_committees SET name = ? WHERE committee_id = ?",
        (replacement, archived_id),
    )


def _next_classifier_deleted_name(
    connection: sqlite3.Connection,
    original_name: str,
    *,
    exclude_classifier_spec_id: int | None = None,
) -> str:
    """Return the first available archival name for a newly deleted classifier."""
    index = 0
    while True:
        candidate = (
            f"{original_name}_deleted" if index == 0 else f"{original_name}_deleted_{index}"
        )
        query = "SELECT classifier_spec_id FROM classifier_specs WHERE name = ?"
        params: list[Any] = [candidate]
        if exclude_classifier_spec_id is not None:
            query += " AND classifier_spec_id != ?"
            params.append(int(exclude_classifier_spec_id))
        if connection.execute(query, params).fetchone() is None:
            return candidate
        index += 1


def _bump_archived_classifier_name(
    connection: sqlite3.Connection,
    archived_name: str,
    *,
    exclude_classifier_spec_id: int | None = None,
) -> str:
    """Move a deleted-name collision farther into the archival namespace."""
    import re

    match = re.match(r"^(.*_deleted)(?:_(\d+))?$", archived_name)
    if match is None:
        return _next_classifier_deleted_name(
            connection, archived_name,
            exclude_classifier_spec_id=exclude_classifier_spec_id,
        )
    prefix = str(match.group(1))
    index = int(match.group(2) or 0) + 1
    while True:
        candidate = f"{prefix}_{index}"
        query = "SELECT classifier_spec_id FROM classifier_specs WHERE name = ?"
        params: list[Any] = [candidate]
        if exclude_classifier_spec_id is not None:
            query += " AND classifier_spec_id != ?"
            params.append(int(exclude_classifier_spec_id))
        if connection.execute(query, params).fetchone() is None:
            return candidate
        index += 1


def _free_classifier_name(
    connection: sqlite3.Connection,
    requested_name: str,
    *,
    exclude_classifier_spec_id: int | None = None,
) -> None:
    """Free a classifier name while enforcing the shared active predictor namespace."""
    _assert_active_predictor_name_available(
        connection,
        requested_name,
        exclude_classifier_spec_id=exclude_classifier_spec_id,
    )

    # classifier_specs has a table-wide UNIQUE(name) constraint that includes deleted
    # rows, so an archival collision must be bumped before the active name can be used.
    query = "SELECT classifier_spec_id, status FROM classifier_specs WHERE name = ?"
    params: list[Any] = [requested_name]
    if exclude_classifier_spec_id is not None:
        query += " AND classifier_spec_id != ?"
        params.append(int(exclude_classifier_spec_id))
    row = connection.execute(query, params).fetchone()
    if row is None:
        return
    if str(row["status"]) == "active":
        raise RuntimeError(
            "Active classifier name collision escaped shared predictor validation."
        )
    archived_id = int(row["classifier_spec_id"])
    replacement = _bump_archived_classifier_name(
        connection, requested_name, exclude_classifier_spec_id=archived_id
    )
    connection.execute(
        "UPDATE classifier_specs SET name = ? WHERE classifier_spec_id = ?",
        (replacement, archived_id),
    )


def _extract_hashtags(markdown: str) -> list[str]:
    import re
    return list(dict.fromkeys(re.findall(r"(?<!\w)#([A-Za-z0-9_-]+)", markdown)))

def _code_version_row(row: sqlite3.Row) -> dict[str, Any]:
    description = str(row["description_markdown"])
    return {
        "code_id": int(row["code_id"]),
        "code_version_id": int(row["code_version_id"]),
        "version_number": int(row["version_number"]),
        "name": str(row["name"]),
        "description": description,
        "definition": description,
        "created_at": str(row["created_at"]),
        "version_created_at": str(row["version_created_at"]),
    }

def _require_draft_apply_run(
    connection: sqlite3.Connection,
    apply_run_id: int,
) -> sqlite3.Row:
    row = connection.execute(
        "SELECT * FROM apply_runs WHERE apply_run_id = ?",
        (apply_run_id,),
    ).fetchone()
    if row is None:
        raise KeyError(f"Unknown apply_run_id: {apply_run_id}")
    if row["status"] != "draft":
        raise ValueError(
            f"Apply run {apply_run_id} is {row['status']!r} and can no longer be edited."
        )
    return row


def _validate_apply_decision(decision: str, review_mode: str) -> None:
    valid = {"pending", "accept", "positive", "negative", "unsure", "unreviewed"}
    if decision not in valid:
        raise ValueError(f"Unknown Apply decision: {decision!r}")
    if review_mode not in {"individual", "bulk"}:
        raise ValueError("review_mode must be 'individual' or 'bulk'")


def _apply_run_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "apply_run_id": int(row["apply_run_id"]),
        "code_id": int(row["code_id"]),
        "code_name": str(row["code_name"]),
        "aggregation": str(row["aggregation"]),
        "source_kind": str(row["source_kind"]),
        "source_id": int(row["source_id"]),
        "committee_fit_id": (
            int(row["committee_fit_id"]) if row["committee_fit_id"] is not None else None
        ),
        "threshold": float(row["threshold"]),
        "eligibility": str(row["eligibility"]),
        "status": str(row["status"]),
        "created_at": str(row["created_at"]),
        "committed_at": row["committed_at"],
        "discarded_at": row["discarded_at"],
        "proposal_count": int(row["proposal_count"] or 0),
        "reviewed_count": int(row["reviewed_count"] or 0),
        "unreviewed_count": int(row["unreviewed_count"] or 0),
    }


def _classifier_spec_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "classifier_spec_id": int(row["classifier_spec_id"]),
        "code_id": int(row["code_id"]),
        "code_name": str(row["code_name"]) if "code_name" in row.keys() and row["code_name"] is not None else None,
        "name": str(row["name"]),
        "geometry_id": int(row["geometry_id"]),
        "geometry_name": str(row["geometry_name"]),
        "algorithm": str(row["algorithm"]),
        "hyperparameters": json.loads(row["hyperparameters_json"]),
        "status": str(row["status"]),
        "created_at": str(row["created_at"]),
    }


def _classifier_fit_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "classifier_fit_id": int(row["classifier_fit_id"]),
        "classifier_spec_id": int(row["classifier_spec_id"]),
        "classifier_name": str(row["classifier_name"]),
        "code_id": int(row["code_id"]),
        "geometry_id": int(row["geometry_id"]),
        "geometry_name": str(row["geometry_name"]),
        "algorithm": str(row["algorithm"]),
        "hyperparameters": json.loads(row["hyperparameters_json"]),
        "training_snapshot": json.loads(row["training_snapshot_json"]),
        "artifact_path": str(row["artifact_path"]),
        "score_kind": str(row["score_kind"]),
        "created_at": str(row["created_at"]),
    }


def _classifier_prediction_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "classifier_fit_id": int(row["classifier_fit_id"]),
        "observation_id": int(row["observation_id"]),
        "kind": str(row["kind"]),
        "text": str(row["text"]),
        "unit_id": int(row["unit_id"]) if row["unit_id"] is not None else None,
        "row_position": (
            int(row["row_position"]) if row["row_position"] is not None else None
        ),
        "predicted_label": str(row["predicted_label"]),
        "probability": (
            float(row["probability"]) if row["probability"] is not None else None
        ),
        "decision_score": (
            float(row["decision_score"])
            if row["decision_score"] is not None
            else None
        ),
        "uncertainty": float(row["uncertainty"]),
    }


def _classifier_committee_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "committee_id": int(row["committee_id"]),
        "code_id": int(row["code_id"]),
        "name": str(row["name"]),
        "aggregation": str(row["aggregation"]),
        "status": str(row["status"]),
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
    }


def _classifier_committee_fit_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "committee_fit_id": int(row["committee_fit_id"]),
        "committee_id": int(row["committee_id"]),
        "code_id": int(row["code_id"]),
        "aggregation": str(row["aggregation"]),
        "member_fit_ids": json.loads(str(row["member_fit_ids_json"])),
        "training_snapshot": json.loads(str(row["training_snapshot_json"])),
        "artifact_path": str(row["artifact_path"]),
        "created_at": str(row["created_at"]),
    }


def _classifier_evaluation_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "evaluation_event_id": int(row["evaluation_event_id"]),
        "event_type": str(row["event_type"]),
        "code_id": int(row["code_id"]),
        "classifier_fit_id": int(row["classifier_fit_id"]),
        "classifier_name": str(row["classifier_name"]),
        "geometry_name": str(row["geometry_name"]),
        "observation_id": (
            int(row["observation_id"]) if row["observation_id"] is not None else None
        ),
        "human_label": str(row["human_label"]),
        "predicted_label": str(row["predicted_label"]),
        "probability": (
            float(row["probability"]) if row["probability"] is not None else None
        ),
        "decision_score": (
            float(row["decision_score"])
            if row["decision_score"] is not None
            else None
        ),
        "added_to_training": bool(row["added_to_training"]),
        "created_at": str(row["created_at"]),
    }


def _geometry_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "geometry_id": int(row["geometry_id"]),
        "name": str(row["name"]),
        "class_name": str(row["class_name"]),
        "modality": str(row["modality"]),
        "public": bool(row["public"]),
        "config": json.loads(row["config_json"]),
        "dependency_names": json.loads(row["dependency_names_json"]),
        "matrix_path": str(row["matrix_path"]),
        "matrix_sparse": bool(row["matrix_sparse"]),
        "state_path": row["state_path"],
        "supports_query": bool(row["supports_query"]),
        "supports_text_transform": bool(row["supports_text_transform"]),
        "storage_kind": str(row["storage_kind"]),
        "external_ref": (json.loads(row["external_ref_json"]) if row["external_ref_json"] is not None else None),
        "created_at": str(row["created_at"]),
    }


def _view_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "view_id": int(row["view_id"]),
        "geometry_id": int(row["geometry_id"]),
        "name": str(row["name"]),
        "method": str(row["method"]),
        "parameters": json.loads(row["parameters_json"]),
        "artifact_path": str(row["artifact_path"]),
        "state_path": row["state_path"],
        "supports_transform": bool(row["supports_transform"]),
        "storage_kind": str(row["storage_kind"]),
        "external_ref": (json.loads(row["external_ref_json"]) if row["external_ref_json"] is not None else None),
        "created_at": str(row["created_at"]),
    }


def _unit_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "unit_id": int(row["unit_id"]),
        "observation_id": int(row["observation_id"]),
        "kind": "atomic",
        "row_position": int(row["row_position"]),
        "text": row["text"],
        "user_key": json.loads(row["user_key_json"]),
        "metadata": json.loads(row["metadata_json"]),
    }


def _observation_row(row: sqlite3.Row) -> dict[str, Any]:
    user_key = json.loads(row["user_key_json"]) if "user_key_json" in row.keys() and row["user_key_json"] is not None else {}
    metadata = json.loads(row["metadata_json"]) if "metadata_json" in row.keys() and row["metadata_json"] is not None else {}
    return {
        "observation_id": int(row["observation_id"]),
        "kind": str(row["kind"]),
        "text": str(row["text"]),
        "created_at": str(row["created_at"]),
        "unit_id": int(row["unit_id"]) if "unit_id" in row.keys() and row["unit_id"] is not None else None,
        "row_position": int(row["row_position"]) if "row_position" in row.keys() and row["row_position"] is not None else None,
        "user_key": user_key,
        "metadata": metadata,
        "teaching_code_id": int(row["teaching_code_id"]) if "teaching_code_id" in row.keys() and row["teaching_code_id"] is not None else None,
        "teaching_label": str(row["teaching_label"]) if "teaching_label" in row.keys() and row["teaching_label"] is not None else None,
        "teaching_status": str(row["teaching_status"]) if "teaching_status" in row.keys() and row["teaching_status"] is not None else None,
        "teaching_note": str(row["teaching_note"]) if "teaching_note" in row.keys() and row["teaching_note"] is not None else "",
    }


def _find_span_id(connection: sqlite3.Connection, members: Sequence[int]) -> int | None:
    candidates = connection.execute(
        """SELECT span_observation_id
           FROM span_members
           GROUP BY span_observation_id
           HAVING COUNT(*) = ?
           ORDER BY span_observation_id""",
        (len(members),),
    ).fetchall()
    expected = [int(value) for value in members]
    for candidate in candidates:
        observation_id = int(candidate["span_observation_id"])
        rows = connection.execute(
            """SELECT unit_id FROM span_members
               WHERE span_observation_id = ? ORDER BY member_order""",
            (observation_id,),
        ).fetchall()
        if [int(row["unit_id"]) for row in rows] == expected:
            return observation_id
    return None


def _validate_contiguous_span(
    connection: sqlite3.Connection, members: Sequence[int]
) -> None:
    rows = connection.execute(
        f"SELECT unit_id, row_position FROM units WHERE unit_id IN ({','.join('?' for _ in members)}) "
        "ORDER BY row_position",
        tuple(members),
    ).fetchall()
    ordered = [int(row["unit_id"]) for row in rows]
    if ordered != list(members):
        raise ValueError("Span members must follow canonical row order.")
    positions = [int(row["row_position"]) for row in rows]
    if any(right != left + 1 for left, right in zip(positions, positions[1:], strict=False)):
        raise ValueError("Span members must be contiguous atomic observations.")
    atomic_level_row = connection.execute(
        "SELECT MAX(level_index) AS atomic_level FROM hierarchy_levels"
    ).fetchone()
    atomic_level = int(atomic_level_row["atomic_level"])
    if atomic_level > 0:
        parent_level = atomic_level - 1
        parent_rows = connection.execute(
            f"SELECT DISTINCT group_id FROM unit_groups WHERE level_index = ? "
            f"AND unit_id IN ({','.join('?' for _ in members)})",
            (parent_level, *members),
        ).fetchall()
        if len(parent_rows) != 1:
            raise ValueError(
                "Span members must belong to the same immediate parent group."
            )


def _json_value(value: Any) -> Any:
    """Convert common pandas/numpy values into JSON-compatible values."""
    if value is None:
        return None
    try:
        missing = pd.isna(value)
        if isinstance(missing, bool) and missing:
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
