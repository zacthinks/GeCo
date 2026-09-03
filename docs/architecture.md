# GeCo architecture

This document gives a compact public overview of the current GeCo architecture. GeCo 0.7.0 uses project schema 12.

## Project model

`GeometricCoder` is the main project object used by Python code and the local Dash interface. Structured state is stored in SQLite; large numerical and model artifacts are stored as registered sidecars.

```text
project.geco/
├── project.sqlite3
└── artifacts/
    ├── geometries/
    ├── views/
    ├── observations/
    └── models/
```

Runtime opening requires an exact project-schema match. Schema migrations are explicit standalone scripts rather than automatic application behavior.

## Documents, keys, and observations

Project creation accepts a pandas DataFrame with one text column, one or more key columns, and optional metadata columns. Input row order is canonical and original key values are preserved.

Each imported row is an **atomic observation**. GeCo also supports two derived observation kinds:

- **span** — a contiguous interval of atomic observations within the same immediate context group;
- **teaching example** — researcher-authored text attached to a code as positive or negative training evidence.

`unit_id` identifies imported atomic rows for canonical corpus order and navigation. `observation_id` is the general identity used by coding and representation access across atomic and derived observations.

## Human assignments

Human coding is represented as append-only assignment events with three substantive values:

```text
positive
negative
unsure
```

The latest assignment for an observation/code pair is the current judgment. No assignment means the observation has not been reviewed for that code. Model predictions and proposals are stored separately and never silently become human assignments.

## Geometries and views

A **geometry** maps observation text into a numerical space. Geometries may be local or externally backed.

For local geometries, GeCo stores the fitted transformation and the atomic corpus matrix. Transform-capable geometries can place later spans and teaching examples into the same frozen space without refitting the corpus representation.

A **view** is a two-dimensional display associated with a parent geometry. Views are used for exploration and are not treated as analytic geometries unless a researcher explicitly constructs a geometry from them.

The stable access boundaries are:

```python
project.geometry_matrix(...)
project.view_coordinates(...)
```

External integrations implement these through a runtime provider rather than special-casing another application's storage system in the UI.

## Sessions, filters, and memos

Explore sessions persist navigation and resumable search/filter state. An atomic unit counts as visited only when it becomes focal; appearing on a map or in context does not itself count as a visit.

Memos are versioned Markdown documents with evidence references that can reopen the corresponding observation in the interface. Presentation mode provides a read-only analytic narrative alongside a minimal map and context reader without changing Explore visit history.

## Codes and classifiers

A code may own multiple classifiers. Every active classifier:

- belongs to exactly one code;
- has a project-wide unique active name;
- records its geometry, algorithm, and hyperparameters;
- has at most one retained heavy fitted estimator and current prediction cache;
- is explicitly Current, Stale, or Not trained.

Retraining creates and validates the replacement fit before switching live state. Superseded heavy estimator and prediction artifacts are then removed, while lightweight historical metadata needed to interpret prior Apply/evaluation events remains.

Classifier training is explicit. Ordinary coding or navigation does not silently retrain models.

## Apply and Review

Apply drafts persist machine proposals separately from human assignments. Researchers can accept a proposal, override it with Present/Absent/Unsure, or leave it unreviewed. Only committed decisions create assignment events. Bulk operations preserve their commit provenance and can be undone at the commit-operation level without erasing later revisions.

## External numerical resources

Externally backed projects preserve the normal GeCo document/text/metadata snapshot in SQLite while obtaining geometry matrices and view coordinates from a runtime provider. External references are opaque JSON values; GeCo persists and round-trips them without interpretation.

The provider receives the project's original user keys in canonical atomic-row order and is responsible for returning correctly aligned rows. This keeps GeCo independent of external catalogs, lineage systems, query engines, and physical storage formats.

See [`teal-integration.md`](teal-integration.md) for the public provider protocol.

## Incremental ingestion

Standalone projects can append new atomic rows through `GeometricCoder.ingest(...)` when their local fitted geometries/views can transform new text without refitting existing representations.

Externally backed projects have a fixed document universe in GeCo 0.7.0; ingestion is disabled so the GeCo workspace cannot diverge from the external key universe.
