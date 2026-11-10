"""Large hierarchical context must never decode an entire parent group."""

import pandas as pd

from geometric_coder import GeometricCoder


def test_bounded_context_never_reads_entire_hierarchy_group(tmp_path):
    count = 300
    project = GeometricCoder.create(
        project_dir=tmp_path / "context.geco",
        data=pd.DataFrame({
            "doc": ["same"] * count,
            "sentence": list(range(count)),
            "text": [f"Sentence {i}" for i in range(count)],
        }),
        keys=["doc", "sentence"], text="text", metadata=[],
        geometries=None,
    )
    focal = 153
    unbounded = project.context(focal, level=0, window=1)
    bounded = project.context(focal, level=0, window=1, max_units=19)
    assert len(unbounded) == count
    assert len(bounded) == 19
    assert focal in [int(row["unit_id"]) for row in bounded]
    assert [row["row_position"] for row in bounded] == list(range(143, 162))
    # The lower-level sentence group also stays bounded.
    leaf = project.context(focal, level=1, window=1, max_units=19)
    assert len(leaf) == 1


def test_context_bounded_invalid_limit(tmp_path):
    project = GeometricCoder.create(
        project_dir=tmp_path / "small.geco",
        data=pd.DataFrame({"doc": ["a"], "sentence": [0], "text": ["a"]}),
        keys=["doc", "sentence"], text="text", metadata=[],
        geometries=None,
    )
    import pytest
    with pytest.raises(ValueError, match="max_units"):
        project.context(1, level=0, max_units=0)
