"""Phase 2 regression checks for batched latest-version searches."""

from pathlib import Path

import pandas as pd

from geometric_coder import GeometricCoder


def _project(tmp_path: Path) -> GeometricCoder:
    project = GeometricCoder.create(
        project_dir=tmp_path / "batched-search.geco",
        data=pd.DataFrame({"doc": [1, 2, 3], "text": ["alpha", "beta", "gamma"]}),
        keys=["doc"],
        text="text",
    )
    return project


def test_memo_search_matches_single_version_retrieval(tmp_path: Path, monkeypatch) -> None:
    project = _project(tmp_path)
    first = project.create_memo(
        title="First", body_markdown="old body",
        hashtags=["old"], unit_references=[1],
    )
    project.update_memo(
        first, title="First revised", body_markdown="new body",
        hashtags=["shared", "primary"], unit_references=[2, 3],
    )
    second = project.create_memo(
        title="Second", body_markdown="newer body",
        hashtags=["shared"], unit_references=[1],
    )
    expected = [project.memo(int(row["memo_id"])) for row in project.memos()]
    assert project.search_memos() == expected
    assert [row["memo_id"] for row in project.search_memos(
        hashtags=["shared", "primary"], hashtag_operator="and"
    )] == [first]
    assert {row["memo_id"] for row in project.search_memos(
        hashtags=["shared", "primary"], hashtag_operator="xor"
    )} == {second}
    assert [row["memo_id"] for row in project.search_memos(
        query="revised", scope="title"
    )] == [first]
    assert project.search_memos(query="old body") == []
    assert len(project.memo_versions(first)) == 2

    def reject_get_memo(*args, **kwargs):
        raise AssertionError("search must not fetch each memo separately")

    monkeypatch.setattr(project.database, "get_memo", reject_get_memo)
    assert len(project.search_memos()) == 2


def test_code_search_matches_latest_versions_without_n_plus_one(tmp_path: Path, monkeypatch) -> None:
    project = _project(tmp_path)
    first = project.create_code("First", "old description")
    second = project.create_code("Second", "other description")
    project.update_code(first, name="First revised", description="new description")
    expected = [project.code(int(row["code_id"])) for row in project.codes()]
    assert project.search_codes() == expected
    assert [row["code_id"] for row in project.search_codes(
        query="revised", scope="label"
    )] == [first]
    assert project.search_codes(query="old description") == []
    assert len(project.code_versions(first)) == 2

    def reject_get_code(*args, **kwargs):
        raise AssertionError("search must not fetch each code separately")

    monkeypatch.setattr(project.database, "get_code", reject_get_code)
    assert {row["code_id"] for row in project.search_codes()} == {first, second}
