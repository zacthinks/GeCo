"""Release metadata and importable package version must stay synchronized."""

from pathlib import Path
import tomllib

from geometric_coder import __version__


def test_runtime_version_matches_project_metadata():
    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as handle:
        metadata = tomllib.load(handle)
    assert metadata["project"]["version"] == __version__
