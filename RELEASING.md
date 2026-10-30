# Releasing GeCo

GeCo is in pre-1.0 development. GitHub releases are deliberately tagged and initiated by a maintainer; package releases are **not** automatically published to PyPI.

1. Merge the release changes and any intended product changes into the protected default branch. Verify CI on that commit.
2. Decide the next version using the pre-1.0 versioning policy and update `[project].version` in `pyproject.toml`, plus `CHANGELOG.md`. Open/merge a version-bump PR if necessary.
3. Check that the intended tag does not already exist. Create an annotated tag `vX.Y.Z` at the tested default-branch commit and push it.
4. The `Release` workflow checks the tag against `pyproject.toml`, installs dependencies, runs unit and acceptance tests, builds wheel and sdist, verifies a clean wheel installation, and attaches distributions to a GitHub Release.
5. Verify the tag, associated commit, uploaded assets and release page. Check the CLI through a clean install before announcing.
6. If CI fails, fix the problem through a new PR; do not silently move or replace an already published version tag.

The optional `master` → `main` default-branch rename is a coordinated GeCo/TeAL operation, separate from the release process. The release workflow triggers on version tags and does not depend on the branch name.
