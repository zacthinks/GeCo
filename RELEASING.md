# Releasing GeCo

GeCo is in pre-1.0 development. GitHub releases are deliberately tagged and initiated by a maintainer; package releases are **not** automatically published to PyPI.

1. Select the new pre-1.0 version and prepare a PR from the current default branch. Keep the version in pyproject.toml and src/geometric_coder/_version.py synchronized; add the corresponding dated CHANGELOG.md entry and update the user-facing README and architecture documentation. The version bump may be part of the final feature PR.
2. Verify all required CI checks at the **final PR head** after the documentation/version commit; then merge into the protected default branch.
3. Verify CI on the **merged default-branch commit** and confirm the intended tag does not already exist. Create an annotated tag vX.Y.Z at that merged, tested commit and push it. Do not tag the pre-merge PR head.
4. The tag-triggered Release workflow validates the tag against pyproject.toml, installs dependencies, runs unit and acceptance tests, builds wheel and sdist, verifies a clean wheel installation, and publishes a GitHub Release with those distributions.
5. Verify the tag target, matching package/runtime versions, GitHub Actions result, uploaded wheel/sdist assets, and release page. Check the CLI through a clean wheel install before announcing.
6. If release CI fails, fix the problem in a new PR; never silently move or replace a published tag. Update the changelog and project documentation as appropriate.
