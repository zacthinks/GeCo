# Scale Develop UI — researcher acceptance walkthrough

This view is an exploratory tool. Predicted Scale positions are signed,
reference-centered Davidson utilities, **not** probabilities or binary Code
assignments. Positional epistemic uncertainty is a model-conditional posterior
standard deviation, **not** total measurement error or evidence of construct
validity.

## Open an isolated real-project copy

Do **not** run development builds against an irreplaceable research project.
Copy your `.geco` directory before testing; schema upgrades and new judgments
modify the opened project.

From an installed GeCo v0.14.0 release candidate or its reviewed source checkout:

```bash
uv sync --dev
uv run geco launch /path/to/your-project-copy.geco
```

Open the localhost URL printed by `geco launch`, then select **Develop**.

## Walkthrough

1. Select a **Scale** Construct, or create one through **New Construct**.
   When using Code-to-Scale bootstrap, confirm the explicit Present > Absent
   assumption and check the full comparison-count preview.
2. With no Scale fit, inspect **Scale prediction positions** (shown below the comparisons, not a collapsible panel).
   The view should explain that training is needed; it must not fabricate a
   plot of learned scores.
3. Check that the default comparison interface presents observations A and B
   around four centrally positioned judgments. Use A higher, parity, B
   higher, and unsure/skip; verify saved history. **Next pair** must advance
   without creating a judgment.
4. Select an analytic geometry and **Train** after enough directional evidence.
   In **Scale prediction positions**: x-axis is signed, centered model
   position; y-axis is posterior positional standard deviation where available.
   If the posterior is unavailable, the y-axis clearly says *page order
   (not uncertainty)*, retaining clickable observations.
5. Change the score page, click a plotted observation, and verify Explore
   focal-navigation history and source key. Only visible eligible atomic
   observations should be scored on each page.
6. Turn automatic training off, save a comparison, and verify that the model
   is marked stale while the previous fit and recommendation order remain
   available. Refresh/inspect scores without retraining. Then explicitly
   Train to update the fit.
7. Switch to a binary Code and back. The Code plot and classifier interfaces
   must remain unchanged; stale Scale plot data must not appear under a Code.
8. Reopen the project and verify session restoration, comparison history,
   manual corrections, Scale fit status and EIG/random acquisition notices.

For large or linked geometries, also verify that plot paging stays responsive
without source-text or feature-matrix materialization across the whole corpus.

## Automated checks

```bash
uv run pytest -q tests/test_scale_prediction_ui.py tests/test_scale_comparison_ui.py
uv run pytest -q
```

**Before merging:** exact-head CI, visual browser inspection, and hands-on
acceptance on an actual project copy. Do not merge to `scaling` without
researcher approval, and do not merge `scaling` into `main` yet.
