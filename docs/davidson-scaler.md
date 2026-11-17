# Linear Davidson numerical scaler (Phase 2A)

`geometric_coder.davidson_scaler.LinearDavidsonScaler` is a standalone, database-agnostic numerical component. Its inputs and outputs are NumPy arrays or SciPy CSR matrices, **not** GeCo identifiers. All arrays use float64 computation.

## Interface for integration

```python
model = LinearDavidsonScaler(
    feature_normalization="none",  # or "rms"
    score_prior_sd=1.0,
    eta_prior=np.log(0.5),
    eta_prior_sd=1.0,
    posterior_method="auto",  # exact "primal" or "dual"
)
model.fit(X, pairs, outcomes, reference_X=None)
model.build_posterior()
scores = model.predict_scores(X)
probabilities = model.predict_comparisons(X, candidate_pairs)
joint_means, joint_covariances = model.pair_posterior_moments(X, candidate_pairs)
position_variances = model.posterior_position_variance(X)
state = model.to_state()
restored = LinearDavidsonScaler.from_state(state)
```

- **`X`**: two-dimensional finite, dense NumPy or CSR matrix, shape `(n,d)` with `n>=2,d>=1`. It must be the same registered numerical representation for all participating observations. Sparse preprocessing retains a sparse matrix.
- **`pairs`**: `(m,2)` integral array, with distinct zero-based row indices into *the passed X*, never stable IDs. An arbitrary number of repeated or conflicting judgments is allowed.
- **`outcomes`**: integral `(m,)` array, **0** first higher, **1** parity, **2** second higher, **-1** unsure/skip. Skips are counted and omitted from the likelihood.
- **`reference_X`**: optional frozen, finite `(r,d)` eligible reference distribution, with `r>=2`, in the same original feature coordinates. Defaults to the rows of the training `X`. It determines the RMS normalization, fixed score origin and feature-variance prior scale; do not derive it from selected labels or outcomes.
- **`predict_scores`**: one centered real number per row, not a probability or a 0–100 score. The frozen origin is the mean feature vector of the reference representation.
- **`predict_comparisons`**: `(m,3)` softmax probabilities for `(first, parity, second)`, at the MAP point estimate. They represent judgment noise as well as the modeled ordering; they are *not* an estimate of positional epistemic uncertainty.
- **`pair_posterior_moments`**: mean `(m,2)`, covariance `(m,2,2)` for `(delta,eta)`. In particular `cov[:,0,1]` retains score-difference/tie correlation.
- **`posterior_position_variance`**: `(n,)` centered epistemic score-position variance under a fixed, frozen reference mean.
- **Fit diagnostics**: `success_`, `message_`, `optimality_`, `n_iterations_`, `n_comparisons_`, `n_skipped_`, `evidence_status_`. `prior-only` indicates no usable human comparisons, not sufficient learned evidence. An unsuccessful fit must not be used to build a posterior.
- **Posterior**: `posterior_.mode` is `prior-only`, `primal-exact` or `dual-exact`. `auto` selects dual for `m<=min(d,512)` and primal otherwise. No truncated spectral substitution is implemented. Primal has a 2,048-feature cap and quadratic space; dual is capped at 768 comparisons. These limits are explicit; successful computation and approximate normality are separate questions.

## Numerical model

Scores are `s_i = beta @ (z_i - reference_mean)`, and differences are invariant to the chosen origin. For each training pair, logits are `[delta/2, eta, -delta/2]`, where `delta = beta @ (z_i-z_j)`. Sum negative log-likelihood is regularized by independent proper Gaussian priors for `beta` and `eta`:

```
0.5 * lambda_beta * ||beta||^2
+ 0.5 * lambda_eta * (eta - eta_prior)^2
```

The reference trace variance `V = E_ref ||z-E_ref z||^2` determines `lambda_beta=max(V,1e-8)/score_prior_sd**2`. `lambda_eta=1/eta_prior_sd**2`. This is a proper prior with an explicit numerical floor for constant representations, not a dimension-count penalty. Optional per-column RMS normalization is frozen and operates inside the model; it never overwrites geometries.

MAP uses analytic gradients and L-BFGS-B; the joint Laplace covariance uses the analytic Hessian. Dual mode uses a Woodbury/Cholesky representation; primal mode uses a dense parameter-space Cholesky solve and covariance. Both are **exact numerical covariance within a Laplace approximation**, not exact posterior inference. Use the approximate posterior only conditionally on representation, independent-judgment model and priors. Comparisons arising from Code-to-Scale bootstrapping use the same likelihood as any other human comparisons, without any weighting.

## Fitted state and serialization

`to_state()` returns a **version 1 Python mapping containing NumPy arrays**, not a JSON document. This deliberately delegates the encoding, checksum, fit identity, geometry fingerprint, version lifecycle, and persistent storage to the application. The map includes constructor `config`, `n_features`, fit counts and diagnostics, frozen `column_scale`/`feature_mean`/`reference_variance`, `lambda_beta`/`lambda_eta`, `beta`/`eta`, and an optional `posterior` dictionary with `mode` and optional `dual_G`/`dual_L`/`full_cov`. `from_state` restores prediction and posterior moment computation; original training matrices are *not* persisted and cannot be reconstructed from fitted state.

The integration layer must keep a correspondence from stable observation identities to rows, provide eligible reference statistics, choose when a fit is stale, preserve every individual human judgment, and handle failures. Phase 3 can consume joint bivariate Gaussian pair moments for acquisition without depending on optimizer internals.

## Methodological limitations

Scale values depend on the chosen representation, proper prior and human judgments. Gaussian/Laplace uncertainty may not be calibrated for limited data, correlated annotators, construct misspecification, or distribution shift. A fully prior-driven solution is not an informative acquired ranking. This module deliberately does not implement pair sampling, acquisition, UI actions, SQLite persistence, auto-training, or confirmatory inference.
