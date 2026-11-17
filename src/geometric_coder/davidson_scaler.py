"""Numerical linear Davidson scaler: regularized MAP and joint Laplace posterior.

Outcome codes: 0 first higher, 1 parity, 2 second higher, -1 unsure/skip.
No observation identifiers or application state enter this module. Parameter
uncertainty is conditional on the linear model, Gaussian priors and Laplace fit;
it is not a guarantee of calibration or construct validity.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.linalg import cholesky, solve_triangular, cho_solve
from scipy.special import logsumexp


def davidson_probabilities(delta, eta):
    """Stable (first wins, tie, second wins) probabilities, broadcasting."""
    logits = np.stack(np.broadcast_arrays(np.asarray(delta) / 2,
                                            np.asarray(eta) + np.zeros_like(delta),
                                            -np.asarray(delta) / 2), axis=-1)
    return np.exp(logits - logsumexp(logits, axis=-1, keepdims=True))


def _row_sumsq(M):
    if sparse.issparse(M):
        return np.asarray(M.multiply(M).sum(axis=1)).ravel()
    return np.einsum('ij,ij->i', M, M)


def _factor_dual_gram(G):
    """Return a stable Woodbury factor for the same G.T @ G.

    Cholesky(I + G @ G.T) loses small eigenvalues when large or repeated
    comparison directions make its Gram matrix ill-conditioned. In that
    regime, rotate G with a thin SVD: (S @ V.T).T @ (S @ V.T) is unchanged,
    but the dual Gram becomes diagonal, so its Cholesky factor is accurate.
    The ordinary Cholesky fast path remains unchanged for well-scaled data.
    """
    if np.sum(_row_sumsq(G)) > 1e8:
        _, singular_values, vh = np.linalg.svd(G, full_matrices=False)
        rotated = singular_values[:, None] * vh
        return rotated, np.diag(np.hypot(1., singular_values))
    L = cholesky(np.eye(G.shape[0]) + G @ G.T,
                 lower=True, check_finite=False)
    return G, L


def _stable_dual_variances(G, L, whitened):
    """Evaluate Woodbury covariance quadratics without subtractive cancellation.

    For A=I+GG.T and u=A^-1 Gv, the quadratic v.T(I+G.T G)^-1 v
    equals ||v-G.T u||^2 + ||u||^2, a nonnegative sum of squares.
    """
    u = cho_solve((L, True), G @ whitened.T, check_finite=False)
    residual = whitened - u.T @ G
    return _row_sumsq(residual) + np.sum(u * u, axis=0)


def _repair_dual_beta_variances(G, L, rows, beta_sd, prior, variance):
    """Recompute cancellation-sensitive beta variances in bounded chunks."""
    bad = np.flatnonzero((prior > 0) & (variance <= 1e-8 * prior))
    chunk_size = min(256, max(1, 8_388_608 // (8 * G.shape[1])))
    for start in range(0, len(bad), chunk_size):
        indices = bad[start:start + chunk_size]
        chunk = rows[indices]
        if sparse.issparse(chunk):
            chunk = chunk.toarray()
        whitened = np.zeros((len(indices), G.shape[1]), dtype=float)
        whitened[:, :-1] = np.asarray(chunk) * beta_sd
        variance[indices] = _stable_dual_variances(G, L, whitened)
    return variance


def _stable_dual_eta_variance(G, L, eta_sd):
    whitened = np.zeros((1, G.shape[1]))
    whitened[0, -1] = eta_sd
    return float(_stable_dual_variances(G, L, whitened)[0])


def _validate_pair_indices(indices, n_rows):
    """Validate row indices before conversion, including unsigned overflows."""
    raw = np.asarray(indices)
    if (raw.ndim != 2 or raw.shape[1] != 2
            or not np.issubdtype(raw.dtype, np.integer)):
        raise ValueError('pairs must have shape (m,2) with integer indices')
    if raw.size and (
        (raw < 0).any() or (raw >= n_rows).any()
        or (raw[:, 0] == raw[:, 1]).any()
    ):
        raise ValueError('invalid pair indices')
    return raw.astype(np.int64, copy=False)


def _centered_reference_variance(R, center):
    """Stable reference trace variance without dense CSR materialization."""
    if sparse.issparse(R):
        R = R.tocsr(copy=True)
        R.sum_duplicates()
        indices = R.indices
        observed = np.bincount(indices, minlength=R.shape[1])
        centered_sq = np.square(R.data - center[indices])
        total = np.bincount(indices, weights=centered_sq, minlength=R.shape[1])
        # Implicit CSR zeros contribute (0 - center[j])**2.
        total += (R.shape[0] - observed) * np.square(center)
        return float(total.sum() / R.shape[0])
    return float(np.square(np.asarray(R) - center).sum() / R.shape[0])


def objective(theta, D, y, lambda_beta, lambda_eta, eta_prior):
    """NLP + exact analytical gradient, unnormalized *sum* NLL."""
    delta = np.asarray(D @ theta[:-1]).ravel()
    eta = theta[-1]
    logits = np.column_stack((delta / 2, np.full(len(y), eta), -delta / 2))
    ll = logsumexp(logits, axis=1) - logits[np.arange(len(y)), y]
    p = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    ad = .5 * (p[:, 0] - p[:, 2]) - .5 * (y == 0) + .5 * (y == 2)
    ae = p[:, 1] - (y == 1)
    f = ll.sum() + .5 * lambda_beta * np.dot(theta[:-1], theta[:-1]) \
        + .5 * lambda_eta * (eta - eta_prior)**2
    gb = np.asarray(D.T @ ad).ravel() + lambda_beta * theta[:-1]
    ge = np.sum(ae) + lambda_eta * (eta - eta_prior)
    return float(f), np.r_[gb, ge]


def likelihood_curvature(delta, eta):
    p = davidson_probabilities(delta, eta)
    mean_a = .5 * (p[:, 0] - p[:, 2])
    w_dd = .25 * (p[:, 0] + p[:, 2]) - mean_a**2
    w_de = -mean_a * p[:, 1]
    w_ee = p[:, 1] * (1 - p[:, 1])
    return w_dd, w_de, w_ee


@dataclass
class LaplaceSpectrum:
    """Exact-within-Laplace parameter covariance in primal or Woodbury form."""
    mode: str
    dual_G: np.ndarray | None = None
    dual_L: np.ndarray | None = None
    full_cov: np.ndarray | None = None

class LinearDavidsonScaler:
    def __init__(self, *, feature_normalization='none', score_prior_sd=1.,
                 eta_prior=np.log(.5), eta_prior_sd=1., maxiter=350,
                 posterior_method='auto'):
        if feature_normalization not in ('none', 'rms'):
            raise ValueError('feature_normalization must be none or rms')
        if (not np.isfinite(score_prior_sd) or score_prior_sd <= 0
                or not np.isfinite(eta_prior_sd) or eta_prior_sd <= 0
                or not np.isfinite(eta_prior)):
            raise ValueError('finite proper Gaussian priors required')
        if not isinstance(maxiter, int) or maxiter < 1:
            raise ValueError('maxiter must be a positive integer')
        self.feature_normalization = feature_normalization
        self.score_prior_sd = float(score_prior_sd)
        self.eta_prior = float(eta_prior)
        self.eta_prior_sd = float(eta_prior_sd)
        self.maxiter = maxiter
        if posterior_method not in ('auto', 'dual', 'primal'):
            raise ValueError('posterior_method must be auto, dual or primal')
        self.posterior_method = posterior_method

    def _scale_X(self, X):
        if X.shape[1] != len(self.column_scale_):
            raise ValueError('feature dimension mismatch')
        if sparse.issparse(X):
            return X.tocsr().astype(float).multiply(1. / self.column_scale_).tocsr()
        return np.asarray(X, dtype=float) / self.column_scale_

    def fit(self, X, pair_indices, outcomes, *, reference_X=None):
        X = X.tocsr() if sparse.issparse(X) else np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1:
            raise ValueError('X must have shape (n>=2, d>=1)')
        pairs = _validate_pair_indices(pair_indices, X.shape[0])
        raw_y = np.asarray(outcomes)
        if not np.issubdtype(raw_y.dtype, np.integer):
            raise ValueError('outcomes must be integer arrays')
        ys = np.asarray(raw_y, dtype=np.int64)
        if ys.shape != (len(pairs),):
            raise ValueError('outcomes must have shape (m,) matching pairs')
        if not np.isin(ys, [-1, 0, 1, 2]).all():
            raise ValueError('outcomes must be -1, 0, 1 or 2')
        if not np.isfinite(X.data if sparse.issparse(X) else X).all():
            raise ValueError('nonfinite X')
        if reference_X is None:
            reference = X
        elif sparse.issparse(reference_X):
            reference = reference_X.tocsr()
        else:
            reference = np.asarray(reference_X, dtype=float)
        if reference.ndim != 2 or reference.shape[1] != X.shape[1] or reference.shape[0] < 2:
            raise ValueError('reference_X must have shape (r>=2, d) matching X')
        if not np.isfinite(reference.data if sparse.issparse(reference) else reference).all():
            raise ValueError('nonfinite reference_X')
        if self.feature_normalization == 'rms':
            if sparse.issparse(reference):
                rms = np.sqrt(np.asarray(reference.multiply(reference).mean(axis=0)).ravel())
            else:
                rms = np.sqrt(np.mean(np.square(reference), axis=0))
            self.column_scale_ = np.where(rms <= 1e-12, 1., rms)
        else:
            self.column_scale_ = np.ones(X.shape[1])
        Z = self._scale_X(X)
        R = self._scale_X(reference)
        use = ys != -1
        self.n_comparisons_ = int(np.sum(use))
        self.n_skipped_ = int(np.sum(~use))
        self.n_features_ = Z.shape[1]
        D = Z[pairs[use, 0]] - Z[pairs[use, 1]]
        # Set isotropic beta prior to achieve an approximately fixed *utility*
        # prior SD for this reference representation. This is invariant to
        # multiplying all feature vectors by a global positive constant.
        self.feature_mean_ = np.asarray(R.mean(axis=0)).ravel()
        centered_energy = _centered_reference_variance(R, self.feature_mean_)
        self.reference_variance_ = centered_energy
        self.lambda_beta_ = max(centered_energy, 1e-8) / self.score_prior_sd**2
        self.lambda_eta_ = 1. / self.eta_prior_sd**2
        if (not np.isfinite(self.lambda_beta_) or self.lambda_beta_ <= 0
                or not np.isfinite(self.lambda_eta_) or self.lambda_eta_ <= 0):
            raise ValueError('prior precision is not finite and positive')
        init = np.zeros(self.n_features_ + 1)
        init[-1] = self.eta_prior
        res = minimize(objective, init, args=(D, ys[use], self.lambda_beta_,
                    self.lambda_eta_, self.eta_prior), method='L-BFGS-B', jac=True,
                    options={'maxiter': self.maxiter, 'ftol': 1e-10, 'gtol': 1e-6})
        self.beta_ = res.x[:-1]
        self.eta_ = float(res.x[-1])
        self.success_ = bool(res.success)
        self.message_ = str(res.message)
        self.n_iterations_ = int(res.nit)
        self.optimality_ = float(np.linalg.norm(objective(res.x,D,ys[use],
                                        self.lambda_beta_,self.lambda_eta_,
                                        self.eta_prior)[1], ord=np.inf))
        self.origin_ = float(self.feature_mean_ @ self.beta_)
        self.fitted_ = bool(np.isfinite(res.x).all() and np.isfinite(res.fun))
        self.success_ = self.success_ and self.fitted_
        self.evidence_status_ = (
            'prior-only' if self.n_comparisons_ == 0
            else ('fitted' if self.success_ else 'failed')
        )
        self._D = D
        self._curvature = likelihood_curvature(np.asarray(D @ self.beta_).ravel(), self.eta_)
        self.posterior_ = None
        return self

    def build_posterior(self):
        if not getattr(self, 'success_', False):
            raise RuntimeError('cannot build reliable posterior: fit was unsuccessful')
        D = self._D
        dd, de, ee = self._curvature
        n, d = D.shape
        sig_b = 1 / np.sqrt(self.lambda_beta_)
        sig_e = 1 / np.sqrt(self.lambda_eta_)
        p = d + 1
        trace = float(sig_b**2 * (dd @ _row_sumsq(D)) + sig_e**2 * ee.sum())
        if n == 0 or trace == 0:
            self.posterior_ = LaplaceSpectrum('prior-only')
            self.posterior_variance_eta_ = sig_e**2
            return self
        mode_selected=self.posterior_method
        if mode_selected=='auto':
            mode_selected='dual' if n<=min(512,d) else 'primal'
        if mode_selected=='primal':
            if p>2049:
                raise ValueError('primal exact covariance capped at 2049 parameters')
            Ddense=D.toarray() if sparse.issparse(D) else D
            H=np.zeros((p,p),dtype=float)
            H[:-1,:-1]=Ddense.T@(dd[:,None]*Ddense)
            H[:-1,:-1].flat[::d+1]+=self.lambda_beta_
            H[-1,:-1]=Ddense.T@de
            H[:-1,-1]=H[-1,:-1]
            H[-1,-1]=ee.sum()+self.lambda_eta_
            L=cholesky(H,lower=True,check_finite=False)
            C=cho_solve((L,True),np.eye(p),check_finite=False)
            C=(C+C.T)*.5
            self.posterior_=LaplaceSpectrum('primal-exact', full_cov=C)
            self.posterior_variance_eta_=max(0.,float(C[-1,-1]))
            return self
        if mode_selected=='dual':
            if n>768:
                raise ValueError('dual exact covariance only for small comparison count')
            t=np.sqrt(np.maximum(dd,0))
            b=np.divide(de,t,out=np.zeros_like(de),where=t>1e-14)*sig_e
            residual=ee.sum()-np.sum(np.divide(de**2,dd,out=np.zeros_like(de),where=dd>1e-14))
            if residual < -1e-8:
                raise ArithmeticError('negative Fisher Schur residual')
            G=np.zeros((n+1,p),dtype=float)
            G[:n,:d]=(D.toarray() if sparse.issparse(D) else D)*(t*sig_b)[:,None]
            G[:n,-1]=b
            G[n,-1]=sig_e*np.sqrt(max(residual,0))
            G, L = _factor_dual_gram(G)
            self.posterior_=LaplaceSpectrum('dual-exact', dual_G=G, dual_L=L)
            self.posterior_variance_eta_=_stable_dual_eta_variance(G,L,sig_e)
            return self
        raise RuntimeError('unsupported posterior covariance method')

    def predict_scores(self, X):
        Z = self._scale_X(X)
        return np.asarray(Z @ self.beta_).ravel() - self.origin_

    def predict_comparisons(self, X, pairs):
        scores = self.predict_scores(X)
        p = _validate_pair_indices(pairs, len(scores))
        return davidson_probabilities(scores[p[:,0]]-scores[p[:,1]], self.eta_)

    def posterior_pair_moments(self, X, pairs, *, scores=None):
        """Return mean delta, var(delta), cov(delta,eta), var(eta)."""
        if self.posterior_ is None:
            raise RuntimeError('call build_posterior first')
        Z = self._scale_X(X)
        p = _validate_pair_indices(pairs, Z.shape[0])
        return self._moments_scaled(Z,p,scores=scores)

    def _moments_scaled(self, Z, p, *, scores=None):
        D = Z[p[:,0]]-Z[p[:,1]]
        mu = np.asarray(D @ self.beta_).ravel() if scores is None else \
            np.asarray(scores)[p[:,0]]-np.asarray(scores)[p[:,1]]
        sig_b = 1/np.sqrt(self.lambda_beta_)
        sig_e = 1/np.sqrt(self.lambda_eta_)
        if self.posterior_.full_cov is not None:
            C=self.posterior_.full_cov
            difference = D.toarray() if sparse.issparse(D) else D
            variance = np.sum(np.asarray(D @ C[:-1, :-1]) * difference, axis=1)
            cov=np.asarray(D@C[:-1,-1]).ravel()
            variance_eta=C[-1,-1]
        elif self.posterior_.dual_G is not None:
            G,L=self.posterior_.dual_G,self.posterior_.dual_L
            projected = np.asarray(D @ G[:, :-1].T) * sig_b
            proj = solve_triangular(L, projected.T, lower=True, check_finite=False)
            eta_proj=solve_triangular(L,G[:,-1]*sig_e,lower=True,check_finite=False)
            prior = sig_b**2*_row_sumsq(D)
            variance=prior-np.sum(proj**2,axis=0)
            variance=_repair_dual_beta_variances(G,L,D,sig_b,prior,variance)
            cov=-(proj.T@eta_proj)
            variance_eta=self.posterior_variance_eta_
        else:
            variance = sig_b**2 * _row_sumsq(D)
            cov = np.zeros(len(mu))
            variance_eta = sig_e**2
        return mu,np.maximum(variance,0),cov,float(max(variance_eta,0))

    def posterior_position_variance(self, X, *, chunk_size=512):
        """Position variance of s(x)=beta^T(x-mean training x), with mean x fixed."""
        if self.posterior_ is None:raise RuntimeError('call build_posterior first')
        if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
            raise ValueError('chunk_size must be a positive integer')
        Z=self._scale_X(X)
        ret=np.empty(Z.shape[0])
        sig=1/np.sqrt(self.lambda_beta_)
        # CSR centering by differences of squared norms suffers catastrophic
        # cancellation for large offsets. Densify only bounded score chunks.
        rows_per_chunk = int(chunk_size)
        if sparse.issparse(Z):
            rows_per_chunk = min(rows_per_chunk, max(1, 8_388_608 // (8*Z.shape[1])))
        for i in range(0,len(ret),rows_per_chunk):
            Xc=Z[i:i+rows_per_chunk]
            centered = (Xc.toarray() if sparse.issparse(Xc) else Xc) - self.feature_mean_
            xc2=_row_sumsq(centered)
            if self.posterior_.full_cov is not None:
                C=self.posterior_.full_cov[:-1,:-1]
                variance=np.einsum('ij,ij->i',centered@C,centered)
            elif self.posterior_.dual_G is not None:
                G,L=self.posterior_.dual_G,self.posterior_.dual_L
                P=np.asarray(centered@G[:,:-1].T)*sig
                W=solve_triangular(L,P.T,lower=True,check_finite=False)
                prior=sig**2*xc2
                variance=prior-np.sum(W**2,axis=0)
                variance=_repair_dual_beta_variances(G,L,centered,sig,prior,variance)
            else:
                variance=sig**2*xc2
            ret[i:i+rows_per_chunk]=np.maximum(0,variance)
        return ret

    def pair_posterior_moments(self, X, pairs):
        """Return (means[m,2], covariance[m,2,2]) for (delta, eta)."""
        delta, var_delta, cross, var_eta = self.posterior_pair_moments(X, pairs)
        means = np.column_stack((delta, np.full(len(delta), self.eta_)))
        covariance = np.zeros((len(delta), 2, 2))
        covariance[:, 0, 0] = var_delta
        covariance[:, 0, 1] = cross
        covariance[:, 1, 0] = cross
        covariance[:, 1, 1] = var_eta
        if not np.isfinite(means).all() or not np.isfinite(covariance).all():
            raise ArithmeticError("nonfinite Laplace pair moments")
        return means, covariance

    def to_state(self):
        """Return versioned NumPy-array state; the application owns durable encoding."""
        if not getattr(self, 'fitted_', False):
            raise RuntimeError('no fitted parameter state')
        posterior = self.posterior_
        return {
            'schema_version': 1,
            'config': {
                'feature_normalization': self.feature_normalization,
                'score_prior_sd': self.score_prior_sd,
                'eta_prior': self.eta_prior,
                'eta_prior_sd': self.eta_prior_sd,
                'maxiter': self.maxiter,
                'posterior_method': self.posterior_method,
            },
            'n_features': self.n_features_,
            'n_comparisons': self.n_comparisons_,
            'n_skipped': self.n_skipped_,
            'column_scale': self.column_scale_.copy(),
            'feature_mean': self.feature_mean_.copy(),
            'reference_variance': self.reference_variance_,
            'lambda_beta': self.lambda_beta_,
            'lambda_eta': self.lambda_eta_,
            'beta': self.beta_.copy(),
            'eta': self.eta_,
            'success': self.success_,
            'message': self.message_,
            'n_iterations': self.n_iterations_,
            'optimality': self.optimality_,
            'evidence_status': self.evidence_status_,
            'posterior': None if posterior is None else {
                'mode': posterior.mode,
                'dual_G': None if posterior.dual_G is None else posterior.dual_G.copy(),
                'dual_L': None if posterior.dual_L is None else posterior.dual_L.copy(),
                'full_cov': None if posterior.full_cov is None else posterior.full_cov.copy(),
            },
        }

    @classmethod
    def from_state(cls, state):
        """Reload a fitted state without the original training observations."""
        if state.get('schema_version') != 1:
            raise ValueError('unknown Davidson scaler state version')
        model = cls(**state['config'])
        for name, key in (
            ('n_features_', 'n_features'), ('n_comparisons_', 'n_comparisons'),
            ('n_skipped_', 'n_skipped'), ('reference_variance_', 'reference_variance'),
            ('lambda_beta_', 'lambda_beta'), ('lambda_eta_', 'lambda_eta'),
            ('eta_', 'eta'), ('success_', 'success'), ('message_', 'message'),
            ('n_iterations_', 'n_iterations'), ('optimality_', 'optimality'),
            ('evidence_status_', 'evidence_status')
        ):
            setattr(model, name, state[key])
        for name, key in (('column_scale_', 'column_scale'),
                          ('feature_mean_', 'feature_mean'), ('beta_', 'beta')):
            setattr(model, name, np.asarray(state[key], dtype=float).copy())
        if model.beta_.shape != (model.n_features_,) or (
            model.feature_mean_.shape != model.beta_.shape
            or model.column_scale_.shape != model.beta_.shape
            or not np.isfinite(model.beta_).all()
            or not np.isfinite(model.column_scale_).all()
            or np.any(model.column_scale_ <= 0)
        ):
            raise ValueError('invalid Davidson scaler state')
        if (not np.isfinite(model.lambda_beta_) or model.lambda_beta_ <= 0
                or not np.isfinite(model.lambda_eta_) or model.lambda_eta_ <= 0
                or not np.isfinite(model.eta_)):
            raise ValueError('invalid Gaussian prior or tie parameter in state')
        model.origin_ = float(model.feature_mean_ @ model.beta_)
        model.fitted_ = True
        p = state['posterior']
        model.posterior_ = None if p is None else LaplaceSpectrum(
            p['mode'],
            dual_G=None if p['dual_G'] is None else np.asarray(p['dual_G'], dtype=float).copy(),
            dual_L=None if p['dual_L'] is None else np.asarray(p['dual_L'], dtype=float).copy(),
            full_cov=(None if p['full_cov'] is None
                      else np.asarray(p['full_cov'], dtype=float).copy())
        )
        if model.posterior_ is not None:
            q = model.posterior_
            if q.full_cov is not None:
                model.posterior_variance_eta_ = float(q.full_cov[-1, -1])
            elif q.dual_G is not None:
                sig_e = 1. / np.sqrt(model.lambda_eta_)
                model.posterior_variance_eta_ = _stable_dual_eta_variance(q.dual_G, q.dual_L, sig_e)
            else:
                model.posterior_variance_eta_ = 1. / model.lambda_eta_
        return model
