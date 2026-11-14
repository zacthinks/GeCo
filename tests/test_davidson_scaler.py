"""Numerical regression coverage for the Davidson scaler."""
import numpy as np
import pytest
from scipy import sparse
from scipy.optimize._numdiff import approx_derivative
from scipy.stats import spearmanr

from geometric_coder.davidson_scaler import (
    LinearDavidsonScaler, davidson_probabilities, likelihood_curvature, objective,
)


def example(seed=4, n=60, d=8, m=130):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    beta = rng.normal(size=d)
    beta /= np.linalg.norm(beta)
    pairs = rng.integers(0, n, size=(m, 2))
    pairs[:, 1] = (pairs[:, 0] + 1 + rng.integers(0, n-1, size=m)) % n
    probs = davidson_probabilities((X[pairs[:, 0]]-X[pairs[:, 1]]) @ beta, -.5)
    y = np.array([rng.choice(3, p=p) for p in probs])
    return X, pairs, y, beta


def test_probabilities_and_side_swap():
    delta = np.array([-700., -2., 0., 3., 700.])
    p = davidson_probabilities(delta, -.3)
    assert np.isfinite(p).all()
    assert (p > 0).all()
    assert np.allclose(p.sum(axis=1), 1)
    swapped = davidson_probabilities(-delta, -.3)
    assert np.allclose(p[:, 0], swapped[:, 2])
    assert np.allclose(p[:, 1], swapped[:, 1])


def test_gradient_and_hessian_finite_differences():
    rng = np.random.default_rng(0)
    D = rng.normal(size=(14, 4))
    y = rng.integers(0, 3, size=14)
    t = rng.normal(size=5)
    f, g = objective(t, D, y, .7, 1.3, -.1)
    gn = approx_derivative(lambda x: objective(x, D, y, .7, 1.3, -.1)[0],
                           t, method='3-point').ravel()
    assert np.max(abs(g-gn)) < 1e-6
    dd, de, ee = likelihood_curvature(D @ t[:-1], t[-1])
    H = np.block([[D.T @ (dd[:, None]*D)+.7*np.eye(4), (D.T @ de)[:, None]],
                  [(de @ D)[None, :], np.array([[ee.sum()+1.3]])]])
    hn = approx_derivative(lambda x: objective(x, D, y, .7, 1.3, -.1)[1],
                           t, method='3-point')
    assert np.max(abs(H-hn)) < 1e-6
    assert np.linalg.eigvalsh(H).min() > 0


def test_parity_vs_skip_and_prior_only():
    X, pairs, y, _ = example(m=10)
    skip = LinearDavidsonScaler().fit(X, pairs, np.full(10, -1)).build_posterior()
    assert skip.n_comparisons_ == 0 and skip.n_skipped_ == 10
    assert skip.evidence_status_ == 'prior-only'
    assert np.allclose(skip.beta_, 0) and np.isclose(skip.eta_, np.log(.5))
    assert np.all(skip.posterior_position_variance(X) >= 0)
    ties = LinearDavidsonScaler().fit(X, pairs, np.ones(10, dtype=int))
    assert ties.n_comparisons_ == 10 and ties.n_skipped_ == 0
    assert ties.eta_ > skip.eta_


def test_all_parity_and_no_parity():
    X, pairs, y, _ = example()
    for outcome in (0, 1, 2):
        model = LinearDavidsonScaler().fit(X, pairs, np.full(len(y), outcome))
        assert model.success_
        assert np.isfinite(model.beta_).all() and np.isfinite(model.eta_)


def test_conflicting_repeated_pairs_are_independent():
    X = np.array([[1., 0.], [0., 0.], [0., 1.]])
    pairs = np.tile([[0, 1]], (12, 1))
    y = np.array([0]*6 + [2]*6)
    model = LinearDavidsonScaler().fit(X, pairs, y).build_posterior()
    assert model.n_comparisons_ == 12
    assert np.isclose(model.predict_comparisons(X, [[0,1]])[0,0],
                      model.predict_comparisons(X, [[0,1]])[0,2], atol=1e-7)
    v = model.pair_posterior_moments(X, [[0,1]])[1][0,0,0]
    single = LinearDavidsonScaler().fit(X, pairs[:2], y[[0,6]]).build_posterior()
    assert v < single.pair_posterior_moments(X, [[0,1]])[1][0,0,0]


@pytest.mark.parametrize('normalization', ['none', 'rms'])
def test_dense_csr_and_prior_rescaling(normalization):
    X, pairs, y, _ = example(d=12)
    X[abs(X) < 1] = 0
    a = LinearDavidsonScaler(feature_normalization=normalization).fit(X, pairs, y)
    b = LinearDavidsonScaler(feature_normalization=normalization).fit(
        sparse.csr_matrix(X), pairs, y)
    assert np.allclose(a.predict_scores(X), b.predict_scores(sparse.csr_matrix(X)), atol=1e-7)
    assert np.allclose(a.lambda_beta_, b.lambda_beta_)
    c = LinearDavidsonScaler(feature_normalization=normalization).fit(X*5, pairs, y)
    assert np.allclose(a.predict_scores(X), c.predict_scores(X*5), atol=2e-4)
    assert np.isclose(a.predict_scores(X).mean(), 0, atol=1e-10)


def test_frozen_reference_statistics_and_centered_position():
    X, pairs, y, _ = example(n=40, d=6)
    ref = X + 4
    model = LinearDavidsonScaler(feature_normalization='rms').fit(
        X, pairs, y, reference_X=ref).build_posterior()
    assert np.allclose(model.predict_scores(ref).mean(), 0, atol=1e-10)
    assert np.allclose(model.posterior_position_variance(ref.mean(axis=0)[None, :]), 0,
                       atol=1e-9)
    assert np.allclose(model.column_scale_, np.sqrt(np.mean(ref**2, axis=0)))


def test_joint_covariance_and_primal_dual_agreement():
    X, pairs, y, _ = example(d=9, m=42)
    p = LinearDavidsonScaler(posterior_method='primal').fit(X, pairs, y).build_posterior()
    d = LinearDavidsonScaler(posterior_method='dual').fit(X, pairs, y).build_posterior()
    mu1, C1 = p.pair_posterior_moments(X, pairs[:15])
    mu2, C2 = d.pair_posterior_moments(X, pairs[:15])
    assert np.allclose(mu1, mu2)
    assert np.allclose(C1, C2, atol=1e-8, rtol=1e-7)
    assert np.allclose(C1, C1.transpose(0,2,1))
    assert np.linalg.eigvalsh(C1).min() > -1e-9
    assert np.allclose(p.posterior_position_variance(X),
                       d.posterior_position_variance(X), atol=1e-8)


@pytest.mark.parametrize('method', ['primal', 'dual'])
def test_serialization_roundtrip(method):
    X, pairs, y, _ = example(d=8, m=24)
    m = LinearDavidsonScaler(feature_normalization='rms',
                             posterior_method=method).fit(X, pairs, y).build_posterior()
    restored = LinearDavidsonScaler.from_state(m.to_state())
    assert np.allclose(restored.predict_scores(X), m.predict_scores(X))
    assert np.allclose(restored.predict_comparisons(X, pairs),
                       m.predict_comparisons(X, pairs))
    assert np.allclose(restored.pair_posterior_moments(X, pairs)[1],
                       m.pair_posterior_moments(X, pairs)[1])
    assert np.allclose(restored.posterior_position_variance(X),
                       m.posterior_position_variance(X))


def test_validation_and_extreme_fit():
    X, pairs, y, _ = example(n=40, d=6, m=70)
    for bad in (np.full(X.shape, np.nan),):
        with pytest.raises(ValueError):
            LinearDavidsonScaler().fit(bad, pairs, y)
    with pytest.raises(ValueError):
        LinearDavidsonScaler().fit(X, pairs.astype(float), y)
    with pytest.raises(ValueError):
        LinearDavidsonScaler().fit(X, pairs, y.astype(float))
    with pytest.raises(ValueError):
        LinearDavidsonScaler().fit(X, np.array([[0,0]]), np.array([0]))
    m = LinearDavidsonScaler().fit(X*10000, pairs, y).build_posterior()
    assert np.isfinite(m.predict_comparisons(X*10000, pairs)).all()


def test_small_m_large_d_dense_and_csr():
    X, pairs, y, _ = example(n=48, d=768, m=12)
    for matrix in (X, sparse.csr_matrix(X)):
        m = LinearDavidsonScaler(posterior_method='auto').fit(matrix, pairs, y)
        m.build_posterior()
        assert m.posterior_.mode == 'dual-exact'
        mu, C = m.pair_posterior_moments(matrix, pairs[:3])
        assert np.isfinite(mu).all() and np.isfinite(C).all()


def test_synthetic_ordering_recovery():
    X, pairs, y, truth = example(seed=9, n=120, d=6, m=2500)
    m = LinearDavidsonScaler().fit(X, pairs, y)
    assert m.success_
    assert spearmanr(X @ truth, m.predict_scores(X)).statistic > .8
