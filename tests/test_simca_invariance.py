"""
Invariance tests for SIMCA.

Each test fits SIMCA on generated data, applies a transformation that
must not change the classifier (extra constant or empty columns, empty
or sparse training rows, feature permutation or rescaling, row order)
and compares limits, distances and predictions with the untouched fit.
This covers the edge cases several adversarial reviews found one at a
time.
"""

import warnings

import numpy as np
import pytest

from open_nipals.simca import SIMCA

DATASETS = ["gaussian", "missing", "correlated"]
SETTINGS = [
    dict(n_components=2, scale=True),
    dict(n_components=2, scale=False),
    dict(n_components="auto", component_selection="r2", scale=True),
    dict(n_components="auto", component_selection="eigenvalue", scale=True),
    dict(n_components="auto", component_selection="q2", scale=True),
]
TOL = dict(rtol=1e-5, atol=1e-8)


def make_data(kind, seed=0):
    """Three classes of low-rank data plus noise, and evaluation data."""
    rng = np.random.default_rng(seed)
    n_features = 8
    X, y = [], []
    for label, shift in enumerate([0.0, 4.0, -4.0]):
        latent = rng.normal(size=(40, 3))
        basis = rng.normal(size=(3, n_features))
        noise = 0.3 * rng.normal(size=(40, n_features))
        X.append(shift + latent @ basis + noise)
        y.append(np.full(40, label))
    X, y = np.vstack(X), np.concatenate(y)
    if kind == "correlated":
        X[:, 1] = X[:, 0] + 0.01 * rng.normal(size=len(X))
    X_eval = X + 0.5 * rng.normal(size=X.shape)
    if kind == "missing":
        X[rng.random(X.shape) < 0.1] = np.nan
        X_eval[rng.random(X_eval.shape) < 0.1] = np.nan
    return X, y, X_eval


def fit(X, y, settings):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SIMCA(**settings).fit(X, y)


def assert_same_classifier(a, b, X_a, X_b, s0_factor=1.0):
    """Same component counts, limits, distances and predictions.

    s0 is in data units, so a common rescaling by s0_factor rescales it.
    """
    for label in a.classes_:
        ma, mb = a.class_models_[label], b.class_models_[label]
        assert ma.n_components == mb.n_components
        assert ma.n_samples == mb.n_samples
        np.testing.assert_allclose(mb.t2_limit, ma.t2_limit, **TOL)
        np.testing.assert_allclose(mb.dmodx_limit, ma.dmodx_limit, **TOL)
        np.testing.assert_allclose(mb.s0, s0_factor * ma.s0, **TOL)
    da, db = a.get_distances(X_a), b.get_distances(X_b)
    np.testing.assert_allclose(db["t2"], da["t2"], **TOL)
    np.testing.assert_allclose(db["dmodx"], da["dmodx"], **TOL)
    assert list(b.predict(X_b)) == list(a.predict(X_a))


@pytest.fixture(params=DATASETS)
def data(request):
    return make_data(request.param)


@pytest.fixture(params=range(len(SETTINGS)), ids=lambda i: str(SETTINGS[i]))
def settings(request):
    return SETTINGS[request.param]


def test_constant_columns(data, settings):
    X, y, X_eval = data
    base = fit(X, y, settings)
    extra = np.full((len(X), 2), 3.0)
    wide = fit(np.hstack([X, extra]), y, settings)
    assert_same_classifier(
        base, wide, X_eval, np.hstack([X_eval, extra[: len(X_eval)]])
    )


def test_empty_columns(data, settings):
    X, y, X_eval = data
    base = fit(X, y, settings)
    empty = np.full((len(X), 2), np.nan)
    wide = fit(np.hstack([X, empty]), y, settings)
    assert_same_classifier(base, wide, X_eval, np.hstack([X_eval, empty]))


def test_empty_training_rows(data, settings):
    X, y, X_eval = data
    base = fit(X, y, settings)
    X_pad = np.vstack([X, np.full((30, X.shape[1]), np.nan)])
    y_pad = np.concatenate([y, np.repeat(np.unique(y), 10)])
    assert_same_classifier(base, fit(X_pad, y_pad, settings), X_eval, X_eval)


def test_single_value_training_rows(data, settings):
    """Rows with one observed value inform no model with >= 1 component."""
    X, y, X_eval = data
    base = fit(X, y, settings)
    rng = np.random.default_rng(1)
    sparse = np.full((30, X.shape[1]), np.nan)
    sparse[np.arange(30), rng.integers(0, X.shape[1], 30)] = rng.normal(
        size=30
    )
    X_pad = np.vstack([X, sparse])
    y_pad = np.concatenate([y, np.repeat(np.unique(y), 10)])
    assert_same_classifier(base, fit(X_pad, y_pad, settings), X_eval, X_eval)


def test_feature_permutation(data, settings):
    X, y, X_eval = data
    perm = np.random.default_rng(2).permutation(X.shape[1])
    assert_same_classifier(
        fit(X, y, settings),
        fit(X[:, perm], y, settings),
        X_eval,
        X_eval[:, perm],
    )


def test_training_row_order(data, settings):
    X, y, X_eval = data
    order = np.random.default_rng(3).permutation(len(X))
    assert_same_classifier(
        fit(X, y, settings), fit(X[order], y[order], settings), X_eval, X_eval
    )


def test_feature_units(data, settings):
    """Per-feature units with scale=True, a common unit with scale=False."""
    X, y, X_eval = data
    if settings.get("scale", True):
        factor, s0_factor = np.logspace(-3, 3, X.shape[1]), 1.0
    else:
        factor = s0_factor = 1e3
    assert_same_classifier(
        fit(X, y, settings),
        fit(X * factor, y, settings),
        X_eval,
        X_eval * factor,
        s0_factor=s0_factor,
    )


def test_far_outliers_are_rejected(data, settings):
    X, y, _ = data
    model = fit(X, y, dict(settings, unknown_handling="reject"))
    far = np.nanmean(X, axis=0) + 1e3 * np.nanstd(X, axis=0)
    assert model.predict(far[None])[0] is None


def test_probabilities(data, settings):
    X, y, X_eval = data
    proba = fit(X, y, settings).predict_proba(X_eval)
    assert np.all(np.isfinite(proba))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_numpy_and_jax_agree(data, settings):
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    from open_nipals.jax.simca import SIMCA as SIMCA_JAX

    X, y, X_eval = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        jax_model = SIMCA_JAX(**settings).fit(X, y)
    assert_same_classifier(fit(X, y, settings), jax_model, X_eval, X_eval)
