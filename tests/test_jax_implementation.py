"""
Tests for JAX implementation of NIPALS algorithms.

Verifies that JAX implementations produce results equivalent to NumPy versions.
"""

import numpy as np
import pytest

# Try to import JAX
try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

if JAX_AVAILABLE:
    from open_nipals.jax import NipalsPCA as NipalsPCA_JAX
    from open_nipals.jax import NipalsPLS as NipalsPLS_JAX
    from open_nipals.jax.utils import _nan_mult as _nan_mult_jax

from open_nipals.nipalsPCA import NipalsPCA
from open_nipals.nipalsPLS import NipalsPLS
from open_nipals.utils import _nan_mult


@pytest.fixture
def sample_data_clean():
    """Generate clean sample data (no NaNs)."""
    np.random.seed(42)
    n, m = 100, 20
    X = np.random.randn(n, m)
    # Mean center
    X = X - X.mean(axis=0)
    return X


@pytest.fixture
def sample_data_with_nan():
    """Generate sample data with NaN values."""
    np.random.seed(42)
    n, m = 100, 20
    X = np.random.randn(n, m)
    # Mean center
    X = X - X.mean(axis=0)
    # Add 5% NaN values
    nan_mask = np.random.rand(n, m) < 0.05
    X[nan_mask] = np.nan
    return X


@pytest.fixture
def sample_xy_data():
    """Generate sample X and Y data for PLS."""
    np.random.seed(42)
    n, m_x, m_y = 100, 20, 3
    X = np.random.randn(n, m_x)
    # Create Y correlated with X
    true_weights = np.random.randn(m_x, m_y)
    Y = X @ true_weights + 0.1 * np.random.randn(n, m_y)
    # Mean center
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    return X, Y


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestNanMult:
    """Test _nan_mult JAX implementation against NumPy."""

    def test_clean_data(self, sample_data_clean):
        """Test that _nan_mult produces same results for clean data."""
        X = sample_data_clean
        y = np.random.randn(X.shape[1], 1)
        nan_mask = np.isnan(X)

        np_result = _nan_mult(X, y, nan_mask, use_denom=True)
        jax_result = _nan_mult_jax(
            jnp.array(X), jnp.array(y), jnp.array(nan_mask), use_denom=True
        )

        np.testing.assert_allclose(np_result, np.array(jax_result), **PARITY)

    def test_with_nan(self, sample_data_with_nan):
        """Test that _nan_mult produces same results with NaN data."""
        X = sample_data_with_nan
        y = np.random.randn(X.shape[1], 1)
        nan_mask = np.isnan(X)

        np_result = _nan_mult(X, y, nan_mask, use_denom=True)
        jax_result = _nan_mult_jax(
            jnp.array(X), jnp.array(y), jnp.array(nan_mask), use_denom=True
        )

        np.testing.assert_allclose(np_result, np.array(jax_result), **PARITY)

    def test_no_denom(self, sample_data_clean):
        """Test _nan_mult with use_denom=False."""
        X = sample_data_clean
        y = np.random.randn(X.shape[1], 1)
        nan_mask = np.isnan(X)

        np_result = _nan_mult(X, y, nan_mask, use_denom=False)
        jax_result = _nan_mult_jax(
            jnp.array(X), jnp.array(y), jnp.array(nan_mask), use_denom=False
        )

        np.testing.assert_allclose(np_result, np.array(jax_result), **PARITY)


# The JAX fits are a line-by-line port of the NumPy ones, so in float64
# they have to agree to rounding error, not just point the same direction.
PARITY = dict(rtol=1e-6, atol=1e-9)


@pytest.fixture
def sample_xy_data_with_nan(sample_xy_data):
    """X and Y data with NaNs in both blocks, no row of Y fully missing."""
    X, Y = sample_xy_data
    rng = np.random.default_rng(42)
    X = X.copy()
    Y = Y.copy()
    X[rng.random(X.shape) < 0.05] = np.nan
    y_mask = rng.random(Y.shape) < 0.05
    y_mask[y_mask.all(axis=1)] = False
    Y[y_mask] = np.nan
    return X - np.nanmean(X, axis=0), Y - np.nanmean(Y, axis=0)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestNipalsPCAJax:
    """Test JAX PCA implementation against NumPy."""

    @pytest.mark.parametrize("data", ["sample_data_clean", "sample_data_with_nan"])
    def test_fit(self, data, request):
        """Scores and loadings match the NumPy fit, with and without NaNs."""
        X = request.getfixturevalue(data)

        pca_np = NipalsPCA(n_components=5).fit(X)
        pca_jax = NipalsPCA_JAX(n_components=5).fit(X)

        np.testing.assert_allclose(pca_jax.loadings, pca_np.loadings, **PARITY)
        np.testing.assert_allclose(pca_jax.fit_scores, pca_np.fit_scores, **PARITY)

    @pytest.mark.parametrize("method", ["naive", "projection", "conditional_mean"])
    def test_transform_with_nan(self, sample_data_with_nan, method):
        """All missing data transform methods match NumPy."""
        X = sample_data_with_nan

        pca_np = NipalsPCA(n_components=3).fit(X)
        pca_jax = NipalsPCA_JAX(n_components=3).fit(X)

        # NumPy conditional_mean fills its input in place, so hand out copies
        scores_np = pca_np.transform(X.copy(), method=method)
        scores_jax = pca_jax.transform(X.copy(), method=method)

        np.testing.assert_allclose(scores_jax, scores_np, **PARITY)
        assert pca_jax.n_components == 3

    def test_transform_bad_method(self, sample_data_with_nan):
        pca_jax = NipalsPCA_JAX(n_components=2).fit(sample_data_with_nan)
        with pytest.raises(ValueError):
            pca_jax.transform(sample_data_with_nan, method="nonsense")

    @pytest.mark.parametrize("data", ["sample_data_clean", "sample_data_with_nan"])
    def test_set_components(self, data, request):
        """Adding components later gives the same model as NumPy."""
        X = request.getfixturevalue(data)

        pca_np = NipalsPCA(n_components=2).fit(X).set_components(5)
        pca_jax = NipalsPCA_JAX(n_components=2).fit(X).set_components(5)

        np.testing.assert_allclose(pca_jax.loadings, pca_np.loadings, **PARITY)
        np.testing.assert_allclose(pca_jax.fit_scores, pca_np.fit_scores, **PARITY)

    def test_set_components_shrink_then_grow(self, sample_data_with_nan):
        """Components added after shrinking must not repeat fitted ones."""
        X = sample_data_with_nan

        pca_ref = NipalsPCA_JAX(n_components=6).fit(X)
        pca_jax = NipalsPCA_JAX(n_components=4).fit(X)
        pca_jax.set_components(2).set_components(6)

        np.testing.assert_allclose(pca_jax.loadings, pca_ref.loadings, **PARITY)

    def test_conditional_mean_ignores_fit_history(self, sample_data_with_nan):
        """Same scores whether or not more components were fitted before."""
        X = sample_data_with_nan

        pca_ref = NipalsPCA_JAX(n_components=2).fit(X)
        pca_jax = NipalsPCA_JAX(n_components=4).fit(X).set_components(2)

        np.testing.assert_allclose(
            pca_jax.transform(X.copy(), method="conditional_mean"),
            pca_ref.transform(X.copy(), method="conditional_mean"),
            rtol=1e-6,
            atol=1e-7,
        )

    @pytest.mark.parametrize(
        "dtype, scale, rtol", [("float64", 1e-8, 1e-6), ("float32", 1e-3, 1e-2)]
    )
    def test_conditional_mean_scale_equivariant(
        self, sample_data_with_nan, dtype, scale, rtol
    ):
        """Rescaling the data must rescale the scores, nothing else."""
        X = sample_data_with_nan

        pca_unit = NipalsPCA_JAX(n_components=3, dtype=dtype).fit(X)
        pca_small = NipalsPCA_JAX(n_components=3, dtype=dtype).fit(X * scale)

        scores_unit = pca_unit.transform(X.copy(), method="conditional_mean")
        scores_small = pca_small.transform(X.copy() * scale, method="conditional_mean")

        np.testing.assert_allclose(
            scores_small / scale, scores_unit, rtol=rtol, atol=rtol
        )

    def test_distances(self, sample_data_with_nan):
        X = sample_data_with_nan

        pca_np = NipalsPCA(n_components=3).fit(X)
        pca_jax = NipalsPCA_JAX(n_components=3).fit(X)

        np.testing.assert_allclose(
            pca_jax.calc_imd(input_array=X), pca_np.calc_imd(input_array=X), **PARITY
        )
        np.testing.assert_allclose(pca_jax.calc_oomd(X), pca_np.calc_oomd(X), **PARITY)

    def test_inverse_transform(self, sample_data_clean):
        X = sample_data_clean

        pca_np = NipalsPCA(n_components=5).fit(X)
        pca_jax = NipalsPCA_JAX(n_components=5).fit(X)

        np.testing.assert_allclose(
            pca_jax.inverse_transform(pca_jax.transform(X)),
            pca_np.inverse_transform(pca_np.transform(X)),
            **PARITY,
        )

    def test_float32(self, sample_data_with_nan):
        """float32 is an approximation: same subspace, looser numbers."""
        X = sample_data_with_nan

        pca_np = NipalsPCA(n_components=3).fit(X)
        pca_jax = NipalsPCA_JAX(n_components=3, dtype="float32").fit(X)

        assert pca_jax.loadings.dtype == np.float64
        np.testing.assert_allclose(pca_jax.loadings, pca_np.loadings, atol=1e-2)

    def test_max_iter_warns(self, sample_data_clean):
        with pytest.warns(UserWarning, match="max_iter reached on LV 0"):
            NipalsPCA_JAX(n_components=1, max_iter=2).fit(sample_data_clean)

    def test_no_recompile_on_refit(self, sample_data_with_nan):
        """A second fit of the same shape must reuse the compiled code."""
        from open_nipals.jax.nipalsPCA import _fit_components

        NipalsPCA_JAX(n_components=3).fit(sample_data_with_nan)
        cache_size = _fit_components._cache_size()
        NipalsPCA_JAX(n_components=3).fit(sample_data_with_nan[::-1].copy())
        assert _fit_components._cache_size() == cache_size


PLS_ATTRIBUTES = [
    "fit_scores_x",
    "loadings_x",
    "weights_x",
    "fit_scores_y",
    "loadings_y",
    "regression_matrix",
]


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestNipalsPLSJax:
    """Test JAX PLS implementation against NumPy."""

    @pytest.mark.parametrize("data", ["sample_xy_data", "sample_xy_data_with_nan"])
    def test_fit(self, data, request):
        """Every fitted attribute matches NumPy, with and without NaNs."""
        X, Y = request.getfixturevalue(data)

        pls_np = NipalsPLS(n_components=3).fit(X, Y)
        pls_jax = NipalsPLS_JAX(n_components=3).fit(X, Y)

        for name in PLS_ATTRIBUTES:
            np.testing.assert_allclose(
                getattr(pls_jax, name), getattr(pls_np, name), err_msg=name, **PARITY
            )

    @pytest.mark.parametrize("data", ["sample_xy_data", "sample_xy_data_with_nan"])
    def test_predict_and_transform(self, data, request):
        X, Y = request.getfixturevalue(data)

        pls_np = NipalsPLS(n_components=3).fit(X, Y)
        pls_jax = NipalsPLS_JAX(n_components=3).fit(X, Y)

        np.testing.assert_allclose(pls_jax.predict(X), pls_np.predict(X), **PARITY)
        np.testing.assert_allclose(
            pls_jax.get_reg_vector(), pls_np.get_reg_vector(), **PARITY
        )
        for scores_jax, scores_np in zip(
            pls_jax.transform(X, Y), pls_np.transform(X, Y)
        ):
            np.testing.assert_allclose(scores_jax, scores_np, **PARITY)

    @pytest.mark.parametrize("data", ["sample_xy_data", "sample_xy_data_with_nan"])
    def test_set_components(self, data, request):
        X, Y = request.getfixturevalue(data)

        pls_np = NipalsPLS(n_components=2).fit(X, Y).set_components(4)
        pls_jax = NipalsPLS_JAX(n_components=2).fit(X, Y).set_components(4)

        for name in PLS_ATTRIBUTES:
            np.testing.assert_allclose(
                getattr(pls_jax, name), getattr(pls_np, name), err_msg=name, **PARITY
            )

    def test_set_components_shrink_then_grow(self, sample_xy_data_with_nan):
        """Components added after shrinking must not repeat fitted ones."""
        X, Y = sample_xy_data_with_nan

        # Reference grows without shrinking first: with NaNs in Y a grown
        # model differs slightly from a direct fit, also in the NumPy version
        pls_ref = NipalsPLS_JAX(n_components=4).fit(X, Y).set_components(5)
        pls_jax = NipalsPLS_JAX(n_components=4).fit(X, Y)
        pls_jax.set_components(2).set_components(5)

        for name in PLS_ATTRIBUTES:
            np.testing.assert_allclose(
                getattr(pls_jax, name), getattr(pls_ref, name), err_msg=name, **PARITY
            )

    def test_distances(self, sample_xy_data_with_nan):
        X, Y = sample_xy_data_with_nan

        pls_np = NipalsPLS(n_components=3).fit(X, Y)
        pls_jax = NipalsPLS_JAX(n_components=3).fit(X, Y)

        np.testing.assert_allclose(
            pls_jax.calc_imd(input_array=X), pls_np.calc_imd(input_array=X), **PARITY
        )
        np.testing.assert_allclose(pls_jax.calc_oomd(X), pls_np.calc_oomd(X), **PARITY)

    def test_all_nan_y_row_is_dropped(self, sample_xy_data):
        X, Y = sample_xy_data
        Y = Y.copy()
        Y[5, :] = np.nan

        with pytest.warns(UserWarning, match="dropped"):
            pls_jax = NipalsPLS_JAX(n_components=2).fit(X, Y)

        assert pls_jax.fit_scores_x.shape == (X.shape[0] - 1, 2)
        assert np.all(np.isfinite(pls_jax.predict(X)))

    def test_float32(self, sample_xy_data_with_nan):
        """float32 is an approximation: predictions agree to a few digits."""
        X, Y = sample_xy_data_with_nan

        pls_np = NipalsPLS(n_components=3).fit(X, Y)
        pls_jax = NipalsPLS_JAX(n_components=3, dtype="float32").fit(X, Y)

        np.testing.assert_allclose(pls_jax.predict(X), pls_np.predict(X), atol=1e-2)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestJAXDevices:
    """Test JAX device detection."""

    def test_jax_available(self):
        """Test that JAX is available."""
        import jax

        devices = jax.devices()
        assert len(devices) > 0, "No JAX devices found"
        print(f"JAX devices: {devices}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
