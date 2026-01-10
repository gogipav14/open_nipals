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
        jax_result = _nan_mult_jax(jnp.array(X), jnp.array(y), jnp.array(nan_mask), use_denom=True)

        np.testing.assert_allclose(np_result, np.array(jax_result), rtol=1e-5, atol=1e-8)

    def test_with_nan(self, sample_data_with_nan):
        """Test that _nan_mult produces same results with NaN data."""
        X = sample_data_with_nan
        y = np.random.randn(X.shape[1], 1)
        nan_mask = np.isnan(X)

        np_result = _nan_mult(X, y, nan_mask, use_denom=True)
        jax_result = _nan_mult_jax(jnp.array(X), jnp.array(y), jnp.array(nan_mask), use_denom=True)

        np.testing.assert_allclose(np_result, np.array(jax_result), rtol=1e-5, atol=1e-8)

    def test_no_denom(self, sample_data_clean):
        """Test _nan_mult with use_denom=False."""
        X = sample_data_clean
        y = np.random.randn(X.shape[1], 1)
        nan_mask = np.isnan(X)

        np_result = _nan_mult(X, y, nan_mask, use_denom=False)
        jax_result = _nan_mult_jax(jnp.array(X), jnp.array(y), jnp.array(nan_mask), use_denom=False)

        np.testing.assert_allclose(np_result, np.array(jax_result), rtol=1e-5, atol=1e-8)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestNipalsPCAJax:
    """Test JAX PCA implementation against NumPy."""

    def test_fit_clean_data(self, sample_data_clean):
        """Test PCA fit on clean data produces similar results."""
        X = sample_data_clean
        n_components = 5

        # NumPy version
        pca_np = NipalsPCA(n_components=n_components)
        pca_np.fit(X)

        # JAX version
        pca_jax = NipalsPCA_JAX(n_components=n_components)
        pca_jax.fit(X)

        # Compare loadings (may have sign differences)
        for i in range(n_components):
            # Check if loadings are aligned or anti-aligned
            dot_product = np.abs(pca_np.loadings[:, i] @ pca_jax.loadings[:, i])
            assert dot_product > 0.999, f"Loadings differ significantly for component {i}"

    def test_fit_nan_data(self, sample_data_with_nan):
        """Test PCA fit on data with NaNs produces similar results."""
        X = sample_data_with_nan
        n_components = 3

        # NumPy version
        pca_np = NipalsPCA(n_components=n_components)
        pca_np.fit(X)

        # JAX version
        pca_jax = NipalsPCA_JAX(n_components=n_components)
        pca_jax.fit(X)

        # Compare loadings
        for i in range(n_components):
            dot_product = np.abs(pca_np.loadings[:, i] @ pca_jax.loadings[:, i])
            assert dot_product > 0.99, f"Loadings differ significantly for component {i}"

    def test_transform(self, sample_data_clean):
        """Test transform produces similar results."""
        X = sample_data_clean
        n_components = 5

        pca_np = NipalsPCA(n_components=n_components)
        pca_np.fit(X)
        scores_np = pca_np.transform(X)

        pca_jax = NipalsPCA_JAX(n_components=n_components)
        pca_jax.fit(X)
        scores_jax = pca_jax.transform(X)

        # Scores may have sign differences, check absolute correlation
        for i in range(n_components):
            corr = np.abs(np.corrcoef(scores_np[:, i], scores_jax[:, i])[0, 1])
            assert corr > 0.99, f"Scores differ significantly for component {i}"

    def test_inverse_transform(self, sample_data_clean):
        """Test inverse_transform produces similar results."""
        X = sample_data_clean
        n_components = 5

        pca_jax = NipalsPCA_JAX(n_components=n_components)
        pca_jax.fit(X)
        scores = pca_jax.transform(X)
        X_reconstructed = pca_jax.inverse_transform(scores)

        # Check reconstruction error is reasonable
        reconstruction_error = np.mean((X - X_reconstructed) ** 2)
        assert reconstruction_error < 1.0, "Reconstruction error too high"


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestNipalsPLSJax:
    """Test JAX PLS implementation against NumPy."""

    def test_fit_clean_data(self, sample_xy_data):
        """Test PLS fit on clean data produces similar results."""
        X, Y = sample_xy_data
        n_components = 3

        # NumPy version
        pls_np = NipalsPLS(n_components=n_components)
        pls_np.fit(X, Y)

        # JAX version
        pls_jax = NipalsPLS_JAX(n_components=n_components)
        pls_jax.fit(X, Y)

        # Compare X loadings
        for i in range(n_components):
            dot_product = np.abs(pls_np.loadings_x[:, i] @ pls_jax.loadings_x[:, i])
            assert dot_product > 0.99, f"X loadings differ for component {i}"

    def test_predict(self, sample_xy_data):
        """Test predict produces similar results."""
        X, Y = sample_xy_data
        n_components = 3

        pls_np = NipalsPLS(n_components=n_components)
        pls_np.fit(X, Y)
        y_pred_np = pls_np.predict(X)

        pls_jax = NipalsPLS_JAX(n_components=n_components)
        pls_jax.fit(X, Y)
        y_pred_jax = pls_jax.predict(X)

        # Compare predictions
        for i in range(Y.shape[1]):
            corr = np.corrcoef(y_pred_np[:, i], y_pred_jax[:, i])[0, 1]
            assert corr > 0.99, f"Predictions differ for Y column {i}"


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
