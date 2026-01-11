"""
Tests for JAX SIMCA implementation.

Verifies that JAX SIMCA produces results equivalent to NumPy version.
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
    from open_nipals.jax.simca import SIMCA as SIMCA_JAX
    from open_nipals.jax.simca import (
        calc_r2_x as calc_r2_x_jax,
        calc_r2_y as calc_r2_y_jax,
        calc_r2_cumulative_pca as calc_r2_cumulative_pca_jax,
        calc_press as calc_press_jax,
        calc_q2 as calc_q2_jax,
        KFoldCV,
    )
    from open_nipals.jax import NipalsPCA as NipalsPCA_JAX

from open_nipals.simca import SIMCA
from open_nipals.simca import (
    calc_r2_x,
    calc_r2_y,
    calc_press,
    calc_q2,
)


@pytest.fixture
def two_class_data():
    """Generate two-class classification data."""
    np.random.seed(42)
    n_per_class = 50
    n_features = 10

    X0 = np.random.randn(n_per_class, n_features) * 0.5
    y0 = np.zeros(n_per_class)

    X1 = np.random.randn(n_per_class, n_features) * 0.5 + 2.0
    y1 = np.ones(n_per_class)

    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])

    return X, y


@pytest.fixture
def sample_pca_data():
    """Generate sample PCA data."""
    np.random.seed(42)
    n, m = 100, 10
    X = np.random.randn(n, m)
    X = X - X.mean(axis=0)
    return X


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestJAXSIMCABasic:
    """Basic JAX SIMCA functionality tests."""

    def test_fit(self, two_class_data):
        """Test JAX SIMCA fitting."""
        X, y = two_class_data
        model = SIMCA_JAX(n_components=2, scale=True)
        model.fit(X, y)

        assert model.classes_ is not None
        assert len(model.classes_) == 2
        assert len(model.class_models_) == 2

    def test_predict(self, two_class_data):
        """Test JAX SIMCA prediction."""
        X, y = two_class_data
        model = SIMCA_JAX(n_components=2, scale=True)
        model.fit(X, y)

        y_pred = model.predict(X)
        assert len(y_pred) == len(y)
        accuracy = np.mean(y_pred == y)
        assert accuracy > 0.8

    def test_predict_proba(self, two_class_data):
        """Test JAX SIMCA probability prediction."""
        X, y = two_class_data
        model = SIMCA_JAX(n_components=2, scale=True)
        model.fit(X, y)

        proba = model.predict_proba(X)
        assert proba.shape == (len(y), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestJAXSIMCAvsNumPy:
    """Compare JAX SIMCA with NumPy version."""

    def test_same_predictions(self, two_class_data):
        """JAX and NumPy SIMCA should produce similar predictions."""
        X, y = two_class_data

        # NumPy SIMCA
        model_np = SIMCA(n_components=2, scale=True, unknown_handling="closest")
        model_np.fit(X, y)
        y_pred_np = model_np.predict(X)

        # JAX SIMCA
        model_jax = SIMCA_JAX(n_components=2, scale=True, unknown_handling="closest")
        model_jax.fit(X, y)
        y_pred_jax = model_jax.predict(X)

        # Predictions should mostly match
        agreement = np.mean(y_pred_np == y_pred_jax)
        assert agreement > 0.9

    def test_similar_accuracy(self, two_class_data):
        """JAX and NumPy SIMCA should have similar accuracy."""
        X, y = two_class_data

        model_np = SIMCA(n_components=2, scale=True)
        model_np.fit(X, y)
        acc_np = model_np.score(X, y)

        model_jax = SIMCA_JAX(n_components=2, scale=True)
        model_jax.fit(X, y)
        acc_jax = model_jax.score(X, y)

        # Accuracies should be within 5%
        assert abs(acc_np - acc_jax) < 0.05


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestJAXMetrics:
    """Test JAX metrics implementations."""

    def test_r2_x_matches_numpy(self, sample_pca_data):
        """JAX R² should match NumPy R²."""
        X = sample_pca_data
        X_noisy = X + 0.1 * np.random.randn(*X.shape)

        r2_np = calc_r2_x(X, X_noisy)
        r2_jax = calc_r2_x_jax(jnp.array(X), jnp.array(X_noisy))

        np.testing.assert_allclose(r2_np, r2_jax, rtol=1e-5)

    def test_r2_y_matches_numpy(self, sample_pca_data):
        """JAX R² Y should match NumPy."""
        Y = sample_pca_data[:, :3]
        Y_pred = Y + 0.1 * np.random.randn(*Y.shape)

        r2_np = calc_r2_y(Y, Y_pred)
        r2_jax = calc_r2_y_jax(jnp.array(Y), jnp.array(Y_pred))

        np.testing.assert_allclose(r2_np, r2_jax, rtol=1e-5)

    def test_press_matches_numpy(self, sample_pca_data):
        """JAX PRESS should match NumPy."""
        X = sample_pca_data
        X_pred = X + 0.1 * np.random.randn(*X.shape)

        press_np = calc_press(X, X_pred)
        press_jax = calc_press_jax(jnp.array(X), jnp.array(X_pred))

        np.testing.assert_allclose(press_np, press_jax, rtol=1e-5)

    def test_q2_matches_numpy(self, sample_pca_data):
        """JAX Q² should match NumPy."""
        X = sample_pca_data
        X_pred = X + 0.1 * np.random.randn(*X.shape)

        q2_np = calc_q2(X, X_pred)
        q2_jax = calc_q2_jax(jnp.array(X), jnp.array(X_pred))

        np.testing.assert_allclose(q2_np, q2_jax, rtol=1e-5)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestJAXSIMCAAuto:
    """Test automatic component selection in JAX SIMCA."""

    def test_auto_r2(self, two_class_data):
        """Test automatic R² component selection."""
        X, y = two_class_data
        model = SIMCA_JAX(
            n_components="auto",
            component_selection="r2",
            r2_threshold=0.8,
            scale=True,
        )
        model.fit(X, y)

        for class_model in model.class_models_.values():
            assert class_model.n_components >= 1

    def test_auto_eigenvalue(self, two_class_data):
        """Test automatic eigenvalue component selection."""
        X, y = two_class_data
        model = SIMCA_JAX(
            n_components="auto",
            component_selection="eigenvalue",
            scale=True,
        )
        model.fit(X, y)

        for class_model in model.class_models_.values():
            assert class_model.n_components >= 1


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestJAXSIMCAScaling:
    """Test JAX SIMCA scaling behavior."""

    def test_scale_raw_data(self, two_class_data):
        """Test scaling on raw data."""
        X, y = two_class_data
        model = SIMCA_JAX(n_components=2, scale=True)
        model.fit(X, y)

        assert model.scaler_ is not None

    def test_skip_prescaled(self, two_class_data):
        """Test skipping scaling for pre-scaled data."""
        X, y = two_class_data
        X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)

        with pytest.warns(UserWarning, match="pre-scaled"):
            model = SIMCA_JAX(n_components=2, scale=True)
            model.fit(X_scaled, y)

        assert model.scaler_ is None


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestJAXSIMCAUnknown:
    """Test JAX SIMCA unknown handling."""

    def test_unknown_closest(self, two_class_data):
        """Test unknown_handling='closest'."""
        X, y = two_class_data
        model = SIMCA_JAX(n_components=2, scale=True, unknown_handling="closest")
        model.fit(X, y)

        outlier = np.random.randn(1, X.shape[1]) * 10
        pred = model.predict(outlier)

        assert pred[0] in model.classes_

    def test_unknown_reject(self, two_class_data):
        """Test unknown_handling='reject'."""
        X, y = two_class_data
        model = SIMCA_JAX(n_components=2, scale=True, unknown_handling="reject", alpha=0.99)
        model.fit(X, y)

        outlier = np.ones((1, X.shape[1])) * 100
        pred = model.predict(outlier)

        assert pred[0] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
