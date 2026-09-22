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

    # open_nipals.jax needs 64-bit mode; request it before importing
    jax.config.update("jax_enable_x64", True)
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
        model_np = SIMCA(
            n_components=2, scale=True, unknown_handling="closest"
        )
        model_np.fit(X, y)
        y_pred_np = model_np.predict(X)

        # JAX SIMCA
        model_jax = SIMCA_JAX(
            n_components=2, scale=True, unknown_handling="closest"
        )
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
        """Each class is autoscaled on its own training rows."""
        X, y = two_class_data
        model = SIMCA_JAX(n_components=2, scale=True).fit(X, y)

        for label, class_model in model.class_models_.items():
            np.testing.assert_allclose(
                class_model.class_std, X[y == label].std(axis=0, ddof=1)
            )


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestJAXSIMCAUnknown:
    """Test JAX SIMCA unknown handling."""

    def test_unknown_closest(self, two_class_data):
        """Test unknown_handling='closest'."""
        X, y = two_class_data
        model = SIMCA_JAX(
            n_components=2, scale=True, unknown_handling="closest"
        )
        model.fit(X, y)

        outlier = np.random.randn(1, X.shape[1]) * 10
        pred = model.predict(outlier)

        assert pred[0] in model.classes_

    def test_unknown_reject(self, two_class_data):
        """Test unknown_handling='reject'."""
        X, y = two_class_data
        model = SIMCA_JAX(
            n_components=2, scale=True, unknown_handling="reject", alpha=0.99
        )
        model.fit(X, y)

        outlier = np.ones((1, X.shape[1])) * 100
        pred = model.predict(outlier)

        assert pred[0] is None


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestJAXSIMCAReworkParity:
    """The JAX wrapper must inherit the corrected SIMCA behaviour."""

    def test_class_models_are_centred(self, two_class_data):
        """Finding 2: class means are removed before fitting."""
        X, y = two_class_data
        model = SIMCA_JAX(n_components=2, scale=True).fit(X, y)

        for label, class_model in model.class_models_.items():
            expected = X[y == label].mean(axis=0)
            np.testing.assert_allclose(
                class_model.class_mean, expected, atol=1e-12
            )
            fit_data = np.asarray(class_model.pca_model.fit_data)
            assert np.abs(fit_data.mean(axis=0)).max() < 1e-10

    def test_distances_match_numpy(self, two_class_data):
        """T² and normalised DModX agree with the NumPy reference."""
        X, y = two_class_data

        model_np = SIMCA(n_components=2, scale=True).fit(X, y)
        model_jax = SIMCA_JAX(n_components=2, scale=True).fit(X, y)

        d_np = model_np.get_distances(X)
        d_jax = model_jax.get_distances(X)

        np.testing.assert_allclose(d_np["t2"], d_jax["t2"], rtol=1e-6)
        np.testing.assert_allclose(d_np["dmodx"], d_jax["dmodx"], rtol=1e-6)
        np.testing.assert_allclose(
            d_np["dmodx_limits"], d_jax["dmodx_limits"], rtol=1e-12
        )

    def test_acceptance_is_scale_invariant(self):
        """Finding 3: normalised DModX does not care about units."""
        rng = np.random.default_rng(0)
        X = np.vstack(
            [
                rng.standard_normal((60, 6)),
                rng.standard_normal((60, 6)) + 5.0,
            ]
        )
        X = X - X.mean(axis=0)
        y = np.array([0] * 60 + [1] * 60)

        base = SIMCA_JAX(n_components=2, scale=False).fit(X, y)
        scaled = SIMCA_JAX(n_components=2, scale=False).fit(X * 10.0, y)

        d_base = base.get_distances(X)
        d_scaled = scaled.get_distances(X * 10.0)

        np.testing.assert_allclose(
            d_base["dmodx"], d_scaled["dmodx"], rtol=1e-7
        )
        np.testing.assert_array_equal(
            d_base["dmodx"] <= d_base["dmodx_limits"],
            d_scaled["dmodx"] <= d_scaled["dmodx_limits"],
        )

    def test_predict_agrees_with_membership(self, two_class_data):
        """Finding 1: predict and membership see the same scaling."""
        X, y = two_class_data
        model = SIMCA_JAX(
            n_components=2, scale=True, unknown_handling="reject"
        ).fit(X, y)

        membership = model.get_class_membership(X)
        y_pred = model.predict(X)

        members = 0
        for i, classes in enumerate(membership["member_of"]):
            if len(classes) == 0:
                assert y_pred[i] is None
            elif len(classes) == 1:
                members += 1
                assert y_pred[i] == classes[0]

        assert members > 0, "fixture no longer exercises membership"

    def test_auto_q2_matches_numpy(self, two_class_data):
        """Finding 4: Q² selection runs and agrees with NumPy."""
        X, y = two_class_data

        model_np = SIMCA(
            n_components="auto",
            component_selection="q2",
            q2_cv_folds=5,
            scale=True,
        ).fit(X, y)
        model_jax = SIMCA_JAX(
            n_components="auto",
            component_selection="q2",
            q2_cv_folds=5,
            scale=True,
        ).fit(X, y)

        for label in model_np.classes_:
            assert (
                model_jax.class_models_[label].n_components
                == model_np.class_models_[label].n_components
            )

    def test_rejects_unusable_component_counts(self, two_class_data):
        """Finding 5: the validation is inherited too."""
        X, y = two_class_data
        with pytest.raises(ValueError, match="residual degrees of freedom"):
            SIMCA_JAX(n_components=X.shape[1], scale=True).fit(X, y)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestJAXPRESSFailedPredictions:
    """Finding 6: the JAX PRESS must not reward failed predictions."""

    def test_press_infinite_when_all_predictions_fail(self, sample_pca_data):
        """All-NaN predictions give infinite PRESS in both backends."""
        X = sample_pca_data
        X_pred = np.full_like(X, np.nan)

        assert calc_press_jax(jnp.array(X), jnp.array(X_pred)) == np.inf
        assert calc_q2_jax(jnp.array(X), jnp.array(X_pred)) == -np.inf
        assert calc_press(X, X_pred) == calc_press_jax(
            jnp.array(X), jnp.array(X_pred)
        )

    def test_missing_targets_are_still_masked(self, sample_pca_data):
        """A genuinely missing observation contributes nothing."""
        X = sample_pca_data.copy()
        X[0, 0] = np.nan
        X_pred = X + 0.1

        press_jax = calc_press_jax(jnp.array(X), jnp.array(X_pred))

        assert np.isfinite(press_jax)
        np.testing.assert_allclose(press_jax, calc_press(X, X_pred))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestJAXSIMCAReviewRound3:
    """The JAX wrapper inherits the third-review fixes."""

    def test_constant_decimal_column_is_not_scaled_up(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 4))
        X[:, 1] = 0.1
        y = np.array([0] * 10 + [1] * 10)

        model = SIMCA_JAX(n_components=2, scale=True).fit(X, y)

        for class_model in model.class_models_.values():
            assert class_model.class_std[1] == 1.0
            assert np.abs(class_model.pca_model.loadings[1, :]).max() < 1e-8

    def test_wrong_feature_count_is_rejected(self, two_class_data):
        X, y = two_class_data
        model = SIMCA_JAX(n_components=2).fit(X, y)
        with pytest.raises(ValueError, match="features"):
            model.predict(X[:, :1])

    def test_component_count_outside_limit_domain_is_rejected(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(20, 10))
        y = np.zeros(20, dtype=int)
        with pytest.raises(ValueError, match="no DModX limit"):
            SIMCA_JAX(n_components=8, scale=False).fit(X, y)
        model = SIMCA_JAX(
            n_components="auto", component_selection="r2", r2_threshold=1.0
        ).fit(X, y)
        assert np.isfinite(model.class_models_[0].dmodx_limit)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestJAXSIMCAReviewRound4:
    """The JAX helpers inherit the fourth-review fixes."""

    def test_cross_val_predict_integer_input(self):
        from open_nipals.jax.simca.cross_validation import (
            cross_val_predict_pca,
        )
        from open_nipals.jax import NipalsPCA as NipalsPCA_JAX

        rng = np.random.default_rng(0)
        X_int = rng.integers(-5, 6, size=(40, 5))
        cv = KFoldCV(n_splits=4)

        pred_int = cross_val_predict_pca(NipalsPCA_JAX, X_int, 2, cv)
        pred_float = cross_val_predict_pca(
            NipalsPCA_JAX, X_int.astype(float), 2, cv
        )
        np.testing.assert_allclose(pred_int, pred_float)

    def test_predict_keeps_label_dtype(self, two_class_data):
        X, y = two_class_data
        model = SIMCA_JAX(n_components=2).fit(X, y)
        assert model.predict(X).dtype == np.asarray(y).dtype


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestJAXSIMCAReviewRound5:
    """The JAX classifier and metrics inherit the fifth-review fixes."""

    def test_is_a_sklearn_classifier(self):
        from sklearn.base import is_classifier

        assert is_classifier(SIMCA_JAX())

    def test_r2_failed_prediction_is_minus_inf(self):
        from open_nipals.jax.simca.metrics import calc_r2_x

        X = np.array([[1.0, -1.0], [-1.0, 1.0]])
        X_rec = X.copy()
        X_rec[0, 0] = np.nan
        assert calc_r2_x(X, X_rec) == -np.inf

    def test_cumulative_r2_after_reducing_components(self, two_class_data):
        from open_nipals.jax import NipalsPCA as NipalsPCA_JAX
        from open_nipals.jax.simca.metrics import calc_r2_cumulative_pca

        X, _ = two_class_data
        X = X - X.mean(axis=0)
        pca = NipalsPCA_JAX(n_components=3).fit(X)
        expected = calc_r2_cumulative_pca(pca, X)
        pca.set_components(1)
        np.testing.assert_allclose(calc_r2_cumulative_pca(pca, X), expected)
        assert pca.n_components == 1


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestJAXSIMCAReviewRound6:
    def test_constant_feature_with_nan_first(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 6))
        X[:, 0] = 2.5
        X[::5, 0] = np.nan
        y = np.array([0] * 20 + [1] * 20)

        model = SIMCA_JAX(n_components=2).fit(X, y)

        assert np.all(np.isfinite(model.get_distances(X)["dmodx"]))

    def test_all_nan_training_rows_are_dropped(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 5))
        X_padded = np.vstack([X, np.full((10, 5), np.nan)])
        with pytest.warns(UserWarning, match="Dropping 10"):
            model = SIMCA_JAX(n_components=2).fit(
                X_padded, np.zeros(50, dtype=int)
            )
        assert model.class_models_[0].n_samples == 40
