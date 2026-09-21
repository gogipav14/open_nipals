"""
Tests for SIMCA metrics (R², Q², PRESS).
"""

import numpy as np
import pytest

from open_nipals.simca import (
    calc_r2_x,
    calc_r2_y,
    calc_r2_cumulative_pca,
    calc_r2_cumulative_pls,
    calc_press,
    calc_q2,
    calc_q2_cumulative_pca,
    KFoldCV,
    cross_val_predict_pca,
)
from open_nipals.nipalsPCA import NipalsPCA
from open_nipals.nipalsPLS import NipalsPLS


@pytest.fixture
def sample_pca_data():
    """Generate sample data for PCA tests."""
    np.random.seed(42)
    n, m = 100, 10
    X = np.random.randn(n, m)
    X = X - X.mean(axis=0)
    return X


@pytest.fixture
def sample_pls_data():
    """Generate sample data for PLS tests."""
    np.random.seed(42)
    n, m_x, m_y = 100, 10, 3
    X = np.random.randn(n, m_x)
    true_weights = np.random.randn(m_x, m_y)
    Y = X @ true_weights + 0.1 * np.random.randn(n, m_y)
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    return X, Y


class TestR2X:
    """Tests for R² X-block calculations."""

    def test_perfect_reconstruction(self, sample_pca_data):
        """R² should be 1 for perfect reconstruction."""
        X = sample_pca_data
        r2 = calc_r2_x(X, X)
        assert r2 == pytest.approx(1.0, abs=1e-10)

    def test_zero_reconstruction(self, sample_pca_data):
        """R² should be 0 for zero reconstruction (if data is centered)."""
        X = sample_pca_data
        X_zeros = np.zeros_like(X)
        r2 = calc_r2_x(X, X_zeros)
        assert r2 == pytest.approx(0.0, abs=1e-10)

    def test_r2_between_zero_and_one(self, sample_pca_data):
        """R² should be between 0 and 1 for partial reconstruction."""
        X = sample_pca_data
        pca = NipalsPCA(n_components=3)
        pca.fit(X)
        X_reconstructed = pca.inverse_transform(pca.transform(X))

        r2 = calc_r2_x(X, X_reconstructed)
        assert 0 < r2 < 1

    def test_per_variable(self, sample_pca_data):
        """Test per-variable R² calculation."""
        X = sample_pca_data
        pca = NipalsPCA(n_components=3)
        pca.fit(X)
        X_reconstructed = pca.inverse_transform(pca.transform(X))

        r2_per_var = calc_r2_x(X, X_reconstructed, per_variable=True)
        assert len(r2_per_var) == X.shape[1]
        assert all(0 <= r <= 1 for r in r2_per_var)


class TestR2Y:
    """Tests for R² Y-block calculations."""

    def test_perfect_prediction(self, sample_pls_data):
        """R² should be 1 for perfect prediction."""
        X, Y = sample_pls_data
        r2 = calc_r2_y(Y, Y)
        assert r2 == pytest.approx(1.0, abs=1e-10)

    def test_zero_prediction(self, sample_pls_data):
        """R² should be 0 for zero prediction (if Y is centered)."""
        X, Y = sample_pls_data
        Y_zeros = np.zeros_like(Y)
        r2 = calc_r2_y(Y, Y_zeros)
        assert r2 == pytest.approx(0.0, abs=1e-10)

    def test_per_variable(self, sample_pls_data):
        """Test per-variable R² for Y-block."""
        X, Y = sample_pls_data
        pls = NipalsPLS(n_components=3)
        pls.fit(X, Y)
        Y_pred = pls.predict(X)

        r2_per_var = calc_r2_y(Y, Y_pred, per_variable=True)
        assert len(r2_per_var) == Y.shape[1]


class TestR2Cumulative:
    """Tests for cumulative R² calculations."""

    def test_r2_cumulative_pca_increasing(self, sample_pca_data):
        """Cumulative R² should be monotonically increasing."""
        X = sample_pca_data
        pca = NipalsPCA(n_components=5)
        pca.fit(X)

        r2_cum = calc_r2_cumulative_pca(pca, X)
        assert len(r2_cum) == 5
        for i in range(1, len(r2_cum)):
            assert r2_cum[i] >= r2_cum[i - 1] - 1e-10

    def test_r2_cumulative_pls_increasing(self, sample_pls_data):
        """Cumulative R² for PLS should be monotonically increasing."""
        X, Y = sample_pls_data
        pls = NipalsPLS(n_components=3)
        pls.fit(X, Y)

        r2_cum = calc_r2_cumulative_pls(pls, X, Y)
        assert len(r2_cum) == 3
        for i in range(1, len(r2_cum)):
            assert r2_cum[i] >= r2_cum[i - 1] - 1e-10


class TestPRESS:
    """Tests for PRESS calculations."""

    def test_press_zero_for_perfect(self, sample_pca_data):
        """PRESS should be 0 for perfect predictions."""
        X = sample_pca_data
        press = calc_press(X, X)
        assert press == pytest.approx(0.0, abs=1e-10)

    def test_press_positive(self, sample_pca_data):
        """PRESS should be positive for imperfect predictions."""
        X = sample_pca_data
        X_noisy = X + 0.1 * np.random.randn(*X.shape)
        press = calc_press(X, X_noisy)
        assert press > 0

    def test_press_per_variable(self, sample_pca_data):
        """Test per-variable PRESS calculation."""
        X = sample_pca_data
        X_noisy = X + 0.1 * np.random.randn(*X.shape)
        press_per_var = calc_press(X, X_noisy, per_variable=True)
        assert len(press_per_var) == X.shape[1]
        assert all(p >= 0 for p in press_per_var)


class TestQ2:
    """Tests for Q² calculations."""

    def test_q2_perfect(self, sample_pca_data):
        """Q² should be 1 for perfect CV predictions."""
        X = sample_pca_data
        q2 = calc_q2(X, X)
        assert q2 == pytest.approx(1.0, abs=1e-10)

    def test_q2_zero(self, sample_pca_data):
        """Q² should be 0 for zero CV predictions."""
        X = sample_pca_data
        X_zeros = np.zeros_like(X)
        q2 = calc_q2(X, X_zeros)
        assert q2 == pytest.approx(0.0, abs=1e-10)

    def test_q2_cumulative_pca(self, sample_pca_data):
        """Test cumulative Q² for PCA."""
        X = sample_pca_data
        cv = KFoldCV(n_splits=5)

        q2_cum = calc_q2_cumulative_pca(NipalsPCA, X, cv, max_components=3)
        assert len(q2_cum) == 3


class TestCrossValPredict:
    """Tests for cross-validation prediction functions."""

    def test_cross_val_predict_pca(self, sample_pca_data):
        """Test cross-validated PCA predictions."""
        X = sample_pca_data
        cv = KFoldCV(n_splits=5)

        X_pred = cross_val_predict_pca(NipalsPCA, X, n_components=3, cv=cv)
        assert X_pred.shape == X.shape


class TestWithNaN:
    """Tests with NaN values."""

    def test_r2_x_with_nan(self, sample_pca_data):
        """R² should handle NaN values."""
        X = sample_pca_data.copy()
        X[0, 0] = np.nan
        X[5, 3] = np.nan

        pca = NipalsPCA(n_components=3)
        pca.fit(X)
        X_reconstructed = pca.inverse_transform(pca.transform(X))

        r2 = calc_r2_x(X, X_reconstructed)
        assert 0 <= r2 <= 1

    def test_press_with_nan(self, sample_pca_data):
        """PRESS should handle NaN values."""
        X = sample_pca_data.copy()
        X[0, 0] = np.nan

        X_pred = X.copy() + 0.1
        press = calc_press(X, X_pred)
        assert press >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
