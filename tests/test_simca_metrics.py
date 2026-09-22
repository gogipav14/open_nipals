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
    ComponentSelector,
    KFoldCV,
    cross_val_predict_pca,
    cross_val_press_pca,
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


@pytest.fixture
def noise_data():
    """200 x 5 independent Gaussian noise, centred."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 5))
    return X - X.mean(axis=0)


@pytest.fixture
def rank_two_data():
    """200 x 8 rank-2 signal with a little noise, centred."""
    rng = np.random.default_rng(1)
    scores = rng.standard_normal((200, 2))
    loadings = rng.standard_normal((2, 8))
    X = scores @ loadings + 0.05 * rng.standard_normal((200, 8))
    return X - X.mean(axis=0)


class TestElementWiseQ2:
    """Finding 4: an element must never predict itself."""

    def test_q2_of_noise_is_not_positive(self, noise_data):
        """Pure noise cannot be predicted, at any component count."""
        cv = KFoldCV(n_splits=7)
        q2 = calc_q2_cumulative_pca(NipalsPCA, noise_data, cv, 4)

        assert len(q2) == 4
        assert np.all(q2 <= 0.05)

    def test_q2_of_noise_does_not_grow(self, noise_data):
        """Q² must not climb towards 1 as components are added."""
        cv = KFoldCV(n_splits=7)
        q2 = calc_q2_cumulative_pca(NipalsPCA, noise_data, cv, 4)

        assert q2[-1] < q2[0] + 0.05

    def test_selection_on_noise_picks_at_most_one(self, noise_data):
        """Automatic selection must not chase noise."""
        n_comp = ComponentSelector.select_by_q2(
            NipalsPCA, noise_data, max_components=4, cv_folds=7
        )
        assert n_comp <= 1

    def test_q2_peaks_at_the_true_rank(self, rank_two_data):
        """A rank-2 signal peaks at two components."""
        cv = KFoldCV(n_splits=7)
        q2 = calc_q2_cumulative_pca(NipalsPCA, rank_two_data, cv, 5)

        assert int(np.argmax(q2)) + 1 == 2
        assert q2[1] > 0.9

    def test_selection_on_rank_two_signal(self, rank_two_data):
        """Automatic selection recovers the true rank."""
        n_comp = ComponentSelector.select_by_q2(
            NipalsPCA, rank_two_data, max_components=5, cv_folds=7
        )
        assert n_comp == 2

    def test_press_is_fold_preprocessed(self, noise_data):
        """A constant offset cannot make the data look predictable."""
        cv = KFoldCV(n_splits=6)

        q2_centred = calc_q2_cumulative_pca(NipalsPCA, noise_data, cv, 3)
        q2_offset = calc_q2_cumulative_pca(
            NipalsPCA, noise_data + 100.0, cv, 3
        )

        np.testing.assert_allclose(q2_centred, q2_offset, atol=1e-8)

    def test_press_and_ss_total_are_finite(self, rank_two_data):
        """cross_val_press_pca reports both PRESS and its reference."""
        cv = KFoldCV(n_splits=5)
        press, ss_total = cross_val_press_pca(
            NipalsPCA, rank_two_data, 2, cv
        )

        assert np.isfinite(press) and press > 0
        assert np.isfinite(ss_total) and ss_total > 0
        assert press < ss_total

    def test_raises_on_too_small_training_folds(self, noise_data):
        """Folds that cannot carry the model are an error, not a score."""
        cv = KFoldCV(n_splits=2)
        with pytest.raises(ValueError, match="too few"):
            cross_val_press_pca(NipalsPCA, noise_data[:6], 4, cv)


class TestPRESSFailedPredictions:
    """Finding 6: failed predictions must not earn Q² credit."""

    def test_press_infinite_when_all_predictions_fail(
        self, sample_pca_data
    ):
        """All-NaN predictions are a failure, not missing data."""
        X = sample_pca_data
        X_pred = np.full_like(X, np.nan)

        assert calc_press(X, X_pred) == np.inf
        assert calc_q2(X, X_pred) == -np.inf

    def test_press_infinite_for_a_single_failure(self, sample_pca_data):
        """One non-finite prediction is enough to poison PRESS."""
        X = sample_pca_data
        X_pred = X.copy()
        X_pred[3, 2] = np.inf

        assert calc_press(X, X_pred) == np.inf

    def test_press_per_variable_marks_failed_columns(
        self, sample_pca_data
    ):
        """Only the column that failed becomes infinite."""
        X = sample_pca_data
        X_pred = X.copy()
        X_pred[0, 1] = np.nan

        press = calc_press(X, X_pred, per_variable=True)

        assert np.isinf(press[1])
        assert np.all(np.isfinite(np.delete(press, 1)))

    def test_missing_targets_are_still_masked(self, sample_pca_data):
        """A genuinely missing observation contributes nothing."""
        X = sample_pca_data.copy()
        X[0, 0] = np.nan
        X_pred = X + 0.1

        press = calc_press(X, X_pred)

        assert np.isfinite(press)
        assert press == pytest.approx(0.01 * (X.size - 1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
