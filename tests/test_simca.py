"""
Tests for SIMCA classifier implementation.
"""

import numpy as np
import pytest

from open_nipals.simca import (
    SIMCA,
    SIMCAClass,
    ComponentSelector,
    KFoldCV,
    LeaveOneOutCV,
    VenetianBlindsCV,
)
from open_nipals.nipalsPCA import NipalsPCA


@pytest.fixture
def two_class_data():
    """Generate two-class classification data."""
    np.random.seed(42)
    n_per_class = 50
    n_features = 10

    # Class 0: centered at origin
    X0 = np.random.randn(n_per_class, n_features) * 0.5
    y0 = np.zeros(n_per_class)

    # Class 1: shifted
    X1 = np.random.randn(n_per_class, n_features) * 0.5 + 2.0
    y1 = np.ones(n_per_class)

    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])

    return X, y


@pytest.fixture
def three_class_data():
    """Generate three-class classification data."""
    np.random.seed(42)
    n_per_class = 40
    n_features = 10

    # Class 0
    X0 = np.random.randn(n_per_class, n_features) * 0.5
    y0 = np.zeros(n_per_class)

    # Class 1
    X1 = np.random.randn(n_per_class, n_features) * 0.5 + np.array([2, 0] + [0] * 8)
    y1 = np.ones(n_per_class)

    # Class 2
    X2 = np.random.randn(n_per_class, n_features) * 0.5 + np.array([0, 2] + [0] * 8)
    y2 = np.full(n_per_class, 2)

    X = np.vstack([X0, X1, X2])
    y = np.concatenate([y0, y1, y2])

    return X, y


@pytest.fixture
def five_class_data():
    """Generate five-class classification data."""
    np.random.seed(42)
    n_per_class = 30
    n_features = 15

    X_all = []
    y_all = []

    for class_idx in range(5):
        shift = np.zeros(n_features)
        shift[class_idx % n_features] = 3.0
        X_class = np.random.randn(n_per_class, n_features) * 0.5 + shift
        y_class = np.full(n_per_class, class_idx)
        X_all.append(X_class)
        y_all.append(y_class)

    X = np.vstack(X_all)
    y = np.concatenate(y_all)

    return X, y


class TestSIMCABasic:
    """Basic SIMCA functionality tests."""

    def test_fit_two_class(self, two_class_data):
        """Test fitting on two-class data."""
        X, y = two_class_data
        model = SIMCA(n_components=2, scale=True)
        model.fit(X, y)

        assert model.classes_ is not None
        assert len(model.classes_) == 2
        assert len(model.class_models_) == 2

    def test_fit_three_class(self, three_class_data):
        """Test fitting on three-class data."""
        X, y = three_class_data
        model = SIMCA(n_components=2, scale=True)
        model.fit(X, y)

        assert len(model.classes_) == 3
        assert len(model.class_models_) == 3

    def test_fit_five_class(self, five_class_data):
        """Test fitting on five-class data."""
        X, y = five_class_data
        model = SIMCA(n_components=2, scale=True)
        model.fit(X, y)

        assert len(model.classes_) == 5
        assert len(model.class_models_) == 5

    def test_predict(self, two_class_data):
        """Test prediction."""
        X, y = two_class_data
        model = SIMCA(n_components=2, scale=True)
        model.fit(X, y)

        y_pred = model.predict(X)
        assert len(y_pred) == len(y)
        # Should have reasonable accuracy on training data
        accuracy = np.mean(y_pred == y)
        assert accuracy > 0.8

    def test_predict_proba(self, two_class_data):
        """Test probability prediction."""
        X, y = two_class_data
        model = SIMCA(n_components=2, scale=True)
        model.fit(X, y)

        proba = model.predict_proba(X)
        assert proba.shape == (len(y), 2)
        # Probabilities should sum to 1
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

    def test_score(self, two_class_data):
        """Test accuracy score."""
        X, y = two_class_data
        model = SIMCA(n_components=2, scale=True)
        model.fit(X, y)

        score = model.score(X, y)
        assert 0 <= score <= 1
        assert score > 0.8  # Training accuracy should be high


class TestSIMCAScaling:
    """Tests for SIMCA scaling behavior."""

    def test_scale_raw_data(self, two_class_data):
        """Test that scaling is applied to raw data."""
        X, y = two_class_data
        model = SIMCA(n_components=2, scale=True)
        model.fit(X, y)

        assert model.scaler_ is not None

    def test_skip_scaling_prescaled(self, two_class_data):
        """Test that scaling is skipped for pre-scaled data."""
        X, y = two_class_data
        # Pre-scale the data
        X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)

        with pytest.warns(UserWarning, match="pre-scaled"):
            model = SIMCA(n_components=2, scale=True)
            model.fit(X_scaled, y)

        assert model.scaler_ is None

    def test_no_scale(self, two_class_data):
        """Test with scaling disabled."""
        X, y = two_class_data
        model = SIMCA(n_components=2, scale=False)
        model.fit(X, y)

        assert model.scaler_ is None


class TestSIMCAUnknownHandling:
    """Tests for unknown sample handling."""

    def test_unknown_handling_closest(self, two_class_data):
        """Test unknown_handling='closest' assigns to nearest class."""
        X, y = two_class_data
        model = SIMCA(n_components=2, scale=True, unknown_handling="closest")
        model.fit(X, y)

        # Create outlier sample
        outlier = np.random.randn(1, X.shape[1]) * 10
        pred = model.predict(outlier)

        # Should still assign to a class
        assert pred[0] in model.classes_

    def test_unknown_handling_reject(self, two_class_data):
        """Test unknown_handling='reject' returns None for outliers."""
        X, y = two_class_data
        model = SIMCA(n_components=2, scale=True, unknown_handling="reject", alpha=0.99)
        model.fit(X, y)

        # Create extreme outlier
        outlier = np.ones((1, X.shape[1])) * 100
        pred = model.predict(outlier)

        # Should reject (return None)
        assert pred[0] is None


class TestSIMCAClassMembership:
    """Tests for class membership functionality."""

    def test_get_class_membership(self, two_class_data):
        """Test get_class_membership returns correct structure."""
        X, y = two_class_data
        model = SIMCA(n_components=2, scale=True)
        model.fit(X, y)

        membership = model.get_class_membership(X[:5])

        assert "t2_in" in membership
        assert "dmodx_in" in membership
        assert "both_in" in membership
        assert "member_of" in membership
        assert len(membership["member_of"]) == 5


class TestSIMCADistances:
    """Tests for distance calculations."""

    def test_get_distances(self, two_class_data):
        """Test get_distances returns correct structure."""
        X, y = two_class_data
        model = SIMCA(n_components=2, scale=True)
        model.fit(X, y)

        distances = model.get_distances(X[:5])

        assert "t2" in distances
        assert "dmodx" in distances
        assert "t2_limits" in distances
        assert "dmodx_limits" in distances
        assert distances["t2"].shape == (5, 2)
        assert distances["dmodx"].shape == (5, 2)


class TestComponentSelector:
    """Tests for automatic component selection."""

    def test_select_by_r2(self, two_class_data):
        """Test R² component selection."""
        X, y = two_class_data
        X_class = X[y == 0]

        pca = NipalsPCA(n_components=5)
        pca.fit(X_class)

        n_comp = ComponentSelector.select_by_r2(pca, X_class, threshold=0.8)
        assert 1 <= n_comp <= 5

    def test_select_by_eigenvalue(self, two_class_data):
        """Test eigenvalue component selection."""
        X, y = two_class_data
        X_class = X[y == 0]

        pca = NipalsPCA(n_components=5)
        pca.fit(X_class)

        n_comp = ComponentSelector.select_by_eigenvalue(pca, X_class)
        assert 1 <= n_comp <= 5


class TestSIMCAAuto:
    """Tests for automatic component selection in SIMCA."""

    def test_auto_r2(self, two_class_data):
        """Test automatic component selection with R²."""
        X, y = two_class_data
        model = SIMCA(
            n_components="auto",
            component_selection="r2",
            r2_threshold=0.8,
            scale=True,
        )
        model.fit(X, y)

        for class_label, class_model in model.class_models_.items():
            assert class_model.n_components >= 1

    def test_auto_q2(self, two_class_data):
        """Test automatic component selection with Q²."""
        X, y = two_class_data
        model = SIMCA(
            n_components="auto",
            component_selection="q2",
            q2_cv_folds=5,
            scale=True,
        )
        model.fit(X, y)

        for class_label, class_model in model.class_models_.items():
            assert class_model.n_components >= 1

    def test_auto_eigenvalue(self, two_class_data):
        """Test automatic component selection with eigenvalue."""
        X, y = two_class_data
        model = SIMCA(
            n_components="auto",
            component_selection="eigenvalue",
            scale=True,
        )
        model.fit(X, y)

        for class_label, class_model in model.class_models_.items():
            assert class_model.n_components >= 1


class TestCrossValidation:
    """Tests for cross-validation utilities."""

    def test_kfold_cv(self, two_class_data):
        """Test KFold cross-validation."""
        X, y = two_class_data
        cv = KFoldCV(n_splits=5)

        n_splits = 0
        for train_idx, test_idx in cv.split(X):
            n_splits += 1
            assert len(train_idx) + len(test_idx) == len(X)
            assert len(np.intersect1d(train_idx, test_idx)) == 0

        assert n_splits == 5

    def test_leave_one_out_cv(self, two_class_data):
        """Test Leave-One-Out cross-validation."""
        X, y = two_class_data
        cv = LeaveOneOutCV()

        n_splits = 0
        for train_idx, test_idx in cv.split(X):
            n_splits += 1
            assert len(test_idx) == 1
            assert len(train_idx) == len(X) - 1

        assert n_splits == len(X)

    def test_venetian_blinds_cv(self, two_class_data):
        """Test Venetian Blinds cross-validation."""
        X, y = two_class_data
        cv = VenetianBlindsCV(n_splits=5)

        n_splits = 0
        for train_idx, test_idx in cv.split(X):
            n_splits += 1
            assert len(train_idx) + len(test_idx) == len(X)

        assert n_splits == 5


class TestSIMCAStringLabels:
    """Test SIMCA with string class labels."""

    def test_string_labels(self, two_class_data):
        """Test with string class labels."""
        X, y = two_class_data
        y_str = np.array(["class_a" if yi == 0 else "class_b" for yi in y])

        model = SIMCA(n_components=2, scale=True)
        model.fit(X, y_str)

        assert "class_a" in model.classes_
        assert "class_b" in model.classes_

        y_pred = model.predict(X)
        assert all(p in ["class_a", "class_b", None] for p in y_pred)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
