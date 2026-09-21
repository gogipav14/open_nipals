"""
Tests for SIMCA classifier implementation.
"""

import warnings

import numpy as np
import pytest

from open_nipals.simca import (
    SIMCA,
    SIMCAClass,
    ComponentSelector,
    KFoldCV,
    LeaveOneOutCV,
    VenetianBlindsCV,
    dmodx_limit,
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


@pytest.fixture
def centred_two_class_data():
    """Two separated classes, centred on the pooled mean."""
    rng = np.random.default_rng(0)
    n_per_class = 60
    n_features = 6

    X0 = rng.standard_normal((n_per_class, n_features))
    X1 = rng.standard_normal((n_per_class, n_features)) + 5.0

    X = np.vstack([X0, X1])
    X = X - X.mean(axis=0)
    y = np.array([0] * n_per_class + [1] * n_per_class)

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


class TestSIMCAPreprocessingOnce:
    """Finding 1: preprocessing must be applied exactly once."""

    def test_predict_agrees_with_membership(self, two_class_data):
        """A sample inside exactly one class is predicted as that class."""
        X, y = two_class_data
        model = SIMCA(n_components=2, scale=True).fit(X, y)

        membership = model.get_class_membership(X)
        y_pred = model.predict(X)

        single = 0
        for i, members in enumerate(membership["member_of"]):
            if len(members) == 1:
                single += 1
                assert y_pred[i] == members[0]

        assert single > 0, "fixture no longer exercises single membership"

    def test_reject_predictions_match_membership(self, two_class_data):
        """With rejection there is no fallback to hide a second scaling."""
        X, y = two_class_data
        model = SIMCA(
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

    def test_scaler_is_applied_once_per_call(self, two_class_data):
        """predict() must not push the data through the scaler twice."""
        X, y = two_class_data
        model = SIMCA(n_components=2, scale=True).fit(X, y)

        original = model.scaler_.transform
        calls = []

        def counting_transform(data):
            calls.append(data)
            return original(data)

        model.scaler_.transform = counting_transform
        model.predict(X)

        assert len(calls) == 1

    def test_predict_matches_externally_scaled_model(self, two_class_data):
        """Scaling inside the model must match scaling by hand."""
        X, y = two_class_data

        inside = SIMCA(n_components=2, scale=True).fit(X, y)
        X_scaled = inside.scaler_.transform(X)
        outside = SIMCA(n_components=2, scale=False).fit(X_scaled, y)

        np.testing.assert_allclose(
            inside.get_distances(X)["dmodx"],
            outside.get_distances(X_scaled)["dmodx"],
            rtol=1e-8,
        )
        assert list(inside.predict(X)) == list(outside.predict(X_scaled))

    def test_membership_matches_distances(self, two_class_data):
        """Membership must be reproducible from the reported distances."""
        X, y = two_class_data
        model = SIMCA(n_components=2, scale=True).fit(X, y)

        membership = model.get_class_membership(X)
        distances = model.get_distances(X)

        for i in range(X.shape[0]):
            within = [
                label
                for j, label in enumerate(model.classes_)
                if distances["t2"][i, j] <= distances["t2_limits"][j]
                and distances["dmodx"][i, j] <= distances["dmodx_limits"][j]
            ]
            assert membership["member_of"][i] == within


class TestSIMCAClassCentering:
    """Finding 2: class models must be centred within their class."""

    def test_class_mean_is_stored_and_removed(self, two_class_data):
        """Each PCA model sees data centred on its own class mean."""
        X, y = two_class_data
        model = SIMCA(n_components=2, scale=True).fit(X, y)
        X_scaled = model.scaler_.transform(X)

        for label, class_model in model.class_models_.items():
            expected = X_scaled[y == label].mean(axis=0)
            np.testing.assert_allclose(
                class_model.class_mean, expected, atol=1e-12
            )

            fit_data = np.asarray(class_model.pca_model.fit_data)
            assert np.abs(fit_data.mean(axis=0)).max() < 1e-10

    def test_r2_describes_within_class_variation(self, two_class_data):
        """R² must not be inflated by the offset between classes."""
        X, y = two_class_data
        model = SIMCA(n_components=3, scale=True).fit(X, y)

        for class_model in model.class_models_.values():
            r2 = class_model.r2_cumulative
            # The classes are spherical, so no single component can
            # explain most of the within-class variation
            assert r2[0] < 0.5
            assert np.all(np.diff(r2) >= -1e-10)
            assert np.all(r2 <= 1.0 + 1e-10)

    def test_distances_use_the_class_mean(self, centred_two_class_data):
        """The class mean itself sits at the centre of its own model."""
        X, y = centred_two_class_data
        model = SIMCA(n_components=2, scale=False).fit(X, y)

        for i, label in enumerate(model.classes_):
            class_model = model.class_models_[label]
            mean_row = class_model.class_mean.reshape(1, -1)
            distances = model.get_distances(mean_row)
            assert distances["t2"][0, i] == pytest.approx(0.0, abs=1e-8)
            assert distances["dmodx"][0, i] == pytest.approx(0.0, abs=1e-8)


class TestSIMCADModXNormalisation:
    """Finding 3: DModX and its limit must live on the same scale."""

    def test_acceptance_is_scale_invariant(self, centred_two_class_data):
        """Multiplying centred data by 10 changes nothing."""
        X, y = centred_two_class_data

        base = SIMCA(n_components=2, scale=False).fit(X, y)
        scaled = SIMCA(n_components=2, scale=False).fit(X * 10.0, y)

        d_base = base.get_distances(X)
        d_scaled = scaled.get_distances(X * 10.0)

        np.testing.assert_allclose(
            d_base["dmodx"], d_scaled["dmodx"], rtol=1e-8
        )
        np.testing.assert_allclose(
            d_base["dmodx_limits"], d_scaled["dmodx_limits"], rtol=1e-12
        )

        accept_base = d_base["dmodx"] <= d_base["dmodx_limits"]
        accept_scaled = d_scaled["dmodx"] <= d_scaled["dmodx_limits"]
        np.testing.assert_array_equal(accept_base, accept_scaled)

    def test_own_class_acceptance_near_alpha(self, centred_two_class_data):
        """Most training samples stay inside their own DModX limit."""
        X, y = centred_two_class_data
        model = SIMCA(n_components=2, scale=False, alpha=0.95).fit(X, y)
        distances = model.get_distances(X)

        for i, label in enumerate(model.classes_):
            own = distances["dmodx"][y == label, i]
            rate = np.mean(own <= distances["dmodx_limits"][i])
            assert 0.85 <= rate <= 1.0

    def test_normalised_dmodx_is_about_one(self, centred_two_class_data):
        """s0 normalises the training DModX to roughly unity."""
        X, y = centred_two_class_data
        model = SIMCA(n_components=2, scale=False).fit(X, y)
        distances = model.get_distances(X)

        for i, label in enumerate(model.classes_):
            own = distances["dmodx"][y == label, i]
            assert 0.8 < np.mean(own) < 1.2

    def test_limit_matches_f_distribution(self, centred_two_class_data):
        """The stored limit is the F-based critical value."""
        X, y = centred_two_class_data
        model = SIMCA(n_components=2, scale=False, alpha=0.95).fit(X, y)
        n_features = X.shape[1]

        for class_model in model.class_models_.values():
            expected = dmodx_limit(
                0.95,
                class_model.n_samples,
                n_features,
                class_model.n_components,
            )
            assert class_model.dmodx_limit == pytest.approx(expected)
            assert np.isfinite(class_model.dmodx_limit)
            assert class_model.s0 > 0


class TestSIMCAComponentValidation:
    """Finding 5: unusable component counts must fail loudly."""

    def test_rejects_n_components_equal_to_n_features(self, two_class_data):
        """A = K leaves no residual degrees of freedom."""
        X, y = two_class_data
        with pytest.raises(ValueError, match="residual degrees of freedom"):
            SIMCA(n_components=X.shape[1], scale=True).fit(X, y)

    def test_rejects_n_components_equal_to_class_size(self):
        """A = n_class - 1 leaves no residual degrees of freedom."""
        rng = np.random.default_rng(3)
        X = rng.standard_normal((12, 30))
        y = np.array([0] * 6 + [1] * 6)

        with pytest.raises(ValueError, match="residual degrees of freedom"):
            SIMCA(n_components=5, scale=False).fit(X, y)

    def test_rejects_tiny_class(self):
        """A class of three samples cannot carry a two-component model."""
        rng = np.random.default_rng(4)
        X = rng.standard_normal((9, 5))
        y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1])

        with pytest.raises(ValueError, match="residual degrees of freedom"):
            SIMCA(n_components=2, scale=False).fit(X, y)

    def test_auto_selection_never_exhausts_features(self):
        """An unreachable R² threshold must not consume every feature."""
        rng = np.random.default_rng(5)
        X = np.vstack(
            [
                rng.standard_normal((25, 4)),
                rng.standard_normal((25, 4)) + 4.0,
            ]
        )
        y = np.array([0] * 25 + [1] * 25)

        model = SIMCA(
            n_components="auto",
            component_selection="r2",
            r2_threshold=1.0,
            scale=False,
        ).fit(X, y)

        for class_model in model.class_models_.values():
            assert class_model.n_components <= X.shape[1] - 1
            assert np.isfinite(class_model.dmodx_limit)
            assert class_model.dmodx_limit > 0
            assert np.isfinite(class_model.t2_limit)
            assert class_model.t2_limit > 0

    @pytest.mark.parametrize(
        "selection", ["r2", "q2", "eigenvalue"]
    )
    def test_auto_selection_respects_limits(self, selection):
        """Every selector stays inside the usable component range."""
        rng = np.random.default_rng(6)
        X = np.vstack(
            [
                rng.standard_normal((25, 5)),
                rng.standard_normal((25, 5)) + 4.0,
            ]
        )
        y = np.array([0] * 25 + [1] * 25)

        model = SIMCA(
            n_components="auto",
            component_selection=selection,
            q2_cv_folds=5,
            scale=False,
        ).fit(X, y)

        for class_model in model.class_models_.values():
            upper = min(X.shape[1] - 1, class_model.n_samples - 2)
            assert 1 <= class_model.n_components <= upper
            assert np.isfinite(class_model.dmodx_limit)
            assert np.isfinite(class_model.t2_limit)

    def test_all_nan_row_is_rejected(self, two_class_data):
        """A row without data must not win an argmin over NaN."""
        X, y = two_class_data
        model = SIMCA(
            n_components=2, scale=True, unknown_handling="closest"
        ).fit(X, y)

        row = np.full((1, X.shape[1]), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prediction = model.predict(row)
            proba = model.predict_proba(row)

        assert prediction[0] is None
        assert np.all(np.isfinite(proba))
        assert proba.sum(axis=1)[0] == pytest.approx(1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
