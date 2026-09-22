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
    X1 = np.random.randn(n_per_class, n_features) * 0.5 + np.array(
        [2, 0] + [0] * 8
    )
    y1 = np.ones(n_per_class)

    # Class 2
    X2 = np.random.randn(n_per_class, n_features) * 0.5 + np.array(
        [0, 2] + [0] * 8
    )
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

    def test_each_class_scaled_by_its_own_statistics(self, two_class_data):
        """Class models are autoscaled on their own training rows."""
        X, y = two_class_data
        model = SIMCA(n_components=2, scale=True).fit(X, y)

        for label, class_model in model.class_models_.items():
            X_class = X[y == label]
            np.testing.assert_allclose(
                class_model.class_mean, X_class.mean(axis=0), atol=1e-12
            )
            np.testing.assert_allclose(
                class_model.class_std, X_class.std(axis=0, ddof=1)
            )
            fit_data = np.asarray(class_model.pca_model.fit_data)
            assert np.abs(fit_data.mean(axis=0)).max() < 1e-10
            np.testing.assert_allclose(fit_data.std(axis=0, ddof=1), 1.0)

    def test_scaling_matches_manual_per_class_autoscaling(
        self, two_class_data
    ):
        """scale=True equals autoscaling every class by hand."""
        X, y = two_class_data
        inside = SIMCA(n_components=2, scale=True).fit(X, y)

        for i, label in enumerate(inside.classes_):
            X_class = X[y == label]
            mean, std = X_class.mean(axis=0), X_class.std(axis=0, ddof=1)
            outside = SIMCA(n_components=2, scale=False).fit(
                (X_class - mean) / std, np.full(len(X_class), label)
            )
            X_manual = (X - mean) / std
            np.testing.assert_allclose(
                inside.get_distances(X)["dmodx"][:, i],
                outside.get_distances(X_manual)["dmodx"][:, 0],
                rtol=1e-8,
            )
            np.testing.assert_allclose(
                inside.get_distances(X)["t2"][:, i],
                outside.get_distances(X_manual)["t2"][:, 0],
                rtol=1e-8,
            )

    def test_no_scale(self, two_class_data):
        """Test with scaling disabled."""
        X, y = two_class_data
        model = SIMCA(n_components=2, scale=False).fit(X, y)

        for class_model in model.class_models_.values():
            np.testing.assert_array_equal(class_model.class_std, 1.0)

    def test_constant_feature_does_not_break_scaling(self, two_class_data):
        """A constant column within a class gets std 1, not 0."""
        X, y = two_class_data
        X = X.copy()
        X[y == y[0], 0] = 3.0

        model = SIMCA(n_components=2, scale=True).fit(X, y)

        assert model.class_models_[y[0]].class_std[0] == 1.0
        assert np.all(np.isfinite(model.get_distances(X)["dmodx"]))


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
        model = SIMCA(
            n_components=2, scale=True, unknown_handling="reject", alpha=0.99
        )
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

        for label, class_model in model.class_models_.items():
            expected = X[y == label].mean(axis=0)
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

    def test_limit_is_the_core_package_limit(self, centred_two_class_data):
        """One DModX limit definition: NipalsPCA.calc_limit."""
        X, y = centred_two_class_data
        model = SIMCA(n_components=2, scale=False, alpha=0.95).fit(X, y)

        for class_model in model.class_models_.values():
            expected = class_model.pca_model.calc_limit(
                metric="DModX", alpha=0.95
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

    @pytest.mark.parametrize("selection", ["r2", "q2", "eigenvalue"])
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


class TestSIMCAReviewRound3:
    """Findings of the third adversarial review."""

    def test_constant_decimal_column_is_not_scaled_up(self):
        """A column of identical 0.1 values must get scale 1, not ~1e-17."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 4))
        X[:, 1] = 0.1
        y = np.array([0] * 10 + [1] * 10)

        model = SIMCA(n_components=2, scale=True).fit(X, y)

        for class_model in model.class_models_.values():
            assert class_model.class_std[1] == 1.0
            fit_data = np.asarray(class_model.pca_model.fit_data)
            assert np.abs(fit_data[:, 1]).max() < 1e-12
            assert np.abs(class_model.pca_model.loadings[1, :]).max() < 1e-8

    def test_cv_constant_decimal_column(self):
        """Same guard inside the cross-validation scaling."""
        from open_nipals.simca.cross_validation import _class_statistics

        X = np.full((10, 3), 0.1)
        X[:, 0] = np.arange(10)
        mean, std = _class_statistics(X, scale=True)
        np.testing.assert_array_equal(std[1:], 1.0)
        np.testing.assert_array_equal(mean[1:], 0.1)

    @pytest.mark.parametrize("scale", [True, False])
    def test_large_constant_column_leaves_no_residual(self, scale):
        """Centring a constant 1.8e18 column must not leave rounding error."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 4))
        X[:, 1] = 1.783098331888911e18
        y = np.array([0] * 10 + [1] * 10)

        model = SIMCA(n_components=1, scale=scale).fit(X, y)

        for class_model in model.class_models_.values():
            fit_data = np.asarray(class_model.pca_model.fit_data)
            assert np.all(fit_data[:, 1] == 0.0)
        distances = model.get_distances(X)
        assert np.all(np.isfinite(distances["t2"]))
        assert np.all(np.isfinite(distances["dmodx"]))
        assert all(p is not None for p in model.predict(X))

    def test_autoscaling_is_unit_invariant(self, two_class_data):
        """Rescaling a feature (even by 1e-14) must not change predictions."""
        X, y = two_class_data
        X_small = X.copy()
        X_small[:, 0] *= 1e-14

        base = SIMCA(n_components=2, scale=True).fit(X, y)
        small = SIMCA(n_components=2, scale=True).fit(X_small, y)

        np.testing.assert_allclose(
            small.get_distances(X_small)["dmodx"],
            base.get_distances(X)["dmodx"],
            rtol=1e-6,
        )
        assert list(small.predict(X_small)) == list(base.predict(X))

    def test_rejected_refit_keeps_previous_model(self, two_class_data):
        X, y = two_class_data
        model = SIMCA(n_components=2).fit(X, y)

        with pytest.raises(ValueError):
            model.fit(X[:, 0], y)  # 1-D input is rejected

        assert model.n_features_in_ == X.shape[1]
        with pytest.raises(ValueError, match="features"):
            model.predict(X[:, :1])
        assert list(model.predict(X)) == list(
            SIMCA(n_components=2).fit(X, y).predict(X)
        )

    def test_wrong_feature_count_is_rejected(self, two_class_data):
        X, y = two_class_data
        model = SIMCA(n_components=2).fit(X, y)

        for method in (
            model.predict,
            model.predict_proba,
            model.get_class_membership,
            model.get_distances,
        ):
            with pytest.raises(ValueError, match="features"):
                method(X[:, :1])

        # Refitting with another feature count is still allowed
        model.fit(X[:, :4], y)
        assert model.n_features_in_ == 4

    def test_component_count_outside_limit_domain_is_rejected(self):
        """calc_limit's DModX formula has no value for A close to n or K."""
        rng = np.random.default_rng(1)
        X = rng.normal(size=(20, 10))
        y = np.zeros(20, dtype=int)

        with pytest.raises(ValueError, match="no DModX limit"):
            SIMCA(n_components=8, scale=False).fit(X, y)

        max_comp = SIMCA(scale=False)._max_components(20, 10)
        assert max_comp < 8
        model = SIMCA(n_components=max_comp, scale=False).fit(X, y)
        assert np.isfinite(model.class_models_[0].dmodx_limit)

    @pytest.mark.parametrize("selection", ["r2", "q2", "eigenvalue"])
    def test_auto_selection_stays_inside_limit_domain(self, selection):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(20, 10))
        y = np.zeros(20, dtype=int)

        model = SIMCA(
            n_components="auto",
            component_selection=selection,
            r2_threshold=1.0,
            scale=False,
        ).fit(X, y)

        class_model = model.class_models_[0]
        assert class_model.n_components <= model._max_components(20, 10)
        assert np.isfinite(class_model.dmodx_limit)


class TestSIMCAReviewRound4:
    """Findings of the fourth adversarial review."""

    def test_eigenvalue_selection_uses_total_variance(self):
        """A truncated fit must not inflate the retained eigenvalues.

        Correlation spectrum 2.0, 0.8 x 5: only one eigenvalue is above
        the Kaiser threshold, whatever the number of fitted components.
        Normalising by the fitted variance only (3 components, 3.6 in
        total) would put 0.8 at 0.8 * 6 / 3.6 = 1.33 and select 3.
        """
        rng = np.random.default_rng(0)
        eigenvalues = np.array([2.0, 0.8, 0.8, 0.8, 0.8, 0.8])
        n_samples = 500
        # Orthonormal, zero-mean score columns give the exact spectrum
        centred = rng.normal(size=(n_samples, 6))
        centred -= centred.mean(axis=0)
        scores, _ = np.linalg.qr(centred)
        basis, _ = np.linalg.qr(rng.normal(size=(6, 6)))
        X = scores * np.sqrt((n_samples - 1) * eigenvalues) @ basis.T

        pca = NipalsPCA(n_components=3).fit(X)

        assert ComponentSelector.select_by_eigenvalue(pca, X) == 1

    @pytest.mark.parametrize("helper", ["pca", "pls"])
    def test_cross_val_predict_integer_input(self, helper):
        """Integer inputs must give the same predictions as floats."""
        from open_nipals.simca.cross_validation import (
            cross_val_predict_pca,
            cross_val_predict_pls,
        )
        from open_nipals.nipalsPLS import NipalsPLS

        rng = np.random.default_rng(0)
        X_int = rng.integers(-5, 6, size=(40, 5))
        y_int = rng.integers(-5, 6, size=(40, 2))
        cv = KFoldCV(n_splits=4)

        if helper == "pca":
            pred_int = cross_val_predict_pca(NipalsPCA, X_int, 2, cv)
            pred_float = cross_val_predict_pca(
                NipalsPCA, X_int.astype(float), 2, cv
            )
            np.testing.assert_allclose(pred_int, pred_float)
        else:
            pred_int = cross_val_predict_pls(NipalsPLS, X_int, y_int, 2, cv)
            pred_float = cross_val_predict_pls(
                NipalsPLS, X_int.astype(float), y_int.astype(float), 2, cv
            )
            for a, b in zip(pred_int, pred_float):
                np.testing.assert_allclose(a, b)

    def test_predict_keeps_label_dtype(self, two_class_data):
        """Plain labels come back with their own dtype, usable by sklearn."""
        from sklearn.metrics import accuracy_score

        X, y = two_class_data
        model = SIMCA(n_components=2).fit(X, y)

        predictions = model.predict(X)
        assert predictions.dtype == np.asarray(y).dtype
        assert accuracy_score(y, predictions) > 0.8

        # Rejections still need an object array to hold None
        rejecting = SIMCA(n_components=2, unknown_handling="reject").fit(X, y)
        far_away = np.full((2, X.shape[1]), 1e3)
        assert rejecting.predict(far_away).dtype == object


class TestSIMCAReviewRound5:
    """Findings of the fifth adversarial review."""

    def test_is_a_sklearn_classifier(self, two_class_data):
        from sklearn.base import clone, is_classifier
        from sklearn.metrics import get_scorer

        X, y = two_class_data
        assert is_classifier(SIMCA())
        model = clone(SIMCA(n_components=2)).fit(X, y)
        assert get_scorer("accuracy")(model, X, y) > 0.8

    def test_empty_refit_keeps_previous_model(self, two_class_data):
        X, y = two_class_data
        model = SIMCA(n_components=2).fit(X, y)
        before = list(model.predict(X))

        with pytest.raises(ValueError, match="zero samples"):
            model.fit(np.empty((0, X.shape[1])), np.empty(0, dtype=y.dtype))

        assert len(model.class_models_) == len(np.unique(y))
        assert list(model.predict(X)) == before


class TestMetricsReviewRound5:
    """R² must not reward failed predictions or crash on reduced models."""

    def test_r2_failed_prediction_is_minus_inf(self):
        from open_nipals.simca.metrics import calc_r2_x, calc_r2_y

        y_true = np.array([[1.0, -1.0], [-1.0, 1.0], [0.5, -0.5]])
        y_pred = y_true.copy()
        y_pred[:, 1] = np.nan  # the second column failed entirely

        assert calc_r2_y(y_true, y_pred) == -np.inf
        per_var = calc_r2_y(y_true, y_pred, per_variable=True)
        assert per_var[0] == pytest.approx(1.0)
        assert per_var[1] == -np.inf
        assert calc_r2_x(y_true, y_pred) == -np.inf

    def test_r2_ignores_missing_observations_only(self):
        from open_nipals.simca.metrics import calc_r2_y

        y_true = np.array([[1.0, np.nan], [-1.0, 1.0], [0.5, -1.0]])
        y_pred = np.where(np.isnan(y_true), 123.0, y_true)  # anything there

        assert calc_r2_y(y_true, y_pred) == pytest.approx(1.0)

    def test_cumulative_r2_after_reducing_components(self, two_class_data):
        from open_nipals.simca.metrics import calc_r2_cumulative_pca

        X, _ = two_class_data
        X = X - X.mean(axis=0)
        pca = NipalsPCA(n_components=3).fit(X)
        expected = calc_r2_cumulative_pca(pca, X)

        pca.set_components(1)
        r2 = calc_r2_cumulative_pca(pca, X)

        np.testing.assert_allclose(r2, expected)
        assert pca.n_components == 1  # restored


class TestSIMCAReviewRound6:
    """Findings of the sixth adversarial review."""

    def test_all_nan_training_rows_are_dropped(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 5))
        y = np.zeros(40, dtype=int)
        reference = SIMCA(n_components=2, scale=False).fit(X, y)

        X_padded = np.vstack([X, np.full((360, 5), np.nan)])
        y_padded = np.zeros(400, dtype=int)
        with pytest.warns(UserWarning, match="Dropping 360"):
            model = SIMCA(n_components=2, scale=False).fit(X_padded, y_padded)

        ref_model = reference.class_models_[0]
        new_model = model.class_models_[0]
        assert new_model.n_samples == 40
        assert new_model.s0 == pytest.approx(ref_model.s0)
        assert new_model.dmodx_limit == pytest.approx(ref_model.dmodx_limit)
        assert list(model.predict(X)) == list(reference.predict(X))

    @pytest.mark.parametrize("position", [0, -1])
    def test_constant_feature_with_nan_anywhere(self, position):
        """A constant, partly missing feature must fit in any column."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 6))
        X[:, position] = 2.5
        X[::5, position] = np.nan
        y = np.array([0] * 20 + [1] * 20)

        model = SIMCA(n_components=2).fit(X, y)

        distances = model.get_distances(X)
        assert np.all(np.isfinite(distances["t2"]))
        assert np.all(np.isfinite(distances["dmodx"]))
