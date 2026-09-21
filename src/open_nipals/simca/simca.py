"""
SIMCA (Soft Independent Modelling of Class Analogy) classifier.

Uses NIPALS PCA to build per-class models and classifies samples based on
Hotelling's T² (in-model distance) and DModX (out-of-model distance).
"""

import numpy as np
import warnings
from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Any, Literal
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

from open_nipals.nipalsPCA import NipalsPCA
from .metrics import calc_r2_cumulative_pca, calc_q2_cumulative_pca
from .cross_validation import KFoldCV


@dataclass
class SIMCAClass:
    """Container for a single class model in SIMCA.

    Attributes
    ----------
    label : Any
        Class label.
    pca_model : NipalsPCA
        Fitted PCA model for this class.
    n_components : int
        Number of components used.
    t2_limit : float
        Hotelling's T² threshold.
    dmodx_limit : float
        DModX threshold.
    n_samples : int
        Number of training samples in this class.
    r2_cumulative : np.ndarray
        Cumulative R² values per component.
    """

    label: Any
    pca_model: NipalsPCA
    n_components: int
    t2_limit: float
    dmodx_limit: float
    n_samples: int
    r2_cumulative: Optional[np.ndarray] = None


class ComponentSelector:
    """Methods for automatic component selection."""

    @staticmethod
    def select_by_r2(
        pca_model: NipalsPCA,
        X: np.ndarray,
        threshold: float = 0.80,
        max_components: int = 10
    ) -> int:
        """
        Select number of components where cumulative R² >= threshold.

        Parameters
        ----------
        pca_model : NipalsPCA
            Fitted PCA model with enough components.
        X : np.ndarray
            Data used for fitting.
        threshold : float, default=0.80
            R² threshold to achieve.
        max_components : int, default=10
            Maximum components to consider.

        Returns
        -------
        int
            Selected number of components.
        """
        n_comp_available = pca_model.loadings.shape[1]
        max_comp = min(max_components, n_comp_available)

        r2_values = calc_r2_cumulative_pca(pca_model, X)[:max_comp]

        # Find first component where R² exceeds threshold
        for i, r2 in enumerate(r2_values):
            if r2 >= threshold:
                return i + 1

        # If threshold not reached, return max components
        return max_comp

    @staticmethod
    def select_by_q2(
        model_class,
        X: np.ndarray,
        max_components: int = 10,
        cv_folds: int = 7,
        min_improvement: float = 0.05,
        **model_kwargs
    ) -> int:
        """
        Select components where Q² stops improving significantly.

        Parameters
        ----------
        model_class : class
            PCA model class.
        X : np.ndarray
            Data for fitting.
        max_components : int, default=10
            Maximum components to evaluate.
        cv_folds : int, default=7
            Number of CV folds.
        min_improvement : float, default=0.05
            Minimum Q² improvement to add another component.
        **model_kwargs
            Additional arguments for model constructor.

        Returns
        -------
        int
            Selected number of components.
        """
        cv = KFoldCV(n_splits=cv_folds)
        q2_values = calc_q2_cumulative_pca(
            model_class, X, cv, max_components, **model_kwargs
        )

        # Find where Q² stops improving
        best_n = 1
        for i in range(1, len(q2_values)):
            improvement = q2_values[i] - q2_values[i - 1]
            if improvement >= min_improvement:
                best_n = i + 1
            else:
                break

        return best_n

    @staticmethod
    def select_by_eigenvalue(
        pca_model: NipalsPCA,
        X: np.ndarray,
        threshold: float = 1.0
    ) -> int:
        """
        Select components using Kaiser criterion (eigenvalue > threshold).

        Parameters
        ----------
        pca_model : NipalsPCA
            Fitted PCA model.
        X : np.ndarray
            Data used for fitting.
        threshold : float, default=1.0
            Eigenvalue threshold (Kaiser criterion uses 1.0).

        Returns
        -------
        int
            Number of components with eigenvalue > threshold.
        """
        # Eigenvalues are proportional to variance of scores
        scores = pca_model.fit_scores
        variances = np.var(scores, axis=0, ddof=1)

        # Normalize by number of variables (like correlation matrix)
        n_features = X.shape[1]
        eigenvalues = variances * n_features / np.sum(variances)

        n_above = np.sum(eigenvalues > threshold)
        return max(1, n_above)  # At least 1 component


class SIMCA(BaseEstimator, ClassifierMixin):
    """
    SIMCA (Soft Independent Modelling of Class Analogy) classifier.

    Builds separate PCA models for each class and classifies samples based on
    their distance to each class model using Hotelling's T² and DModX.

    Parameters
    ----------
    n_components : int or 'auto', default=2
        Number of components for each class model.
        If 'auto', automatically selects using component_selection method.
    alpha : float, default=0.95
        Significance level for T² and DModX limits.
    component_selection : str, default='q2'
        Method for automatic component selection: 'r2', 'q2', 'eigenvalue'.
    r2_threshold : float, default=0.80
        R² threshold when component_selection='r2'.
    q2_cv_folds : int, default=7
        CV folds when component_selection='q2'.
    q2_min_improvement : float, default=0.05
        Minimum Q² improvement when component_selection='q2'.
    scale : bool, default=True
        Whether to apply StandardScaler to data.
    unknown_handling : str, default='closest'
        How to handle samples not in any class:
        - 'closest': Assign to closest class by combined distance.
        - 'reject': Return None for outliers.
    max_iter : int, default=10000
        Maximum iterations for NIPALS convergence.
    tol_criteria : float, default=1e-8
        Convergence tolerance for NIPALS.

    Attributes
    ----------
    classes_ : np.ndarray
        Unique class labels.
    class_models_ : dict
        Dictionary mapping class labels to SIMCAClass objects.
    scaler_ : StandardScaler or None
        Fitted scaler if scaling was applied.
    """

    def __init__(
        self,
        n_components: Union[int, Literal["auto"]] = 2,
        alpha: float = 0.95,
        component_selection: str = "q2",
        r2_threshold: float = 0.80,
        q2_cv_folds: int = 7,
        q2_min_improvement: float = 0.05,
        scale: bool = True,
        unknown_handling: Literal["closest", "reject"] = "closest",
        max_iter: int = 10000,
        tol_criteria: float = 1e-8,
    ):
        self.n_components = n_components
        self.alpha = alpha
        self.component_selection = component_selection
        self.r2_threshold = r2_threshold
        self.q2_cv_folds = q2_cv_folds
        self.q2_min_improvement = q2_min_improvement
        self.scale = scale
        self.unknown_handling = unknown_handling
        self.max_iter = max_iter
        self.tol_criteria = tol_criteria

        self.classes_ = None
        self.class_models_: Dict[Any, SIMCAClass] = {}
        self.scaler_: Optional[StandardScaler] = None

    def _check_if_scaled(self, X: np.ndarray) -> bool:
        """
        Check if data appears already scaled (mean≈0, std≈1).

        Parameters
        ----------
        X : np.ndarray
            Data to check.

        Returns
        -------
        bool
            True if data appears pre-scaled.
        """
        # Handle NaN values
        means = np.abs(np.nanmean(X, axis=0))
        stds = np.nanstd(X, axis=0)

        is_centered = np.all(means < 0.1)
        is_scaled = np.all(np.abs(stds - 1.0) < 0.1)

        return is_centered and is_scaled

    def _apply_scaling(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted scaler to data."""
        if self.scaler_ is not None:
            return self.scaler_.transform(X)
        return X

    def _create_pca_model(self, n_components: int) -> NipalsPCA:
        """Create a PCA model instance. Override in JAX version."""
        return NipalsPCA(
            n_components=n_components,
            max_iter=self.max_iter,
            tol_criteria=self.tol_criteria,
        )

    def _select_components(self, X_class: np.ndarray) -> int:
        """
        Select number of components for a class.

        Parameters
        ----------
        X_class : np.ndarray
            Data for single class.

        Returns
        -------
        int
            Selected number of components.
        """
        if isinstance(self.n_components, int):
            return self.n_components

        # Auto selection
        n_samples, n_features = X_class.shape
        max_components = min(10, n_samples - 1, n_features)

        if max_components < 1:
            return 1

        if self.component_selection == "r2":
            # Fit temporary model with max components
            temp_pca = self._create_pca_model(max_components)
            temp_pca.fit(X_class)
            return ComponentSelector.select_by_r2(
                temp_pca, X_class, self.r2_threshold, max_components
            )

        elif self.component_selection == "q2":
            return ComponentSelector.select_by_q2(
                type(self._create_pca_model(1)),  # Pass class
                X_class,
                max_components,
                self.q2_cv_folds,
                self.q2_min_improvement,
                max_iter=self.max_iter,
                tol_criteria=self.tol_criteria,
            )

        elif self.component_selection == "eigenvalue":
            temp_pca = self._create_pca_model(max_components)
            temp_pca.fit(X_class)
            return ComponentSelector.select_by_eigenvalue(
                temp_pca, X_class
            )

        else:
            raise ValueError(
                f"Unknown component_selection: {self.component_selection}"
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SIMCA":
        """
        Fit SIMCA model by building PCA models for each class.

        Parameters
        ----------
        X : np.ndarray
            Training data (n_samples, n_features).
        y : np.ndarray
            Class labels (n_samples,).

        Returns
        -------
        self
            Fitted SIMCA model.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")

        # Handle scaling
        if self.scale:
            if self._check_if_scaled(X):
                warnings.warn(
                    "Data appears pre-scaled (mean≈0, std≈1), skipping scaling"
                )
                self.scaler_ = None
            else:
                self.scaler_ = StandardScaler()
                X = self.scaler_.fit_transform(X)
        else:
            self.scaler_ = None

        # Get unique classes
        self.classes_ = np.unique(y)
        self.class_models_ = {}

        # Build PCA model for each class
        for class_label in self.classes_:
            mask = y == class_label
            X_class = X[mask]
            n_samples_class = X_class.shape[0]

            # Select number of components
            n_comp = self._select_components(X_class)

            # Fit PCA model
            pca = self._create_pca_model(n_comp)
            pca.fit(X_class)

            # Calculate limits
            t2_limit = pca.calc_limit(
                metric="HotellingT2", alpha=self.alpha
            )
            dmodx_limit = pca.calc_limit(
                metric="DModX", alpha=self.alpha
            )

            # Calculate R² for diagnostics
            r2_cumulative = calc_r2_cumulative_pca(pca, X_class)

            # Store class model
            self.class_models_[class_label] = SIMCAClass(
                label=class_label,
                pca_model=pca,
                n_components=n_comp,
                t2_limit=t2_limit,
                dmodx_limit=dmodx_limit,
                n_samples=n_samples_class,
                r2_cumulative=r2_cumulative,
            )

        return self

    def _calc_distances(
        self, X: np.ndarray, class_label: Any
    ) -> tuple:
        """
        Calculate T² and DModX for samples to a specific class.

        Returns
        -------
        t2 : np.ndarray
            Hotelling's T² values.
        dmodx : np.ndarray
            DModX values.
        """
        model = self.class_models_[class_label]
        pca = model.pca_model

        t2 = pca.calc_imd(input_array=X)
        dmodx = pca.calc_oomd(X, metric="DModX")

        return t2.flatten(), dmodx.flatten()

    def _calc_combined_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate combined normalized distances to all classes.

        Returns
        -------
        np.ndarray
            Shape (n_samples, n_classes) with normalized distances.
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        distances = np.zeros((n_samples, n_classes))

        for i, class_label in enumerate(self.classes_):
            model = self.class_models_[class_label]
            t2, dmodx = self._calc_distances(X, class_label)

            # Normalize by limits
            t2_norm = t2 / model.t2_limit
            dmodx_norm = dmodx / model.dmodx_limit

            # Combined distance (Euclidean in normalized space)
            distances[:, i] = np.sqrt(t2_norm ** 2 + dmodx_norm ** 2)

        return distances

    def get_class_membership(
        self, X: np.ndarray
    ) -> Dict[str, List]:
        """
        Determine class membership for samples.

        Parameters
        ----------
        X : np.ndarray
            Samples to classify.

        Returns
        -------
        dict
            Dictionary with keys:
            - 't2_in': List of class labels where T² is within limit
            - 'dmodx_in': List of class labels where DModX is within limit
            - 'both_in': List of class labels where both are within limits
            - 'member_of': Final membership (intersection of t2_in and dmodx_in)
        """
        X = self._apply_scaling(np.asarray(X))
        n_samples = X.shape[0]

        results = {
            "t2_in": [[] for _ in range(n_samples)],
            "dmodx_in": [[] for _ in range(n_samples)],
            "both_in": [[] for _ in range(n_samples)],
            "member_of": [[] for _ in range(n_samples)],
        }

        for class_label in self.classes_:
            model = self.class_models_[class_label]
            t2, dmodx = self._calc_distances(X, class_label)

            t2_within = t2 <= model.t2_limit
            dmodx_within = dmodx <= model.dmodx_limit
            both_within = t2_within & dmodx_within

            for i in range(n_samples):
                if t2_within[i]:
                    results["t2_in"][i].append(class_label)
                if dmodx_within[i]:
                    results["dmodx_in"][i].append(class_label)
                if both_within[i]:
                    results["both_in"][i].append(class_label)
                    results["member_of"][i].append(class_label)

        return results

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples.

        Parameters
        ----------
        X : np.ndarray
            Samples to classify.

        Returns
        -------
        np.ndarray
            Predicted class labels. May contain None if unknown_handling='reject'.
        """
        if self.classes_ is None:
            raise NotFittedError("SIMCA model has not been fitted")

        X = self._apply_scaling(np.asarray(X))
        membership = self.get_class_membership(X)

        predictions = []
        distances = self._calc_combined_distances(X)

        for i, member_classes in enumerate(membership["member_of"]):
            if len(member_classes) == 1:
                # Single membership - assign to that class
                predictions.append(member_classes[0])

            elif len(member_classes) > 1:
                # Multiple memberships - pick closest
                # Only consider the classes that sample belongs to
                class_indices = [
                    np.where(self.classes_ == c)[0][0]
                    for c in member_classes
                ]
                closest_idx = class_indices[
                    np.argmin(distances[i, class_indices])
                ]
                predictions.append(self.classes_[closest_idx])

            else:
                # No membership
                if self.unknown_handling == "reject":
                    predictions.append(None)
                else:  # 'closest'
                    closest_idx = np.argmin(distances[i])
                    predictions.append(self.classes_[closest_idx])

        return np.array(predictions, dtype=object)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return probability-like scores based on distance to each class.

        Scores are based on inverse combined distance, normalized to sum to 1.
        These are not true probabilities but can be useful for ranking.

        Parameters
        ----------
        X : np.ndarray
            Samples to score.

        Returns
        -------
        np.ndarray
            Shape (n_samples, n_classes) with probability-like scores.
        """
        if self.classes_ is None:
            raise NotFittedError("SIMCA model has not been fitted")

        X = self._apply_scaling(np.asarray(X))
        distances = self._calc_combined_distances(X)

        # Convert distances to similarity (inverse)
        # Add small epsilon to avoid division by zero
        similarities = 1.0 / (distances + 1e-10)

        # Normalize to sum to 1
        proba = similarities / similarities.sum(axis=1, keepdims=True)

        return proba

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return accuracy score.

        Parameters
        ----------
        X : np.ndarray
            Test samples.
        y : np.ndarray
            True labels.

        Returns
        -------
        float
            Accuracy score.
        """
        y_pred = self.predict(X)
        # Handle None predictions
        correct = sum(
            1 for pred, true in zip(y_pred, y)
            if pred is not None and pred == true
        )
        return correct / len(y)

    def get_distances(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get T² and DModX distances to all classes.

        Parameters
        ----------
        X : np.ndarray
            Samples to evaluate.

        Returns
        -------
        dict
            Dictionary with:
            - 't2': Shape (n_samples, n_classes) T² distances
            - 'dmodx': Shape (n_samples, n_classes) DModX distances
            - 't2_limits': T² limits for each class
            - 'dmodx_limits': DModX limits for each class
        """
        if self.classes_ is None:
            raise NotFittedError("SIMCA model has not been fitted")

        X = self._apply_scaling(np.asarray(X))
        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        t2_all = np.zeros((n_samples, n_classes))
        dmodx_all = np.zeros((n_samples, n_classes))
        t2_limits = np.zeros(n_classes)
        dmodx_limits = np.zeros(n_classes)

        for i, class_label in enumerate(self.classes_):
            model = self.class_models_[class_label]
            t2, dmodx = self._calc_distances(X, class_label)
            t2_all[:, i] = t2
            dmodx_all[:, i] = dmodx
            t2_limits[i] = model.t2_limit
            dmodx_limits[i] = model.dmodx_limit

        return {
            "t2": t2_all,
            "dmodx": dmodx_all,
            "t2_limits": t2_limits,
            "dmodx_limits": dmodx_limits,
        }
