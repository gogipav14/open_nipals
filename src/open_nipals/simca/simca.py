"""
SIMCA (Soft Independent Modelling of Class Analogy) classifier.

Uses NIPALS PCA to build per-class models and classifies samples based on
Hotelling's T² (in-model distance) and a normalised DModX (out-of-model
distance).

Each class model is centred on its own training mean, so the PCA planes
describe the *within-class* variation. DModX is reported relative to the
pooled residual standard deviation s0 of that class, which makes it
dimensionless and comparable with the F-based critical value.
"""

import numpy as np
import warnings
from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Any, Literal
from scipy.stats import f as F_dist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

from open_nipals.nipalsPCA import NipalsPCA
from .metrics import calc_r2_cumulative_pca, calc_q2_cumulative_pca
from .cross_validation import KFoldCV


def dmodx_limit(
    alpha: float, n_samples: int, n_features: int, n_components: int
) -> float:
    """
    Critical value for the normalised DModX statistic.

    DModX divided by the pooled training residual standard deviation is
    approximately the square root of an F ratio with (K - A) and
    (n - A - 1)(K - A) degrees of freedom, following the SIMCA-P
    definition.

    Parameters
    ----------
    alpha : float
        Confidence level.
    n_samples : int
        Number of training rows in the class, n.
    n_features : int
        Number of features, K.
    n_components : int
        Number of components, A.

    Returns
    -------
    float
        Critical value for the normalised DModX.
    """
    dof_obs = n_features - n_components
    dof_mod = (n_samples - n_components - 1) * dof_obs
    return float(np.sqrt(F_dist.ppf(alpha, dof_obs, dof_mod)))


@dataclass
class SIMCAClass:
    """Container for a single class model in SIMCA.

    Attributes
    ----------
    label : Any
        Class label.
    pca_model : NipalsPCA
        Fitted PCA model for this class, fitted on class-centred data.
    n_components : int
        Number of components used.
    t2_limit : float
        Hotelling's T² threshold.
    dmodx_limit : float
        Threshold for the *normalised* DModX.
    n_samples : int
        Number of training samples in this class.
    class_mean : np.ndarray
        Mean of the class training data, after global scaling. It is
        subtracted before the PCA model is applied.
    s0 : float
        Pooled residual standard deviation of the class training data.
        The absolute DModX of new samples is divided by it.
    r2_cumulative : np.ndarray
        Cumulative R² values per component, against the class-centred
        training variation.
    """

    label: Any
    pca_model: NipalsPCA
    n_components: int
    t2_limit: float
    dmodx_limit: float
    n_samples: int
    class_mean: np.ndarray
    s0: float
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
            Centred data used for fitting.
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

        Q² comes from the element-wise (Wold) cross-validation in
        :func:`~open_nipals.simca.cross_validation.cross_val_press_pca`,
        with the centring re-estimated inside every fold.

        Parameters
        ----------
        model_class : class
            PCA model class.
        X : np.ndarray
            Centred data for fitting.
        max_components : int, default=10
            Maximum components to evaluate. Clipped so that the class
            keeps residual degrees of freedom and every CV training fold
            stays large enough.
        cv_folds : int, default=7
            Number of CV folds. Clipped to the number of samples.
        min_improvement : float, default=0.05
            Minimum Q² improvement to add another component.
        **model_kwargs
            Additional arguments for model constructor.

        Returns
        -------
        int
            Selected number of components.

        Raises
        ------
        ValueError
            If no component count leaves usable degrees of freedom.
        """
        n_samples, n_features = X.shape
        cv = KFoldCV(n_splits=max(2, min(cv_folds, n_samples)))
        min_train = min(len(train) for train, _ in cv.split(X))

        max_components = int(
            min(max_components, n_features - 1, n_samples - 2, min_train - 2)
        )
        if max_components < 1:
            raise ValueError(
                f"A class with {n_samples} samples and {n_features} "
                f"features cross-validated over {cv.get_n_splits()} folds "
                "leaves no usable number of components. Provide more "
                "samples or fewer folds."
            )

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
            Centred data used for fitting.
        threshold : float, default=1.0
            Eigenvalue threshold (Kaiser criterion uses 1.0).

        Returns
        -------
        int
            Number of components with eigenvalue > threshold.
        """
        # Eigenvalues are proportional to variance of scores
        scores = np.asarray(pca_model.fit_scores)
        variances = np.var(scores, axis=0, ddof=1)

        # Normalize by number of variables (like correlation matrix)
        n_features = X.shape[1]
        eigenvalues = variances * n_features / np.sum(variances)

        n_above = int(np.sum(eigenvalues > threshold))
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
        Every class needs n_class - n_components - 1 >= 1 and
        n_features - n_components >= 1, otherwise fit() raises.
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

    Notes
    -----
    Preprocessing is applied exactly once per call, in
    :meth:`_prepare_input`. All private helpers take data that has
    already been through it.
    """

    # Swapped by subclasses that use a different PCA implementation
    _pca_class = NipalsPCA

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

    def _prepare_input(self, X: np.ndarray) -> np.ndarray:
        """
        Convert X to a 2-D float array and scale it exactly once.

        This is the single entry point for preprocessing. Every public
        method calls it once; the private helpers never scale again.

        Parameters
        ----------
        X : np.ndarray
            Raw samples in the original feature space.

        Returns
        -------
        np.ndarray
            Globally scaled data, still uncentred per class.
        """
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(
                f"X must be a 2-D array, got {X.ndim} dimension(s)."
            )

        if self.scaler_ is not None:
            X = self.scaler_.transform(X)

        return X

    def _check_fitted(self):
        """Raise NotFittedError if fit() has not been called."""
        if self.classes_ is None:
            raise NotFittedError("SIMCA model has not been fitted")

    def _create_pca_model(self, n_components: int) -> NipalsPCA:
        """Create a PCA model instance from ``_pca_class``."""
        return self._pca_class(
            n_components=n_components,
            max_iter=self.max_iter,
            tol_criteria=self.tol_criteria,
        )

    def _max_components(self, n_samples: int, n_features: int) -> int:
        """
        Largest component count that leaves residual degrees of freedom.

        Parameters
        ----------
        n_samples : int
            Number of samples in the class.
        n_features : int
            Number of features.

        Returns
        -------
        int
            Maximum usable number of components (may be < 1).
        """
        return int(min(10, n_samples - 2, n_features - 1))

    def _validate_n_components(
        self,
        n_components: int,
        n_samples: int,
        n_features: int,
        class_label: Any,
    ):
        """
        Check that a component count leaves a usable class model.

        Parameters
        ----------
        n_components : int
            Component count to check.
        n_samples : int
            Number of samples in the class.
        n_features : int
            Number of features.
        class_label : Any
            Label of the class, used in the error message.

        Raises
        ------
        ValueError
            If the component count leaves no residual degrees of freedom
            for the DModX statistic or its limit.
        """
        if not isinstance(n_components, (int, np.integer)):
            raise ValueError(
                "n_components must be an int or 'auto', got "
                f"{n_components!r}."
            )

        if n_components < 1:
            raise ValueError(
                f"Class {class_label!r}: n_components must be >= 1, got "
                f"{n_components}."
            )

        if n_features - n_components < 1:
            raise ValueError(
                f"Class {class_label!r}: n_components={n_components} leaves "
                f"no residual degrees of freedom for {n_features} features. "
                f"Use n_components <= {n_features - 1}."
            )

        if n_samples - n_components - 1 < 1:
            raise ValueError(
                f"Class {class_label!r}: n_components={n_components} leaves "
                f"no residual degrees of freedom for {n_samples} samples. "
                f"Use n_components <= {n_samples - 2} or collect more "
                "samples for this class."
            )

    def _check_limits(
        self,
        class_label: Any,
        t2_limit: float,
        dmodx_lim: float,
        s0: float,
    ):
        """
        Reject a class model whose limits are not usable.

        Parameters
        ----------
        class_label : Any
            Label of the class, used in the error message.
        t2_limit : float
            Hotelling's T² limit.
        dmodx_lim : float
            Normalised DModX limit.
        s0 : float
            Pooled residual standard deviation.

        Raises
        ------
        ValueError
            If any of the values is not finite and positive.
        """
        checks = (
            ("Hotelling's T² limit", t2_limit),
            ("DModX limit", dmodx_lim),
            ("residual scale s0", s0),
        )
        for name, value in checks:
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    f"Class {class_label!r}: {name} is {value}, which is "
                    "not a finite positive number. The class model would "
                    "be unusable; use fewer components or more samples."
                )

    def _select_components(self, X_class: np.ndarray) -> int:
        """
        Select number of components for a class.

        Parameters
        ----------
        X_class : np.ndarray
            Class-centred data for a single class.

        Returns
        -------
        int
            Selected number of components.

        Raises
        ------
        ValueError
            If the class is too small for any usable model, or if
            component_selection is unknown.
        """
        n_samples, n_features = X_class.shape
        max_components = self._max_components(n_samples, n_features)

        if max_components < 1:
            raise ValueError(
                f"A class with {n_samples} samples and {n_features} "
                "features is too small for a SIMCA model: no component "
                "count leaves residual degrees of freedom."
            )

        if self.component_selection == "r2":
            temp_pca = self._create_pca_model(max_components)
            temp_pca.fit(X_class)
            n_comp = ComponentSelector.select_by_r2(
                temp_pca, X_class, self.r2_threshold, max_components
            )

        elif self.component_selection == "q2":
            n_comp = ComponentSelector.select_by_q2(
                self._pca_class,
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
            n_comp = ComponentSelector.select_by_eigenvalue(
                temp_pca, X_class
            )

        else:
            raise ValueError(
                f"Unknown component_selection: {self.component_selection}"
            )

        return int(min(max(n_comp, 1), max_components))

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

        Raises
        ------
        ValueError
            If X and y disagree in length, or if a class model would have
            no residual degrees of freedom or non-usable limits.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")

        if not (
            isinstance(self.n_components, (int, np.integer))
            or self.n_components == "auto"
        ):
            raise ValueError(
                "n_components must be an int or 'auto', got "
                f"{self.n_components!r}."
            )

        # Handle scaling
        if self.scale:
            if self._check_if_scaled(X):
                warnings.warn(
                    "Data appears pre-scaled (mean≈0, std≈1), skipping scaling"
                )
                self.scaler_ = None
            else:
                self.scaler_ = StandardScaler().fit(X)
        else:
            self.scaler_ = None

        X_scaled = self._prepare_input(X)

        # Get unique classes
        self.classes_ = np.unique(y)
        self.class_models_ = {}

        # Build PCA model for each class
        for class_label in self.classes_:
            mask = y == class_label
            X_class = X_scaled[mask]
            n_samples_class, n_features = X_class.shape

            # Each class model describes the variation around its own mean
            class_mean = np.nanmean(X_class, axis=0)
            X_centered = X_class - class_mean

            # Select and validate the number of components
            if isinstance(self.n_components, (int, np.integer)):
                n_comp = int(self.n_components)
            else:
                n_comp = self._select_components(X_centered)

            self._validate_n_components(
                n_comp, n_samples_class, n_features, class_label
            )

            # Fit PCA model on the centred class data
            pca = self._create_pca_model(n_comp)
            pca.fit(X_centered)

            # Pooled residual standard deviation of the training class,
            # the scale that makes DModX dimensionless
            scores = np.asarray(pca.transform(X_centered))
            residuals = X_centered - np.asarray(
                pca.inverse_transform(scores)
            )
            dof = (n_samples_class - n_comp - 1) * (n_features - n_comp)
            s0 = float(np.sqrt(np.nansum(residuals ** 2) / dof))

            # Calculate limits
            t2_limit = float(
                pca.calc_limit(metric="HotellingT2", alpha=self.alpha)
            )
            dmodx_lim = dmodx_limit(
                self.alpha, n_samples_class, n_features, n_comp
            )
            self._check_limits(class_label, t2_limit, dmodx_lim, s0)

            # Calculate R² for diagnostics, against centred variation
            r2_cumulative = calc_r2_cumulative_pca(pca, X_centered)

            # Store class model
            self.class_models_[class_label] = SIMCAClass(
                label=class_label,
                pca_model=pca,
                n_components=n_comp,
                t2_limit=t2_limit,
                dmodx_limit=dmodx_lim,
                n_samples=n_samples_class,
                class_mean=class_mean,
                s0=s0,
                r2_cumulative=r2_cumulative,
            )

        return self

    def _calc_distances(
        self, X_scaled: np.ndarray, class_label: Any
    ) -> tuple:
        """
        Calculate T² and normalised DModX to a specific class.

        Parameters
        ----------
        X_scaled : np.ndarray
            Data that already went through :meth:`_prepare_input`.
        class_label : Any
            Class to measure the distance to.

        Returns
        -------
        t2 : np.ndarray
            Hotelling's T² values.
        dmodx : np.ndarray
            DModX values divided by the class s0, directly comparable
            with ``SIMCAClass.dmodx_limit``.
        """
        model = self.class_models_[class_label]
        pca = model.pca_model

        X_centered = X_scaled - model.class_mean

        t2 = np.asarray(pca.calc_imd(input_array=X_centered)).ravel()
        dmodx = np.asarray(
            pca.calc_oomd(X_centered, metric="DModX")
        ).ravel()

        return t2, dmodx / model.s0

    def _calc_combined_distances(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Calculate combined normalized distances to all classes.

        Parameters
        ----------
        X_scaled : np.ndarray
            Data that already went through :meth:`_prepare_input`.

        Returns
        -------
        np.ndarray
            Shape (n_samples, n_classes) with normalized distances.
        """
        n_samples = X_scaled.shape[0]
        n_classes = len(self.classes_)
        distances = np.zeros((n_samples, n_classes))

        for i, class_label in enumerate(self.classes_):
            model = self.class_models_[class_label]
            t2, dmodx = self._calc_distances(X_scaled, class_label)

            # Normalize by limits
            t2_norm = t2 / model.t2_limit
            dmodx_norm = dmodx / model.dmodx_limit

            # Combined distance (Euclidean in normalized space)
            distances[:, i] = np.sqrt(t2_norm ** 2 + dmodx_norm ** 2)

        return distances

    def _class_membership(self, X_scaled: np.ndarray) -> Dict[str, List]:
        """
        Determine class membership for already prepared samples.

        Parameters
        ----------
        X_scaled : np.ndarray
            Data that already went through :meth:`_prepare_input`.

        Returns
        -------
        dict
            See :meth:`get_class_membership`.
        """
        n_samples = X_scaled.shape[0]

        results = {
            "t2_in": [[] for _ in range(n_samples)],
            "dmodx_in": [[] for _ in range(n_samples)],
            "both_in": [[] for _ in range(n_samples)],
            "member_of": [[] for _ in range(n_samples)],
        }

        for class_label in self.classes_:
            model = self.class_models_[class_label]
            t2, dmodx = self._calc_distances(X_scaled, class_label)

            # Non-finite distances never compare <=, so they are rejected
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

    def get_class_membership(
        self, X: np.ndarray
    ) -> Dict[str, List]:
        """
        Determine class membership for samples.

        Parameters
        ----------
        X : np.ndarray
            Samples to classify, in the original feature space.

        Returns
        -------
        dict
            Dictionary with keys:
            - 't2_in': List of class labels where T² is within limit
            - 'dmodx_in': List of class labels where DModX is within limit
            - 'both_in': List of class labels where both are within limits
            - 'member_of': Final membership (intersection of t2_in and
              dmodx_in)
        """
        self._check_fitted()
        return self._class_membership(self._prepare_input(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples.

        Parameters
        ----------
        X : np.ndarray
            Samples to classify, in the original feature space.

        Returns
        -------
        np.ndarray
            Predicted class labels. May contain None if
            unknown_handling='reject', or if every distance is
            non-finite.
        """
        self._check_fitted()

        X_scaled = self._prepare_input(X)
        membership = self._class_membership(X_scaled)
        distances = self._calc_combined_distances(X_scaled)

        # Non-finite distances must never win an argmin
        finite_distances = np.where(
            np.isfinite(distances), distances, np.inf
        )

        predictions = []

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
                    np.argmin(finite_distances[i, class_indices])
                ]
                predictions.append(self.classes_[closest_idx])

            elif self.unknown_handling == "reject":
                predictions.append(None)

            elif np.all(np.isinf(finite_distances[i])):
                # No usable distance at all, so nothing to be close to
                predictions.append(None)

            else:  # 'closest'
                closest_idx = int(np.argmin(finite_distances[i]))
                predictions.append(self.classes_[closest_idx])

        return np.array(predictions, dtype=object)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return probability-like scores based on distance to each class.

        Scores are based on inverse combined distance, normalized to sum to 1.
        These are not true probabilities but can be useful for ranking.
        Classes whose distance could not be computed get a score of 0;
        if that holds for every class the row is uniform.

        Parameters
        ----------
        X : np.ndarray
            Samples to score, in the original feature space.

        Returns
        -------
        np.ndarray
            Shape (n_samples, n_classes) with probability-like scores.
        """
        self._check_fitted()

        distances = self._calc_combined_distances(self._prepare_input(X))
        distances = np.where(np.isfinite(distances), distances, np.inf)

        # Convert distances to similarity (inverse)
        # Add small epsilon to avoid division by zero
        similarities = 1.0 / (distances + 1e-10)

        # Normalize to sum to 1, falling back to uniform where every
        # class is unreachable
        totals = similarities.sum(axis=1, keepdims=True)
        n_classes = distances.shape[1]
        proba = np.where(
            totals > 0, similarities / np.where(totals > 0, totals, 1.0),
            1.0 / n_classes
        )

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
        Get T² and normalised DModX distances to all classes.

        Parameters
        ----------
        X : np.ndarray
            Samples to evaluate, in the original feature space.

        Returns
        -------
        dict
            Dictionary with:
            - 't2': Shape (n_samples, n_classes) T² distances
            - 'dmodx': Shape (n_samples, n_classes) normalised DModX
            - 't2_limits': T² limits for each class
            - 'dmodx_limits': DModX limits for each class
        """
        self._check_fitted()

        X_scaled = self._prepare_input(X)
        n_samples = X_scaled.shape[0]
        n_classes = len(self.classes_)

        t2_all = np.zeros((n_samples, n_classes))
        dmodx_all = np.zeros((n_samples, n_classes))
        t2_limits = np.zeros(n_classes)
        dmodx_limits = np.zeros(n_classes)

        for i, class_label in enumerate(self.classes_):
            model = self.class_models_[class_label]
            t2, dmodx = self._calc_distances(X_scaled, class_label)
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
