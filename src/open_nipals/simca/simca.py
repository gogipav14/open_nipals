"""
SIMCA (Soft Independent Modelling of Class Analogy) classifier.

Uses NIPALS PCA to build per-class models and classifies samples based on
Hotelling's T² (in-model distance) and a normalised DModX (out-of-model
distance).

Each class model is centred, and with ``scale=True`` autoscaled, on its
own training data, so the PCA planes describe the *within-class*
variation (as in SIMCA-P and other SIMCA implementations). DModX is
reported relative to the pooled residual standard deviation s0 of that
class, which makes it dimensionless and comparable with the critical
value from :meth:`~open_nipals.nipalsPCA.NipalsPCA.calc_limit`.
"""

import numpy as np
import warnings
from dataclasses import dataclass
from typing import Tuple, Optional, Union, List, Dict, Any, Literal
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError

from open_nipals.nipalsPCA import NipalsPCA
from .metrics import calc_r2_cumulative_pca, calc_q2_cumulative_pca
from .cross_validation import KFoldCV, _class_statistics


def _varying_mask(X: np.ndarray) -> np.ndarray:
    """Columns with at least two different observed values.

    Constant and entirely missing columns have no variance to model, so
    they do not count as features for component bounds, degrees of
    freedom or the informative-row test.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        observed = np.any(~np.isnan(X), axis=0)
        constant = np.nanmax(X, axis=0) == np.nanmin(X, axis=0)
    return observed & ~constant


def _n_varying(X: np.ndarray) -> int:
    """Number of columns selected by :func:`_varying_mask`."""
    return int(np.sum(_varying_mask(X)))


def _residual_sums(
    pca: NipalsPCA, X_centered: np.ndarray, varying: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-row residual sum of squares and its feature count.

    The sum runs over every observed feature, so a sample that deviates
    in a feature that was constant in training is still out of model.
    The count is the number of observed *varying* features, the degrees
    of freedom those residuals are spread over.
    """
    scores = np.asarray(pca.transform(X_centered))
    residuals = X_centered - np.asarray(pca.inverse_transform(scores))
    observed = ~np.isnan(X_centered)
    sse = np.sum(np.where(observed, residuals, 0.0) ** 2, axis=1)
    n_observed = np.sum(observed & varying, axis=1)
    return sse, n_observed


def _numerical_rank(X: np.ndarray) -> int:
    """Numerical rank of the centred data, missing values set to zero.

    Zero filling can only add rank, so this is an upper bound when
    values are missing and exact otherwise.
    """
    X = np.where(np.isnan(X), 0.0, X)
    return int(np.linalg.matrix_rank(X - X.mean(axis=0)))


def _determined_rows(X_class: np.ndarray, n_components: int) -> np.ndarray:
    """
    Rows with at least n_components + 1 observed varying features.

    Dropping rows can make a feature constant, which can leave further
    rows short of observed varying features, so the test is repeated
    until no more rows are removed.
    """
    keep = np.ones(X_class.shape[0], dtype=bool)
    while True:
        varying = _varying_mask(X_class[keep])
        n_observed = np.sum(~np.isnan(X_class[:, varying]), axis=1)
        new_keep = keep & (n_observed >= n_components + 1)
        if np.array_equal(new_keep, keep) or not np.any(new_keep):
            return new_keep
        keep = new_keep


def _informative_rows(X_class: np.ndarray, class_label: Any) -> np.ndarray:
    """
    Drop training rows that cannot inform any class model.

    Every model has at least one component, so a row needs at least two
    observed varying features to have a determined score and a residual.
    Entirely missing rows, rows observed only in class-constant features
    and singletons are dropped (repeatedly, see _determined_rows), with
    a warning; they would otherwise count as samples in the component
    selection, the limits and s0.
    """
    keep = _determined_rows(X_class, 1)
    n_dropped = int(np.sum(~keep))
    if n_dropped:
        warnings.warn(
            f"Class {class_label!r}: dropping {n_dropped} training rows "
            "with fewer than 2 observed varying features."
        )
    return X_class[keep]


class _ResidualExhausted(ValueError):
    """A component count that leaves no residual variation."""


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
        Mean of the class training data. Subtracted before the PCA
        model is applied.
    class_std : np.ndarray
        Standard deviation (ddof=1) of the class training data, all ones
        when the model was fitted with ``scale=False``. The centred data
        is divided by it before the PCA model is applied.
    varying : np.ndarray
        Boolean mask of the features that vary within the class. Only
        they count as degrees of freedom for DModX.
    s0 : float
        Pooled residual standard deviation per degree of freedom of the
        class training data, over rows with at least one residual degree
        of freedom. The absolute DModX of new samples is divided by it.
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
    class_std: np.ndarray
    varying: np.ndarray
    s0: float
    r2_cumulative: Optional[np.ndarray] = None


class ComponentSelector:
    """Methods for automatic component selection."""

    @staticmethod
    def select_by_r2(
        pca_model: NipalsPCA,
        X: np.ndarray,
        threshold: float = 0.80,
        max_components: int = 10,
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
        scale: bool = False,
        **model_kwargs,
    ) -> int:
        """
        Select the fewest components whose Q² is close to the best Q².

        The count is the smallest one whose cumulative Q² is within
        ``min_improvement`` of the maximum over all evaluated counts.
        Stopping at the first small improvement instead is fragile: a
        weak component followed by a strong one (Q² 0.55, 0.59, 0.90)
        would stop at 1, and which side of the threshold the weak step
        falls on depends on the CV folds.

        Q² comes from the element-wise (Wold) cross-validation in
        :func:`~open_nipals.simca.cross_validation.cross_val_press_pca`,
        with the centring (and scaling) re-estimated inside every fold.

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
            Tolerance below the best Q² within which fewer components
            are preferred.
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
        # Canonical row order: the CV folds (and so the selection) then
        # depend on which rows are in the class, not on their order
        X = X[np.lexsort(np.nan_to_num(X, nan=np.inf).T[::-1])]
        n_samples = X.shape[0]
        n_features = _n_varying(X)
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

        # A fold can be numerically singular for the largest counts even
        # when the full class is not; fall back to fewer components
        while True:
            try:
                q2_values = calc_q2_cumulative_pca(
                    model_class,
                    X,
                    cv,
                    max_components,
                    scale=scale,
                    **model_kwargs,
                )
                break
            except np.linalg.LinAlgError:
                if max_components == 1:
                    raise
                max_components -= 1

        # Fewest components within min_improvement of the best Q²
        close_to_best = q2_values >= np.max(q2_values) - min_improvement
        return int(np.argmax(close_to_best)) + 1

    @staticmethod
    def select_by_eigenvalue(
        pca_model: NipalsPCA, X: np.ndarray, threshold: float = 1.0
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
        # Eigenvalues are the variances of the scores. They are put on
        # the correlation-matrix scale (sum of all eigenvalues equals
        # the number of features) using the total variance of X, not
        # the variance of the fitted components only: SIMCA fits a
        # truncated spectrum and that would inflate every eigenvalue.
        scores = np.asarray(pca_model.fit_scores)
        variances = np.var(scores, axis=0, ddof=1)

        # Constant and empty columns hold no variance and do not count
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            column_variance = np.nanvar(X, axis=0, ddof=1)
        varying = np.nan_to_num(column_variance) > 0
        n_features = int(np.sum(varying))
        total_variance = np.sum(column_variance[varying])
        eigenvalues = variances * n_features / total_variance

        # Leading components only: stop at the first eigenvalue below the
        # threshold. With missing data NIPALS can fail to converge on a
        # small trailing component and return a huge score variance,
        # which must not count.
        below = np.flatnonzero(~(eigenvalues > threshold))
        n_above = int(below[0]) if below.size else len(eigenvalues)
        return max(1, n_above)  # At least 1 component


class SIMCA(ClassifierMixin, BaseEstimator):
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
        With component_selection='q2', the fewest components whose Q² is
        within this of the best Q² are used.
    scale : bool, default=True
        Autoscale each class by its own training mean and standard
        deviation (ddof=1), as SIMCA-P does. With False the classes are
        only centred.
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
        self.n_features_in_: Optional[int] = None
        self.class_models_: Dict[Any, SIMCAClass] = {}

    def _prepare_input(
        self, X: np.ndarray, check_features: bool = True
    ) -> np.ndarray:
        """
        Convert X to a 2-D float array in the original feature space.

        Centring and scaling are per class, see :meth:`_to_class_space`.

        Parameters
        ----------
        X : np.ndarray
            Raw samples in the original feature space.
        check_features : bool, default=True
            Reject a feature count different from the fitted one. Off
            during fit, which may legitimately change it.

        Returns
        -------
        np.ndarray
            The same samples as a float array.
        """
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(
                f"X must be a 2-D array, got {X.ndim} dimension(s)."
            )

        if (
            check_features
            and self.n_features_in_ is not None
            and X.shape[1] != self.n_features_in_
        ):
            raise ValueError(
                f"X has {X.shape[1]} features, but this SIMCA model was "
                f"fitted with {self.n_features_in_} features."
            )

        return X

    def _to_class_space(self, X: np.ndarray, model: SIMCAClass) -> np.ndarray:
        """Centre (and scale) samples the way the class model was fitted."""
        return (X - model.class_mean) / model.class_std

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
        upper = int(min(10, n_samples - 2, n_features - 1))
        # The DModX limit formula has its own domain, see _limit_is_usable
        n_max = 0
        for n_comp in range(1, upper + 1):
            if not self._limit_is_usable(n_samples, n_features, n_comp):
                break
            n_max = n_comp
        return n_max

    def _limit_is_usable(
        self, n_samples: int, n_features: int, n_components: int
    ) -> bool:
        """
        Whether the core DModX limit exists for these dimensions.

        NipalsPCA.calc_limit(metric="DModX") follows the SIMCA-P help
        formula, whose observation degrees of freedom drop below 1 (the
        core code warns) and then turn negative (NaN limit) for component
        counts close to the sample or feature count, even when
        (n - A - 1)(K - A) is still positive. Both cases are rejected.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            limit = self._pca_class().calc_limit(
                metric="DModX",
                n=n_samples,
                num_lvs=n_components,
                m=n_features,
                alpha=self.alpha,
            )
        return not caught and bool(np.isfinite(limit)) and limit > 0

    def _count_is_feasible(self, X_class: np.ndarray, n_comp: int) -> bool:
        """Whether n_comp leaves a usable model after row filtering."""
        keep = _determined_rows(X_class, n_comp)
        n_samples = int(np.sum(keep))
        if n_samples == 0:
            return False
        X_kept = X_class[keep]
        n_features = _n_varying(X_kept)
        return (
            n_samples - n_comp - 1 >= 1
            and n_features - n_comp >= 1
            # the kept rows must still have variation beyond n_comp
            and n_comp <= _numerical_rank(X_kept) - 1
            and self._limit_is_usable(n_samples, n_features, n_comp)
        )

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
                f"n_components must be an int or 'auto', got {n_components!r}."
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
        if not self._limit_is_usable(n_samples, n_features, n_components):
            raise ValueError(
                f"Class {class_label!r}: no DModX limit exists for "
                f"n_components={n_components} with {n_samples} samples and "
                f"{n_features} features. Use n_components <= "
                f"{self._max_components(n_samples, n_features)}."
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
        n_samples = X_class.shape[0]
        n_features = _n_varying(X_class)
        max_components = self._max_components(n_samples, n_features)

        # Beyond rank - 1 no residual is left (duplicated channels etc.)
        max_components = min(max_components, _numerical_rank(X_class) - 1)

        # With missing data a larger count drops more rows (see
        # _determined_rows); only offer counts that still leave a usable
        # model after that filtering
        feasible = 0
        for n_comp in range(1, max_components + 1):
            if not self._count_is_feasible(X_class, n_comp):
                break
            feasible = n_comp
        max_components = feasible

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
                scale=self.scale,
                max_iter=self.max_iter,
                tol_criteria=self.tol_criteria,
            )

        elif self.component_selection == "eigenvalue":
            temp_pca = self._create_pca_model(max_components)
            temp_pca.fit(X_class)
            n_comp = ComponentSelector.select_by_eigenvalue(temp_pca, X_class)

        else:
            raise ValueError(
                f"Unknown component_selection: {self.component_selection}"
            )

        return int(min(max(n_comp, 1), max_components))

    def _fit_class_model(
        self, X_class: np.ndarray, n_comp: int, class_label: Any
    ) -> "SIMCAClass":
        """
        Fit one class model with a given component count.

        Parameters
        ----------
        X_class : np.ndarray
            Training rows of the class (already without rows unusable by
            any model), in the original feature space.
        n_comp : int
            Number of components.
        class_label : Any
            Label of the class, used in messages.

        Returns
        -------
        SIMCAClass
            The fitted class model.

        Raises
        ------
        _ResidualExhausted
            If n_comp leaves no residual variation.
        ValueError
            If the class cannot support n_comp components.
        """
        n_samples_class = X_class.shape[0]
        n_varying = _n_varying(X_class)
        class_mean, class_std = _class_statistics(X_class, self.scale)
        X_centered = (X_class - class_mean) / class_std

        # A row needs more observed varying features than components:
        # otherwise its scores are underdetermined and it has no
        # residual, yet it would count in the T2 score variances, the
        # sample count of the limits and s0. Drop such rows and
        # re-estimate the class statistics without them.
        determined = _determined_rows(X_class, n_comp)
        if not np.all(determined):
            warnings.warn(
                f"Class {class_label!r}: dropping "
                f"{int(np.sum(~determined))} training rows with fewer "
                f"than {n_comp + 1} observed varying features, too "
                f"few for {n_comp} components."
            )
            X_class = X_class[determined]
            n_samples_class = X_class.shape[0]
            if n_samples_class == 0:
                raise ValueError(
                    f"Class {class_label!r}: no training row has "
                    f"{n_comp + 1} observed varying features."
                )
            n_varying = _n_varying(X_class)
            class_mean, class_std = _class_statistics(X_class, self.scale)
            X_centered = (X_class - class_mean) / class_std

        self._validate_n_components(
            n_comp, n_samples_class, n_varying, class_label
        )

        # Fit PCA model on the centred class data
        pca = self._create_pca_model(n_comp)
        pca.fit(X_centered)

        # Pooled residual standard deviation per degree of freedom of
        # the training class, the scale that makes DModX
        # dimensionless. Each row has (observed varying features - A)
        # degrees of freedom; rows with none (too sparse for this A)
        # cannot contribute a residual estimate and are left out.
        varying = _varying_mask(X_class)
        sse, n_observed = _residual_sums(pca, X_centered, varying)
        usable = n_observed - n_comp >= 1
        dof = float(np.sum(n_observed[usable] - n_comp))
        s0 = float(np.sqrt(np.sum(sse[usable]) / dof)) if dof > 0 else np.nan
        # A residual at rounding level means the components used up
        # all the variation: DModX would measure numerical noise. NIPALS
        # can also run out of variation mid-fit and return NaN loadings.
        data_scale = float(np.sqrt(np.nanmean(X_centered**2)))
        # Rounding leaves residuals of about eps (measured ~0.2 eps for
        # an exhausted rank); 100 eps separates that from real noise,
        # which in float32 can be as small as 1e-4 of the data scale
        tolerance = 100 * self._compute_eps()
        fit_failed = not (
            np.all(np.isfinite(np.asarray(pca.loadings))) and np.isfinite(s0)
        )
        if fit_failed or s0 <= tolerance * data_scale:
            raise _ResidualExhausted(
                f"Class {class_label!r}: n_components={n_comp} leaves "
                "no residual variation (the class data has rank "
                f"<= {n_comp}). Use fewer components."
            )

        # Calculate limits
        t2_limit = float(
            pca.calc_limit(metric="HotellingT2", alpha=self.alpha)
        )
        dmodx_lim = float(
            pca.calc_limit(metric="DModX", m=n_varying, alpha=self.alpha)
        )
        self._check_limits(class_label, t2_limit, dmodx_lim, s0)

        # Calculate R² for diagnostics, against centred variation
        r2_cumulative = calc_r2_cumulative_pca(pca, X_centered)

        # Store class model
        return SIMCAClass(
            label=class_label,
            pca_model=pca,
            n_components=n_comp,
            t2_limit=t2_limit,
            dmodx_limit=dmodx_lim,
            n_samples=n_samples_class,
            class_mean=class_mean,
            class_std=class_std,
            varying=varying,
            s0=s0,
            r2_cumulative=r2_cumulative,
        )

    def _compute_eps(self) -> float:
        """Machine epsilon of the precision the PCA models compute in."""
        return float(np.finfo(np.float64).eps)

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
        if X.shape[0] == 0:
            raise ValueError("Cannot fit SIMCA on zero samples")

        if not (
            isinstance(self.n_components, (int, np.integer))
            or self.n_components == "auto"
        ):
            raise ValueError(
                "n_components must be an int or 'auto', got "
                f"{self.n_components!r}."
            )

        X = self._prepare_input(X, check_features=False)

        # Fitted state is only replaced once every class model succeeded,
        # a rejected refit leaves the previous model usable
        classes = np.unique(y)
        class_models = {}

        # Build PCA model for each class
        for class_label in classes:
            mask = y == class_label
            X_class = _informative_rows(X[mask], class_label)
            n_samples_class, n_features = X_class.shape
            if n_samples_class == 0:
                raise ValueError(
                    f"Class {class_label!r} has no training row with at "
                    "least 2 observed varying features."
                )
            # Degrees of freedom and limits count varying features only
            n_varying = _n_varying(X_class)

            # Each class model describes the variation around its own
            # mean, in units of its own standard deviation if scale is set
            class_mean, class_std = _class_statistics(X_class, self.scale)
            X_centered = (X_class - class_mean) / class_std

            # Select and validate the number of components
            if isinstance(self.n_components, (int, np.integer)):
                n_comp = int(self.n_components)
            else:
                n_comp = self._select_components(X_centered)

            self._validate_n_components(
                n_comp, n_samples_class, n_varying, class_label
            )

            # In automatic mode a count whose fit leaves no residual (the
            # rank estimate is only an upper bound with missing values)
            # falls back to fewer components
            automatic = not isinstance(self.n_components, (int, np.integer))
            while True:
                try:
                    class_models[class_label] = self._fit_class_model(
                        X_class, n_comp, class_label
                    )
                    break
                except _ResidualExhausted:
                    if not automatic or n_comp == 1:
                        raise
                    n_comp -= 1

        self.classes_ = classes
        self.class_models_ = class_models
        self.n_features_in_ = X.shape[1]

        return self

    def _calc_distances(self, X: np.ndarray, class_label: Any) -> tuple:
        """
        Calculate T² and normalised DModX to a specific class.

        Parameters
        ----------
        X : np.ndarray
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

        X_centered = self._to_class_space(X, model)

        t2 = np.asarray(pca.calc_imd(input_array=X_centered)).ravel()

        # DModX of a row: residual standard deviation per degree of
        # freedom over its observed varying features, relative to the
        # pooled training value s0. (SIMCA-P's n / (n - A - 1) factor
        # multiplies both and cancels.) Rows with no residual degree of
        # freedom cannot be judged and are infinitely far (rejected).
        n_comp = model.n_components
        sse, n_observed = _residual_sums(pca, X_centered, model.varying)
        dof = n_observed - n_comp
        with np.errstate(divide="ignore", invalid="ignore"):
            dmodx = np.sqrt(sse / dof)
        dmodx = np.where(dof >= 1, dmodx, np.inf)

        return t2, dmodx / model.s0

    def _calc_combined_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate combined normalized distances to all classes.

        Parameters
        ----------
        X : np.ndarray
            Data that already went through :meth:`_prepare_input`.

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
            distances[:, i] = np.sqrt(t2_norm**2 + dmodx_norm**2)

        return distances

    def _class_membership(self, X: np.ndarray) -> Dict[str, List]:
        """
        Determine class membership for already prepared samples.

        Parameters
        ----------
        X : np.ndarray
            Data that already went through :meth:`_prepare_input`.

        Returns
        -------
        dict
            See :meth:`get_class_membership`.
        """
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

    def get_class_membership(self, X: np.ndarray) -> Dict[str, List]:
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

        X = self._prepare_input(X)
        membership = self._class_membership(X)
        distances = self._calc_combined_distances(X)

        # Non-finite distances must never win an argmin
        finite_distances = np.where(np.isfinite(distances), distances, np.inf)

        predictions = []

        for i, member_classes in enumerate(membership["member_of"]):
            if len(member_classes) == 1:
                # Single membership - assign to that class
                predictions.append(member_classes[0])

            elif len(member_classes) > 1:
                # Multiple memberships - pick closest
                # Only consider the classes that sample belongs to
                class_indices = [
                    np.where(self.classes_ == c)[0][0] for c in member_classes
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

        if any(p is None for p in predictions):
            # None marks rejected samples, so the array must be object
            return np.array(predictions, dtype=object)
        return np.array(predictions, dtype=self.classes_.dtype)

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
            totals > 0,
            similarities / np.where(totals > 0, totals, 1.0),
            1.0 / n_classes,
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
            1
            for pred, true in zip(y_pred, y)
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

        X = self._prepare_input(X)
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
