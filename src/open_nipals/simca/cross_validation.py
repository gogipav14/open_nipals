"""
Cross-validation utilities for SIMCA and dimensionality reduction models.

Provides cross-validators and cross-validation prediction functions
for both PCA and PLS models.
"""

import numpy as np
import warnings
from typing import Iterator, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from open_nipals.nipalsPCA import NipalsPCA
    from open_nipals.nipalsPLS import NipalsPLS


class KFoldCV:
    """
    K-Fold cross-validation splitter.

    Parameters
    ----------
    n_splits : int, default=7
        Number of folds.
    shuffle : bool, default=False
        Whether to shuffle data before splitting.
    random_state : int, optional
        Random seed for shuffling.
    """

    def __init__(
        self,
        n_splits: int = 7,
        shuffle: bool = False,
        random_state: Optional[int] = None,
    ):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for each fold.

        Parameters
        ----------
        X : np.ndarray
            Data to split.

        Yields
        ------
        train_idx : np.ndarray
            Training indices.
        test_idx : np.ndarray
            Test indices.
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(indices)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits)
        fold_sizes[: n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_idx = indices[start:stop]
            train_idx = np.concatenate([indices[:start], indices[stop:]])
            yield train_idx, test_idx
            current = stop

    def get_n_splits(self) -> int:
        """Return number of splits."""
        return self.n_splits


class LeaveOneOutCV:
    """
    Leave-One-Out cross-validation splitter.

    Each sample is used once as test set while remaining samples form training.
    """

    def split(self, X: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices.

        Parameters
        ----------
        X : np.ndarray
            Data to split.

        Yields
        ------
        train_idx : np.ndarray
            Training indices.
        test_idx : np.ndarray
            Test indices (single sample).
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        for i in range(n_samples):
            test_idx = np.array([i])
            train_idx = np.concatenate([indices[:i], indices[i + 1 :]])
            yield train_idx, test_idx

    def get_n_splits(self, X: np.ndarray = None) -> int:
        """Return number of splits (equals number of samples)."""
        if X is None:
            raise ValueError(
                "X must be provided for LeaveOneOutCV.get_n_splits()"
            )
        return X.shape[0]


class VenetianBlindsCV:
    """
    Venetian Blinds cross-validation splitter.

    Systematically selects every n-th sample for each fold,
    ensuring even distribution across the dataset.

    Parameters
    ----------
    n_splits : int, default=7
        Number of folds.
    """

    def __init__(self, n_splits: int = 7):
        self.n_splits = n_splits

    def split(self, X: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices using venetian blinds pattern.

        Parameters
        ----------
        X : np.ndarray
            Data to split.

        Yields
        ------
        train_idx : np.ndarray
            Training indices.
        test_idx : np.ndarray
            Test indices.
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        for fold in range(self.n_splits):
            # Select every n_splits-th sample starting at fold
            test_idx = indices[fold :: self.n_splits]
            train_idx = np.setdiff1d(indices, test_idx)
            yield train_idx, test_idx

    def get_n_splits(self) -> int:
        """Return number of splits."""
        return self.n_splits


def _element_groups(n_rows: int, n_cols: int, n_groups: int) -> np.ndarray:
    """
    Assign every element of a matrix to one of n_groups groups.

    A diagonal (venetian blind) pattern is used, so every group touches
    every row and every column roughly evenly.

    Parameters
    ----------
    n_rows : int
        Number of rows.
    n_cols : int
        Number of columns.
    n_groups : int
        Number of element groups.

    Returns
    -------
    np.ndarray
        Integer group index for each element, shape (n_rows, n_cols).
    """
    rows = np.arange(n_rows).reshape(-1, 1)
    cols = np.arange(n_cols).reshape(1, -1)
    return (rows + cols) % n_groups


def _class_statistics(
    X: np.ndarray, scale: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Centre and scale vectors of a class, as used by SIMCA.

    Constant columns (all observed values identical) are centred on
    that value, not on np.nanmean, whose rounding error is not zero
    (256 for identical values of 1.8e18) and would otherwise be left
    as a spurious residual. Their scale is 1 instead of the ~1e-17
    standard deviation that rounding produces for identical decimals,
    which would turn the noise into a full-variance feature. Any
    genuine variation, however small, is kept.

    Parameters
    ----------
    X : np.ndarray
        Training rows of one class.
    scale : bool
        Return the standard deviations (ddof=1), otherwise all ones.

    Returns
    -------
    mean : np.ndarray
        Centre per column.
    std : np.ndarray
        Scale per column.
    """
    with warnings.catch_warnings():
        # all-NaN columns are constant too, no need for the warning
        warnings.simplefilter("ignore", RuntimeWarning)
        upper = np.nanmax(X, axis=0)
        constant = upper == np.nanmin(X, axis=0)
        mean = np.where(constant, upper, np.nanmean(X, axis=0))
    if scale:
        std = np.where(constant, 1.0, np.nanstd(X, axis=0, ddof=1))
    else:
        std = np.ones(X.shape[1])
    return mean, std


def cross_val_press_pca(
    model_class,
    X: np.ndarray,
    n_components: int,
    cv,
    n_element_groups: int = 7,
    scale: bool = False,
    **model_kwargs,
) -> Tuple[float, float]:
    """
    Element-wise (Wold) cross-validated PRESS for a PCA model.

    For every fold the model is fitted on the training rows only, with
    the mean (and, with ``scale``, the standard deviation) re-estimated
    on those rows. The elements of the validation
    rows are then split into ``n_element_groups`` groups; one group at a
    time is set to NaN, the scores of the validation rows are computed
    from the *remaining* elements with
    ``transform(method='projection')`` and the held-out elements are
    reconstructed with ``inverse_transform``. PRESS is accumulated only
    over the held-out elements, so no element ever contributes to its
    own prediction.

    Parameters
    ----------
    model_class : class
        PCA model class (e.g., NipalsPCA).
    X : np.ndarray
        Data matrix (n_samples, n_features).
    n_components : int
        Number of components.
    cv : CrossValidator
        Cross-validation object with split() method.
    n_element_groups : int, default=7
        Number of element groups per fold. Clipped to [2, n_features].
    scale : bool, default=False
        Also divide by the training rows' standard deviation (ddof=1),
        as :class:`~open_nipals.simca.SIMCA` does per class.
    **model_kwargs
        Additional arguments for model constructor.

    Returns
    -------
    press : float
        Sum of squared prediction errors over all held-out elements.
    ss_total : float
        Sum of squares of the same elements, after the within-fold
        centring, i.e. the reference for Q² = 1 - press / ss_total.

    Raises
    ------
    ValueError
        If a training fold has too few rows for n_components.
    """
    X = np.asarray(X, dtype=float)
    n_groups = int(max(2, min(n_element_groups, X.shape[1])))

    press = 0.0
    ss_total = 0.0

    for train_idx, test_idx in cv.split(X):
        if len(test_idx) == 0:
            continue
        if len(train_idx) <= n_components + 1:
            raise ValueError(
                f"A cross-validation fold has {len(train_idx)} training "
                f"rows, which is too few for {n_components} components. "
                "Use fewer folds, fewer components or more samples."
            )

        # Preprocessing is fitted on the training part of the fold only
        X_train = X[train_idx]
        mean, std = _class_statistics(X_train, scale)
        X_train = (X_train - mean) / std
        X_val = (X[test_idx] - mean) / std

        model = model_class(n_components=n_components, **model_kwargs)
        model.fit(X_train)

        observed = ~np.isnan(X_val)
        groups = _element_groups(*X_val.shape, n_groups)
        ss_total += float(np.sum(np.where(observed, X_val, 0.0) ** 2))

        for group in range(n_groups):
            held_out = (groups == group) & observed
            if not np.any(held_out):
                continue

            X_masked = np.where(held_out, np.nan, X_val)
            scores = np.asarray(model.transform(X_masked, method="projection"))
            recon = np.asarray(model.inverse_transform(scores))
            resid = X_val[held_out] - recon[held_out]
            press += float(np.sum(resid**2))

    return press, ss_total


def cross_val_predict_pca(
    model_class, X: np.ndarray, n_components: int, cv, **model_kwargs
) -> np.ndarray:
    """
    Generate cross-validated PCA reconstructions.

    Every validation row is reconstructed from its own scores, so the
    row informs its own reconstruction. That makes this function useful
    for inspecting reconstructions, but *not* for Q²: use
    :func:`cross_val_press_pca` for that.

    Parameters
    ----------
    model_class : class
        PCA model class (e.g., NipalsPCA).
    X : np.ndarray
        Data matrix (n_samples, n_features).
    n_components : int
        Number of components.
    cv : CrossValidator
        Cross-validation object.
    **model_kwargs
        Additional arguments for model constructor.

    Returns
    -------
    np.ndarray
        Cross-validated X reconstructions.
    """
    n_samples, n_features = X.shape
    # float buffers: integer inputs must not truncate the predictions
    X_pred = np.zeros(X.shape, dtype=float)

    for train_idx, test_idx in cv.split(X):
        X_train = X[train_idx]
        X_test = X[test_idx]

        # Fit model on training data
        model = model_class(n_components=n_components, **model_kwargs)
        model.fit(X_train)

        # Transform and inverse transform test data
        scores = np.asarray(model.transform(X_test))
        X_reconstructed = np.asarray(model.inverse_transform(scores))

        X_pred[test_idx] = X_reconstructed

    return X_pred


def cross_val_predict_pls(
    model_class,
    X: np.ndarray,
    y: np.ndarray,
    n_components: int,
    cv,
    **model_kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate cross-validated PLS predictions.

    Parameters
    ----------
    model_class : class
        PLS model class (e.g., NipalsPLS).
    X : np.ndarray
        X data matrix.
    y : np.ndarray
        Y data matrix.
    n_components : int
        Number of components.
    cv : CrossValidator
        Cross-validation object.
    **model_kwargs
        Additional arguments for model constructor.

    Returns
    -------
    X_pred : np.ndarray
        Cross-validated X reconstructions.
    y_pred : np.ndarray
        Cross-validated Y predictions.
    """
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    n_samples, n_features_x = X.shape
    n_targets = y.shape[1]

    # float buffers: integer inputs must not truncate the predictions
    X_pred = np.zeros(X.shape, dtype=float)
    y_pred = np.zeros(y.shape, dtype=float)

    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit model on training data
        model = model_class(n_components=n_components, **model_kwargs)
        model.fit(X_train, y_train)

        # Predict test data
        y_pred[test_idx] = model.predict(X_test)

        # Reconstruct X
        scores = model.transform(X_test)
        X_pred[test_idx] = model.inverse_transform(scores)

    return X_pred, y_pred
