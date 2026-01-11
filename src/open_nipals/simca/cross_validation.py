"""
Cross-validation utilities for SIMCA and dimensionality reduction models.

Provides cross-validators and cross-validation prediction functions
for both PCA and PLS models.
"""

import numpy as np
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
        random_state: Optional[int] = None
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
            train_idx = np.concatenate([indices[:i], indices[i + 1:]])
            yield train_idx, test_idx

    def get_n_splits(self, X: np.ndarray = None) -> int:
        """Return number of splits (equals number of samples)."""
        if X is None:
            raise ValueError("X must be provided for LeaveOneOutCV.get_n_splits()")
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
            test_idx = indices[fold::self.n_splits]
            train_idx = np.setdiff1d(indices, test_idx)
            yield train_idx, test_idx

    def get_n_splits(self) -> int:
        """Return number of splits."""
        return self.n_splits


def cross_val_predict_pca(
    model_class,
    X: np.ndarray,
    n_components: int,
    cv,
    **model_kwargs
) -> np.ndarray:
    """
    Generate cross-validated PCA reconstructions.

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
    X_pred = np.zeros_like(X)

    for train_idx, test_idx in cv.split(X):
        X_train = X[train_idx]
        X_test = X[test_idx]

        # Fit model on training data
        model = model_class(n_components=n_components, **model_kwargs)
        model.fit(X_train)

        # Transform and inverse transform test data
        scores = model.transform(X_test)
        X_reconstructed = model.inverse_transform(scores)

        X_pred[test_idx] = X_reconstructed

    return X_pred


def cross_val_predict_pls(
    model_class,
    X: np.ndarray,
    y: np.ndarray,
    n_components: int,
    cv,
    **model_kwargs
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

    X_pred = np.zeros_like(X)
    y_pred = np.zeros_like(y)

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
