"""
JAX-compatible cross-validation utilities for SIMCA.

Uses NumPy for CV splits but JAX models for fitting/prediction.
"""

import numpy as np
import jax.numpy as jnp
from typing import Iterator, Tuple, Optional

# Import CV splitters from base module (they're just index generators)
from open_nipals.simca.cross_validation import (
    KFoldCV,
    LeaveOneOutCV,
    VenetianBlindsCV,
)


def cross_val_predict_pca(
    model_class,
    X: np.ndarray,
    n_components: int,
    cv,
    **model_kwargs
) -> np.ndarray:
    """
    Generate cross-validated PCA reconstructions using JAX models.

    Parameters
    ----------
    model_class : class
        JAX PCA model class.
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

        # Fit JAX model on training data
        model = model_class(n_components=n_components, **model_kwargs)
        model.fit(X_train)

        # Transform and inverse transform test data
        scores = model.transform(X_test)
        X_reconstructed = model.inverse_transform(scores)

        # Convert JAX arrays back to NumPy
        X_pred[test_idx] = np.asarray(X_reconstructed)

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
    Generate cross-validated PLS predictions using JAX models.

    Parameters
    ----------
    model_class : class
        JAX PLS model class.
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

        # Fit JAX model on training data
        model = model_class(n_components=n_components, **model_kwargs)
        model.fit(X_train, y_train)

        # Predict test data
        y_pred[test_idx] = np.asarray(model.predict(X_test))

        # Reconstruct X
        scores = model.transform(X_test)
        X_pred[test_idx] = np.asarray(model.inverse_transform(scores))

    return X_pred, y_pred
