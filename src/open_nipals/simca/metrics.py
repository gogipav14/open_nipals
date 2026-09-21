"""
Metrics for SIMCA and dimensionality reduction models.

Provides R², Q², and PRESS calculations for both PCA (X-block)
and PLS (Y-block) models.
"""

import numpy as np
from typing import Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from open_nipals.nipalsPCA import NipalsPCA
    from open_nipals.nipalsPLS import NipalsPLS


def calc_r2_x(
    X: np.ndarray,
    X_reconstructed: np.ndarray,
    per_variable: bool = False
) -> Union[float, np.ndarray]:
    """
    Calculate R² for X-block (explained variance).

    R² = 1 - (SS_residual / SS_total)

    Parameters
    ----------
    X : np.ndarray
        Original data matrix (n_samples, n_features).
    X_reconstructed : np.ndarray
        Reconstructed data matrix from model.
    per_variable : bool, default=False
        If True, return R² for each variable. Otherwise return overall R².

    Returns
    -------
    float or np.ndarray
        R² value(s). Float if per_variable=False, array if per_variable=True.
    """
    # Handle NaN values - only use non-NaN pairs
    nan_mask = np.isnan(X) | np.isnan(X_reconstructed)
    X_clean = np.where(nan_mask, 0.0, X)
    X_rec_clean = np.where(nan_mask, 0.0, X_reconstructed)
    valid_count = (~nan_mask).astype(float)

    residuals = X_clean - X_rec_clean

    if per_variable:
        # Per-variable R²
        ss_res = np.sum(residuals ** 2, axis=0)
        ss_tot = np.sum(X_clean ** 2, axis=0)
        # Avoid division by zero
        ss_tot = np.where(ss_tot == 0, 1.0, ss_tot)
        return 1.0 - (ss_res / ss_tot)
    else:
        # Overall R²
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum(X_clean ** 2)
        if ss_tot == 0:
            return 0.0
        return 1.0 - (ss_res / ss_tot)


def calc_r2_y(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    per_variable: bool = False
) -> Union[float, np.ndarray]:
    """
    Calculate R² for Y-block (prediction explained variance).

    R² = 1 - (SS_residual / SS_total)

    Parameters
    ----------
    y_true : np.ndarray
        True Y values (n_samples, n_targets).
    y_pred : np.ndarray
        Predicted Y values.
    per_variable : bool, default=False
        If True, return R² for each Y variable.

    Returns
    -------
    float or np.ndarray
        R² value(s).
    """
    # Ensure 2D
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    # Handle NaN
    nan_mask = np.isnan(y_true) | np.isnan(y_pred)
    y_true_clean = np.where(nan_mask, 0.0, y_true)
    y_pred_clean = np.where(nan_mask, 0.0, y_pred)

    residuals = y_true_clean - y_pred_clean

    if per_variable:
        ss_res = np.sum(residuals ** 2, axis=0)
        ss_tot = np.sum(y_true_clean ** 2, axis=0)
        ss_tot = np.where(ss_tot == 0, 1.0, ss_tot)
        return 1.0 - (ss_res / ss_tot)
    else:
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum(y_true_clean ** 2)
        if ss_tot == 0:
            return 0.0
        return 1.0 - (ss_res / ss_tot)


def calc_r2_cumulative_pca(
    model: "NipalsPCA",
    X: np.ndarray
) -> np.ndarray:
    """
    Calculate cumulative R² for each component in a PCA model.

    Parameters
    ----------
    model : NipalsPCA
        Fitted PCA model.
    X : np.ndarray
        Data matrix used for fitting.

    Returns
    -------
    np.ndarray
        Cumulative R² values, one per component.
    """
    n_components = model.loadings.shape[1]
    r2_values = np.zeros(n_components)

    for i in range(1, n_components + 1):
        # Use first i components for reconstruction
        scores = model.transform(X)[:, :i]
        loadings = model.loadings[:, :i]
        X_reconstructed = scores @ loadings.T
        r2_values[i - 1] = calc_r2_x(X, X_reconstructed)

    return r2_values


def calc_r2_cumulative_pls(
    model: "NipalsPLS",
    X: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """
    Calculate cumulative R² for Y-block predictions in a PLS model.

    Parameters
    ----------
    model : NipalsPLS
        Fitted PLS model.
    X : np.ndarray
        X data matrix.
    y : np.ndarray
        Y data matrix.

    Returns
    -------
    np.ndarray
        Cumulative R² values for Y predictions, one per component.
    """
    n_components = model.loadings_x.shape[1]
    r2_values = np.zeros(n_components)

    # Store original n_components
    original_n_components = model.n_components

    for i in range(1, n_components + 1):
        # Temporarily set number of components
        model.n_components = i
        y_pred = model.predict(X)
        r2_values[i - 1] = calc_r2_y(y, y_pred)

    # Restore original n_components
    model.n_components = original_n_components

    return r2_values


def calc_press(
    y_true: np.ndarray,
    y_pred_cv: np.ndarray,
    per_variable: bool = False
) -> Union[float, np.ndarray]:
    """
    Calculate PRESS (Predicted Residual Error Sum of Squares).

    PRESS = sum((y_true - y_pred_cv)^2)

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred_cv : np.ndarray
        Cross-validated predictions.
    per_variable : bool, default=False
        If True, return PRESS for each variable.

    Returns
    -------
    float or np.ndarray
        PRESS value(s).
    """
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred_cv.ndim == 1:
        y_pred_cv = y_pred_cv.reshape(-1, 1)

    nan_mask = np.isnan(y_true) | np.isnan(y_pred_cv)
    residuals = np.where(nan_mask, 0.0, y_true - y_pred_cv)

    if per_variable:
        return np.sum(residuals ** 2, axis=0)
    else:
        return np.sum(residuals ** 2)


def calc_q2(
    y_true: np.ndarray,
    y_pred_cv: np.ndarray,
    ss_total: Optional[float] = None
) -> float:
    """
    Calculate Q² (cross-validated R²).

    Q² = 1 - (PRESS / SS_total)

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred_cv : np.ndarray
        Cross-validated predictions.
    ss_total : float, optional
        Total sum of squares. If None, calculated from y_true.

    Returns
    -------
    float
        Q² value.
    """
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)

    press = calc_press(y_true, y_pred_cv)

    if ss_total is None:
        nan_mask = np.isnan(y_true)
        y_clean = np.where(nan_mask, 0.0, y_true)
        ss_total = np.sum(y_clean ** 2)

    if ss_total == 0:
        return 0.0

    return 1.0 - (press / ss_total)


def calc_q2_cumulative_pca(
    model_class,
    X: np.ndarray,
    cv,
    max_components: int,
    **model_kwargs
) -> np.ndarray:
    """
    Calculate cumulative Q² for PCA using cross-validation.

    Parameters
    ----------
    model_class : class
        PCA model class (e.g., NipalsPCA).
    X : np.ndarray
        Data matrix.
    cv : CrossValidator
        Cross-validation object with split() method.
    max_components : int
        Maximum number of components to evaluate.
    **model_kwargs
        Additional arguments for model constructor.

    Returns
    -------
    np.ndarray
        Q² values, one per component.
    """
    from .cross_validation import cross_val_predict_pca

    n_samples, n_features = X.shape
    nan_mask = np.isnan(X)
    X_clean = np.where(nan_mask, 0.0, X)
    ss_total = np.sum(X_clean ** 2)

    q2_values = np.zeros(max_components)

    for n_comp in range(1, max_components + 1):
        X_pred_cv = cross_val_predict_pca(
            model_class, X, n_comp, cv, **model_kwargs
        )
        press = calc_press(X, X_pred_cv)
        q2_values[n_comp - 1] = 1.0 - (press / ss_total) if ss_total > 0 else 0.0

    return q2_values


def calc_q2_cumulative_pls(
    model_class,
    X: np.ndarray,
    y: np.ndarray,
    cv,
    max_components: int,
    **model_kwargs
) -> np.ndarray:
    """
    Calculate cumulative Q² for PLS Y-block using cross-validation.

    Parameters
    ----------
    model_class : class
        PLS model class (e.g., NipalsPLS).
    X : np.ndarray
        X data matrix.
    y : np.ndarray
        Y data matrix.
    cv : CrossValidator
        Cross-validation object with split() method.
    max_components : int
        Maximum number of components to evaluate.
    **model_kwargs
        Additional arguments for model constructor.

    Returns
    -------
    np.ndarray
        Q² values for Y predictions, one per component.
    """
    from .cross_validation import cross_val_predict_pls

    if y.ndim == 1:
        y = y.reshape(-1, 1)

    nan_mask = np.isnan(y)
    y_clean = np.where(nan_mask, 0.0, y)
    ss_total = np.sum(y_clean ** 2)

    q2_values = np.zeros(max_components)

    for n_comp in range(1, max_components + 1):
        _, y_pred_cv = cross_val_predict_pls(
            model_class, X, y, n_comp, cv, **model_kwargs
        )
        press = calc_press(y, y_pred_cv)
        q2_values[n_comp - 1] = 1.0 - (press / ss_total) if ss_total > 0 else 0.0

    return q2_values
