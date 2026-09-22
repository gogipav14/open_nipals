"""
JAX-accelerated metrics for SIMCA and dimensionality reduction models.

Provides R², Q², and PRESS calculations using JAX for GPU acceleration.
"""

import jax.numpy as jnp
import numpy as np
from typing import Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from open_nipals.jax.nipalsPCA import NipalsPCA
    from open_nipals.jax.nipalsPLS import NipalsPLS


def calc_r2_x(
    X: jnp.ndarray, X_reconstructed: jnp.ndarray, per_variable: bool = False
) -> Union[float, jnp.ndarray]:
    """
    Calculate R² for X-block (explained variance) using JAX.

    R² = 1 - (SS_residual / SS_total)

    Parameters
    ----------
    X : jnp.ndarray
        Original data matrix (n_samples, n_features).
    X_reconstructed : jnp.ndarray
        Reconstructed data matrix from model.
    per_variable : bool, default=False
        If True, return R² for each variable.

    Returns
    -------
    float or jnp.ndarray
        R² value(s).
    """
    X = jnp.asarray(X)
    X_reconstructed = jnp.asarray(X_reconstructed)

    nan_mask = jnp.isnan(X) | jnp.isnan(X_reconstructed)
    X_clean = jnp.where(nan_mask, 0.0, X)
    X_rec_clean = jnp.where(nan_mask, 0.0, X_reconstructed)

    residuals = X_clean - X_rec_clean

    if per_variable:
        ss_res = jnp.sum(residuals**2, axis=0)
        ss_tot = jnp.sum(X_clean**2, axis=0)
        ss_tot = jnp.where(ss_tot == 0, 1.0, ss_tot)
        return 1.0 - (ss_res / ss_tot)
    else:
        ss_res = jnp.sum(residuals**2)
        ss_tot = jnp.sum(X_clean**2)
        return float(jnp.where(ss_tot == 0, 0.0, 1.0 - (ss_res / ss_tot)))


def calc_r2_y(
    y_true: jnp.ndarray, y_pred: jnp.ndarray, per_variable: bool = False
) -> Union[float, jnp.ndarray]:
    """
    Calculate R² for Y-block (prediction explained variance) using JAX.

    Parameters
    ----------
    y_true : jnp.ndarray
        True Y values.
    y_pred : jnp.ndarray
        Predicted Y values.
    per_variable : bool, default=False
        If True, return R² for each Y variable.

    Returns
    -------
    float or jnp.ndarray
        R² value(s).
    """
    y_true = jnp.asarray(y_true)
    y_pred = jnp.asarray(y_pred)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    nan_mask = jnp.isnan(y_true) | jnp.isnan(y_pred)
    y_true_clean = jnp.where(nan_mask, 0.0, y_true)
    y_pred_clean = jnp.where(nan_mask, 0.0, y_pred)

    residuals = y_true_clean - y_pred_clean

    if per_variable:
        ss_res = jnp.sum(residuals**2, axis=0)
        ss_tot = jnp.sum(y_true_clean**2, axis=0)
        ss_tot = jnp.where(ss_tot == 0, 1.0, ss_tot)
        return 1.0 - (ss_res / ss_tot)
    else:
        ss_res = jnp.sum(residuals**2)
        ss_tot = jnp.sum(y_true_clean**2)
        return float(jnp.where(ss_tot == 0, 0.0, 1.0 - (ss_res / ss_tot)))


def calc_r2_cumulative_pca(model: "NipalsPCA", X: jnp.ndarray) -> np.ndarray:
    """
    Calculate cumulative R² for each component in a PCA model.

    Parameters
    ----------
    model : NipalsPCA
        Fitted JAX PCA model.
    X : jnp.ndarray
        Data matrix used for fitting.

    Returns
    -------
    np.ndarray
        Cumulative R² values, one per component.
    """
    X = jnp.asarray(X)
    n_components = model.loadings.shape[1]
    r2_values = np.zeros(n_components)

    for i in range(1, n_components + 1):
        scores = model.transform(X)[:, :i]
        loadings = model.loadings[:, :i]
        X_reconstructed = scores @ loadings.T
        r2_values[i - 1] = calc_r2_x(X, X_reconstructed)

    return r2_values


def calc_r2_cumulative_pls(
    model: "NipalsPLS", X: jnp.ndarray, y: jnp.ndarray
) -> np.ndarray:
    """
    Calculate cumulative R² for Y-block predictions in a PLS model.

    Parameters
    ----------
    model : NipalsPLS
        Fitted JAX PLS model.
    X : jnp.ndarray
        X data matrix.
    y : jnp.ndarray
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
    y_true: jnp.ndarray, y_pred_cv: jnp.ndarray, per_variable: bool = False
) -> Union[float, jnp.ndarray]:
    """
    Calculate PRESS using JAX.

    Only genuinely missing observations (NaN in ``y_true``) are
    excluded. A non-finite *prediction* for an observed target is a
    failure of the model, not missing data, so it gives an infinite
    PRESS rather than free credit.

    Parameters
    ----------
    y_true : jnp.ndarray
        True values.
    y_pred_cv : jnp.ndarray
        Cross-validated predictions.
    per_variable : bool, default=False
        If True, return PRESS for each variable.

    Returns
    -------
    float or jnp.ndarray
        PRESS value(s). Infinite where a prediction failed.
    """
    y_true = jnp.asarray(y_true)
    y_pred_cv = jnp.asarray(y_pred_cv)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred_cv.ndim == 1:
        y_pred_cv = y_pred_cv.reshape(-1, 1)

    observed = ~jnp.isnan(y_true)
    failed = observed & ~jnp.isfinite(y_pred_cv)
    valid = observed & ~failed

    residuals = jnp.where(valid, y_true - y_pred_cv, 0.0)

    if per_variable:
        press = jnp.sum(residuals**2, axis=0)
        return jnp.where(jnp.any(failed, axis=0), jnp.inf, press)

    if bool(jnp.any(failed)):
        return float("inf")
    return float(jnp.sum(residuals**2))


def calc_q2(
    y_true: jnp.ndarray,
    y_pred_cv: jnp.ndarray,
    ss_total: Optional[float] = None,
) -> float:
    """
    Calculate Q² (cross-validated R²) using JAX.

    Parameters
    ----------
    y_true : jnp.ndarray
        True values.
    y_pred_cv : jnp.ndarray
        Cross-validated predictions.
    ss_total : float, optional
        Total sum of squares.

    Returns
    -------
    float
        Q² value.
    """
    y_true = jnp.asarray(y_true)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)

    press = calc_press(y_true, y_pred_cv)

    if ss_total is None:
        nan_mask = jnp.isnan(y_true)
        y_clean = jnp.where(nan_mask, 0.0, y_true)
        ss_total = float(jnp.sum(y_clean**2))

    if ss_total == 0:
        return 0.0

    return 1.0 - (press / ss_total)


def calc_q2_cumulative_pca(
    model_class,
    X: jnp.ndarray,
    cv,
    max_components: int,
    n_element_groups: int = 7,
    scale: bool = False,
    **model_kwargs,
) -> np.ndarray:
    """
    Calculate cumulative Q² for PCA using element-wise cross-validation.

    The cross-validation loop is the NumPy one; only the PCA models are
    JAX models, so the Q² definition stays identical to the reference
    implementation.

    Parameters
    ----------
    model_class : class
        JAX PCA model class.
    X : jnp.ndarray
        Data matrix.
    cv : CrossValidator
        Cross-validation object.
    max_components : int
        Maximum number of components.
    n_element_groups : int, default=7
        Number of element groups held out within the validation rows.
    scale : bool, default=False
        Re-estimate the column standard deviations inside each fold too.
    **model_kwargs
        Additional arguments for model constructor.

    Returns
    -------
    np.ndarray
        Q² values, one per component.
    """
    from open_nipals.simca.metrics import (
        calc_q2_cumulative_pca as _calc_q2_cumulative_pca,
    )

    return _calc_q2_cumulative_pca(
        model_class,
        np.asarray(X),
        cv,
        max_components,
        n_element_groups,
        scale=scale,
        **model_kwargs,
    )


def calc_q2_cumulative_pls(
    model_class,
    X: jnp.ndarray,
    y: jnp.ndarray,
    cv,
    max_components: int,
    **model_kwargs,
) -> np.ndarray:
    """
    Calculate cumulative Q² for PLS Y-block using cross-validation with JAX.

    Parameters
    ----------
    model_class : class
        JAX PLS model class.
    X : jnp.ndarray
        X data matrix.
    y : jnp.ndarray
        Y data matrix.
    cv : CrossValidator
        Cross-validation object.
    max_components : int
        Maximum number of components.
    **model_kwargs
        Additional arguments for model constructor.

    Returns
    -------
    np.ndarray
        Q² values for Y predictions, one per component.
    """
    from .cross_validation import cross_val_predict_pls

    y = np.asarray(y)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    nan_mask = np.isnan(y)
    y_clean = np.where(nan_mask, 0.0, y)
    ss_total = np.sum(y_clean**2)

    q2_values = np.zeros(max_components)

    for n_comp in range(1, max_components + 1):
        _, y_pred_cv = cross_val_predict_pls(
            model_class, X, y, n_comp, cv, **model_kwargs
        )
        press = calc_press(y, y_pred_cv)
        q2_values[n_comp - 1] = (
            1.0 - (press / ss_total) if ss_total > 0 else 0.0
        )

    return q2_values
