"""
JAX-accelerated utility functions for NIPALS algorithms.

The key function here is _nan_mult which handles matrix multiplication
with NaN values - the foundation of NIPALS missing data handling.

(c) 2020: Ryan Wall (original NumPy version)
JAX conversion 2024
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Optional, Tuple

# Relative convergence tests below this are dominated by float32 rounding
FLOAT32_MIN_TOL = 1e-5


def _resolve_dtype(
    dtype: Optional[str], tol_criteria: float
) -> Tuple[jnp.dtype, float]:
    """Translate the user-facing dtype string and floor the tolerance.

    Args:
        dtype (Optional[str]): 'float64', 'float32', or None to follow
            JAX's x64 setting.
        tol_criteria (float): The requested convergence tolerance.

    Raises:
        ValueError: If dtype is not recognized, or float64 is requested
            while jax_enable_x64 is off.

    Returns:
        Tuple[jnp.dtype, float]: The JAX dtype and the usable tolerance.
    """
    if dtype is None:
        dtype = "float64" if jax.config.jax_enable_x64 else "float32"

    if dtype == "float64":
        if not jax.config.jax_enable_x64:
            raise ValueError(
                "dtype='float64' requires JAX's 64-bit mode: call "
                'jax.config.update("jax_enable_x64", True) before fitting, '
                "or use dtype='float32'."
            )
        return jnp.float64, tol_criteria
    elif dtype == "float32":
        # Silently floored: the default tol_criteria is already below it
        return jnp.float32, max(tol_criteria, FLOAT32_MIN_TOL)
    else:
        raise ValueError("dtype must be 'float64' or 'float32'")


def _obs_mask(nan_mask, dtype: jnp.dtype) -> jnp.ndarray:
    """Observation mask in the form that is fastest on the active backend.

    On accelerators a boolean mask is used: _masked_mult is limited by
    memory bandwidth there, and reading one byte per entry in a fused
    kernel measured ~30% faster than a second dense matrix product (and
    the mask takes 1/8 of the memory). On CPU the dense product wins.
    """
    if jax.default_backend() == "cpu":
        return jnp.asarray(~nan_mask, dtype=dtype)
    return jnp.asarray(~nan_mask)


def _split_nan(
    data: np.ndarray, dtype: jnp.dtype
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """Move data to the device as a zero-filled array plus observation mask.

    Args:
        data (np.ndarray): Input data, possibly containing NaNs.
        dtype (jnp.dtype): The dtype to compute in.

    Returns:
        Tuple[jnp.ndarray, Optional[jnp.ndarray]]: Data with NaNs set to
            zero, and a mask that is 1/True where data was observed, see
            _obs_mask. The mask is None if there are no NaNs.
    """
    nan_mask = np.isnan(data)
    if not nan_mask.any():
        return jnp.asarray(data, dtype=dtype), None
    x0 = jnp.asarray(np.where(nan_mask, 0.0, data), dtype=dtype)
    return x0, _obs_mask(nan_mask, dtype)


def _safe_divide(numer: jnp.ndarray, denom: jnp.ndarray) -> jnp.ndarray:
    """numer / denom, returning 0 where denom is 0."""
    safe_denom = jnp.where(denom == 0.0, 1.0, denom)
    return jnp.where(denom == 0.0, 0.0, numer / safe_denom)


def _masked_mult(
    x0: jnp.ndarray, obs: jnp.ndarray, y: jnp.ndarray, use_denom: bool = True
) -> jnp.ndarray:
    """_nan_mult on data that has already been split by _split_nan.

    Row i of the result is the regression of the observed part of x[i, :]
    on the matching rows of y, computed for all rows at once.

    Args:
        x0 (jnp.ndarray): The left matrix with NaNs set to zero.
        obs (jnp.ndarray): 1/True where x was observed, 0/False where it
            was NaN. Float or boolean, see _obs_mask.
        y (jnp.ndarray): The right matrix.
        use_denom (bool, optional): Scale by 1/(y.T @ y) with appropriate
            nulls. Defaults to True.

    Returns:
        jnp.ndarray: The product of the matrix multiplication.
    """
    numer = x0 @ y
    if not use_denom:
        return numer
    if obs.dtype == jnp.bool_:
        denom = jnp.where(obs[:, :, None], (y**2)[None, :, :], 0.0).sum(axis=1)
    else:
        denom = obs @ y**2
    return _safe_divide(numer, denom)


@partial(jax.jit, static_argnames=["use_denom"])
def _nan_mult(
    x: jnp.ndarray,
    y: jnp.ndarray,
    nan_mask: Optional[jnp.ndarray] = None,
    use_denom: bool = True,
) -> jnp.ndarray:
    """Matrix multiplication for when the left matrix (X) contains NaNs.

    Args:
        x (jnp.ndarray): The left matrix in the multiplication.
        y (jnp.ndarray): The right matrix in the multiplication.
        nan_mask (jnp.ndarray, optional): The nan_mask for the left matrix.
            Defaults to None (will be computed from x).
        use_denom (bool, optional): Flag for using a normalization method.
            Defaults to True. Scales the resultant vector by 1/(y.T @ y)
            with appropriate nulls.

    Returns:
        jnp.ndarray: The product of the matrix multiplication.
    """
    if nan_mask is None:
        nan_mask = jnp.isnan(x)

    x0 = jnp.where(nan_mask, 0.0, x)
    obs = _obs_mask(nan_mask, x.dtype)

    return _masked_mult(x0, obs, y, use_denom=use_denom)
