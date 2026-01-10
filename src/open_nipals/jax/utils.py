"""
JAX-accelerated utility functions for NIPALS algorithms.

The key function here is _nan_mult which handles matrix multiplication
with NaN values - the foundation of NIPALS missing data handling.

(c) 2020: Ryan Wall (original NumPy version)
JAX conversion 2024
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Optional


@partial(jax.jit, static_argnames=["use_denom"])
def _nan_mult(
    x: jnp.ndarray,
    y: jnp.ndarray,
    nan_mask: Optional[jnp.ndarray] = None,
    use_denom: bool = True,
) -> jnp.ndarray:
    """Matrix multiplication for when the left matrix (X) contains NaNs.

    This is the JAX-vectorized version using vmap instead of Python loops.

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

    def single_row_mult(x_row: jnp.ndarray, mask_row: jnp.ndarray) -> jnp.ndarray:
        """Process a single row with NaN handling.

        Uses jnp.where instead of boolean indexing (not supported in JAX).
        """
        # Zero out NaN positions in x_row
        x_clean = jnp.where(mask_row, 0.0, x_row)

        # Zero out corresponding rows in y (where x has NaN)
        # mask_row is shape (cols_x,), y is shape (cols_x, cols_y)
        y_clean = jnp.where(mask_row[:, None], 0.0, y)

        # Compute numerator: x_clean @ y_clean
        numerator = x_clean @ y_clean

        if use_denom:
            # Denominator: sum of y_clean^2 for each column
            # This is equivalent to y[not_null].T @ y[not_null] for each output col
            denom = jnp.sum(y_clean**2, axis=0)
            # Avoid division by zero - if denom is 0, result should be 0
            # Use where to handle this safely
            safe_denom = jnp.where(denom == 0.0, 1.0, denom)
            result = jnp.where(denom == 0.0, 0.0, numerator / safe_denom)
            return result
        else:
            return numerator

    # Vectorize over all rows using vmap
    return jax.vmap(single_row_mult)(x, nan_mask)


@jax.jit
def _compute_convergence(t_new: jnp.ndarray, t_old: jnp.ndarray) -> jnp.ndarray:
    """Compute convergence metric for NIPALS iteration.

    Args:
        t_new: New score vector
        t_old: Old score vector

    Returns:
        Convergence metric (should approach 0)
    """
    score_diff = t_old - t_new
    return jnp.sqrt(score_diff.T @ score_diff) / jnp.sqrt(t_new.T @ t_new)


@jax.jit
def _normalize_vector(v: jnp.ndarray) -> jnp.ndarray:
    """Normalize a vector to unit length.

    Args:
        v: Input vector

    Returns:
        Normalized vector
    """
    return v / jnp.sqrt(v.T @ v)
