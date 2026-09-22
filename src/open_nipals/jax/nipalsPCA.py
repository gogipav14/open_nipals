"""
JAX-accelerated PCA using the NIPALS algorithm.

This provides GPU acceleration for PCA with missing data support.
The algorithm is the same as used in SIMCA.

One of the most concise definitions can be found in this paper on page 7:
    Geladi, P.; Kowalski, B. R. Partial Least-Squares Regression: A Tutorial.
    Analytica Chimica Acta 1986, 185, 1–17.
    https://doi.org/10.1016/0003-2670(86)80028-9.

For the transformation part also see:
    Nelson, P. R. C.; Taylor, P. A.; MacGregor, J. F. Missing data methods
    in PCA and PLS: Score calculations with incomplete observations.
    Chemometrics and Intelligent Laboratory Systems 1996, 35(1), 45-65.

(c) 2020-2021: Ryan Wall (lead), David Ochsenbein (original NumPy version)
revised 2024: Niels Schlusser
JAX conversion 2024
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
import warnings
from scipy.stats import f as F_dist
from functools import partial
from open_nipals.nipalsPCA import NipalsPCA as _ReferenceNipalsPCA
from open_nipals.jax.utils import (
    _full_precision_methods,
    _masked_mult,
    _resolve_dtype,
    _split_nan,
)
from typing import Optional, Tuple


def _pinv_rtol(dtype: jnp.dtype) -> float:
    """Relative singular value cutoff for pinv.

    numpy's default in float64 and its float32 counterpart. Fixed values,
    so the cutoff does not depend on the (padded) matrix dimensions.
    """
    return 1e-15 if dtype == jnp.float64 else 1e-6


def _start_column(data: jnp.ndarray, preferred: int) -> jnp.ndarray:
    """Column of data to start the NIPALS iteration from.

    The preferred column, unless it is all zeros (that would make every
    iteration NaN), then the column with the largest sum of squares.
    """
    sum_sq = jnp.sum(data**2, axis=0)
    fallback = jnp.argmax(sum_sq)
    col = jnp.where(sum_sq[preferred] > 0, preferred, fallback)
    return jax.lax.dynamic_slice_in_dim(data, col, 1, axis=1)


@partial(jax.jit, static_argnames=["n_add"])
def _fit_components(
    x0: jnp.ndarray,
    obs: Optional[jnp.ndarray],
    n_add: int,
    tol: float,
    max_iter: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Fit n_add NIPALS components, deflating after each one.

    Mirrors the loop in open_nipals.nipalsPCA.NipalsPCA._add_components.
    Compiled once per (shape, n_add, NaN/no NaN) and reused after that.

    Args:
        x0 (jnp.ndarray): The (already deflated) data, NaNs set to zero.
        obs (Optional[jnp.ndarray]): 1 where x0 was observed, None if
            there are no NaNs.
        n_add (int): Number of components to fit.
        tol (float): The convergence threshold.
        max_iter (int): The maximum number of iterations per component.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: scores (n x n_add),
            loadings (m x n_add) and iterations used per component.
    """
    _, m = x0.shape

    def not_converged(state):
        t_new, t_old, _, num_iter = state
        # guard against zero norm
        den = jnp.maximum(jnp.linalg.norm(t_new), 1e-12)
        conv_test = jnp.linalg.norm(t_old - t_new) / den
        # written so that a NaN conv_test counts as not converged
        not_done = ~(conv_test < tol) & (num_iter < max_iter)
        # a non-finite iterate never recovers, stop and report it
        return (num_iter == 0) | (not_done & jnp.isfinite(conv_test))

    def one_component(x0, _):
        def iterate(state):
            t_old, _, _, num_iter = state
            if obs is None:
                loadings_loc = (x0.T @ t_old) / (t_old.T @ t_old)
                loadings_loc = loadings_loc / jnp.linalg.norm(loadings_loc)
                t_new = (x0 @ loadings_loc) / (loadings_loc.T @ loadings_loc)
            else:
                loadings_loc = _masked_mult(x0.T, obs.T, t_old)
                loadings_loc = loadings_loc / jnp.linalg.norm(loadings_loc)
                t_new = _masked_mult(x0, obs, loadings_loc)
            return (t_new, t_old, loadings_loc, num_iter + 1)

        # choose a column of input_array, NaNs are already zero
        t_init = _start_column(x0, 0)
        state = (
            t_init,
            jnp.zeros_like(t_init),
            jnp.zeros((m, 1), x0.dtype),
            0,
        )
        t_new, _, loadings_loc, num_iter = jax.lax.while_loop(
            not_converged, iterate, state
        )

        # Deflate the input matrix, keeping missing entries at zero
        x0 = x0 - t_new @ loadings_loc.T
        if obs is not None:
            x0 = x0 * obs
        return x0, (t_new[:, 0], loadings_loc[:, 0], num_iter)

    _, (scores, loadings, num_iters) = jax.lax.scan(
        one_component, x0, None, length=n_add
    )
    return scores.T, loadings.T, num_iters


@jax.jit
def _transform_naive(
    x0: jnp.ndarray, obs: Optional[jnp.ndarray], loadings: jnp.ndarray
) -> jnp.ndarray:
    """Single component projection, one loading after the other."""

    def one_component(resids, loading):
        loading = loading[:, None]
        if obs is None:
            score_col = (resids @ loading) / (loading.T @ loading)
        else:
            score_col = _masked_mult(resids, obs, loading)
        resids = resids - score_col @ loading.T
        if obs is not None:
            resids = resids * obs
        return resids, score_col[:, 0]

    _, scores = jax.lax.scan(one_component, x0, loadings.T)
    return scores.T


@jax.jit
def _transform_projection(
    x0: jnp.ndarray, obs: jnp.ndarray, loadings: jnp.ndarray
) -> jnp.ndarray:
    """Projection to the model plane, one least squares fit per row.

    Row i solves (P_i.T @ P_i) t = P_i.T @ x_i with P_i the loadings of
    the observed columns. All P_i.T @ P_i come out of a single matrix
    product of obs with the pairwise products of the loadings.
    """
    n, _ = x0.shape
    m, k = loadings.shape
    pairwise = (loadings[:, :, None] * loadings[:, None, :]).reshape(m, k * k)
    gram = (obs.astype(x0.dtype) @ pairwise).reshape(n, k, k)
    nom = x0 @ loadings
    gram_inv = jnp.linalg.pinv(gram, rtol=_pinv_rtol(x0.dtype), hermitian=True)
    return (gram_inv @ nom[:, :, None])[:, :, 0]


@partial(jax.jit, static_argnames=["batch_size"])
def _fill_conditional_mean(
    x0: jnp.ndarray, obs: jnp.ndarray, data_cov: jnp.ndarray, batch_size: int
) -> jnp.ndarray:
    """Replace missing values by their conditional mean given the rest.

    For each row z_missing = S12 @ inv(S22) @ x_observed, with S the model
    covariance of the data. To keep shapes static the observed block S22
    is embedded in an (m x m) system that is diagonal on the missing
    entries, which leaves the (pseudo) inverse on the observed entries
    unchanged.
    """

    def fill_row(row):
        x_row, obs_row = row
        obs_row = obs_row.astype(x_row.dtype)
        # Pad with the largest observed variance: it never exceeds the top
        # eigenvalue of S22, so the pinv cutoff stays that of S22 alone
        pad = jnp.max(jnp.diag(data_cov) * obs_row)
        system = data_cov * jnp.outer(obs_row, obs_row)
        system = system + pad * jnp.diag(1 - obs_row)
        system_inv = jnp.linalg.pinv(
            system, rtol=_pinv_rtol(x0.dtype), hermitian=True
        )
        z_hash = data_cov @ (system_inv @ x_row)
        return jnp.where(obs_row > 0, x_row, z_hash)

    return jax.lax.map(fill_row, (x0, obs), batch_size=batch_size)


@_full_precision_methods
class NipalsPCA(BaseEstimator, TransformerMixin):
    """JAX-accelerated PCA using the NIPALS algorithm.

    This class provides the same interface as the NumPy version but uses
    JAX for GPU acceleration on large datasets.

    Attributes:
    ----------
    n_components : int
        The number of principal components.
    max_iter : int
        The max number of iterations for the fitting step.
    tol_criteria : float
        The convergence tolerance criterion.
    loadings : jnp.ndarray
        The loadings vectors of the PCA model.
    fit_scores : jnp.ndarray
        The fitted scores of the PCA model.
    fit_data : jnp.ndarray
        The data used to fit the model.
    mean_centered : bool
        Whether or not the original data is mean-centered.
    fitted_components : int
        The number of current LVs in the model (0 if not fitted yet.)
    """

    def __init__(
        self,
        n_components: int = 2,
        max_iter: int = 10000,
        tol_criteria: float = 1e-6,
        mean_centered: bool = True,
        dtype: Optional[str] = None,
    ):
        """The constructor for the NipalsPCA class.

        Args:
            n_components (int, optional): The number of principle components.
                Defaults to 2.
            max_iter (int, optional): The maximum number of iterations until
                convergence. Defaults to 10000.
            tol_criteria (float, optional): The convergence threshold.
                Defaults to 1e-6.
            mean_centered (bool, optional): Whether or not the data is already
                mean-centered. Defaults to True.
            dtype (str, optional): Precision to fit and transform in, either
                'float64' (matches the NumPy version, requires JAX's 64-bit
                mode) or 'float32' (faster and half the memory on GPU,
                tol_criteria floored at 1e-5). Results are always returned
                as float64 numpy arrays. Defaults to None, which follows
                JAX's jax_enable_x64 setting.

        Returns:
            NipalsPCA object
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol_criteria = tol_criteria
        self.mean_centered = mean_centered
        self.dtype = dtype

        self.loadings = None  # m x num_lvs matrix
        self.fit_scores = None  # n x num_lvs matrix
        self.fit_data = None  # n x m training data

        warnings.simplefilter("always", UserWarning)

    @property
    def fitted_components(self) -> int:
        """Get total # of LVs in model."""
        if self.loadings is None:
            return 0
        else:
            return self.loadings.shape[1]

    def _add_components(self, n_add: int, verbose: bool = False):
        """Method for adding components to an already-constructed model.

        The whole fit of the added components runs as one compiled call
        on the JAX device, see _fit_components.

        Args:
            n_add (int): number of components to add
            verbose (bool): Whether or not to print out additional
                convergence information. Defaults to False.
        """
        data = self.fit_data
        fitted_components = self.fitted_components

        if fitted_components > 0:
            # There are loadings, so must deflate
            sim_data = self.inverse_transform(self.transform(data))
            data = data - sim_data

        dtype, tol = _resolve_dtype(self.dtype, self.tol_criteria)
        x0, obs = _split_nan(data, dtype)
        if verbose:
            print("nan_mask Generated")

        scores, loadings, num_iters = _fit_components(
            x0, obs, n_add, tol, self.max_iter
        )

        # Results live in numpy, like the rest of the sklearn ecosystem
        scores = np.asarray(scores, dtype=np.float64)
        loadings = np.asarray(loadings, dtype=np.float64)

        for i, num_iter in enumerate(np.asarray(num_iters)):
            if verbose:
                print(f"LV {fitted_components + i} took {num_iter} iterations")
            if not np.all(np.isfinite(loadings[:, i])):
                warnings.warn(
                    f"Non-finite values on LV {fitted_components + i}, "
                    "the model is not usable"
                )
            elif num_iter >= self.max_iter:
                warnings.warn(
                    f"max_iter reached on LV {fitted_components + i}"
                )

        if fitted_components == 0:
            self.fit_scores = scores
            self.loadings = loadings
        else:
            self.fit_scores = np.concatenate([self.fit_scores, scores], axis=1)
            self.loadings = np.concatenate([self.loadings, loadings], axis=1)

    def set_components(self, n_component: int, verbose: bool = False):
        """Method for setting the number of components in an
        already-constructed model.

        Args:
            n_component (int): the desired number of components.
            verbose (bool): Whether or not to print out additional
                convergence information. Defaults to False.

        Raises:
            TypeError: if n_component is not an int
            ValueError: if n_component < 1
        """
        if not isinstance(n_component, int):
            raise TypeError("n_component must be an integer!")
        elif n_component < 1:
            raise ValueError("n_component must be an int > 0")

        max_fit_lvs = self.fitted_components
        if n_component <= max_fit_lvs:
            self.n_components = n_component
        else:
            n_to_add = n_component - max_fit_lvs
            # Deflation in _add_components uses the active components, so
            # activate all fitted ones or the new ones repeat old directions
            self.n_components = max_fit_lvs
            self._add_components(n_to_add, verbose=verbose)
            self.set_components(n_component)

        return self

    def transform(self, X: np.ndarray, method: str = "naive") -> np.ndarray:
        """Transform input data to scores using the fitted model.

        Args:
            X (np.ndarray): The nxm input array to be projected.
            method (str, optional): The method to use for projection.
                Valid options are {'naive','projection','conditional_mean'}
                Defaults to 'naive'.

        Raises:
            NotFittedError: If model has not been fit yet.
            ValueError: Method 'conditional_mean' requires fit_data.

        Returns:
            np.ndarray: The corresponding scores.
        """
        if not self.__sklearn_is_fitted__():
            raise NotFittedError(
                "Model has not yet been fit. Consider using fit_transform."
            )

        num_lvs = self.n_components
        dtype, _ = _resolve_dtype(self.dtype, self.tol_criteria)
        x0, obs = _split_nan(np.asarray(X), dtype)
        loadings = jnp.asarray(self.loadings[:, :num_lvs], dtype=dtype)

        if (obs is None) or (method == "naive"):
            # Single component projection algorithm
            scores = _transform_naive(x0, obs, loadings)

        elif method == "projection":
            # Projection to model plane method
            scores = _transform_projection(x0, obs, loadings)

        elif method == "conditional_mean":
            # Conditional mean replacement method
            if self.fit_data is None:
                raise ValueError(
                    "This transformation method requires fit_data, "
                    "which is not available in this model."
                )

            fit_rows, fit_cols = self.fit_data.shape

            if fit_cols == self.n_components:
                T = self.fit_scores
                P = self.loadings
            elif fit_cols > self.n_components:
                # if more fit_cols required
                # fit them temporarily and
                # go back to lower number
                old_components = self.n_components
                self.set_components(fit_cols)
                T = self.fit_scores
                P = self.loadings
                self.set_components(old_components)

            theta = (T.T @ T) / (fit_rows - 1)
            data_cov = jnp.asarray(P @ theta @ P.T, dtype=dtype)

            # Only rows with missing values need the (m x m) solve,
            # batched so that the stacked systems stay around 256 MB
            nan_rows = np.isnan(X).any(axis=1)
            batch_size = max(1, 2**28 // (fit_cols**2 * dtype.dtype.itemsize))
            filled = _fill_conditional_mean(
                x0[nan_rows], obs[nan_rows], data_cov, batch_size
            )
            x0 = x0.at[nan_rows].set(filled)
            scores = x0 @ jnp.asarray(P[:, :num_lvs], dtype=dtype)

        else:
            raise ValueError(
                "method must be one of "
                "{'naive','projection','conditional_mean'}"
            )

        return np.asarray(scores, dtype=np.float64)

    def fit(self, X: np.ndarray, verbose: bool = False) -> NipalsPCA:
        """Fits PCA model to input data.

        Args:
            X (np.ndarray): The input data to fit on.
            verbose (bool, optional): Whether or not to print out additional
                convergence information. Defaults to False.

        Returns:
            NipalsPCA: A reference to the object.
        """
        if self.fitted_components > 0:
            raise ValueError(
                "Model Object has already been fit. "
                "Try set_components() or build a new model object."
            )

        # Check to see if the data is mean_centered; if not raise a warning
        if (not self._check_mean_centered(X)) and (self.mean_centered):
            warnings.warn(
                "Data appears to not be mean centered. "
                "This may cause errors in interpretation!"
            )

        self.fit_data = np.copy(X)
        self._add_components(n_add=self.n_components, verbose=verbose)

        return self

    def fit_transform(
        self, X: np.ndarray, verbose: bool = False
    ) -> np.ndarray:
        """Fit, then transform input data.

        Args:
            X (np.ndarray): The input data to fit on and transform.
            verbose (bool, optional): Whether or not to print out additional
                convergence information. Defaults to False.

        Raises:
            ValueError: Model has already been fit.

        Returns:
            np.ndarray: The corresponding scores.
        """
        if not self.__sklearn_is_fitted__():
            self.fit(X, verbose=verbose)
            return self.fit_scores.copy()
        else:
            raise ValueError(
                "Model has already been fit. Try transform instead."
            )

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Approximate original data from scores.

        Args:
            X (np.ndarray): An array containing the scores.

        Raises:
            NotFittedError: PCA model has not been fit yet.
            ValueError: Shape of provided scores does not match n_components.

        Returns:
            np.ndarray: The approximation of the original data.
        """
        if not self.__sklearn_is_fitted__():
            raise NotFittedError(
                "Model has not yet been fit. Try fit() or fit_transform() instead."
            )

        _, m = X.shape
        if m != self.n_components:
            raise ValueError(
                "X has number of columns different than number of components in model"
            )

        # Use JAX for the matrix multiplication
        X_jax = jnp.array(X)
        loadings_jax = jnp.array(self.loadings[:, : self.n_components])
        out_data = X_jax @ loadings_jax.T

        return np.array(out_data)

    # Hotelling's T2 only needs the scores and a (components x components)
    # covariance; the NumPy code does that inversion in float64, which
    # keeps low-variance components that a float32 pinv cutoff would drop
    calc_imd = _ReferenceNipalsPCA.calc_imd

    def calc_oomd(
        self, input_array: np.ndarray, metric: str = "QRes"
    ) -> np.ndarray:
        """Calculate out-of-model distance (Q-residuals or DModX).

        Args:
            input_array (np.ndarray): The data for which to calculate oomds.
            metric (str, optional): The metric to use.
                Valid options are {'Qres','DModX'}. Defaults to 'QRes'.

        Raises:
            NotImplementedError: Unknown metric.

        Returns:
            np.ndarray: The distances for the provided observations.
        """
        if metric == "QRes":
            transform_dat = input_array.copy()
            n, _ = input_array.shape

            scores = self.transform(transform_dat)
            modeled_data = self.inverse_transform(scores)

            # Calculate residuals using JAX
            resids_jax = jnp.array(transform_dat) - jnp.array(modeled_data)
            nan_mask = jnp.isnan(resids_jax)

            # Vectorized Q-residuals calculation
            def calc_row_qres(resid_row, mask_row):
                """Calculate Q-residual for a single row."""
                clean_resid = jnp.where(mask_row, 0.0, resid_row)
                return jnp.sum(clean_resid**2)

            out_oomd = jax.vmap(calc_row_qres)(resids_jax, nan_mask)
            out_oomd = np.array(out_oomd).reshape(-1, 1)

        elif metric == "DModX":
            fit_n, _ = self.fit_scores.shape
            num_lvs = self.n_components

            out_oomd = self.calc_oomd(input_array, metric="QRes")

            if self.mean_centered:
                A0 = 1
            else:
                A0 = 0

            nan_mask = np.isnan(input_array)
            not_null = ~nan_mask
            K = not_null.sum(axis=1)
            factor = np.sqrt(fit_n / ((fit_n - num_lvs - A0) * (K - num_lvs)))
            out_oomd = factor.reshape(-1, 1) * np.sqrt(out_oomd)

        else:
            raise NotImplementedError("Input metric not recognized. See doc.")

        return out_oomd

    def calc_limit(
        self,
        metric: str = "HotellingT2",
        n: Optional[int] = None,
        num_lvs: Optional[int] = None,
        m: Optional[int] = None,
        alpha: float = 0.95,
    ) -> float:
        """Calculate limits for imd and oomd.

        Args:
            metric (str, optional): The metric to use.
                Valid options are {'HotellingT2','DModX'}. Defaults to 'HotellingT2'.
            n (Optional[int], optional): Number of observations. Defaults to None.
            num_lvs (Optional[int], optional): Number of latent variables.
                Defaults to None.
            m (Optional[int], optional): Number of original features.
                Defaults to None.
            alpha (float, optional): Confidence value. Defaults to 0.95.

        Returns:
            float: The limit threshold.
        """
        if n is None:
            n = self.fit_scores.shape[0]
        if m is None:
            m = self.fit_data.shape[1]
        if num_lvs is None:
            num_lvs = self.n_components

        if metric == "HotellingT2":
            tsqcl = (
                F_dist.ppf(alpha, num_lvs, n - num_lvs)
                * num_lvs
                * (n - 1)
                / (n - num_lvs)
            )
            return tsqcl

        elif metric == "DModX":
            if self.mean_centered:
                A0 = 1
            else:
                A0 = 0

            dof_mod = np.sqrt((n - A0 - num_lvs) * (m - num_lvs))
            M = np.min([m, 100, dof_mod])
            corr = n / (n - A0 - num_lvs)

            if m > dof_mod:
                dof_obs = (M + np.sqrt(m - dof_mod) - num_lvs) / corr
            else:
                dof_obs = (M - num_lvs) / corr

            if np.any(np.array([dof_mod, dof_obs, (n - A0 - num_lvs)]) < 1):
                warnings.warn(
                    "One of the factors in the calculation of the DModX limits "
                    "was smaller than 1. This shouldn't happen."
                )

            d_crit = np.sqrt(F_dist.ppf(alpha, dof_obs, dof_mod))
            return d_crit

    def _check_mean_centered(self, data: np.ndarray) -> bool:
        """Check if data is mean centered along the rows within tolerance.

        Args:
            data (np.ndarray): Data to check.

        Returns:
            bool: Whether or not the data is mean-centered.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore", message="Mean of empty slice"
            )
            try:
                maxmean = np.nanmax(np.abs(np.nanmean(data, axis=0)))
            except RuntimeWarning:
                maxmean = np.nan

            return maxmean < 1e-10

    def get_explained_variance_ratio(
        self,
        in_data: np.ndarray = None,
    ) -> np.ndarray:
        """Calculate the explained variance ratios per fitted component.

        Args:
            in_data (np.ndarray, optional):
                Alternative input data. Defaults to None.

        Raises:
            ValueError: if in_data not mean centered.

        Returns:
            np.ndarray: explained variances
        """
        if in_data is not None:
            if self._check_mean_centered(in_data):
                data = in_data
            else:
                raise ValueError("Variance input data is not mean centered.")
        else:
            data = self.fit_data

        orig_n_comp = self.n_components
        ret = np.zeros(orig_n_comp + 1)

        # compute explained variances per component
        for i in range(1, orig_n_comp + 1):
            self.set_components(i)

            # compute data as per model
            sim_data = self.inverse_transform(self.transform(data))

            # compute residual variance
            resid_var = np.nanvar(data - sim_data, axis=0)

            # variance of data scaled to 1, average over variables
            ret[i] = np.nanmean(1 - resid_var)

        # go back to original components
        self.set_components(orig_n_comp)

        # subtract previous component
        ret = ret[1:] - ret[:-1]

        return ret

    # Property alias for sklearn compatibility
    explained_variance_ratio_ = property(get_explained_variance_ratio)

    def __sklearn_is_fitted__(self) -> bool:
        """Determine if this is fitted or not."""
        return not (self.fitted_components == 0)
