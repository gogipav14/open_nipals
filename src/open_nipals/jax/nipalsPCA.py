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
from sklearn.covariance import LedoitWolf
import warnings
from scipy.stats import f as F_dist
from open_nipals.jax.utils import _nan_mult
from typing import Optional


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
    ):
        """The constructor for the NipalsPCA class.

        Args:
            n_components (int, optional): The number of principle components.
                Defaults to 2.
            max_iter (int, optional): The maximum number of iterations until
                convergence. Defaults to 10000.
            tol_criteria (float, optional): The convergence threshold.
                Defaults to 10**-8.
            mean_centered (bool, optional): Whether or not the data is already
                mean-centered. Defaults to True.

        Returns:
            NipalsPCA object
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol_criteria = tol_criteria
        self.mean_centered = mean_centered

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

        Uses JAX's lax.while_loop for GPU-accelerated convergence iterations.

        Args:
            n_add (int): number of components to add
            verbose (bool): Whether or not to print out additional
                convergence information. Defaults to False.
        """
        # Convert to JAX arrays
        data = jnp.array(self.fit_data)
        n, m = data.shape

        fitted_components = self.fitted_components

        # Initialize scores and loadings arrays
        if fitted_components == 0:
            num_lvs = range(n_add)
            scores = jnp.zeros((n, n_add))
            loadings = jnp.zeros((m, n_add))
        else:
            # Deflate existing data
            sim_data = self.inverse_transform(self.transform(np.array(data)))
            data = data - jnp.array(sim_data)

            num_lvs = range(fitted_components, fitted_components + n_add)
            scores = jnp.concatenate(
                [jnp.array(self.fit_scores), jnp.zeros((n, n_add))], axis=1
            )
            loadings = jnp.concatenate(
                [jnp.array(self.loadings), jnp.zeros((m, n_add))], axis=1
            )

        if verbose:
            print("Scores and Loads preallocated")

        # Logical mask of NaN data
        nan_mask = jnp.isnan(data)
        nan_flag = bool(jnp.any(nan_mask))  # Convert to Python bool for if-statement

        if verbose:
            print("nan_mask Generated")

        # Define the NIPALS iteration functions for while_loop
        def nipals_cond(state):
            """Condition function: continue while not converged and under max_iter."""
            _, t_new, t_old, _, iteration = state
            score_diff = t_old - t_new
            # Use squeeze to convert (1,1) arrays to scalars
            conv_test = jnp.sqrt((score_diff.T @ score_diff).squeeze()) / jnp.sqrt((t_new.T @ t_new).squeeze())
            not_converged = conv_test >= self.tol_criteria
            under_max = iteration < self.max_iter
            return jnp.logical_and(not_converged, under_max).squeeze()

        def nipals_body_nan(state):
            """Body function for NaN case: single NIPALS iteration."""
            data_local, t_new, _, loadings_loc, iteration = state
            t_old = t_new

            # Calculate loadings with NaN handling
            loadings_loc = _nan_mult(data_local.T, t_old, nan_mask.T, use_denom=True)
            loadings_loc = loadings_loc / jnp.sqrt((loadings_loc.T @ loadings_loc).squeeze())

            # Calculate scores with NaN handling
            t_new = _nan_mult(data_local, loadings_loc, nan_mask, use_denom=True)

            return (data_local, t_new, t_old, loadings_loc, iteration + 1)

        def nipals_body_clean(state):
            """Body function for clean data (no NaN): single NIPALS iteration."""
            data_local, t_new, _, loadings_loc, iteration = state
            t_old = t_new

            # Calculate loadings (standard formula)
            loadings_loc = (data_local.T @ t_old) / (t_old.T @ t_old).squeeze()
            loadings_loc = loadings_loc / jnp.sqrt((loadings_loc.T @ loadings_loc).squeeze())

            # Calculate scores
            t_new = data_local @ loadings_loc
            t_new = t_new / (loadings_loc.T @ loadings_loc).squeeze()

            return (data_local, t_new, t_old, loadings_loc, iteration + 1)

        # Select body function based on NaN presence
        nipals_body = nipals_body_nan if nan_flag else nipals_body_clean

        # Loop for all LVs (kept as Python loop - small count, sequential dependency)
        for i in num_lvs:
            # Initialize score vector from first column
            t_new = data[:, [0]]
            # Replace NaNs with zero
            t_new = jnp.where(jnp.isnan(t_new), 0.0, t_new)

            if verbose:
                print("Score initialized")

            # Initialize state for while_loop
            t_old = jnp.zeros_like(t_new)
            loadings_loc = jnp.zeros((m, 1))
            initial_state = (data, t_new, t_old, loadings_loc, 0)

            # Run convergence loop using JAX while_loop
            final_state = jax.lax.while_loop(nipals_cond, nipals_body, initial_state)
            _, t_new, _, loadings_loc, num_iter = final_state

            if num_iter >= self.max_iter:
                print(f"max_iter Reached on LV {i}")

            if verbose:
                print(f"Iteration finished on LV {i}")

            # Store scores and loadings
            scores = scores.at[:, i : i + 1].set(t_new)
            loadings = loadings.at[:, i : i + 1].set(loadings_loc)

            # Deflate the input matrix for next component
            data = data - t_new @ loadings_loc.T

            if verbose:
                print("Deflation Complete")

        # Store results (convert back to numpy for sklearn compatibility)
        self.fit_scores = np.array(scores)
        self.loadings = np.array(loadings)

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

        # Convert to JAX array
        X_jax = jnp.array(X)
        n, _ = X_jax.shape

        num_lvs = self.n_components
        loadings = jnp.array(self.loadings)

        scores = jnp.zeros((n, num_lvs))

        nan_mask = jnp.isnan(X_jax)
        nan_flag = jnp.any(nan_mask)

        if (not nan_flag) or (method == "naive"):
            # Single component projection algorithm
            X_deflated = X_jax
            for ind_lv in range(num_lvs):
                if nan_flag:
                    score_col = _nan_mult(
                        X_deflated,
                        loadings[:, [ind_lv]],
                        nan_mask,
                        use_denom=True,
                    )
                else:
                    score_col = X_deflated @ loadings[:, [ind_lv]]
                    score_col = score_col / (
                        loadings[:, [ind_lv]].T @ loadings[:, [ind_lv]]
                    )
                scores = scores.at[:, [ind_lv]].set(score_col)
                # Deflate
                X_deflated = X_deflated - score_col @ loadings[:, [ind_lv]].T

        elif nan_flag and (method == "projection"):
            # Projection to model plane method
            # This requires matrix inversion per row, so we use numpy
            X_np = np.array(X_jax)
            nan_mask_np = np.array(nan_mask)
            loadings_np = np.array(loadings)
            scores_np = np.zeros((n, num_lvs))

            for row in range(n):
                not_null = np.invert(nan_mask_np[row, :])
                denom = np.linalg.inv(
                    loadings_np[not_null, :num_lvs].T @ loadings_np[not_null, :num_lvs]
                )
                nom = loadings_np[not_null, :num_lvs].T @ X_np[row, not_null].T
                scores_np[row, :] = (denom @ nom).T

            return scores_np

        elif nan_flag and (method == "conditional_mean"):
            # Conditional mean replacement method
            if self.fit_data is None:
                raise ValueError(
                    "This transformation method requires fit_data, "
                    "which is not available in this model."
                )

            fit_rows, fit_cols = self.fit_data.shape
            X_np = np.array(X_jax)
            nan_mask_np = np.array(nan_mask)

            if fit_cols == self.n_components:
                T = self.fit_scores
                P = self.loadings
            elif fit_cols > self.n_components:
                full_model = NipalsPCA(fit_cols)
                full_model.fit(self.fit_data, verbose=False)
                P = full_model.loadings
                T = full_model.fit_scores

            theta = (T.T @ T) / (fit_rows - 1)
            scores_np = np.zeros((n, num_lvs))

            for row in range(n):
                is_null = nan_mask_np[row, :]
                not_null = np.invert(is_null)
                if np.any(is_null):
                    S12 = P[is_null, :] @ theta @ P[not_null, :].T
                    S22 = P[not_null, :] @ theta @ P[not_null, :].T
                    z_hash = S12 @ np.linalg.inv(S22) @ X_np[row, not_null].T
                    X_np[row, is_null] = z_hash
                scores_np[row, :] = (P.T @ X_np[row, :].T)[0 : self.n_components]

            return scores_np

        return np.array(scores)

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

    def fit_transform(self, X: np.ndarray, verbose: bool = False) -> np.ndarray:
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
            raise ValueError("Model has already been fit. Try transform instead.")

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

    def calc_imd(
        self,
        input_scores: Optional[np.ndarray] = None,
        input_array: Optional[np.ndarray] = None,
        metric: str = "HotellingT2",
        covariance: str = "diag",
    ) -> np.ndarray:
        """Calculate within-model distance (Hotelling's T2).

        Args:
            input_scores (Optional[np.ndarray], optional): The scores.
                Defaults to None.
            input_array (Optional[np.ndarray], optional): The input data.
                Defaults to None.
            metric (str, optional): The metric to use.
                Valid options are {'HotellingT2'}. Defaults to 'HotellingT2'.
            covariance (str, optional): Method to compute covariance. Valid
                options are {'diag', 'full', 'ledoit_wolf'}.
                Defaults to 'diag'.

        Raises:
            NotFittedError: Model has not been fit yet.
            ValueError: Neither scores nor input data provided.
            NotImplementedError: Unknown metric or covariance method.

        Returns:
            np.ndarray: The calculated within-model distance.
        """
        if not self.__sklearn_is_fitted__():
            raise NotFittedError(
                "Model has not yet been fit. Try fit() or fit_transform() instead."
            )

        if metric == "HotellingT2":
            if (input_array is None) and (input_scores is None):
                raise ValueError("No values provided.")

            elif (input_array is not None) and (input_scores is not None):
                warnings.warn(
                    "Both Scores and Data are given. Operating on Data alone."
                )
                out_t2 = self.calc_imd(input_array=input_array, covariance=covariance)

            elif (input_scores is None) and (input_array is not None):
                scores = self.transform(input_array)
                out_t2 = self.calc_imd(input_scores=scores, covariance=covariance)

            else:
                # Use JAX for vectorized computation
                scores_jax = jnp.array(input_scores)
                fit_scores_jax = jnp.array(self.fit_scores[:, : self.n_components])

                _, num_lvs = scores_jax.shape
                num_lvs_fit = self.n_components

                if num_lvs != num_lvs_fit:
                    raise ValueError(
                        "input_scores have different number of columns/latent "
                        "variables than model n_components."
                    )

                fit_means = jnp.mean(fit_scores_jax, axis=0)

                if covariance == "diag":
                    fit_vars = jnp.var(fit_scores_jax, axis=0, ddof=1)
                    out_t2 = jnp.sum(
                        (scores_jax - fit_means) ** 2 / fit_vars, axis=1
                    ).reshape(-1, 1)
                elif covariance == "full":
                    # Use full covariance matrix
                    cov_matrix = jnp.cov(fit_scores_jax.T, ddof=1)
                    cov_inv = jnp.linalg.pinv(cov_matrix)
                    diff = scores_jax - fit_means
                    out_t2 = jnp.diagonal(diff @ cov_inv @ diff.T).reshape(-1, 1)
                elif covariance == "ledoit_wolf":
                    # Compute full covariance matrix with Ledoit-Wolf shrinkage
                    lw_obj = LedoitWolf(
                        assume_centered=self.mean_centered
                    ).fit(np.array(fit_scores_jax))
                    cov_inv = jnp.linalg.pinv(jnp.array(lw_obj.covariance_))
                    diff = scores_jax - fit_means
                    out_t2 = jnp.diagonal(diff @ cov_inv @ diff.T).reshape(-1, 1)
                else:
                    raise NotImplementedError(
                        f"Covariance method {covariance} not implemented. "
                        "Possible methods are {'diag', 'full', 'ledoit_wolf'}."
                    )
                out_t2 = np.array(out_t2)
        else:
            raise NotImplementedError("This metric has not been implemented. See doc.")

        return out_t2

    def calc_oomd(self, input_array: np.ndarray, metric: str = "QRes") -> np.ndarray:
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
            warnings.filterwarnings(action="ignore", message="Mean of empty slice")
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
