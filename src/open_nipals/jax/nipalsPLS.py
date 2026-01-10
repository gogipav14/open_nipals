"""
JAX-accelerated PLS using the NIPALS algorithm.

This provides GPU acceleration for PLS regression with missing data support.

Algorithm implemented from Chapter 6 of
    Chiang, Leo H., Evan L. Russell, and Richard D. Braatz.
    Fault detection and diagnosis in industrial systems.
    Springer Science & Business Media, 2000.

Alternative algorithm derivation from:
    Geladi, P.; Kowalski, B. R.
    Partial Least-Squares Regression: A Tutorial.
    Analytica Chimica Acta 1986, 185, 1–17.
    https://doi.org/10.1016/0003-2670(86)80028-9.

For the transformation part also see:
    Nelson, P. R. C.; Taylor, P. A.; MacGregor, J. F.
    Missing data methods in PCA and PLS: Score calculations
    with incomplete observations.
    Chemometrics and Intelligent Laboratory Systems 1996, 35(1), 45-65.

(C) 2020-2021: Ryan Wall (lead), David Ochsenbein, YBaranwal (original NumPy)
revised 2024: Niels Schlusser
JAX conversion 2024
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.cross_decomposition._pls import _PLS
from sklearn.exceptions import NotFittedError
import warnings
from open_nipals.jax.utils import _nan_mult
from typing import Optional, Tuple, Union


class NipalsPLS(_PLS):
    """JAX-accelerated PLS using the NIPALS algorithm.

    This class provides the same interface as the NumPy version but uses
    JAX for GPU acceleration on large datasets.

    Attributes:
    ----------
    n_components : int
        The number of latent variables.
    max_iter : int
        The max number of iterations for the fitting step.
    tol_criteria : float
        The convergence tolerance criterion.
    mean_centered : bool
        Whether or not the original data is mean-centered.
    force_include : bool
        Force including rows with all NaNs in Y-block.
    """

    def __init__(
        self,
        n_components: int = 2,
        max_iter: int = 10000,
        tol_criteria: float = 10**-10,
        mean_centered: bool = True,
        force_include: bool = False,
    ):
        """Constructor for initialization.

        Args:
            n_components (int): The number of components to use in the model.
            max_iter (int): The maximum number of iterations when fitting.
            tol_criteria (float): Tolerance limit for convergence.
            mean_centered (bool): Whether the data is mean centered.
            force_include (bool): Force include rows with all NaNs in Y-block.

        Returns:
            NipalsPLS object
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol_criteria = tol_criteria
        self.mean_centered = mean_centered
        self.force_include = force_include

        self.fit_data_x = None  # n_rows_x x n_cols_x
        self.fit_data_y = None  # n_rows_y x n_cols_y

        self.loadings_x = None  # n_cols_x x n_components
        self.fit_scores_x = None  # n_rows_x x n_components
        self.weights_x = None  # n_cols_x x n_components

        self.loadings_y = None  # n_cols_y x n_components
        self.fit_scores_y = None  # n_rows_y x n_components

        self.regression_matrix = None  # n_components x n_predictors

        warnings.simplefilter("always", UserWarning)

    @property
    def fitted_components(self) -> int:
        """Get total # of LVs in model."""
        if self.loadings_x is not None:
            return self.loadings_x.shape[1]
        else:
            return 0

    def _filter_nan_rows(self, X, y):
        """Filter rows where all Y-values are NaN."""
        y_nan_rows = np.all(np.isnan(y), axis=1)
        if np.any(y_nan_rows):
            warnings.warn("Some Y-rows have no data in them!")
            if self.force_include:
                warnings.warn("Rows still included due to force_include.")
            else:
                warnings.warn(
                    "Rows with all NaNs in Y are dropped, see force_include."
                )
                rows_to_keep = np.invert(y_nan_rows)
                y = y[rows_to_keep, :]
                X = X[rows_to_keep, :]

        return X.copy(), y.copy()

    def _add_components(self, n_add: int, verbose: bool = False):
        """Method for adding components to an already-constructed model.

        Uses JAX's lax.while_loop for GPU-accelerated convergence iterations.

        Args:
            n_add (int): number of components to add
            verbose (bool): Whether to print convergence info. Defaults to False.
        """
        X = self.fit_data_x.copy()
        y = self.fit_data_y.copy()

        # Filter and store
        self.fit_data_x, self.fit_data_y = self._filter_nan_rows(X, y)

        # Pull out shapes
        n_rows_x, n_cols_x = self.fit_data_x.shape
        n_rows_y, n_cols_y = self.fit_data_y.shape
        if n_rows_x != n_rows_y:
            raise ValueError("Rows in X do not match rows in Y!")

        fitted_components = self.fitted_components

        # Initialize arrays
        if fitted_components == 0:
            num_lvs = range(n_add)
            p = jnp.zeros((n_cols_x, n_add))  # x loadings
            t = jnp.zeros((n_rows_x, n_add))  # x scores
            u = jnp.zeros((n_rows_x, n_add))  # y scores
            w = jnp.zeros((n_cols_x, n_add))  # weights
            q = jnp.zeros((n_cols_y, n_add))  # y loadings
            b = jnp.zeros((n_add, n_add))  # regression coeff
        else:
            num_lvs = range(fitted_components, fitted_components + n_add)

            # Deflate existing data
            sim_data_x = self.inverse_transform(self.transform(X))
            sim_data_y = self.predict(X, self.fit_scores_x)
            X = X - sim_data_x
            y = y - sim_data_y

            p = jnp.concatenate(
                [jnp.array(self.loadings_x), jnp.zeros((n_cols_x, n_add))], axis=1
            )
            t = jnp.concatenate(
                [jnp.array(self.fit_scores_x), jnp.zeros((n_rows_x, n_add))], axis=1
            )
            u = jnp.concatenate(
                [jnp.array(self.fit_scores_y), jnp.zeros((n_rows_x, n_add))], axis=1
            )
            w = jnp.concatenate(
                [jnp.array(self.weights_x), jnp.zeros((n_cols_x, n_add))], axis=1
            )
            q = jnp.concatenate(
                [jnp.array(self.loadings_y), jnp.zeros((n_cols_y, n_add))], axis=1
            )
            b = jnp.zeros((fitted_components + n_add, fitted_components + n_add))
            b = b.at[:fitted_components, :fitted_components].set(
                jnp.array(self.regression_matrix)
            )

        # Convert to JAX arrays
        x_res = jnp.array(X)
        y_res = jnp.array(y)

        # NaN masks
        nan_mask_x = jnp.isnan(x_res)
        nan_mask_y = jnp.isnan(y_res)
        nan_flag = bool(jnp.any(nan_mask_x) | jnp.any(nan_mask_y))  # Convert to Python bool

        # Loop over each LV
        for ind_lv in num_lvs:
            # Define body functions INSIDE the loop so they capture updated x_res/y_res
            def pls_cond(state):
                """Condition: continue while not converged and under max_iter."""
                ti, ti_old, _, _, _, iteration = state
                diff_norm = jnp.linalg.norm(ti - ti_old) / jnp.linalg.norm(ti)
                not_converged = diff_norm >= self.tol_criteria
                under_max = iteration < self.max_iter
                return jnp.logical_and(not_converged, under_max).squeeze()

            # Capture current x_res, y_res in closure
            x_res_local = x_res
            y_res_local = y_res

            def pls_body_nan(state):
                """Body function for NaN case."""
                ti, _, ui, wi, qi, iteration = state
                ti_old = ti

                wi = _nan_mult(x_res_local.T, ui, nan_mask_x.T, use_denom=False)
                wi = wi / jnp.linalg.norm(wi)
                ti = _nan_mult(x_res_local, wi, nan_mask_x, use_denom=True)
                qi = _nan_mult(y_res_local.T, ti, nan_mask_y.T, use_denom=False)
                qi = qi / jnp.linalg.norm(qi)
                ui = _nan_mult(y_res_local, qi, nan_mask_y, use_denom=False)

                return (ti, ti_old, ui, wi, qi, iteration + 1)

            def pls_body_clean(state):
                """Body function for clean data (no NaN)."""
                ti, _, ui, wi, qi, iteration = state
                ti_old = ti

                wi = x_res_local.T @ ui
                wi = wi / jnp.linalg.norm(wi)
                ti = x_res_local @ wi
                qi = y_res_local.T @ ti
                qi = qi / jnp.linalg.norm(qi)
                ui = y_res_local @ qi

                return (ti, ti_old, ui, wi, qi, iteration + 1)

            pls_body = pls_body_nan if nan_flag else pls_body_clean
            # Start with column of Y having max variance
            std_y = jnp.nanstd(jnp.array(self.fit_data_y), axis=0)
            start_col = int(jnp.argmax(std_y))  # Convert to Python int for indexing

            # Initialize Y scores
            ui = y_res[:, start_col : start_col + 1]
            ui = jnp.where(jnp.isnan(ui), 0.0, ui)
            ti = jnp.array(ui)  # Explicit copy

            if verbose:
                print(f"LV {ind_lv} Started")

            # Initialize state
            ti_old = jnp.zeros_like(ti)
            wi = jnp.zeros((n_cols_x, 1))
            qi = jnp.zeros((n_cols_y, 1))
            initial_state = (ti, ti_old, ui, wi, qi, 0)

            # Run convergence loop
            final_state = jax.lax.while_loop(pls_cond, pls_body, initial_state)
            ti, _, ui, wi, qi, iter_count = final_state

            if iter_count >= self.max_iter:
                print(f"max_iter Reached on LV {ind_lv}.")

            # Compute X loadings
            if nan_flag:
                pi = _nan_mult(x_res.T, ti, nan_mask_x.T, use_denom=False)
            else:
                pi = (x_res.T @ ti) / (ti.T @ ti)

            p_weight = jnp.linalg.norm(pi)
            pi = pi / p_weight

            # Scale weights
            wi = wi * p_weight

            # Recompute scores if NaNs
            if nan_flag:
                ti = _nan_mult(x_res, wi, nan_mask_x, use_denom=True)
            else:
                ti = ti * p_weight

            # Regression coefficient
            bi = (ui.T @ ti) / (ti.T @ ti)

            # Store values
            p = p.at[:, ind_lv : ind_lv + 1].set(pi)
            t = t.at[:, ind_lv : ind_lv + 1].set(ti)
            u = u.at[:, ind_lv : ind_lv + 1].set(ui)
            w = w.at[:, ind_lv : ind_lv + 1].set(wi)
            q = q.at[:, ind_lv : ind_lv + 1].set(qi)
            b = b.at[ind_lv, ind_lv].set(bi.item())

            # Deflate residual matrices
            x_res = x_res - ti @ pi.T
            y_res = y_res - bi * ti @ qi.T

        # Store results (convert to numpy for sklearn compatibility)
        self.fit_scores_x = np.array(t)
        self.loadings_x = np.array(p)
        self.weights_x = np.array(w)
        self.fit_scores_y = np.array(u)
        self.loadings_y = np.array(q)
        self.regression_matrix = np.array(b)

    def set_components(self, n_component: int, verbose: bool = False):
        """Set the number of components in an already-constructed model.

        Args:
            n_component (int): the desired number of components.
            verbose (bool): Print convergence info. Defaults to False.

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

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = False,
    ) -> NipalsPLS:
        """Fit PLS model from X/Y Data.

        Args:
            X (np.ndarray): Input X data.
            y (np.ndarray): Input Y data.
            verbose (bool): Turn verbosity on/off. Defaults to False.

        Raises:
            NotFittedError: Model has already been fit.

        Returns:
            NipalsPLS: A reference to the object.
        """
        if self.__sklearn_is_fitted__():
            raise NotFittedError(
                "Model Object has already been fit. "
                "Try set_components() or build a new model object."
            )

        if (not self._check_mean_centered(X)) and (self.mean_centered):
            warnings.warn(
                "X-Block appears to not be mean centered. "
                "This will cause errors in prediction!"
            )

        if (not self._check_mean_centered(y)) and (self.mean_centered):
            warnings.warn(
                "Y-Block appears to not be mean centered. "
                "This will cause errors in prediction!"
            )

        self.fit_data_x = np.copy(X)
        self.fit_data_y = np.copy(y)

        self._add_components(n_add=self.n_components, verbose=verbose)

        return self

    def transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Compute scores using model.

        Args:
            X (np.ndarray): X-data.
            y (np.ndarray, optional): Y-data. Defaults to None.

        Raises:
            NotFittedError: If model is not fit.

        Returns:
            Union[np.ndarray, Tuple]: Scores for X, or tuple of (X, Y) scores.
        """
        if not self.__sklearn_is_fitted__():
            raise NotFittedError("Model has not yet been fit.")

        if y is None:
            scores_x = self._transform_xy(X, self.loadings_x, weights=self.weights_x)
            return scores_x
        else:
            scores_x = self._transform_xy(X, self.loadings_x, weights=self.weights_x)
            scores_y = self._transform_xy(y, self.loadings_y)
            return scores_x, scores_y

    def _transform_xy(
        self,
        input_array: np.ndarray,
        loadings: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Transform X or Y data to scores.

        Args:
            input_array (np.ndarray): The input array (X or Y data).
            loadings (np.ndarray): The loadings.
            weights (np.ndarray, optional): The weights. Defaults to None.

        Returns:
            np.ndarray: The scores.
        """
        # Convert to JAX
        input_jax = jnp.array(input_array)
        loadings_jax = jnp.array(loadings)

        n, _ = input_jax.shape
        num_lvs = self.n_components

        if weights is None:
            weights_jax = loadings_jax.copy()
        else:
            weights_jax = jnp.array(weights)

        scores = jnp.zeros((n, num_lvs))
        nan_mask = jnp.isnan(input_jax)
        resids = input_jax.copy()

        for ind_lv in range(num_lvs):
            score_col = _nan_mult(
                resids, weights_jax[:, [ind_lv]], nan_mask, use_denom=True
            )
            scores = scores.at[:, [ind_lv]].set(score_col)
            resids = resids - score_col @ loadings_jax[:, [ind_lv]].T

        return np.array(scores)

    def _check_mean_centered(self, data: np.ndarray) -> bool:
        """Check if data is mean centered."""
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", message="Mean of empty slice")
            try:
                maxmean = np.nanmax(np.abs(np.nanmean(data, axis=0)))
            except RuntimeWarning:
                maxmean = np.nan

            return maxmean < 10**-10

    def fit_transform(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and transform in one step.

        Args:
            X (np.ndarray): The X-data.
            y (np.ndarray): The Y-data.

        Raises:
            ValueError: If model already fit.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Fitted scores for X and Y.
        """
        if not self.__sklearn_is_fitted__():
            self.fit(X, y)
            return self.fit_scores_x, self.fit_scores_y
        else:
            raise ValueError("Model has already been fit. Try transform instead.")

    def calc_imd(
        self,
        input_scores: Optional[np.ndarray] = None,
        input_array: Optional[np.ndarray] = None,
        metric: str = "HotellingT2",
    ):
        """Calculate in-model distance (Hotelling's T2).

        Args:
            input_scores (Optional[np.ndarray]): Scores array.
            input_array (Optional[np.ndarray]): Data array.
            metric (str): Metric to compute. Defaults to 'HotellingT2'.

        Raises:
            NotFittedError: If model not fit.
            ValueError: If no values provided.

        Returns:
            float: The within-model distance(s).
        """
        if not self.__sklearn_is_fitted__():
            raise NotFittedError(
                "Model has not yet been fit. Try fit() or fit_transform()."
            )
        elif (input_array is None) and (input_scores is None):
            raise ValueError("No values provided.")

        if input_array is not None:
            if input_scores is not None:
                warnings.warn(
                    "Both Scores and Data given. Operating on Data alone."
                )
            scores = self.transform(X=input_array)
        elif input_scores is not None:
            scores = input_scores

        if metric == "HotellingT2":
            num_lvs_fit = self.n_components

            if scores.shape[1] != num_lvs_fit:
                raise ValueError(
                    "input_scores have more columns/latent variables than "
                    "model was fit with."
                )

            # Use JAX for computation
            scores_jax = jnp.array(scores)
            fit_scores_jax = jnp.array(self.fit_scores_x[:, :num_lvs_fit])

            fit_means = jnp.mean(fit_scores_jax, axis=0)
            fit_vars = jnp.var(fit_scores_jax, axis=0, ddof=1)
            out_imd = jnp.sum(
                (scores_jax - fit_means) ** 2 / fit_vars, axis=1
            ).reshape(-1, 1)

            return np.array(out_imd)
        else:
            raise ValueError("Unknown metric requested (metric = HotellingT2).")

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Transform scores back to simulated data.

        Args:
            X (np.ndarray): The scores to transform back.

        Raises:
            NotFittedError: If model not fit.
            ValueError: If input scores shape mismatch.

        Returns:
            np.ndarray: The simulated data.
        """
        if not self.__sklearn_is_fitted__():
            raise NotFittedError(
                "Model has not yet been fit. Try fit() or fit_transform()."
            )

        _, m = X.shape
        if m != self.n_components:
            raise ValueError(
                "input_scores has number of columns different than "
                "number of components in model"
            )

        # Use JAX
        X_jax = jnp.array(X)
        loadings_jax = jnp.array(self.loadings_x[:, : self.n_components])
        out_data_x = X_jax @ loadings_jax.T

        return np.array(out_data_x)

    def calc_oomd(
        self, input_array: np.ndarray, metric: str = "QRes"
    ) -> np.ndarray:
        """Calculate Out of Model Distance.

        Args:
            input_array (np.ndarray): The input data.
            metric (str): Metric to compute. Defaults to 'QRes'.

        Raises:
            ValueError: If unknown metric.

        Returns:
            np.ndarray: The out-of-model distance(s).
        """
        if metric == "QRes":
            n, _ = input_array.shape

            scores = self.transform(input_array)
            modeled_data = self.inverse_transform(X=scores)

            # Calculate residuals with JAX
            residual_jax = jnp.array(input_array) - jnp.array(modeled_data)
            nan_mask = jnp.isnan(residual_jax)

            def calc_row_qres(resid_row, mask_row):
                clean_resid = jnp.where(mask_row, 0.0, resid_row)
                return jnp.sum(clean_resid**2)

            out_oomd = jax.vmap(calc_row_qres)(residual_jax, nan_mask)
            out_oomd = np.array(out_oomd).reshape(-1, 1)

        elif metric == "DModX":
            n, _ = self.fit_scores_x.shape
            num_lvs = self.n_components

            out_oomd = self.calc_oomd(input_array, metric="QRes")

            if self.mean_centered:
                A0 = 1
            else:
                A0 = 0

            K = self.fit_data_x.shape[1]
            factor = np.sqrt(n / ((n - num_lvs - A0) * (K - num_lvs)))
            out_oomd = factor.reshape(-1, 1) * np.sqrt(out_oomd)
        else:
            raise ValueError("Input metric not recognized")

        return out_oomd

    def predict(
        self, X: np.ndarray = None, scores_x: np.ndarray = None
    ) -> np.ndarray:
        """Predict y from data or scores.

        Args:
            X (np.ndarray, optional): Input X data.
            scores_x (np.ndarray, optional): Input scores.

        Raises:
            NotFittedError: If model not fit.
            ValueError: If neither data nor scores provided.

        Returns:
            np.ndarray: The predicted y-values.
        """
        if not self.__sklearn_is_fitted__():
            raise NotFittedError("Model has not yet been fit")

        if (X is None) and (scores_x is None):
            raise ValueError("Provide either data or scores")
        elif scores_x is not None:
            num_lvs = self.n_components
            # Use JAX
            scores_jax = jnp.array(scores_x[:, :num_lvs])
            reg_jax = jnp.array(self.regression_matrix[:num_lvs, :num_lvs])
            loadings_y_jax = jnp.array(self.loadings_y[:, :num_lvs])

            pred_y = scores_jax @ reg_jax @ loadings_y_jax.T
            return np.array(pred_y)
        else:
            scores_x = self.transform(X)
            return self.predict(scores_x=scores_x)

    def get_reg_vector(self) -> np.ndarray:
        """Get the regression vector for the model.

        Raises:
            NotFittedError: If model not fit.

        Returns:
            np.ndarray: The regression vector.
        """
        if not self.__sklearn_is_fitted__():
            raise NotFittedError("Model has not yet been fit")

        # Use JAX
        weights_jax = jnp.array(self.weights_x[:, : self.n_components])
        reg_jax = jnp.array(self.regression_matrix[: self.n_components, :])
        loadings_y_jax = jnp.array(self.loadings_y[:, : self.n_components])

        reg_vects = weights_jax @ (reg_jax @ loadings_y_jax.T)
        return np.array(reg_vects)

    def __sklearn_is_fitted__(self) -> bool:
        """Determine if this is fitted or not."""
        return not (self.fitted_components == 0)
