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
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.exceptions import NotFittedError
import warnings
from functools import partial
from open_nipals.jax.nipalsPCA import _start_column
from open_nipals.nipalsPCA import _start_weights
from open_nipals.nipalsPLS import NipalsPLS as _ReferenceNipalsPLS
from open_nipals.jax.utils import (
    _full_precision_methods,
    _masked_mult,
    _resolve_dtype,
    _split_nan,
)
from typing import Optional, Tuple, Union


@partial(jax.jit, static_argnames=["n_add"])
def _fit_components(
    x_res: jnp.ndarray,
    obs_x: Optional[jnp.ndarray],
    y_res: jnp.ndarray,
    obs_y: Optional[jnp.ndarray],
    start_col: int,
    n_add: int,
    tol: float,
    max_iter: int,
    start_weights: jnp.ndarray,
) -> Tuple[jnp.ndarray, ...]:
    """Fit n_add NIPALS PLS components, deflating after each one.

    Mirrors the loop in open_nipals.nipalsPLS.NipalsPLS._add_components.
    Compiled once per (shapes, n_add, NaN/no NaN) and reused after that.

    Args:
        x_res (jnp.ndarray): The (already deflated) X data, NaNs set to zero.
        obs_x (Optional[jnp.ndarray]): 1 where x_res was observed. None if
            neither block has NaNs.
        y_res (jnp.ndarray): The (already deflated) Y data, NaNs set to zero.
        obs_y (Optional[jnp.ndarray]): 1 where y_res was observed. None if
            neither block has NaNs.
        start_col (int): Column of Y that sets the sign convention.
        n_add (int): Number of components to fit.
        tol (float): The convergence threshold.
        max_iter (int): The maximum number of iterations per component.
        start_weights (jnp.ndarray): Y column weights of the start
            vector, see open_nipals.nipalsPCA._start_weights.

    Returns:
        Tuple[jnp.ndarray, ...]: x scores, x loadings, x weights, y scores,
            y loadings (all with n_add columns), the n_add inner regression
            coefficients and the iterations used per component.
    """
    nan_flag = obs_x is not None

    def not_converged(state):
        ti, ti_old, _, _, _, iter_count = state
        # guard against zero norm
        den = jnp.maximum(jnp.linalg.norm(ti), 1e-12)
        diff_norm = jnp.linalg.norm(ti - ti_old) / den
        # written so that a NaN diff_norm counts as not converged
        not_done = ~(diff_norm < tol) & (iter_count < max_iter)
        # a non-finite iterate never recovers, stop and report it
        return (iter_count == 0) | (not_done & jnp.isfinite(diff_norm))

    def one_component(residuals, _):
        x_res, y_res = residuals

        def iterate(state):
            ti_old, _, ui, _, _, iter_count = state
            if not nan_flag:  # Loop for no NaN
                wi = (x_res.T @ ui) / (ui.T @ ui)  # x weight
                wi = wi / jnp.linalg.norm(wi)  # Normalize weights
                ti = (x_res @ wi) / (wi.T @ wi)  # x score
                qi = (y_res.T @ ti) / (ti.T @ ti)  # y weight
                ui = (y_res @ qi) / (qi.T @ qi)  # y score
            else:  # Loop for handling NaNs
                wi = _masked_mult(x_res.T, obs_x.T, ui)
                wi = wi / jnp.linalg.norm(wi)
                ti = _masked_mult(x_res, obs_x, wi)
                qi = _masked_mult(y_res.T, obs_y.T, ti)
                ui = _masked_mult(y_res, obs_y, qi)
            return (ti, ti_old, ui, wi, qi, iter_count + 1)

        # Fixed random combination of the Y columns as start, as in the
        # NumPy version; the start column only sets the sign convention.
        # NaNs are already zero.
        u_sign = jax.lax.dynamic_slice_in_dim(y_res, start_col, 1, axis=1)
        ui = y_res @ start_weights[:, None]
        ui = jnp.where(jnp.any(ui != 0), ui, _start_column(y_res, start_col))
        # No X score yet (zeros): comparing the first one with the Y
        # guess could report convergence after a single iteration
        state = (
            jnp.zeros_like(ui),
            jnp.zeros_like(ui),
            ui,
            jnp.zeros((x_res.shape[1], 1), x_res.dtype),
            jnp.zeros((y_res.shape[1], 1), y_res.dtype),
            0,
        )
        ti, _, ui, wi, qi, iter_count = jax.lax.while_loop(
            not_converged, iterate, state
        )

        # Sign convention: positive correlation with the start column
        sign = jnp.where((ui.T @ u_sign)[0, 0] < 0, -1.0, 1.0)
        wi, ti, qi, ui = sign * wi, sign * ti, sign * qi, sign * ui

        # x loading
        if not nan_flag:
            pi = (x_res.T @ ti) / (ti.T @ ti)
        else:
            pi = _masked_mult(x_res.T, obs_x.T, ti)

        # regression coefficient, is a scalar!
        bi = (ui.T @ ti) / (ti.T @ ti)

        # update residual matrices for next LV, missing entries stay zero
        x_res = x_res - ti @ pi.T
        y_res = y_res - ti @ qi.T
        if nan_flag:
            x_res = x_res * obs_x
            y_res = y_res * obs_y

        out = (ti[:, 0], pi[:, 0], wi[:, 0], ui[:, 0], qi[:, 0], bi[0, 0])
        return (x_res, y_res), (out, iter_count)

    _, ((t, p, w, u, q, b_diag), iter_counts) = jax.lax.scan(
        one_component, (x_res, y_res), None, length=n_add
    )
    return t.T, p.T, w.T, u.T, q.T, b_diag, iter_counts


@partial(jax.jit, static_argnames=["use_denom"])
def _transform_components(
    resids: jnp.ndarray,
    obs: Optional[jnp.ndarray],
    weights: jnp.ndarray,
    loadings: jnp.ndarray,
    use_denom: bool,
) -> jnp.ndarray:
    """Single component projection, deflating with the loadings.

    Uses approach described in section 3.1 of McGregor paper
    (Single component projection algorithm for missing data in PCA/PLS)
    """

    def one_component(resids, weight_loading):
        weight, loading = weight_loading
        weight = weight[:, None]
        if obs is None:
            score_col = resids @ weight
            if use_denom:
                score_col = score_col / (weight.T @ weight)
        else:
            score_col = _masked_mult(resids, obs, weight, use_denom=use_denom)
        # deflate input data
        resids = resids - score_col @ loading[:, None].T
        if obs is not None:
            resids = resids * obs
        return resids, score_col[:, 0]

    _, scores = jax.lax.scan(one_component, resids, (weights.T, loadings.T))
    return scores.T


@_full_precision_methods
class NipalsPLS(BaseEstimator, TransformerMixin, RegressorMixin):
    """JAX-accelerated PLS using the NIPALS algorithm.

    This class provides the same interface as the NumPy version but uses
    JAX for GPU acceleration on large datasets.

    Attributes:
    -----------
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
        tol_criteria: float = 1e-6,
        mean_centered: bool = True,
        force_include: bool = False,
        dtype: Optional[str] = None,
    ):
        """Constructor for initialization.

        Args:
            n_components (int): The number of components to use in the model.
            max_iter (int): The maximum number of iterations when fitting.
            tol_criteria (float): Tolerance limit for convergence.
            mean_centered (bool): Whether the data is mean centered.
            force_include (bool): Force include rows with all NaNs in Y-block.
            dtype (str, optional): Precision to fit and transform in, either
                'float64' (matches the NumPy version, requires JAX's 64-bit
                mode) or 'float32' (faster and half the memory on GPU,
                tol_criteria floored at 1e-5). Results are always returned
                as float64 numpy arrays. Defaults to None, which follows
                JAX's jax_enable_x64 setting.

        Returns:
            NipalsPLS object
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol_criteria = tol_criteria
        self.mean_centered = mean_centered
        self.force_include = force_include
        self.dtype = dtype

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

        The whole fit of the added components runs as one compiled call
        on the JAX device, see _fit_components.

        Args:
            n_add (int): number of components to add
            verbose (bool): Whether to print convergence info. Defaults to False.
        """
        # Filter and store
        self.fit_data_x, self.fit_data_y = self._filter_nan_rows(
            self.fit_data_x, self.fit_data_y
        )
        X = self.fit_data_x
        y = self.fit_data_y

        # Pull out shapes
        n_rows_x, _ = self.fit_data_x.shape
        n_rows_y, _ = self.fit_data_y.shape
        if n_rows_x != n_rows_y:
            raise ValueError("Rows in X do not match rows in Y!")

        fitted_components = self.fitted_components

        if fitted_components > 0:
            # Deflate exactly as the fit loop does (t p', t q'), see the
            # NumPy version
            t_fit = self.fit_scores_x[:, :fitted_components]
            X = X - t_fit @ self.loadings_x[:, :fitted_components].T
            y = y - t_fit @ self.loadings_y[:, :fitted_components].T

        # Start with column of Y having max variance
        std_y = np.nanstd(self.fit_data_y, axis=0)
        start_col = int(np.argmax(std_y))

        dtype, tol = _resolve_dtype(self.dtype, self.tol_criteria)
        x_res, obs_x = _split_nan(X, dtype)
        y_res, obs_y = _split_nan(y, dtype)

        # NaNs in either block switch both blocks to the NaN-aware loop
        if (obs_x is None) != (obs_y is None):
            if obs_x is None:
                obs_x = jnp.ones(x_res.shape, obs_y.dtype)
            else:
                obs_y = jnp.ones(y_res.shape, obs_x.dtype)

        start_weights = jnp.asarray(
            _start_weights(y_res.shape[1]), dtype=y_res.dtype
        )
        t, p, w, u, q, b_diag, iter_counts = _fit_components(
            x_res,
            obs_x,
            y_res,
            obs_y,
            start_col,
            n_add,
            tol,
            self.max_iter,
            start_weights,
        )

        for i, iter_count in enumerate(np.asarray(iter_counts)):
            if verbose:
                print(
                    f"LV {fitted_components + i} took {iter_count} iterations"
                )
            if not np.all(np.isfinite(np.asarray(p)[:, i])):
                warnings.warn(
                    f"Non-finite values on LV {fitted_components + i}, "
                    "the model is not usable"
                )
            elif iter_count >= self.max_iter:
                warnings.warn(
                    f"max_iter reached on LV {fitted_components + i}."
                )

        # Results live in numpy, like the rest of the sklearn ecosystem
        t, p, w, u, q, b_diag = (
            np.asarray(arr, dtype=np.float64)
            for arr in (t, p, w, u, q, b_diag)
        )
        n_total = fitted_components + n_add
        b = np.zeros((n_total, n_total))
        b[fitted_components:, fitted_components:] = np.diag(b_diag)

        if fitted_components > 0:
            t = np.concatenate([self.fit_scores_x, t], axis=1)
            p = np.concatenate([self.loadings_x, p], axis=1)
            w = np.concatenate([self.weights_x, w], axis=1)
            u = np.concatenate([self.fit_scores_y, u], axis=1)
            q = np.concatenate([self.loadings_y, q], axis=1)
            b[:fitted_components, :fitted_components] = self.regression_matrix

        self.fit_scores_x = t
        self.loadings_x = p
        self.weights_x = w
        self.fit_scores_y = u
        self.loadings_y = q
        self.regression_matrix = b

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
            # Deflation in _add_components uses the active components, so
            # activate all fitted ones or the new ones repeat old directions
            self.n_components = max_fit_lvs
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
            scores_x = self._transform_xy(
                X, self.loadings_x, weights=self.weights_x
            )
            return scores_x
        else:
            scores_x = self._transform_xy(
                X, self.loadings_x, weights=self.weights_x
            )
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
        num_lvs = self.n_components
        dtype, _ = _resolve_dtype(self.dtype, self.tol_criteria)
        resids, obs = _split_nan(np.asarray(input_array), dtype)
        loadings_jax = jnp.asarray(loadings[:, :num_lvs], dtype=dtype)

        # For Y-scores (no weights provided), use loadings which are unit
        # normalized, so use_denom=True (divides by 1).
        # For X-scores (weights provided), behavior depends on NaN presence:
        # - Non-NaN data: use_denom=False (weights are unit normalized)
        # - NaN data: use_denom=True (matches the fitting behavior)
        if weights is None:
            weights_jax = loadings_jax
            use_denom = True
        else:
            weights_jax = jnp.asarray(weights[:, :num_lvs], dtype=dtype)
            use_denom = obs is not None

        scores = _transform_components(
            resids, obs, weights_jax, loadings_jax, use_denom
        )

        return np.asarray(scores, dtype=np.float64)

    def _check_mean_centered(self, data: np.ndarray) -> bool:
        """Check if data is mean centered."""
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore", message="Mean of empty slice"
            )
            try:
                maxmean = np.nanmax(np.abs(np.nanmean(data, axis=0)))
            except RuntimeWarning:
                maxmean = np.nan

            return maxmean < 1e-10

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
            raise ValueError(
                "Model has already been fit. Try transform instead."
            )

    # Hotelling's T2 only needs the scores and a (components x components)
    # covariance; the NumPy code does that inversion in float64, which
    # keeps low-variance components that a float32 pinv cutoff would drop
    calc_imd = _ReferenceNipalsPLS.calc_imd

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
            out_oomd = factor * np.sqrt(out_oomd)
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

        The regression vector B satisfies y_pred = X @ B for complete X,
        matching predict(X): B = W @ (I + triu(P.T @ W, 1))^-1 @ diag(b)
        @ Q.T, see open_nipals.nipalsPLS.NipalsPLS.get_reg_vector.

        Raises:
            NotFittedError: If model not fit.

        Returns:
            np.ndarray: The regression vector (n_features_x, n_targets_y).
        """
        if not self.__sklearn_is_fitted__():
            raise NotFittedError("Model has not yet been fit")

        num_lvs = self.n_components
        W = jnp.array(self.weights_x[:, :num_lvs])
        P = jnp.array(self.loadings_x[:, :num_lvs])
        Q = jnp.array(self.loadings_y[:, :num_lvs])
        B_inner = jnp.array(self.regression_matrix[:num_lvs, :num_lvs])

        # Correction for the sequential deflation in transform(), see the
        # NumPy version for the derivation
        deflation = jnp.eye(num_lvs) + jnp.triu(P.T @ W, 1)

        reg_vects = W @ jnp.linalg.inv(deflation) @ B_inner @ Q.T

        return np.array(reg_vects)

    def get_explained_variance_ratio(
        self,
        in_x_data: np.ndarray = None,
        in_y_data: np.ndarray = None,
    ) -> (np.ndarray, np.ndarray):
        """Calculate the explained variance ratios for X and y arrays
        per fitted component.

        Args:
            in_x_data (np.ndarray, optional):
                Alternative input X data. Defaults to None.
            in_y_data (np.ndarray, optional):
                Alternative input y data. Defaults to None.

        Raises:
            ValueError: If in_x_data not mean centered.
            ValueError: If in_y_data not mean centered.

        Returns:
            (np.ndarray, np.ndarray): explained variance ratios for X and y
        """
        if in_x_data is not None:
            if self._check_mean_centered(in_x_data):
                x_data = in_x_data
            else:
                raise ValueError("Variance input X data is not mean centered.")
        else:
            x_data = self.fit_data_x

        if in_y_data is not None:
            if self._check_mean_centered(in_y_data):
                y_data = in_y_data
            else:
                raise ValueError("Variance input y data is not mean centered.")
        else:
            y_data = self.fit_data_y

        orig_n_comp = self.n_components
        ret_x = np.zeros(orig_n_comp + 1)
        ret_y = np.zeros(orig_n_comp + 1)

        # compute explained variance ratios per component
        for i in range(1, orig_n_comp + 1):
            self.set_components(i)

            # compute data as per model
            sim_data_x = self.inverse_transform(self.transform(x_data))
            sim_data_y = self.predict(x_data, self.fit_scores_x)

            # compute residual variance
            resid_x_var = np.nanvar(x_data - sim_data_x, axis=0)
            resid_y_var = np.nanvar(y_data - sim_data_y, axis=0)

            # variance of data scaled to 1, average over variables
            ret_x[i] = np.nanmean(1 - resid_x_var)
            ret_y[i] = np.nanmean(1 - resid_y_var)

        # go back to original components
        self.set_components(orig_n_comp)

        # subtract previous components
        ret_x = ret_x[1:] - ret_x[:-1]
        ret_y = ret_y[1:] - ret_y[:-1]

        return ret_x, ret_y

    # Property alias for sklearn compatibility
    explained_variance_ratio_ = property(get_explained_variance_ratio)

    def __sklearn_is_fitted__(self) -> bool:
        """Determine if this is fitted or not."""
        return not (self.fitted_components == 0)
