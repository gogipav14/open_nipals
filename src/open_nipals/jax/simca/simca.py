"""
JAX-accelerated SIMCA (Soft Independent Modelling of Class Analogy) classifier.

Uses JAX NipalsPCA for GPU acceleration while inheriting classification
logic from the base SIMCA implementation.
"""

import jax.numpy as jnp
import numpy as np
from typing import Union, Literal

from open_nipals.simca.simca import SIMCA as SIMCA_Base, SIMCAClass, ComponentSelector
from open_nipals.jax.nipalsPCA import NipalsPCA as NipalsPCA_JAX
from .metrics import calc_r2_cumulative_pca
from .cross_validation import KFoldCV


class SIMCA(SIMCA_Base):
    """
    JAX-accelerated SIMCA classifier.

    Inherits all classification logic from the base SIMCA class but uses
    JAX NipalsPCA models for GPU acceleration.

    Parameters
    ----------
    n_components : int or 'auto', default=2
        Number of components for each class model.
    alpha : float, default=0.95
        Significance level for T² and DModX limits.
    component_selection : str, default='q2'
        Method for automatic component selection: 'r2', 'q2', 'eigenvalue'.
    r2_threshold : float, default=0.80
        R² threshold when component_selection='r2'.
    q2_cv_folds : int, default=7
        CV folds when component_selection='q2'.
    q2_min_improvement : float, default=0.05
        Minimum Q² improvement when component_selection='q2'.
    scale : bool, default=True
        Whether to apply StandardScaler to data.
    unknown_handling : str, default='closest'
        How to handle samples not in any class: 'closest' or 'reject'.
    max_iter : int, default=10000
        Maximum iterations for NIPALS convergence.
    tol_criteria : float, default=1e-8
        Convergence tolerance for NIPALS.

    See Also
    --------
    open_nipals.simca.SIMCA : NumPy version of SIMCA classifier.
    """

    def _create_pca_model(self, n_components: int) -> NipalsPCA_JAX:
        """Create a JAX PCA model instance."""
        return NipalsPCA_JAX(
            n_components=n_components,
            max_iter=self.max_iter,
            tol_criteria=self.tol_criteria,
        )

    def _select_components(self, X_class: np.ndarray) -> int:
        """
        Select number of components for a class using JAX models.

        Parameters
        ----------
        X_class : np.ndarray
            Data for single class.

        Returns
        -------
        int
            Selected number of components.
        """
        if isinstance(self.n_components, int):
            return self.n_components

        n_samples, n_features = X_class.shape
        max_components = min(10, n_samples - 1, n_features)

        if max_components < 1:
            return 1

        if self.component_selection == "r2":
            temp_pca = self._create_pca_model(max_components)
            temp_pca.fit(X_class)
            # Use JAX metrics
            r2_values = calc_r2_cumulative_pca(temp_pca, X_class)
            for i, r2 in enumerate(r2_values):
                if r2 >= self.r2_threshold:
                    return i + 1
            return max_components

        elif self.component_selection == "q2":
            from .cross_validation import cross_val_predict_pca
            from .metrics import calc_press

            cv = KFoldCV(n_splits=self.q2_cv_folds)
            nan_mask = np.isnan(X_class)
            X_clean = np.where(nan_mask, 0.0, X_class)
            ss_total = np.sum(X_clean ** 2)

            q2_values = []
            for n_comp in range(1, max_components + 1):
                X_pred_cv = cross_val_predict_pca(
                    NipalsPCA_JAX, X_class, n_comp, cv,
                    max_iter=self.max_iter, tol_criteria=self.tol_criteria
                )
                press = calc_press(X_class, X_pred_cv)
                q2 = 1.0 - (press / ss_total) if ss_total > 0 else 0.0
                q2_values.append(q2)

            # Find where Q² stops improving
            best_n = 1
            for i in range(1, len(q2_values)):
                improvement = q2_values[i] - q2_values[i - 1]
                if improvement >= self.q2_min_improvement:
                    best_n = i + 1
                else:
                    break

            return best_n

        elif self.component_selection == "eigenvalue":
            temp_pca = self._create_pca_model(max_components)
            temp_pca.fit(X_class)
            # Use NumPy for eigenvalue calculation
            scores = np.asarray(temp_pca.fit_scores)
            variances = np.var(scores, axis=0, ddof=1)
            eigenvalues = variances * n_features / np.sum(variances)
            n_above = np.sum(eigenvalues > 1.0)
            return max(1, n_above)

        else:
            raise ValueError(
                f"Unknown component_selection: {self.component_selection}"
            )

    def _calc_distances(self, X: np.ndarray, class_label) -> tuple:
        """
        Calculate T² and DModX for samples to a specific class.

        Converts between NumPy and JAX arrays as needed.
        """
        model = self.class_models_[class_label]
        pca = model.pca_model

        t2 = pca.calc_imd(input_array=X)
        dmodx = pca.calc_oomd(X, metric="DModX")

        # Convert to NumPy and flatten
        return np.asarray(t2).flatten(), np.asarray(dmodx).flatten()
