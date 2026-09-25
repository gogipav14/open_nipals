"""
JAX-accelerated SIMCA (Soft Independent Modelling of Class Analogy) classifier.

Uses JAX NipalsPCA for GPU acceleration while inheriting classification
logic from the base SIMCA implementation.
"""

from open_nipals.simca.simca import SIMCA as SIMCA_Base
from open_nipals.jax.nipalsPCA import NipalsPCA as NipalsPCA_JAX


class SIMCA(SIMCA_Base):
    """
    JAX-accelerated SIMCA classifier.

    Inherits all classification logic from the base SIMCA class and only
    swaps the PCA implementation: component selection, class centring,
    the normalised DModX and the limits are the ones of the NumPy
    reference implementation.

    Parameters
    ----------
    n_components : int or 'auto', default=2
        Number of components for each class model.
    alpha : float, default=0.95
        Confidence level of the T² and DModX limits (0.95 corresponds
        to a 5 % significance level).
    component_selection : str, default='q2'
        Method for automatic component selection: 'r2', 'q2', 'eigenvalue'.
    r2_threshold : float, default=0.80
        R² threshold when component_selection='r2'.
    q2_cv_folds : int, default=7
        CV folds when component_selection='q2'.
    q2_min_improvement : float, default=0.05
        With component_selection='q2', the fewest components whose Q² is
        within this of the best Q² are used.
    scale : bool, default=True
        Autoscale each class by its own training mean and standard
        deviation, see the NumPy version.
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

    _pca_class = NipalsPCA_JAX

    def _convergence_tolerance(self) -> float:
        """NIPALS tolerance after the float32 floor of the JAX models."""
        from open_nipals.jax.utils import _resolve_dtype

        _, tol = _resolve_dtype(None, self.tol_criteria)
        return float(tol)

    def _compute_eps(self) -> float:
        """Epsilon of the dtype the JAX PCA models compute in.

        The models follow JAX's x64 setting, float32 when it is off, and
        the residual exhaustion test must use that precision.
        """
        import numpy as np
        from open_nipals.jax.utils import _resolve_dtype

        dtype, _ = _resolve_dtype(None, 1.0)
        return float(np.finfo(dtype).eps)
