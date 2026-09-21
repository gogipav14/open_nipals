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

    _pca_class = NipalsPCA_JAX
