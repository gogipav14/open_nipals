"""
JAX-accelerated NIPALS implementations for GPU acceleration.

This module provides JAX versions of NipalsPCA and NipalsPLS that can
leverage GPU acceleration for massive speedups on large datasets.

Usage:
    from open_nipals.jax import NipalsPCA, NipalsPLS

    # Automatically uses GPU if available
    pca = NipalsPCA(n_components=5)
    pca.fit(X)

    # SIMCA classifier
    from open_nipals.jax.simca import SIMCA
    simca = SIMCA(n_components=3)
    simca.fit(X, y)

The models compute in float64 when JAX's 64-bit mode is on and in float32
otherwise (JAX defaults to 32-bit). Reproducing the NumPy results needs
float64, so enable it in your application before fitting:

    import jax
    jax.config.update("jax_enable_x64", True)

This module does not change that setting itself, as it is global to the
process. Pass dtype="float64" or dtype="float32" to the models to insist on
a precision instead of following the setting.
"""

from open_nipals.jax.nipalsPCA import NipalsPCA
from open_nipals.jax.nipalsPLS import NipalsPLS
from open_nipals.jax.utils import _nan_mult

__all__ = ["NipalsPCA", "NipalsPLS", "_nan_mult", "simca"]
