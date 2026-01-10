"""
JAX-accelerated NIPALS implementations for GPU acceleration.

This module provides JAX versions of NipalsPCA and NipalsPLS that can
leverage GPU acceleration for massive speedups on large datasets.

Usage:
    from open_nipals.jax import NipalsPCA, NipalsPLS

    # Automatically uses GPU if available
    pca = NipalsPCA(n_components=5)
    pca.fit(X)
"""

from open_nipals.jax.nipalsPCA import NipalsPCA
from open_nipals.jax.nipalsPLS import NipalsPLS
from open_nipals.jax.utils import _nan_mult

__all__ = ["NipalsPCA", "NipalsPLS", "_nan_mult"]
