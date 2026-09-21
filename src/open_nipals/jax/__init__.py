"""
JAX-accelerated NIPALS implementations for GPU acceleration.

This module provides JAX versions of NipalsPCA and NipalsPLS that can
leverage GPU acceleration for massive speedups on large datasets.

Usage:
    from open_nipals.jax import NipalsPCA, NipalsPLS

    # Automatically uses GPU if available
    pca = NipalsPCA(n_components=5)
    pca.fit(X)

Importing this module enables 64-bit mode in JAX (jax_enable_x64), which is
needed to reproduce the NumPy results. Pass dtype="float32" to the models
to trade that precision for speed and memory on the GPU.
"""

import jax

jax.config.update("jax_enable_x64", True)

from open_nipals.jax.nipalsPCA import NipalsPCA  # noqa: E402
from open_nipals.jax.nipalsPLS import NipalsPLS  # noqa: E402
from open_nipals.jax.utils import _nan_mult  # noqa: E402

__all__ = ["NipalsPCA", "NipalsPLS", "_nan_mult"]
