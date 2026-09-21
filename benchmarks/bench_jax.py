"""Benchmark NumPy vs JAX NIPALS fits.

Run once per backend (JAX picks its device per process):

    python benchmarks/bench_jax.py --backend numpy
    JAX_PLATFORMS=cpu python benchmarks/bench_jax.py --backend jax
    python benchmarks/bench_jax.py --backend jax          # GPU if available
    python benchmarks/bench_jax.py --backend jax --dtype float32

Each line of output is one JSON record. "first_s" includes JIT compilation,
"best_s" is the fastest of the repeat fits on already-compiled code.
"""

import argparse
import json
import time

import numpy as np

SIZES = [(200, 50), (2000, 200), (10000, 500), (20000, 2000), (50000, 2000)]
NAN_FRACS = [0.0, 0.1]
N_COMPONENTS = 5
N_TARGETS = 3


def make_data(n, m, nan_frac, seed=0):
    """Low-rank data plus noise, mean centered, with optional NaNs in X."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 8))
    X = latent @ rng.normal(size=(8, m)) + 0.5 * rng.normal(size=(n, m))
    Y = latent[:, :N_TARGETS] + 0.1 * rng.normal(size=(n, N_TARGETS))
    if nan_frac > 0:
        X[rng.random(size=X.shape) < nan_frac] = np.nan
    X -= np.nanmean(X, axis=0)
    Y -= Y.mean(axis=0)
    return X, Y


def time_fit(make_model, fit_args, repeats, budget_s):
    """Return (first fit time, best repeat time, model from last fit)."""
    times = []
    for _ in range(1 + repeats):
        model = make_model()
        start = time.perf_counter()
        model.fit(*fit_args)
        times.append(time.perf_counter() - start)
        if sum(times) > budget_s:
            break
    best = min(times[1:]) if len(times) > 1 else times[0]
    return times[0], best, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["numpy", "jax"], required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--budget",
        type=float,
        default=300.0,
        help="Stop repeating a config after this many seconds",
    )
    parser.add_argument("--max-rows", type=int, default=50000)
    parser.add_argument(
        "--dtype",
        choices=["float64", "float32"],
        default="float64",
        help="JAX backend only",
    )
    args = parser.parse_args()

    if args.backend == "jax":
        import jax
        from open_nipals.jax import NipalsPCA, NipalsPLS

        device = f"{jax.devices()[0]} {args.dtype}"
        kwargs = {"dtype": args.dtype}
    else:
        from open_nipals.nipalsPCA import NipalsPCA
        from open_nipals.nipalsPLS import NipalsPLS

        device = "numpy"
        kwargs = {}

    for n, m in SIZES:
        if n > args.max_rows:
            continue
        for nan_frac in NAN_FRACS:
            X, Y = make_data(n, m, nan_frac)
            for name, make_model, fit_args in [
                ("pca", lambda: NipalsPCA(n_components=N_COMPONENTS, **kwargs), (X,)),
                ("pls", lambda: NipalsPLS(n_components=N_COMPONENTS, **kwargs), (X, Y)),
            ]:
                first, best, model = time_fit(
                    make_model, fit_args, args.repeats, args.budget
                )
                scores = model.fit_scores if name == "pca" else model.fit_scores_x
                print(
                    json.dumps(
                        {
                            "device": device,
                            "model": name,
                            "n": n,
                            "m": m,
                            "nan_frac": nan_frac,
                            "first_s": round(first, 4),
                            "best_s": round(best, 4),
                            "score_norm": float(np.linalg.norm(scores[:, 0])),
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
