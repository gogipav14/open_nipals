# Known limitations

Behaviour that is understood and accepted rather than fixed, each with the reason and a way to reproduce it.
Findings of code reviews that are not fixed are recorded here.

## JAX in float32

**Agreement with NumPy.** With `dtype="float32"` (or JAX's 64-bit mode off) fitted scores and loadings agree with the NumPy classes to about 1e-4 relative, set by the convergence tolerance, which is floored at `1e-5` in float32.
Matrix products run at full float32 precision also on GPUs whose default is TF32.

**SIMCA "no residual" threshold.** SIMCA treats a class model as having used up all variation when the pooled residual scale is below `max(100 eps, 10 x NIPALS tolerance)` of the data scale. In float32 that is `1e-4`: genuine noise below 1e-4 of the data scale counts as no residual, and a count that leaves only such noise is refused (explicit `n_components`) or reduced (automatic).
Reason: an iterative fit cannot tell residuals near its own tolerance from rounding; rank-exhausted fits were measured between 4.5e-10 and 1.6e-8 relative in float64 depending on the start vector.
Reproduce: `tests/test_jax_simca.py::TestJAXSIMCAReviewRound17` (noise at 1.3e-4 is kept, just above the threshold).

## SIMCA component selection

**Q² depends on the fold count.** Q² uses Wold's element-wise cross validation (`q2_cv_folds`, default 7). Rows and columns are put in a canonical order first, so the selection does not depend on how the data is ordered, but a different fold count can select a different number of components when two counts have similar Q².

**Selection sees rows the final model drops.** In automatic mode the component count is chosen on all usable training rows (at least two observed varying features). Rows with fewer than A + 1 observed varying features are dropped afterwards for the chosen A. With very sparse training rows the chosen count can therefore reflect variation that the final model does not see.
Reproduce: `tests/test_simca.py::TestSIMCAReviewRound16`.

## SIMCA training rows

Rows that cannot be modelled are dropped with a warning rather than raising: rows with fewer than 2 observed varying features before component selection, and rows with fewer than A + 1 after it. A class left without rows raises `ValueError`.

## Comparison with R mdatools

Class-membership counts differ from `mdatools` by a few samples per class because the limits differ (SIMCA-P F-based limits here; `ddmoments`, `jm` or `chisq` in `mdatools`, which also disagree with each other by a similar margin). Distances themselves agree to 5e-7 relative.
Details: `validation/simca_mdatools/REPORT.md`.

## PLS start column

`NipalsPLS` (NumPy and JAX) starts each component from the Y column with the largest variance. If that column is orthogonal to X (e.g. `X = [1, -1, 1, -1]`, `Y = [[2, 1], [2, -1], [-2, 1], [-2, -1]]`) the iteration cannot leave it and the loadings are NaN. Scheduled to be fixed together with the next item (project plan, WP3).

## PLS grown with set_components when Y has missing values

A PLS model grown with `set_components` differs by about 1e-2 in the loadings from a direct fit with the same number of components when Y has missing values (identical for complete Y). Growing deflates Y with `predict()` (b t q'), the fit loop with t q'. Scheduled to be fixed (see the project plan, WP3); until then, fit directly with the final number of components when Y has missing values.
