# `open_nipals` PCA and PLS package

This package implements the nonlinear iterative partial least squares (NIPALS) algorithm for principal component analysis (PCA) and partial least squares (PLS) regression in a `scikit-learn` compatible fashion. 
In contrast to orthodox methods for PCA and PLS, the NIPALS algorithm is an iterative method, allowing free tuning of desired numerical performance and precision.
Moreover, it naturally integrates with Nelson's Single Component Projection method for missing data imputation.

# Quickstart

Install the package with `pip install open-nipals`.

Training a `NipalsPCA` model can look as simple as:
```python
from sklearn.preprocessing import StandardScaler
from open_nipals.nipalsPCA import NipalsPCA

# input data frame df

# standard-scale data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# train PCA model
model = NipalsPCA()
transformed_data = model.fit_transform(X=data)
```

A minimal example of fitting a `NipalsPLS` model:
```python
from sklearn.preprocessing import StandardScaler
from open_nipals.nipalsPLS import NipalsPLS

# input data frames df_x, df_y

# standard-scale data
scaler_x = StandardScaler()
scaler_y = StandardScaler()
scaled_x_data = scaler_x.fit_transform(df_x)
scaled_y_data = scaler_y.fit_transform(df_y)

# train PLS model
model = NipalsPLS()
transformed_x_data, transformed_y_data = model.fit_transform(X=scaled_x_data, y=scaled_y_data)
```

# Key Features and API Overview

## Preprocessing

Both the `NipalsPCA` and `NipalsPLS` classes expect a numpy array as an input with rows as samples and columns as features. 
Additionally, these array columns should have zero mean for best performance; typically this is done with a sklearn `StandardScaler` object. 
Note that it is *highly* encouraged to mean-center the input data before training an `open_nipals` model on it.

Note: If the input data is a `pandas` dataframe, you can fit and instantiate an `ArrangeData` object which will ensure all future datasets come to the appropriate shape and column order.

```python
from open_nipals.utils import ArrangeData
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load some arbitrary data
df = pd.read_csv('my_data.csv')

# Invoke preprocessing pipeline
arrdat = ArrangeData()
scaler = StandardScaler()

# Both scaler and arrdat should be saved for future use
data = scaler.fit_transform(arrdat.fit_transform(df))

# data is ready to model
```

## Model fitting and transforming

The number of components can be specified as an argument to the constructor, with the default `n_components=2`. 
After fitting, components can be added or removed by the `set_components()` method without having to fit the entire model again from scratch.
Components that were once fitted but not needed any more are saved for possible later use. 

Functions of the `scikit-learn` API implemented by `open_nipals`:

- `fit()` for model fitting
- `transform()` for transforming data given a fitted model
- `fit_transform()` as a combination of `fit()` and `transform()`
- a pseudo-inverse transformation `inverse_transform()`, making the model predict how the data would look like


## PLS prediction

One particular feature of PLS models is that they can predict dependent variables. To this end, run `model.predict()`, where either a matrix of X data `X`, 
or a matrix of X scores `scores_x` need to be given as arguments, e.g. 
```python
predicted_y_data = model.predict(X=data_x)
```


## Distances

In-model distances (IMD) and out-of-model distances (OOMD) are metrics of model accuracy.
They can be calculated for PCA and PLS models with:
```python
# Must be scaled data
hotelling_t2 = model.calc_imd(input_array = data)

# also must be scaled, default metric is QResiduals or 'QRes'
dmodx = model.calc_oomd(input_array = data, metric = "DModX")
```

## Explainability

Similar to `scikit-learn`, the attribute `explained_variance_ratio_` measures the ratio of variance that each component of the model explains.
`NipalsPLS` has two of those arrays, one for the X and one for the y data.
Note that the NIPALS algorithm avoids calculating eigenvalues, therefore they are not accessible as the `explained_variance_` attribute.

Additionally, the regression vector can be calculated for a `NipalsPLS` model with `get_reg_vector()`.
The regression vector is a measure of how relevant each X feature is for the prediction of the y data.

## GPU acceleration with JAX (optional)

`open_nipals.jax` contains drop-in versions of `NipalsPCA` and `NipalsPLS` whose fit and transform run as compiled [JAX](https://docs.jax.dev) programs, on a GPU if one is available. It needs Python 3.10 or newer.
They implement the same algorithm, including the missing data handling, and reproduce the results of the NumPy classes to rounding error.
```bash
pip install open_nipals[jax]        # CPU only
pip install open_nipals[jax-cuda]   # NVIDIA GPU
```
```python
import jax
from open_nipals.jax import NipalsPCA, NipalsPLS

jax.config.update("jax_enable_x64", True)  # see below
model = NipalsPCA(n_components=5).fit(data)
```

Things to know:

- Inputs and all fitted attributes stay NumPy arrays, so the models can be mixed with the rest of `open_nipals` and `scikit-learn`.
- JAX computes in 32-bit unless its 64-bit mode is on. The models follow that setting: to reproduce the NumPy results, call `jax.config.update("jax_enable_x64", True)` in your application before fitting (the package does not change this process-wide setting for you). In float32 the models are faster and use half the GPU memory, but agree with the NumPy results to only a few digits, limited by the convergence tolerance (`tol_criteria` is floored at `1e-5`). Matrix products always run at full float32 precision, also on GPUs whose default would be TF32. Pass `dtype="float64"` or `dtype="float32"` to insist on a precision; `dtype="float64"` raises a clear error if 64-bit mode is off.
- The first fit for a given data shape and number of components includes compilation. Later fits of the same shape, e.g. in cross validation, reuse the compiled code.
- Small data sets do not benefit from a GPU; see the measurements below and `benchmarks/bench_jax.py` to reproduce them on your hardware.

Time in seconds for `fit()` with 5 components, excluding the one-off compilation (0.4 to 5 s), on an Intel Core Ultra 5 225F and an NVIDIA RTX 5060 (8 GB). "10 % NaN" means 10 % of the X entries are missing at random.

| model | rows x columns | data | NumPy | JAX GPU float64 | JAX GPU float32 |
|---|---|---|---|---|---|
| PCA | 200 x 50 | 10 % NaN | 0.013 | 0.029 | 0.025 |
| PCA | 2000 x 200 | 10 % NaN | 1.5 | 0.13 | 0.046 |
| PCA | 10000 x 500 | complete | 9.6 | 0.46 | 0.11 |
| PCA | 10000 x 500 | 10 % NaN | 80 | 0.64 | 0.19 |
| PLS | 10000 x 500 | complete | 0.79 | 0.11 | 0.058 |
| PLS | 10000 x 500 | 10 % NaN | 12.6 | 0.15 | 0.095 |
| PCA | 20000 x 2000 | complete | 28 | 3.5 | 1.5 |
| PCA | 20000 x 2000 | 10 % NaN | 966 | 5.1 | 2.0 |
| PLS | 20000 x 2000 | complete | 11.8 | 1.1 | 0.56 |
| PCA | 50000 x 2000 | 10 % NaN | not run | 18.8 | 6.7 |

Without a GPU the JAX classes are still about 7x faster than NumPy on data with missing values, but can be slower on large complete data.


## SIMCA classification

`open_nipals.simca.SIMCA` is a SIMCA (Soft Independent Modelling of Class Analogy) classifier built on `NipalsPCA`, with a `scikit-learn` classifier interface (`fit`, `predict`, `predict_proba`, `score`; `is_classifier` is true, so scorers and ensembles accept it).
`open_nipals.jax.simca.SIMCA` is the same classifier on top of the JAX `NipalsPCA`.
```python
from open_nipals.simca import SIMCA

model = SIMCA(n_components="auto", alpha=0.95).fit(X_train, y_train)
labels = model.predict(X_new)                 # None for rejected samples with unknown_handling="reject"
membership = model.get_class_membership(X_new)
distances = model.get_distances(X_new)        # T2, normalised DModX and their limits per class
```

How it works:

- One PCA model per class. With `scale=True` (default) every class is centred and scaled by its own training mean and standard deviation (as in SIMCA-P and R `mdatools`); constant features keep scale 1.
- A sample belongs to a class when its Hotelling T² and its DModX are both within the class limits at confidence `alpha`. DModX is the residual standard deviation over the sample's observed, varying features, relative to the pooled training value; its limit is `NipalsPCA.calc_limit(metric="DModX")`.
- `predict` returns the single class a sample belongs to, the closest one (combined normalised distance) if it belongs to several, and for none either the closest class (`unknown_handling="closest"`, default) or `None` (`"reject"`).
- `n_components` is an int or `"auto"`: `component_selection="q2"` (default) uses element-wise (Wold) cross validation and takes the fewest components whose Q² is within `q2_min_improvement` of the best; `"r2"` and `"eigenvalue"` (Kaiser) are also available. Automatic counts are capped by the class size, the number of varying features, the rank of the data and the missing-data pattern.
- Missing values are supported. Training rows with too few observed varying features for the model are dropped with a warning; new samples with too few are rejected (infinite DModX).

Validation against R `mdatools` 0.16.0 on Iris and Wine (`validation/simca_mdatools/`): per-sample T² and residuals agree to within 5e-7 relative, and class membership agrees for 96.7 % (Iris) and 97.8 % (Wine) of sample/class pairs; the remaining differences come from the different limit definitions (SIMCA-P F-based here, data-driven in `mdatools`).
`tests/test_simca_invariance.py` checks that constant or empty columns, empty rows, row and column order and feature units do not change the classifier; `validation/sweep_simca_invariance.py` runs it over many random data sets.

## Numerical notes

- NIPALS starts each component from a fixed pseudo-random combination of all columns (with missing values, refined by power iteration on the zero-filled data). The fitted components therefore do not depend on the column order, and a single column orthogonal to the leading component cannot trap the iteration. Component signs follow the convention of positive correlation with the first column.
- Accepted limitations and their reasons are listed in [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md).

# References

PLS algorithm implemented from Chapter 6 of:
> Chiang, Leo H., Evan L. Russell, and Richard D. Braatz.
> Fault detection and diagnosis in industrial systems.
> Springer Science & Business Media, 2000.

One of the most concise definitions can be found in this paper on page 7:
> Geladi, P.; Kowalski, B. R. Partial Least-Squares Regression: A Tutorial.
> Analytica Chimica Acta 1986, 185, 1–17.
> https://doi.org/10.1016/0003-2670(86)80028-9.

For the transformation part also see:
> Nelson, P. R. C.; Taylor, P. A.; MacGregor, J. F. Missing data methods
> in PCA and PLS: Score calculations with incomplete observations.
> Chemometrics and Intelligent Laboratory Systems 1996, 35(1), 45-65.

# Documentation

An online version of the documentation is hosted at [ReadTheDocs](https://open-nipals.readthedocs.io/en/latest/).


# Contributing

If you would like to contribute to `open_nipals`, please check out our [github repo](https://github.com/johnsonandjohnson/open_nipals).
For contribution guidelines please refer to the `CONTRIBUTING.md` in the repo, or the [contributor's guide](https://open-nipals.readthedocs.io/en/stable/contributing.html) in the online documentation.

# License

`open_nipals` is distributed under the BSD 3-clause license.

# Citation
This documentation refers to [`open_nipals v2.0.1`](https://github.com/johnsonandjohnson/open_nipals/tree/v2.0.1). 
An archived version of the code can be found under this DOI [10.5281/zenodo.18375840](https://doi.org/10.5281/zenodo.18375840).
