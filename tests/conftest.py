"""
conftest.py containing auxiliary test methods and fixtures
for open_nipals
"""

import os
import warnings
from typing import Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler

# The tox "jax" environments set this, so that a broken JAX install fails
# the run instead of silently skipping every JAX test
if os.environ.get("OPEN_NIPALS_REQUIRE_JAX"):
    import jax  # noqa: F401


def nan_conc_coeff(y: np.ndarray, yhat: np.ndarray) -> float:
    """Calculate the Lin's Concordance Coefficient, a
    linearity metric that shows how close to 1:1 a line is"""
    # Note that using the standard numpy var,cov, etc caused some
    # weird errors. could get correlations of 1.001001 etc.

    nan_mask = np.isnan(y)
    nan_mask_yhat = np.isnan(yhat)
    new_y = y[np.invert(nan_mask)].copy()
    new_yhat = yhat[np.invert(nan_mask_yhat)].copy()

    # averages
    ybar = np.mean(new_y)
    yhatbar = np.mean(new_yhat)

    # variances
    sy = np.sum((new_y - ybar) ** 2) / len(new_y)
    syhat = np.sum((new_yhat - yhatbar) ** 2) / len(new_yhat)
    syyhat = np.sum((new_y - ybar) * (new_yhat - yhatbar)) / len(
        new_y
    )  # covariance

    numer = 2 * syyhat
    denom = sy + syhat + (ybar - yhatbar) ** 2
    return numer / denom


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    """Calculate Root Mean Square Error"""
    y = y.ravel()
    yhat = yhat.ravel()

    return np.sqrt(np.mean((y - yhat) ** 2))


def init_scaler(dat: np.array) -> Tuple[StandardScaler, np.array]:
    scaler = StandardScaler()
    scaler.fit(dat)
    scaler.scale_ = np.nanstd(
        dat, axis=0, ddof=1
    )  # standardscaler uses biased variance, but we want unbiased estimator

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        transformed_data = scaler.transform(dat)
        # throws unproblematic: "RuntimeWarning: invalid value encountered in divide"

    return scaler, transformed_data
