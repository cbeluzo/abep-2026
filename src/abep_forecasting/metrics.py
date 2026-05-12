"""Forecast metric helpers shared by SARIMA and LSTM pipelines.

The metric functions align pandas Series by index when possible and drop
missing values before scoring. This avoids inconsistent behavior between
modeling sections and makes the final comparison table comparable.

Author:
    Carlos Beluzo - beluzo@ifsp.edu.br
"""

from __future__ import annotations

__author__ = "Carlos Beluzo"
__email__ = "beluzo@ifsp.edu.br"

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _aligned_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Convert metric inputs to aligned one-dimensional float arrays."""
    # For pandas inputs, align by date index before computing errors. This is
    # important for rolling forecasts, where some horizons can be missing.
    if isinstance(y_true, pd.Series) and isinstance(y_pred, pd.Series):
        aligned = pd.concat([y_true.rename("actual"), y_pred.rename("predicted")], axis=1).dropna()
        return aligned["actual"].to_numpy(dtype="float64"), aligned["predicted"].to_numpy(dtype="float64")

    true = np.asarray(y_true, dtype="float64").reshape(-1)
    pred = np.asarray(y_pred, dtype="float64").reshape(-1)
    if true.shape != pred.shape:
        raise ValueError(f"Metric inputs must have the same shape. Got {true.shape} and {pred.shape}.")
    mask = ~(np.isnan(true) | np.isnan(pred))
    return true[mask], pred[mask]


def maerr(y_true, y_pred) -> float:
    """Mean absolute error."""
    true, pred = _aligned_arrays(y_true, y_pred)
    return float(mean_absolute_error(true, pred))


def mse(y_true, y_pred) -> float:
    """Mean squared error."""
    true, pred = _aligned_arrays(y_true, y_pred)
    return float(mean_squared_error(true, pred))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error returned as percentage points."""
    true, pred = _aligned_arrays(y_true, y_pred)
    # Avoid division by zero. Months with zero observed value are excluded from
    # MAPE while still allowing MAE/RMSE to be computed normally.
    non_zero = true != 0
    if not np.any(non_zero):
        return np.nan
    return float(np.mean(np.abs((true[non_zero] - pred[non_zero]) / true[non_zero])) * 100)


def get_mean_errors(y_true, y_pred) -> tuple[float, float, float]:
    """Return MAE, RMSE, and MAPE for a pair of series."""
    return maerr(y_true, y_pred), rmse(y_true, y_pred), mape(y_true, y_pred)
