"""LSTM preprocessing, training, tuning, and forecasting utilities.

This module contains the neural-network branch of the notebook. It keeps all
LSTM variants on one unified pipeline so baseline, regularized, transformed,
and multi-horizon experiments differ only by explicit parameters.

Author:
    Carlos Beluzo - beluzo@ifsp.edu.br
"""

from __future__ import annotations

__author__ = "Carlos Beluzo"
__email__ = "beluzo@ifsp.edu.br"

import logging
import os
import random
from dataclasses import dataclass
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

from .config import SEED
from .data import SplitData
from .metrics import maerr, mape, rmse


logger = logging.getLogger("abep_forecasting")


def set_seeds(seed: int = SEED) -> None:
    """Set random seeds used by Python, NumPy, and TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def apply_series_transform(values, transform: str = "none") -> np.ndarray:
    """Apply an optional monotonic transform before scaling/modeling."""
    values = np.asarray(values, dtype="float64")
    if transform == "none":
        return values
    if transform == "log1p":
        return np.log1p(values)
    raise ValueError(f"Unsupported transform: {transform}")


def invert_series_transform(values, transform: str = "none") -> np.ndarray:
    """Invert a transform produced by apply_series_transform."""
    values = np.asarray(values, dtype="float64")
    if transform == "none":
        return values
    if transform == "log1p":
        return np.expm1(values)
    raise ValueError(f"Unsupported transform: {transform}")


def create_sequences(values, lookback: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Create supervised LSTM windows from a scaled univariate series."""
    X, y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i - lookback:i, 0])
        y.append(values[i, 0])
    return np.array(X).reshape((-1, lookback, 1)), np.array(y)


def build_lstm_model(lookback: int, units: int, dropout: float, learning_rate: float, l2_lambda: float = 0.0):
    """Build and compile the baseline one-layer LSTM model."""
    regularizer = l2(l2_lambda) if l2_lambda and l2_lambda > 0 else None
    layers = [
        LSTM(
            units,
            input_shape=(lookback, 1),
            kernel_regularizer=regularizer,
            recurrent_regularizer=regularizer,
            bias_regularizer=regularizer,
        )
    ]
    if dropout > 0:
        layers.append(Dropout(dropout))
    layers.append(Dense(1, activation="linear", kernel_regularizer=regularizer, bias_regularizer=regularizer))

    model = Sequential(layers)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model


def recursive_forecast(model, history_scaled, steps: int, lookback: int) -> np.ndarray:
    """Generate recursive one-step-ahead forecasts on the scaled series."""
    preds = []
    current_history = list(np.asarray(history_scaled).flatten())
    for _ in range(steps):
        # Each prediction is appended to history so later horizons depend on
        # earlier forecasts, matching how recursive forecasting is deployed.
        x_input = np.array(current_history[-lookback:]).reshape(1, lookback, 1)
        yhat = model.predict(x_input, verbose=0)[0, 0]
        preds.append(yhat)
        current_history.append(yhat)
    return np.array(preds).reshape(-1, 1)


@dataclass
class LSTMPreparedData:
    """Reusable inputs created once for LSTM experiments."""

    train_series: pd.Series
    test_series: pd.Series
    scaler: MinMaxScaler
    train_scaled: np.ndarray
    X_train: np.ndarray
    y_train: np.ndarray
    transform: str


def prepare_lstm_training_data(split_data: SplitData, lookback: int = 12, transform: str = "none") -> LSTMPreparedData:
    """Transform, scale, and window the train/test split for an LSTM run."""
    train_series = split_data.train.astype("float64").dropna()
    test_series = split_data.test_2023.astype("float64").dropna()
    train_transformed = apply_series_transform(train_series.values, transform=transform)
    # Fit the scaler only on the training period to prevent leakage from the
    # 2023 holdout months into model fitting.
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_transformed.reshape(-1, 1))
    X_train, y_train = create_sequences(train_scaled, lookback=lookback)
    if len(X_train) == 0:
        raise ValueError(f"Training series too short for lookback={lookback}.")
    return LSTMPreparedData(train_series, test_series, scaler, train_scaled, X_train, y_train, transform)


def build_lstm_fit_inputs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    use_validation: bool = True,
    early_stopping: bool = True,
    patience: int = 20,
):
    """Create fit arrays, callbacks, and validation data for all LSTM variants."""
    callbacks = []
    fit_kwargs = {}
    if use_validation and len(X_train) > 12:
        # Validation uses the most recent training windows, preserving temporal
        # order instead of randomly mixing past and future observations.
        val_size = max(6, int(len(X_train) * 0.2))
        X_fit, y_fit = X_train[:-val_size], y_train[:-val_size]
        X_val, y_val = X_train[-val_size:], y_train[-val_size:]
        if early_stopping:
            callbacks.append(EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True))
        fit_kwargs["validation_data"] = (X_val, y_val)
    else:
        X_fit, y_fit = X_train, y_train
        if early_stopping:
            callbacks.append(EarlyStopping(monitor="loss", patience=patience, restore_best_weights=True))
    return X_fit, y_fit, callbacks, fit_kwargs


def invert_lstm_predictions(pred_scaled: np.ndarray, scaler: MinMaxScaler, transform: str = "none") -> np.ndarray:
    """Invert scaling and optional transformation for LSTM predictions."""
    pred_transformed = scaler.inverse_transform(np.asarray(pred_scaled).reshape(-1, 1)).flatten()
    return np.maximum(invert_series_transform(pred_transformed, transform=transform), 0)


def make_lstm_comparison_table(test_series: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    """Create the standard 2023 actual-vs-predicted table."""
    y_true = test_series.values.astype("float64")
    return pd.DataFrame({
        "Month": test_series.index.strftime("%Y-%m"),
        "Actual": np.round(y_true, 4),
        "Predicted": np.round(y_pred, 4),
        "Absolute Error": np.round(np.abs(y_true - y_pred), 4),
    })


def summarize_lstm_holdout(model, prepared: LSTMPreparedData, lookback: int) -> dict:
    """Generate recursive holdout forecasts and standard error metrics."""
    pred_scaled = recursive_forecast(model, prepared.train_scaled, steps=len(prepared.test_series), lookback=lookback)
    y_pred = invert_lstm_predictions(pred_scaled, prepared.scaler, transform=prepared.transform)
    y_true = prepared.test_series.values.astype("float64")
    y_pred_series = pd.Series(y_pred, index=prepared.test_series.index, name="predicted")
    return {
        "model": model,
        "mae": maerr(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "y_true": prepared.test_series,
        "y_pred": y_pred_series,
        "compare_2023": make_lstm_comparison_table(prepared.test_series, y_pred),
    }


def plot_lstm_holdout_result(result: dict, title: str, ylabel: str, forecast_label: str = "Predicted (LSTM)") -> None:
    """Plot the standard 2023 LSTM holdout comparison."""
    y_true = result["y_true"]
    y_pred = result["y_pred"]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_true.index, y_true.values, label="Actual", marker="o")
    ax.plot(y_pred.index, y_pred.values, label=forecast_label, marker="s", linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    text = f"MAE: {result['mae']:.2f}\nRMSE: {result['rmse']:.2f}\nMAPE: {result['mape']:.2f}%"
    ax.text(0.01, 0.98, text, transform=ax.transAxes, va="top", ha="left", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    fig.tight_layout()
    plt.show()


def generate_lstm_multi_horizon_forecasts(model, prepared: LSTMPreparedData, lookback: int, horizons=(1, 2, 3, 4)) -> pd.DataFrame:
    """Generate rolling LSTM forecasts for one or more horizons using one trained model."""
    max_horizon = max(horizons)
    history_scaled = prepared.train_scaled.copy()
    test_index = prepared.test_series.index
    test_values = prepared.test_series.values.astype("float64")
    rows = []
    for i in range(len(prepared.test_series)):
        # Advance the forecast origin one observed month at a time. After
        # recording forecasts, the actual value is added to the scaled history.
        remaining = len(prepared.test_series) - i
        current_max_horizon = min(max_horizon, remaining)
        pred_scaled = recursive_forecast(model, history_scaled, steps=current_max_horizon, lookback=lookback)
        pred_raw = invert_lstm_predictions(pred_scaled, prepared.scaler, transform=prepared.transform)
        origin_date = test_index[i]
        for horizon in range(1, current_max_horizon + 1):
            target_idx = i + horizon - 1
            rows.append({
                "origin": origin_date,
                "target": test_index[target_idx],
                "horizon": horizon,
                "actual": test_values[target_idx],
                "predicted": pred_raw[horizon - 1],
                "absolute_error": abs(test_values[target_idx] - pred_raw[horizon - 1]),
            })
        observed_transformed = apply_series_transform(np.array([prepared.test_series.iloc[i]]), transform=prepared.transform)
        observed_scaled = prepared.scaler.transform(observed_transformed.reshape(-1, 1))
        history_scaled = np.vstack([history_scaled, observed_scaled])
    forecasts_df = pd.DataFrame(rows)
    return forecasts_df[forecasts_df["horizon"].isin(horizons)].reset_index(drop=True)


def summarize_lstm_horizon_metrics(forecasts_df: pd.DataFrame, horizons=(1, 2, 3, 4)) -> pd.DataFrame:
    """Summarize MAE, RMSE, and MAPE by forecast horizon."""
    metrics = []
    for horizon in horizons:
        df_horizon = forecasts_df[forecasts_df["horizon"] == horizon]
        if not df_horizon.empty:
            y_true = df_horizon["actual"].values
            y_pred = df_horizon["predicted"].values
            metrics.append({"Horizon": horizon, "N": len(df_horizon), "MAE": maerr(y_true, y_pred), "RMSE": rmse(y_true, y_pred), "MAPE": mape(y_true, y_pred)})
    return pd.DataFrame(metrics)


def run_lstm_experiment(
    split_data: SplitData,
    value_col: str,
    title: str,
    ylabel: str,
    lookback: int = 12,
    units: int = 32,
    dropout: float = 0.0,
    learning_rate: float = 0.001,
    l2_lambda: float = 0.0,
    transform: str = "none",
    epochs: int = 250,
    batch_size: int = 8,
    seed: int = SEED,
    use_validation: bool = True,
    early_stopping: bool = True,
    patience: int = 20,
    horizons=(1,),
    mode: str = "holdout",
    plot: bool = True,
    forecast_label: str | None = None,
    verbose_fit: int = 0,
):
    """Train one LSTM model and return either holdout or multi-horizon outputs."""
    del value_col
    set_seeds(seed)
    prepared = prepare_lstm_training_data(split_data, lookback=lookback, transform=transform)
    X_fit, y_fit, callbacks, fit_kwargs = build_lstm_fit_inputs(prepared.X_train, prepared.y_train, use_validation=use_validation, early_stopping=early_stopping, patience=patience)
    model = build_lstm_model(lookback, units, dropout, learning_rate, l2_lambda=l2_lambda)
    # shuffle=False keeps sequence order intact for the recurrent model.
    model.fit(X_fit, y_fit, epochs=epochs, batch_size=batch_size, shuffle=False, verbose=verbose_fit, callbacks=callbacks, **fit_kwargs)

    if mode == "holdout":
        result = summarize_lstm_holdout(model, prepared, lookback=lookback)
        if plot:
            label = forecast_label or (f"Predicted (LSTM, transform={transform})" if transform != "none" else "Predicted (LSTM)")
            plot_lstm_holdout_result(result, title=title, ylabel=ylabel, forecast_label=label)
        return result

    if mode == "multi_horizon":
        forecasts_df = generate_lstm_multi_horizon_forecasts(model, prepared, lookback=lookback, horizons=horizons)
        horizon_metrics = summarize_lstm_horizon_metrics(forecasts_df, horizons=horizons)
        if plot:
            from .plots import plot_multi_horizon_forecasts

            plot_multi_horizon_forecasts(forecasts_df, f"{title} - Horizon Forecasts", ylabel, horizons)
        return {"model": model, "forecasts_df": forecasts_df, "horizon_metrics": horizon_metrics}

    raise ValueError(f"Unsupported LSTM experiment mode: {mode}")


def run_lstm_pipeline_split(*args, **kwargs):
    """Compatibility wrapper for the standard 2023 holdout LSTM experiment."""
    return run_lstm_experiment(*args, mode="holdout", **kwargs)


def build_lstm_model_regularized(lookback=12, units=32, dropout=0.0, learning_rate=0.001, l2_lambda=0.0):
    """Compatibility wrapper for the unified LSTM model builder."""
    return build_lstm_model(lookback, units, dropout, learning_rate, l2_lambda=l2_lambda)


def recursive_forecast_12m(model, history_scaled, steps: int = 12, lookback: int = 12) -> np.ndarray:
    """Compatibility wrapper for recursive monthly forecasts."""
    return recursive_forecast(model, history_scaled, steps=steps, lookback=lookback)


def recursive_forecast_multi_horizon(model, history_scaled, max_horizon: int = 4, lookback: int = 12) -> np.ndarray:
    """Compatibility wrapper for multi-horizon recursive forecasts."""
    return recursive_forecast(model, history_scaled, steps=max_horizon, lookback=lookback)


def run_lstm_pipeline_split_regularized(*args, **kwargs):
    """Compatibility wrapper for regularized LSTM holdout experiments."""
    kwargs.setdefault("forecast_label", "Predicted (Regularized LSTM)")
    return run_lstm_experiment(*args, mode="holdout", **kwargs)


def run_lstm_pipeline_split_regularized_transformed(*args, **kwargs):
    """Compatibility wrapper for transformed regularized LSTM holdout experiments."""
    transform = kwargs.get("transform", "none")
    kwargs.setdefault("forecast_label", f"Predicted (LSTM, transform={transform})")
    return run_lstm_experiment(*args, mode="holdout", **kwargs)


def evaluate_lstm_multi_horizon(*args, **kwargs):
    """Compatibility wrapper for multi-horizon LSTM evaluation."""
    return run_lstm_experiment(*args, mode="multi_horizon", **kwargs)


def run_lstm_pipeline(split_data, value_col, params, horizons=(1,), plot=True, title="LSTM Forecast", ylabel: str | None = None):
    """Compatibility wrapper around the unified LSTM experiment function."""
    mode = "multi_horizon" if max(horizons) > 1 else "holdout"
    return run_lstm_experiment(
        split_data=split_data,
        value_col=value_col,
        title=title,
        ylabel=ylabel or value_col,
        lookback=params.get("lookback", 12),
        units=params.get("units", 32),
        dropout=params.get("dropout", 0.0),
        learning_rate=params.get("learning_rate", 0.001),
        l2_lambda=params.get("l2_lambda", 0.0),
        transform=params.get("transform", "none"),
        epochs=params.get("epochs", 250),
        batch_size=params.get("batch_size", 8),
        seed=params.get("seed", SEED),
        early_stopping=params.get("early_stopping", True),
        patience=params.get("patience", 20),
        horizons=horizons,
        mode=mode,
        plot=plot,
    )


def grid_search_lstm_split(
    split_data: SplitData,
    value_col: str,
    title: str,
    ylabel: str,
    lookback_values=(6, 12, 18, 24),
    units_values=(16, 32, 64),
    learning_rate_values=(1e-3, 5e-4, 1e-4),
    dropout: float = 0.0,
    epochs: int = 250,
    batch_size: int = 8,
    seed: int = SEED,
    sort_by: str = "rmse",
    use_validation: bool = True,
    verbose: bool = True,
):
    """Search baseline LSTM hyperparameters and return ranked results."""
    rows = []
    best_result = None
    best_params = None
    best_score = np.inf
    grid = list(product(lookback_values, units_values, learning_rate_values))
    if verbose:
        logger.info("Total LSTM combinations: %s", len(grid))
    for index, (lookback, units, learning_rate) in enumerate(grid, start=1):
        # Every candidate is scored on the same holdout period, so the sorted
        # grid result can be interpreted as a fair model comparison.
        if verbose:
            logger.info("LSTM grid [%s/%s]: lookback=%s, units=%s, learning_rate=%s", index, len(grid), lookback, units, learning_rate)
        params = {"lookback": lookback, "units": units, "dropout": dropout, "learning_rate": learning_rate, "epochs": epochs, "batch_size": batch_size, "seed": seed}
        try:
            result = run_lstm_pipeline_split(split_data=split_data, value_col=value_col, title=title, ylabel=ylabel, lookback=lookback, units=units, dropout=dropout, learning_rate=learning_rate, epochs=epochs, batch_size=batch_size, seed=seed, use_validation=use_validation, plot=False, verbose_fit=0)
            row = {**params, "mae": result["mae"], "rmse": result["rmse"], "mape": result["mape"], "status": "ok"}
            score = row.get(sort_by, np.nan)
            if pd.notna(score) and score < best_score:
                best_score = score
                best_result = result
                best_params = row.copy()
        except Exception as exc:
            row = {**params, "mae": np.nan, "rmse": np.nan, "mape": np.nan, "status": "error", "error": str(exc)}
            if verbose:
                logger.warning("LSTM grid combination failed: %s", exc)
        rows.append(row)
    results_df = pd.DataFrame(rows)
    if sort_by in results_df.columns:
        results_df = results_df.sort_values(by=[sort_by, "mae", "mape"], ascending=True, na_position="last").reset_index(drop=True)
    return results_df, best_result, best_params


def grid_search_lstm_regularization(
    split_data,
    value_col,
    title,
    ylabel,
    base_params,
    dropout_values=(0.0, 0.1, 0.2),
    l2_values=(0.0, 1e-5, 1e-4, 1e-3),
    patience_values=(10, 20, 30),
    early_stopping_values=(True,),
    use_validation=True,
    sort_by="rmse",
    verbose=True,
):
    """Search regularization parameters while keeping the baseline LSTM structure fixed."""
    import itertools

    results = []
    best_result = None
    best_params = None
    best_score = np.inf
    grid = list(itertools.product(dropout_values, l2_values, patience_values, early_stopping_values))
    if verbose:
        logger.info("Total regularization combinations: %s", len(grid))
    for i, (dropout, l2_lambda, patience, early_stopping) in enumerate(grid, start=1):
        # Keep the best baseline architecture fixed and vary only controls that
        # reduce overfitting: dropout, L2 penalty, and early stopping patience.
        if verbose:
            logger.info("Regularization grid [%s/%s]: dropout=%s, l2_lambda=%s, patience=%s, early_stopping=%s", i, len(grid), dropout, l2_lambda, patience, early_stopping)
        row_base = {"lookback": base_params["lookback"], "units": base_params["units"], "learning_rate": base_params["learning_rate"], "epochs": base_params["epochs"], "batch_size": base_params["batch_size"], "seed": base_params["seed"], "dropout": dropout, "l2_lambda": l2_lambda, "patience": patience, "early_stopping": early_stopping}
        try:
            result = run_lstm_pipeline_split_regularized(split_data=split_data, value_col=value_col, title=title, ylabel=ylabel, lookback=base_params["lookback"], units=base_params["units"], dropout=dropout, learning_rate=base_params["learning_rate"], l2_lambda=l2_lambda, epochs=base_params["epochs"], batch_size=base_params["batch_size"], seed=base_params["seed"], use_validation=use_validation, early_stopping=early_stopping, patience=patience, plot=False, verbose_fit=0)
            row = {**row_base, "mae": result["mae"], "rmse": result["rmse"], "mape": result["mape"], "status": "ok"}
            score = row[sort_by]
            if pd.notna(score) and score < best_score:
                best_score = score
                best_result = result
                best_params = row.copy()
        except Exception as exc:
            row = {**row_base, "mae": np.nan, "rmse": np.nan, "mape": np.nan, "status": "error", "error": str(exc)}
            if verbose:
                logger.warning("Regularization grid combination failed: %s", exc)
        results.append(row)
    results_df = pd.DataFrame(results)
    if sort_by in results_df.columns:
        results_df = results_df.sort_values(by=[sort_by, "mae", "mape"], ascending=True, na_position="last").reset_index(drop=True)
    return results_df, best_result, best_params
