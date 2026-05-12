"""SARIMA diagnostics, model search, forecasting, and narrative reporting.

The functions here implement the statistical time-series branch of the
notebook: stationarity checks, seasonal differencing suggestions, grid search,
holdout forecasts, future projections, and the textual model report.

Author:
    Carlos Beluzo - beluzo@ifsp.edu.br
"""

from __future__ import annotations

__author__ = "Carlos Beluzo"
__email__ = "beluzo@ifsp.edu.br"

import warnings
from dataclasses import asdict, dataclass
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss

from .config import FORECAST_YEAR, TEST_YEAR
from .data import SplitData
from .metrics import maerr, mape, rmse

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=InterpolationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def ensure_datetime_series(series: pd.Series) -> pd.Series:
    s = pd.Series(series).copy()
    s = s.astype("float64").dropna()
    if not isinstance(s.index, pd.DatetimeIndex):
        raise ValueError("The series must have a DatetimeIndex.")
    return s.sort_index()


def test_adf(series: pd.Series) -> float:
    try:
        return adfuller(series.dropna(), autolag="AIC")[1]
    except Exception:
        return np.nan


def test_kpss(series: pd.Series, regression="c") -> float:
    try:
        return kpss(series.dropna(), regression=regression, nlags="auto")[1]
    except Exception:
        return np.nan


def test_ljungbox_residuals(residuals: pd.Series, lag: int = 12) -> float:
    try:
        residuals = pd.Series(residuals).dropna()
        if len(residuals) <= lag:
            return np.nan
        out = acorr_ljungbox(residuals, lags=[lag], return_df=True)
        return out["lb_pvalue"].iloc[0]
    except Exception:
        return np.nan


def stationarity_diagnostic(series: pd.Series, regression="c") -> dict:
    p_adf = test_adf(series)
    p_kpss = test_kpss(series, regression=regression)
    if np.isnan(p_adf) or np.isnan(p_kpss):
        status = "inconclusive"
    elif p_adf < 0.05 and p_kpss >= 0.05:
        status = "stationary"
    elif p_adf >= 0.05 and p_kpss < 0.05:
        status = "non_stationary"
    else:
        status = "conflicting"
    return {"adf_pvalue": p_adf, "kpss_pvalue": p_kpss, "status": status}


def seasonal_strength_stl(series: pd.Series, period: int = 12) -> float:
    try:
        s = series.dropna()
        if len(s) < 2 * period:
            return np.nan
        res = STL(s, period=period, robust=True).fit()
        denom = np.var(res.resid + res.seasonal)
        if denom == 0:
            return np.nan
        return max(0.0, 1.0 - np.var(res.resid) / denom)
    except Exception:
        return np.nan


def suggest_d_D(series: pd.Series, period: int = 12) -> dict:
    """Suggest non-seasonal and seasonal differencing orders."""
    candidates = []
    # Evaluate only a compact and interpretable differencing grid. This keeps
    # the SARIMA search stable for short monthly public-health series.
    transformations = {
        (0, 0): series,
        (1, 0): series.diff(),
        (0, 1): series.diff(period),
        (1, 1): series.diff().diff(period),
    }
    for (d, D), s in transformations.items():
        s = s.dropna()
        if len(s) < max(24, 2 * period):
            continue
        diag = stationarity_diagnostic(s)
        candidates.append({"d": d, "D": D, "adf_pvalue": diag["adf_pvalue"], "kpss_pvalue": diag["kpss_pvalue"], "status": diag["status"]})
    if not candidates:
        # Fallback favors one regular difference when the sample is too short
        # for reliable stationarity diagnostics across all candidates.
        return {"best": {"d": 1, "D": 0, "status": "fallback"}, "candidates": []}
    ranking = sorted(candidates, key=lambda x: (0 if x["status"] == "stationary" else 1 if x["status"] == "conflicting" else 2, x["d"] + x["D"], x["D"], x["d"]))
    return {"best": ranking[0], "candidates": ranking}


def forecast_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    base = pd.concat([y_true.rename("y"), y_pred.rename("yhat")], axis=1).dropna()
    error = base["y"] - base["yhat"]
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))
    mape = np.nan if np.any(base["y"] == 0) else np.mean(np.abs(error / base["y"])) * 100
    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape) if not np.isnan(mape) else np.nan}


@dataclass
class ModelResult:
    order: tuple
    seasonal_order: tuple
    aic: float
    bic: float
    mae: float
    rmse: float
    mape: float
    lb_pvalue: float
    converged: bool
    n_params: int


def adjust_one_sarima(train: pd.Series, test: pd.Series, order: tuple, seasonal_order: tuple, trend: str = "n", enforce_stationarity: bool = False, enforce_invertibility: bool = False):
    try:
        model = SARIMAX(train, order=order, seasonal_order=seasonal_order, trend=trend, enforce_stationarity=enforce_stationarity, enforce_invertibility=enforce_invertibility)
        fit = model.fit(disp=False)
        forecast = fit.get_forecast(steps=len(test)).predicted_mean
        forecast.index = test.index
        mets = forecast_metrics(test, forecast)
        # Use a lag compatible with the available sample size, avoiding Ljung-
        # Box tests with more lags than the fitted residual series supports.
        lag_lb = min(12, max(1, len(train) // 5))
        p_lb = test_ljungbox_residuals(fit.resid, lag=lag_lb)
        return ModelResult(order=order, seasonal_order=seasonal_order, aic=float(fit.aic), bic=float(fit.bic), mae=mets["mae"], rmse=mets["rmse"], mape=mets["mape"], lb_pvalue=float(p_lb) if not np.isnan(p_lb) else np.nan, converged=bool(fit.mle_retvals.get("converged", True)), n_params=len(fit.params))
    except Exception:
        return None


def grid_search_sarima(series: pd.Series, name: str = "Series", period: int = 12, test_size: int = 12, p_values=(0, 1, 2), q_values=(0, 1, 2), P_values=(0, 1, 2), Q_values=(0, 1, 2), trend: str = "n"):
    s = ensure_datetime_series(series)
    if len(s) <= test_size + 24:
        raise ValueError("Series too short for SARIMA grid search with holdout validation.")
    # Reserve the last 12 months as a true out-of-sample holdout. The grid is
    # ranked by both information criteria and predictive errors on this split.
    train = s.iloc[:-test_size]
    test = s.iloc[-test_size:]
    diag_original = stationarity_diagnostic(train)
    seasonality = seasonal_strength_stl(train, period=period)
    suggestion_dd = suggest_d_D(train, period=period)
    d = suggestion_dd["best"]["d"]
    D = suggestion_dd["best"]["D"]
    results = []
    for p, q, P, Q in product(p_values, q_values, P_values, Q_values):
        if p == q == P == Q == 0:
            continue
        res = adjust_one_sarima(train=train, test=test, order=(p, d, q), seasonal_order=(P, D, Q, period), trend=trend)
        if res is not None:
            results.append(asdict(res))
    if not results:
        raise RuntimeError("No model converged during search.")
    df_results = pd.DataFrame(results)
    df_results["residuals_ok"] = df_results["lb_pvalue"] >= 0.05
    # Composite score balances parsimony and predictive accuracy. The weights
    # intentionally keep RMSE/AIC influential while still penalizing complexity.
    df_results["score"] = df_results["aic"].rank(method="min") * 0.30 + df_results["bic"].rank(method="min") * 0.20 + df_results["rmse"].rank(method="min") * 0.30 + df_results["mae"].rank(method="min") * 0.15 + df_results["n_params"].rank(method="min") * 0.05
    valid_candidates = df_results[(df_results["converged"]) & (df_results["residuals_ok"])].copy()
    if valid_candidates.empty:
        valid_candidates = df_results[df_results["converged"]].copy()
    valid_candidates = valid_candidates.sort_values(["score", "rmse", "aic", "bic"]).reset_index(drop=True)
    return {
        "series_name": name,
        "n_obs": len(s),
        "train_size": len(train),
        "test_size": len(test),
        "seasonal_period": period,
        "seasonal_strength": seasonality,
        "original_diagnostic": diag_original,
        "suggestion_d_D": suggestion_dd,
        "best_model": valid_candidates.iloc[0].to_dict(),
        "top_models": valid_candidates.head(10).copy(),
        "all_models": df_results.sort_values(["score", "rmse", "aic"]).reset_index(drop=True),
    }


def run_and_plot_sarima(split_data: SplitData, params: dict, label: str, ylabel: str, test_year: int = TEST_YEAR):
    """Fit the selected SARIMA model and plot the holdout forecast."""
    model = SARIMAX(
        split_data.train,
        order=params["order"],
        seasonal_order=params["seasonal_order"],
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    results = model.fit(disp=False)

    y_true = split_data.test_2023
    y_pred = results.get_forecast(steps=len(y_true)).predicted_mean
    y_pred.index = y_true.index

    mae_val = maerr(y_true, y_pred)
    rmse_val = rmse(y_true, y_pred)
    mape_val = mape(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_true.index, y_true.values, label=f"Actual {test_year}", marker="o", color="black", linewidth=2)
    ax.plot(y_pred.index, y_pred.values, label=f"Predicted {test_year}", marker="s", linestyle="--", color="#1f77b4", linewidth=2)
    ax.set_title(f"SARIMA Forecast - {label}", fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Month")
    ax.legend(frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()

    print(f"Metrics for {label}:")
    print(f"MAE: {mae_val:.2f} | RMSE: {rmse_val:.2f} | MAPE: {mape_val:.2f}%")
    return results, mae_val, rmse_val, mape_val


def generate_final_pipeline(
    series: pd.Series,
    test_split: SplitData,
    best_params: dict,
    y_pred_2023: pd.Series,
    name: str,
    forecast_year: int = FORECAST_YEAR,
    forecast_steps: int = 6,
):
    """Build validation metrics, the holdout comparison table, and a future forecast table."""
    y_true_2023 = test_split.test_2023
    y_pred_2023 = pd.Series(y_pred_2023, index=y_true_2023.index)

    metrics_df = pd.DataFrame({
        "Indicator": [name],
        "MAE": [round(maerr(y_true_2023, y_pred_2023), 1)],
        "RMSE": [round(rmse(y_true_2023, y_pred_2023), 1)],
        "MAPE (%)": [f"{mape(y_true_2023, y_pred_2023):.1f}%"],
    })

    compare_2023 = pd.DataFrame({
        "Month": y_true_2023.index.strftime("%Y-%m"),
        "Actual": y_true_2023.values,
        "Predicted": y_pred_2023.values.round(2),
        "Absolute Error": np.abs(y_true_2023.values - y_pred_2023.values).round(1),
    })

    model_full = SARIMAX(
        series,
        order=best_params["order"],
        seasonal_order=best_params["seasonal_order"],
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    results_full = model_full.fit(disp=False)
    # After holdout evaluation, refit on the full observed series to produce
    # short-term operational projections for the following months.
    forecast = results_full.get_forecast(steps=forecast_steps).predicted_mean
    forecast.index = pd.date_range(series.index.max() + pd.offsets.MonthBegin(1), periods=forecast_steps, freq="MS")

    forecast_df = pd.DataFrame({
        "Month": forecast.index.strftime("%Y-%m"),
        f"{name} Projection": forecast.values.round(1),
    })
    forecast_df.attrs["forecast_year"] = forecast_year

    return metrics_df, compare_2023, forecast_df


def generate_sarima_report(report: dict) -> str:
    diag = report["original_diagnostic"]
    best = report["best_model"]
    dd = report["suggestion_d_D"]["best"]
    seasonal_strength = report["seasonal_strength"]
    text = []
    text.append(f"An analysis was performed for series '{report['series_name']}' with {report['n_obs']} observations, of which {report['train_size']} were used for fitting and {report['test_size']} reserved for out-of-sample evaluation.")
    text.append(f"On the training series, the ADF test showed a p-value of {diag['adf_pvalue']:.4f}, while the KPSS test showed a p-value of {diag['kpss_pvalue']:.4f}. Together, these results indicated a diagnostic status of '{diag['status']}'.")
    if not np.isnan(seasonal_strength):
        text.append(f"STL decomposition indicated seasonal strength of {seasonal_strength:.4f}, which was considered in defining the model's seasonal structure.")
    text.append(f"The analysis of candidate transformations suggested the combination d={dd['d']} and D={dd['D']}, as it presented the best balance between stationarity and parsimony.")
    text.append(f"In the grid search, the selected model was SARIMA{tuple(best['order'])}x{tuple(best['seasonal_order'])}, with AIC={best['aic']:.2f}, BIC={best['bic']:.2f}, MAE={best['mae']:.4f}, RMSE={best['rmse']:.4f}" + (f", MAPE={best['mape']:.2f}%" if not np.isnan(best["mape"]) else ", MAPE not calculable due to zero/null values in test base") + f" and Ljung-Box p-value on residuals={best['lb_pvalue']:.4f}.")
    if not np.isnan(best["lb_pvalue"]) and best["lb_pvalue"] >= 0.05:
        text.append("The Ljung-Box test applied to the residuals did not reject the hypothesis of absence of residual autocorrelation, favoring the statistical adequacy of the chosen model.")
    else:
        text.append("The residuals of the model still show signs of autocorrelation; the selected model should be interpreted as the best relative alternative within the tested grid, not as a final definitive specification.")
    text.append("As a next step, it is recommended to visually inspect residuals, expand the search grid if necessary and compare the selected SARIMA with alternative approaches such as ETS or machine learning models, especially if the primary goal is forecasting.")
    return "\n\n".join(text)
