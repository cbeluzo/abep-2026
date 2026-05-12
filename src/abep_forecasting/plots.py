"""Plotting and rolling-forecast table utilities for notebook outputs.

The plotting functions are intentionally kept outside the notebook so figures
for SARIMA and LSTM use consistent labels, colors, and table formats.

Author:
    Carlos Beluzo - beluzo@ifsp.edu.br
"""

from __future__ import annotations

__author__ = "Carlos Beluzo"
__email__ = "beluzo@ifsp.edu.br"

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .config import DATE_COLUMN, SEASONAL_PERIOD, TEST_YEAR


def get_month_series(df: pd.DataFrame, year: int, value_col: str, date_column: str = DATE_COLUMN) -> np.ndarray:
    """Return the 12 monthly values for a calendar year."""
    data = df[df[date_column].dt.year == year].sort_values(date_column)
    return data[value_col].to_numpy()


def plot_decomposition(series: pd.Series, title: str, filename: str, output_dir: Path, figsize: tuple[int, int] = (10, 7)) -> None:
    """Create and save an additive seasonal decomposition plot."""
    decomposition = seasonal_decompose(series, model="additive", period=SEASONAL_PERIOD)
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    # Keep component order aligned with standard time-series decomposition:
    # observed, trend, seasonal pattern, and residual noise.
    components = [(decomposition.observed, "Observed", "#2E86AB"), (decomposition.trend, "Trend", "#F6C85F"), (decomposition.seasonal, "Seasonal", "#6C5B7B"), (decomposition.resid, "Residual", "#C06C84")]
    for ax, (values, label, color) in zip(axes, components):
        ax.plot(values, color=color, linewidth=1.5)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    axes[-1].set_xlabel("Date")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / filename, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def generate_rolling_forecasts(full_series: pd.Series, params: dict, year: int = TEST_YEAR, steps: int = 4):
    """Generate SARIMA rolling forecasts and horizon-specific error tables."""
    target_months = pd.date_range(f"{year}-01-01", f"{year}-12-01", freq="MS")
    forecast_months = pd.date_range(target_months.min(), periods=len(target_months) + steps - 1, freq="MS")
    pred_map = {month: {} for month in forecast_months}

    month_labels = {
        "January": "Jan",
        "February": "Feb",
        "March": "Mar",
        "April": "Apr",
        "May": "May",
        "June": "Jun",
        "July": "Jul",
        "August": "Aug",
        "September": "Sep",
        "October": "Oct",
        "November": "Nov",
        "December": "Dec",
    }

    for current_month in target_months:
        # Train each origin only with information available before the target
        # month, which simulates real rolling forecast production.
        train_end = current_month - pd.offsets.MonthBegin(1)
        train_data = full_series[:train_end]
        if len(train_data) < 24:
            continue

        model = SARIMAX(
            train_data,
            order=params["order"],
            seasonal_order=params["seasonal_order"],
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False)
        forecast = result.get_forecast(steps=steps).predicted_mean

        for horizon in range(1, steps + 1):
            target_date = current_month + pd.offsets.MonthBegin(horizon - 1)
            if target_date in pred_map and horizon <= len(forecast):
                pred_map[target_date][horizon] = forecast.iloc[horizon - 1]

    absolute_rows = []
    percentage_rows = []
    for month in target_months:
        actual = full_series.get(month, np.nan)
        row_abs = {"Month (Actual)": month_labels[month.strftime("%B")]}
        row_pct = {"Month (Actual)": month_labels[month.strftime("%B")]}

        for horizon in range(1, steps + 1):
            predicted = pred_map[month].get(horizon, np.nan)
            if pd.notnull(predicted) and pd.notnull(actual):
                abs_error = abs(actual - predicted)
                row_abs[f"H{horizon}"] = abs_error
                row_pct[f"H{horizon}"] = (abs_error / actual) * 100 if actual != 0 else np.nan
            else:
                row_abs[f"H{horizon}"] = np.nan
                row_pct[f"H{horizon}"] = np.nan

        absolute_rows.append(row_abs)
        percentage_rows.append(row_pct)

    horizons_plot = {
        horizon: [pred_map[month].get(horizon, np.nan) for month in target_months]
        for horizon in range(1, steps + 1)
    }
    forecast_df = pd.DataFrame(horizons_plot, index=target_months)
    return forecast_df, pd.DataFrame(absolute_rows), pd.DataFrame(percentage_rows)


def plot_rolling_comparison(full_series: pd.Series, forecast_df: pd.DataFrame, title: str, ploc: str = "upper right") -> None:
    """Plot actual values against SARIMA rolling forecasts for each horizon."""
    actual = full_series.loc[forecast_df.index]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(actual.index, actual.values, label="Actual", marker="o", color="black", linewidth=3)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for horizon in range(1, min(4, len(forecast_df.columns)) + 1):
        ax.plot(
            forecast_df.index,
            forecast_df[horizon],
            label=f"{horizon}-Month Horizon",
            linestyle="--",
            marker="x",
            alpha=0.7,
            color=colors[horizon - 1],
        )

    ax.axvline(pd.to_datetime(f"{TEST_YEAR}-12-01"), color="gray", linestyle=":", linewidth=2, label=f"End of {TEST_YEAR}")
    ax.set_xticks(forecast_df.index)
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_title(f"SARIMA - 4-Month Horizon Forecasts ({TEST_YEAR}): {title}", fontsize=14)
    ax.legend(loc=ploc, frameon=True, shadow=True, fontsize="small")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def plot_error_heatmap(df_results: pd.DataFrame, title: str, error_type: str = "Absolute") -> None:
    """Plot absolute or percentage SARIMA rolling-forecast errors as a heatmap."""
    df_heatmap = df_results.set_index("Month (Actual)")
    fig, ax = plt.subplots(figsize=(6, 5))

    if error_type == "Percentage":
        cbar_label = "Percentage Error (%)"
        annot_df = df_heatmap.map(lambda x: f"{x:.1f}%" if pd.notnull(x) else "")
        fmt_str = ""
    else:
        cbar_label = "Absolute Error"
        annot_df = True
        fmt_str = ".1f"

    sns.heatmap(df_heatmap, annot=annot_df, fmt=fmt_str, cmap="YlOrRd", cbar_kws={"label": cbar_label}, ax=ax)
    ax.set_title(f"SARIMA - {cbar_label} ({TEST_YEAR}): {title}", fontsize=14)
    ax.set_xlabel("Projection Horizon (Months)")
    ax.set_ylabel("Reference Month")
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    plt.show()


def plot_multi_horizon_forecasts(forecasts_df, title, ylabel, horizons=(1, 2, 3, 4)):
    """Plot actual values and LSTM forecasts for each requested horizon."""
    df = forecasts_df.copy()
    df["target"] = pd.to_datetime(df["target"])
    real_df = df[["target", "actual"]].drop_duplicates(subset=["target"]).sort_values("target")
    plt.figure(figsize=(12, 5))
    plt.plot(real_df["target"], real_df["actual"], marker="o", linewidth=2, label="Actual")
    for h in horizons:
        df_h = df[df["horizon"] == h][["target", "predicted"]].sort_values("target")
        if not df_h.empty:
            plt.plot(df_h["target"], df_h["predicted"], marker="s", linestyle="--", label=f"Predicted (h={h})")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def generate_lstm_rolling_outputs(lstm_multi_result: dict, year: int = TEST_YEAR, steps: int = 4):
    """Build LSTM rolling forecast and absolute-error tables from multi-horizon output."""
    forecasts_df = lstm_multi_result["forecasts_df"].copy()
    forecasts_df["target"] = pd.to_datetime(forecasts_df["target"])
    forecasts_df["origin"] = pd.to_datetime(forecasts_df["origin"])

    months = pd.date_range(f"{year}-01-01", f"{year}-12-01", freq="MS")
    month_labels = {
        "January": "Jan",
        "February": "Feb",
        "March": "Mar",
        "April": "Apr",
        "May": "May",
        "June": "Jun",
        "July": "Jul",
        "August": "Aug",
        "September": "Sep",
        "October": "Oct",
        "November": "Nov",
        "December": "Dec",
    }

    forecast_table = forecasts_df.pivot_table(
        index="target",
        columns="horizon",
        values="predicted",
        aggfunc="first",
    ).reindex(months)

    # Ensure the exported table always has the same H1-H4 columns even when the
    # last months cannot support longer forecast horizons.
    for horizon in range(1, steps + 1):
        if horizon not in forecast_table.columns:
            forecast_table[horizon] = np.nan
    forecast_table = forecast_table[[horizon for horizon in range(1, steps + 1)]]

    error_rows = []
    for month in months:
        row = {"Month (Actual)": month_labels[month.strftime("%B")]}
        for horizon in range(1, steps + 1):
            match = forecasts_df[(forecasts_df["target"] == month) & (forecasts_df["horizon"] == horizon)]
            row[f"H{horizon}"] = float(match["absolute_error"].iloc[0]) if not match.empty else np.nan
        error_rows.append(row)

    return forecast_table, pd.DataFrame(error_rows)


def plot_lstm_rolling_comparison(full_series, forecast_df, title, ploc="upper right"):
    """Plot actual values against LSTM rolling forecasts for each horizon."""
    real_2023 = full_series.loc[forecast_df.index]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(real_2023.index, real_2023.values, label="Actual", marker="o", color="black", linewidth=3)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for h in range(1, 5):
        if h in forecast_df.columns:
            ax.plot(forecast_df.index, forecast_df[h], label=f"{h}-Month Horizon", linestyle="--", marker="x", alpha=0.7, color=colors[h - 1])
    ax.axvline(pd.to_datetime(f"{TEST_YEAR}-12-01"), color="gray", linestyle=":", linewidth=2, label=f"End of {TEST_YEAR}")
    ax.set_xticks(forecast_df.index)
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    plt.title(f"LSTM - 4-Month Horizon Forecasts ({TEST_YEAR}): {title}", fontsize=14)
    plt.legend(loc=ploc, frameon=True, shadow=True, fontsize="small")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_lstm_error_heatmap(df_results, title):
    """Plot LSTM absolute rolling-forecast errors as a heatmap."""
    df_heatmap = df_results.set_index("Month (Actual)")
    plt.figure(figsize=(6, 5))
    sns.heatmap(df_heatmap, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={"label": "Absolute Error"})
    plt.title(f"LSTM - Absolute Error ({TEST_YEAR}): {title}", fontsize=14)
    plt.xlabel("Projection Horizon (Months)")
    plt.ylabel("Reference Month")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
