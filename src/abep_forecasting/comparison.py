"""Final model-comparison table builder.

This module consolidates metrics produced by independent SARIMA and LSTM
sections into a single ranked table. It accepts a context dictionary so the
notebook can call it after optional model sections have run.

Author:
    Carlos Beluzo - beluzo@ifsp.edu.br
"""

from __future__ import annotations

__author__ = "Carlos Beluzo"
__email__ = "beluzo@ifsp.edu.br"

import logging

import pandas as pd


logger = logging.getLogger("abep_forecasting")


def append_comparison_row(rows: list[dict], model: str, variant: str, metrics: dict, notes: str = "") -> None:
    """Append one standardized model-comparison row when all metrics are available."""
    required = ("mae", "rmse", "mape")
    if not all(metric in metrics and pd.notna(metrics[metric]) for metric in required):
        logger.warning("Skipping %s / %s because one or more metrics are missing.", model, variant)
        return
    rows.append({"Model": model, "Variant": variant, "MAE": float(metrics["mae"]), "RMSE": float(metrics["rmse"]), "MAPE (%)": float(metrics["mape"]), "Notes": notes})


def metrics_from_result(result: dict) -> dict:
    """Extract MAE, RMSE, and MAPE from a model result dictionary."""
    return {"mae": result.get("mae"), "rmse": result.get("rmse"), "mape": result.get("mape")}


def build_final_model_comparison(context: dict) -> pd.DataFrame:
    """Build a ranked comparison table for SARIMA and available LSTM variants."""
    rows = []
    # SARIMA metrics are stored as scalar notebook variables because they come
    # from the selected statsmodels fit.
    if all(name in context for name in ("mae_val", "rmse_val", "mape_val", "best_params")):
        best_params = context["best_params"]
        append_comparison_row(rows, "SARIMA", "Selected grid model", {"mae": context["mae_val"], "rmse": context["rmse_val"], "mape": context["mape_val"]}, f"SARIMA{tuple(best_params['order'])}x{tuple(best_params['seasonal_order'])}")
    model_results = [
        ("LSTM", "Baseline", "lstm_result", "Manual baseline parameters"),
        ("LSTM", "Best grid model", "best_lstm_final", "Best baseline grid-search configuration"),
        ("LSTM", "Regularized", "best_lstm_reg_final", "Dropout/L2/early-stopping configuration"),
        ("LSTM", "Best transformation", "best_lstm_transform_final", f"Transform={context.get('best_transform', 'not selected')}"),
    ]
    # LSTM sections are optional in the notebook; add each row only when that
    # model result exists in the current execution context.
    for model, variant, result_name, notes in model_results:
        if result_name in context:
            append_comparison_row(rows, model, variant, metrics_from_result(context[result_name]), notes)
    if "lstm_multi_result" in context and "horizon_metrics" in context["lstm_multi_result"]:
        horizon_metrics = context["lstm_multi_result"]["horizon_metrics"]
        if not horizon_metrics.empty:
            # Horizon 1 is directly comparable with the one-step holdout
            # forecasts used by SARIMA and the standard LSTM variants.
            horizon_1 = horizon_metrics[horizon_metrics["Horizon"] == 1]
            if not horizon_1.empty:
                row = horizon_1.iloc[0]
                append_comparison_row(rows, "LSTM", "Multi-horizon H1", {"mae": row["MAE"], "rmse": row["RMSE"], "mape": row["MAPE"]}, "One-month horizon from rolling multi-horizon evaluation")
    if not rows:
        logger.warning("No model metrics available for the final comparison table.")
        return pd.DataFrame(columns=["Rank by RMSE", "Model", "Variant", "MAE", "RMSE", "MAPE (%)", "Notes"])
    comparison = pd.DataFrame(rows)
    comparison["Rank by RMSE"] = comparison["RMSE"].rank(method="min").astype("Int64")
    comparison = comparison.sort_values(["RMSE", "MAE", "MAPE (%)"], ascending=True).reset_index(drop=True)
    comparison[["MAE", "RMSE", "MAPE (%)"]] = comparison[["MAE", "RMSE", "MAPE (%)"]].round(4)
    return comparison[["Rank by RMSE", "Model", "Variant", "MAE", "RMSE", "MAPE (%)", "Notes"]]
