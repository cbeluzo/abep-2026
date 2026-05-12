"""Input/output helpers for reproducible notebook artifacts.

The functions in this module centralize file naming, CSV/JSON exports, and
small result tables. Keeping this logic here makes the output directory
structure consistent across SARIMA and LSTM experiments.

Author:
    Carlos Beluzo - beluzo@ifsp.edu.br
"""

from __future__ import annotations

__author__ = "Carlos Beluzo"
__email__ = "beluzo@ifsp.edu.br"

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd


logger = logging.getLogger("abep_forecasting")


def slugify(value: str) -> str:
    """Return a filesystem-safe label for generated outputs."""
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def output_path(filename: str, output_dir: Path) -> Path:
    """Build a path inside the output directory."""
    return output_dir / filename


def result_path(name: str, table_dir: Path, indicator_choice: str, suffix: str = "csv") -> Path:
    """Build a standardized path for structured result files."""
    # Prefixing every result with the indicator avoids accidental overwrites
    # when the notebook is rerun for both configured outcomes.
    indicator_slug = slugify(indicator_choice)
    return table_dir / f"{indicator_slug}_{slugify(name)}.{suffix}"


def save_table(
    df: pd.DataFrame,
    name: str,
    table_dir: Path,
    indicator_choice: str,
    index: bool = False,
) -> Path:
    """Save a DataFrame as CSV and return the generated path."""
    path = result_path(name, table_dir=table_dir, indicator_choice=indicator_choice, suffix="csv")
    df.to_csv(path, index=index)
    logger.info("Saved table: %s", path)
    return path


def _json_default(value):
    """Convert common scientific Python objects to JSON-compatible values."""
    # NumPy and pandas scalar/date objects are common in model metadata but are
    # not serializable by Python's standard json encoder without conversion.
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(data, name: str, table_dir: Path, indicator_choice: str) -> Path:
    """Save structured metadata or selected parameters as JSON."""
    path = result_path(name, table_dir=table_dir, indicator_choice=indicator_choice, suffix="json")
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=_json_default)
    logger.info("Saved JSON: %s", path)
    return path


def model_metrics_frame(model_name: str, result: dict) -> pd.DataFrame:
    """Create a one-row metrics table from a model result dictionary."""
    return pd.DataFrame([{
        "Model": model_name,
        "MAE": result["mae"],
        "RMSE": result["rmse"],
        "MAPE": result["mape"],
    }])
