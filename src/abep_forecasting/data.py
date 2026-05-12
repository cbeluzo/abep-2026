"""Data loading, validation, and temporal split helpers.

This module standardizes the raw monthly indicator files into English column
names, validates the expected 2013-2023 monthly coverage, and builds the
training/test split reused by all models.

Author:
    Carlos Beluzo - beluzo@ifsp.edu.br
"""

from __future__ import annotations

__author__ = "Carlos Beluzo"
__email__ = "beluzo@ifsp.edu.br"

from dataclasses import dataclass

import pandas as pd
from IPython.display import Markdown, display
from tabulate import tabulate

from .config import DATE_COLUMN, START_YEAR, TEST_YEAR, TRAIN_END_YEAR, IndicatorConfig


STANDARD_COLUMN_NAMES = {
    # Raw files use Portuguese names. The analysis code uses English names for
    # consistency with the notebook text, plots, and exported tables.
    "ano": "year",
    "mes": "month",
    "total_nascimentos": "total_births",
    "total_baixo_peso": "total_low_birth_weight",
    "total_prematuros": "total_preterm_births",
}


def load_indicator_data(source, indicator_config: IndicatorConfig, date_column: str = DATE_COLUMN) -> pd.DataFrame:
    """Load, sort, and standardize the selected indicator dataset."""
    df = pd.read_csv(source, parse_dates=["mes_ref"])

    if indicator_config.raw_value_col not in df.columns:
        raise ValueError(f"Expected column '{indicator_config.raw_value_col}' was not found in the source data.")

    rename_map = {
        **STANDARD_COLUMN_NAMES,
        "mes_ref": date_column,
        indicator_config.raw_value_col: indicator_config.value_col,
    }
    df = df.rename(columns=rename_map)
    # Only the configured target column is rescaled; count columns remain in
    # their original units.
    df[indicator_config.value_col] = df[indicator_config.value_col].astype("float64") * indicator_config.scale_factor

    if indicator_config.decimals is not None:
        df[indicator_config.value_col] = df[indicator_config.value_col].round(indicator_config.decimals)

    return df.sort_values(date_column).reset_index(drop=True)


def _colorize(value: str, color: str) -> str:
    """Format a value with inline HTML color for Jupyter Markdown output."""
    return f"<span style='color:{color}'>{value}</span>"


def validate_monthly_dataframe(
    df: pd.DataFrame,
    name: str,
    date_col: str = DATE_COLUMN,
    expected_range: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Validate date type, period coverage, and missing/extra monthly rows."""
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' was not found.")

    if expected_range is None:
        expected_range = pd.date_range(f"{START_YEAR}-01-01", f"{TEST_YEAR}-12-01", freq="MS")

    # Compare observed months against the complete expected monthly range so
    # missing or duplicated periods are visible before modeling.
    rows = len(df)
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    observed = pd.DatetimeIndex(df[date_col].sort_values().unique())
    missing = expected_range.difference(observed)
    extra = observed.difference(expected_range)

    display(Markdown(f"**=== Validation: {name} ===**"))

    table = [
        ["Rows", rows],
        ["Minimum Date", min_date.date() if pd.notnull(min_date) else "N/A"],
        ["Maximum Date", max_date.date() if pd.notnull(max_date) else "N/A"],
        ["Column Types", "<br>".join(f"{column}: {dtype}" for column, dtype in df.dtypes.items())],
        ["Expected Months", len(expected_range)],
        ["Observed Months", len(observed)],
        ["Missing Months", len(missing)],
        ["Extra Months", len(extra)],
    ]
    display(Markdown(tabulate(table, tablefmt="github", headers=["Item", "Value"])))

    if missing.size:
        missing_text = ", ".join(str(month.date()) for month in missing)
        display(Markdown(f"*Missing months:* {_colorize(missing_text, 'red')}"))
    if extra.size:
        extra_text = ", ".join(str(month.date()) for month in extra)
        display(Markdown(f"*Extra months:* {_colorize(extra_text, 'orange')}"))

    return pd.DataFrame([{
        "name": name,
        "rows": rows,
        "min_date": min_date,
        "max_date": max_date,
        "expected_months": len(expected_range),
        "observed_months": len(observed),
        "missing_months": len(missing),
        "extra_months": len(extra),
    }])


validate_monthly_csv = validate_monthly_dataframe


@dataclass
class SplitData:
    """Container for the full, train, and 2023 holdout series."""

    full: pd.DataFrame
    train: pd.Series
    test_2023: pd.Series


def prepare_monthly_split(
    df: pd.DataFrame,
    value_col: str,
    date_col: str = DATE_COLUMN,
    train_end_year: int = TRAIN_END_YEAR,
    test_year: int = TEST_YEAR,
) -> SplitData:
    """Sort the monthly data and create the train/test split used by all models."""
    df_sorted = df.sort_values(date_col).set_index(date_col)
    # The holdout period is kept completely outside model fitting so SARIMA and
    # LSTM metrics are directly comparable on the same 2023 months.
    train = df_sorted[df_sorted.index.year <= train_end_year][value_col]
    test = df_sorted[df_sorted.index.year == test_year][value_col]
    return SplitData(full=df_sorted, train=train, test_2023=test)


def show_split_info(name: str, data: SplitData, logger, train_end_year: int = TRAIN_END_YEAR, test_year: int = TEST_YEAR) -> None:
    """Log the number of observations in each modeling split."""
    logger.info("%s preparation summary", name)
    logger.info("Total observations: %s", len(data.full))
    logger.info("Training set up to %s: %s months", train_end_year, len(data.train))
    logger.info("Test set %s: %s months", test_year, len(data.test_2023))
