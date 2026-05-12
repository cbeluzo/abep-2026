"""Regression tests for the refactored ABEP forecasting notebook.

These tests validate that the notebook still uses the local package, that core
helpers preserve their contracts, and that structured outputs remain
standardized.

Author:
    Carlos Beluzo - beluzo@ifsp.edu.br
"""

from __future__ import annotations

__author__ = "Carlos Beluzo"
__email__ = "beluzo@ifsp.edu.br"

import inspect
import math
import sys
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
NOTEBOOK_PATH = PROJECT_ROOT / "abep_2026_sarima_lstm_low_birth_and_prematurity.ipynb"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from abep_forecasting.comparison import build_final_model_comparison
from abep_forecasting.config import INDICATOR_CONFIGS, resolve_data_source
from abep_forecasting.data import load_indicator_data, prepare_monthly_split, validate_monthly_dataframe
from abep_forecasting.io_utils import result_path as build_result_path
from abep_forecasting.lstm import (
    apply_series_transform,
    create_sequences,
    invert_series_transform,
    prepare_lstm_training_data,
)
from abep_forecasting import lstm as lstm_module
from abep_forecasting.metrics import maerr, mape, rmse


def _load_notebook():
    return nbformat.read(NOTEBOOK_PATH, as_version=4)


@pytest.fixture(scope="session")
def notebook_context():
    indicator_choice = "Low Birth Weight"
    config = INDICATOR_CONFIGS[indicator_choice]
    data_source = resolve_data_source(config)
    df_selected = load_indicator_data(data_source, config)
    split_data = prepare_monthly_split(df_selected, config.value_col)
    table_dir = PROJECT_ROOT / "output" / "tables"

    return {
        "indicator_choice": indicator_choice,
        "config": config,
        "df_selected": df_selected,
        "split_data": split_data,
        "VALUE_COL": config.value_col,
        "result_path": lambda name, suffix="csv": build_result_path(
            name,
            table_dir=table_dir,
            indicator_choice=indicator_choice,
            suffix=suffix,
        ),
    }


def test_notebook_uses_requirements_file():
    notebook = _load_notebook()
    install_cells = [
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and "install -r requirements.txt" in cell.source
    ]

    assert install_cells, "The notebook should install dependencies from requirements.txt."


def test_notebook_uses_extracted_modules():
    notebook = _load_notebook()
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert "from abep_forecasting.lstm import" in code
    assert "from abep_forecasting.sarima import" in code
    assert "from abep_forecasting.plots import" in code
    assert "def run_lstm_experiment(" not in code
    assert "def grid_search_sarima(" not in code
    assert "def build_final_model_comparison(" not in code


def test_methodological_decisions_are_documented():
    notebook = _load_notebook()
    markdown = "\n".join(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "markdown"
    )

    assert "## Methodological Decisions" in markdown
    for topic in [
        "Indicator standardization",
        "Train/test split",
        "SARIMA specification",
        "LSTM preprocessing",
        "Evaluation metrics",
        "Final comparison",
        "Reproducibility",
    ]:
        assert topic in markdown


def test_data_loading_standardizes_columns(notebook_context):
    df_selected = notebook_context["df_selected"]

    assert len(df_selected) == 132
    assert notebook_context["VALUE_COL"] == "low_birth_weight_index"
    assert list(df_selected.columns) == [
        "reference_month",
        "year",
        "month",
        "total_births",
        "total_low_birth_weight",
        "low_birth_weight_index",
    ]
    assert df_selected["reference_month"].min() == pd.Timestamp("2013-01-01")
    assert df_selected["reference_month"].max() == pd.Timestamp("2023-12-01")


def test_validation_detects_complete_monthly_series(notebook_context):
    summary = validate_monthly_dataframe(
        notebook_context["df_selected"],
        notebook_context["indicator_choice"],
    )

    row = summary.iloc[0]
    assert row["rows"] == 132
    assert row["expected_months"] == 132
    assert row["observed_months"] == 132
    assert row["missing_months"] == 0
    assert row["extra_months"] == 0


def test_train_test_split_boundaries(notebook_context):
    split_data = notebook_context["split_data"]

    assert len(split_data.train) == 120
    assert len(split_data.test_2023) == 12
    assert split_data.train.index.min() == pd.Timestamp("2013-01-01")
    assert split_data.train.index.max() == pd.Timestamp("2022-12-01")
    assert split_data.test_2023.index.min() == pd.Timestamp("2023-01-01")
    assert split_data.test_2023.index.max() == pd.Timestamp("2023-12-01")


def test_metric_helpers_known_values():
    y_true = np.array([1.0, 2.0, 4.0])
    y_pred = np.array([1.0, 1.0, 7.0])

    assert maerr(y_true, y_pred) == pytest.approx(4 / 3)
    assert rmse(y_true, y_pred) == pytest.approx(math.sqrt(10 / 3))
    assert mape(y_true, y_pred) == pytest.approx((0 + 0.5 + 0.75) / 3 * 100)


def test_transform_round_trip():
    values = np.array([0.0, 1.5, 10.0])
    transformed = apply_series_transform(values, transform="log1p")
    restored = invert_series_transform(transformed, transform="log1p")

    np.testing.assert_allclose(restored, values)

    with pytest.raises(ValueError, match="Unsupported transform"):
        apply_series_transform(values, transform="invalid")


def test_create_sequences_shape_and_values():
    values = np.arange(1, 7, dtype="float64").reshape(-1, 1)
    X, y = create_sequences(values, lookback=3)

    assert X.shape == (3, 3, 1)
    assert y.tolist() == [4.0, 5.0, 6.0]
    np.testing.assert_array_equal(X[0, :, 0], np.array([1.0, 2.0, 3.0]))


def test_lstm_training_preparation_contract(notebook_context):
    prepared = prepare_lstm_training_data(
        notebook_context["split_data"],
        lookback=12,
        transform="none",
    )

    assert len(prepared.train_series) == 120
    assert len(prepared.test_series) == 12
    assert prepared.train_scaled.shape == (120, 1)
    assert prepared.X_train.shape == (108, 12, 1)
    assert prepared.y_train.shape == (108,)
    assert prepared.transform == "none"


def test_unified_lstm_pipeline_wrappers_delegate_to_run_lstm_experiment():
    wrapper_names = [
        "run_lstm_pipeline_split",
        "run_lstm_pipeline_split_regularized",
        "run_lstm_pipeline_split_regularized_transformed",
        "evaluate_lstm_multi_horizon",
    ]

    for name in wrapper_names:
        source = inspect.getsource(getattr(lstm_module, name))
        assert "run_lstm_experiment(" in source

    multi_horizon_source = inspect.getsource(lstm_module.evaluate_lstm_multi_horizon)
    assert 'mode="multi_horizon"' in multi_horizon_source


def test_final_model_comparison_builder_with_synthetic_metrics():
    context = {
        "mae_val": 2.0,
        "rmse_val": 2.5,
        "mape_val": 10.0,
        "best_params": {"order": (1, 0, 1), "seasonal_order": (0, 1, 1, 12)},
        "lstm_result": {"mae": 1.5, "rmse": 2.0, "mape": 8.0},
        "best_lstm_final": {"mae": 1.2, "rmse": 1.8, "mape": 7.0},
        "best_lstm_reg_final": {"mae": 1.3, "rmse": 1.9, "mape": 7.5},
        "best_transform": "log1p",
        "best_lstm_transform_final": {"mae": 1.1, "rmse": 1.7, "mape": 6.5},
        "lstm_multi_result": {
            "horizon_metrics": pd.DataFrame([
                {"Horizon": 1, "N": 12, "MAE": 1.4, "RMSE": 2.1, "MAPE": 8.5}
            ])
        },
    }

    comparison = build_final_model_comparison(context)

    assert not comparison.empty
    assert comparison.iloc[0]["Variant"] == "Best transformation"
    assert comparison.iloc[0]["Rank by RMSE"] == 1
    assert set(comparison["Variant"]) == {
        "Selected grid model",
        "Baseline",
        "Best grid model",
        "Regularized",
        "Best transformation",
        "Multi-horizon H1",
    }


def test_result_path_uses_indicator_prefix(notebook_context):
    path = notebook_context["result_path"]("Unit Test Sample")

    assert path.parent.name == "tables"
    assert path.name == "low_birth_weight_unit_test_sample.csv"
