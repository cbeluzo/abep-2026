# ABEP 2026 - SARIMA and LSTM Forecasting

Author: Carlos Beluzo - beluzo@ifsp.edu.br

This project contains a reproducible forecasting pipeline for monthly low birth weight and prematurity indicators. The main analysis is in the notebook `abep_2026_sarima_lstm_low_birth_and_prematurity.ipynb`, with reusable Python functions extracted to the local package `src/abep_forecasting`.

## Objective

The notebook compares statistical and neural-network approaches for monthly public-health time series:

- SARIMA for interpretable seasonal time-series modeling.
- LSTM for recurrent neural-network forecasting.
- Rolling multi-horizon diagnostics for 1- to 4-month forecasts.
- Structured CSV/JSON outputs for validation, model metrics, forecasts, and final comparison tables.

## Project Structure

```text
.
├── abep_2026_sarima_lstm_low_birth_and_prematurity.ipynb
├── requirements.txt
├── README.md
├── data/
│   ├── vw_indice_baixo_peso_mensal.csv
│   └── vw_indice_prematuridade_mensal.csv
├── output/
│   ├── tables/
│   └── *.pdf
├── src/
│   └── abep_forecasting/
│       ├── __init__.py
│       ├── comparison.py
│       ├── config.py
│       ├── data.py
│       ├── io_utils.py
│       ├── lstm.py
│       ├── metrics.py
│       ├── plots.py
│       └── sarima.py
└── tests/
    └── test_notebook_helpers.py
```

## Main Files

- `abep_2026_sarima_lstm_low_birth_and_prematurity.ipynb`: main execution notebook.
- `requirements.txt`: Python dependencies used by the notebook and tests.
- `src/abep_forecasting/config.py`: indicator metadata, constants, and data-source resolution.
- `src/abep_forecasting/data.py`: data loading, column standardization, validation, and train/test split.
- `src/abep_forecasting/metrics.py`: MAE, MSE, RMSE, MAPE, and shared metric alignment.
- `src/abep_forecasting/sarima.py`: SARIMA diagnostics, grid search, fitting, forecasting, and reporting.
- `src/abep_forecasting/lstm.py`: unified LSTM training, tuning, regularization, transformation, and multi-horizon evaluation.
- `src/abep_forecasting/plots.py`: decomposition plots, rolling forecast plots, and heatmaps.
- `src/abep_forecasting/io_utils.py`: standardized output paths and CSV/JSON saving.
- `src/abep_forecasting/comparison.py`: final ranked model comparison table.
- `tests/test_notebook_helpers.py`: regression tests for the extracted helpers and notebook integration.

## Requirements

Use Python 3.10+ when possible. The project was organized to run in a notebook environment such as Jupyter, JupyterLab, or Google Colab.

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

If you are running locally and do not already have a notebook server installed:

```bash
python3 -m pip install notebook
```

## How to Use

1. Open the notebook:

```bash
jupyter notebook abep_2026_sarima_lstm_low_birth_and_prematurity.ipynb
```

2. In the first configuration cell, choose the indicator:

```python
indicator_choice = "Low Birth Weight"
```

Available options:

- `"Low Birth Weight"`
- `"Prematurity"`

3. Run the notebook cells from top to bottom.

4. Review generated outputs in:

```text
output/
output/tables/
```

The notebook first uses local CSV files from `data/`. If a local file is missing, it falls back to the configured remote source URL in `config.py`.

## Outputs

Structured tables are saved with an indicator prefix to avoid overwriting results from different runs. Examples:

- `output/tables/low_birth_weight_validation_summary.csv`
- `output/tables/low_birth_weight_sarima_metrics_2023.csv`
- `output/tables/low_birth_weight_lstm_baseline_metrics_2023.csv`
- `output/tables/low_birth_weight_lstm_multi_horizon_metrics_2023.csv`
- `output/tables/low_birth_weight_final_model_comparison_2023.csv`

Figures are saved under `output/` as PDF files when the corresponding notebook cell produces an export.

## Methodological Summary

- Data are monthly from January 2013 through December 2023.
- The training period uses 2013-2022.
- The holdout test period uses 2023.
- SARIMA models are selected with stationarity diagnostics, seasonal differencing suggestions, information criteria, residual checks, and holdout metrics.
- LSTM models fit scalers only on the training period to avoid data leakage.
- LSTM variants share one unified pipeline and differ by explicit parameters such as lookback, units, dropout, L2 penalty, early stopping, and optional transformation.
- Final model ranking uses 2023 holdout metrics, primarily RMSE, with MAE and MAPE included for interpretation.

## Running Tests

Run the test suite after changing code:

```bash
python3 -m pytest -q
```

The current tests validate:

- dependency installation cell in the notebook;
- methodological documentation in the notebook;
- data loading and standardized column names;
- monthly coverage validation;
- train/test split boundaries;
- metric helper behavior;
- LSTM sequence and preprocessing contracts;
- LSTM wrapper delegation to the unified pipeline;
- final model comparison table construction;
- standardized output file naming.

## Development Notes

- Keep reusable logic inside `src/abep_forecasting/`.
- Keep the notebook focused on narrative, configuration, execution, and result display.
- Add or update tests in `tests/` when changing helper behavior.
- Preserve the 2013-2022 training and 2023 holdout convention unless the methodology is intentionally changed.
- Use `save_table` and `save_json` for structured outputs so generated files remain consistently named.

## Authorship

All Python source files include author metadata:

```python
__author__ = "Carlos Beluzo"
__email__ = "beluzo@ifsp.edu.br"
```

Author: Carlos Beluzo - beluzo@ifsp.edu.br
