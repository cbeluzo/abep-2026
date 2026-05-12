"""Central configuration for indicator metadata and modeling constants.

This module keeps dataset locations, public labels, date columns, and shared
time-series constants in one place so the notebook and helper modules do not
duplicate configuration logic.

Author:
    Carlos Beluzo - beluzo@ifsp.edu.br
"""

from __future__ import annotations

__author__ = "Carlos Beluzo"
__email__ = "beluzo@ifsp.edu.br"

from dataclasses import dataclass
from pathlib import Path


SEED = 42
START_YEAR = 2013
TRAIN_END_YEAR = 2022
TEST_YEAR = 2023
FORECAST_YEAR = 2024
SEASONAL_PERIOD = 12
DATE_COLUMN = "reference_month"


@dataclass(frozen=True)
class IndicatorConfig:
    """Configuration required to load and label one modeling target."""

    source_url: str
    local_file: str
    raw_value_col: str
    value_col: str
    target_label: str
    y_axis_label: str
    scale_factor: float = 1.0
    decimals: int | None = None


INDICATOR_CONFIGS = {
    # Low birth weight is stored as a percentage in the source data. The
    # scale_factor converts it to the index convention used in the analysis.
    "Low Birth Weight": IndicatorConfig(
        source_url="https://raw.githubusercontent.com/fjmeneguini/TCC/refs/heads/main/data/processed/series_temporais/vw_indice_baixo_peso_mensal.csv",
        local_file="data/vw_indice_baixo_peso_mensal.csv",
        raw_value_col="indice_baixo_peso_pct",
        value_col="low_birth_weight_index",
        target_label="Low Birth Weight Index",
        y_axis_label="Index (per 1,000 live births)",
        scale_factor=10.0,
        decimals=1,
    ),
    # Prematurity already uses the percentage scale expected by the notebook.
    "Prematurity": IndicatorConfig(
        source_url="https://raw.githubusercontent.com/fjmeneguini/TCC/refs/heads/main/data/processed/series_temporais/vw_indice_prematuridade_mensal.csv",
        local_file="data/vw_indice_prematuridade_mensal.csv",
        raw_value_col="indice_prematuridade_pct",
        value_col="prematurity_index",
        target_label="Prematurity Index",
        y_axis_label="Index (% of live births)",
        scale_factor=1.0,
        decimals=1,
    ),
}


def resolve_data_source(config: IndicatorConfig):
    """Prefer a local CSV when present, otherwise fall back to the configured URL."""
    local_data_path = Path(config.local_file)
    return local_data_path if local_data_path.exists() else config.source_url
