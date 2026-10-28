"""Données des modèles de prévision : fenêtres mises à l'échelle pour chaque ensemble.

Les réseaux (LSTM, GRU) lisent la fenêtre complète `(W, F)` du jour t ; les modèles
tabulaires (Ridge, LightGBM) ne lisent que sa dernière ligne, les features du jour t, qui
contiennent déjà les rendements passés sur 1, 5, 10 et 20 jours.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gold_agents.data.features import RobustScaler
from gold_agents.data.splits import SPLIT_NAMES
from gold_agents.data.windows import Windows, make_windows


@dataclass(frozen=True)
class SplitData:
    windows: Windows
    rows: np.ndarray  # position de chaque exemple dans le fichier de features

    @property
    def x(self) -> np.ndarray:
        return self.windows.x

    @property
    def y(self) -> np.ndarray:
        return self.windows.y

    @property
    def tabular(self) -> np.ndarray:
        """Features du jour t seulement, `(N, F)`."""
        return self.windows.x[:, -1, :]


@dataclass(frozen=True)
class ForecastData:
    frame: pd.DataFrame  # fichier de features (non mis à l'échelle), une ligne par jour
    columns: list[str]
    window: int
    splits: dict[str, SplitData]

    def __getitem__(self, name: str) -> SplitData:
        return self.splits[name]


def build_forecast_data(frame: pd.DataFrame, scaler: RobustScaler, window: int) -> ForecastData:
    """Met les features à l'échelle (scaler ajusté sur le train) et découpe les fenêtres."""
    frame = frame.reset_index(drop=True)
    columns = list(scaler.columns)
    scaled = scaler.transform(frame)
    splits = {}
    for name in SPLIT_NAMES:
        mask = frame["sample"] & (frame["split"] == name)
        splits[name] = SplitData(
            make_windows(scaled, columns, window, mask), np.flatnonzero(mask.to_numpy())
        )
    return ForecastData(frame, columns, window, splits)
