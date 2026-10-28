"""Fenêtres glissantes pour les modèles séquentiels (§4.2, étape `make_windows`).

L'exemple du jour t est la matrice des features des jours t-W+1 … t (W lignes, F colonnes)
et sa cible est `target_return` du jour t, c'est-à-dire le rendement de t à t+1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Windows:
    x: np.ndarray  # (N, W, F)
    y: np.ndarray  # (N,)
    dates: pd.DatetimeIndex  # date t de chaque exemple (jour de la décision)


def make_windows(
    df: pd.DataFrame,
    columns: list[str],
    window: int,
    mask: pd.Series,
    target: str = "target_return",
) -> Windows:
    """Fenêtres des lignes où `mask` est vrai (voir `splits.sample_mask`)."""
    values = df[columns].to_numpy(dtype="float32")
    idx = np.flatnonzero(mask.to_numpy())
    if len(idx) and idx.min() < window - 1:
        raise ValueError("une fenêtre déborde avant le début de la série")
    offsets = np.arange(-window + 1, 1)
    x = values[idx[:, None] + offsets[None, :]] if len(idx) else np.empty((0, window, len(columns)))
    if np.isnan(x).any():
        raise ValueError("fenêtre avec des features manquantes")
    y = df[target].to_numpy(dtype="float32")[idx]
    return Windows(x.astype("float32"), y, pd.DatetimeIndex(df["date"].to_numpy()[idx]))
