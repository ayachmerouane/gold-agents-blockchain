"""Découpage chronologique train / validation / test (§4.2, étape `split_chrono`).

Les dates viennent de `config.yaml` (section `splits`). La purge de W jours en tête de
validation et de test sera ajoutée avec les fenêtres (phase 2).
"""

from __future__ import annotations

import pandas as pd

from gold_agents.config import SplitsConfig

SPLIT_NAMES = ("train", "val", "test")


def assign_split(dates: pd.Series | pd.Index, splits: SplitsConfig) -> pd.Series:
    """Nom de l'ensemble (train/val/test) de chaque date, ou `None` hors ensembles."""
    days = pd.Series(pd.DatetimeIndex(dates).tz_localize(None).date, index=dates)
    out = pd.Series(None, index=dates, dtype="object")
    out[days <= splits.train_end] = "train"
    out[(days >= splits.val_start) & (days <= splits.val_end)] = "val"
    out[days >= splits.test_start] = "test"
    return out
