"""Découpage chronologique train / validation / test (§4.2, étape `split_chrono`).

Les dates viennent de `config.yaml` (section `splits`). Un jour t devient un exemple
d'entraînement, de validation ou de test (`sample_mask`) seulement si :

- ses W derniers jours (la fenêtre du modèle) ont toutes leurs features ;
- ces W jours et le lendemain (la cible) sont dans le même ensemble que t : aucune
  fenêtre ni aucune cible ne chevauche deux ensembles ;
- pour la validation et le test, t n'est pas dans les `purge` premiers jours de son
  ensemble (par défaut W) : ces jours servent seulement d'historique.
"""

from __future__ import annotations

import numpy as np
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


def sample_mask(df: pd.DataFrame, window: int, purge: int | None = None) -> pd.Series:
    """Lignes utilisables comme exemples (voir le docstring du module).

    `df` doit avoir les colonnes `split`, `features_valid` et `target_valid`.
    """
    purge = window if purge is None else purge
    code = df["split"].map({name: i for i, name in enumerate(SPLIT_NAMES)}).astype("float64")
    feats = df["features_valid"].astype("float64")

    full_window = feats.rolling(window, min_periods=window).min() == 1
    same_split = (
        code.rolling(window, min_periods=window).min()
        == code.rolling(window, min_periods=window).max()
    )
    target_same_split = code.shift(-1) == code
    position = df.groupby("split", sort=False).cumcount()
    purged = (df["split"] != "train") & (position < purge)

    mask = (
        code.notna()
        & full_window
        & same_split
        & target_same_split
        & df["target_valid"].astype(bool)
        & ~purged
    )
    return mask.astype(bool)


def split_bounds(df: pd.DataFrame, mask: pd.Series) -> dict[str, dict[str, object]]:
    """Première et dernière date, nombre d'exemples de chaque ensemble."""
    out: dict[str, dict[str, object]] = {}
    for name in SPLIT_NAMES:
        sel = df.loc[mask & (df["split"] == name), "date"]
        out[name] = {
            "n_samples": len(sel),
            "first_date": sel.min().strftime("%Y-%m-%d") if len(sel) else None,
            "last_date": sel.max().strftime("%Y-%m-%d") if len(sel) else None,
        }
    return out


def check_splits(df: pd.DataFrame, mask: pd.Series, window: int) -> list[str]:
    """Règles anti-fuite du découpage ; liste vide si tout va bien (§10.3)."""
    problems: list[str] = []
    idx = np.flatnonzero(mask.to_numpy())
    split = df["split"].to_numpy()
    dates = df["date"].to_numpy()
    last: dict[str, np.datetime64] = {}
    first: dict[str, np.datetime64] = {}
    for name in SPLIT_NAMES:
        sel = idx[split[idx] == name]
        if len(sel) == 0:
            problems.append(f"ensemble {name} vide")
            continue
        first[name], last[name] = dates[sel].min(), dates[sel].max()
    for a, b in [("train", "val"), ("val", "test")]:
        if a in last and b in first and not last[a] < first[b]:
            problems.append(f"{a} et {b} se chevauchent")
    for i in idx:
        if i - window + 1 < 0 or len(set(split[i - window + 1 : i + 2])) != 1:
            problems.append(f"fenêtre ou cible à cheval sur deux ensembles ({dates[i]})")
            break
    return problems
