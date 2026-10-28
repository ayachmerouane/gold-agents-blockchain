"""Tests anti-fuite du futur (§10.3) : aucune feature du jour t ne dépend d'un jour > t."""

import numpy as np
import pandas as pd
import pytest

from gold_agents.config import Config
from gold_agents.data import features as ft
from gold_agents.data.prepare import prepare
from gold_agents.data.splits import check_splits, sample_mask

CUTS = [120, 1000, 3000, 3731, 4600]  # dont la frontière train/val


def _features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return ft.add_features(df, cfg.features)[ft.feature_columns(cfg.features)]


@pytest.mark.parametrize("cut", CUTS)
def test_truncation(clean_series: pd.DataFrame, cfg: Config, cut: int) -> None:
    """Features calculées sur toute la série = features calculées sur la série coupée à t."""
    full = _features(clean_series, cfg).iloc[: cut + 1]
    short = _features(clean_series.iloc[: cut + 1], cfg)
    pd.testing.assert_frame_equal(full, short, check_exact=False, rtol=1e-12)


@pytest.mark.parametrize("cut", CUTS)
def test_future_replaced_by_noise(clean_series: pd.DataFrame, cfg: Config, cut: int) -> None:
    """Remplacer tous les prix après t par du bruit ne change aucune feature jusqu'à t."""
    rng = np.random.default_rng(cut)
    noisy = clean_series.copy()
    after = noisy.index > cut
    base = rng.uniform(100, 5000, after.sum())
    noisy.loc[after, "open"] = base
    noisy.loc[after, "close"] = base * rng.uniform(0.9, 1.1, after.sum())
    noisy.loc[after, "high"] = noisy.loc[after, ["open", "close"]].max(axis=1) * 1.01
    noisy.loc[after, "low"] = noisy.loc[after, ["open", "close"]].min(axis=1) * 0.99
    before = _features(clean_series, cfg).iloc[: cut + 1]
    pd.testing.assert_frame_equal(before, _features(noisy, cfg).iloc[: cut + 1])


def test_target_is_the_only_future_column(clean_series: pd.DataFrame, cfg: Config) -> None:
    """Changer le dernier prix ne modifie que la cible de l'avant-dernier jour."""
    shifted = clean_series.copy()
    shifted.loc[shifted.index[-1], ["open", "high", "low", "close"]] *= 1.5
    a = ft.build_features(clean_series, cfg.features).iloc[:-1]
    b = ft.build_features(shifted, cfg.features).iloc[:-1]
    diff = [c for c in a.columns if not a[c].equals(b[c])]
    assert "target_return" in diff and set(diff) <= {"target_return", "target_up"}


def test_scaler_uses_train_only(clean_series: pd.DataFrame, cfg: Config) -> None:
    """Les statistiques du scaler ne bougent pas si l'on remplace val et test par du bruit."""
    noisy = clean_series.copy()
    after = noisy["split"] != "train"
    rng = np.random.default_rng(0)
    noisy.loc[after, ["open", "high", "low", "close"]] *= rng.uniform(0.5, 2.0, (after.sum(), 1))
    noisy.loc[after, "high"] = noisy.loc[after, ["open", "high", "low", "close"]].max(axis=1)
    noisy.loc[after, "low"] = noisy.loc[after, ["open", "high", "low", "close"]].min(axis=1)
    assert prepare(clean_series, cfg).scaler == prepare(noisy, cfg).scaler


def test_splits_are_ordered_with_gap(clean_series: pd.DataFrame, cfg: Config) -> None:
    """max(train) + W < min(val), idem val/test ; aucune fenêtre ni cible à cheval."""
    w = cfg.features.window
    df = ft.build_features(clean_series, cfg.features)
    mask = sample_mask(df, w)
    assert check_splits(df, mask, w) == []
    pos = {name: df.index[mask & (df["split"] == name)] for name in ("train", "val", "test")}
    assert pos["train"].max() + w < pos["val"].min()
    assert pos["val"].max() + w < pos["test"].min()
