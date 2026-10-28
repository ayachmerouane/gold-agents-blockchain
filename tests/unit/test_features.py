"""Tests des features, de la cible, du découpage, du scaler et des fenêtres (§10.2)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from gold_agents.cli import app
from gold_agents.config import Config
from gold_agents.data import features as ft
from gold_agents.data.prepare import check_prepared, load_prepared, prepare
from gold_agents.data.splits import SPLIT_NAMES, check_splits, sample_mask
from gold_agents.data.windows import make_windows


def _s(values: list[float]) -> pd.Series:
    return pd.Series(values, dtype="float64")


# --- Indicateurs calculés à la main -------------------------------------------


def test_sma_and_ema_by_hand() -> None:
    x = _s([1, 2, 3, 4, 5])
    assert ft.sma(x, 3).tolist()[2:] == [2.0, 3.0, 4.0]
    assert ft.sma(x, 3).isna().sum() == 2
    # alpha = 2 / (3 + 1) = 0,5 : 1 -> 1,5 -> 2,25 ; défini à partir de 3 valeurs
    assert ft.ema(x[:3], 3).tolist()[2] == pytest.approx(2.25)
    assert ft.ema(x, 3).isna().sum() == 2


def test_rsi_by_hand() -> None:
    # variations +1 -1 +1 +1 ; n = 2 : graine (1+0)/2 et (0+1)/2 -> 50, puis 75, puis 87,5
    out = ft.rsi(_s([1, 2, 1, 2, 3]), 2)
    assert out.isna().sum() == 2
    assert out.tolist()[2:] == pytest.approx([50.0, 75.0, 87.5])


def test_rsi_extremes() -> None:
    assert ft.rsi(_s(list(range(1, 30))), 14).dropna().eq(100).all()
    assert ft.rsi(_s(list(range(30, 1, -1))), 14).dropna().eq(0).all()
    assert ft.rsi(_s([5.0] * 20), 14).dropna().eq(50).all()


def test_atr_by_hand() -> None:
    high, low, close = _s([3, 4, 8]), _s([1, 2, 6]), _s([2, 3, 7])
    # TR = 2 ; max(2, |4-2|, |2-2|) = 2 ; max(2, |8-3|, |6-3|) = 5
    assert ft.true_range(high, low, close).tolist() == [2, 2, 5]
    # graine (2+2)/2 = 2, puis (2*1 + 5)/2 = 3,5
    assert ft.atr(high, low, close, 2).tolist()[1:] == [2.0, 3.5]


def test_target_is_next_day_return(clean_series: pd.DataFrame, cfg: Config) -> None:
    df = ft.build_features(clean_series, cfg.features)
    r = np.log(df["close"]).diff()
    ok = df["target_valid"].to_numpy()
    np.testing.assert_allclose(df["target_return"][ok], r.shift(-1)[ok])
    assert (df["target_up"][ok] == (df["target_return"][ok] > 0)).all()
    assert not df["target_valid"].iloc[-1]
    # Le lendemain du trou n'est pas un rendement d'un jour : ni cible, ni feature.
    after_gap = df.index[~df["valid_return"]][1]
    assert not df.loc[after_gap - 1, "target_valid"]
    assert np.isnan(df.loc[after_gap, "ret_1d"])


def test_features_columns_and_warmup(clean_series: pd.DataFrame, cfg: Config) -> None:
    df = ft.add_features(clean_series, cfg.features)
    cols = ft.feature_columns(cfg.features)
    assert len(cols) == len(set(cols)) and set(cols) <= set(df.columns)
    assert "volume" not in cols
    first = df.index[df["features_valid"]][0]
    assert first == max(cfg.features.sma_periods + cfg.features.ema_periods) - 1
    assert df.loc[df["features_valid"], f"rsi_{cfg.features.rsi_period}"].between(0, 100).all()
    # Après l'échauffement, seul le trou laisse des features manquantes.
    assert (~df["features_valid"]).sum() < first + 1 + max(cfg.features.return_lags) + 25


# --- Découpage ----------------------------------------------------------------


def test_sample_mask_rules(clean_series: pd.DataFrame, cfg: Config) -> None:
    w = cfg.features.window
    df = ft.build_features(clean_series, cfg.features)
    mask = sample_mask(df, w)
    assert check_splits(df, mask, w) == []
    for name in ("val", "test"):
        part = df[df["split"] == name]
        first_sample = part.index[mask[part.index]][0]
        assert first_sample - part.index[0] == w  # purge de W jours en tête
    train_last = df.index[mask & (df["split"] == "train")][-1]
    assert df.loc[train_last + 1, "split"] == "train"  # sa cible reste dans le train


def test_check_splits_detects_straddling_window(clean_series: pd.DataFrame, cfg: Config) -> None:
    w = cfg.features.window
    df = ft.build_features(clean_series, cfg.features)
    mask = sample_mask(df, w, purge=0)
    first_val = df.index[df["split"] == "val"][0]
    mask.loc[first_val] = True  # fenêtre à cheval train/val
    assert any("à cheval" in p for p in check_splits(df, mask, w))


# --- Scaler et fenêtres -------------------------------------------------------


def test_robust_scaler_round_trip() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 100.0], "b": [5.0] * 5})
    scaler = ft.RobustScaler.fit(df, ["a", "b"])
    assert scaler.center == (3.0, 5.0)
    assert scaler.scale == (2.0, 1.0)  # IQR de a = 4 - 2 ; b constante -> 1
    out = scaler.transform(df)
    assert out["a"].tolist() == [-1.0, -0.5, 0.0, 0.5, 48.5]
    assert ft.RobustScaler.from_dict(scaler.to_dict()) == scaler
    with pytest.raises(ValueError):
        ft.RobustScaler.fit(pd.DataFrame({"a": [1.0, np.nan]}), ["a"])


def test_make_windows_shape_and_content(clean_series: pd.DataFrame, cfg: Config) -> None:
    w = cfg.features.window
    prepared = prepare(clean_series, cfg)
    df, cols = prepared.frame, prepared.columns
    win = make_windows(df, cols, w, df["sample"] & (df["split"] == "val"))
    assert win.x.shape == (len(win.y), w, len(cols))
    t = df.index[df["date"] == win.dates[0]][0]
    np.testing.assert_allclose(win.x[0], df.loc[t - w + 1 : t, cols].to_numpy(), rtol=1e-6)
    assert win.y[0] == pytest.approx(df.loc[t, "target_return"])


def test_make_windows_rejects_overflow(clean_series: pd.DataFrame, cfg: Config) -> None:
    prepared = prepare(clean_series, cfg)
    mask = pd.Series(False, index=prepared.frame.index)
    mask.iloc[0] = True
    with pytest.raises(ValueError):
        make_windows(prepared.frame, prepared.columns, 5, mask)


def test_prepare_is_consistent(clean_series: pd.DataFrame, cfg: Config) -> None:
    prepared = prepare(clean_series, cfg)
    assert check_prepared(prepared, cfg.features.window) == []
    counts = prepared.frame[prepared.frame["sample"]]["split"].value_counts()
    assert all(counts[name] > 0 for name in SPLIT_NAMES)


# --- Commande `prepare` -------------------------------------------------------


def test_cli_prepare(tmp_path: Path, config_path: Path, clean_series: pd.DataFrame) -> None:
    text = config_path.read_text(encoding="utf-8").replace(
        "data/processed", (tmp_path / "processed").as_posix()
    )
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(text, encoding="utf-8")
    runner = CliRunner()

    res = runner.invoke(app, ["prepare", "-c", str(cfg_path)])
    assert res.exit_code == 1 and "make clean" in res.output

    (tmp_path / "processed").mkdir()
    clean_series.to_parquet(tmp_path / "processed" / "xau_1d_clean.parquet", index=False)
    res = runner.invoke(app, ["prepare", "-c", str(cfg_path)])
    assert res.exit_code == 0, res.output
    assert "train" in res.output and "test" in res.output
    frame, scaler = load_prepared(tmp_path / "processed")
    assert list(scaler.columns) == [c for c in frame.columns if c in scaler.columns]
    assert (tmp_path / "processed" / "xau_1d_features.json").exists()
