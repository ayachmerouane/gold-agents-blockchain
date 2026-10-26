"""Tests de l'EDA sur une série synthétique dont on connaît les défauts plantés."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from gold_agents.cli import app
from gold_agents.config import Config, load_config
from gold_agents.data.load import normalize
from gold_agents.eda import analysis
from gold_agents.eda.report import analyze, summary_lines, write_report

SPIKE_DAY = "2010-06-15"  # +15 % puis retour : erreur de cotation plantée
BAD_BAR_DAY = "2012-03-07"  # high < close plantée
REMOVED = pd.bdate_range("2015-08-03", "2015-08-07")  # trou de 5 jours ouvrés


def _synthetic(start: str = "2004-06-11", end: str = "2025-12-31", seed: int = 0) -> pd.DataFrame:
    """Marche aléatoire à volatilité GARCH(1,1) : rendements imprévisibles, |r| autocorrélés."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end, tz="UTC")
    n = len(dates)
    r, var = np.empty(n), 1e-4
    for t in range(n):
        r[t] = rng.normal(0, np.sqrt(var))
        var = 2e-6 + 0.08 * r[t] ** 2 + 0.9 * var
    close = 400 * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.001, n))
    spread = np.abs(rng.normal(0, 0.004, n)) * close
    df = pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": rng.integers(1000, 5000, n).astype(float),
        }
    )
    spike = df["date"] == pd.Timestamp(SPIKE_DAY, tz="UTC")
    df.loc[spike, ["close", "high"]] = df.loc[spike, "close"] * 1.15
    # high entre open et close : sous max(open, close) mais au-dessus de low
    bad = df["date"] == pd.Timestamp(BAD_BAR_DAY, tz="UTC")
    df.loc[bad, "high"] = (df.loc[bad, "open"] + df.loc[bad, "close"]) / 2
    df = df[~df["date"].isin(REMOVED.tz_localize("UTC"))]
    return normalize(df, "kaggle")


@pytest.fixture(scope="module")
def kaggle() -> pd.DataFrame:
    return _synthetic()


@pytest.fixture(scope="module")
def cfg() -> Config:
    return load_config(Path(__file__).resolve().parents[2] / "config" / "config.yaml")


def _update(kaggle: pd.DataFrame, jump: float = 0.0) -> pd.DataFrame:
    last = kaggle["close"].iloc[-1] * (1 + jump)
    dates = pd.bdate_range("2026-01-02", periods=20, tz="UTC")
    c = last * np.ones(len(dates))
    return normalize(
        pd.DataFrame({"date": dates, "open": c, "high": c + 1, "low": c - 1, "close": c}),
        "twelvedata",
    )


def test_calendar_finds_planted_gap(kaggle: pd.DataFrame) -> None:
    cal = analysis.calendar_report(kaggle)
    assert cal["weekend_rows"] == 0
    assert cal["missing_business_days"] == len(REMOVED)
    assert cal["largest_gaps"][0] == {
        "after": "2015-07-31",
        "before": "2015-08-10",
        "missing_business_days": 5,
    }
    assert cal["missing_business_days_per_year"] == {2015: 5}


def test_calendar_counts_weekend_bars() -> None:
    df = _synthetic("2024-01-01", "2024-01-31")
    sunday = df.iloc[[0]].assign(date=pd.Timestamp("2024-01-07", tz="UTC"))
    cal = analysis.calendar_report(pd.concat([df, sunday]).sort_values("date"))
    assert cal["weekend_rows"] == 1
    assert cal["rows_per_weekday"]["dim"] == 1


def test_ohlc_flags_planted_bar(kaggle: pd.DataFrame) -> None:
    ohlc = analysis.ohlc_report(kaggle)
    assert ohlc["high_below_open_or_close"] == 1
    assert ohlc["violation_examples"]["high_below_open_or_close"] == [BAD_BAR_DAY]
    assert ohlc["high_below_low"] == ohlc["nan_price"] == ohlc["non_positive_price"] == 0


def test_outliers_detect_spike_reversal(kaggle: pd.DataFrame, cfg: Config) -> None:
    out = analysis.outliers_report(kaggle, cfg.splits)
    assert SPIKE_DAY in out["spike_reversals"]
    assert out["top"][0]["date"] in {SPIKE_DAY, "2010-06-16"}
    assert out["top"][0]["split"] == "train"


def test_returns_show_fat_tails(kaggle: pd.DataFrame, cfg: Config) -> None:
    rep = analysis.returns_report(analysis.train_part(analysis.log_returns(kaggle), cfg.splits))
    assert rep["excess_kurtosis"] > 1
    assert rep["jarque_bera"]["p_value"] < 0.01
    assert rep["max"]["date"] == SPIKE_DAY


def test_thresholds_ignore_future(kaggle: pd.DataFrame, cfg: Config) -> None:
    """Changer la validation et le test ne doit pas bouger les seuils estimés sur le train."""
    r = analysis.log_returns(kaggle)
    later = analysis.assign_split(r.index, cfg.splits).to_numpy() != "train"
    shocked = r.copy()
    shocked[later] *= 5
    base = analysis.volatility_regimes(r, cfg.splits, 20)["thresholds_from_train"]
    assert analysis.volatility_regimes(shocked, cfg.splits, 20)["thresholds_from_train"] == base


def test_rolling_volatility_is_causal(kaggle: pd.DataFrame) -> None:
    r = analysis.log_returns(kaggle)
    vol = analysis.rolling_volatility(r, 20)
    cut = 3000
    truncated = analysis.rolling_volatility(r.iloc[:cut], 20)
    pd.testing.assert_series_equal(vol.iloc[:cut], truncated)


def test_dependence_and_stationarity(kaggle: pd.DataFrame, cfg: Config) -> None:
    r = analysis.train_part(analysis.log_returns(kaggle), cfg.splits)
    acf = analysis.autocorrelation_report(r)
    assert acf["abs_returns"]["ljung_box"][10]["p_value"] < 0.01
    assert abs(acf["returns"]["acf"][0]) < 0.1
    logp = analysis.train_part(np.log(kaggle.set_index("date")["close"]), cfg.splits)
    st = analysis.stationarity_report(logp, r)
    assert st["returns"]["verdict"] == "stationnaire"
    assert st["log_price"]["verdict"] == "non stationnaire"


def test_split_preview_matches_config(kaggle: pd.DataFrame, cfg: Config) -> None:
    sp = analysis.split_preview(kaggle, cfg.splits, cfg.features.window)
    assert sp["splits"]["train"]["last_date"] == "2018-12-31"
    assert sp["splits"]["val"]["first_date"] == "2019-01-01"
    assert sp["splits"]["test"]["first_date"] == "2022-01-03"
    val = sp["splits"]["val"]
    assert val["n_usable_after_purge"] == val["n_rows"] - cfg.features.window
    assert sp["unassigned_rows"] == 0
    assert sum(sp["share"].values()) == pytest.approx(1)


def test_leakage_checks_pass_and_catch_overlap(kaggle: pd.DataFrame, cfg: Config) -> None:
    w = cfg.features.window
    assert all(analysis.leakage_checks(kaggle, _update(kaggle), cfg.splits, w).values())
    overlapping = _update(kaggle).assign(date=lambda d: d["date"] - pd.Timedelta(days=60))
    checks = analysis.leakage_checks(kaggle, overlapping, cfg.splits, w)
    assert checks["update_rows_strictly_after_kaggle"] is False


def test_source_junction_reports_level_jump(kaggle: pd.DataFrame, cfg: Config) -> None:
    smooth = analysis.source_junction(kaggle, _update(kaggle), cfg.splits)
    assert abs(smooth["junction_robust_z"]) < 1
    assert smooth["business_days_between"] == 1  # 1er janvier
    jumped = analysis.source_junction(kaggle, _update(kaggle, jump=0.05), cfg.splits)
    assert jumped["junction_robust_z"] > 5
    assert analysis.source_junction(kaggle, kaggle.iloc[0:0], cfg.splits) == {"update_rows": 0}


def test_write_report(tmp_path: Path, kaggle: pd.DataFrame, cfg: Config) -> None:
    result = write_report(kaggle, _update(kaggle), cfg, tmp_path)
    assert result.report_path.read_text(encoding="utf-8").startswith("# Rapport")
    assert all(p.exists() and p.stat().st_size > 1000 for p in result.figure_paths)
    assert result.summary_path.exists()
    text = "\n".join(summary_lines(analyze(kaggle, kaggle.iloc[0:0], cfg)))
    assert "aucun fichier" in text


def _write_mt4(path: Path, df: pd.DataFrame) -> Path:
    lines = ["Date;Open;High;Low;Close;Volume"]
    lines += [
        f"{r.date.strftime('%Y.%m.%d 00:00')};{r.open};{r.high};{r.low};{r.close};{r.volume}"
        for r in df.itertuples()
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_cli_eda(tmp_path: Path, kaggle: pd.DataFrame, config_path: Path) -> None:
    raw = _write_mt4(tmp_path / "XAU_1d_data.csv", kaggle)
    text = config_path.read_text(encoding="utf-8")
    text = text.replace("data/raw/XAU_1d_data.csv", raw.as_posix())
    text = text.replace("data/raw/XAU_1d_update.csv", (tmp_path / "absent.csv").as_posix())
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(text, encoding="utf-8")
    res = CliRunner().invoke(app, ["eda", "-c", str(cfg_path), "--out", str(tmp_path / "eda")])
    assert res.exit_code == 0, res.output
    assert "RÉSUMÉ EDA" in res.output
    assert (tmp_path / "eda" / "rapport_eda.md").exists()


def test_cli_eda_without_data(tmp_path: Path, config_path: Path) -> None:
    text = config_path.read_text(encoding="utf-8").replace(
        "data/raw/XAU_1d_data.csv", (tmp_path / "absent.csv").as_posix()
    )
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(text, encoding="utf-8")
    res = CliRunner().invoke(app, ["eda", "-c", str(cfg_path)])
    assert res.exit_code == 1
