"""Tests du nettoyage : détection et comblement des trous, OHLC, colonnes de contrôle."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from gold_agents.cli import app
from gold_agents.config import Config, load_config
from gold_agents.data import clean as cleaning
from gold_agents.data import sources
from gold_agents.data.load import normalize, read_update_csv

GAP = pd.bdate_range("2023-03-06", "2023-03-31", tz="UTC")  # 20 jours ouvrés absents
HOLIDAY = pd.Timestamp("2023-05-01", tz="UTC")  # 1 jour : un férié, pas un trou


@pytest.fixture(scope="module")
def cfg() -> Config:
    return load_config(Path(__file__).resolve().parents[2] / "config" / "config.yaml")


def _truth(start: str = "2022-06-01", end: str = "2023-08-31") -> pd.DataFrame:
    """Série « complète » (7 jours sur 7, comme une API en UTC)."""
    dates = pd.date_range(start, end, tz="UTC")
    rng = np.random.default_rng(1)
    c = 1800 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
    return pd.DataFrame({"date": dates, "open": c, "high": c * 1.005, "low": c * 0.995, "close": c})


def _kaggle(truth: pd.DataFrame) -> pd.DataFrame:
    days = truth[truth["date"].dt.day_of_week < 5]
    days = days[~days["date"].isin(GAP) & (days["date"] != HOLIDAY)]
    return normalize(days, "kaggle")


def _fetcher(truth: pd.DataFrame, scale: float = 1.0005):
    """Faux Twelve Data : même instrument, à 0,05 % près, barres de week-end comprises."""

    def fetch(start: date, end: date) -> pd.DataFrame:
        sel = truth[(truth["date"].dt.date >= start) & (truth["date"].dt.date <= end)]
        prices = sel[["open", "high", "low", "close"]] * scale
        return normalize(sel[["date"]].join(prices), "twelvedata")

    return fetch


def test_find_gaps_ignores_holidays() -> None:
    gaps = cleaning.find_gaps(_kaggle(_truth()))
    assert gaps == [cleaning.Gap(date(2023, 3, 3), date(2023, 4, 3), len(GAP))]


def test_fetch_gap_bars_fills_only_missing_business_days() -> None:
    truth = _truth()
    bars, fills = cleaning.fetch_gap_bars(_kaggle(truth), _fetcher(truth))
    assert list(bars["date"]) == list(GAP)
    assert (bars["source"] == "twelvedata").all()
    assert fills[0].n_filled == len(GAP) and fills[0].report.ok


def test_fetch_gap_bars_rejects_other_instrument() -> None:
    truth = _truth()
    with pytest.raises(cleaning.GapFillError, match="incohérentes"):
        cleaning.fetch_gap_bars(_kaggle(truth), _fetcher(truth, scale=1.5))
    bars, _ = cleaning.fetch_gap_bars(_kaggle(truth), _fetcher(truth, scale=1.5), force=True)
    assert len(bars) == len(GAP)


def test_clean_with_gapfill(cfg: Config) -> None:
    truth = _truth()
    kaggle = _kaggle(truth)
    bars, _ = cleaning.fetch_gap_bars(kaggle, _fetcher(truth))
    out = cleaning.clean(kaggle, cfg.splits, bars)
    assert len(out) == len(kaggle) + len(GAP)
    assert int(out["filled"].sum()) == len(GAP)
    assert (out.loc[out["filled"], "source"] == "twelvedata").all()
    assert int((~out["valid_return"]).sum()) == 1  # seulement le premier jour
    assert cleaning.check_clean(out) == []
    assert set(out["split"]) == {"test"}


def test_clean_without_gapfill_marks_return_across_gap(cfg: Config) -> None:
    out = cleaning.clean(_kaggle(_truth()), cfg.splits)
    invalid = out.loc[~out["valid_return"], "date"].dt.strftime("%Y-%m-%d").tolist()
    assert invalid == ["2022-06-01", "2023-04-03"]
    assert not out["filled"].any()
    assert cleaning.check_clean(out) == []


def test_gapfill_never_overrides_kaggle_or_extends_series(cfg: Config) -> None:
    truth = _truth()
    kaggle = _kaggle(truth)
    other = _fetcher(truth, scale=2.0)(date(2022, 1, 1), date(2024, 1, 1))
    out = cleaning.clean(kaggle, cfg.splits, other)
    kept = out[~out["filled"]].reset_index(drop=True)
    pd.testing.assert_series_equal(kept["close"], kaggle["close"], check_names=False)
    assert out["date"].min() == kaggle["date"].min()
    assert out["date"].max() == kaggle["date"].max()
    assert (out["date"].dt.day_of_week < 5).all()


def test_fix_ohlc() -> None:
    df = pd.DataFrame(
        {"open": [10.0, 10.0], "high": [11.0, 9.0], "low": [9.0, 9.5], "close": [10.5, 9.8]}
    )
    out = cleaning.fix_ohlc(df)
    assert out["ohlc_fixed"].tolist() == [False, True]
    assert out.loc[1, ["high", "low"]].tolist() == [10.0, 9.5]


def test_check_clean_reports_problems(cfg: Config) -> None:
    out = cleaning.clean(_kaggle(_truth()), cfg.splits)
    out.loc[5, "valid_return"] = True
    out.loc[5, "missing_bdays_before"] = 10
    assert cleaning.check_clean(out) == ["trou non marqué"]


def _write_mt4(path: Path, df: pd.DataFrame) -> Path:
    lines = ["Date;Open;High;Low;Close;Volume"]
    lines += [
        f"{r.date.strftime('%Y.%m.%d 00:00')};{r.open};{r.high};{r.low};{r.close};100"
        for r in df.itertuples()
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_cli_fill_gaps_then_clean(
    tmp_path: Path, config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    truth = _truth()
    raw = _write_mt4(tmp_path / "XAU_1d_data.csv", _kaggle(truth))
    text = config_path.read_text(encoding="utf-8")
    for key, name in [
        ("data/raw/XAU_1d_data.csv", raw.name),
        ("data/raw/XAU_1d_gapfill.csv", "XAU_1d_gapfill.csv"),
        ("data/raw/CHECKSUMS.txt", "CHECKSUMS.txt"),
        ("data/processed", "processed"),
    ]:
        text = text.replace(key, (tmp_path / name).as_posix())
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(text, encoding="utf-8")
    runner = CliRunner()

    res = runner.invoke(app, ["clean", "-c", str(cfg_path)])
    assert res.exit_code == 0, res.output
    assert "trous non comblés" in res.output

    fake = _fetcher(truth)
    monkeypatch.setattr(sources, "fetch_twelvedata", lambda start, end: fake(start, end))
    res = runner.invoke(app, ["fill-gaps", "-c", str(cfg_path)])
    assert res.exit_code == 0, res.output
    assert f"{len(GAP)} jour(s) récupéré(s)" in res.output
    assert len(read_update_csv(tmp_path / "XAU_1d_gapfill.csv")) == len(GAP)
    assert "XAU_1d_gapfill.csv" in (tmp_path / "CHECKSUMS.txt").read_text(encoding="utf-8")

    res = runner.invoke(app, ["clean", "-c", str(cfg_path)])
    assert res.exit_code == 0, res.output
    out = pd.read_parquet(tmp_path / "processed" / "xau_1d_clean.parquet")
    assert int(out["filled"].sum()) == len(GAP)
    assert (tmp_path / "processed" / "xau_1d_clean.json").exists()


def test_cli_fill_gaps_refuses_incoherent_source(
    tmp_path: Path, config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    truth = _truth()
    raw = _write_mt4(tmp_path / "XAU_1d_data.csv", _kaggle(truth))
    text = config_path.read_text(encoding="utf-8").replace(
        "data/raw/XAU_1d_data.csv", raw.as_posix()
    )
    text = text.replace("data/raw/XAU_1d_gapfill.csv", (tmp_path / "gap.csv").as_posix())
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(text, encoding="utf-8")
    fake = _fetcher(truth, scale=1.5)
    monkeypatch.setattr(sources, "fetch_twelvedata", lambda start, end: fake(start, end))
    res = CliRunner().invoke(app, ["fill-gaps", "-c", str(cfg_path)])
    assert res.exit_code == 1
    assert not (tmp_path / "gap.csv").exists()
