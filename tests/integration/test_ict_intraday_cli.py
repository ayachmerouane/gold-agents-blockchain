"""Commande `ict-intraday` de bout en bout : CSV MT4 synthétique (heure ny+7), petite grille,
en barres de 1 h (.csv) et de 15 min (.csv.gz)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from gold_agents.cli import app


def _quarter_hours(hourly: pd.DataFrame) -> pd.DataFrame:
    """Chaque barre d'une heure découpée en 4 barres de 15 min (chemin linéaire)."""
    o, c = hourly["open"].to_numpy(), hourly["close"].to_numpy()
    sp = (hourly["high"] - hourly[["open", "close"]].max(axis=1)).to_numpy() / 2
    rows = []
    for j in range(4):
        a, b = o + (c - o) * j / 4, o + (c - o) * (j + 1) / 4
        rows.append(
            pd.DataFrame(
                {
                    "ny": hourly["ny"] + pd.Timedelta(minutes=15 * j),
                    "open": a,
                    "high": np.maximum(a, b) + sp,
                    "low": np.minimum(a, b) - sp,
                    "close": b,
                }
            )
        )
    return pd.concat(rows).sort_values("ny").reset_index(drop=True)


@pytest.mark.parametrize(
    ("timeframe", "minutes", "name"),
    [("1h", 60, "XAU_1h_data.csv"), ("15m", 15, "XAU_15m_data.csv.gz")],
)
def test_ict_intraday_walk_forward(
    tmp_path: Path,
    config_path: Path,
    hourly_series: pd.DataFrame,
    timeframe: str,
    minutes: int,
    name: str,
) -> None:
    bars = hourly_series if minutes == 60 else _quarter_hours(hourly_series)
    server = (bars["ny"].dt.tz_localize(None) + pd.Timedelta(hours=7)).dt.strftime("%Y.%m.%d %H:%M")
    csv = tmp_path / name
    out_csv = bars[["open", "high", "low", "close"]].assign(Date=server, Volume=1)
    out_csv[["Date", "open", "high", "low", "close", "Volume"]].to_csv(
        csv, sep=";", index=False, header=["Date", "Open", "High", "Low", "Close", "Volume"]
    )
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["ict_intraday"]["timeframes"] = {
        timeframe: {"path": csv.as_posix(), "bar_minutes": minutes, "swing": [1]}
    }
    raw["ict_intraday"]["grid"] = {
        "entry": ["limit_fvg", "stop_pdh"],
        "target": ["prev_day_high"],
        "min_reward_risk": [0.5],
    }
    raw["tuning"]["folds"] = [
        ["2019-01-07", "2019-02-28"],
        ["2019-03-01", "2019-04-30"],
        ["2019-05-01", "2019-06-30"],
        ["2019-07-01", "2019-08-25"],
    ]
    raw["tuning"]["random_runs"] = 10
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    out = tmp_path / "res"
    args = ["ict-intraday", "-c", str(cfg_path), "--out", str(out), "-t", timeframe]
    res = CliRunner().invoke(app, args)
    assert res.exit_code == 0, res.output
    assert "test non lu" in res.output and "ny+7" in res.output
    out = out / timeframe
    wf = pd.read_csv(out / "walk_forward.csv")
    assert list(wf["fold"]) == [2, 3, 4]
    assert len(pd.read_csv(out / "grid.csv")) == 2 * 4
    rand = pd.read_csv(out / "random_references.csv")
    assert rand["beats_random"].between(0, 1).all()
    report = (out / "ict_intraday_report.md").read_text(encoding="utf-8")
    assert report.startswith("# Agent ICT")
    assert ("15 min" in report) == (minutes == 15)


def test_ict_intraday_unknown_timeframe(config_path: Path) -> None:
    res = CliRunner().invoke(app, ["ict-intraday", "-c", str(config_path), "-t", "2m"])
    assert res.exit_code == 1
