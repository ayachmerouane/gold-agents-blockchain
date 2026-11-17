"""Commande `ict-intraday` de bout en bout : CSV MT4 synthétique (heure ny+7), petite grille."""

from pathlib import Path

import pandas as pd
import yaml
from typer.testing import CliRunner

from gold_agents.cli import app


def test_ict_intraday_walk_forward(
    tmp_path: Path, config_path: Path, hourly_series: pd.DataFrame
) -> None:
    server = (hourly_series["ny"].dt.tz_localize(None) + pd.Timedelta(hours=7)).dt.strftime(
        "%Y.%m.%d %H:%M"
    )
    csv = tmp_path / "XAU_1h_data.csv"
    out_csv = hourly_series[["open", "high", "low", "close"]].assign(Date=server, Volume=1)
    out_csv[["Date", "open", "high", "low", "close", "Volume"]].to_csv(
        csv, sep=";", index=False, header=["Date", "Open", "High", "Low", "Close", "Volume"]
    )
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["ict_intraday"]["path"] = csv.as_posix()
    raw["ict_intraday"]["grid"] = {
        "entry": ["limit_fvg", "stop_pdh"],
        "swing": [1],
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
    res = CliRunner().invoke(app, ["ict-intraday", "-c", str(cfg_path), "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert "test non lu" in res.output and "ny+7" in res.output
    wf = pd.read_csv(out / "walk_forward.csv")
    assert list(wf["fold"]) == [2, 3, 4]
    assert len(pd.read_csv(out / "grid.csv")) == 2 * 4
    rand = pd.read_csv(out / "random_references.csv")
    assert rand["beats_random"].between(0, 1).all()
    assert (out / "ict_intraday_report.md").read_text(encoding="utf-8").startswith("# Agent ICT")
