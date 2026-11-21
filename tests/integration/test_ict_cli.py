"""Commande `ict` de bout en bout sur la série synthétique (petite grille)."""

import json
from pathlib import Path

import pandas as pd
import yaml
from typer.testing import CliRunner

from gold_agents.cli import app


def test_ict_walk_forward(tmp_path: Path, config_path: Path, clean_series: pd.DataFrame) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    clean_series.to_parquet(processed / "xau_1d_clean.parquet")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["data"]["processed_dir"] = processed.as_posix()
    raw["ict"]["grid"] = {
        "entry": ["limit_fvg", "stop_pdh"],
        "swing": [1],
        "sweep_lookback": [10],
        "discount": [1.0],
        "min_reward_risk": [0.5],
    }
    raw["tuning"]["random_runs"] = 10
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    out = tmp_path / "ict"
    res = CliRunner().invoke(app, ["ict", "-c", str(cfg_path), "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert "test non lu" in res.output
    wf = pd.read_csv(out / "walk_forward.csv")
    assert list(wf["fold"]) == [2, 3, 4, 5]
    assert pd.to_datetime(wf["end"]).max() < pd.Timestamp("2022-01-01")
    grid = pd.read_csv(out / "grid.csv")
    assert len(grid) == 2 * 5
    choice = json.loads((out / "final_choice.json").read_text(encoding="utf-8"))
    assert choice["entry"] in {"limit_fvg", "stop_pdh"}
    assert (out / "ict_report.md").read_text(encoding="utf-8").startswith("# Stratégie ICT")
