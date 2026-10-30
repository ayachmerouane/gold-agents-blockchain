"""Commandes `tune` puis `freeze` de bout en bout (après `train --quick`)."""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

pytest.importorskip("torch")
pytest.importorskip("lightgbm")

from gold_agents.cli import app  # noqa: E402
from gold_agents.config import Config  # noqa: E402
from gold_agents.data.prepare import prepare, write_prepared  # noqa: E402


def test_tune_then_freeze(tmp_path: Path, cfg: Config, clean_series: pd.DataFrame) -> None:
    raw = cfg.model_dump(mode="json")
    raw["data"]["processed_dir"] = str(tmp_path / "processed")
    raw["models"]["seeds"] = [0]
    raw["tuning"]["grid"] = {
        "theta_buy": [0.0, 0.0005],
        "theta_sell": [0.0, 0.0005],
        "min_confidence": [None, 0.5],
        "stop_loss": [0.05],
        "max_drawdown": [0.15, 0.2],
        "models": ["lstm", "ensemble", "lightgbm"],
    }
    local = Config.model_validate(raw)
    clean_path = tmp_path / "clean.parquet"
    clean_series.to_parquet(clean_path)
    write_prepared(prepare(clean_series, local), local, clean_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    runner = CliRunner()
    forecast, models = tmp_path / "forecast", tmp_path / "models"
    common = ["-c", str(config_path), "--models-dir", str(models)]
    res = runner.invoke(
        app,
        [
            "train",
            *common,
            "--quick",
            "--no-mlflow",
            "--models",
            "b4,lstm,gru",
            "--out",
            str(forecast),
        ],
    )
    assert res.exit_code == 0, res.output

    out = tmp_path / "tuning"
    tune = ["tune", *common, "--forecast-dir", str(forecast), "--out", str(out)]
    res = runner.invoke(app, [*tune, "--quick"])
    assert res.exit_code == 0, res.output
    assert "Walk-forward" in res.output and "identique" in res.output
    forecasts = pd.read_parquet(out / "forecasts_walk_forward.parquet")
    assert set(forecasts["fold"]) == {1, 2, 3, 4, 5}
    assert set(forecasts["model"]) == {"lstm", "gru", "ensemble", "lightgbm"}
    # Jamais une date du test.
    assert pd.to_datetime(forecasts["date"]).max().date() < local.splits.test_start
    walk = pd.read_csv(out / "walk_forward.csv")
    assert list(walk["fold"]) == [2, 3, 4, 5]
    assert (out / "tuning.png").stat().st_size > 10_000

    # Deuxième passage : prévisions relues depuis le cache, mêmes résultats.
    res = runner.invoke(app, [*tune, "--quick"])
    assert res.exit_code == 0 and "relues depuis" in res.output
    chosen = json.loads((out / "chosen.json").read_text(encoding="utf-8"))

    ledgers = tmp_path / "ledger"
    freeze = [
        "freeze",
        "-c",
        str(config_path),
        "--tuning-dir",
        str(out),
        "--ledger-dir",
        str(ledgers),
    ]
    res = runner.invoke(app, freeze)
    assert res.exit_code == 1 and "--quick" in res.output  # un essai rapide ne se fige pas
    chosen["quick"] = False
    (out / "chosen.json").write_text(json.dumps(chosen), encoding="utf-8")

    settings = chosen["chosen"]
    raw["strategy"].update(
        {k: settings[k] for k in ("model", "theta_buy", "theta_sell", "min_confidence")}
    )
    raw["strategy"]["theta_buy"] = settings["theta_buy"] + 0.01  # pas encore recopié
    raw["risk"].update({k: settings[k] for k in ("stop_loss", "max_drawdown", "cooldown_days")})
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    res = runner.invoke(app, freeze)
    assert res.exit_code == 1 and "theta_buy" in res.output

    raw["strategy"]["theta_buy"] = settings["theta_buy"]
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    key, public = tmp_path / "k.key", tmp_path / "public_key.pem"
    runner.invoke(app, ["keygen", "--key", str(key), "--public-key", str(public)])
    res = runner.invoke(app, [*freeze, "--key", str(key)])
    assert res.exit_code == 0, res.output
    res = runner.invoke(
        app, ["verify", str(ledgers / "reglages_figes.jsonl"), "--public-key", str(public)]
    )
    assert res.exit_code == 0 and "(1 blocs, v2, signatures vérifiées)" in res.output
    block = json.loads(
        (ledgers / "reglages_figes.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    genesis = block["events"][0]
    assert genesis["kind"] == "FROZEN_SETTINGS" and genesis["settings"] == settings
