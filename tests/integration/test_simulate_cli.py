"""Commande `simulate` de bout en bout : `train --quick` puis les agents sur la validation."""

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


def test_train_then_simulate(tmp_path: Path, cfg: Config, clean_series: pd.DataFrame) -> None:
    raw = cfg.model_dump(mode="json")
    raw["data"]["processed_dir"] = str(tmp_path / "processed")
    raw["models"]["seeds"] = [0, 1]
    raw["models"]["mc_samples"] = 10
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

    out, ledgers = tmp_path / "agents", tmp_path / "ledger"
    res = runner.invoke(
        app,
        [
            "simulate",
            *common,
            "--forecast-dir",
            str(forecast),
            "--out",
            str(out),
            "--ledger-dir",
            str(ledgers),
        ],
    )
    assert res.exit_code == 0, res.output
    assert "Agents + LSTM/GRU, confiance" in res.output and "vs B&H" in res.output
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    names = [m["strategy"] for m in metrics]
    assert names == [
        "buy_and_hold",
        "sma_20_50",
        "agents_lstm",
        "agents_gru",
        "agents_ensemble",
        "agents_ensemble_mc",
        "agents_lightgbm",
    ]
    assert all(m["ledger"]["valid"] for m in metrics)
    # Même période que la validation des modèles, pour toutes les stratégies.
    assert len({(m["start"], m["end"]) for m in metrics}) == 1
    assert metrics[0]["start"] >= "2019-01-01" and metrics[0]["end"] <= "2021-12-31"
    assert (out / "agents_val.png").stat().st_size > 10_000

    decisions = pd.read_csv(out / "agents_ensemble_mc_decisions.csv")
    assert decisions["sigma_hat"].notna().all() and (decisions["sigma_hat"] > 0).all()
    assert set(decisions["rule"]) <= {"signal", "stop_loss", "max_drawdown", "rejet"}
    # Registres non signés (pas de clé dans le test) : sans la clé publique du dépôt.
    files = [str(p) for p in sorted((ledgers / "agents_val").glob("*.jsonl"))]
    res = runner.invoke(app, ["verify", *files, "--public-key", str(tmp_path / "aucune.pem")])
    assert res.exit_code == 0 and res.output.count("chaîne valide") == 7
