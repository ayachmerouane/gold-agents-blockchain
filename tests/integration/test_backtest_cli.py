"""Commande `backtest` de bout en bout sur la série synthétique, puis `verify`."""

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from gold_agents.cli import app
from gold_agents.config import Config
from gold_agents.data.prepare import prepare, write_prepared


def test_backtest_and_verify(
    tmp_path: Path, config_path: Path, cfg: Config, clean_series: pd.DataFrame
) -> None:
    processed = tmp_path / "processed"
    text = config_path.read_text(encoding="utf-8").replace("data/processed", processed.as_posix())
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(text, encoding="utf-8")
    local = cfg.model_copy(
        update={"data": cfg.data.model_copy(update={"processed_dir": processed})}
    )
    clean_path = tmp_path / "clean.parquet"
    clean_series.to_parquet(clean_path)
    write_prepared(prepare(clean_series, local), local, clean_path)

    out, ledgers = tmp_path / "results", tmp_path / "ledger"
    key, public = tmp_path / "keys" / "k.key", tmp_path / "public_key.pem"
    runner = CliRunner()
    res = runner.invoke(app, ["keygen", "--key", str(key), "--public-key", str(public)])
    assert res.exit_code == 0 and key.exists() and public.exists(), res.output
    res = runner.invoke(app, ["keygen", "--key", str(key), "--public-key", str(public)])
    assert res.exit_code == 1  # jamais d'écrasement silencieux de la clé
    res = runner.invoke(
        app,
        ["backtest", "-c", str(cfg_path), "--out", str(out), "--ledger-dir", str(ledgers)]
        + ["--key", str(key)],
    )
    assert res.exit_code == 0, res.output
    assert "NON DÉTECTÉ" not in res.output and res.output.count("détecté") == 5
    assert (out / "backtest_test.png").stat().st_size > 10_000
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert [m["strategy"] for m in metrics] == ["buy_and_hold", "sma_20_50"]
    assert all(m["ledger"]["valid"] for m in metrics)

    # Buy & Hold : achat unique, au lendemain du premier jour de test.
    bh = pd.read_csv(out / "buy_and_hold_trades.csv")
    assert len(bh) == 1 and bh["side"].iloc[0] == "BUY"

    files = [str(p) for p in sorted(ledgers.glob("*.jsonl"))]
    res = runner.invoke(app, ["verify", *files, "--public-key", str(public)])
    assert res.exit_code == 0 and res.output.count("signatures vérifiées") == 2, res.output
    other = tmp_path / "autre.pem"
    runner.invoke(app, ["keygen", "--key", str(tmp_path / "o.key"), "--public-key", str(other)])
    res = runner.invoke(app, ["verify", *files, "--public-key", str(other)])
    assert res.exit_code == 1 and res.output.count("signature invalide") == 2

    path = ledgers / "buy_and_hold.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[5] = lines[5].replace('"action":"HOLD"', '"action":"SELL"', 1)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    res = runner.invoke(app, ["verify", str(path), "--public-key", str(public)])
    assert res.exit_code == 1 and "INVALIDE au bloc 5 : événements modifiés" in res.output
