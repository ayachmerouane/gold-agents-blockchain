"""Commande `evaluate` (phase 8) de bout en bout, sur la série synthétique.

`train --quick`, un faux bloc de réglages figés, puis : refus si la config a bougé,
répétition sur la validation, run unique sur le test (sans ancrage), refus d'un second run.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

pytest.importorskip("torch")
pytest.importorskip("lightgbm")

from gold_agents.blockchain.chain import Chain  # noqa: E402
from gold_agents.blockchain.storage import read_jsonl, write_jsonl  # noqa: E402
from gold_agents.cli import app  # noqa: E402
from gold_agents.config import Config  # noqa: E402
from gold_agents.data.prepare import FEATURES_FILE, prepare, write_prepared  # noqa: E402
from gold_agents.data.update import sha256_file  # noqa: E402
from gold_agents.evaluation.final import current_settings  # noqa: E402

STRATEGIES = [
    "buy_and_hold",
    "sma_20_50",
    "agents_lstm",
    "agents_gru",
    "agents_ensemble",
    "agents_ensemble_mc",
    "agents_lightgbm",
    "aleatoire",
    "oracle",
]


def _frozen(cfg: Config, data_sha: str, path: Path) -> str:
    events = [
        {
            "type": "GENESIS",
            "kind": "FROZEN_SETTINGS",
            "settings": current_settings(cfg),
            "config": json.loads(cfg.model_dump_json()),
            "data_sha256": data_sha,
        }
    ]
    chain = Chain.start(events, version=1)
    write_jsonl(chain, path)
    return chain.head.hash


def test_evaluate_once(tmp_path: Path, cfg: Config, clean_series: pd.DataFrame) -> None:
    raw = cfg.model_dump(mode="json")
    raw["data"]["processed_dir"] = str(tmp_path / "processed")
    raw["models"]["seeds"] = [0]
    raw["models"]["max_epochs"] = 3
    raw["models"]["mc_samples"] = 5
    raw["tuning"]["seeds"] = [0]
    raw["tuning"]["random_runs"] = 10
    # Règle d'achat relative (tentative 2) : quantile 70 % des prévisions passées.
    raw["strategy"].update(buy_rule="quantile", theta_buy=0.7, min_confidence=None)
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

    key, public = tmp_path / "k.key", tmp_path / "public_key.pem"
    runner.invoke(app, ["keygen", "--key", str(key), "--public-key", str(public)])
    frozen = tmp_path / "reglages_figes.jsonl"
    data_sha = sha256_file(local.data.processed_dir / FEATURES_FILE)
    head = _frozen(local, data_sha, frozen)
    out, ledgers = tmp_path / "final", tmp_path / "ledger"
    evaluate = [
        "evaluate",
        *common,
        "--forecast-dir",
        str(forecast),
        "--tuning-dir",
        str(tmp_path / "pas_de_tune"),
        "--frozen",
        str(frozen),
        "--frozen-head",
        head,
        "--out",
        str(out),
        "--ledger-dir",
        str(ledgers),
        "--key",
        str(key),
        "--public-key",
        str(public),
        "--no-anchor",
        "--moves",
        str(tmp_path / "moves.json"),
    ]

    # Un réglage modifié après le gel : refus, et rien n'est écrit.
    moved = {**raw, "risk": {**raw["risk"], "stop_loss": 0.05}}
    moved_path = tmp_path / "moved.yaml"
    moved_path.write_text(yaml.safe_dump(moved), encoding="utf-8")
    res = runner.invoke(app, [*evaluate, "-c", str(moved_path)])
    assert res.exit_code == 1 and "stop_loss" in res.output
    assert not ledgers.exists()
    # Un autre bloc genèse que celui attendu : refus.
    res = runner.invoke(app, [*evaluate, "--frozen-head", "0" * 64])
    assert res.exit_code == 1 and "bloc genèse" in res.output
    # --quick est réservé à la répétition.
    res = runner.invoke(app, [*evaluate, "--quick"])
    assert res.exit_code == 1 and not ledgers.exists()

    # Répétition générale sur la validation : le test n'est pas lu.
    rehearsal = tmp_path / "rehearsal"
    res = runner.invoke(
        app,
        [
            *evaluate,
            "--rehearsal",
            "--quick",
            "--out",
            str(rehearsal),
            "--ledger-dir",
            str(tmp_path / "ledger_val"),
        ],
    )
    assert res.exit_code == 0, res.output
    metrics = json.loads((rehearsal / "metrics.json").read_text(encoding="utf-8"))["metrics"]
    assert metrics[0]["end"] <= "2021-12-31"
    fc = pd.read_csv(rehearsal / "forecasts.csv")
    assert fc["date"].str[:10].max() <= "2021-12-31"

    # Le run sur le test.
    res = runner.invoke(app, evaluate)
    assert res.exit_code == 0, res.output
    assert "Tableau comparatif" in res.output and "Sensibilité aux frais" in res.output
    assert "Alertes" in res.output
    result = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    metrics = result["metrics"]
    assert [m["strategy"] for m in metrics] == STRATEGIES
    assert len({(m["start"], m["end"]) for m in metrics}) == 1
    assert metrics[0]["start"] >= "2022-01-01"
    assert all(m["ledger"]["valid"] and m["ledger"]["signatures"] == "vérifiées" for m in metrics)
    # Le hasard fait autant de trades que le système retenu ; l'oracle bat tout le monde.
    by_name = {m["strategy"]: m for m in metrics}
    assert by_name["aleatoire"]["n_trades"] == by_name["agents_ensemble_mc"]["n_trades"]
    assert all(m["final_value"] <= by_name["oracle"]["final_value"] for m in metrics)

    # Les prévisions viennent de modèles appris avant le test, sur tous les jours du test.
    fc = pd.read_csv(out / "forecasts.csv")
    assert set(fc["model"]) == {"lstm", "gru", "ensemble", "lightgbm"}
    assert fc["date"].str[:10].min() >= "2022-01-01"
    costs = pd.read_csv(out / "costs.csv")
    assert sorted(costs["costs_factor"].unique()) == [0, 1, 2, 5]
    assert set(costs["strategy"]) == set(STRATEGIES)
    # Frais x0 : pas de coûts ; frais x1 : mêmes chiffres que le run principal.
    assert (costs[costs["costs_factor"] == 0]["total_costs"] == 0).all()
    one = costs[costs["costs_factor"] == 1].set_index("strategy")["final_value"]
    for m in metrics:
        assert one[m["strategy"]] == pytest.approx(m["final_value"])
    for name in ("final_table.md", "final_table.csv", "forecast_metrics.csv"):
        assert (out / name).stat().st_size > 100
    assert (out / "final_test.png").stat().st_size > 10_000
    assert (out / "robustesse_test.png").stat().st_size > 10_000

    # Plan et résultats : signés, liés au plan, avec l'empreinte de chaque fichier.
    plan = read_jsonl(ledgers / "plan.jsonl")
    plan_event = plan.blocks[0].events[0]
    assert plan_event["kind"] == "TEST_PLAN" and plan_event["attempt"] == 1
    assert plan_event["frozen_settings"]["head"] == head
    results = read_jsonl(ledgers / "resultats.jsonl")
    genesis = results.blocks[0].events[0]
    assert genesis["plan_head"] == plan.head.hash
    assert "final_table.md" in genesis["files_sha256"]
    files = sorted(ledgers.glob("*.jsonl"))
    res = runner.invoke(app, ["verify", *map(str, files), "--public-key", str(public)])
    assert res.exit_code == 0 and res.output.count("signatures vérifiées") == len(files) == 11

    # Un seul run : le second est refusé, sauf avec une raison inscrite au registre.
    res = runner.invoke(app, evaluate)
    assert res.exit_code == 1 and "déjà été lancé" in res.output
    res = runner.invoke(app, [*evaluate, "--again", "essai du test"])
    assert res.exit_code == 0, res.output
    second = read_jsonl(ledgers / "plan.jsonl").blocks[0].events[0]
    assert second["attempt"] == 2 and second["reason_again"] == "essai du test"
    assert second["previous_attempts"][0]["head"] == plan.head.hash
    # Rien n'est effacé : la tentative 1 est archivée, registres et résultats.
    kept = tmp_path / "ledger_tentative1"
    assert sorted(p.name for p in kept.glob("*.jsonl")) == [p.name for p in files]
    assert (tmp_path / "final_tentative1" / "metrics.json").exists()
    moves = json.loads((tmp_path / "moves.json").read_text(encoding="utf-8"))
    assert len(moves) == 11 and all("ledger_tentative1" in m["to"] for m in moves)
    # Les agents ont bien tradé avec la règle relative, et le registre le dit.
    assert by_name["agents_ensemble"]["n_trades"] > 0
    blocks = (ledgers / "agents_ensemble.jsonl").read_text(encoding="utf-8")
    assert "quantile 70%" in blocks
