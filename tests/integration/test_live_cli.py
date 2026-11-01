"""Simulation en direct de bout en bout, sur la série synthétique (sans API ni Sepolia).

Règle relative gelée, `live-init` (modèles figés, plan signé, premier jour), puis `live`
après trois nouveaux jours : deux rattrapés, un à temps.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

pytest.importorskip("torch")

from gold_agents.blockchain.chain import Chain  # noqa: E402
from gold_agents.blockchain.signing import load_public_key  # noqa: E402
from gold_agents.blockchain.storage import read_jsonl, write_jsonl  # noqa: E402
from gold_agents.cli import app  # noqa: E402
from gold_agents.config import Config  # noqa: E402
from gold_agents.data.features import build_features, feature_columns  # noqa: E402
from gold_agents.evaluation.final import current_settings  # noqa: E402
from gold_agents.live import session as ls  # noqa: E402
from gold_agents.models.train import TrainConfig  # noqa: E402


def _frozen(cfg: Config, path: Path) -> None:
    """Faux gel de la règle (comme `freeze`) : la config doit lui correspondre."""
    events = [{"type": "GENESIS", "kind": "FROZEN_SETTINGS", "settings": current_settings(cfg)}]
    write_jsonl(Chain.start(events, version=1), path)


def test_live_init_then_daily(
    tmp_path: Path, cfg: Config, clean_series: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = cfg.model_dump(mode="json")
    raw["data"]["processed_dir"] = str(tmp_path / "processed")
    raw["models"]["mc_samples"] = 3
    raw["strategy"].update(
        buy_rule="quantile", theta_buy=0.6, min_confidence=None, quantile_window=40
    )
    raw["strategy"]["quantile_min_days"] = 20
    local = Config.model_validate(raw)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    frozen = tmp_path / "reglages_figes.jsonl"
    _frozen(local, frozen)

    # Hyperparamètres « de la phase 4 » : petits réseaux pour aller vite.
    models = tmp_path / "models"
    n = len(feature_columns(local.features))
    for kind in ("lstm", "gru"):
        (models / kind).mkdir(parents=True)
        meta = {"net": {"n_features": n, "hidden": 8, "layers": 1, "dropout": 0.2}}
        meta["train"] = TrainConfig(batch_size=256).to_dict()
        (models / kind / "model.json").write_text(json.dumps(meta), encoding="utf-8")

    full = build_features(clean_series, local.features).reset_index(drop=True)
    shown = {"n": len(full) - 3}
    monkeypatch.setattr(ls, "live_frame", lambda _cfg: full.iloc[: shown["n"]].copy())

    runner = CliRunner()
    key, pub = tmp_path / "k.key", tmp_path / "k.pem"
    assert (
        runner.invoke(app, ["keygen", "--key", str(key), "--public-key", str(pub)]).exit_code == 0
    )
    ledgers, live_models = tmp_path / "live", tmp_path / "live_models"
    common = [
        "-c", str(config_path), "--no-update", "--no-anchor",
        "--live-models", str(live_models), "--ledger-dir", str(ledgers),
        "--key", str(key), "--public-key", str(pub),
    ]  # fmt: skip
    init = [*common, "--models-dir", str(models), "--quick", "--frozen", str(frozen)]
    # Une config qui n'est plus celle du gel : refus, rien ne démarre.
    other = {**raw, "strategy": {**raw["strategy"], "theta_buy": 0.7}}
    moved = tmp_path / "autre.yaml"
    moved.write_text(yaml.safe_dump(other), encoding="utf-8")
    res = runner.invoke(app, ["live-init", *init, "-c", str(moved)])
    assert res.exit_code == 1 and "REFUS" in res.output
    assert not (ledgers / ls.LEDGER_FILE).exists()

    res = runner.invoke(app, ["live-init", *init])
    assert res.exit_code == 0, res.output
    path = ledgers / ls.LEDGER_FILE
    chain = read_jsonl(path)
    assert len(chain) == 2  # genèse + premier jour
    plan = ls.genesis(chain)
    first = full["date"].iloc[shown["n"] - 1].date().isoformat()
    assert plan["first_day"] == first and chain.head.sim_date == first
    assert len(plan["warmup"]) == 40 and plan["models"]["files"]
    timing = next(e for e in chain.head.events if e["type"] == "TIMING")
    assert timing["on_time"] is True

    # Un second live-init est refusé.
    assert runner.invoke(app, ["live-init", *init]).exit_code == 1

    # Rien de nouveau : aucun bloc.
    res = runner.invoke(app, ["live", *common])
    assert res.exit_code == 0 and "Aucun nouveau jour" in res.output
    assert len(read_jsonl(path)) == 2

    # Trois jours plus tard : deux rattrapés (en retard), le dernier à temps.
    shown["n"] = len(full)
    res = runner.invoke(app, ["live", *common])
    assert res.exit_code == 0, res.output
    chain = read_jsonl(path)
    assert len(chain) == 5
    assert chain.verify(load_public_key(pub)).valid
    on_time = [
        next(e for e in b.events if e["type"] == "TIMING")["on_time"] for b in chain.blocks[2:]
    ]
    assert on_time == [False, False, True]
    state = ls.last_state(chain)
    assert state.last_date == full["date"].iloc[-1].date().isoformat()
    assert state.bh_qty > 0  # Buy & Hold a acheté à l'ouverture du deuxième jour
    assert len(state.history) == 40
    forecast = next(e for e in chain.head.events if e["type"] == "FORECAST")
    assert forecast["buy_threshold"] is not None  # le seuil relatif est calculé

    # Des poids modifiés après le plan : refus.
    weights = next(live_models.rglob("*.pt"))
    weights.write_bytes(weights.read_bytes() + b"x")
    res = runner.invoke(app, ["live", *common])
    assert res.exit_code == 1 and "modifié" in res.output

    # La page « Simulation en direct » du tableau de bord affiche ce registre.
    st_testing = pytest.importorskip("streamlit.testing.v1")
    from gold_agents.dashboard import data

    weights.write_bytes(weights.read_bytes()[:-1])
    monkeypatch.setattr(data, "live_ledger", lambda _root: path)
    app_path = Path(__file__).resolve().parents[2] / "dashboard" / "app.py"
    at = st_testing.AppTest.from_file(str(app_path), default_timeout=120)
    at.run()
    at.switch_page("pages/direct.py").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Décision du" in m.value for m in [*at.success, *at.error, *at.info])
