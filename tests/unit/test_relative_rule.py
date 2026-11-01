"""Tentative 2 : seuil d'achat relatif, contrainte sur le nombre de trades, archivage."""

from __future__ import annotations

import json
import math
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from gold_agents.agents.messages import Forecast
from gold_agents.agents.strategy import StrategyAgent, past_quantile, relative_threshold
from gold_agents.blockchain.archive import archive_file, read_moves, resolve
from gold_agents.config import Config, StrategyConfig, apply_settings, load_config
from gold_agents.dashboard import data
from gold_agents.evaluation import tuning as tu
from gold_agents.simulation.fast import AgentParams, buy_thresholds

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "config.yaml"


def test_past_quantile_uses_only_the_window_and_needs_history() -> None:
    assert math.isnan(past_quantile([0.1, 0.2], 0.5, 10, 3))
    assert past_quantile([9.0, 1.0, 2.0, 3.0], 0.5, 3, 3) == 2.0  # 9 est hors fenêtre
    assert relative_threshold([-3.0, -2.0, -1.0], 0.5, 10, 3) == 0.0  # jamais sous 0


def test_relative_rule_is_scale_free() -> None:
    """Même prévisions à une échelle 10 fois plus petite : mêmes décisions."""
    rng = np.random.default_rng(0)
    r = rng.normal(0.0003, 0.001, 300)
    p = AgentParams(0.8, buy_rule="quantile", quantile_window=100, quantile_min_days=20)
    big = r > buy_thresholds(r, p)
    small = r / 10 > buy_thresholds(r / 10, p)
    assert (big == small).all() and big.sum() > 30
    assert not big[:20].any()  # historique trop court au début


def test_agent_threshold_ignores_today_and_future() -> None:
    agent = StrategyAgent(0.5, 0.0, None, "quantile", 3, 3)
    day = date(2024, 1, 1)
    for r in (0.001, 0.002, 0.003):
        sig = agent.act(Forecast(day, r, None, "m"), invested=False)
        assert sig.action.value == "HOLD"  # moins de 3 prévisions passées
    sig = agent.act(Forecast(day, 0.0025, None, "m"), invested=False)
    assert sig.action.value == "BUY"  # 0,0025 > médiane(0,001 ; 0,002 ; 0,003)
    assert "quantile 50%" in sig.reason
    assert "_history" not in __import__("dataclasses").asdict(agent)


def test_quantile_rule_needs_a_level_between_0_and_1() -> None:
    with pytest.raises(ValueError):
        StrategyConfig(buy_rule="quantile", theta_buy=1.5, theta_sell=0.0)
    StrategyConfig(buy_rule="quantile", theta_buy=0.8, theta_sell=0.0)


def test_trades_constraint_comes_first() -> None:
    grid = load_config(CONFIG).tuning
    assert grid is not None
    rows = []
    for q, trades, sharpe in ((0.5, 0, 9.0), (0.9, 10, 0.5)):
        rows.append(
            {
                "model": "ensemble",
                "theta_buy": q,
                "theta_sell": 0.0,
                "min_confidence": 1.0,
                "stop_loss": 0.08,
                "max_drawdown": 0.25,
                "fold": 1,
                "sharpe": sharpe,
                "total_return": 0.0,
                "worst_drawdown": -0.1,
                "n_trades": trades,
                "trades_per_year": trades / 2,
                "exposure": 0.5,
                "dd_ok": True,
            }
        )
    table = pd.DataFrame(rows)
    free = tu.best(tu.summarize_grid(table, grid.grid, [1]))
    bounded = tu.best(tu.summarize_grid(table, grid.grid, [1], 4, 60))
    assert free.theta_buy == 0.5  # sans contrainte : le Sharpe d'un système qui ne trade pas
    assert bounded.theta_buy == 0.9


def test_apply_settings_rewrites_only_the_settings(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    shutil.copy(CONFIG, path)
    before = path.read_text(encoding="utf-8").splitlines()
    settings = {
        "buy_rule": "quantile",
        "theta_buy": 0.8,
        "theta_sell": 0.0,
        "min_confidence": None,
        "model": "ensemble",
        "stop_loss": 0.05,
        "max_drawdown": 0.25,
        "cooldown_days": 20,
    }
    cfg = load_config(path)
    now = {**cfg.strategy.model_dump(), **cfg.risk.model_dump()}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    missing = {k for k in settings if k not in {**raw["strategy"], **raw["risk"]}}
    changed = apply_settings(path, settings)
    assert set(changed) == {k for k in settings if now[k] != settings[k]}
    cfg = load_config(path)
    assert cfg.strategy.buy_rule == "quantile" and cfg.strategy.theta_buy == 0.8
    assert cfg.strategy.min_confidence is None and cfg.risk.stop_loss == 0.05
    assert cfg.tuning is not None and cfg.tuning.grid.theta_buy == [0.5, 0.7, 0.8, 0.9]
    after = path.read_text(encoding="utf-8").splitlines()
    assert len(after) == len(before) + len(missing)  # clés absentes ajoutées, rien d'autre
    assert apply_settings(path, settings) == []  # deuxième passage : rien à changer
    Config.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def copy_test_anchors(src: Path, dst: Path) -> None:
    """Copie les seuls ancrages du test final (sans ceux de la simulation en direct)."""
    anchors = json.loads(src.read_text(encoding="utf-8"))
    kept = [
        a
        for a in anchors
        if all(x["file"].startswith("outputs/ledger/test/") for x in a["ledgers"])
    ]
    dst.write_text(json.dumps(kept), encoding="utf-8")


def test_archived_ledger_keeps_its_anchor_valid(tmp_path: Path) -> None:
    shutil.copytree(ROOT / "outputs" / "ledger" / "test", tmp_path / "outputs" / "ledger" / "test")
    (tmp_path / "ledger").mkdir()
    copy_test_anchors(ROOT / "ledger" / "anchors.json", tmp_path / "ledger" / "anchors.json")
    assert data.anchor_checks(tmp_path)[-1]["status"] == "conforme"

    moves = tmp_path / "ledger" / "moves.json"
    src = tmp_path / "outputs" / "ledger" / "test"
    for f in sorted(src.glob("*.jsonl")):
        archive_file(f, tmp_path / "outputs" / "ledger" / "test_tentative1" / f.name, moves)
    # Chemins notés tels quels ; on les remet relatifs à la racine, comme dans le dépôt.
    entries = json.loads(moves.read_text(encoding="utf-8"))
    for e in entries:
        e["from"] = Path(e["from"]).relative_to(tmp_path).as_posix()
        e["to"] = Path(e["to"]).relative_to(tmp_path).as_posix()
    moves.write_text(json.dumps(entries), encoding="utf-8")
    assert resolve("outputs/ledger/test/plan.jsonl", read_moves(moves)) == (
        "outputs/ledger/test_tentative1/plan.jsonl"
    )
    assert data.anchor_checks(tmp_path)[-1]["status"] == "conforme"

    # Tentative 2 : un nouveau registre au même chemin, ancré APRÈS le déplacement. Chaque
    # ancrage doit être comparé au bon fichier (l'ancien archivé, ou le nouveau).
    new = tmp_path / "outputs" / "ledger" / "test" / "oracle.jsonl"
    old_lines = (
        (tmp_path / "outputs" / "ledger" / "test_tentative1" / "oracle.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    new.parent.mkdir(parents=True, exist_ok=True)
    new.write_text("\n".join(old_lines[:10]) + "\n", encoding="utf-8")
    head = data.ledger_head(new, tmp_path)
    anchors_path = tmp_path / "ledger" / "anchors.json"
    anchors = json.loads(anchors_path.read_text(encoding="utf-8"))
    anchors.append(
        {
            **anchors[-1],
            "anchored_at": "2999-01-01T00:00:00+00:00",
            "ledgers": [{"file": head.file, "blocks": head.blocks, "head": head.head}],
            "digest": data.anchor_digest([head]),
        }
    )
    anchors_path.write_text(json.dumps(anchors), encoding="utf-8")
    checks = data.anchor_checks(tmp_path)
    assert checks[-1]["status"] == "conforme" and checks[-2]["status"] == "conforme"

    with pytest.raises(FileExistsError):
        dst = tmp_path / "outputs" / "ledger" / "test_tentative1" / "oracle.jsonl"
        archive_file(dst, dst, moves)
