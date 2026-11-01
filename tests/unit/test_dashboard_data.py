"""Lecture des registres pour le tableau de bord (§9), sur les registres du test final."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from gold_agents.dashboard import data

ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = ROOT / "outputs" / "ledger" / "test"
PUBLIC_KEY = ROOT / "ledger" / "public_key.pem"

pytestmark = pytest.mark.skipif(
    not (TEST_DIR / "buy_and_hold.jsonl").is_file(), reason="registres du test absents"
)


def test_run_sets_lists_test_first_with_strategies() -> None:
    sets = data.run_sets(ROOT)
    assert sets[0].name == "test"
    assert sets[0].strategies[0] == "buy_and_hold"
    assert "plan" not in sets[0].strategies and "resultats" not in sets[0].strategies


def test_daily_frame_matches_ledger() -> None:
    blocks = data.read_blocks(TEST_DIR / "agents_gru.jsonl")
    frame = data.daily_frame(blocks)
    assert len(frame) == len(blocks) - 1  # tout sauf la genèse
    assert frame["date"].is_monotonic_increasing
    assert {"value", "r_hat", "signal", "action", "rule"} <= set(frame.columns)
    trades = data.trades_frame(blocks)
    assert set(trades["side"]) == {"BUY", "SELL"}
    # Le nombre d'ordres BUY validés égale le nombre d'achats exécutés.
    assert (frame["action"] == "BUY").sum() == (trades["side"] == "BUY").sum()


def test_prices_and_realized_returns() -> None:
    frames = {"buy_and_hold": data.daily_frame(data.read_blocks(TEST_DIR / "buy_and_hold.jsonl"))}
    close = data.prices(frames)
    assert close.iloc[0] == pytest.approx(1801.2)
    target = data.realized_returns(close)
    assert target.iloc[0] == pytest.approx(np.log(close.iloc[1] / close.iloc[0]))
    assert np.isnan(target.iloc[-1])


def test_sealed_metrics_match_recomputed_ones() -> None:
    sealed = data.results_events(TEST_DIR)["metrics"]
    name = "sma_20_50"
    blocks = data.read_blocks(TEST_DIR / f"{name}.jsonl")
    mine = data.computed_metrics(name, data.daily_frame(blocks), data.trades_frame(blocks))
    ref = next(m for m in sealed if m["strategy"] == name)
    for key in ("final_value", "total_return", "sharpe", "max_drawdown", "n_trades"):
        assert mine[key] == pytest.approx(ref[key], rel=1e-9), key
    assert mine["total_costs"] == pytest.approx(ref["total_costs"], rel=1e-9)


def test_metrics_table_falls_back_to_computed() -> None:
    blocks = data.read_blocks(TEST_DIR / "buy_and_hold.jsonl")
    frames = {"buy_and_hold": data.daily_frame(blocks)}
    trades = {"buy_and_hold": data.trades_frame(blocks)}
    table = data.metrics_table(None, frames, trades)
    assert table.loc[0, "n_trades"] == 1
    assert table.loc[0, "final_value"] == pytest.approx(frames["buy_and_hold"]["value"].iloc[-1])


def test_verify_and_every_tamper_mode_is_detected() -> None:
    path = TEST_DIR / "agents_gru.jsonl"
    assert data.verify(path, PUBLIC_KEY).valid
    for mode in data.TAMPER_MODES:
        result = data.tamper_value(path, 300, 1e6, mode, PUBLIC_KEY)
        assert not result.verification.valid, mode
        assert result.verification.bad_index == 300
    assert data.verify(path, PUBLIC_KEY).valid  # le fichier n'a pas bougé


def test_tamper_scenarios_all_detected() -> None:
    table = data.tamper_scenarios(TEST_DIR / "buy_and_hold.jsonl", PUBLIC_KEY)
    assert len(table) == 5
    assert table["détectée"].all()


def test_anchor_checks_detect_a_changed_ledger(tmp_path: Path) -> None:
    checks = data.anchor_checks(ROOT)
    assert checks[-1]["status"] == "conforme"

    # Copie du dépôt minimal : on change le dernier bloc d'un registre ancré.
    shutil.copytree(TEST_DIR, tmp_path / "outputs" / "ledger" / "test")
    (tmp_path / "ledger").mkdir()
    shutil.copy(ROOT / "ledger" / "anchors.json", tmp_path / "ledger" / "anchors.json")
    path = tmp_path / "outputs" / "ledger" / "test" / "oracle.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    last = json.loads(lines[-1])
    last["hash"] = "f" * 64
    path.write_text("\n".join([*lines[:-1], json.dumps(last)]) + "\n", encoding="utf-8")
    check = data.anchor_checks(tmp_path)[-1]
    assert check["status"] == "DIFFÉRENT"
    states = dict(zip(check["files"]["fichier"], check["files"]["état"], strict=True))
    assert states["outputs/ledger/test/oracle.jsonl"] == "DIFFÉRENT"
    assert states["outputs/ledger/test/plan.jsonl"] == "identique"


def test_mlflow_runs_reads_sqlite(tmp_path: Path) -> None:
    import sqlite3

    db = tmp_path / "mlflow.db"
    with sqlite3.connect(db) as con:
        con.executescript(
            """
            CREATE TABLE experiments (experiment_id INTEGER, name TEXT);
            CREATE TABLE runs (run_uuid TEXT, experiment_id INTEGER, name TEXT, status TEXT,
                               start_time INTEGER, lifecycle_stage TEXT);
            CREATE TABLE params (run_uuid TEXT, key TEXT, value TEXT);
            CREATE TABLE latest_metrics (run_uuid TEXT, key TEXT, value REAL);
            INSERT INTO experiments VALUES (1, 'phase4');
            INSERT INTO runs VALUES ('a', 1, 'gru_seed0', 'FINISHED', 1700000000000, 'active');
            INSERT INTO runs VALUES ('b', 1, 'old', 'FINISHED', 1700000000000, 'deleted');
            INSERT INTO params VALUES ('a', 'hidden', '64');
            INSERT INTO latest_metrics VALUES ('a', 'mae', 0.0075);
            INSERT INTO latest_metrics VALUES ('a', 'val_loss', 1.0);
            """
        )
    runs = data.mlflow_runs(db)
    assert list(runs["run"]) == ["gru_seed0"]
    assert runs.loc[0, "param.hidden"] == "64"
    assert runs.loc[0, "mae"] == pytest.approx(0.0075)
    assert "val_loss" not in runs
    assert data.mlflow_runs(tmp_path / "absent.db").empty


def test_formatting_helpers() -> None:
    assert data.fmt_pct(0.1234) == "+12,3 %"
    assert data.fmt_money(27094.2) == "27 094 $"
    assert data.fmt_num(1.5259) == "1,53"
    assert data.fmt_pct(None) == "–"
    assert data.flatten({"a": {"b": 1}, "c": 2}) == {"a.b": 1, "c": 2}
