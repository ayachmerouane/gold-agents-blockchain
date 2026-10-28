"""Intégrité du registre v1 (§10.4) : toute falsification est détectée."""

import copy
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gold_agents.blockchain.block import GENESIS_PREV_HASH, Block
from gold_agents.blockchain.chain import Chain, tamper_demo
from gold_agents.blockchain.storage import read_jsonl, write_jsonl
from gold_agents.simulation.costs import CostModel
from gold_agents.simulation.engine import run_backtest
from gold_agents.simulation.strategies import SmaCrossover


def _chain(n: int = 6) -> Chain:
    chain = Chain.start([{"type": "GENESIS", "seed": 42}])
    for i in range(1, n):
        chain.append(f"2024-01-{i:02d}", [{"type": "DECISION", "action": "HOLD", "close": 1.5 * i}])
    return chain


def test_hash_is_deterministic_and_covers_content() -> None:
    a = Block(1, "2024-01-01", GENESIS_PREV_HASH, [{"b": 1, "a": 2.5}]).sealed()
    b = Block(1, "2024-01-01", GENESIS_PREV_HASH, [{"a": 2.5, "b": 1}]).sealed()
    c = Block(1, "2024-01-01", GENESIS_PREV_HASH, [{"a": 2.5, "b": 2}]).sealed()
    assert a.hash == b.hash != c.hash and len(a.hash) == 64


def test_valid_chain_links_blocks() -> None:
    chain = _chain()
    assert chain.verify().valid and chain.verify().n_blocks == 6
    for prev, block in zip(chain.blocks, chain.blocks[1:], strict=False):
        assert block.prev_hash == prev.hash


def test_append_freezes_events() -> None:
    chain = _chain(2)
    events = [{"type": "DECISION", "action": "BUY"}]
    chain.append("2024-02-01", events)
    events[0]["action"] = "SELL"
    assert chain.head.events[0]["action"] == "BUY" and chain.verify().valid


def test_modified_event_detected_at_its_block() -> None:
    blocks = copy.deepcopy(_chain().blocks)
    blocks[3].events[0]["close"] = 999.0
    v = Chain(blocks).verify()
    assert not v.valid and v.bad_index == 3


def test_rehashed_block_detected_at_next_block() -> None:
    blocks = copy.deepcopy(_chain().blocks)
    blocks[3].events[0]["action"] = "BUY"
    blocks[3] = blocks[3].sealed()
    v = Chain(blocks).verify()
    assert not v.valid and v.bad_index == 4


@pytest.mark.parametrize("edit", ["delete", "swap", "insert"])
def test_structure_changes_detected(edit: str) -> None:
    blocks = copy.deepcopy(_chain().blocks)
    if edit == "delete":
        del blocks[2]
    elif edit == "swap":
        blocks[2], blocks[3] = blocks[3], blocks[2]
    else:
        blocks.insert(2, Block(2, "2024-01-01", blocks[1].hash, []).sealed())
    assert not Chain(blocks).verify().valid


def test_tamper_demo_detects_everything_and_leaves_chain_intact() -> None:
    chain = _chain(10)
    results = tamper_demo(chain)
    assert len(results) == 4 and not any(v.valid for _, v in results)
    assert chain.verify().valid


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    chain = _chain()
    back = read_jsonl(write_jsonl(chain, tmp_path / "c.jsonl"))
    assert back.verify().valid and [b.hash for b in back.blocks] == [b.hash for b in chain.blocks]


def test_backtest_replay_gives_identical_ledger(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    closes = 1800 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))
    dates = pd.bdate_range("2024-01-01", periods=200, tz="UTC")
    frame = pd.DataFrame({"date": dates, "open": closes, "close": closes})
    files = []
    for i in range(2):
        chain = Chain.start([{"type": "GENESIS"}])
        run_backtest(
            frame, SmaCrossover(5, 20), CostModel(0.001), 1000, date(2024, 2, 1), ledger=chain
        )
        files.append(write_jsonl(chain, tmp_path / f"{i}.jsonl").read_bytes())
    assert files[0] == files[1]
    chain = read_jsonl(tmp_path / "0.jsonl")
    assert chain.verify().valid
    assert any(e["type"] == "TRADE" for b in chain.blocks for e in b.events)
