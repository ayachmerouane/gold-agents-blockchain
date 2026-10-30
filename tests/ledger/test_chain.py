"""Intégrité du registre (§10.4) : toute falsification est détectée.

Chaînes v2 (Merkle, preuve de travail, signature Ed25519) et chaînes v1 des phases 3 à 6.
"""

import copy
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gold_agents.blockchain.block import GENESIS_PREV_HASH, Block
from gold_agents.blockchain.chain import Chain, rewrite_from, tamper_demo
from gold_agents.blockchain.merkle import merkle_root
from gold_agents.blockchain.signing import Signer
from gold_agents.blockchain.storage import read_jsonl, write_jsonl
from gold_agents.simulation.costs import CostModel
from gold_agents.simulation.engine import run_backtest
from gold_agents.simulation.strategies import SmaCrossover

GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "ledger"
FROZEN_V1_HEAD = "ca1c3fd2a834b035cdb1b8c55a1726faca8b89637276d7c1eb037c5a192d60c9"
SIGNER = Signer.from_seed(bytes(range(32)))


def _chain(n: int = 6, signer: Signer | None = SIGNER, version: int = 2) -> Chain:
    chain = Chain.start([{"type": "GENESIS", "seed": 42}], signer, version=version)
    for i in range(1, n):
        chain.append(
            f"2024-01-{i:02d}",
            [
                {"type": "FORECAST", "r_hat": 0.001 * i},
                {"type": "DECISION", "action": "HOLD", "close": 1.5 * i},
                {"type": "PORTFOLIO", "value": 1000.0 + i},
            ],
        )
    return chain


def _check(blocks: list[Block]) -> tuple[bool, int | None, str]:
    v = Chain(blocks).verify(SIGNER.public_key)
    return v.valid, v.bad_index, v.reason


# --- Blocs v2 ---------------------------------------------------------------


def test_v2_block_has_merkle_pow_and_signature() -> None:
    chain = _chain()
    v = chain.verify(SIGNER.public_key)
    assert v.valid and v.n_blocks == 6 and v.signatures == "vérifiées"
    for prev, block in zip(chain.blocks, chain.blocks[1:], strict=False):
        assert block.prev_hash == prev.hash
        assert block.merkle_root == merkle_root(block.events)
        assert block.hash.startswith("000") and block.signature
        assert block.compute_hash() == block.hash  # minage rapide = calcul de référence


def test_hash_is_deterministic_and_covers_events() -> None:
    def block(events: list[dict[str, float]]) -> Block:
        return Block(1, "2024-01-01", GENESIS_PREV_HASH, events, version=2, difficulty=2)

    a = block([{"b": 1, "a": 2.5}]).sealed()
    b = block([{"a": 2.5, "b": 1}]).sealed()
    c = block([{"a": 2.5, "b": 2}]).sealed()
    assert a.hash == b.hash != c.hash and len(a.hash) == 64


def test_append_freezes_events() -> None:
    chain = _chain(2)
    events = [{"type": "DECISION", "action": "BUY"}]
    chain.append("2024-02-01", events)
    events[0]["action"] = "SELL"
    assert chain.head.events[0]["action"] == "BUY" and chain.verify(SIGNER.public_key).valid


def test_unsigned_chain_is_valid_but_says_so() -> None:
    chain = _chain(signer=None)
    assert chain.verify().valid and chain.verify().signatures == "absentes"
    v = chain.verify(SIGNER.public_key)  # une clé publique exige des signatures
    assert not v.valid and v.bad_index == 0 and v.reason == "signature absente"
    assert _chain().verify().signatures.startswith("non vérifiées")


# --- Falsifications (§10.4) -------------------------------------------------


def test_modified_event_detected_by_merkle_root() -> None:
    blocks = copy.deepcopy(_chain().blocks)
    blocks[3].events[1]["close"] = 999.0
    assert _check(blocks) == (False, 3, "événements modifiés (racine de Merkle différente)")


def test_modified_header_detected_by_hash() -> None:
    blocks = copy.deepcopy(_chain().blocks)
    blocks[3] = replace(blocks[3], sim_date="2024-12-31")
    assert _check(blocks) == (False, 3, "contenu modifié (hash recalculé différent)")


def test_resealed_block_detected_by_signature() -> None:
    blocks = copy.deepcopy(_chain().blocks)
    blocks[3].events[1]["action"] = "BUY"
    blocks[3] = blocks[3].sealed()  # Merkle, nonce et hash refaits : la signature manque
    assert _check(blocks) == (False, 3, "signature absente")
    blocks[3] = replace(blocks[3], signature=_chain().blocks[3].signature)  # ancienne
    assert _check(blocks)[:2] == (False, 3)


def test_full_rewrite_with_another_key_detected() -> None:
    blocks = copy.deepcopy(_chain().blocks)
    blocks[3].events[1]["action"] = "BUY"
    forged = rewrite_from(blocks, 3, Signer.generate())
    assert Chain(forged).verify().valid  # sans clé publique, le faux semble valide...
    valid, index, reason = _check(forged)  # ... pas avec la clé publiée
    assert (valid, index) == (False, 3) and "signature invalide" in reason


def test_missing_proof_of_work_detected() -> None:
    blocks = copy.deepcopy(_chain().blocks)
    # Le fraudeur signe (avec la bonne clé, pour isoler la règle) un bloc non miné.
    nonce = blocks[2].nonce + 1
    while True:
        lazy = replace(blocks[2], nonce=nonce)
        lazy = replace(lazy, hash=lazy.compute_hash())
        if not lazy.hash.startswith("000"):
            break
        nonce += 1
    blocks[2] = replace(lazy, signature=SIGNER.sign(lazy.hash))
    assert _check(blocks) == (False, 2, "preuve de travail absente (hash sans les zéros requis)")
    easy = replace(blocks[2], difficulty=0)
    blocks[2] = replace(easy, hash=easy.compute_hash())
    assert _check(blocks) == (False, 2, "difficulté 0 au lieu de 3")


@pytest.mark.parametrize("edit", ["delete", "swap", "insert", "truncate_genesis"])
def test_structure_changes_detected(edit: str) -> None:
    blocks = copy.deepcopy(_chain().blocks)
    if edit == "delete":
        del blocks[2]
    elif edit == "swap":
        blocks[2], blocks[3] = blocks[3], blocks[2]
    elif edit == "insert":
        blocks.insert(2, replace(blocks[1], index=2, prev_hash=blocks[1].hash))
    else:
        del blocks[0]
    assert not _check(blocks)[0]


def test_mixed_versions_rejected() -> None:
    blocks = copy.deepcopy(_chain().blocks)
    v1 = Block(3, "2024-01-03", blocks[2].hash, blocks[3].events).sealed()
    blocks[3] = v1
    assert not _check(blocks)[0]


def test_tamper_demo_detects_everything_and_leaves_chain_intact() -> None:
    chain = _chain(10)
    results = tamper_demo(chain, SIGNER.public_key)
    assert len(results) == 5 and not any(v.valid for _, v in results)
    assert chain.verify(SIGNER.public_key).valid


def test_tamper_demo_shows_why_signatures_matter() -> None:
    # Sans signature, un fraudeur qui réécrit toute la suite passe la vérification.
    results = dict(tamper_demo(_chain(10, signer=None)))
    assert results["chaîne réécrite depuis le bloc 5"].valid
    assert sum(v.valid for v in results.values()) == 1


# --- v1 (phases 3 à 6) ------------------------------------------------------


def test_v1_chain_still_detects_tampering() -> None:
    chain = _chain(signer=None, version=1)
    assert chain.verify().valid and chain.verify().version == 1
    blocks = copy.deepcopy(chain.blocks)
    blocks[3].events[1]["close"] = 999.0
    v = Chain(blocks).verify()
    assert not v.valid and v.bad_index == 3
    blocks[3] = blocks[3].sealed()
    v = Chain(blocks).verify()
    assert not v.valid and v.bad_index == 4


def test_frozen_settings_of_phase_6_still_verify(tmp_path: Path) -> None:
    path = GOLDEN / "reglages_figes_v1.jsonl"
    chain = read_jsonl(path)
    v = chain.verify(SIGNER.public_key)  # une clé publique ne change rien en v1
    assert v.valid and v.version == 1 and chain.head.hash == FROZEN_V1_HEAD
    assert write_jsonl(chain, tmp_path / "copie.jsonl").read_bytes() == path.read_bytes()


# --- Fichiers et rejeu ------------------------------------------------------


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    chain = _chain()
    back = read_jsonl(write_jsonl(chain, tmp_path / "c.jsonl"))
    assert back.verify(SIGNER.public_key).valid
    assert [b.to_dict() for b in back.blocks] == [b.to_dict() for b in chain.blocks]


def test_backtest_replay_gives_identical_ledger(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    closes = 1800 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))
    dates = pd.bdate_range("2024-01-01", periods=200, tz="UTC")
    frame = pd.DataFrame({"date": dates, "open": closes, "close": closes})
    files = []
    for i in range(2):
        chain = Chain.start([{"type": "GENESIS"}], SIGNER)
        run_backtest(
            frame, SmaCrossover(5, 20), CostModel(0.001), 1000, date(2024, 2, 1), ledger=chain
        )
        files.append(write_jsonl(chain, tmp_path / f"{i}.jsonl").read_bytes())
    assert files[0] == files[1]  # Ed25519 est déterministe : même octets
    chain = read_jsonl(tmp_path / "0.jsonl")
    assert chain.verify(SIGNER.public_key).valid
    assert any(e["type"] == "TRADE" for b in chain.blocks for e in b.events)
