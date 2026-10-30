"""Arbre de Merkle (§8.2) : racine, preuves d'inclusion."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from gold_agents.blockchain.merkle import EMPTY_ROOT, merkle_proof, merkle_root, verify_proof

EVENTS = [{"type": t, "i": i} for i, t in enumerate(["FORECAST", "SIGNAL", "ORDER", "TRADE", "X"])]


def test_root_depends_on_every_event_and_order() -> None:
    root = merkle_root(EVENTS)
    assert merkle_root([]) == EMPTY_ROOT and len(root) == 64
    assert merkle_root(EVENTS[::-1]) != root
    assert merkle_root([*EVENTS, EVENTS[-1]]) != root  # pas de doublon silencieux
    changed = [dict(e) for e in EVENTS]
    changed[2]["i"] = 99
    assert merkle_root(changed) != root


def test_single_leaf_is_not_a_node() -> None:
    # Préfixes 0x00 / 0x01 : une feuille ne vaut pas le nœud de deux feuilles.
    assert merkle_root([EVENTS[0]]) != merkle_root(EVENTS[:2])


@given(st.lists(st.dictionaries(st.text(max_size=3), st.integers()), min_size=1, max_size=17))
def test_every_event_has_a_valid_proof(events: list[dict[str, int]]) -> None:
    root = merkle_root(events)
    for i, event in enumerate(events):
        assert verify_proof(event, merkle_proof(events, i), root)


def test_proof_rejects_another_event() -> None:
    root = merkle_root(EVENTS)
    proof = merkle_proof(EVENTS, 3)
    assert not verify_proof({"type": "TRADE", "i": 4}, proof, root)
    with pytest.raises(IndexError):
        merkle_proof(EVENTS, 5)
