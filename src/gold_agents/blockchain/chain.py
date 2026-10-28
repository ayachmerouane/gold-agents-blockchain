"""Chaîne de blocs v1 : ajout, vérification d'intégrité, démonstration de falsification."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from gold_agents.blockchain.block import GENESIS_PREV_HASH, Block, canonical_json


@dataclass(frozen=True)
class Verification:
    valid: bool
    n_blocks: int
    bad_index: int | None = None
    reason: str = ""

    def describe(self) -> str:
        if self.valid:
            return f"chaîne valide ({self.n_blocks} blocs)"
        return f"chaîne INVALIDE au bloc {self.bad_index} : {self.reason}"


class Chain:
    def __init__(self, blocks: list[Block] | None = None) -> None:
        self.blocks: list[Block] = list(blocks or [])

    @classmethod
    def start(cls, genesis_events: list[dict[str, Any]]) -> Chain:
        """Nouvelle chaîne dont le bloc genèse contient `genesis_events` (config, données)."""
        chain = cls()
        chain.blocks.append(Block(0, None, GENESIS_PREV_HASH, genesis_events).sealed())
        return chain

    def __len__(self) -> int:
        return len(self.blocks)

    @property
    def head(self) -> Block:
        return self.blocks[-1]

    def append(self, sim_date: str, events: list[dict[str, Any]]) -> Block:
        # On fige les événements tels qu'ils sont au moment de l'ajout.
        frozen = copy.deepcopy(events)
        canonical_json(frozen)  # refuse tout de suite un contenu non sérialisable
        block = Block(len(self.blocks), sim_date, self.head.hash, frozen).sealed()
        self.blocks.append(block)
        return block

    def verify(self) -> Verification:
        """Recalcule chaque hash et chaque lien ; renvoie le premier bloc fautif."""
        n = len(self.blocks)
        if n == 0:
            return Verification(False, 0, 0, "chaîne vide")
        prev_hash = GENESIS_PREV_HASH
        for i, block in enumerate(self.blocks):
            if block.index != i:
                return Verification(False, n, i, f"index {block.index} au lieu de {i}")
            if block.prev_hash != prev_hash:
                return Verification(False, n, i, "prev_hash ne correspond pas au bloc précédent")
            if block.compute_hash() != block.hash:
                return Verification(False, n, i, "contenu modifié (hash recalculé différent)")
            prev_hash = block.hash
        return Verification(True, n)


def tamper_demo(chain: Chain) -> list[tuple[str, Verification]]:
    """Falsifie des copies de la chaîne et montre que la vérification le détecte (§10.4).

    La chaîne d'origine n'est pas modifiée.
    """
    if len(chain) < 4:
        raise ValueError("il faut au moins 4 blocs pour la démonstration")
    target = len(chain) // 2
    out: list[tuple[str, Verification]] = []

    # 1. Changer un chiffre dans un ancien bloc.
    blocks = copy.deepcopy(chain.blocks)
    events = blocks[target].events
    events.append({"type": "FALSIFICATION", "note": "décision ajoutée après coup"})
    out.append((f"événement ajouté dans le bloc {target}", Chain(blocks).verify()))

    # 2. Même chose, mais le fraudeur recalcule le hash du bloc modifié.
    blocks[target] = blocks[target].sealed()
    out.append((f"bloc {target} modifié et re-haché", Chain(blocks).verify()))

    # 3. Supprimer un bloc.
    blocks = copy.deepcopy(chain.blocks)
    del blocks[target]
    out.append((f"bloc {target} supprimé", Chain(blocks).verify()))

    # 4. Inverser deux blocs.
    blocks = copy.deepcopy(chain.blocks)
    blocks[target], blocks[target + 1] = blocks[target + 1], blocks[target]
    out.append((f"blocs {target} et {target + 1} inversés", Chain(blocks).verify()))
    return out
