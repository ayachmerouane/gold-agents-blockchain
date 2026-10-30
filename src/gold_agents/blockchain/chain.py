"""Chaîne de blocs : ajout, vérification d'intégrité, démonstration de falsification.

Les nouvelles chaînes sont en v2 (Merkle, preuve de travail, signature Ed25519 si une clé
privée est fournie). Les chaînes v1 des phases 3 à 6 restent lisibles et vérifiables.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from gold_agents.blockchain.block import (
    GENESIS_PREV_HASH,
    Block,
    canonical_json,
    meets_difficulty,
)
from gold_agents.blockchain.merkle import merkle_root
from gold_agents.blockchain.signing import Signer, verify_signature

# Zéros hexadécimaux en tête du hash : 3 = environ 4 000 essais par bloc (quelques ms).
# Pédagogique : c'est la signature, pas le calcul, qui empêche de réécrire la chaîne.
DEFAULT_DIFFICULTY = 3


@dataclass(frozen=True)
class Verification:
    valid: bool
    n_blocks: int
    bad_index: int | None = None
    reason: str = ""
    version: int = 2
    # "vérifiées", "absentes" (chaîne non signée) ou "non vérifiées" (pas de clé publique).
    signatures: str = "absentes"

    def describe(self) -> str:
        if not self.valid:
            return f"chaîne INVALIDE au bloc {self.bad_index} : {self.reason}"
        if self.version == 1:
            return f"chaîne valide ({self.n_blocks} blocs, v1 sans signature)"
        return f"chaîne valide ({self.n_blocks} blocs, v2, signatures {self.signatures})"


class Chain:
    def __init__(
        self,
        blocks: list[Block] | None = None,
        signer: Signer | None = None,
        difficulty: int = DEFAULT_DIFFICULTY,
        version: int = 2,
    ) -> None:
        self.blocks: list[Block] = list(blocks or [])
        self.signer = signer
        if self.blocks:  # chaîne relue : on continue dans son format
            version, difficulty = self.blocks[0].version, self.blocks[0].difficulty
        self.version = version
        self.difficulty = difficulty

    @classmethod
    def start(
        cls,
        genesis_events: list[dict[str, Any]],
        signer: Signer | None = None,
        difficulty: int = DEFAULT_DIFFICULTY,
        version: int = 2,
    ) -> Chain:
        """Nouvelle chaîne dont le bloc genèse contient `genesis_events` (config, données)."""
        chain = cls(signer=signer, difficulty=difficulty, version=version)
        chain._add(None, genesis_events)
        return chain

    def __len__(self) -> int:
        return len(self.blocks)

    @property
    def head(self) -> Block:
        return self.blocks[-1]

    @property
    def signed(self) -> bool:
        return self.signer is not None

    def append(self, sim_date: str, events: list[dict[str, Any]]) -> Block:
        return self._add(sim_date, events)

    def _add(self, sim_date: str | None, events: list[dict[str, Any]]) -> Block:
        # On fige les événements tels qu'ils sont au moment de l'ajout.
        frozen = copy.deepcopy(events)
        canonical_json(frozen)  # refuse tout de suite un contenu non sérialisable
        prev_hash = self.head.hash if self.blocks else GENESIS_PREV_HASH
        block = Block(len(self.blocks), sim_date, prev_hash, frozen, version=self.version)
        if self.version > 1:
            block = replace(block, difficulty=self.difficulty)
        block = _seal(block, self.signer)
        self.blocks.append(block)
        return block

    def verify(self, public_key: Ed25519PublicKey | None = None) -> Verification:
        """Recalcule chaque lien, racine de Merkle, hash, preuve de travail et signature.

        Renvoie le premier bloc fautif. Avec `public_key`, chaque bloc v2 doit porter une
        signature valide pour cette clé.
        """
        n = len(self.blocks)
        if n == 0:
            return Verification(False, 0, 0, "chaîne vide")
        version, difficulty = self.blocks[0].version, self.blocks[0].difficulty
        if version not in (1, 2):
            return Verification(False, n, 0, f"version {version} inconnue", version)

        def fail(i: int, reason: str) -> Verification:
            return Verification(False, n, i, reason, version)

        prev_hash = GENESIS_PREV_HASH
        for i, block in enumerate(self.blocks):
            if block.index != i:
                return fail(i, f"index {block.index} au lieu de {i}")
            if block.version != version:
                return fail(i, f"version {block.version} dans une chaîne v{version}")
            if block.prev_hash != prev_hash:
                return fail(i, "prev_hash ne correspond pas au bloc précédent")
            if version > 1:
                if block.merkle_root != merkle_root(block.events):
                    return fail(i, "événements modifiés (racine de Merkle différente)")
                if block.difficulty != difficulty:
                    return fail(i, f"difficulté {block.difficulty} au lieu de {difficulty}")
            if block.compute_hash() != block.hash:
                return fail(i, "contenu modifié (hash recalculé différent)")
            if version > 1:
                if not meets_difficulty(block.hash, difficulty):
                    return fail(i, "preuve de travail absente (hash sans les zéros requis)")
                if public_key is not None:
                    if not block.signature:
                        return fail(i, "signature absente")
                    if not verify_signature(public_key, block.hash, block.signature):
                        return fail(i, "signature invalide pour la clé publique publiée")
            prev_hash = block.hash

        if version == 1:
            signatures = "absentes"
        elif public_key is not None:
            signatures = "vérifiées"
        elif any(b.signature for b in self.blocks):
            signatures = "non vérifiées (clé publique non fournie)"
        else:
            signatures = "absentes"
        return Verification(True, n, version=version, signatures=signatures)


def _seal(block: Block, signer: Signer | None) -> Block:
    block = block.sealed()
    if signer is not None and block.version > 1:
        block = replace(block, signature=signer.sign(block.hash))
    return block


def rewrite_from(blocks: list[Block], start: int, signer: Signer | None) -> list[Block]:
    """Ce que ferait un fraudeur : refaire hash, nonce et signature de `start` à la fin."""
    out = list(blocks)
    for i in range(start, len(out)):
        prev = out[i - 1].hash if i > 0 else GENESIS_PREV_HASH
        out[i] = _seal(replace(out[i], prev_hash=prev), signer)
    return out


def tamper_demo(
    chain: Chain, public_key: Ed25519PublicKey | None = None
) -> list[tuple[str, Verification]]:
    """Falsifie des copies de la chaîne et montre que la vérification le détecte (§10.4).

    La chaîne d'origine n'est pas modifiée. Sans clé publique (chaîne v1 ou non signée),
    la réécriture complète passe : c'est la limite que la signature corrige.
    """
    if len(chain) < 4:
        raise ValueError("il faut au moins 4 blocs pour la démonstration")
    target = len(chain) // 2
    out: list[tuple[str, Verification]] = []

    def check(blocks: list[Block]) -> Verification:
        return Chain(blocks).verify(public_key)

    # 1. Ajouter une décision dans un ancien bloc.
    blocks = copy.deepcopy(chain.blocks)
    blocks[target].events.append({"type": "FALSIFICATION", "note": "décision ajoutée après coup"})
    out.append((f"événement ajouté dans le bloc {target}", check(blocks)))

    # 2. Même chose, mais le fraudeur recalcule Merkle, nonce et hash du bloc modifié.
    blocks[target] = blocks[target].sealed()
    out.append((f"bloc {target} modifié et re-haché", check(blocks)))

    # 3. Le fraudeur réécrit toute la suite de la chaîne, signée avec SA clé.
    fraud_key = Signer.generate() if public_key is not None else None
    out.append(
        (f"chaîne réécrite depuis le bloc {target}", check(rewrite_from(blocks, target, fraud_key)))
    )

    # 4. Supprimer un bloc.
    blocks = copy.deepcopy(chain.blocks)
    del blocks[target]
    out.append((f"bloc {target} supprimé", check(blocks)))

    # 5. Inverser deux blocs.
    blocks = copy.deepcopy(chain.blocks)
    blocks[target], blocks[target + 1] = blocks[target + 1], blocks[target]
    out.append((f"blocs {target} et {target + 1} inversés", check(blocks)))
    return out
