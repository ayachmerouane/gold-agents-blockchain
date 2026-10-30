"""Bloc du registre (§8.2).

Version 1 (phases 3 à 6) : le hash est le SHA-256 de la sérialisation canonique (JSON trié,
sans espaces) de `index`, `sim_date`, `prev_hash` et `events`. Ce format reste lisible et
vérifiable : le registre des réglages figés de la phase 6 est un bloc v1.

Version 2 (phase 7) : le hash couvre l'en-tête `version`, `index`, `sim_date`, `prev_hash`,
`merkle_root`, `difficulty` et `nonce`. Les événements y sont couverts par leur racine de
Merkle. Le hash doit commencer par `difficulty` zéros hexadécimaux (preuve de travail : on
cherche le `nonce`). La signature Ed25519 porte sur le hash ; elle n'est donc pas dans
l'en-tête. Aucune heure réelle dans le bloc : rejouer un backtest redonne le même registre.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Any

from gold_agents.blockchain.canonical import canonical_json, sha256_hex
from gold_agents.blockchain.merkle import merkle_root

__all__ = ["GENESIS_PREV_HASH", "Block", "canonical_json", "meets_difficulty", "sha256_hex"]

GENESIS_PREV_HASH = "0" * 64


def meets_difficulty(block_hash: str, difficulty: int) -> bool:
    return block_hash.startswith("0" * difficulty)


@dataclass(frozen=True)
class Block:
    index: int
    sim_date: str | None
    prev_hash: str
    events: list[dict[str, Any]]
    hash: str = field(default="")
    version: int = 1
    merkle_root: str = ""
    difficulty: int = 0
    nonce: int = 0
    signature: str | None = None

    def content(self) -> dict[str, Any]:
        """Champs couverts par le hash : tout en v1, l'en-tête en v2."""
        if self.version == 1:
            return {
                "index": self.index,
                "sim_date": self.sim_date,
                "prev_hash": self.prev_hash,
                "events": self.events,
            }
        return {
            "version": self.version,
            "index": self.index,
            "sim_date": self.sim_date,
            "prev_hash": self.prev_hash,
            "merkle_root": self.merkle_root,
            "difficulty": self.difficulty,
            "nonce": self.nonce,
        }

    def compute_hash(self) -> str:
        return sha256_hex(canonical_json(self.content()))

    def sealed(self) -> Block:
        """Copie avec son hash calculé ; en v2, racine de Merkle recalculée et nonce miné.

        En v2, la signature éventuelle est retirée : elle ne vaut plus pour le nouveau hash.
        """
        if self.version == 1:
            return replace(self, hash=self.compute_hash())
        block = replace(self, merkle_root=merkle_root(self.events), nonce=0, signature=None)
        nonce, block_hash = _mine(block)
        return replace(block, nonce=nonce, hash=block_hash)

    def to_dict(self) -> dict[str, Any]:
        data = {**self.content(), "hash": self.hash}
        if self.version > 1:
            data["events"] = self.events
            data["signature"] = self.signature
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Block:
        version = int(data.get("version", 1))
        block = cls(
            index=int(data["index"]),
            sim_date=data["sim_date"],
            prev_hash=str(data["prev_hash"]),
            events=list(data["events"]),
            hash=str(data["hash"]),
            version=version,
        )
        if version == 1:
            return block
        return replace(
            block,
            merkle_root=str(data["merkle_root"]),
            difficulty=int(data["difficulty"]),
            nonce=int(data["nonce"]),
            signature=data.get("signature"),
        )


def _mine(block: Block) -> tuple[int, str]:
    """Cherche le premier nonce dont le hash d'en-tête atteint la difficulté.

    Le JSON canonique de l'en-tête est coupé autour du nonce : on ne hache que la partie
    qui change (même résultat que `compute_hash`, en bien plus rapide).
    """
    text = canonical_json(block.content())
    head, tail = text.split('"nonce":0', 1)
    base = hashlib.sha256((head + '"nonce":').encode("utf-8"))
    tail_bytes = tail.encode("utf-8")
    nonce = 0
    while True:
        h = base.copy()
        h.update(str(nonce).encode("ascii") + tail_bytes)
        digest = h.hexdigest()
        if meets_difficulty(digest, block.difficulty):
            return nonce, digest
        nonce += 1
