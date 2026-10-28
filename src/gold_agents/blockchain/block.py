"""Bloc du registre (§8.2), version 1 : blocs chaînés par SHA-256.

Le hash d'un bloc est le SHA-256 de la sérialisation canonique (JSON trié, sans espaces)
de tous ses champs sauf `hash`. Il contient `prev_hash`, le hash du bloc précédent :
modifier un ancien bloc change son hash et casse tous les liens qui suivent.
La racine de Merkle, la preuve de travail et la signature Ed25519 viendront en v2.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

GENESIS_PREV_HASH = "0" * 64


def canonical_json(data: Any) -> str:
    """JSON déterministe : clés triées, sans espaces, NaN interdit."""
    return json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Block:
    index: int
    sim_date: str | None
    prev_hash: str
    events: list[dict[str, Any]]
    hash: str = field(default="")

    def content(self) -> dict[str, Any]:
        """Champs couverts par le hash (tous sauf `hash`)."""
        return {
            "index": self.index,
            "sim_date": self.sim_date,
            "prev_hash": self.prev_hash,
            "events": self.events,
        }

    def compute_hash(self) -> str:
        return sha256_hex(canonical_json(self.content()))

    def sealed(self) -> Block:
        """Copie du bloc avec son hash calculé."""
        return Block(self.index, self.sim_date, self.prev_hash, self.events, self.compute_hash())

    def to_dict(self) -> dict[str, Any]:
        return {**self.content(), "hash": self.hash}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Block:
        return cls(
            index=int(data["index"]),
            sim_date=data["sim_date"],
            prev_hash=str(data["prev_hash"]),
            events=list(data["events"]),
            hash=str(data["hash"]),
        )
