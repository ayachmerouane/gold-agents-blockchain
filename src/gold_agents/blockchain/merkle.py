"""Arbre de Merkle des événements d'un bloc (§8.2, v2).

Chaque événement est haché (feuille), puis les empreintes sont combinées deux par deux
jusqu'à une seule : la racine. Le bloc ne stocke que la racine ; changer un seul champ
d'un seul événement change la racine.

Préfixes 0x00 (feuille) et 0x01 (nœud) : une feuille ne peut pas se faire passer pour un
nœud interne. Un nœud sans voisin remonte tel quel au niveau supérieur (il n'est pas
dupliqué, ce qui évite qu'une liste [a, b, c] et [a, b, c, c] aient la même racine).
"""

from __future__ import annotations

import hashlib
from typing import Any

from gold_agents.blockchain.canonical import canonical_json

EMPTY_ROOT = hashlib.sha256(b"").hexdigest()


def leaf_hash(event: Any) -> bytes:
    return hashlib.sha256(b"\x00" + canonical_json(event).encode("utf-8")).digest()


def node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + left + right).digest()


def _next_level(level: list[bytes]) -> list[bytes]:
    pairs = [node_hash(level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)]
    return pairs + ([level[-1]] if len(level) % 2 else [])


def merkle_root(events: list[Any]) -> str:
    """Racine (hexadécimal) de l'arbre des événements, dans leur ordre."""
    if not events:
        return EMPTY_ROOT
    level = [leaf_hash(e) for e in events]
    while len(level) > 1:
        level = _next_level(level)
    return level[0].hex()


def merkle_proof(events: list[Any], position: int) -> list[tuple[str, str]]:
    """Preuve d'inclusion de l'événement `position` : voisins ("L" ou "R", empreinte).

    Avec la preuve et la racine, on vérifie qu'un événement est dans le bloc sans
    recevoir les autres événements.
    """
    if not 0 <= position < len(events):
        raise IndexError(f"événement {position} absent ({len(events)} événements)")
    level = [leaf_hash(e) for e in events]
    proof: list[tuple[str, str]] = []
    while len(level) > 1:
        sibling = position ^ 1
        if sibling < len(level):
            proof.append(("L" if sibling < position else "R", level[sibling].hex()))
        level = _next_level(level)
        position //= 2
    return proof


def verify_proof(event: Any, proof: list[tuple[str, str]], root: str) -> bool:
    h = leaf_hash(event)
    for side, sibling in proof:
        other = bytes.fromhex(sibling)
        h = node_hash(other, h) if side == "L" else node_hash(h, other)
    return h.hex() == root
