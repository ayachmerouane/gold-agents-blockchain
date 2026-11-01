"""Archivage d'un registre sans casser la vérification de son ancrage.

L'empreinte ancrée sur Sepolia couvre le nom de chaque fichier (`anchor.anchor_digest`).
Un registre archivé (tentative 1 du test, réglages figés v1) change de chemin : on note
l'ancien et le nouveau chemin dans `ledger/moves.json` (versionné), et la vérification
relit le fichier à son nouvel emplacement sous son nom d'origine. Le contenu, lui, ne
change pas : un seul octet modifié et l'ancrage ne correspond plus.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

MOVES_FILE = Path("ledger/moves.json")


def read_moves(path: Path = MOVES_FILE) -> dict[str, str]:
    """Ancien chemin -> nouveau chemin (chemins relatifs à la racine du dépôt)."""
    if not path.is_file():
        return {}
    return {m["from"]: m["to"] for m in json.loads(path.read_text(encoding="utf-8"))}


def record_move(old: Path, new: Path, path: Path = MOVES_FILE) -> None:
    moves = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    moves.append(
        {
            "from": old.as_posix(),
            "to": new.as_posix(),
            "moved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(moves, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def archive_file(old: Path, new: Path, moves_path: Path = MOVES_FILE) -> Path:
    """Déplace `old` vers `new` (jamais par-dessus un fichier existant) et le note."""
    if new.exists():
        raise FileExistsError(f"{new} existe déjà : archive non écrasée")
    new.parent.mkdir(parents=True, exist_ok=True)
    old.replace(new)
    record_move(old, new, moves_path)
    return new


def resolve(file: str, moves: dict[str, str]) -> str:
    """Emplacement actuel d'un fichier ancré (suit les déplacements successifs)."""
    seen = set()
    while file in moves and file not in seen:
        seen.add(file)
        file = moves[file]
    return file
