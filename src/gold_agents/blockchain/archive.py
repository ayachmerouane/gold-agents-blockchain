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


def read_moves(path: Path = MOVES_FILE) -> list[dict[str, str]]:
    """Les déplacements, dans l'ordre : ancien chemin, nouveau chemin, date."""
    if not path.is_file():
        return []
    moves: list[dict[str, str]] = json.loads(path.read_text(encoding="utf-8"))
    return moves


def record_move(old: Path, new: Path, path: Path = MOVES_FILE) -> None:
    moves = read_moves(path)
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


def resolve(file: str, moves: list[dict[str, str]], anchored_at: str | None = None) -> str:
    """Emplacement actuel du fichier ancré le `anchored_at`.

    Seuls comptent les déplacements faits APRÈS l'ancrage : un nouveau registre écrit
    plus tard au même chemin (tentative 2) n'est pas celui qui a été ancré avant.
    """
    for move in moves:  # dans l'ordre chronologique : suit les déplacements successifs
        if move["from"] == file and (anchored_at is None or move["moved_at"] > anchored_at):
            file = move["to"]
    return file
