# /// script
# requires-python = ">=3.11"
# dependencies = ["cryptography>=42"]
# ///
"""Vérification autonome du registre (§8.3), pour le professeur.

Ne dépend pas du projet : seulement Python 3.11+ et `cryptography`. Toutes les règles sont
réécrites ici, indépendamment de `gold_agents.blockchain` : deux implémentations qui
donnent le même verdict se contrôlent l'une l'autre.

Utilisation, depuis la racine du dépôt :

    uv run scripts/verify_ledger.py                      # ledger/*.jsonl, clé ledger/public_key.pem
    uv run scripts/verify_ledger.py chemin/registre.jsonl --public-key cle.pem
    uv run scripts/verify_ledger.py registre.jsonl --head <hash attendu du dernier bloc>

(ou `pip install cryptography` puis `python scripts/verify_ledger.py ...`)

Contrôles, bloc par bloc : index, `prev_hash` égal au hash du bloc précédent, hash recalculé.
En v2 en plus : racine de Merkle recalculée depuis les événements, difficulté constante,
preuve de travail, signature Ed25519 avec la clé publique publiée. Code de sortie 0 si tout
est valide, 1 sinon.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any

GENESIS_PREV_HASH = "0" * 64
DEFAULT_DIRS = (Path("ledger"), Path("outputs/ledger"))
DEFAULT_PUBLIC_KEY = Path("ledger/public_key.pem")


class Invalid(Exception):
    def __init__(self, index: int, reason: str) -> None:
        super().__init__(f"bloc {index} : {reason}")


def canonical(data: Any) -> bytes:
    return json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def merkle_root(events: list[Any]) -> str:
    if not events:
        return hashlib.sha256(b"").hexdigest()
    level = [hashlib.sha256(b"\x00" + canonical(e)).digest() for e in events]
    while len(level) > 1:
        nxt = [
            hashlib.sha256(b"\x01" + level[i] + level[i + 1]).digest()
            for i in range(0, len(level) - 1, 2)
        ]
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    return level[0].hex()


def block_hash(block: dict[str, Any]) -> str:
    if block.get("version", 1) == 1:
        keys = ("index", "sim_date", "prev_hash", "events")
    else:
        keys = ("version", "index", "sim_date", "prev_hash", "merkle_root", "difficulty", "nonce")
    return hashlib.sha256(canonical({k: block[k] for k in keys})).hexdigest()


def load_public_key(path: Path) -> Any:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from cryptography.hazmat.primitives.serialization import load_pem_public_key

    key = load_pem_public_key(path.read_bytes())
    if not isinstance(key, Ed25519PublicKey):
        raise SystemExit(f"{path} n'est pas une clé publique Ed25519")
    return key


def signature_ok(public_key: Any, digest: str, signature: str | None) -> bool:
    from cryptography.exceptions import InvalidSignature

    if not signature or not signature.startswith("ed25519:"):
        return False
    try:
        public_key.verify(bytes.fromhex(signature[8:]), bytes.fromhex(digest))
    except (InvalidSignature, ValueError):
        return False
    return True


def verify_blocks(blocks: list[dict[str, Any]], public_key: Any) -> tuple[int, str]:
    """Lève `Invalid` au premier bloc fautif ; renvoie (version, état des signatures)."""
    if not blocks:
        raise Invalid(0, "registre vide")
    version = blocks[0].get("version", 1)
    difficulty = blocks[0].get("difficulty", 0)
    if version not in (1, 2):
        raise Invalid(0, f"version {version} inconnue")
    prev = GENESIS_PREV_HASH
    for i, b in enumerate(blocks):
        try:
            if b["index"] != i:
                raise Invalid(i, f"index {b['index']} au lieu de {i}")
            if b.get("version", 1) != version:
                raise Invalid(i, f"version {b.get('version', 1)} dans un registre v{version}")
            if b["prev_hash"] != prev:
                raise Invalid(i, "prev_hash différent du hash du bloc précédent")
            if version == 2:
                if b["merkle_root"] != merkle_root(b["events"]):
                    raise Invalid(i, "événements modifiés (racine de Merkle différente)")
                if b["difficulty"] != difficulty:
                    raise Invalid(i, f"difficulté {b['difficulty']} au lieu de {difficulty}")
            if block_hash(b) != b["hash"]:
                raise Invalid(i, "contenu modifié (hash recalculé différent)")
            if version == 2:
                if not b["hash"].startswith("0" * difficulty):
                    raise Invalid(i, "preuve de travail absente")
                if public_key is not None and not signature_ok(
                    public_key, b["hash"], b.get("signature")
                ):
                    raise Invalid(i, "signature absente ou invalide pour la clé publique")
        except (KeyError, TypeError, ValueError) as exc:
            raise Invalid(i, f"bloc mal formé ({exc!r})") from exc
        prev = b["hash"]
    if version == 1:
        return 1, "v1, sans signature (format des phases 3 à 6)"
    if public_key is not None:
        return 2, f"v2, difficulté {difficulty}, signatures vérifiées"
    return 2, f"v2, difficulté {difficulty}, signatures NON vérifiées (pas de clé publique)"


def read_blocks(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def find_ledgers(paths: list[Path]) -> list[Path]:
    if not paths:
        paths = [d for d in DEFAULT_DIRS if d.is_dir()]
    found: list[Path] = []
    for p in paths:
        found.extend(sorted(p.glob("*.jsonl")) if p.is_dir() else [p])
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vérifie l'intégrité des registres .jsonl.")
    parser.add_argument("ledgers", nargs="*", type=Path, help="fichiers ou dossiers")
    parser.add_argument("--public-key", type=Path, help="clé publique Ed25519 (PEM)")
    parser.add_argument("--head", help="hash attendu du dernier bloc (un seul registre)")
    parser.add_argument(
        "--strict", action="store_true", help="refuser les registres v1 ou non signés"
    )
    args = parser.parse_args(argv)
    # Sous Windows, une sortie redirigée prend l'encodage cp1252 : on force l'UTF-8.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")

    key_path = args.public_key or (DEFAULT_PUBLIC_KEY if DEFAULT_PUBLIC_KEY.exists() else None)
    public_key = load_public_key(key_path) if key_path else None
    if public_key is not None:
        print(f"Clé publique : {key_path}")
    elif args.strict:
        print("--strict demande une clé publique (--public-key).")
        return 1
    ledgers = find_ledgers(args.ledgers)
    if not ledgers:
        print("Aucun registre .jsonl trouvé.")
        return 1
    if args.head and len(ledgers) != 1:
        print("--head s'utilise avec un seul registre.")
        return 1

    ok = True
    for path in ledgers:
        try:
            blocks = read_blocks(path)
            version, detail = verify_blocks(blocks, public_key)
        except Invalid as exc:
            print(f"INVALIDE  {path} : {exc}")
            ok = False
            continue
        except (OSError, json.JSONDecodeError) as exc:
            print(f"ILLISIBLE {path} : {exc}")
            ok = False
            continue
        head = blocks[-1]["hash"]
        if args.head and head != args.head:
            print(f"INVALIDE  {path} : dernier bloc {head[:12]}… au lieu de {args.head[:12]}…")
            ok = False
            continue
        if args.strict and version == 1:
            print(f"REFUSÉ    {path} : registre v1 non signé (--strict)")
            ok = False
            continue
        print(f"valide    {path} : {len(blocks)} blocs, {detail}, dernier hash {head}")
    print("\nRésultat : tous les registres sont valides." if ok else "\nRésultat : ÉCHEC.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
