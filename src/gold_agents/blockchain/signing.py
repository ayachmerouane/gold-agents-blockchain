"""Signature Ed25519 des blocs (§8.2, v2).

L'agent d'exécution signe l'empreinte (hash) de chaque bloc avec une clé privée qui reste
sur le PC de l'auteur (`keys/`, jamais versionné). La clé publique est publiée dans
`ledger/public_key.pem` : n'importe qui peut vérifier les signatures, personne ne peut en
fabriquer sans la clé privée. Ed25519 est déterministe : même clé et même bloc donnent la
même signature, donc rejouer un backtest redonne un registre identique octet pour octet.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

PRIVATE_KEY_PATH = Path("keys/ledger_ed25519.key")
PUBLIC_KEY_PATH = Path("ledger/public_key.pem")
PREFIX = "ed25519:"


class Signer:
    """Détient la clé privée et signe des empreintes de blocs."""

    def __init__(self, private_key: Ed25519PrivateKey) -> None:
        self._key = private_key
        self.public_key = private_key.public_key()

    @classmethod
    def generate(cls) -> Signer:
        return cls(Ed25519PrivateKey.generate())

    @classmethod
    def from_seed(cls, seed: bytes) -> Signer:
        """Clé dérivée de 32 octets : seulement pour les tests et la démonstration."""
        return cls(Ed25519PrivateKey.from_private_bytes(seed))

    def sign(self, block_hash: str) -> str:
        return PREFIX + self._key.sign(bytes.fromhex(block_hash)).hex()

    def private_pem(self) -> bytes:
        return self._key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )


def public_pem(public_key: Ed25519PublicKey) -> bytes:
    return public_key.public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )


def fingerprint(public_key: Ed25519PublicKey) -> str:
    """Empreinte courte de la clé publique (SHA-256 des 32 octets bruts, 16 caractères)."""
    raw = public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return hashlib.sha256(raw).hexdigest()[:16]


def verify_signature(public_key: Ed25519PublicKey, block_hash: str, signature: str) -> bool:
    if not signature.startswith(PREFIX):
        return False
    try:
        public_key.verify(bytes.fromhex(signature[len(PREFIX) :]), bytes.fromhex(block_hash))
    except (InvalidSignature, ValueError):
        return False
    return True


def generate_keypair(
    private_path: Path = PRIVATE_KEY_PATH,
    public_path: Path = PUBLIC_KEY_PATH,
    overwrite: bool = False,
) -> Signer:
    """Crée la paire de clés. Refuse d'écraser une clé existante sauf `overwrite`."""
    if private_path.exists() and not overwrite:
        raise FileExistsError(f"{private_path} existe déjà : la clé n'a pas été remplacée")
    signer = Signer.generate()
    private_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_bytes(signer.private_pem())
    private_path.chmod(0o600)
    public_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.write_bytes(public_pem(signer.public_key))
    return signer


def load_signer(path: Path = PRIVATE_KEY_PATH) -> Signer:
    key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError(f"{path} n'est pas une clé privée Ed25519")
    return Signer(key)


def load_public_key(path: Path = PUBLIC_KEY_PATH) -> Ed25519PublicKey:
    key = serialization.load_pem_public_key(path.read_bytes())
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError(f"{path} n'est pas une clé publique Ed25519")
    return key
