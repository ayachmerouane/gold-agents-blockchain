"""Ancrage public des registres sur Sepolia, le réseau de test d'Ethereum (§8.1, option C).

On publie dans une transaction une empreinte qui résume le dernier hash de chaque registre.
La transaction est datée par la blockchain publique et visible sur Etherscan : elle prouve
que les registres existaient déjà, tels quels, à cette date.

L'empreinte est le SHA-256 du JSON canonique de la liste triée
`[{"file": ..., "blocks": ..., "head": ...}]`. Les données de la transaction sont le texte
`gold-agents ledger sha256:<empreinte>` (lisible sur Etherscan, « View input as UTF-8 »).
La transaction envoie 0 ETH de l'adresse vers elle-même. Seul Sepolia est accepté.

Le portefeuille vient du fichier `.env` (jamais versionné) : `SEPOLIA_RPC_URL` et
`SEPOLIA_PRIVATE_KEY`, la clé privée d'un compte de test sans vrai argent.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gold_agents.blockchain.canonical import canonical_json, sha256_hex

SEPOLIA_CHAIN_ID = 11155111
PAYLOAD_PREFIX = "gold-agents ledger sha256:"
ANCHORS_PATH = Path("ledger/anchors.json")
ENV_KEYS = ("SEPOLIA_RPC_URL", "SEPOLIA_PRIVATE_KEY")
GAS_LIMIT = 60_000  # 21 000 + les octets de données, avec de la marge


@dataclass(frozen=True)
class LedgerHead:
    file: str
    blocks: int
    head: str


@dataclass(frozen=True)
class Anchor:
    anchored_at: str
    network: str
    chain_id: int
    tx_hash: str
    block_number: int
    sender: str
    digest: str
    ledgers: list[LedgerHead]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def anchor_digest(heads: list[LedgerHead]) -> str:
    entries = sorted((asdict(h) for h in heads), key=lambda e: e["file"])
    return sha256_hex(canonical_json(entries))


def payload(digest: str) -> bytes:
    return (PAYLOAD_PREFIX + digest).encode("ascii")


def read_env(path: Path) -> dict[str, str]:
    """Lit `CLE=valeur` dans `.env` ; les variables d'environnement l'emportent."""
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip("\"'")
    for key in ENV_KEYS:
        # Un secret collé garde souvent un retour à la ligne final : on l'ignore.
        if os.environ.get(key, "").strip():
            values[key] = os.environ[key].strip().strip("\"'")
    missing = [k for k in ENV_KEYS if not values.get(k)]
    if missing:
        raise KeyError(f"{', '.join(missing)} absent de {path} et de l'environnement")
    return values


def build_transaction(
    sender: str, digest: str, nonce: int, max_fee: int, priority_fee: int
) -> dict[str, Any]:
    """Transaction EIP-1559 de 0 ETH vers soi-même, l'empreinte en données."""
    return {
        "type": 2,
        "chainId": SEPOLIA_CHAIN_ID,
        "from": sender,
        "to": sender,
        "value": 0,
        "data": "0x" + payload(digest).hex(),
        "nonce": nonce,
        "gas": GAS_LIMIT,
        "maxFeePerGas": max_fee,
        "maxPriorityFeePerGas": priority_fee,
    }


def send_anchor(heads: list[LedgerHead], rpc_url: str, private_key: str) -> Anchor:
    """Envoie la transaction et attend qu'elle soit dans un bloc (extra `anchor`)."""
    from web3 import Web3

    w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 60}))
    chain_id = w3.eth.chain_id
    if chain_id != SEPOLIA_CHAIN_ID:
        raise ValueError(f"le RPC est sur la chaîne {chain_id}, pas Sepolia ({SEPOLIA_CHAIN_ID})")
    account = w3.eth.account.from_key(private_key)
    digest = anchor_digest(heads)
    priority = w3.to_wei(2, "gwei")
    base_fee = int(w3.eth.get_block("latest")["baseFeePerGas"])
    tx = build_transaction(
        account.address,
        digest,
        w3.eth.get_transaction_count(account.address),
        2 * base_fee + priority,
        priority,
    )
    signed = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
    if receipt["status"] != 1:
        raise RuntimeError(f"transaction {tx_hash.to_0x_hex()} refusée par le réseau")
    return Anchor(
        anchored_at=datetime.now(UTC).isoformat(timespec="seconds"),
        network="sepolia",
        chain_id=chain_id,
        tx_hash=tx_hash.to_0x_hex(),
        block_number=int(receipt["blockNumber"]),
        sender=account.address,
        digest=digest,
        ledgers=heads,
    )


def append_anchor(anchor: Anchor, path: Path = ANCHORS_PATH) -> Path:
    anchors = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    anchors.append(anchor.to_dict())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(anchors, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def etherscan_url(tx_hash: str) -> str:
    return f"https://sepolia.etherscan.io/tx/{tx_hash}"
