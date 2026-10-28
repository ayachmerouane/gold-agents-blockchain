"""Lecture et écriture du registre au format JSON Lines (un bloc par ligne)."""

from __future__ import annotations

import json
from pathlib import Path

from gold_agents.blockchain.block import Block, canonical_json
from gold_agents.blockchain.chain import Chain


def write_jsonl(chain: Chain, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [canonical_json(block.to_dict()) for block in chain.blocks]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def read_jsonl(path: Path) -> Chain:
    text = path.read_text(encoding="utf-8")
    return Chain([Block.from_dict(json.loads(line)) for line in text.splitlines() if line])
