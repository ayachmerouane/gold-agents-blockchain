"""Script autonome `scripts/verify_ledger.py` (§8.3) : même verdict que le paquet.

Le registre de démonstration `tests/golden/ledger/demo_v2.jsonl` est signé avec une clé de
test dérivée d'une graine publique (voir `demo_chain`) : il sert à vérifier que le format
ne change pas et que le script tourne sans le projet (job CI dédié).
Régénérer : `uv run python tests/ledger/test_verify_script.py`.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest

from gold_agents.blockchain.block import Block
from gold_agents.blockchain.chain import Chain, rewrite_from
from gold_agents.blockchain.signing import Signer, public_pem
from gold_agents.blockchain.storage import read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "verify_ledger.py"
GOLDEN = ROOT / "tests" / "golden" / "ledger"
DEMO, DEMO_KEY, FROZEN_V1 = (
    GOLDEN / "demo_v2.jsonl",
    GOLDEN / "demo_public_key.pem",
    GOLDEN / "reglages_figes_v1.jsonl",
)
DEMO_SIGNER = Signer.from_seed(bytes(range(32)))  # clé de DÉMONSTRATION, publique


def demo_chain() -> Chain:
    chain = Chain.start([{"type": "GENESIS", "note": "registre de démonstration"}], DEMO_SIGNER)
    for day, (r_hat, action) in enumerate([(0.004, "BUY"), (-0.002, "HOLD"), (-0.006, "SELL")]):
        chain.append(
            f"2021-03-0{day + 1}",
            [
                {"type": "FORECAST", "agent_id": "forecast-v1", "r_hat": r_hat, "sigma_hat": 0.01},
                {"type": "SIGNAL", "agent_id": "strategy-v1", "action": action},
                {"type": "ORDER", "agent_id": "risk-v1", "approved": True},
                {"type": "PORTFOLIO", "value": 10_000.0 + 25 * day},
            ],
        )
    return chain


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verify_ledger", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(*args: str | Path) -> subprocess.CompletedProcess[str]:
    # -I : Python isolé, comme sur la machine du professeur (pas de PYTHONPATH, pas de cwd).
    return subprocess.run(
        [sys.executable, "-I", str(SCRIPT), *map(str, args)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        # Pas à la racine : le script y lirait ledger/public_key.pem, la vraie clé de l'auteur.
        cwd=Path(__file__).parent,
        check=False,
    )


def test_script_does_not_import_the_project() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    modules = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in (
            node.names if isinstance(node, ast.Import) else [ast.alias(node.module or "")]
        )
    }
    assert modules <= {
        "__future__",
        "argparse",
        "hashlib",
        "io",
        "urllib",
        "datetime",
        "json",
        "sys",
        "pathlib",
        "typing",
        "cryptography",
    }


def test_golden_demo_is_reproducible(tmp_path: Path) -> None:
    assert write_jsonl(demo_chain(), tmp_path / "d.jsonl").read_bytes() == DEMO.read_bytes()
    # Comparaison sans les fins de ligne : Git peut les convertir en CRLF sous Windows.
    assert public_pem(DEMO_SIGNER.public_key).splitlines() == DEMO_KEY.read_bytes().splitlines()


def test_script_accepts_demo_and_frozen_settings() -> None:
    res = _run(DEMO, FROZEN_V1, "--public-key", DEMO_KEY)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "signatures vérifiées" in res.stdout and "v1, sans signature" in res.stdout
    head = read_jsonl(DEMO).head.hash
    assert _run(DEMO, "--public-key", DEMO_KEY, "--head", head).returncode == 0
    assert _run(DEMO, "--public-key", DEMO_KEY, "--head", "0" * 64).returncode == 1


def test_script_strict_refuses_v1_and_missing_key() -> None:
    assert _run(FROZEN_V1, "--public-key", DEMO_KEY, "--strict").returncode == 1
    assert _run(DEMO, "--strict").returncode == 1


def test_script_detects_tampering(tmp_path: Path) -> None:
    lines = DEMO.read_text(encoding="utf-8").splitlines()
    block = json.loads(lines[2])
    block["events"][1]["action"] = "BUY"
    lines[2] = json.dumps(block)
    bad = tmp_path / "bad.jsonl"
    bad.write_text("\n".join(lines) + "\n", encoding="utf-8")
    res = _run(bad, "--public-key", DEMO_KEY)
    assert res.returncode == 1 and "bloc 2 : événements modifiés" in res.stdout

    forged = rewrite_from(copy.deepcopy(read_jsonl(bad).blocks), 2, Signer.generate())
    path = write_jsonl(Chain(forged), tmp_path / "forged.jsonl")
    assert _run(path).returncode == 0  # sans clé publique, le faux passe...
    res = _run(path, "--public-key", DEMO_KEY)  # ... pas avec la clé publiée
    assert res.returncode == 1 and "bloc 2 : signature" in res.stdout


@pytest.mark.parametrize(
    "tamper",
    ["none", "event", "header", "reseal", "rewrite", "delete", "swap", "unsigned", "version"],
)
def test_script_and_package_agree(tamper: str) -> None:
    script = _load_script()
    blocks = copy.deepcopy(demo_chain().blocks)
    if tamper == "event":
        blocks[1].events[0]["r_hat"] = 0.5
    elif tamper == "header":
        blocks[1] = replace(blocks[1], sim_date="2020-01-01")
    elif tamper == "reseal":
        blocks[1].events[0]["r_hat"] = 0.5
        blocks[1] = blocks[1].sealed()
    elif tamper == "rewrite":
        blocks = rewrite_from(blocks, 1, Signer.generate())
    elif tamper == "delete":
        del blocks[1]
    elif tamper == "swap":
        blocks[1], blocks[2] = blocks[2], blocks[1]
    elif tamper == "unsigned":
        blocks = rewrite_from(blocks, 0, None)
    elif tamper == "version":
        blocks[2] = Block(2, blocks[2].sim_date, blocks[1].hash, []).sealed()  # bloc v1
    expected = Chain(blocks).verify(DEMO_SIGNER.public_key)
    try:
        script.verify_blocks([b.to_dict() for b in blocks], DEMO_SIGNER.public_key)
        got: tuple[bool, int | None] = (True, None)
    except script.Invalid as exc:
        got = (False, int(str(exc).split()[1]))
    assert got == (expected.valid, expected.bad_index)
    assert expected.valid == (tamper == "none")


if __name__ == "__main__":
    write_jsonl(demo_chain(), DEMO)
    DEMO_KEY.write_bytes(public_pem(DEMO_SIGNER.public_key))
    print(f"{DEMO} et {DEMO_KEY} régénérés.")
