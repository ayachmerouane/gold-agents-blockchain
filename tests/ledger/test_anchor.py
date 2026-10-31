"""Ancrage Sepolia (bonus, §8.1 C) : empreinte, transaction, vérification par le script.

Aucun accès réseau : la transaction est signée hors ligne et le nœud RPC est simulé.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from typer.testing import CliRunner

from gold_agents.blockchain.anchor import (
    PAYLOAD_PREFIX,
    SEPOLIA_CHAIN_ID,
    Anchor,
    LedgerHead,
    anchor_digest,
    append_anchor,
    build_transaction,
    payload,
    read_env,
)
from gold_agents.blockchain.storage import read_jsonl
from gold_agents.cli import app

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "tests" / "golden" / "ledger" / "demo_v2.jsonl"
DEMO_KEY = ROOT / "tests" / "golden" / "ledger" / "demo_public_key.pem"
SENDER = "0x9Ed3500000000000000000000000000000efb2B0"
HEADS = [LedgerHead("b.jsonl", 3, "00ab"), LedgerHead("a.jsonl", 2, "00cd")]


def test_digest_ignores_order_and_covers_every_field() -> None:
    digest = anchor_digest(HEADS)
    assert digest == anchor_digest(HEADS[::-1]) and len(digest) == 64
    assert digest != anchor_digest([HEADS[0], LedgerHead("a.jsonl", 3, "00cd")])
    assert payload(digest).decode("ascii") == PAYLOAD_PREFIX + digest


def test_read_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("SEPOLIA_RPC_URL", "SEPOLIA_PRIVATE_KEY"):
        monkeypatch.delenv(key, raising=False)
    env = tmp_path / ".env"
    with pytest.raises(KeyError):
        read_env(env)
    env.write_text('# test\nSEPOLIA_RPC_URL="https://rpc"\nSEPOLIA_PRIVATE_KEY=0xabc\n')
    assert read_env(env) == {"SEPOLIA_RPC_URL": "https://rpc", "SEPOLIA_PRIVATE_KEY": "0xabc"}
    monkeypatch.setenv("SEPOLIA_RPC_URL", "https://autre")
    assert read_env(env)["SEPOLIA_RPC_URL"] == "https://autre"


def test_transaction_is_sepolia_only_and_signs_offline() -> None:
    eth_account = pytest.importorskip("eth_account")
    account = eth_account.Account.create()
    digest = anchor_digest(HEADS)
    tx = build_transaction(account.address, digest, 0, 10**10, 10**9)
    assert tx["chainId"] == SEPOLIA_CHAIN_ID and tx["value"] == 0 and tx["to"] == account.address
    assert bytes.fromhex(tx["data"][2:]) == payload(digest)
    tx.pop("from")
    assert len(account.sign_transaction(tx).raw_transaction) > 100


def _anchor_demo(tmp_path: Path, head: str | None = None) -> tuple[Path, dict[str, Any]]:
    chain = read_jsonl(DEMO)
    heads = [LedgerHead(DEMO.as_posix(), len(chain), head or chain.head.hash)]
    anchor = Anchor(
        "2026-10-31T20:00:00+00:00", "sepolia", SEPOLIA_CHAIN_ID, "0x" + "ab" * 32, 123,
        SENDER, anchor_digest(heads), heads,
    )  # fmt: skip
    path = append_anchor(anchor, tmp_path / "anchors.json")
    tx = {
        "blockNumber": hex(123),
        "from": SENDER.lower(),
        "input": "0x" + payload(anchor.digest).hex(),
    }
    return path, tx


def _fake_rpc(tx: dict[str, Any]) -> Any:
    def call(url: str, method: str, params: list[Any]) -> Any:
        return {
            "eth_chainId": hex(SEPOLIA_CHAIN_ID),
            "eth_getTransactionByHash": tx,
            "eth_getBlockByNumber": {"timestamp": hex(1_793_000_000)},
        }[method]

    return call


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "verify_ledger", ROOT / "scripts" / "verify_ledger.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _main(script: Any, anchors: Path, capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    code = script.main([str(DEMO), "--public-key", str(DEMO_KEY), "--anchors", str(anchors)])
    return code, capsys.readouterr().out


def test_script_checks_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _load_script()
    anchors, tx = _anchor_demo(tmp_path)
    monkeypatch.setattr(script, "rpc_call", _fake_rpc(tx))
    code, out = _main(script, anchors, capsys)
    assert code == 0 and "ancré (transaction 0xababababab" in out and "bloc Sepolia 123" in out

    # La transaction ne contient pas la même empreinte.
    monkeypatch.setattr(script, "rpc_call", _fake_rpc({**tx, "input": "0x00"}))
    code, out = _main(script, anchors, capsys)
    assert code == 1 and "ne contient pas cette empreinte" in out

    # Le registre a changé depuis l'ancrage (ou l'ancrage a été modifié après coup).
    anchors, tx = _anchor_demo(tmp_path / "autre", head="0" * 64)
    monkeypatch.setattr(script, "rpc_call", _fake_rpc(tx))
    code, out = _main(script, anchors, capsys)
    assert code == 1 and "différent de l'ancrage" in out
    data = json.loads(anchors.read_text(encoding="utf-8"))
    data[0]["ledgers"][0]["head"] = read_jsonl(DEMO).head.hash  # sans recalculer l'empreinte
    anchors.write_text(json.dumps(data), encoding="utf-8")
    code, out = _main(script, anchors, capsys)
    assert code == 1 and "empreinte recalculée différente" in out


def test_anchor_command_needs_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("SEPOLIA_RPC_URL", "SEPOLIA_PRIVATE_KEY"):
        monkeypatch.delenv(key, raising=False)
    args = ["anchor", str(DEMO), "--public-key", str(DEMO_KEY), "--env", str(tmp_path / ".env")]
    res = CliRunner().invoke(app, args)
    assert res.exit_code == 1 and "SEPOLIA_RPC_URL" in res.output


def test_script_tries_other_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _load_script()
    anchors, tx = _anchor_demo(tmp_path)
    good = _fake_rpc(tx)

    def first_refuses(url: str, method: str, params: list[Any]) -> Any:
        if url == script.DEFAULT_RPCS[0]:
            raise OSError("HTTP Error 403: Forbidden")
        return good(url, method, params)

    monkeypatch.setattr(script, "rpc_call", first_refuses)
    code, out = _main(script, anchors, capsys)
    assert code == 0 and "bloc Sepolia 123" in out

    def all_refuse(url: str, method: str, params: list[Any]) -> Any:
        raise OSError("HTTP Error 403: Forbidden")

    monkeypatch.setattr(script, "rpc_call", all_refuse)
    code, out = _main(script, anchors, capsys)
    assert code == 1 and "NON VÉRIFIÉ" in out and "--rpc" in out


def test_rpc_call_sends_a_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    script = _load_script()
    seen = {}

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"jsonrpc": "2.0", "id": 1, "result": "0xaa36a7"}'

    def urlopen(request: Any, timeout: float) -> Response:
        seen["agent"] = request.get_header("User-agent")
        return Response()

    monkeypatch.setattr(script.urllib.request, "urlopen", urlopen)
    assert script.rpc_call("https://rpc", "eth_chainId", []) == "0xaa36a7"
    assert seen["agent"] and "urllib" not in seen["agent"].lower()
