"""Les sept pages du tableau de bord s'affichent sans erreur sur les registres du test."""

from __future__ import annotations

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1", reason="extra dashboard absent")

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "dashboard" / "app.py"
PAGES = [
    "direct",
    "vue_ensemble",
    "previsions",
    "portefeuille",
    "decisions",
    "blockchain",
    "experiences",
]

pytestmark = pytest.mark.skipif(
    not (ROOT / "outputs" / "ledger" / "test" / "buy_and_hold.jsonl").is_file(),
    reason="registres du test absents",
)


def _app(page: str) -> object:
    at = st_testing.AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    at.switch_page(f"pages/{page}.py").run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page: str) -> None:
    at = _app(page)
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.title) == 1


def test_blockchain_buttons_verify_and_detect_tampering() -> None:
    at = _app("blockchain")
    at.button[0].click().run()  # vérifier tous les registres
    assert not at.exception
    assert any("intacts" in s.value for s in at.success)

    tamper = next(b for b in at.button if b.label.startswith("Falsifier"))
    tamper.click().run()
    assert any("Falsification détectée" in e.value for e in at.error)


@pytest.mark.parametrize("page", PAGES)
def test_no_file_names_or_commands_on_screen(page: str) -> None:
    at = _app(page)
    texts = [
        e.value
        for kind in ("markdown", "caption", "info", "title", "subheader")
        for e in getattr(at, kind)
    ]
    texts += [str(o) for box in at.selectbox for o in box.options]
    shown = "\n".join(str(t) for t in texts)
    for word in (".jsonl", "outputs/", "ledger/", "make ", "uv run", ".pem"):
        assert word not in shown, (page, word)
