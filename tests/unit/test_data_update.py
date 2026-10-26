"""Tests des empreintes SHA-256, du contrôle de cohérence et de la mise à jour incrémentale."""

import hashlib
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from gold_agents.cli import app
from gold_agents.data import update
from gold_agents.data.load import normalize, read_update_csv


def _series(start: str, closes: list[float], source: str) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(closes), tz="UTC")
    c = np.asarray(closes)
    return normalize(
        pd.DataFrame({"date": dates, "open": c, "high": c + 1, "low": c - 1, "close": c}), source
    )


def _walk(n: int, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    return list(2000 * np.exp(np.cumsum(rng.normal(0, 0.01, n))))


def _write_mt4(path: Path, df: pd.DataFrame) -> Path:
    lines = ["Date;Open;High;Low;Close;Volume"]
    for row in df.itertuples():
        d = row.date.strftime("%Y.%m.%d 00:00")
        lines.append(f"{d};{row.open};{row.high};{row.low};{row.close};100")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --- Empreintes ---


def test_record_checksum_replaces_entry_and_keeps_comments(tmp_path: Path) -> None:
    csv = tmp_path / "XAU_1d_data.csv"
    sums = tmp_path / "CHECKSUMS.txt"
    sums.write_text("# commentaire\n" + "0" * 64 + "  XAU_1d_data.csv\n")
    csv.write_bytes(b"abc")
    digest = update.record_checksum(csv, sums)
    assert digest == hashlib.sha256(b"abc").hexdigest()
    assert sums.read_text() == f"# commentaire\n{digest}  XAU_1d_data.csv\n"
    assert update.verify_checksum(csv, sums) is True
    csv.write_bytes(b"modifie")
    assert update.verify_checksum(csv, sums) is False
    assert update.verify_checksum(tmp_path / "autre.csv", sums) is None


# --- Cohérence ---


def test_coherence_accepts_small_broker_offset() -> None:
    closes = _walk(80)
    ref = _series("2026-01-01", closes, "kaggle")
    new = _series("2026-01-01", [c * 1.002 for c in closes], "twelvedata")
    report = update.check_coherence(ref, new)
    assert report.ok and report.n_overlap == 80
    assert report.median_rel_diff == pytest.approx(0.002)
    assert report.return_corr == pytest.approx(1.0)


def test_coherence_rejects_other_unit_or_shifted_dates() -> None:
    closes = _walk(80)
    ref = _series("2026-01-01", closes, "kaggle")
    per_gram = _series("2026-01-01", [c / 31.1035 for c in closes], "x")
    assert not update.check_coherence(ref, per_gram).ok
    shifted = _series("2026-01-01", closes[1:] + [closes[-1]], "x")
    assert not update.check_coherence(ref, shifted).ok


def test_coherence_without_overlap_fails() -> None:
    ref = _series("2026-01-01", _walk(10), "kaggle")
    new = _series("2027-01-01", _walk(10), "x")
    assert update.check_coherence(ref, new).n_overlap == 0
    assert not update.check_coherence(ref, new).ok


# --- Mise à jour incrémentale ---


@pytest.fixture
def market() -> pd.DataFrame:
    """Une « vraie » série de 200 jours ouvrés ; Kaggle n'en contient que les 150 premiers."""
    return _series("2026-01-01", _walk(200), "truth")


def _fetcher(market: pd.DataFrame, factor: float = 1.001) -> update.Fetcher:
    def fetch(start: date, end: date) -> pd.DataFrame:
        days = market["date"].dt.date
        part = market[(days >= start) & (days <= end)].copy()
        part[["open", "high", "low", "close"]] *= factor
        return normalize(part, "twelvedata")

    return fetch


def test_update_appends_only_new_days(tmp_path: Path, market: pd.DataFrame) -> None:
    raw = _write_mt4(tmp_path / "XAU_1d_data.csv", market.iloc[:150])
    raw_bytes = raw.read_bytes()
    out = tmp_path / "XAU_1d_update.csv"
    today = market["date"].iloc[-1].date()

    result = update.update_series(raw, out, _fetcher(market), today=today)

    assert result.n_new_rows == 50
    assert result.last_date_before == market["date"].iloc[149].date()
    assert result.report is not None and result.report.ok
    assert raw.read_bytes() == raw_bytes  # le brut Kaggle n'est jamais modifié
    written = read_update_csv(out)
    assert written["date"].min() > market["date"].iloc[149]
    assert (written["source"] == "twelvedata").all()

    # Deuxième passage le même jour : rien à ajouter, fichier inchangé.
    again = update.update_series(raw, out, _fetcher(market), today=today)
    assert again.n_new_rows == 0
    assert len(read_update_csv(out)) == 50


def test_update_continues_from_existing_update_file(tmp_path: Path, market: pd.DataFrame) -> None:
    raw = _write_mt4(tmp_path / "XAU_1d_data.csv", market.iloc[:150])
    out = tmp_path / "XAU_1d_update.csv"
    update.update_series(raw, out, _fetcher(market), today=market["date"].iloc[169].date())
    result = update.update_series(raw, out, _fetcher(market), today=market["date"].iloc[-1].date())
    assert result.last_date_before == market["date"].iloc[169].date()
    assert result.n_new_rows == 30
    assert len(read_update_csv(out)) == 50


def test_update_refuses_incoherent_source(tmp_path: Path, market: pd.DataFrame) -> None:
    raw = _write_mt4(tmp_path / "XAU_1d_data.csv", market.iloc[:150])
    out = tmp_path / "XAU_1d_update.csv"
    today = market["date"].iloc[-1].date()
    with pytest.raises(update.CoherenceError):
        update.update_series(raw, out, _fetcher(market, factor=1.5), today=today)
    assert not out.exists()
    forced = update.update_series(raw, out, _fetcher(market, factor=1.5), today=today, force=True)
    assert forced.n_new_rows == 50


# --- CLI ---


def test_cli_update_without_raw_file_explains(tmp_path: Path, config_path: Path) -> None:
    cfg = config_path.read_text(encoding="utf-8")
    cfg = cfg.replace("data/raw/XAU_1d_data.csv", str(tmp_path / "absent.csv"))
    custom = tmp_path / "config.yaml"
    custom.write_text(cfg, encoding="utf-8")
    result = CliRunner().invoke(app, ["update-data", "--config", str(custom)])
    assert result.exit_code == 1
    assert "make download" in result.output
