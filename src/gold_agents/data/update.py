"""Données brutes figées (SHA-256) et mise à jour incrémentale par API.

- Le CSV Kaggle est figé : son empreinte est notée dans `CHECKSUMS.txt` et il n'est
  jamais réécrit sans `force`. Train/validation/test se construisent sur lui seul.
- Les jours postérieurs sont ajoutés dans un fichier séparé (`data.update_path`),
  après un contrôle de cohérence avec la série existante sur une période commune.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from gold_agents.data.load import COLUMNS, normalize_concat, read_kaggle_csv, read_update_csv

# (début, fin) inclus -> barres au schéma commun
Fetcher = Callable[[date, date], pd.DataFrame]

_CHECKSUMS_HEADER = (
    "# Empreintes SHA-256 des fichiers bruts (format : <sha256>  <nom_de_fichier>)\n"
)


# --- Empreintes -----------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_checksums(checksums_path: Path) -> dict[str, str]:
    if not checksums_path.exists():
        return {}
    entries: dict[str, str] = {}
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            digest, name = line.split(maxsplit=1)
            entries[Path(name.strip()).name] = digest
    return entries


def record_checksum(path: Path, checksums_path: Path) -> str:
    """Écrit (ou remplace) la ligne de `path` dans `CHECKSUMS.txt`, en gardant les commentaires."""
    digest = sha256_file(path)
    lines = (
        checksums_path.read_text(encoding="utf-8").splitlines(keepends=True)
        if checksums_path.exists()
        else [_CHECKSUMS_HEADER]
    )
    kept = [
        line
        for line in lines
        if line.startswith("#")
        or not line.strip()
        or Path(line.split(maxsplit=1)[-1].strip()).name != path.name
    ]
    if kept and not kept[-1].endswith("\n"):
        kept[-1] += "\n"
    kept.append(f"{digest}  {path.name}\n")
    checksums_path.write_text("".join(kept), encoding="utf-8")
    return digest


def verify_checksum(path: Path, checksums_path: Path) -> bool | None:
    """True/False si une empreinte est notée pour `path`, None sinon."""
    expected = read_checksums(checksums_path).get(path.name)
    return None if expected is None else expected == sha256_file(path)


# --- Cohérence entre sources ----------------------------------------------


@dataclass(frozen=True)
class CoherenceReport:
    n_overlap: int
    median_rel_diff: float
    max_rel_diff: float
    return_corr: float
    ok: bool

    def describe(self) -> str:
        return (
            f"{self.n_overlap} jours communs, écart relatif des clôtures médian "
            f"{self.median_rel_diff:.3%} (max {self.max_rel_diff:.3%}), "
            f"corrélation des rendements {self.return_corr:.3f}"
        )


class CoherenceError(RuntimeError):
    def __init__(self, report: CoherenceReport) -> None:
        super().__init__(f"sources incohérentes : {report.describe()}")
        self.report = report


def check_coherence(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    max_median_rel_diff: float = 0.02,
    min_return_corr: float = 0.8,
    min_overlap: int = 5,
) -> CoherenceReport:
    """Compare les clôtures et les rendements log de deux séries sur leurs dates communes.

    Les seuils par défaut tolèrent un écart de courtier (spot) ou de portage (futures),
    mais rejettent une autre unité, un autre instrument ou un décalage de dates.
    """
    both = reference[["date", "close"]].merge(
        candidate[["date", "close"]], on="date", suffixes=("_ref", "_new")
    )
    both = both.dropna().sort_values("date")
    n = len(both)
    if n == 0:
        return CoherenceReport(0, float("nan"), float("nan"), float("nan"), ok=False)
    rel = (both["close_new"] - both["close_ref"]).abs() / both["close_ref"]
    returns = both[["close_ref", "close_new"]].apply(np.log).diff().dropna()
    corr = float(returns["close_ref"].corr(returns["close_new"])) if len(returns) > 2 else np.nan
    ok = bool(
        n >= min_overlap
        and float(rel.median()) <= max_median_rel_diff
        and np.isfinite(corr)
        and corr >= min_return_corr
    )
    return CoherenceReport(n, float(rel.median()), float(rel.max()), corr, ok)


# --- Mise à jour ----------------------------------------------------------


@dataclass(frozen=True)
class UpdateResult:
    last_date_before: date
    n_new_rows: int
    report: CoherenceReport | None


def trading_days(df: pd.DataFrame) -> pd.DataFrame:
    """Retire les samedis et dimanches.

    En UTC, les API renvoient des barres de week-end (réouverture du dimanche soir,
    cotations résiduelles) que l'export MT4 de Kaggle n'a pas.
    """
    weekday: pd.Series = df["date"].dt.day_of_week
    return df.loc[weekday < 5]


def update_series(
    raw_path: Path,
    update_path: Path,
    fetch: Fetcher,
    *,
    today: date,
    overlap_days: int = 60,
    force: bool = False,
) -> UpdateResult:
    """Ajoute à `update_path` les jours clôturés manquants depuis la fin de la série connue.

    Seuls les jours ouvrés strictement antérieurs à `today` sont ajoutés (la barre du
    jour n'est pas encore close). La requête recouvre les `overlap_days` derniers jours
    connus pour vérifier la cohérence ; si le contrôle échoue, rien n'est écrit (sauf `force`).
    """
    raw = read_kaggle_csv(raw_path)
    existing = read_update_csv(update_path) if update_path.exists() else raw.iloc[0:0]
    n_before = len(existing)
    existing = trading_days(existing[existing["date"] > raw["date"].max()])
    existing = existing[existing["date"].dt.date < today]
    known = normalize_concat([raw, existing])
    last = known["date"].max().date()

    report: CoherenceReport | None = None
    new = existing.iloc[0:0]
    if last < today - timedelta(days=1):
        fetched = trading_days(fetch(last - timedelta(days=overlap_days), today))
        report = check_coherence(known, fetched)
        if not report.ok and not force:
            raise CoherenceError(report)
        days = fetched["date"].dt.date
        new = fetched[(days > last) & (days < today)]
    elif len(existing) == n_before:
        return UpdateResult(last, 0, None)

    out = normalize_concat([existing, new])[COLUMNS].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    update_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(update_path, index=False)
    return UpdateResult(last, len(new), report)
