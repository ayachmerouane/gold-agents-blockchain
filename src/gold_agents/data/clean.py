"""Nettoyage de la série XAU/USD quotidienne (§4.2, étape `clean`).

Décisions issues de l'EDA (docs/constats_eda.md, notebook 01_eda_nettoyage) :

- les rendements extrêmes sont de vrais événements : on les garde ;
- les bougies OHLC incohérentes (aucune à ce jour) sont élargies et marquées `ohlc_fixed` ;
- les trous anormaux du CSV Kaggle (plus de `min_missing` jours ouvrés d'affilée) sont
  comblés avec Twelve Data, même instrument XAU/USD. Les barres récupérées sont figées dans
  un fichier séparé (`data.gapfill_path`, empreinte dans CHECKSUMS.txt) pour que le
  nettoyage soit reproductible sans réseau. Elles gardent `source = "twelvedata"` ;
- un rendement qui traverse un trou encore ouvert n'est pas un rendement d'un jour :
  `valid_return = False`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from gold_agents.config import SplitsConfig
from gold_agents.data.load import COLUMNS, normalize_concat
from gold_agents.data.splits import assign_split
from gold_agents.data.update import (
    CoherenceReport,
    Fetcher,
    check_coherence,
    sha256_file,
    trading_days,
)

# Un jour férié isolé fait 1 jour ouvré manquant, un pont 2 ; au-delà, c'est un trou d'export.
DEFAULT_MIN_MISSING = 3


@dataclass(frozen=True)
class Gap:
    """`missing` jours ouvrés absents entre deux dates présentes `after` et `before`."""

    after: date
    before: date
    missing: int


def missing_bdays_before(dates: pd.Series) -> pd.Series:
    """Nombre de jours ouvrés absents entre chaque date et la précédente (0 pour la première)."""
    days = pd.DatetimeIndex(dates).tz_localize(None).to_numpy().astype("datetime64[D]")
    out = np.zeros(len(days), dtype=int)
    out[1:] = np.busday_count(days[:-1] + 1, days[1:])
    return pd.Series(out, index=dates.index)


def find_gaps(df: pd.DataFrame, min_missing: int = DEFAULT_MIN_MISSING) -> list[Gap]:
    """Trous de plus de `min_missing` jours ouvrés consécutifs, dans l'ordre chronologique."""
    missing = missing_bdays_before(df["date"])
    days = df["date"].dt.date.tolist()
    return [
        Gap(days[i - 1], days[i], int(missing.iloc[i]))
        for i in range(1, len(df))
        if missing.iloc[i] > min_missing
    ]


# --- Comblement des trous ---------------------------------------------------


@dataclass(frozen=True)
class GapFill:
    gap: Gap
    n_filled: int
    report: CoherenceReport


class GapFillError(RuntimeError):
    pass


def fetch_gap_bars(
    raw: pd.DataFrame,
    fetch: Fetcher,
    *,
    min_missing: int = DEFAULT_MIN_MISSING,
    context_days: int = 45,
    force: bool = False,
) -> tuple[pd.DataFrame, list[GapFill]]:
    """Télécharge les jours ouvrés absents de chaque trou anormal de `raw`.

    Chaque requête couvre aussi `context_days` jours de part et d'autre du trou : ces
    jours communs servent au contrôle de cohérence avec Kaggle (même contrôle que
    `update-data`). Si un contrôle échoue, rien n'est renvoyé (sauf `force`).
    """
    present = set(raw["date"])
    frames, fills = [], []
    for gap in find_gaps(raw, min_missing):
        bars = trading_days(
            fetch(
                gap.after - timedelta(days=context_days), gap.before + timedelta(days=context_days)
            )
        )
        report = check_coherence(raw, bars)
        if not report.ok and not force:
            raise GapFillError(
                f"trou {gap.after} -> {gap.before} : sources incohérentes ({report.describe()})"
            )
        inside = bars["date"].dt.date
        new = bars[(inside > gap.after) & (inside < gap.before) & ~bars["date"].isin(present)]
        frames.append(new)
        fills.append(GapFill(gap, len(new), report))
    return normalize_concat(frames), fills


def write_gapfill(bars: pd.DataFrame, path: Path) -> None:
    out = bars[COLUMNS].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


# --- Nettoyage ---------------------------------------------------------------


def fix_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Élargit high/low pour contenir open et close ; marque les lignes corrigées."""
    out = df.copy()
    top = out[["open", "close"]].max(axis=1)
    bottom = out[["open", "close"]].min(axis=1)
    out["ohlc_fixed"] = (out["high"] < top) | (out["low"] > bottom)
    out["high"] = out["high"].clip(lower=top)
    out["low"] = out["low"].clip(upper=bottom)
    return out


def clean(
    raw: pd.DataFrame,
    splits: SplitsConfig,
    gapfill: pd.DataFrame | None = None,
    *,
    min_missing: int = DEFAULT_MIN_MISSING,
) -> pd.DataFrame:
    """Série Kaggle nettoyée, éventuellement comblée, avec ses colonnes de contrôle.

    Colonnes ajoutées : `filled` (ligne venant du comblement), `ohlc_fixed`,
    `missing_bdays_before`, `valid_return`, `log_return`, `split`.
    Les lignes de comblement ne remplacent jamais une date Kaggle existante.
    """
    frames = [raw.assign(filled=False)]
    if gapfill is not None and not gapfill.empty:
        extra = gapfill[~gapfill["date"].isin(raw["date"])]
        extra = extra[(extra["date"] > raw["date"].min()) & (extra["date"] < raw["date"].max())]
        frames.append(trading_days(extra).assign(filled=True))
    out = fix_ohlc(normalize_concat(frames))
    out["filled"] = out["filled"].astype(bool)
    out["missing_bdays_before"] = missing_bdays_before(out["date"])
    out["valid_return"] = out["missing_bdays_before"] <= min_missing
    out.loc[out.index[0], "valid_return"] = False  # pas de veille pour le premier jour
    out["log_return"] = np.log(out["close"]).diff()
    out["split"] = assign_split(out["date"], splits).to_numpy()
    return out.reset_index(drop=True)


def check_clean(df: pd.DataFrame, min_missing: int = DEFAULT_MIN_MISSING) -> list[str]:
    """Liste des règles violées par un tableau nettoyé (vide si tout va bien)."""
    prices = df[["open", "high", "low", "close"]]
    rules = {
        "dates non triées": not df["date"].is_monotonic_increasing,
        "dates en double": not df["date"].is_unique,
        "prix manquants": bool(prices.isna().any().any()),
        "prix négatifs ou nuls": bool((prices <= 0).any().any()),
        "high sous open/close": bool((df["high"] < prices[["open", "close"]].max(axis=1)).any()),
        "low au-dessus de open/close": bool(
            (df["low"] > prices[["open", "close"]].min(axis=1)).any()
        ),
        "barre de week-end": bool((df["date"].dt.day_of_week >= 5).any()),
        "trou non marqué": bool(
            ((df["missing_bdays_before"] > min_missing) & df["valid_return"]).any()
        ),
        "ligne hors train/val/test": bool(df["split"].isna().any()),
    }
    return [name for name, broken in rules.items() if broken]


def write_clean(df: pd.DataFrame, out_dir: Path, raw_path: Path, gapfill_path: Path) -> Path:
    """Écrit `xau_1d_clean.parquet` et sa fiche JSON (empreintes des sources, comptes)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "xau_1d_clean.parquet"
    df.to_parquet(path, index=False)
    meta = {
        "raw": {"path": str(raw_path), "sha256": sha256_file(raw_path)},
        "gapfill": (
            {"path": str(gapfill_path), "sha256": sha256_file(gapfill_path)}
            if gapfill_path.exists() and df["filled"].any()
            else None
        ),
        "rows": len(df),
        "filled_rows": int(df["filled"].sum()),
        "ohlc_fixed_rows": int(df["ohlc_fixed"].sum()),
        "invalid_returns": int((~df["valid_return"]).sum()),
        "first_date": df["date"].min().strftime("%Y-%m-%d"),
        "last_date": df["date"].max().strftime("%Y-%m-%d"),
        "rows_per_split": {str(k): int(v) for k, v in df["split"].value_counts().items()},
    }
    (out_dir / "xau_1d_clean.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path
