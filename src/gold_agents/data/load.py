"""Lecture des fichiers bruts XAU/USD vers un schéma unique (§4.2, étape `load`).

Schéma commun à toutes les sources :
`date` (UTC, minuit), `open`, `high`, `low`, `close`, `volume`, `source`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = ["date", "open", "high", "low", "close", "volume", "source"]
PRICE_COLUMNS = ["open", "high", "low", "close"]

# Format des dates exportées par MetaTrader 4 : « 2004.06.11 00:00 ».
_MT4_DATE_FORMAT = "%Y.%m.%d %H:%M"


def _detect_separator(path: Path) -> str:
    with path.open(encoding="utf-8-sig") as f:
        header = f.readline()
    return ";" if header.count(";") > header.count(",") else ","


def _parse_dates(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip()
    parsed = pd.to_datetime(text, format=_MT4_DATE_FORMAT, errors="coerce")
    if parsed.isna().any():
        # Autres variantes (« 2004.06.11 », ISO…) : repli sur le parseur générique.
        fallback = pd.to_datetime(text.str.replace(".", "-", regex=False), errors="coerce")
        parsed = parsed.fillna(fallback)
    if parsed.isna().any():
        bad = text[parsed.isna()].head(3).tolist()
        raise ValueError(f"dates illisibles dans le CSV : {bad}")
    # Barres quotidiennes : on garde le jour calendaire, exprimé en UTC.
    return parsed.dt.normalize().dt.tz_localize("UTC")


def normalize(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Met un tableau OHLC(V) au schéma commun, trié par date, sans doublon de date."""
    out = df.rename(columns=str.lower).copy()
    missing = [c for c in ["date", *PRICE_COLUMNS] if c not in out.columns]
    if missing:
        raise ValueError(f"colonnes manquantes : {missing}")
    if "volume" not in out.columns:
        out["volume"] = float("nan")
    for col in [*PRICE_COLUMNS, "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    out["source"] = source
    out = out[COLUMNS].sort_values("date", kind="stable")
    return out.drop_duplicates("date", keep="last").reset_index(drop=True)


def read_kaggle_csv(path: Path | str) -> pd.DataFrame:
    """Lit le CSV Kaggle « XAUUSD Gold Price Tracker » (export MT4, séparateur `;` ou `,`)."""
    path = Path(path)
    raw = pd.read_csv(path, sep=_detect_separator(path), encoding="utf-8-sig")
    raw.columns = [str(c).strip() for c in raw.columns]
    raw = raw.rename(columns=str.lower)
    if "date" not in raw.columns:
        raise ValueError(f"colonne `Date` absente de {path.name} : {list(raw.columns)}")
    raw["date"] = _parse_dates(raw["date"])
    return normalize(raw, source="kaggle")


def read_update_csv(path: Path | str) -> pd.DataFrame:
    """Relit le fichier de mise à jour écrit par `data.update` (schéma commun, dates ISO)."""
    raw = pd.read_csv(path)
    raw["date"] = pd.to_datetime(raw["date"], utc=True).dt.normalize()
    sources = raw["source"].astype(str)
    frames = [normalize(part, src) for src, part in raw.groupby(sources, sort=False)]
    return normalize_concat(frames)


def normalize_concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatène des tableaux au schéma commun ; à date égale, le dernier gagne."""
    if not frames:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in COLUMNS})
    out = pd.concat(frames, ignore_index=True).sort_values("date", kind="stable")
    return out.drop_duplicates("date", keep="last").reset_index(drop=True)


def load_combined(raw_path: Path | str, update_path: Path | str) -> pd.DataFrame:
    """Historique Kaggle figé + jours ajoutés par l'API (pour le tableau de bord, pas l'évaluation).

    À date égale, la ligne Kaggle est conservée : la mise à jour ne fait que prolonger la série.
    """
    raw = read_kaggle_csv(raw_path)
    update_path = Path(update_path)
    if not update_path.exists():
        return raw
    update = read_update_csv(update_path)
    update = update[update["date"] > raw["date"].max()]
    return normalize_concat([raw, update])
