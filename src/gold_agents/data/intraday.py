"""Barres intraday XAU/USD (export MT4 du dataset Kaggle) : lecture et heure de New York.

Les fichiers MT4 sont datés à l'heure du SERVEUR du courtier, pas en UTC. Beaucoup de
courtiers alignent leur journée sur la clôture de New York (17 h) : leur heure est alors
celle de New York + 7 h, été comme hiver (« ny+7 »). Les séances ICT (Asie, Londres,
New York) sont définies à l'heure de New York : on convertit donc chaque barre.

Colonnes ajoutées : `date` (UTC), `ny` (heure de New York), `ny_hour` et `trading_day`,
le jour de cotation qui commence à 18 h à New York (la barre de 18 h appartient au
lendemain), comme la bougie journalière du courtier.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

NEW_YORK = "America/New_York"
_MT4_FORMAT = "%Y.%m.%d %H:%M"


def server_to_utc(local: pd.Series, server_tz: str) -> pd.Series:
    """Heures du serveur (naïves) -> UTC. `server_tz` : « ny+7 », « UTC » ou un fuseau IANA."""
    if server_tz == "ny+7":
        ny = (local - pd.Timedelta(hours=7)).dt.tz_localize(
            NEW_YORK, ambiguous="NaT", nonexistent="shift_forward"
        )
        return ny.dt.tz_convert("UTC")
    utc: pd.Series = local.dt.tz_localize(
        server_tz, ambiguous="NaT", nonexistent="shift_forward"
    ).dt.tz_convert("UTC")
    return utc


def add_session_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Ajoute `ny`, `ny_hour` et `trading_day` à un tableau dont `date` est en UTC."""
    out = frame.copy()
    ny = pd.to_datetime(out["date"], utc=True).dt.tz_convert(NEW_YORK)
    out["ny"] = ny
    out["ny_hour"] = ny.dt.hour.astype("int64")
    # Le jour de cotation commence à 18 h (New York) : on avance de 6 h puis on prend la date.
    out["trading_day"] = (ny + pd.Timedelta(hours=6)).dt.date
    return out


def read_mt4_intraday(path: Path | str, server_tz: str = "ny+7") -> pd.DataFrame:
    """Lit un CSV MT4 intraday (`Date;Open;High;Low;Close;Volume`), trié, sans doublon."""
    path = Path(path)
    with path.open(encoding="utf-8-sig") as f:
        header = f.readline()
    sep = ";" if header.count(";") > header.count(",") else ","
    raw = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    local = pd.to_datetime(raw["date"].astype(str).str.strip(), format=_MT4_FORMAT)
    out = pd.DataFrame(
        {
            "date": server_to_utc(local, server_tz),
            **{c: pd.to_numeric(raw[c], errors="coerce") for c in ["open", "high", "low", "close"]},
        }
    )
    out = out.dropna().sort_values("date", kind="stable")
    out = out.drop_duplicates("date", keep="last").reset_index(drop=True)
    bad = (out["high"] < out[["open", "close"]].max(axis=1)) | (
        out["low"] > out[["open", "close"]].min(axis=1)
    )
    out.loc[bad, "high"] = out.loc[bad, ["open", "high", "close"]].max(axis=1)
    out.loc[bad, "low"] = out.loc[bad, ["open", "low", "close"]].min(axis=1)
    return add_session_columns(out)


def daily_from_intraday(frame: pd.DataFrame) -> pd.DataFrame:
    """Bougies journalières reconstruites (jour de cotation de New York)."""
    g = frame.groupby("trading_day", sort=True)
    return pd.DataFrame(
        {
            "open": g["open"].first(),
            "high": g["high"].max(),
            "low": g["low"].min(),
            "close": g["close"].last(),
        }
    ).reset_index()
