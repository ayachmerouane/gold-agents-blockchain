"""Accès aux sources distantes : Kaggle (historique), Twelve Data (mise à jour), yfinance (secours).

Aucune clé n'est lue depuis le dépôt : uniquement des variables d'environnement
ou `~/.kaggle/kaggle.json`. Les appels réseau passent par un `HttpGet` injectable
pour que les tests tournent sans réseau.
"""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable, Mapping
from datetime import date
from pathlib import Path

import pandas as pd

from gold_agents.data.load import normalize

KAGGLE_DATASET = "novandraanugrah/xauusd-gold-price-historical-data-2004present"
KAGGLE_DAILY_FILE = "XAU_1d_data.csv"
TWELVEDATA_SYMBOL = "XAU/USD"
TWELVEDATA_KEY_ENV = "TWELVEDATA_API_KEY"
YFINANCE_SYMBOL = "GC=F"

_KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/{dataset}"
_TWELVEDATA_URL = "https://api.twelvedata.com/time_series"
_TWELVEDATA_MAX_BARS = 5000

# (url, en-têtes) -> corps de la réponse
HttpGet = Callable[[str, Mapping[str, str]], bytes]


class SourceError(RuntimeError):
    """Erreur d'accès ou de format d'une source distante."""


def http_get(url: str, headers: Mapping[str, str]) -> bytes:
    request = urllib.request.Request(url, headers=dict(headers))
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body: bytes = response.read()
            return body
    except urllib.error.HTTPError as exc:
        raise SourceError(f"HTTP {exc.code} sur {url.split('?')[0]}") from exc
    except urllib.error.URLError as exc:
        raise SourceError(f"réseau indisponible ({exc.reason})") from exc


# --- Kaggle ---------------------------------------------------------------


def kaggle_auth_header(
    env: Mapping[str, str] | None = None, config_dir: Path | None = None
) -> dict[str, str]:
    """En-tête d'authentification Kaggle.

    Ordre : jeton récent (`KAGGLE_API_TOKEN`, puis fichier `access_token`), puis
    anciens identifiants (`KAGGLE_USERNAME` + `KAGGLE_KEY`, puis fichier `kaggle.json`).
    Les fichiers sont lus dans `$KAGGLE_CONFIG_DIR`, à défaut `~/.kaggle`.
    """
    env = os.environ if env is None else env
    base = config_dir or Path(env.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    token = env.get("KAGGLE_API_TOKEN", "").strip()
    token_path = base / "access_token"
    if not token and token_path.exists():
        token = token_path.read_text(encoding="utf-8").strip()
    if token:
        return {"Authorization": f"Bearer {token}"}

    username, key = env.get("KAGGLE_USERNAME"), env.get("KAGGLE_KEY")
    if not (username and key):
        creds_path = base / "kaggle.json"
        if not creds_path.exists():
            raise SourceError(
                "jeton Kaggle introuvable : définir KAGGLE_API_TOKEN ou l'enregistrer dans "
                f"{token_path} (Kaggle > Settings > API > Generate New Token)"
            )
        creds = json.loads(creds_path.read_text(encoding="utf-8"))
        username, key = creds.get("username"), creds.get("key")
        if not (username and key):
            raise SourceError(f"{creds_path} doit contenir `username` et `key`")
    basic = base64.b64encode(f"{username}:{key}".encode()).decode()
    return {"Authorization": f"Basic {basic}"}


def download_kaggle(
    dest_dir: Path,
    *,
    dataset: str = KAGGLE_DATASET,
    filename: str = KAGGLE_DAILY_FILE,
    get: HttpGet = http_get,
    headers: Mapping[str, str] | None = None,
) -> Path:
    """Télécharge l'archive du dataset et n'en extrait que `filename` dans `dest_dir`."""
    headers = kaggle_auth_header() if headers is None else headers
    body = get(_KAGGLE_URL.format(dataset=dataset), headers)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / filename
    try:
        archive = zipfile.ZipFile(io.BytesIO(body))
    except zipfile.BadZipFile:
        # Petit dataset mono-fichier : Kaggle peut renvoyer le CSV directement.
        target.write_bytes(body)
        return target
    with archive:
        members = [m for m in archive.namelist() if Path(m).name == filename]
        if not members:
            raise SourceError(f"{filename} absent de l'archive Kaggle : {archive.namelist()}")
        target.write_bytes(archive.read(members[0]))
    return target


# --- Twelve Data ----------------------------------------------------------


def _twelvedata_key(env: Mapping[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    key = env.get(TWELVEDATA_KEY_ENV)
    if not key:
        raise SourceError(
            f"variable d'environnement {TWELVEDATA_KEY_ENV} absente "
            "(clé gratuite sur https://twelvedata.com/register)"
        )
    return key


def fetch_twelvedata(
    start: date,
    end: date,
    *,
    symbol: str = TWELVEDATA_SYMBOL,
    api_key: str | None = None,
    get: HttpGet = http_get,
) -> pd.DataFrame:
    """Barres quotidiennes XAU/USD entre `start` et `end` inclus (fuseau UTC)."""
    params = {
        "symbol": symbol,
        "interval": "1day",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "timezone": "UTC",
        "outputsize": str(_TWELVEDATA_MAX_BARS),
        "order": "ASC",
        "format": "JSON",
        "apikey": api_key or _twelvedata_key(),
    }
    body = get(f"{_TWELVEDATA_URL}?{urllib.parse.urlencode(params)}", {})
    payload = json.loads(body)
    if payload.get("status") == "error":
        if "no data is available" in str(payload.get("message", "")).lower():
            payload = {"values": []}
        else:
            raise SourceError(f"Twelve Data : {payload.get('message', 'erreur inconnue')}")
    values = payload.get("values") or []
    if not values:
        return normalize(
            pd.DataFrame(columns=["date", "open", "high", "low", "close"]), "twelvedata"
        )
    df = pd.DataFrame(values).rename(columns={"datetime": "date"})
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.normalize()
    return normalize(df, source="twelvedata")


# --- yfinance (secours) ---------------------------------------------------


def fetch_yfinance(start: date, end: date, *, symbol: str = YFINANCE_SYMBOL) -> pd.DataFrame:
    """Secours sans clé : contrat à terme COMEX (≠ spot). Nécessite `uv sync --extra fetch`."""
    try:
        import yfinance as yf  # type: ignore[import-not-found,unused-ignore]
    except ImportError as exc:
        raise SourceError("yfinance non installé : `uv sync --extra fetch`") from exc
    # `end` est exclusif chez Yahoo.
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).date()
    raw = yf.download(
        symbol,
        start=start.isoformat(),
        end=end_exclusive.isoformat(),
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if raw is None or raw.empty:
        raise SourceError(f"yfinance n'a renvoyé aucune donnée pour {symbol}")
    if isinstance(raw.columns, pd.MultiIndex):
        # Colonnes (champ, ticker) : on ne garde que le champ.
        raw.columns = raw.columns.get_level_values(0)
    df = raw.reset_index().rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.normalize()
    return normalize(df, source="yfinance")
