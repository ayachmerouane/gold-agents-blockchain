"""Tests des sources distantes, sans réseau (réponses HTTP simulées)."""

import base64
import io
import json
import zipfile
from collections.abc import Mapping
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from gold_agents.data import sources


class FakeGet:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.calls: list[tuple[str, dict[str, str]]] = []

    def __call__(self, url: str, headers: Mapping[str, str]) -> bytes:
        self.calls.append((url, dict(headers)))
        return self.body


def _zip(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, text in files.items():
            zf.writestr(name, text)
    return buf.getvalue()


# --- Kaggle ---


def test_kaggle_auth_from_env_username_key() -> None:
    header = sources.kaggle_auth_header({"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"})
    assert header == {"Authorization": "Basic " + base64.b64encode(b"u:k").decode()}


def test_kaggle_auth_prefers_api_token() -> None:
    header = sources.kaggle_auth_header({"KAGGLE_API_TOKEN": "t", "KAGGLE_USERNAME": "u"})
    assert header == {"Authorization": "Bearer t"}


def test_kaggle_auth_from_kaggle_json(tmp_path: Path) -> None:
    (tmp_path / "kaggle.json").write_text(json.dumps({"username": "u", "key": "k"}))
    header = sources.kaggle_auth_header({}, config_dir=tmp_path)
    assert header["Authorization"].startswith("Basic ")


def test_kaggle_auth_missing_explains_how_to_fix(tmp_path: Path) -> None:
    with pytest.raises(sources.SourceError, match="KAGGLE_USERNAME"):
        sources.kaggle_auth_header({}, config_dir=tmp_path)


def test_download_kaggle_extracts_only_daily_file(tmp_path: Path) -> None:
    get = FakeGet(_zip({"XAU_1d_data.csv": "daily", "XAU_1h_data.csv": "hourly"}))
    path = sources.download_kaggle(tmp_path, get=get, headers={"Authorization": "x"})
    assert path == tmp_path / "XAU_1d_data.csv"
    assert path.read_text() == "daily"
    assert not (tmp_path / "XAU_1h_data.csv").exists()
    url, headers = get.calls[0]
    assert url.endswith(f"/datasets/download/{sources.KAGGLE_DATASET}")
    assert headers == {"Authorization": "x"}


def test_download_kaggle_missing_file_raises(tmp_path: Path) -> None:
    get = FakeGet(_zip({"other.csv": "x"}))
    with pytest.raises(sources.SourceError, match="absent de l'archive"):
        sources.download_kaggle(tmp_path, get=get, headers={})


# --- Twelve Data ---


def test_fetch_twelvedata_parses_values() -> None:
    payload = {
        "meta": {"symbol": "XAU/USD"},
        "values": [
            {
                "datetime": "2026-10-22",
                "open": "4100.1",
                "high": "4120",
                "low": "4090",
                "close": "4110.5",
            },
            {
                "datetime": "2026-10-23",
                "open": "4110.5",
                "high": "4150",
                "low": "4105",
                "close": "4142.0",
            },
        ],
        "status": "ok",
    }
    get = FakeGet(json.dumps(payload).encode())
    df = sources.fetch_twelvedata(date(2026, 10, 1), date(2026, 10, 26), api_key="KEY", get=get)
    assert df["close"].tolist() == [4110.5, 4142.0]
    assert df["date"].iloc[0] == pd.Timestamp("2026-10-22", tz="UTC")
    assert (df["source"] == "twelvedata").all()
    url = get.calls[0][0]
    assert "symbol=XAU%2FUSD" in url and "interval=1day" in url and "apikey=KEY" in url


def test_fetch_twelvedata_error_raises() -> None:
    get = FakeGet(json.dumps({"status": "error", "code": 401, "message": "bad key"}).encode())
    with pytest.raises(sources.SourceError, match="bad key"):
        sources.fetch_twelvedata(date(2026, 1, 1), date(2026, 1, 2), api_key="x", get=get)


def test_fetch_twelvedata_no_data_is_empty() -> None:
    body = {"status": "error", "message": "No data is available on the specified dates."}
    get = FakeGet(json.dumps(body).encode())
    df = sources.fetch_twelvedata(date(2026, 1, 1), date(2026, 1, 2), api_key="x", get=get)
    assert df.empty


def test_twelvedata_key_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(sources.TWELVEDATA_KEY_ENV, raising=False)
    with pytest.raises(sources.SourceError, match=sources.TWELVEDATA_KEY_ENV):
        sources.fetch_twelvedata(date(2026, 1, 1), date(2026, 1, 2), get=FakeGet(b"{}"))


# --- yfinance (module simulé) ---


def test_fetch_yfinance_flattens_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    idx = pd.DatetimeIndex(["2026-10-22", "2026-10-23"], name="Date")
    cols = pd.MultiIndex.from_product([["Open", "High", "Low", "Close", "Volume"], ["GC=F"]])
    frame = pd.DataFrame([[1, 2, 0.5, 1.5, 10], [1.5, 3, 1, 2.5, 20]], index=idx, columns=cols)
    calls: dict[str, object] = {}

    def download(symbol: str, **kwargs: object) -> pd.DataFrame:
        calls.update(symbol=symbol, **kwargs)
        return frame

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=download))
    df = sources.fetch_yfinance(date(2026, 10, 22), date(2026, 10, 23))
    assert df["close"].tolist() == [1.5, 2.5]
    assert (df["source"] == "yfinance").all()
    assert calls["symbol"] == "GC=F" and calls["end"] == "2026-10-24"
