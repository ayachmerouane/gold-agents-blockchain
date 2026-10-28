"""Fixtures partagées par tous les tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gold_agents.config import Config, load_config
from gold_agents.data.clean import clean
from gold_agents.data.load import normalize

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config_path() -> Path:
    return ROOT / "config" / "config.yaml"


@pytest.fixture(scope="session")
def cfg() -> Config:
    return load_config(ROOT / "config" / "config.yaml")


@pytest.fixture(scope="session")
def clean_series(cfg: Config) -> pd.DataFrame:
    """Série nettoyée synthétique (2004-2025) couvrant train, validation et test."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2004-06-11", "2025-12-31", tz="UTC")
    n = len(dates)
    close = 400 * np.exp(np.cumsum(rng.normal(0.0003, 0.011, n)))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, 0.005, n)) * close
    raw = pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
        }
    )
    # Un trou de 10 jours ouvrés non comblé dans le test : rendement invalide au retour.
    raw = raw[~raw["date"].isin(pd.bdate_range("2023-03-06", "2023-03-17", tz="UTC"))]
    return clean(normalize(raw, "kaggle"), cfg.splits)
