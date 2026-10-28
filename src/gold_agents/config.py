"""Chargement et validation (pydantic) de `config/config.yaml`."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


class _Strict(BaseModel):
    # Refuse les clés inconnues : une faute de frappe dans le YAML doit échouer.
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(_Strict):
    raw_path: Path
    processed_dir: Path
    # Jours ajoutés par l'API après la fin du CSV Kaggle (jamais utilisés pour l'évaluation).
    update_path: Path = Path("data/raw/XAU_1d_update.csv")
    checksums_path: Path = Path("data/raw/CHECKSUMS.txt")
    # Jours comblant les trous du CSV Kaggle (Twelve Data, figés par `fill-gaps`).
    gapfill_path: Path = Path("data/raw/XAU_1d_gapfill.csv")


class SplitsConfig(_Strict):
    train_end: date
    val_start: date
    val_end: date
    test_start: date

    @model_validator(mode="after")
    def _chronological(self) -> SplitsConfig:
        if not (self.train_end < self.val_start <= self.val_end < self.test_start):
            raise ValueError("les ensembles train/val/test doivent être disjoints et ordonnés")
        return self


class FeaturesConfig(_Strict):
    window: int = Field(gt=0)
    sma_periods: list[int]
    ema_periods: list[int]
    rsi_period: int = Field(gt=0)
    atr_period: int = Field(gt=0)
    volatility_window: int = Field(gt=0)
    # Rendements passés : somme des rendements log des k derniers jours.
    return_lags: list[int] = Field(default=[1, 5, 10, 20])


class StrategyConfig(_Strict):
    theta_buy: float = Field(ge=0)
    theta_sell: float = Field(ge=0)


class RiskConfig(_Strict):
    max_position: float = Field(gt=0, le=1)
    stop_loss: float = Field(gt=0, lt=1)
    max_drawdown: float = Field(gt=0, lt=1)
    allow_short: bool = False


class SimulationConfig(_Strict):
    initial_capital: float = Field(gt=0)
    fee_rate: float = Field(ge=0, lt=1)
    slippage: float = Field(ge=0, lt=1)
    # Taux sans risque annuel pour le Sharpe et le Sortino (§7.4).
    risk_free_rate: float = Field(default=0.0, ge=0, lt=1)


class Config(_Strict):
    seed: int
    data: DataConfig
    splits: SplitsConfig
    features: FeaturesConfig
    strategy: StrategyConfig
    risk: RiskConfig
    simulation: SimulationConfig


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> Config:
    """Lit le YAML et renvoie une configuration validée."""
    with Path(path).open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)
