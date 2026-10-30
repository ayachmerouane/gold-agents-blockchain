"""Chargement et validation (pydantic) de `config/config.yaml`."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

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


ForecastModel = Literal["lstm", "gru", "ensemble", "lightgbm"]


class StrategyConfig(_Strict):
    theta_buy: float = Field(ge=0)
    theta_sell: float = Field(ge=0)
    # Filtre de confiance (MC Dropout) : acheter seulement si r̂ / σ̂ dépasse ce seuil.
    # None : pas de filtre.
    min_confidence: float | None = Field(default=0.5, ge=0)
    # Modèle de l'agent de prévision retenu (phase 6).
    model: ForecastModel = "ensemble"


class RiskConfig(_Strict):
    max_position: float = Field(gt=0, le=1)
    stop_loss: float = Field(gt=0, lt=1)
    max_drawdown: float = Field(gt=0, lt=1)
    allow_short: bool = False
    # Reprise après le coupe-circuit : jours en cash avant de réautoriser les achats.
    cooldown_days: int | None = Field(default=None, gt=0)


class SimulationConfig(_Strict):
    initial_capital: float = Field(gt=0)
    fee_rate: float = Field(ge=0, lt=1)
    slippage: float = Field(ge=0, lt=1)
    # Taux sans risque annuel pour le Sharpe et le Sortino (§7.4).
    risk_free_rate: float = Field(default=0.0, ge=0, lt=1)


class ModelsConfig(_Strict):
    """Entraînement des modèles de prévision (§5.3). Les défauts tiennent sur un CPU."""

    seeds: list[int] = Field(default=[0, 1, 2, 3, 4], min_length=1)
    max_epochs: int = Field(default=100, gt=0)
    patience: int = Field(default=15, gt=0)  # early stopping sur la perte de validation
    optuna_trials: int = Field(default=20, gt=0)  # essais Optuna par réseau (LSTM, GRU)
    lgbm_trials: int = Field(default=30, gt=0)  # essais Optuna pour LightGBM
    mc_samples: int = Field(default=50, gt=0)  # passages MC Dropout (incertitude)
    direction_alert: float = Field(default=0.60, gt=0.5, lt=1)  # au-delà : fuite probable
    mlflow_uri: str = "sqlite:///mlruns/mlflow.db"
    threads: int = Field(default=0, ge=0)  # threads PyTorch (0 = choix de PyTorch)


class TuningGrid(_Strict):
    """Grille grossière des réglages (R4 : peu de valeurs par paramètre)."""

    theta_buy: list[float] = Field(min_length=1)
    theta_sell: list[float] = Field(min_length=1)
    min_confidence: list[float | None] = Field(min_length=1)  # None : sans filtre
    stop_loss: list[float] = Field(min_length=1)
    max_drawdown: list[float] = Field(min_length=1)
    models: list[ForecastModel] = Field(min_length=1)


class TuningConfig(_Strict):
    """Phase 6 : réglage des agents en walk-forward, avant le test (§4.3, §6.3)."""

    # Périodes jugées, successives ; chacune est prédite par des modèles entraînés sur
    # tout ce qui la précède (fenêtre croissante depuis le début des données).
    folds: list[tuple[date, date]] = Field(min_length=2)
    early_stop_days: int = Field(default=250, gt=0)  # fin du passé : arrêt anticipé
    seeds: list[int] = Field(default=[0, 1, 2], min_length=1)
    grid: TuningGrid
    costs_factors: list[float] = Field(default=[0.0, 1.0, 2.0, 5.0], min_length=1)
    random_runs: int = Field(default=200, gt=0)  # stratégies aléatoires par période

    @model_validator(mode="after")
    def _ordered(self) -> TuningConfig:
        for (a, b), (c, _) in zip(self.folds, self.folds[1:], strict=False):
            if not (a <= b < c):
                raise ValueError(
                    "les périodes du walk-forward doivent se suivre sans chevauchement"
                )
        if self.folds[-1][0] > self.folds[-1][1]:
            raise ValueError("période du walk-forward vide")
        return self


class Config(_Strict):
    seed: int
    data: DataConfig
    splits: SplitsConfig
    features: FeaturesConfig
    strategy: StrategyConfig
    risk: RiskConfig
    simulation: SimulationConfig
    models: ModelsConfig = ModelsConfig()
    tuning: TuningConfig | None = None

    @model_validator(mode="after")
    def _tuning_before_test(self) -> Config:
        if self.tuning is not None and self.tuning.folds[-1][1] >= self.splits.test_start:
            raise ValueError("le walk-forward ne doit jamais toucher le test")
        return self


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> Config:
    """Lit le YAML et renvoie une configuration validée."""
    with Path(path).open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)
