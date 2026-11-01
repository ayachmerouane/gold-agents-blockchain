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


BuyRule = Literal["absolute", "quantile"]


class StrategyConfig(_Strict):
    # Règle absolue : θ_buy est un rendement (0.001 = 0,10 %). Règle relative
    # (« quantile ») : θ_buy est un niveau de quantile (0.8 = acheter si r̂ dépasse 80 %
    # des prévisions des `quantile_window` jours précédents).
    buy_rule: BuyRule = "absolute"
    theta_buy: float = Field(ge=0)
    theta_sell: float = Field(ge=0)
    quantile_window: int = Field(default=250, gt=1)
    quantile_min_days: int = Field(default=60, gt=0)
    # Filtre de confiance (MC Dropout) : acheter seulement si r̂ / σ̂ dépasse ce seuil.
    # None : pas de filtre.
    min_confidence: float | None = Field(default=0.5, ge=0)
    # Modèle de l'agent de prévision retenu (phase 6).
    model: ForecastModel = "ensemble"

    @model_validator(mode="after")
    def _quantile(self) -> StrategyConfig:
        if self.buy_rule == "quantile" and not 0 < self.theta_buy < 1:
            raise ValueError("règle « quantile » : theta_buy doit être entre 0 et 1")
        if self.quantile_min_days > self.quantile_window:
            raise ValueError("quantile_min_days ne peut pas dépasser quantile_window")
        return self


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
    # Règle d'achat de la grille (θ_buy : rendements ou niveaux de quantile).
    buy_rule: BuyRule = "absolute"
    # Nombre de trades par an exigé dans chaque période (« ni 0, ni des milliers ») :
    # un réglage hors de ces bornes n'est retenu que si aucun autre ne les respecte.
    min_trades_per_year: float = Field(default=0.0, ge=0)
    max_trades_per_year: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> TuningConfig:
        for (a, b), (c, _) in zip(self.folds, self.folds[1:], strict=False):
            if not (a <= b < c):
                raise ValueError(
                    "les périodes du walk-forward doivent se suivre sans chevauchement"
                )
        if self.folds[-1][0] > self.folds[-1][1]:
            raise ValueError("période du walk-forward vide")
        if self.buy_rule == "quantile" and not all(0 < q < 1 for q in self.grid.theta_buy):
            raise ValueError("règle « quantile » : les θ_buy de la grille sont entre 0 et 1")
        return self


class LiveConfig(_Strict):
    """Simulation en direct (paper trading) sur les jours qui suivent la fin des données.

    La règle de décision n'est PAS ici : c'est celle de `strategy` et `risk`, scellée par
    `gold-agents freeze` (une seule règle gelée pour le test et le direct). Cette section
    dit seulement comment réentraîner les modèles figés du direct.
    """

    seeds: list[int] = Field(default=[0, 1, 2], min_length=1)  # graines par réseau
    early_stop_days: int = Field(default=250, gt=0)  # fin du passé : arrêt anticipé
    mc_seed: int = 0  # graine du MC Dropout (σ̂ reproductible)


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
    live: LiveConfig | None = None

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


_SECTIONS = {
    "strategy": (
        "buy_rule",
        "theta_buy",
        "theta_sell",
        "quantile_window",
        "quantile_min_days",
        "min_confidence",
        "model",
    ),
    "risk": ("stop_loss", "max_drawdown", "cooldown_days"),
}


def _yaml_value(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def apply_settings(path: Path | str, settings: dict[str, object]) -> list[str]:
    """Écrit un réglage dans les sections `strategy` et `risk` du YAML, ligne à ligne.

    Les autres lignes et commentaires restent tels quels ; une clé absente est ajoutée en
    tête de sa section. Le fichier est validé avant d'être écrit. Renvoie les clés changées.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    stamp = date.today().isoformat()
    changed: list[str] = []
    for section, keys in _SECTIONS.items():
        start = next(i for i, line in enumerate(lines) if line.rstrip() == f"{section}:")
        end = next(
            (i for i in range(start + 1, len(lines)) if lines[i] and not lines[i][0].isspace()),
            len(lines),
        )
        for key in keys:
            if key not in settings:
                continue
            new = f"  {key}: {_yaml_value(settings[key])}"
            found = [i for i in range(start + 1, end) if lines[i].startswith(f"  {key}:")]
            if found:
                old = lines[found[0]].split("#", 1)[0].split(":", 1)[1]
                if yaml.safe_load(old) == settings[key]:
                    continue
                lines[found[0]] = f"{new}  # tune --apply, {stamp}"
            else:
                lines.insert(start + 1, f"{new}  # tune --apply, {stamp}")
                end += 1
            changed.append(key)
    text = "\n".join(lines) + "\n"
    Config.model_validate(yaml.safe_load(text))
    path.write_text(text, encoding="utf-8")
    return changed
