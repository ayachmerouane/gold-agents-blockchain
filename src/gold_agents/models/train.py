"""Entraînement des réseaux M1/M2 (§5.3) : AdamW, gradient clipping, early stopping,
graines fixées, et recherche des hyperparamètres avec Optuna.

La cible est divisée par l'écart-type des rendements du train (`y_scale`) : le réseau
apprend des valeurs d'ordre 1 au lieu de 0,01, ce qui stabilise l'optimisation. Les
prévisions sont remises à l'échelle dans `predict`.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

from gold_agents.models.gru import GRURegressor
from gold_agents.models.lstm import LSTMRegressor
from gold_agents.models.recurrent import NetConfig, RecurrentRegressor

NETWORKS: dict[str, type[RecurrentRegressor]] = {"lstm": LSTMRegressor, "gru": GRURegressor}


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 1e-3
    batch_size: int = 64
    weight_decay: float = 1e-4
    loss: str = "huber"  # "mse" ou "huber" (plus robuste aux chocs de prix)
    max_epochs: int = 100
    patience: int = 15
    clip: float = 1.0  # norme maximale du gradient

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainResult:
    model: RecurrentRegressor
    best_epoch: int
    best_val_loss: float  # MSE de validation (cible mise à l'échelle) à la meilleure époque
    history: list[tuple[float, float]] = field(default_factory=list)  # MSE (train, val)


def set_seed(seed: int) -> None:
    """Même graine -> mêmes poids initiaux et même dropout (l'ordre des mini-lots a son
    propre générateur, lui aussi tiré de la graine)."""
    torch.manual_seed(seed)


def build_network(kind: str, net: NetConfig) -> RecurrentRegressor:
    return NETWORKS[kind](net)


def predict(model: nn.Module, x: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """Prévisions en mode évaluation (dropout coupé), dans l'échelle du réseau."""
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            out.append(model(torch.from_numpy(x[start : start + batch_size])).numpy())
    return np.concatenate(out) if out else np.empty(0, dtype="float32")


def _loss(name: str) -> nn.Module:
    if name == "mse":
        return nn.MSELoss()
    if name == "huber":
        return nn.HuberLoss(delta=1.0)
    raise ValueError(f"perte inconnue : {name}")


def fit(
    kind: str,
    net: NetConfig,
    cfg: TrainConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    on_epoch: Callable[[int, float], None] | None = None,
) -> TrainResult:
    """Entraîne un réseau et garde les poids de la meilleure époque en validation.

    `on_epoch(epoch, best_val_loss)` est appelé après chaque époque (Optuna s'en sert pour
    abandonner tôt les essais sans avenir).
    """
    set_seed(seed)
    model = build_network(kind, net)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = _loss(cfg.loss)
    xt, yt = torch.from_numpy(x_train), torch.from_numpy(y_train)
    generator = torch.Generator().manual_seed(seed)

    best_loss, best_epoch, wait = float("inf"), 0, 0
    best_state = copy.deepcopy(model.state_dict())
    history: list[tuple[float, float]] = []
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        order = torch.randperm(len(xt), generator=generator)
        for start in range(0, len(order), cfg.batch_size):
            batch = order[start : start + cfg.batch_size]
            optimizer.zero_grad()
            loss = criterion(model(xt[batch]), yt[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            optimizer.step()
        # Les deux courbes en MSE, dropout coupé : comparables quelle que soit la perte.
        train_loss = float(np.mean((predict(model, x_train) - y_train) ** 2))
        val_loss = float(np.mean((predict(model, x_val) - y_val) ** 2))
        history.append((train_loss, val_loss))
        if val_loss < best_loss:
            best_loss, best_epoch, wait = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
        if on_epoch is not None:
            on_epoch(epoch, best_loss)
        if wait >= cfg.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return TrainResult(model, best_epoch, best_loss, history)


def suggest(trial: Any, n_features: int, base: TrainConfig) -> tuple[NetConfig, TrainConfig]:
    """Espace de recherche Optuna : réseaux petits (peu de données, §5.2)."""
    net = NetConfig(
        n_features=n_features,
        hidden=trial.suggest_categorical("hidden", [32, 64]),
        layers=trial.suggest_int("layers", 1, 2),
        dropout=trial.suggest_float("dropout", 0.1, 0.4),
    )
    cfg = TrainConfig(
        lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        loss=trial.suggest_categorical("loss", ["mse", "huber"]),
        max_epochs=base.max_epochs,
        patience=base.patience,
        clip=base.clip,
    )
    return net, cfg


def from_params(
    params: dict[str, Any], n_features: int, base: TrainConfig
) -> tuple[NetConfig, TrainConfig]:
    """Reconstruit les configurations à partir des meilleurs paramètres Optuna."""
    net = NetConfig(n_features, params["hidden"], params["layers"], params["dropout"])
    cfg = TrainConfig(
        lr=params["lr"],
        batch_size=params["batch_size"],
        weight_decay=params["weight_decay"],
        loss=params["loss"],
        max_epochs=base.max_epochs,
        patience=base.patience,
        clip=base.clip,
    )
    return net, cfg
