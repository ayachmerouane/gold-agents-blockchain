"""Métriques des prévisions de rendement (§5.4) : MAE, RMSE, précision directionnelle,
gain sur le modèle « rendement nul » et test de Diebold-Mariano.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y - pred)))


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def directional_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    """Part des jours où le signe prévu est le bon.

    Une prévision nulle n'annonce ni hausse ni baisse : ces jours sont ignorés. Si toutes
    les prévisions sont nulles (modèle B0), la mesure n'a pas de sens et vaut NaN.
    """
    called = pred != 0
    if not called.any():
        return float("nan")
    return float(np.mean(np.sign(pred[called]) == np.sign(y[called])))


def r2_vs_zero(y: np.ndarray, pred: np.ndarray) -> float:
    """1 - MSE / MSE du modèle « rendement nul » : > 0 si l'on bat B0, < 0 sinon."""
    return float(1 - np.mean((y - pred) ** 2) / np.mean(y**2))


@dataclass(frozen=True)
class DieboldMariano:
    statistic: float
    p_value: float  # bilatérale : l'écart de précision entre les deux modèles est-il réel ?


def diebold_mariano(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> DieboldMariano:
    """Test de Diebold-Mariano (perte quadratique, horizon 1 jour).

    Statistique < 0 : le modèle A a des erreurs plus petites que le modèle B.
    """
    d = (y - pred_a) ** 2 - (y - pred_b) ** 2
    sd = float(np.std(d, ddof=1))
    if sd == 0:
        return DieboldMariano(0.0, 1.0)
    stat = float(np.mean(d) / (sd / math.sqrt(len(d))))
    p_value = math.erfc(abs(stat) / math.sqrt(2))  # loi normale, bilatéral
    return DieboldMariano(stat, p_value)


def forecast_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Toutes les métriques d'une prévision, comparée à B0 (rendement nul)."""
    dm = diebold_mariano(y, pred, np.zeros_like(y))
    return {
        "mae": mae(y, pred),
        "rmse": rmse(y, pred),
        "direction": directional_accuracy(y, pred),
        "r2_vs_zero": r2_vs_zero(y, pred),
        "dm_vs_zero": dm.statistic,
        "dm_p_value": dm.p_value,
    }
