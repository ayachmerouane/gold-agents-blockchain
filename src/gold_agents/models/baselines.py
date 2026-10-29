"""Baselines B0 à B4 (§5.2) : la barre que les réseaux doivent franchir.

- B0 : rendement nul (marche aléatoire) ;
- B1 : même rendement qu'hier, et sa variante moyenne des 20 derniers jours ;
- B2 : ARIMA ajusté sur le train, prévisions à un jour avec des paramètres figés ;
- B3 : régression Ridge sur les features du jour t ;
- B4 : LightGBM sur les mêmes features, réglé avec Optuna.

Toutes renvoient la prévision de `target_return` (rendement de t à t+1) pour des lignes
du fichier de features, en n'utilisant que les jours ≤ t.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def zero(n: int) -> np.ndarray:
    """B0 : on prévoit toujours un rendement nul."""
    return np.zeros(n, dtype="float64")


def last_return(frame: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    """B1 : le rendement de demain est celui d'aujourd'hui."""
    out: np.ndarray = frame["ret_1d"].to_numpy(dtype="float64")[rows]
    return out


def mean_return(frame: pd.DataFrame, rows: np.ndarray, days: int = 20) -> np.ndarray:
    """B1 (variante) : rendement moyen des `days` derniers jours."""
    out: np.ndarray = frame[f"ret_{days}d"].to_numpy(dtype="float64")[rows] / days
    return out


# --- B2 : ARIMA ---------------------------------------------------------------


def daily_returns(frame: pd.DataFrame) -> np.ndarray:
    """Rendement du jour de chaque ligne ; 0 quand il manque (début, trou non comblé)."""
    out: np.ndarray = np.nan_to_num(frame["ret_1d"].to_numpy(dtype="float64"))
    return out


@dataclass(frozen=True)
class Arima:
    order: tuple[int, int, int]
    params: np.ndarray
    aic: float

    def predict(self, frame: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
        """Prévision du rendement de t+1 pour chaque ligne t, paramètres figés.

        Le filtre de Kalman refait un passage sur toute la série : la prévision à un pas
        de la position t+1 n'utilise que les rendements jusqu'à t.
        """
        from statsmodels.tsa.arima.model import ARIMA

        r = daily_returns(frame)
        model = ARIMA(r, order=self.order, trend="c")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = model.filter(self.params)
        one_step = np.asarray(res.predict(), dtype="float64")
        out: np.ndarray = one_step[rows + 1]
        return out


def fit_arima(frame: pd.DataFrame, max_p: int = 2, max_q: int = 2) -> Arima:
    """Choisit l'ordre (p, 0, q) au plus petit AIC sur les jours du train."""
    from statsmodels.tsa.arima.model import ARIMA

    r = daily_returns(frame[frame["split"] == "train"])
    best: Arima | None = None
    for p, q in itertools.product(range(max_p + 1), range(max_q + 1)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ARIMA(r, order=(p, 0, q), trend="c").fit()
        if best is None or res.aic < best.aic:
            best = Arima((p, 0, q), np.asarray(res.params), float(res.aic))
    assert best is not None
    return best


# --- B3 : Ridge -----------------------------------------------------------------

RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0)


def fit_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    alphas: tuple[float, ...] = RIDGE_ALPHAS,
) -> tuple[Any, float]:
    """Ridge dont la pénalité `alpha` minimise l'erreur quadratique en validation."""
    from sklearn.linear_model import Ridge

    scores = {}
    for alpha in alphas:
        model = Ridge(alpha=alpha).fit(x_train, y_train)
        scores[alpha] = float(np.mean((model.predict(x_val) - y_val) ** 2))
    alpha = min(scores, key=lambda a: scores[a])
    return Ridge(alpha=alpha).fit(x_train, y_train), alpha


# --- B4 : LightGBM --------------------------------------------------------------


def lightgbm_params(trial: Any) -> dict[str, Any]:
    """Espace de recherche Optuna : arbres petits et très régularisés (données bruitées)."""
    return {
        "num_leaves": trial.suggest_int("num_leaves", 4, 32, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 300, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
    }


def fit_lightgbm(
    params: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    threads: int = 0,
) -> Any:
    """LightGBM avec arrêt anticipé sur la validation (50 arbres sans progrès)."""
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        n_estimators=2_000,
        subsample_freq=1,
        random_state=seed,
        n_jobs=threads or None,
        verbose=-1,
        **params,
    )
    model.fit(
        x_train,
        y_train,
        eval_X=(x_val,),
        eval_y=(y_val,),
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model
