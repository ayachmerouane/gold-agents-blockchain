"""Prévisions walk-forward (§4.3) : chaque période est prédite par des modèles qui ne l'ont
jamais vue.

Pour régler les seuils des agents sans tricher, il faut des prévisions « hors échantillon »
sur plusieurs périodes. Les modèles de la phase 4 ont appris sur 2004-2018 : leurs
prévisions de 2011 ou 2015 seraient trop belles. Pour chaque période jugée
`[début, fin]`, on réentraîne donc LSTM, GRU et LightGBM sur tout ce qui précède `début`
(fenêtre croissante), avec les hyperparamètres trouvés en phase 4, sans nouvel Optuna :

- passé = jours dont la cible (le lendemain) est avant `début` ;
- les `early_stop_days` derniers jours du passé servent à l'arrêt anticipé, après une
  purge de W jours ; le reste sert à l'apprentissage ;
- le scaler est réajusté sur le passé seulement ;
- on prédit chaque jour de la période (fenêtre complète, rien au-delà de t).

Le test n'est jamais lu : les lignes à partir de `splits.test_start` sont retirées avant
tout calcul.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gold_agents.data.features import RobustScaler
from gold_agents.data.windows import make_windows

MODEL_KEYS = ("lstm", "gru", "ensemble", "lightgbm")


@dataclass(frozen=True)
class FoldRows:
    fit: pd.Series  # apprentissage
    early_stop: pd.Series  # arrêt anticipé
    predict: pd.Series  # jours de la période jugée qui ont une fenêtre complète
    target_known: pd.Series  # la cible (le lendemain) tombe encore dans la période


def before_test(frame: pd.DataFrame, test_start: date) -> pd.DataFrame:
    """Le fichier de features sans aucune ligne du test."""
    days = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.date
    return frame[days < test_start].reset_index(drop=True)


def fold_rows(
    frame: pd.DataFrame, window: int, start: date, end: date, early_stop_days: int
) -> FoldRows:
    """Lignes d'apprentissage, d'arrêt anticipé et de prévision d'une période."""
    days = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.date
    next_day = days.shift(-1)
    full_window = frame["features_valid"].astype("float64").rolling(window).min() == 1
    past = full_window & frame["target_valid"].astype(bool) & (next_day < start)
    idx = np.flatnonzero(past.to_numpy())
    if len(idx) < early_stop_days + window + 100:
        raise ValueError(f"pas assez d'historique avant le {start}")
    es_idx = idx[-early_stop_days:]
    fit_idx = idx[idx < es_idx[0] - window]  # purge de W jours entre les deux
    fit = pd.Series(False, index=frame.index)
    fit.iloc[fit_idx] = True
    early_stop = pd.Series(False, index=frame.index)
    early_stop.iloc[es_idx] = True
    predict = full_window & (days >= start) & (days <= end)
    return FoldRows(fit, early_stop, predict, next_day <= end)


@dataclass(frozen=True)
class FoldInputs:
    x_fit: np.ndarray
    y_fit: np.ndarray
    x_es: np.ndarray
    y_es: np.ndarray
    x_pred: np.ndarray
    y_pred: np.ndarray  # cible réelle (diagnostic seulement, jamais lue par les modèles)
    dates: pd.DatetimeIndex


def fold_inputs(frame: pd.DataFrame, columns: list[str], window: int, rows: FoldRows) -> FoldInputs:
    scaler = RobustScaler.fit(frame[rows.fit | rows.early_stop], columns)
    scaled = scaler.transform(frame)
    fit = make_windows(scaled, columns, window, rows.fit)
    es = make_windows(scaled, columns, window, rows.early_stop)
    pred = make_windows(scaled, columns, window, rows.predict)
    # Le lendemain du dernier jour sort de la période : sa cible n'est pas lue.
    y_pred = np.where(rows.target_known[rows.predict].to_numpy(), pred.y, np.nan)
    return FoldInputs(fit.x, fit.y, es.x, es.y, pred.x, y_pred, pred.dates)


def _networks(
    kind: str,
    meta: dict[str, Any],
    data: FoldInputs,
    seeds: list[int],
    max_epochs: int,
    patience: int,
    mc_samples: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Prévision moyenne des graines et tirages MC Dropout, dans l'échelle des rendements."""
    from gold_agents.models.recurrent import NetConfig
    from gold_agents.models.train import TrainConfig, fit, predict
    from gold_agents.models.uncertainty import mc_draws

    net = NetConfig(**meta["net"])
    cfg = TrainConfig(**{**meta["train"], "max_epochs": max_epochs, "patience": patience})
    y_scale = float(np.std(data.y_fit))
    preds, draws, epochs = [], [], []
    for seed in seeds:
        result = fit(
            kind, net, cfg, data.x_fit, data.y_fit / y_scale, data.x_es, data.y_es / y_scale, seed
        )
        epochs.append(result.best_epoch)
        preds.append(predict(result.model, data.x_pred).astype("float64") * y_scale)
        draws.append(mc_draws(result.model, data.x_pred, mc_samples, seed) * y_scale)
    return np.mean(preds, axis=0), np.concatenate(draws), epochs


def fold_forecasts(
    data: FoldInputs,
    net_meta: dict[str, dict[str, Any]],
    lgbm_params: dict[str, Any] | None,
    seeds: list[int],
    max_epochs: int,
    patience: int,
    mc_samples: int,
    threads: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Prévisions des quatre modèles sur une période ; une ligne par jour et par modèle."""
    out: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
    info: dict[str, Any] = {"n_fit": len(data.y_fit), "n_early_stop": len(data.y_es)}
    draws = {}
    for kind in ("lstm", "gru"):
        r, d, epochs = _networks(
            kind, net_meta[kind], data, seeds, max_epochs, patience, mc_samples
        )
        out[kind] = r, d.std(axis=0)
        draws[kind] = d
        info[f"{kind}_best_epochs"] = epochs
    ens = (out["lstm"][0] + out["gru"][0]) / 2
    out["ensemble"] = ens, np.concatenate([draws["lstm"], draws["gru"]]).std(axis=0)
    if lgbm_params:
        from gold_agents.models.baselines import fit_lightgbm

        tab = data.x_fit[:, -1, :], data.x_es[:, -1, :], data.x_pred[:, -1, :]
        models = [
            fit_lightgbm(lgbm_params, tab[0], data.y_fit, tab[1], data.y_es, s, threads)
            for s in seeds
        ]
        r = np.mean([m.predict(tab[2]) for m in models], axis=0).astype("float64")
        out["lightgbm"] = r, None
        info["lightgbm_trees"] = [int(m.best_iteration_ or 0) for m in models]
    rows = []
    for key, (r, s) in out.items():
        rows.append(
            pd.DataFrame(
                {
                    "date": data.dates,
                    "model": key,
                    "r_hat": r,
                    "sigma_hat": np.nan if s is None else s,
                    "target_return": data.y_pred.astype("float64"),
                }
            )
        )
    return pd.concat(rows, ignore_index=True), info


def walk_forward_forecasts(
    frame: pd.DataFrame,
    columns: list[str],
    window: int,
    folds: list[tuple[date, date]],
    early_stop_days: int,
    net_meta: dict[str, dict[str, Any]],
    lgbm_params: dict[str, Any] | None,
    seeds: list[int],
    max_epochs: int,
    patience: int,
    mc_samples: int,
    threads: int = 0,
    log: Callable[[str], None] = print,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Prévisions hors échantillon de chaque période (colonne `fold`, à partir de 1)."""
    import time

    if threads:
        import torch

        torch.set_num_threads(threads)
    parts, infos = [], []
    for i, (start, end) in enumerate(folds, start=1):
        t0 = time.perf_counter()
        rows = fold_rows(frame, window, start, end, early_stop_days)
        data = fold_inputs(frame, columns, window, rows)
        part, info = fold_forecasts(
            data, net_meta, lgbm_params, seeds, max_epochs, patience, mc_samples, threads
        )
        part.insert(0, "fold", i)
        parts.append(part)
        info = {"fold": i, "start": str(start), "end": str(end), **info}
        infos.append(info)
        log(
            f"  période {i} ({start} -> {end}) : appris sur {info['n_fit']} jours, "
            f"{len(data.dates)} jours prédits ({time.perf_counter() - t0:.0f} s)"
        )
    return pd.concat(parts, ignore_index=True), infos


def forecast_diagnostics(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Par période et par modèle : RMSE comparée au rendement nul, bonnes directions."""
    rows = []
    for (fold, model), g in forecasts.groupby(["fold", "model"], sort=True):
        y, r = g["target_return"].to_numpy(), g["r_hat"].to_numpy()
        ok = ~np.isnan(y)
        y, r = y[ok], r[ok]
        rmse = float(np.sqrt(np.mean((y - r) ** 2)))
        rows.append(
            {
                "fold": fold,
                "model": model,
                "gain RMSE vs B0": rmse / float(np.sqrt(np.mean(y**2))) - 1,
                "direction": float(np.mean(np.sign(r) == np.sign(y))),
                "part de hausses": float(np.mean(y > 0)),
                "r̂ > 0": float(np.mean(r > 0)),
            }
        )
    return pd.DataFrame(rows)


def read_network_meta(models_dir: Path) -> dict[str, dict[str, Any]]:
    """Hyperparamètres des réseaux de la phase 4 (`outputs/models/<kind>/model.json`)."""
    from gold_agents.models.predict import META_FILE

    return {
        kind: json.loads((models_dir / kind / META_FILE).read_text(encoding="utf-8"))
        for kind in ("lstm", "gru")
    }
