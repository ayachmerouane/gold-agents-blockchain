"""Phase 4 : baselines B0-B4 puis réseaux M1 (LSTM) et M2 (GRU), sur la validation.

Pour chaque modèle : réglage (Optuna pour LightGBM, LSTM et GRU, chaque essai suivi dans
MLflow), entraînement final avec plusieurs graines, métriques par graine (moyenne ±
écart-type) et prévision moyenne des graines. Le test n'est pas touché ici : il est
réservé à l'évaluation finale (§4.3, phase 8).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from gold_agents.evaluation.forecast_metrics import diebold_mariano, forecast_metrics
from gold_agents.models import baselines
from gold_agents.models.dataset import ForecastData
from gold_agents.models.predict import save_network
from gold_agents.models.recurrent import RecurrentRegressor
from gold_agents.models.train import TrainConfig, fit, from_params, predict, suggest
from gold_agents.models.uncertainty import mc_dropout

LABELS = {
    "b0": "B0 rendement nul",
    "b1": "B1 rendement d'hier",
    "b1_mm20": "B1 moyenne 20 j",
    "b2": "B2 ARIMA",
    "b3": "B3 Ridge",
    "b4": "B4 LightGBM",
    "lstm": "M1 LSTM",
    "gru": "M2 GRU",
    "ensemble": "E LSTM + GRU",
}
ALL_MODELS = tuple(LABELS)
NETWORKS = ("lstm", "gru")
METRICS = ("mae", "rmse", "direction", "r2_vs_zero")


@dataclass(frozen=True)
class Settings:
    models: tuple[str, ...] = ALL_MODELS
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    trials: int = 20
    lgbm_trials: int = 30
    max_epochs: int = 100
    patience: int = 15
    mc_samples: int = 50
    direction_alert: float = 0.60
    threads: int = 0
    mlflow_uri: str | None = None


@dataclass
class ModelResult:
    key: str
    per_seed: list[dict[str, float]]
    prediction: np.ndarray  # moyenne des graines, échelle des rendements
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class Benchmark:
    results: dict[str, ModelResult]
    y_val: np.ndarray
    dates: pd.DatetimeIndex
    histories: dict[str, list[tuple[float, float]]]
    alerts: list[str]

    def summary(self) -> pd.DataFrame:
        rows = []
        for key, res in self.results.items():
            seeds = pd.DataFrame(res.per_seed)
            row: dict[str, Any] = {"model": key, "label": LABELS[key], "n_seeds": len(seeds)}
            for m in METRICS:
                row[m] = float(seeds[m].mean())
                row[f"{m}_std"] = float(seeds[m].std(ddof=0))
            dm = diebold_mariano(self.y_val, res.prediction, np.zeros_like(self.y_val))
            row["dm_vs_zero"], row["dm_p_value"] = dm.statistic, dm.p_value
            row["rmse_zero"] = float(np.sqrt(np.mean(self.y_val**2)))  # RMSE de B0
            row.update({k: v for k, v in res.info.items() if k in ("sigma_mean", "coverage_95")})
            rows.append(row)
        return pd.DataFrame(rows).set_index("model")


class Tracker:
    """Suivi MLflow : une exécution par essai Optuna et par graine finale (ou rien)."""

    def __init__(self, uri: str | None, experiment: str = "phase4-prevision") -> None:
        self.enabled = uri is not None
        if uri is None:
            return
        import os

        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        import mlflow

        if uri.startswith("sqlite:///"):
            Path(uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)

    def log(
        self,
        name: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        tags: dict[str, str],
        history: list[tuple[float, float]] | None = None,
    ) -> None:
        if not self.enabled:
            return
        import mlflow

        with mlflow.start_run(run_name=name, tags=tags):
            mlflow.log_params(params)
            mlflow.log_metrics({k: v for k, v in metrics.items() if np.isfinite(v)})
            for step, (train_loss, val_loss) in enumerate(history or [], start=1):
                mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=step)


def _optuna_study(seed: int) -> Any:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )


def run_benchmark(
    data: ForecastData,
    settings: Settings,
    models_dir: Path,
    log: Callable[[str], None] = print,
) -> Benchmark:
    if settings.threads:
        torch.set_num_threads(settings.threads)
    tracker = Tracker(settings.mlflow_uri)
    train, val = data["train"], data["val"]
    y_val = val.y.astype("float64")
    results: dict[str, ModelResult] = {}
    histories: dict[str, list[tuple[float, float]]] = {}
    wanted = set(settings.models)

    def add(key: str, preds: list[np.ndarray], info: dict[str, Any] | None = None) -> None:
        per_seed = [forecast_metrics(y_val, p) for p in preds]
        results[key] = ModelResult(key, per_seed, np.mean(preds, axis=0), info or {})
        m = results[key].per_seed[0]
        log(f"  {LABELS[key]:22s} RMSE {m['rmse']:.5f}  direction {m['direction']:.1%}")
        for seed, metrics in zip(settings.seeds, per_seed, strict=False):
            tracker.log(
                f"{key}-seed{seed}",
                {k: v for k, v in (info or {}).items() if not isinstance(v, dict | list)},
                metrics,
                {"model": key, "stage": "final"},
            )

    log("Baselines :")
    if "b0" in wanted:
        add("b0", [baselines.zero(len(y_val))])
    if "b1" in wanted:
        add("b1", [baselines.last_return(data.frame, val.rows)])
    if "b1_mm20" in wanted:
        add("b1_mm20", [baselines.mean_return(data.frame, val.rows, 20)])
    if "b2" in wanted:
        arima = baselines.fit_arima(data.frame)
        add("b2", [arima.predict(data.frame, val.rows)], {"order": str(arima.order)})
    if "b3" in wanted:
        ridge, alpha = baselines.fit_ridge(train.tabular, train.y, val.tabular, val.y)
        add("b3", [ridge.predict(val.tabular).astype("float64")], {"alpha": alpha})
    if "b4" in wanted:
        add("b4", *_lightgbm(data, settings, tracker, log))

    nets: dict[str, list[np.ndarray]] = {}
    for kind in NETWORKS:
        if kind in wanted or "ensemble" in wanted:
            preds, info, history = _network(kind, data, settings, tracker, models_dir, log)
            histories[kind] = history
            nets[kind] = preds
            if kind in wanted:
                add(kind, preds, info)
    if "ensemble" in wanted:
        add("ensemble", [(a + b) / 2 for a, b in zip(nets["lstm"], nets["gru"], strict=True)])

    alerts = []
    for key, res in results.items():
        direction = float(np.mean([m["direction"] for m in res.per_seed]))
        if direction > settings.direction_alert:  # NaN (B0, sans avis) : jamais d'alerte
            alerts.append(
                f"{LABELS[key]} : {direction:.1%} de bonnes directions "
                f"(> {settings.direction_alert:.0%}), fuite de données probable"
            )
    return Benchmark(results, y_val, val.windows.dates, histories, alerts)


def _lightgbm(
    data: ForecastData, settings: Settings, tracker: Tracker, log: Callable[[str], None]
) -> tuple[list[np.ndarray], dict[str, Any]]:
    train, val = data["train"], data["val"]
    seed0 = settings.seeds[0]

    def objective(trial: Any) -> float:
        params = baselines.lightgbm_params(trial)
        model = baselines.fit_lightgbm(
            params, train.tabular, train.y, val.tabular, val.y, seed0, settings.threads
        )
        mse = float(np.mean((model.predict(val.tabular) - val.y) ** 2))
        tracker.log(
            f"b4-trial{trial.number}",
            {**params, "n_trees": model.best_iteration_},
            {"val_mse": mse},
            {"model": "b4", "stage": "optuna"},
        )
        return mse

    start = time.perf_counter()
    study = _optuna_study(seed0)
    study.optimize(objective, n_trials=settings.lgbm_trials)
    log(f"  LightGBM : {settings.lgbm_trials} essais Optuna en {time.perf_counter() - start:.0f} s")
    preds = []
    for seed in settings.seeds:
        model = baselines.fit_lightgbm(
            study.best_params, train.tabular, train.y, val.tabular, val.y, seed, settings.threads
        )
        preds.append(model.predict(val.tabular).astype("float64"))
    return preds, {**study.best_params}


def _network(
    kind: str,
    data: ForecastData,
    settings: Settings,
    tracker: Tracker,
    models_dir: Path,
    log: Callable[[str], None],
) -> tuple[list[np.ndarray], dict[str, Any], list[tuple[float, float]]]:
    import optuna

    train, val = data["train"], data["val"]
    y_scale = float(np.std(train.y))
    y_train, y_val = train.y / y_scale, val.y / y_scale
    base = TrainConfig(max_epochs=settings.max_epochs, patience=settings.patience)
    n_features = train.x.shape[2]
    seed0 = settings.seeds[0]

    def objective(trial: Any) -> float:
        net, cfg = suggest(trial, n_features, base)

        def report(epoch: int, loss: float) -> None:
            trial.report(loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        result = fit(kind, net, cfg, train.x, y_train, val.x, y_val, seed0, report)
        tracker.log(
            f"{kind}-trial{trial.number}",
            {**trial.params, "best_epoch": result.best_epoch},
            {"val_mse_scaled": result.best_val_loss},
            {"model": kind, "stage": "optuna"},
            result.history,
        )
        return result.best_val_loss

    start = time.perf_counter()
    study = _optuna_study(seed0)
    study.optimize(objective, n_trials=settings.trials)
    pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
    log(
        f"  {kind.upper()} : {settings.trials} essais Optuna ({pruned} abandonnés tôt) "
        f"en {time.perf_counter() - start:.0f} s ; meilleurs réglages {study.best_params}"
    )

    net, cfg = from_params(study.best_params, n_features, base)
    models: dict[int, RecurrentRegressor] = {}
    preds, epochs = [], []
    history: list[tuple[float, float]] = []
    for seed in settings.seeds:
        result = fit(kind, net, cfg, train.x, y_train, val.x, y_val, seed)
        models[seed] = result.model
        epochs.append(result.best_epoch)
        if seed == seed0:
            history = result.history
        preds.append(predict(result.model, val.x).astype("float64") * y_scale)
        tracker.log(
            f"{kind}-final-seed{seed}",
            {**study.best_params, "seed": seed, "best_epoch": result.best_epoch},
            forecast_metrics(val.y.astype("float64"), preds[-1]),
            {"model": kind, "stage": "final"},
            result.history,
        )

    # Incertitude (MC Dropout) du réseau de la première graine : écart-type moyen et part
    # des rendements réels dans l'intervalle ± 1,96 σ.
    mu, sigma = mc_dropout(models[seed0], val.x, settings.mc_samples, seed0)
    mu, sigma = mu * y_scale, sigma * y_scale
    inside = np.abs(val.y - mu) <= 1.96 * sigma
    info = {
        **study.best_params,
        "best_epochs": epochs,
        "sigma_mean": float(np.mean(sigma)),
        "coverage_95": float(np.mean(inside)),
    }
    save_network(
        models_dir / kind,
        kind,
        models,
        {
            "net": net.to_dict(),
            "train": cfg.to_dict(),
            "y_scale": y_scale,
            "columns": data.columns,
            "window": data.window,
            "best_epochs": epochs,
        },
    )
    return preds, info, history


def write_results(bench: Benchmark, out_dir: Path, meta: dict[str, Any]) -> pd.DataFrame:
    """Écrit le tableau, les métriques par graine et les prévisions de validation."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = bench.summary()
    summary.to_csv(out_dir / "summary_val.csv")
    rows = [
        {"model": key, "seed_index": i, **m}
        for key, res in bench.results.items()
        for i, m in enumerate(res.per_seed)
    ]
    pd.DataFrame(rows).to_csv(out_dir / "metrics_val_by_seed.csv", index=False)
    preds = pd.DataFrame({"date": bench.dates, "target_return": bench.y_val})
    for key, res in bench.results.items():
        preds[key] = res.prediction
    preds.to_parquet(out_dir / "predictions_val.parquet", index=False)
    info = {key: res.info for key, res in bench.results.items()}
    payload = {**meta, "alerts": bench.alerts, "settings": info}
    (out_dir / "summary_val.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return summary
