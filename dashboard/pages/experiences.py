"""Expériences : réglages figés, hyperparamètres retenus, runs MLflow, figures produites."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from common import MLFLOW_DB, ROOT, require_period

from gold_agents.dashboard import data

period = require_period()


def fmt(value: object) -> str:
    """4 chiffres significatifs pour les réels, tel quel pour le reste."""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


st.title("Expériences")

plan = period.plan or {}
procedure = plan.get("procedure", {})

st.subheader("Réglages des agents, figés avant le test")
settings = plan.get("settings") or period.geneses[period.strategies[0]].get("config", {}).get(
    "strategy", {}
)
if settings:
    names = {
        "model": "modèle de prévision",
        "theta_buy": "θ_buy : r̂ minimal pour acheter",
        "theta_sell": "θ_sell : r̂ sous lequel on vend",
        "min_confidence": "confiance minimale (|z|)",
        "stop_loss": "stop-loss",
        "max_drawdown": "perte max avant coupe-circuit",
        "cooldown_days": "jours de pause après coupe-circuit",
    }
    st.dataframe(
        pd.DataFrame(
            {
                "réglage": [names.get(k, k) for k in settings],
                "valeur": [str(v) for v in settings.values()],
            }
        ),
        hide_index=True,
        width="stretch",
    )
frozen = plan.get("frozen_settings")
if frozen:
    st.caption(
        f"Contrôlés contre le registre `{frozen.get('file')}` "
        f"(tête `{str(frozen.get('head'))[:16]}…`) "
        f": {'conformes' if frozen.get('ok') else 'DIFFÉRENTS'}. Choisis en phase 6 par "
        "walk-forward sur 2011-2021, jamais sur le test."
    )

if procedure:
    st.subheader("Hyperparamètres des modèles retenus")
    nets = procedure.get("networks", {})
    left, right = st.columns([3, 2])
    if nets:
        table = pd.DataFrame(
            {
                name.upper(): {k: fmt(v) for k, v in data.flatten(spec).items()}
                for name, spec in nets.items()
            }
        )
        left.caption("Réseaux récurrents (choisis par Optuna sur la validation)")
        left.dataframe(table, width="stretch")
    if procedure.get("lightgbm"):
        lgbm = pd.DataFrame({"LightGBM": {k: fmt(v) for k, v in procedure["lightgbm"].items()}})
        right.caption("LightGBM")
        right.dataframe(lgbm, width="stretch")
    st.caption(
        f"Graines : {procedure.get('seeds')}, époques max {procedure.get('max_epochs')}, "
        f"patience {procedure.get('patience')}, {procedure.get('mc_samples')} passages de MC "
        f"dropout. {procedure.get('forecasts', '')}"
    )
    info = period.results.get("genesis", {}).get("forecast_info")
    if info:
        st.caption(
            f"Test : {info.get('n_fit')} jours d'apprentissage, {info.get('n_predicted')} jours "
            f"prédits. Meilleures époques LSTM {info.get('lstm_best_epochs')}, GRU "
            f"{info.get('gru_best_epochs')} ; arbres LightGBM {info.get('lightgbm_trees')}."
        )

tuning = data.read_json(ROOT / "outputs" / "results" / "tuning" / "chosen.json")
if tuning:
    with st.expander("Résultat du réglage walk-forward (phase 6)"):
        st.json(tuning, expanded=1)
grid = ROOT / "outputs" / "results" / "tuning" / "grid_summary.csv"
if grid.is_file():
    with st.expander("Grille de réglage : meilleures combinaisons"):
        st.dataframe(pd.read_csv(grid).head(50), hide_index=True, width="stretch")

st.subheader("Runs MLflow")
runs = data.mlflow_runs(MLFLOW_DB)
if runs.empty:
    st.info(
        "Pas de base MLflow sur ce PC (`mlruns/mlflow.db`, non versionnée). Elle est créée "
        "par `make train`. Pour l'interface complète : `uv run --extra ml mlflow ui "
        "--backend-store-uri sqlite:///mlruns/mlflow.db`."
    )
else:
    experiments = sorted(runs["experiment"].unique())
    picked = st.multiselect("Expériences", experiments, experiments)
    view = runs[runs["experiment"].isin(picked)].dropna(axis=1, how="all")
    metric_cols = [
        c
        for c in view.columns
        if not c.startswith("param.")
        and c not in ("run_uuid", "experiment", "run", "status", "start_time")
    ]
    c = st.columns(3)
    c[0].metric("Runs", len(view))
    c[1].metric("Expériences", len(picked))
    if "mae" in view:
        best = view.loc[view["mae"].idxmin()]
        c[2].metric("Meilleure MAE", f"{best['mae']:.5f}", str(best["run"]), delta_color="off")
    st.dataframe(
        view.drop(columns=["run_uuid"]),
        hide_index=True,
        width="stretch",
        column_config={
            "start_time": st.column_config.DatetimeColumn("début", format="DD/MM/YYYY HH:mm")
        },
    )
    st.caption(
        f"{len(metric_cols)} métrique(s) et "
        f"{sum(c.startswith('param.') for c in view.columns)} paramètre(s) par run."
    )

summary = ROOT / "outputs" / "results" / "forecast" / "summary_val.csv"
if summary.is_file():
    with st.expander("Comparaison des modèles sur la validation (5 graines, phase 4)"):
        st.dataframe(pd.read_csv(summary), hide_index=True, width="stretch")

if period.run.results_dir is not None:
    images = sorted(period.run.results_dir.glob("*.png"))
    if images:
        st.subheader("Figures produites par le script")
        for img in images:
            st.image(str(img), caption=img.name)
