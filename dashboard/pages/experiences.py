"""Expériences : réglages figés, hyperparamètres retenus, runs MLflow, figures produites."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from common import MLFLOW_DB, ROOT, header, require_period, section

from gold_agents.dashboard import data

period = require_period()
header("Expériences", "Réglages figés avant le test, modèles retenus et suivi des entraînements")


def fmt(value: object) -> str:
    """4 chiffres significatifs pour les réels, « aucun » pour une valeur vide."""
    if value is None:
        return "aucun"
    if value == "mse":
        return "Erreur quadratique (MSE)"
    if isinstance(value, float):
        return f"{value:.4g}".replace(".", ",")
    return str(value)


plan = period.plan or {}
procedure = plan.get("procedure", {})

# --- Réglages figés -----------------------------------------------------------------------
settings = plan.get("settings") or period.geneses[period.strategies[0]].get("config", {}).get(
    "strategy", {}
)
if settings:
    section("Réglages des agents, figés avant le test")
    relative = settings.get("buy_rule") == "quantile"
    names = {
        "model": ("Modèle de prévision", data.model_label),
        "buy_rule": (
            "Règle d'achat",
            lambda v: (
                "relative (quantile des prévisions passées)" if v == "quantile" else "seuil fixe"
            ),
        ),
        "theta_buy": (
            "Seuil d'achat (quantile des prévisions passées)"
            if relative
            else "Seuil d'achat (rendement prévu minimal)",
            (lambda v: f"{v:.0%}".replace("%", " %"))
            if relative
            else (lambda v: f"{v:.2%}".replace(".", ",").replace("%", " %")),
        ),
        "quantile_window": ("Prévisions passées comparées", lambda v: f"{v} jours"),
        "quantile_min_days": ("Historique minimal avant d'acheter", lambda v: f"{v} jours"),
        "theta_sell": ("Seuil de vente", lambda v: f"{v:.2%}".replace(".", ",").replace("%", " %")),
        "min_confidence": ("Confiance minimale (|r̂| / σ̂)", fmt),
        "stop_loss": ("Stop-loss", lambda v: f"{v:.0%}".replace("%", " %")),
        "max_drawdown": (
            "Perte maximale avant coupe-circuit",
            lambda v: f"{v:.0%}".replace("%", " %"),
        ),
        "cooldown_days": ("Pause après le coupe-circuit", lambda v: f"{v} jours"),
    }
    rows = []
    ordered = [k for k in names if k in settings] + [k for k in settings if k not in names]
    for key in ordered:
        value = settings[key]
        name, show = names.get(key, (key, fmt))
        rows.append({"Réglage": name, "Valeur": show(value) if value is not None else "aucun"})
    left, right = st.columns([3, 2])
    left.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    frozen = plan.get("frozen_settings") or {}
    with right.container(border=True):
        state = "conformes" if frozen.get("ok") else "non contrôlés"
        st.markdown(
            f"**Réglages {state}** au registre des réglages figés.\n\n"
            "Ils ont été choisis par validation glissante sur 2011-2021, puis scellés "
            "avant de lire le moindre prix du test."
        )
        if plan.get("planned_at"):
            st.caption(f"Plan du test scellé le {plan['planned_at'][:10]}.")
        if plan.get("attempt", 1) > 1:
            st.markdown(
                f"**Tentative {plan['attempt']}.** Raison inscrite au registre : "
                f"« {plan.get('reason_again') or 'non précisée'} ». Les tentatives "
                "précédentes restent consultables (sélecteur « Période »)."
            )

# --- Hyperparamètres ----------------------------------------------------------------------
if procedure:
    section("Hyperparamètres des modèles retenus")
    nets = procedure.get("networks", {})
    left, right = st.columns([3, 2])
    if nets:
        table = pd.DataFrame(
            {
                name.upper(): {
                    data.hyperparam_label(k): fmt(v) for k, v in data.flatten(spec).items()
                }
                for name, spec in nets.items()
            }
        )
        table.index.name = "Réseaux récurrents"
        left.dataframe(table, width="stretch")
    if procedure.get("lightgbm"):
        lgbm = pd.DataFrame(
            {
                "LightGBM": {
                    data.hyperparam_label(k): fmt(v) for k, v in procedure["lightgbm"].items()
                }
            }
        )
        lgbm.index.name = "Arbres de décision"
        right.dataframe(lgbm, width="stretch")
    seeds = procedure.get("seeds") or []
    st.caption(
        f"{len(seeds)} graines par réseau, au plus {procedure.get('max_epochs')} époques avec "
        f"arrêt anticipé après {procedure.get('patience')} époques sans progrès, "
        f"{procedure.get('mc_samples')} passages de MC Dropout pour l'incertitude. Les "
        "modèles sont réentraînés sur tout ce qui précède le début du test."
    )

# --- MLflow ------------------------------------------------------------------------------
section("Suivi des entraînements (MLflow)")
runs = data.mlflow_runs(MLFLOW_DB)
if runs.empty:
    st.info(
        "Le suivi des entraînements n'est pas disponible sur ce poste : il est créé lors de "
        "l'entraînement des modèles.",
        icon=":material/info:",
    )
else:
    experiments = sorted(runs["experiment"].unique())
    picked = st.multiselect("Expériences", experiments, experiments)
    view = runs[runs["experiment"].isin(picked)].dropna(axis=1, how="all")
    c = st.columns(3)
    c[0].metric("Entraînements", len(view), border=True)
    c[1].metric("Expériences", len(picked), border=True)
    if "mae" in view and view["mae"].notna().any():
        best = view.loc[view["mae"].idxmin()]
        c[2].metric(
            "Meilleure MAE", f"{best['mae']:.5f}", str(best["run"]), delta_color="off", border=True
        )
    shown = view.drop(columns=["run_uuid"]).rename(columns=lambda col: col.removeprefix("param."))
    shown = shown.rename(
        columns={"experiment": "Expérience", "run": "Entraînement", "status": "État"}
    )
    st.dataframe(
        shown,
        hide_index=True,
        width="stretch",
        column_config={
            "start_time": st.column_config.DatetimeColumn("Début", format="DD/MM/YYYY HH:mm")
        },
    )

# --- Réglage walk-forward (si présent) ----------------------------------------------------
tuning = data.read_json(ROOT / "outputs" / "results" / "tuning" / "chosen.json")
grid = ROOT / "outputs" / "results" / "tuning" / "grid_summary.csv"
if tuning or grid.is_file():
    section("Réglage par validation glissante")
    if tuning:
        flat = data.flatten(tuning)
        with st.expander("Combinaison retenue"):
            st.dataframe(
                pd.DataFrame({"Paramètre": list(flat), "Valeur": [fmt(v) for v in flat.values()]}),
                hide_index=True,
                width="stretch",
            )
    if grid.is_file():
        with st.expander("Meilleures combinaisons de la grille"):
            st.dataframe(pd.read_csv(grid).head(50), hide_index=True, width="stretch")

summary = ROOT / "outputs" / "results" / "forecast" / "summary_val.csv"
if summary.is_file():
    with st.expander("Comparaison des modèles sur la validation (5 graines)"):
        st.dataframe(pd.read_csv(summary), hide_index=True, width="stretch")

# --- Figures ------------------------------------------------------------------------------
if period.run.results_dir is not None:
    images = sorted(period.run.results_dir.glob("*.png"))
    captions = {
        "final_test": "Synthèse du test final",
        "robustesse_test": "Robustesse : frais et comparaison au hasard",
    }
    if images:
        section("Figures de l'évaluation")
        for img in images:
            st.image(str(img), caption=captions.get(img.stem), width="stretch")
