"""Prévisions : prix et rendement réels contre prévus, incertitude, MAE/RMSE par modèle."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import color, date_range, plot_layout, require_period

from gold_agents.dashboard import data

period = require_period()
st.title("Prévisions")

forecasters = period.forecasters()
if not forecasters or period.close.empty:
    st.info("Ces registres ne contiennent pas de prévisions (seulement des stratégies simples).")
    st.stop()

# Une entrée par modèle (l'ensemble avec et sans incertitude partagent les mêmes prévisions).
models: dict[str, str] = {}
for name in forecasters:
    model_id = str(period.frames[name]["model_id"].dropna().iloc[0])
    models.setdefault(model_id, name)
labels = {m: f"{m} (registre {data.label(s)})" for m, s in models.items()}
left, right = st.columns([1, 2])
model = left.selectbox("Modèle", list(models), format_func=labels.get)
strategy = models[model]
with right:
    lo, hi = date_range(period, "forecast_dates")

frame = period.frames[strategy].set_index("date")
close = period.close
df = pd.DataFrame(
    {
        "close": close,
        "target": data.realized_returns(close),
        "r_hat": frame["r_hat"],
        "sigma": frame.get("sigma_hat"),
    }
).dropna(subset=["r_hat"])
has_sigma = df["sigma"].notna().any() and float(df["sigma"].abs().max()) > 0
level = st.select_slider(
    "Bande d'incertitude",
    options=[0.68, 0.90, 0.95],
    value=0.95,
    format_func="{:.0%}".format,
    disabled=not has_sigma,
    help="Intervalle r̂ ± k·σ̂, où σ̂ vient de la dispersion entre graines ou du MC dropout.",
)
k = {0.68: 1.0, 0.90: 1.645, 0.95: 1.96}[level]
view = df.loc[lo:hi]

# --- Prix réel contre prix prévu ---------------------------------------------------------
# La prévision du jour t porte sur la clôture de t+1 : prix prévu = clôture(t)·exp(r̂).
pred = pd.DataFrame(index=view.index)
pred["price"] = view["close"] * np.exp(view["r_hat"])
pred["low"] = view["close"] * np.exp(view["r_hat"] - k * view["sigma"].fillna(0))
pred["high"] = view["close"] * np.exp(view["r_hat"] + k * view["sigma"].fillna(0))
next_day = view.index.to_series().shift(-1)
pred = pred.set_index(next_day.to_numpy()).iloc[:-1]

fig = go.Figure()
if has_sigma:
    fig.add_trace(
        go.Scatter(
            x=pred.index, y=pred["high"], line={"width": 0}, showlegend=False, hoverinfo="skip"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pred.index,
            y=pred["low"],
            fill="tonexty",
            line={"width": 0},
            fillcolor="rgba(237,161,0,0.25)",
            name=f"Bande {level:.0%}",
            hoverinfo="skip",
        )
    )
fig.add_trace(
    go.Scatter(
        x=view.index,
        y=view["close"],
        name="Prix réel",
        line={"color": "#2a78d6", "width": 2},
        hovertemplate="%{y:,.2f} $",
    )
)
fig.add_trace(
    go.Scatter(
        x=pred.index,
        y=pred["price"],
        name="Prix prévu la veille",
        line={"color": color(strategy), "width": 2, "dash": "dot"},
        hovertemplate="%{y:,.2f} $",
    )
)
fig.update_layout(title="Clôture réelle et clôture prévue la veille")
st.plotly_chart(plot_layout(fig, y_title="USD par once"), width="stretch")
st.caption(
    "Sur un prix, toute prévision « colle » à la courbe : le prix d'hier est déjà une très "
    "bonne estimation. Le vrai test est le rendement, juste en dessous."
)

# --- Rendement réel contre prévu ----------------------------------------------------------
fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=view.index,
        y=view["target"],
        name="Rendement réel (t → t+1)",
        marker_color="rgba(42,120,214,0.45)",
        hovertemplate="%{y:.2%}",
    )
)
if has_sigma:
    fig.add_trace(
        go.Scatter(
            x=view.index,
            y=view["r_hat"] + k * view["sigma"],
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=view.index,
            y=view["r_hat"] - k * view["sigma"],
            fill="tonexty",
            line={"width": 0},
            fillcolor="rgba(237,161,0,0.3)",
            name=f"Bande {level:.0%}",
            hoverinfo="skip",
        )
    )
fig.add_trace(
    go.Scatter(
        x=view.index,
        y=view["r_hat"],
        name="Rendement prévu r̂",
        line={"color": color(strategy), "width": 2},
        hovertemplate="%{y:.3%}",
    )
)
fig.update_layout(title="Rendement réel et rendement prévu, jour par jour", bargap=0)
st.plotly_chart(plot_layout(fig, y_title="Rendement log", y_format=".1%"), width="stretch")

ok = view.dropna(subset=["target"])
c = st.columns(4)
c[0].metric(
    "Bonne direction",
    f"{np.mean(np.sign(ok['r_hat']) == np.sign(ok['target'])):.1%}",
    help="Part des jours où le signe de r̂ est celui du rendement réel.",
)
c[1].metric(
    "Jours en hausse",
    f"{np.mean(ok['target'] > 0):.1%}",
    help="Référence : toujours prévoir « hausse » donne ce score de direction.",
)
c[2].metric("r̂ moyen", f"{ok['r_hat'].mean():+.3%}")
c[3].metric("Rendement réel moyen", f"{ok['target'].mean():+.3%}")

# --- Tableau par modèle -------------------------------------------------------------------
st.subheader("Erreurs par modèle (période entière)")
sealed = period.results.get("forecast_metrics")
if sealed:
    table = pd.DataFrame(sealed)
    st.caption("Lu dans `resultats.jsonl` (scellé au moment du test).")
else:
    rows = []
    for m, s in models.items():
        f = period.frames[s].set_index("date")
        j = pd.concat([f["r_hat"], data.realized_returns(close)], axis=1).dropna()
        err = j["r_hat"] - j["target_return"]
        rows.append(
            {
                "model": m,
                "mae": float(err.abs().mean()),
                "rmse": float(np.sqrt((err**2).mean())),
                "direction": float(np.mean(np.sign(j["r_hat"]) == np.sign(j["target_return"]))),
                "n_days": len(j),
            }
        )
    y = data.realized_returns(close).dropna()
    rows.append(
        {
            "model": "b0_zero",
            "mae": float(y.abs().mean()),
            "rmse": float(np.sqrt((y**2).mean())),
            "direction": None,
            "n_days": len(y),
        }
    )
    table = pd.DataFrame(rows)
    st.caption("Recalculé depuis les registres (pas de résultats scellés pour cette période).")

names = {"b0_zero": "B0 : prévoir 0", "always_up": "Toujours « hausse »"}
show = pd.DataFrame(
    {
        "Modèle": table["model"].map(lambda m: names.get(m, m)),
        "MAE": table["mae"].map(lambda x: f"{x:.5f}"),
        "RMSE": table["rmse"].map(lambda x: f"{x:.5f}"),
        "Bonne direction": table["direction"].map(lambda x: "–" if pd.isna(x) else f"{x:.1%}"),
    }
)
if "r2_vs_zero" in table:
    show["Gain sur B0 (R²)"] = table["r2_vs_zero"].map(lambda x: f"{x:+.2%}")
if "dm_p_value" in table:
    show["p-value Diebold-Mariano"] = table["dm_p_value"].map(lambda x: f"{x:.3f}")
st.dataframe(show, hide_index=True, width="stretch")
st.caption(
    "B0 prévoit un rendement nul chaque jour. Un modèle utile doit faire mieux que B0 "
    "(MAE et RMSE plus petits) ; la p-value de Diebold-Mariano dit si l'écart est "
    "significatif (< 0,05)."
)
