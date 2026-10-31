"""Portefeuille : courbes de valeur, drawdown et exposition de chaque stratégie."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import DEFAULT_COMPARE, color, date_range, plot_layout, require_period

from gold_agents.dashboard import data
from gold_agents.evaluation.financial_metrics import drawdown

period = require_period()
st.title("Portefeuille")

options = period.strategies
default = [s for s in DEFAULT_COMPARE if s in options] or options[:3]
picked = st.multiselect("Stratégies comparées", options, default, format_func=data.label)
if not picked:
    st.info("Choisissez au moins une stratégie.")
    st.stop()
left, right = st.columns([3, 1])
with left:
    lo, hi = date_range(period, "portfolio_dates")
log_scale = right.toggle("Échelle logarithmique", value="oracle" in picked)
rebase = right.toggle(
    "Repartir de 100 au début",
    value=False,
    help="Compare les trajectoires depuis la date de début choisie.",
)

fig = go.Figure()
dd = go.Figure()
for name in picked:
    f = period.frames[name].set_index("date").loc[lo:hi]
    values = f["value"].astype("float64")
    y = values / values.iloc[0] * 100 if rebase else values
    style = {"color": color(name), "width": 2}
    if name == "oracle":
        style["dash"] = "dot"
    fig.add_trace(
        go.Scatter(x=f.index, y=y, name=data.label(name), line=style, hovertemplate="%{y:,.0f}")
    )
    dd.add_trace(
        go.Scatter(
            x=f.index,
            y=drawdown(values),
            name=data.label(name),
            line=style,
            hovertemplate="%{y:.1%}",
        )
    )
fig.update_layout(title="Valeur du portefeuille")
plot_layout(fig, height=460, y_title="Base 100" if rebase else "USD")
if log_scale:
    fig.update_yaxes(type="log")
st.plotly_chart(fig, width="stretch")

dd.update_layout(title="Drawdown : baisse depuis le dernier plus haut")
st.plotly_chart(plot_layout(dd, height=320, y_title="", y_format=".0%"), width="stretch")
if "oracle" in picked:
    st.caption(
        "L'oracle connaît le prix du lendemain : c'est un plafond théorique, pas un "
        "objectif atteignable."
    )

st.subheader("Exposition : part du portefeuille investie en or")
close = period.close
rows = []
for name in picked:
    f = period.frames[name].set_index("date")
    invested = (f["qty"] * close.reindex(f.index)).fillna(0) / f["value"]
    rows.append(invested.rename(name))
expo = pd.concat(rows, axis=1).loc[lo:hi]
fig = go.Figure()
for name in picked:
    fig.add_trace(
        go.Scatter(
            x=expo.index,
            y=expo[name],
            name=data.label(name),
            line={"color": color(name), "width": 2, "shape": "hv"},
            hovertemplate="%{y:.0%}",
        )
    )
fig.update_yaxes(range=[-0.05, 1.05])
st.plotly_chart(plot_layout(fig, height=300, y_format=".0%"), width="stretch")
summary = pd.DataFrame(
    {
        "Stratégie": [data.label(n) for n in picked],
        "Jours investis": [f"{(expo[n] > 0.01).mean():.0%}" for n in picked],
        "Valeur au début": [
            data.fmt_money(period.frames[n].set_index("date").loc[lo:hi, "value"].iloc[0])
            for n in picked
        ],
        "Valeur à la fin": [
            data.fmt_money(period.frames[n].set_index("date").loc[lo:hi, "value"].iloc[-1])
            for n in picked
        ],
    }
)
st.dataframe(summary, hide_index=True, width="stretch")
