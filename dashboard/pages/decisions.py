"""Décisions : achats et ventes sur le prix, motifs, et quel agent a bloqué quoi."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import (
    BUY_COLOR,
    SELL_COLOR,
    date_range,
    header,
    kpis,
    plot_layout,
    require_period,
    section,
    strategy_picker,
)

from gold_agents.dashboard import data

period = require_period()
header("Décisions", "Chaque ordre, son motif et l'agent qui l'a décidé")

left, right = st.columns([1, 2])
with left:
    # Par défaut, le premier système d'agents qui a réellement passé des ordres.
    active = [s for s in period.strategies if s.startswith("agents") and len(period.trades[s])]
    strategy = strategy_picker(period, "decisions_strategy", default=(active or [None])[0])
with right:
    lo, hi = date_range(period, "decisions_dates")

frame = period.frames[strategy]
trades = period.trades[strategy]
close = period.close.loc[lo:hi]

in_range = frame[(frame["date"] >= lo) & (frame["date"] <= hi)]
refused = int((in_range["approved"] == False).sum()) if "approved" in in_range else 0  # noqa: E712
kpis(
    [
        ("Achats", str(int((in_range["action"] == "BUY").sum())), "ordres exécutés", ""),
        ("Ventes", str(int((in_range["action"] == "SELL").sum())), "ordres exécutés", ""),
        ("Jours en attente", str(int((in_range["action"] == "HOLD").sum())), "aucun ordre", ""),
        ("Ordres refusés", str(refused), "par l'agent risque", "down" if refused else ""),
    ]
)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=close.index,
        y=close,
        name="Clôture de l'or",
        line={"color": "#AEB6C6", "width": 1.5},
        hovertemplate="%{y:,.2f} $",
    )
)
if len(trades):
    shown = trades[(trades["exec_date"] >= lo) & (trades["exec_date"] <= hi)]
    for side, symbol, col, name in (
        ("BUY", "triangle-up", BUY_COLOR, "Achat"),
        ("SELL", "triangle-down", SELL_COLOR, "Vente"),
    ):
        t = shown[shown["side"] == side]
        fig.add_trace(
            go.Scatter(
                x=t["exec_date"],
                y=t["price"],
                mode="markers",
                name=f"{name} ({len(t)})",
                marker={
                    "symbol": symbol,
                    "size": 12,
                    "color": col,
                    "line": {"width": 1.5, "color": "#0B1220"},
                },
                customdata=t[["reason", "qty", "fees"]].to_numpy(),
                hovertemplate=f"<b>{name}</b> à %{{y:,.2f}} $<br>%{{customdata[0]}}<br>"
                "quantité %{customdata[1]:.3f} oz, frais %{customdata[2]:.2f} $<extra></extra>",
            )
        )
fig.update_layout(title="Ordres exécutés à l'ouverture du lendemain")
st.plotly_chart(plot_layout(fig, height=440, y_title="USD par once"), width="stretch")
if not len(trades):
    st.info("Aucun ordre exécuté sur la période pour cette stratégie.")

# --- Qui a décidé quoi --------------------------------------------------------------------
counts = data.rule_counts(frame)
if len(counts):
    section("Qui a décidé : règle appliquée par l'agent risque")
    pivot = counts.pivot_table(index="rule", columns="action", values="jours", fill_value=0)
    pivot.index = [data.rule_label(str(r)) for r in pivot.index]
    pivot.columns = [data.action_label(str(a)) for a in pivot.columns]
    pivot.index.name = "Règle"
    st.dataframe(pivot, width="stretch")
    if "signal" in frame:
        changed = frame["signal"].notna() & (frame["signal"] != frame["action"])
        st.caption(
            f"{int(changed.sum())} jour(s) où l'ordre final diffère du signal de l'agent "
            "stratégie (signal bloqué ou remplacé par l'agent risque)."
        )

# --- Tableau filtrable --------------------------------------------------------------------
section("Journal des décisions")
view = frame[(frame["date"] >= lo) & (frame["date"] <= hi)].copy()
c = st.columns([1, 1, 2])
actions = sorted(view["action"].dropna().unique())
default = [a for a in actions if a != "HOLD"] or actions
pick_actions = c[0].multiselect("Action", actions, default, format_func=data.action_label)
rule_options = sorted(view["rule"].dropna().unique()) if "rule" in view else []
pick_rules = (
    c[1].multiselect("Règle", rule_options, rule_options, format_func=data.rule_label)
    if rule_options
    else []
)
text = c[2].text_input("Rechercher dans le motif", placeholder="ex. stop, seuil, incertain")

mask = view["action"].isin(pick_actions)
if rule_options:
    mask &= view["rule"].isin(pick_rules) | view["rule"].isna()
if text:
    mask &= view["reason"].fillna("").str.contains(text, case=False, regex=False)
view = view[mask]
for col in ("action", "signal"):
    if col in view:
        view[col] = view[col].map(data.action_label)
if "rule" in view:
    view["rule"] = view["rule"].map(data.rule_label)

columns = {
    "date": "Date",
    "signal": "Signal de la stratégie",
    "action": "Ordre final",
    "rule": "Décidé par",
    "reason": "Motif",
    "r_hat": "Rendement prévu",
    "sigma_hat": "Incertitude",
    "value": "Valeur du portefeuille",
    "block": "Bloc n°",
}
table = view[[c for c in columns if c in view]].rename(columns=columns)
st.dataframe(
    table,
    hide_index=True,
    width="stretch",
    column_config={
        "Date": st.column_config.DateColumn(format="DD/MM/YYYY"),
        "Rendement prévu": st.column_config.NumberColumn(format="percent"),
        "Incertitude": st.column_config.NumberColumn(format="percent"),
        "Valeur du portefeuille": st.column_config.NumberColumn(format="dollar"),
        "Motif": st.column_config.TextColumn(width="large"),
    },
)
left, right = st.columns([3, 1], vertical_alignment="center")
left.caption(
    f"{len(table)} jour(s) affiché(s). Le bloc n° permet de retrouver la décision dans "
    "le registre blockchain."
)
right.download_button(
    "Exporter en CSV",
    pd.DataFrame(table).to_csv(index=False).encode("utf-8"),
    file_name=f"decisions_{strategy}.csv",
    mime="text/csv",
    icon=":material/download:",
    width="stretch",
)
