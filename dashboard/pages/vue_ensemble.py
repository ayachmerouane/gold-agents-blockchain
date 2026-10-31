"""Vue d'ensemble : KPI du système choisi et comparaison avec les références."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import (
    color,
    header,
    metrics_display,
    plot_layout,
    require_period,
    section,
    strategy_picker,
)

from gold_agents.dashboard import data

period = require_period()
frames = period.frames
first = frames[period.strategies[0]]
plan = period.plan or {}
header(
    "Vue d'ensemble",
    f"{period.run.title} · du {first['date'].min():%d/%m/%Y} au "
    f"{first['date'].max():%d/%m/%Y} · {len(first)} jours de bourse · capital initial "
    f"{data.fmt_money(float(first['value'].iloc[0]))}",
)

table = period.metrics.set_index("strategy")
left, right = st.columns([2, 3], vertical_alignment="bottom")
with left:
    chosen = strategy_picker(period, "overview_strategy", "Système analysé")
if plan and plan.get("main_system") in table.index:
    right.caption(
        f"Système principal annoncé avant le test : **{data.label(plan['main_system'])}**. "
        f"Plan scellé le {plan.get('planned_at', '?')[:10]}, tentative n° "
        f"{plan.get('attempt', '?')}."
    )
row = table.loc[chosen]
ref = table.loc["buy_and_hold"] if "buy_and_hold" in table.index else None


def delta(col: str, pct: bool = True) -> str | None:
    if ref is None or chosen == "buy_and_hold":
        return None
    d = float(row[col] - ref[col])
    text = f"{d * 100:+.1f} pts" if pct else f"{d:+.2f}"
    return text.replace(".", ",")


c = st.columns(6)
c[0].metric("Profit net", data.fmt_money(row["net_profit"]), border=True)
c[1].metric("Rendement", data.fmt_pct(row["total_return"]), delta("total_return"), border=True)
c[2].metric("Ratio de Sharpe", data.fmt_num(row["sharpe"]), delta("sharpe", False), border=True)
c[3].metric(
    "Perte maximale",
    data.fmt_pct(row["max_drawdown"], signed=False),
    delta("max_drawdown"),
    help="Plus grande baisse depuis un plus haut. Plus proche de 0 = mieux.",
    border=True,
)
c[4].metric("Transactions", int(row["n_trades"]), border=True)
c[5].metric("Temps investi", f"{row['exposure'] * 100:.0f} %", border=True)

idle = [s for s in table.index if s.startswith("agents") and table.loc[s, "n_trades"] == 0]
if idle:
    st.info(
        "Aucune position prise sur la période par : "
        + ", ".join(data.label(s) for s in idle)
        + ". Leurs prévisions sont restées sous le seuil d'achat ou trop incertaines ; "
        "le détail est sur la page Décisions.",
        icon=":material/info:",
    )
if ref is not None and chosen != "buy_and_hold":
    st.caption("Les écarts sous les indicateurs sont mesurés par rapport au Buy & Hold.")
for alert in period.results.get("alerts") or []:
    st.warning(alert, icon=":material/warning:")

section("Comparaison des stratégies")
st.dataframe(metrics_display(period.metrics), hide_index=True, width="stretch")

left, right = st.columns([3, 1], vertical_alignment="bottom")
show_oracle = right.toggle(
    "Inclure l'oracle",
    value=False,
    help="L'oracle connaît le futur : c'est un plafond théorique, pas une stratégie possible.",
)
plot = period.metrics[(period.metrics["strategy"] != "oracle") | show_oracle].iloc[::-1]
fig = go.Figure(
    go.Bar(
        x=plot["total_return"],
        y=plot["strategy"].map(data.label),
        orientation="h",
        marker={"color": [color(s) for s in plot["strategy"]], "cornerradius": 4},
        text=plot["total_return"].map(data.fmt_pct),
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y} : %{x:.1%}<extra></extra>",
    )
)
fig.update_layout(title="Rendement total sur la période", showlegend=False, hovermode="closest")
plot_layout(fig, height=70 + 40 * len(plot))
fig.update_xaxes(tickformat=".0%", title_text="")
st.plotly_chart(fig, width="stretch")

costs = period.results.get("costs_sensitivity")
refs = period.results.get("random_references")
if costs or refs:
    section("Robustesse")
    tab_costs, tab_random = st.tabs(["Sensibilité aux frais", "Comparaison au hasard"])
if costs:
    with tab_costs:
        frame = pd.DataFrame(costs)
        frame = frame[frame["strategy"] != "oracle"]
        fig = go.Figure()
        for name, g in frame.groupby("strategy", sort=False):
            fig.add_trace(
                go.Scatter(
                    x=g["costs_factor"],
                    y=g["total_return"],
                    name=data.label(str(name)),
                    mode="lines+markers",
                    line={"color": color(str(name)), "width": 2},
                    marker={"size": 8},
                    hovertemplate="frais ×%{x} : %{y:.1%}<extra>"
                    + data.label(str(name))
                    + "</extra>",
                )
            )
        fig.update_layout(title="Rendement si les frais étaient multipliés", hovermode="closest")
        plot_layout(fig, height=400, y_title="Rendement total", y_format=".0%")
        fig.update_xaxes(title_text="Multiplicateur des frais et du glissement", dtick=1)
        st.plotly_chart(fig, width="stretch")
if refs:
    with tab_random:
        frame = pd.DataFrame(refs)
        st.dataframe(
            pd.DataFrame(
                {
                    "Système": frame["strategy"].map(data.label),
                    "Transactions": frame["n_trades"],
                    "Sharpe": frame["sharpe"].map(data.fmt_num),
                    "Sharpe médian du hasard": frame["random_sharpe_median"].map(data.fmt_num),
                    "Intervalle 5 à 95 % du hasard": [
                        f"{data.fmt_num(a)} à {data.fmt_num(b)}"
                        for a, b in zip(
                            frame["random_sharpe_p05"], frame["random_sharpe_p95"], strict=True
                        )
                    ],
                    "Tirages battus": frame["beats_random"].map(lambda x: f"{x:.0%}"),
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "200 stratégies tirées au hasard avec le même nombre de transactions. Un système "
            "qui ne bat pas la plupart d'entre elles ne prouve pas de savoir-faire."
        )
