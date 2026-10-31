"""Vue d'ensemble : KPI du système choisi et comparaison avec les références."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import color, metrics_display, plot_layout, require_period, strategy_picker

from gold_agents.dashboard import data

period = require_period()
st.title("Vue d'ensemble")

frames = period.frames
first = frames[period.strategies[0]]
plan = period.plan or {}
st.markdown(
    f"**{period.run.title}** : du {first['date'].min():%d/%m/%Y} au "
    f"{first['date'].max():%d/%m/%Y}, {len(first)} jours de bourse, capital de départ "
    f"{data.fmt_money(float(first['value'].iloc[0]))}."
)
if plan:
    code = plan.get("code", {})
    st.caption(
        f"Plan scellé le {plan.get('planned_at', '?')[:16].replace('T', ' ')} UTC, "
        f"code `{str(code.get('commit', '?'))[:7]}`, tentative n° {plan.get('attempt', '?')}, "
        f"système principal annoncé à l'avance : **{data.label(plan.get('main_system', '?'))}**."
    )

table = period.metrics.set_index("strategy")
chosen = strategy_picker(period, "overview_strategy", "Système affiché dans les indicateurs")
row = table.loc[chosen]
ref = table.loc["buy_and_hold"] if "buy_and_hold" in table.index else None


def delta(col: str, pct: bool = True) -> str | None:
    if ref is None or chosen == "buy_and_hold":
        return None
    d = float(row[col] - ref[col])
    text = f"{d * 100:+.1f} pts vs B&H" if pct else f"{d:+.2f} vs B&H"
    return text.replace(".", ",")


c = st.columns(6)
c[0].metric("Profit net", data.fmt_money(row["net_profit"]))
c[1].metric("Rendement", data.fmt_pct(row["total_return"]), delta("total_return"))
c[2].metric("Sharpe", data.fmt_num(row["sharpe"]), delta("sharpe", pct=False))
c[3].metric(
    "Perte max (MDD)",
    data.fmt_pct(row["max_drawdown"]),
    delta("max_drawdown"),
    help="Plus grande baisse depuis un plus haut. Plus proche de 0 = mieux.",
)
c[4].metric("Trades", int(row["n_trades"]))
c[5].metric("Temps investi", f"{row['exposure'] * 100:.0f} %")

idle = [s for s in table.index if s.startswith("agents") and table.loc[s, "n_trades"] == 0]
if idle:
    st.info(
        "Ces systèmes n'ont pris **aucune position** sur la période (prévisions toujours sous "
        "le seuil d'achat ou trop incertaines) : "
        + ", ".join(data.label(s) for s in idle)
        + ". Leur résultat est donc 0 %. Voir la page **Décisions** pour les motifs."
    )

alerts = period.results.get("alerts")
if alerts:
    for a in alerts:
        st.warning(a)

st.subheader("Comparaison des stratégies")
st.dataframe(metrics_display(period.metrics), hide_index=True, width="stretch")

show_oracle = st.toggle(
    "Afficher l'oracle",
    value=False,
    help="L'oracle connaît le futur : c'est un plafond "
    "théorique, pas une stratégie possible. Il écrase l'échelle du graphique.",
)
plot = period.metrics[(period.metrics["strategy"] != "oracle") | show_oracle]
plot = plot.iloc[::-1]
fig = go.Figure(
    go.Bar(
        x=plot["total_return"],
        y=plot["strategy"].map(data.label),
        orientation="h",
        marker_color=[color(s) for s in plot["strategy"]],
        text=plot["total_return"].map(data.fmt_pct),
        textposition="outside",
        hovertemplate="%{y} : %{x:.1%}<extra></extra>",
    )
)
fig.update_layout(title="Rendement total sur la période", showlegend=False)
plot_layout(fig, height=60 + 38 * len(plot), y_format="")
fig.update_xaxes(tickformat=".0%", title_text="")
fig.update_layout(hovermode="closest")
st.plotly_chart(fig, width="stretch")

costs = period.results.get("costs_sensitivity")
if costs:
    with st.expander("Robustesse aux frais : rendement si les frais étaient 0×, 1×, 2× ou 5×"):
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
        plot_layout(fig, height=380, y_title="Rendement total", y_format=".0%")
        fig.update_xaxes(title_text="Multiplicateur des frais et du glissement", dtick=1)
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig, width="stretch")

refs = period.results.get("random_references")
if refs:
    with st.expander("Comparaison au hasard : 200 tirages avec le même nombre de trades"):
        frame = pd.DataFrame(refs)
        st.dataframe(
            pd.DataFrame(
                {
                    "Système": frame["strategy"].map(data.label),
                    "Trades": frame["n_trades"],
                    "Sharpe": frame["sharpe"].round(2),
                    "Sharpe médian du hasard": frame["random_sharpe_median"].round(2),
                    "Intervalle 5-95 % du hasard": [
                        f"{a:.2f} à {b:.2f}"
                        for a, b in zip(
                            frame["random_sharpe_p05"], frame["random_sharpe_p95"], strict=True
                        )
                    ],
                    "Part des tirages battus": frame["beats_random"].map(lambda x: f"{x:.0%}"),
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Un système qui ne bat pas la plupart des tirages au hasard n'apporte pas de "
            "preuve d'un vrai savoir-faire."
        )
