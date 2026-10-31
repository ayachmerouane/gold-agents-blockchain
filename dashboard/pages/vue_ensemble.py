"""Vue d'ensemble : indicateurs du système choisi, courbes, risque/rendement, années."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import (
    DEFAULT_COMPARE,
    GOLD,
    MUTED,
    color,
    header,
    kpis,
    metrics_display,
    plot_layout,
    require_period,
    section,
    strategy_picker,
    tone,
)

from gold_agents.dashboard import data
from gold_agents.evaluation import financial_metrics as fm

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
if plan.get("main_system") in table.index:
    right.caption(
        f"Système principal annoncé avant le test : **{data.label(plan['main_system'])}** "
        f"(plan scellé le {plan.get('planned_at', '?')[:10]})."
    )
row = table.loc[chosen]
ref = table.loc["buy_and_hold"] if "buy_and_hold" in table.index else None


def vs_ref(col: str, pct: bool = True) -> tuple[str, str]:
    if ref is None or chosen == "buy_and_hold":
        return "", ""
    d = float(row[col] - ref[col])
    text = f"{d * 100:+.1f} pts vs Buy & Hold" if pct else f"{d:+.2f} vs Buy & Hold"
    return text.replace(".", ","), tone(d)


ret_sub, ret_tone = vs_ref("total_return")
sh_sub, sh_tone = vs_ref("sharpe", pct=False)
dd_sub, dd_tone = vs_ref("max_drawdown")
kpis(
    [
        (
            "Valeur finale",
            data.fmt_money(row["final_value"]),
            f"profit net {data.fmt_money(row['net_profit'])}",
            tone(row["net_profit"]),
        ),
        ("Rendement total", data.fmt_pct(row["total_return"]), ret_sub, ret_tone),
        ("Rendement annuel", data.fmt_pct(row["cagr"]), "composé, sur la période", ""),
        ("Ratio de Sharpe", data.fmt_num(row["sharpe"]), sh_sub, sh_tone),
        ("Perte maximale", data.fmt_pct(row["max_drawdown"], signed=False), dd_sub, dd_tone),
        (
            "Transactions",
            f"{int(row['n_trades'])}",
            f"investi {row['exposure'] * 100:.0f} % du temps",
            "",
        ),
    ],
    accent=0,
)

idle = [s for s in table.index if s.startswith("agents") and table.loc[s, "n_trades"] == 0]
if chosen in idle:
    st.info(
        f"{data.label(chosen)} n'a pris aucune position : ses prévisions sont restées sous le "
        "seuil d'achat ou trop incertaines. Le détail jour par jour est sur la page Décisions.",
        icon=":material/info:",
    )
for alert in period.results.get("alerts") or []:
    st.warning(alert, icon=":material/warning:")

# --- Courbes de valeur --------------------------------------------------------------------
section("Performance comparée")
shown = [s for s in DEFAULT_COMPARE if s in frames]
if chosen not in shown:
    shown.append(chosen)
fig = go.Figure()
for name in shown:
    f = frames[name]
    is_main = name == chosen
    fig.add_trace(
        go.Scatter(
            x=f["date"],
            y=f["value"],
            name=data.label(name),
            line={"color": color(name), "width": 3 if is_main else 1.8},
            hovertemplate="%{y:,.0f} $",
        )
    )
fig.add_hline(y=float(first["value"].iloc[0]), line={"color": MUTED, "dash": "dot", "width": 1})
fig.update_layout(title="Valeur du portefeuille (10 000 $ investis au départ)")
plot_layout(fig, height=430, y_title="USD")
fig.update_yaxes(tickformat=",.0f")
st.plotly_chart(fig, width="stretch")

# --- Risque / rendement et rendements -----------------------------------------------------
left, right = st.columns(2)
fig = go.Figure()
for name in period.strategies:
    if name == "oracle":
        continue
    values = frames[name]["value"].astype("float64").reset_index(drop=True)
    fig.add_trace(
        go.Scatter(
            x=[fm.annual_volatility(values)],
            y=[float(table.loc[name, "cagr"])],
            mode="markers+text" if float(table.loc[name, "n_trades"]) > 0 else "markers",
            name=data.label(name),
            text=[data.label(name)],
            textposition="top center",
            textfont={"color": MUTED, "size": 11},
            marker={
                "size": 18 if name == chosen else 12,
                "color": color(name),
                "line": {"width": 2, "color": GOLD if name == chosen else "#0B1220"},
            },
            hovertemplate="volatilité %{x:.1%}<br>rendement annuel %{y:.1%}"
            "<extra>%{fullData.name}</extra>",
            showlegend=False,
        )
    )
idle_all = [s for s in period.strategies if s != "oracle" and table.loc[s, "n_trades"] == 0]
if idle_all:
    fig.add_annotation(
        x=0,
        y=0,
        text=f"{len(idle_all)} systèmes restés en liquidités",
        showarrow=True,
        arrowcolor=MUTED,
        ax=70,
        ay=-30,
        font={"color": MUTED, "size": 11},
    )
fig.update_layout(title="Risque et rendement annuels", hovermode="closest")
plot_layout(fig, height=400, y_title="Rendement annuel", y_format=".0%")
fig.update_xaxes(
    title_text="Volatilité annuelle", tickformat=".0%", showgrid=True, title_font={"color": MUTED}
)
left.plotly_chart(fig, width="stretch")

bars = period.metrics[period.metrics["strategy"] != "oracle"].sort_values("total_return")
fig = go.Figure(
    go.Bar(
        x=bars["total_return"],
        y=bars["strategy"].map(data.label),
        orientation="h",
        marker={
            "color": [color(s) for s in bars["strategy"]],
            "cornerradius": 4,
            "line": {"width": [2 if s == chosen else 0 for s in bars["strategy"]], "color": GOLD},
        },
        text=bars["total_return"].map(data.fmt_pct),
        textposition="outside",
        textfont={"color": MUTED},
        cliponaxis=False,
        hovertemplate="%{y} : %{x:.1%}<extra></extra>",
    )
)
fig.update_layout(title="Rendement total par stratégie", showlegend=False, hovermode="closest")
plot_layout(fig, height=400)
fig.update_xaxes(
    tickformat=".0%", showgrid=True, range=[0, float(bars["total_return"].max()) * 1.25]
)
right.plotly_chart(fig, width="stretch")

# --- Rendements annuels -------------------------------------------------------------------
yearly = {}
for name in period.strategies:
    if name == "oracle":
        continue
    f = frames[name].set_index("date")["value"].astype("float64")
    year_end = f.groupby(f.index.year).last()
    start = year_end.shift(1).fillna(float(f.iloc[0]))
    yearly[data.label(name)] = year_end / start - 1
heat = pd.DataFrame(yearly).T
if heat.shape[1] > 1:
    limit = float(heat.abs().max().max()) or 1.0
    fig = go.Figure(
        go.Heatmap(
            z=heat.to_numpy(),
            x=[str(y) for y in heat.columns],
            y=list(heat.index),
            colorscale=[[0, "#e5484d"], [0.5, "#1B2438"], [1, "#2fb344"]],
            zmin=-limit,
            zmax=limit,
            text=[[data.fmt_pct(v, 0) for v in r] for r in heat.to_numpy()],
            texttemplate="%{text}",
            textfont={"color": "#E6E9EF", "size": 12},
            xgap=3,
            ygap=3,
            showscale=False,
            hovertemplate="%{y}, %{x} : %{z:.1%}<extra></extra>",
        )
    )
    fig.update_layout(title="Rendement par année civile", hovermode="closest")
    plot_layout(fig, height=70 + 42 * len(heat))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    st.plotly_chart(fig, width="stretch")

# --- Tableau ------------------------------------------------------------------------------
section("Tableau comparatif")
st.dataframe(metrics_display(period.metrics), hide_index=True, width="stretch")
st.caption(
    "L'oracle connaît le prix du lendemain : c'est un plafond théorique, exclu des graphiques."
)

# --- Robustesse ---------------------------------------------------------------------------
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
                    hovertemplate="frais ×%{x} : %{y:.1%}<extra>%{fullData.name}</extra>",
                )
            )
        fig.update_layout(title="Rendement si les frais étaient multipliés", hovermode="closest")
        plot_layout(fig, height=400, y_title="Rendement total", y_format=".0%")
        fig.update_xaxes(title_text="Multiplicateur des frais et du glissement", dtick=1)
        fig.update_layout(legend={"y": -0.28}, margin={"b": 30})
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
