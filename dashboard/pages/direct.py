"""Simulation en direct : le système jugé sur des prix qui n'existaient pas quand il a décidé."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import (
    BUY_COLOR,
    GOLD,
    ROOT,
    SELL_COLOR,
    blocks,
    color,
    header,
    kpis,
    plot_layout,
    section,
    tone,
    verify,
)

from gold_agents.dashboard import data
from gold_agents.evaluation import financial_metrics as fm

header(
    "Simulation en direct",
    "Chaque jour de cotation, la décision est signée et ancrée sur Sepolia avant que le "
    "prix suivant existe",
)

path = data.live_ledger(ROOT)
if not path.is_file():
    st.info(
        "La simulation en direct n'a pas encore démarré. Une fois lancée, chaque jour de "
        "cotation ajoute ici la prévision, la décision des agents et le résultat du "
        "portefeuille, comparé à Buy & Hold."
    )
    st.stop()

raw = blocks(path)
plan = data.genesis(raw)
settings = plan.get("settings", {})
days = data.live_days(raw)
check = verify(path)
if days.empty:
    st.info("Le plan est scellé ; le premier jour de cotation n'est pas encore inscrit.")
    st.stop()

last = days.iloc[-1]
capital = float(settings.get("initial_capital", days["value"].iloc[0]))
value, bh_value = float(last["value"]), float(last["bh_value"])
ret, bh_ret = value / capital - 1, bh_value / capital - 1
n_trades = int(days["trade_side"].notna().sum()) if "trade_side" in days else 0
on_time = int(days["on_time"].sum())
anchors = data.live_anchors(ROOT, path)

kpis(
    [
        ("Valeur des agents", data.fmt_money(value), data.fmt_pct(ret), tone(ret)),
        ("Buy & Hold", data.fmt_money(bh_value), data.fmt_pct(bh_ret), tone(bh_ret)),
        (
            "Écart",
            f"{(ret - bh_ret) * 100:+.1f} pts".replace(".", ","),
            "agents moins Buy & Hold",
            tone(ret - bh_ret),
        ),
        ("Jours de cotation", f"{len(days)}", f"depuis le {days['date'].min():%d/%m/%Y}", ""),
        ("Transactions", f"{n_trades}", "exécutées à l'ouverture", ""),
        (
            "Décisions à temps",
            f"{on_time}/{len(days)}",
            "écrites avant le prix suivant",
            "up" if on_time == len(days) else "",
        ),
    ],
    accent=0,
)

# Décision en attente : ce que les agents feront à la prochaine ouverture.
action = str(last.get("action", "HOLD"))
words = {"BUY": "ACHETER", "SELL": "VENDRE", "HOLD": "NE RIEN FAIRE"}
seuil = last.get("threshold")
rank_txt = "" if pd.isna(seuil) else f", seuil d'achat du jour {seuil:+.3%}"
r_txt = "pas de prévision" if pd.isna(last.get("r_hat")) else f"{last['r_hat']:+.3%}"
box = st.success if action == "BUY" else st.error if action == "SELL" else st.info
box(
    f"**Décision du {last['date']:%d/%m/%Y} pour la prochaine ouverture : "
    f"{words.get(action, action)}.** Prévision du rendement {r_txt}{rank_txt}. "
    f"Règle : {data.rule_label(last.get('rule'))}. {last.get('reason', '')}. "
    f"Écrite le {last['decided_at']:%d/%m/%Y à %H:%M} UTC."
)
if check["valid"]:
    anchored = int(anchors["blocks"].max()) if not anchors.empty else 0
    total = check["n_blocks"]
    st.caption(
        f"Registre intact : {total} blocs, signatures {check['signatures']}. "
        + (
            f"Dernier ancrage Sepolia : {anchored} blocs sur {total}."
            if anchored
            else "Pas encore d'ancrage Sepolia."
        )
    )
else:
    st.error(f"Registre en direct INVALIDE : {check['text']}")

section("Valeur du portefeuille")
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=days["date"],
        y=days["bh_value"],
        name="Buy & Hold",
        line={"color": color("buy_and_hold"), "width": 2},
        hovertemplate="%{y:,.0f}",
    )
)
fig.add_trace(
    go.Scatter(
        x=days["date"],
        y=days["value"],
        name=data.label("agents_live"),
        line={"color": GOLD, "width": 2.5},
        hovertemplate="%{y:,.0f}",
    )
)
for side, sym, col in (("BUY", "triangle-up", BUY_COLOR), ("SELL", "triangle-down", SELL_COLOR)):
    pts = days[days.get("trade_side") == side] if "trade_side" in days else days.iloc[0:0]
    if len(pts):
        fig.add_trace(
            go.Scatter(
                x=pts["date"],
                y=pts["value"],
                mode="markers",
                name=data.action_label(side),
                marker={"symbol": sym, "size": 12, "color": col},
                hovertemplate="%{x|%d/%m/%Y}<extra></extra>",
            )
        )
st.plotly_chart(plot_layout(fig, height=420, y_title="USD"), width="stretch")

if len(days) >= 5:
    values = days["value"].astype("float64").reset_index(drop=True)
    bh = days["bh_value"].astype("float64").reset_index(drop=True)
    stats = pd.DataFrame(
        {
            "": ["Agents en direct", "Buy & Hold"],
            "Rendement": [data.fmt_pct(ret), data.fmt_pct(bh_ret)],
            "Perte maximale": [
                data.fmt_pct(fm.max_drawdown(values), signed=False),
                data.fmt_pct(fm.max_drawdown(bh), signed=False),
            ],
            "Sharpe (annualisé)": [data.fmt_num(fm.sharpe(values)), data.fmt_num(fm.sharpe(bh))],
            "Temps investi": [f"{(days['qty'] > 0).mean():.0%}", "100 %"],
        }
    )
    st.dataframe(stats, hide_index=True, width="stretch")
    if len(days) < 60:
        st.caption(
            f"Seulement {len(days)} jours : trop peu pour conclure. Il faut plusieurs mois "
            "avant qu'un écart avec Buy & Hold veuille dire quelque chose."
        )

section("Prévision et seuil d'achat du jour")
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=days["date"],
        y=days["r_hat"],
        name="Prévision r̂",
        mode="lines+markers",
        line={"color": GOLD},
        hovertemplate="%{y:.3%}",
    )
)
fig.add_trace(
    go.Scatter(
        x=days["date"],
        y=days["threshold"],
        name="Seuil d'achat",
        line={"color": BUY_COLOR, "dash": "dash"},
        hovertemplate="%{y:.3%}",
    )
)
sell = settings.get("theta_sell")
if sell is not None:
    fig.add_hline(
        y=-float(sell), line={"color": SELL_COLOR, "dash": "dot"}, annotation_text="seuil de vente"
    )
fig.update_layout(title="Rendement prévu pour le lendemain et seuil d'achat")
st.plotly_chart(plot_layout(fig, height=320, y_format=".2%"), width="stretch")
if settings.get("buy_rule") == "quantile":
    st.caption(
        f"Le seuil d'achat est le quantile {data.fmt_pct(settings.get('theta_buy'), 0, False)} "
        f"des {settings.get('quantile_window')} prévisions précédentes du même modèle (jamais "
        "sous 0). Il suit l'échelle des prévisions, au lieu d'un seuil fixe en pourcentage "
        "qui n'était jamais atteint au test."
    )

section("Journal des décisions")
table = pd.DataFrame(
    {
        "Jour": days["date"].dt.strftime("%d/%m/%Y"),
        "Clôture": days["close"].map(lambda x: f"{x:,.2f}".replace(",", " ")),
        "Prévision": days["r_hat"].map(lambda x: "–" if pd.isna(x) else f"{x:+.3%}"),
        "Seuil d'achat": days["threshold"].map(lambda x: "–" if pd.isna(x) else f"{x:+.3%}"),
        "Décision": days["action"].map(data.action_label),
        "Règle": days["rule"].map(data.rule_label),
        "Exécuté à l'ouverture": (
            days["trade_side"].map(data.action_label).fillna("") if "trade_side" in days else ""
        ),
        "Valeur": days["value"].map(data.fmt_money),
        "Buy & Hold": days["bh_value"].map(data.fmt_money),
        "Écrite le (UTC)": days["decided_at"].dt.strftime("%d/%m %H:%M"),
        "À temps": days["on_time"].map({True: "oui", False: "en retard"}),
    }
).iloc[::-1]
st.dataframe(table, hide_index=True, width="stretch")
st.caption(
    "La décision d'un jour n'utilise que les prix jusqu'à sa clôture ; elle est exécutée à "
    "l'ouverture du jour de cotation suivant. « En retard » : jour rattrapé alors que le "
    "prix suivant existait déjà. Le résultat est le même (modèles et seuils figés), mais "
    "la preuve d'antériorité ne vaut que pour les décisions à temps."
)

section("Ancrages sur Sepolia")
if anchors.empty:
    st.caption("Aucun ancrage pour le moment.")
else:
    shown = anchors.sort_values("anchored_at", ascending=False)
    st.dataframe(
        pd.DataFrame(
            {
                "Ancré le (UTC)": shown["anchored_at"].dt.strftime("%d/%m/%Y %H:%M"),
                "Blocs couverts": shown["blocks"],
                "Bloc Sepolia": shown["block_number"],
                "État": shown["status"],
                "Transaction": shown["url"],
            }
        ),
        hide_index=True,
        width="stretch",
        column_config={"Transaction": st.column_config.LinkColumn(display_text="Etherscan")},
    )

section("Plan scellé avant le premier jour")
frozen = plan.get("frozen_settings", {})
if settings.get("buy_rule") == "quantile":
    rule = (
        f"**Achat** si la prévision dépasse le quantile "
        f"{data.fmt_pct(settings.get('theta_buy'), 0, False)} des "
        f"{settings.get('quantile_window')} prévisions précédentes"
    )
else:
    rule = f"**Achat** si la prévision dépasse {data.fmt_pct(settings.get('theta_buy'), 2)}"
lines = [
    f"- **Modèle** : {data.model_label(str(settings.get('model_id')))}, poids figés le "
    f"{str(plan.get('created_at', ''))[:10]} (entraînés jusqu'au "
    f"{plan.get('data', {}).get('last_date', '?')})",
    f"- {rule} ; **vente** si elle passe sous "
    f"{data.fmt_pct(-float(settings.get('theta_sell') or 0), 2)}",
    f"- **Stop-loss** -{data.fmt_pct(settings.get('stop_loss'), 0, False)}, **coupe-circuit** "
    f"-{data.fmt_pct(settings.get('max_drawdown'), 0, False)}",
    f"- **Frais** {data.fmt_pct(settings.get('fee_rate'), 2, False)} et **slippage** "
    f"{data.fmt_pct(settings.get('slippage'), 2, False)} par ordre",
    f"- **Règle gelée** : la même que pour le test (gel {str(frozen.get('head', '?'))[:12]}…), "
    "vérifiée au démarrage",
]
st.markdown("\n".join(lines))
