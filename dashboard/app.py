"""Tableau de bord Streamlit (§9) : sept pages qui lisent seulement outputs/ et ledger/.

Lancer depuis la racine du dépôt avec : make dashboard
(ou : uv run --extra dashboard streamlit run dashboard/app.py)
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from common import apply_style, run_sets

st.set_page_config(
    page_title="Gold Agents · Tableau de bord",
    page_icon=":material/monitoring:",
    layout="wide",
)
apply_style()
st.logo(str(Path(__file__).parent / "assets" / "logo.svg"), size="large")

pages = {
    "En direct": [
        st.Page("pages/direct.py", title="Simulation en direct", icon=":material/bolt:"),
    ],
    "Performance": [
        st.Page("pages/vue_ensemble.py", title="Vue d'ensemble", icon=":material/dashboard:"),
        st.Page("pages/portefeuille.py", title="Portefeuille", icon=":material/account_balance:"),
        st.Page("pages/decisions.py", title="Décisions", icon=":material/gavel:"),
    ],
    "Modèles": [
        st.Page("pages/previsions.py", title="Prévisions", icon=":material/insights:"),
        st.Page("pages/experiences.py", title="Expériences", icon=":material/science:"),
    ],
    "Traçabilité": [
        st.Page("pages/blockchain.py", title="Registre blockchain", icon=":material/link:"),
    ],
}
nav = st.navigation(pages)

with st.sidebar:
    sets = run_sets()
    if sets:
        names = [r.name for r in sets]
        titles = {r.name: r.title for r in sets}
        if st.session_state.get("period") not in names:
            st.session_state["period"] = names[0]
        st.selectbox("Période", names, format_func=titles.get, key="period")
    st.caption("Mode clair ou sombre : menu ⋮ en haut à droite (Light, Dark ou System).")

nav.run()
