"""Tableau de bord Streamlit (§9) : six pages qui lisent seulement outputs/ et ledger/.

Lancer depuis la racine du dépôt avec : make dashboard
(ou : uv run --extra dashboard streamlit run dashboard/app.py)
"""

from __future__ import annotations

import streamlit as st
from common import run_sets

st.set_page_config(page_title="Gold Agents", page_icon=":material/monitoring:", layout="wide")

pages = [
    st.Page("pages/vue_ensemble.py", title="Vue d'ensemble", icon=":material/dashboard:"),
    st.Page("pages/previsions.py", title="Prévisions", icon=":material/insights:"),
    st.Page("pages/portefeuille.py", title="Portefeuille", icon=":material/account_balance:"),
    st.Page("pages/decisions.py", title="Décisions", icon=":material/gavel:"),
    st.Page("pages/blockchain.py", title="Blockchain", icon=":material/link:"),
    st.Page("pages/experiences.py", title="Expériences", icon=":material/science:"),
]
nav = st.navigation(pages)

with st.sidebar:
    sets = run_sets()
    if sets:
        names = [r.name for r in sets]
        titles = {r.name: r.title for r in sets}
        if st.session_state.get("period") not in names:
            st.session_state["period"] = names[0]
        st.selectbox("Registres affichés", names, format_func=titles.get, key="period")
    st.caption(
        "Investissement simulé sur l'or (XAU/USD). Toutes les données viennent des registres "
        "signés de `outputs/ledger/` : rien n'est recalculé ici."
    )

nav.run()
