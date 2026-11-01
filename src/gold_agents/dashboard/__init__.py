"""Lecture des sorties pour le tableau de bord Streamlit (§9).

Ce paquet ne dépend ni de Streamlit ni de Plotly : il lit les registres (`outputs/ledger/`,
`ledger/`) et les résultats facultatifs (`outputs/results/`), et renvoie des tableaux
pandas. Les pages de `dashboard/` ne font que les afficher.
"""
