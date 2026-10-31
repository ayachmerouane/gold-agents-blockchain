"""Blockchain : explorateur de blocs, vérification, démo de falsification, ancrages."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st
from common import PUBLIC_KEY, ROOT, blocks, require_period, verify

from gold_agents.dashboard import data

period = require_period()
st.title("Blockchain : le registre des décisions")
st.markdown(
    "Chaque jour simulé est un **bloc** : il contient les événements du jour (prévision, "
    "signal, ordre, trade, portefeuille), leur racine de Merkle, le hash du bloc précédent, "
    "une preuve de travail et une **signature Ed25519**. Modifier un ancien bloc casse "
    "tous les liens suivants ; refaire les liens exige la clé privée, qui n'a jamais quitté "
    "le PC de l'auteur."
)
fp = data.key_fingerprint(PUBLIC_KEY)
st.caption(
    f"Clé publique : `ledger/public_key.pem` (empreinte `{fp}`)."
    if fp
    else "Clé publique `ledger/public_key.pem` absente : signatures non vérifiables."
)

files = sorted(period.run.ledger_dir.glob("*.jsonl"), key=lambda p: data.strategy_order(p.stem))
tab_explore, tab_verify, tab_tamper, tab_anchor = st.tabs(
    ["Explorateur de blocs", "Vérifier l'intégrité", "Démo de falsification", "Ancrage Sepolia"]
)


def file_label(p: object) -> str:
    stem = getattr(p, "stem", str(p))
    extra = {"plan": "plan du test (scellé avant le run)", "resultats": "résultats scellés"}
    return f"{stem}.jsonl : {extra.get(stem, data.label(stem))}"


# --- Explorateur --------------------------------------------------------------------------
with tab_explore:
    path = st.selectbox("Registre", files, format_func=file_label, key="explore_file")
    chain = blocks(path)
    n = len(chain)
    index = int(st.number_input(f"Numéro du bloc (0 à {n - 1})", 0, n - 1, 0, key="explore_i"))
    block = chain[index]
    st.markdown(
        f"**Bloc {block['index']}** · date simulée **{block.get('sim_date') or 'genèse'}** · "
        f"version {block.get('version', 1)} · {len(block['events'])} événement(s)"
    )
    c = st.columns(3)
    c[0].caption("hash du bloc précédent")
    c[0].code(block["prev_hash"], language=None)
    c[1].caption("hash de ce bloc")
    c[1].code(block["hash"], language=None)
    c[2].caption("bloc suivant pointe vers")
    c[2].code(chain[index + 1]["prev_hash"] if index + 1 < n else "(dernier bloc)", language=None)
    head = pd.DataFrame(
        [
            ("racine de Merkle", block.get("merkle_root", "–")),
            ("difficulté (zéros en tête)", block.get("difficulty", "–")),
            ("nonce (preuve de travail)", block.get("nonce", "–")),
            ("signature", (block.get("signature") or "–")[:40] + "…"),
        ],
        columns=["champ", "valeur"],
    )
    st.dataframe(head.astype(str), hide_index=True, width="stretch")
    st.caption("Événements du bloc")
    if index == 0:
        st.json(block["events"], expanded=1)
    else:
        for event in block["events"]:
            st.json(event, expanded=True)

    st.caption("Aperçu des blocs voisins")
    lo = max(0, index - 5)
    near = pd.DataFrame(
        [
            {
                "bloc": b["index"],
                "date": b.get("sim_date") or "genèse",
                "événements": ", ".join(e.get("type", "?") for e in b["events"]),
                "prev_hash": b["prev_hash"][:16] + "…",
                "hash": b["hash"][:16] + "…",
            }
            for b in chain[lo : lo + 11]
        ]
    )
    st.dataframe(near, hide_index=True, width="stretch")

# --- Vérification -------------------------------------------------------------------------
with tab_verify:
    st.markdown(
        "La vérification recalcule pour chaque bloc : le lien `prev_hash`, la racine de "
        "Merkle des événements, le hash, la preuve de travail et la signature avec la clé "
        "publique publiée. C'est le même code que `make verify`."
    )
    if st.button("Vérifier l'intégrité de tous les registres", type="primary"):
        rows = []
        bar = st.progress(0.0)
        for i, p in enumerate(files):
            v = verify(p)
            rows.append(
                {
                    "registre": p.name,
                    "blocs": v["n_blocks"],
                    "état": "valide" if v["valid"] else "INVALIDE",
                    "signatures": v["signatures"] if v["valid"] else "–",
                    "détail": v["text"],
                }
            )
            bar.progress((i + 1) / len(files))
        bar.empty()
        st.session_state["verify_rows"] = rows
    rows = st.session_state.get("verify_rows")
    if rows:
        bad = [r for r in rows if r["état"] != "valide"]
        if bad:
            st.error(
                f"{len(bad)} registre(s) invalide(s) : " + ", ".join(r["registre"] for r in bad)
            )
        else:
            st.success(
                f"Les {len(rows)} registres sont intacts : "
                f"{sum(r['blocs'] for r in rows)} blocs vérifiés, signatures comprises."
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "Le professeur peut faire la même vérification sans installer le projet : "
        "`uv run scripts/verify_ledger.py` (Python et `cryptography` seulement)."
    )

# --- Falsification ------------------------------------------------------------------------
with tab_tamper:
    st.markdown(
        "On falsifie une **copie en mémoire** du registre (le fichier n'est jamais modifié), "
        "puis on relance la vérification. Exemple : gonfler après coup la valeur du "
        "portefeuille un jour donné."
    )
    strategies = [p for p in files if p.stem not in data.SPECIAL_LEDGERS]
    path = st.selectbox(
        "Registre à falsifier", strategies, format_func=file_label, key="tamper_file"
    )
    chain = blocks(path)
    c = st.columns(3)
    index = int(c[0].number_input("Bloc", 1, len(chain) - 1, len(chain) // 2, key="tamper_i"))
    current = next(
        (e["value"] for e in chain[index]["events"] if e.get("type") == "PORTFOLIO"), 10000.0
    )
    new_value = c[1].number_input(
        "Nouvelle valeur du portefeuille ($)",
        value=float(current) * 1.5,
        step=1000.0,
        key=f"tamper_v_{path.stem}_{index}",
    )
    c[2].metric(
        "Valeur enregistrée",
        data.fmt_money(float(current)),
        f"{chain[index].get('sim_date')}",
        delta_color="off",
    )
    mode = st.radio(
        "Ce que fait le fraudeur",
        list(data.TAMPER_MODES),
        format_func=data.TAMPER_MODES.get,
        key="tamper_mode",
    )
    if st.button("Falsifier la copie et vérifier", type="primary"):
        with st.spinner("Falsification et vérification…"):
            result = data.tamper_value(path, index, float(new_value), mode, PUBLIC_KEY)
        v = result.verification
        if v.valid:
            st.warning(f"Non détecté : {v.describe()}")
        else:
            st.error(f"Falsification détectée au bloc {v.bad_index} : {v.reason}")
        explain = {
            "edit": "Le contenu ne correspond plus à la racine de Merkle enregistrée.",
            "rehash": "Le bloc re-haché perd sa signature (seul l'auteur peut signer) et le "
            "bloc suivant pointe toujours vers l'ancien hash.",
            "rewrite": "Le fraudeur a tout re-haché et re-signé, mais avec SA clé : les "
            "signatures ne correspondent pas à la clé publique publiée.",
        }
        st.caption(
            explain[mode] + " Vérification du fichier d'origine : " + verify(path)["text"] + "."
        )

    with st.expander("Les 5 scénarios de la phase 7 en un clic"):
        if st.button("Lancer les scénarios", key="tamper_all"):
            with st.spinner("Cinq copies falsifiées, cinq vérifications…"):
                st.session_state["tamper_table"] = data.tamper_scenarios(path, PUBLIC_KEY)
        table = st.session_state.get("tamper_table")
        if table is not None:
            st.dataframe(table, hide_index=True, width="stretch")

# --- Ancrage ------------------------------------------------------------------------------
with tab_anchor:
    st.markdown(
        "Le hash de tête des registres est publié dans une transaction du réseau de test "
        "Ethereum **Sepolia**. La blockchain publique date la transaction : elle prouve que "
        "les registres existaient déjà, tels quels, à ce moment."
    )
    anchors = data.anchor_checks(ROOT)
    if not anchors:
        st.info("Aucun ancrage dans `ledger/anchors.json`.")
    for a in reversed(anchors):
        icon = {"conforme": ":material/verified:", "DIFFÉRENT": ":material/error:"}.get(
            a["status"], ":material/help:"
        )
        with st.container(border=True):
            st.markdown(
                f"{icon} **{a['anchored_at'][:16].replace('T', ' ')} UTC** · bloc Sepolia "
                f"{a['block_number']} · état : **{a['status']}**"
            )
            st.link_button("Voir la transaction sur Etherscan", a["url"])
            st.caption(
                f"Empreinte publiée : `gold-agents ledger sha256:{a['digest']}` "
                "(sur Etherscan : « View input as UTF-8 »)."
            )
            st.dataframe(a["files"], hide_index=True, width="stretch")
    st.caption(
        "« absent de ce PC » : le registre n'est pas versionné (par exemple ceux de la "
        "validation) ; seuls les registres présents peuvent être comparés."
    )

with st.expander("Ce que la blockchain garantit, et ce qu'elle ne garantit pas"):
    st.markdown(
        "- Elle garantit l'**intégrité** (aucune décision modifiée, ajoutée ou supprimée) "
        "et l'**antériorité** (les décisions existaient à la date de l'ancrage).\n"
        "- Elle ne garantit **pas** que les prévisions sont bonnes, que la stratégie est "
        "rentable, ni que le code qui a produit les décisions était correct."
    )
    plan = period.plan
    if plan:
        st.caption("Bloc genèse du plan du test (réglages figés avant le run) :")
        st.code(
            json.dumps(
                {
                    k: plan[k]
                    for k in ("planned_at", "settings", "main_system", "frozen_settings")
                    if k in plan
                },
                ensure_ascii=False,
                indent=2,
            ),
            language="json",
        )
