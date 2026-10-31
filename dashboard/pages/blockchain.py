"""Registre blockchain : explorateur de blocs, vérification, démo de falsification, ancrages."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from common import PUBLIC_KEY, ROOT, blocks, header, require_period, section, verify

from gold_agents.dashboard import data

period = require_period()
header(
    "Registre blockchain",
    "Chaque jour simulé est un bloc signé, chaîné au précédent et ancré sur Ethereum",
)

files = sorted(period.run.ledger_dir.glob("*.jsonl"), key=lambda p: data.strategy_order(p.stem))
strategy_files = [p for p in files if p.stem not in data.SPECIAL_LEDGERS]
n_blocks = sum(len(blocks(p)) for p in files)
anchors = data.anchor_checks(ROOT)
fp = data.key_fingerprint(PUBLIC_KEY)

c = st.columns(4)
c[0].metric("Registres", len(files), border=True)
c[1].metric("Blocs enregistrés", f"{n_blocks:,}".replace(",", " "), border=True)
c[2].metric("Ancrages Ethereum", len(anchors), border=True)
c[3].metric(
    "Clé de signature",
    fp[:8] if fp else "absente",
    help="Empreinte de la clé publique Ed25519 qui vérifie les signatures.",
    border=True,
)


def file_label(p: object) -> str:
    return data.ledger_label(getattr(p, "stem", str(p)))


tab_explore, tab_verify, tab_tamper, tab_anchor = st.tabs(
    [
        ":material/search: Explorateur de blocs",
        ":material/verified: Vérifier l'intégrité",
        ":material/gpp_bad: Démo de falsification",
        ":material/public: Ancrage public",
    ]
)

# --- Explorateur --------------------------------------------------------------------------
with tab_explore:
    left, right = st.columns([2, 1])
    path = left.selectbox("Registre", files, format_func=file_label, key="explore_file")
    chain = blocks(path)
    n = len(chain)
    index = int(right.number_input(f"Bloc (0 à {n - 1})", 0, n - 1, min(1, n - 1), key="explore_i"))
    block = chain[index]
    when = block.get("sim_date") or "bloc genèse"
    st.markdown(f"#### Bloc {block['index']} · {when}")

    with st.container(border=True):
        c = st.columns(3)
        c[0].caption("Hash du bloc précédent")
        c[0].code(block["prev_hash"][:24] + "…", language=None)
        c[1].caption("Hash de ce bloc")
        c[1].code(block["hash"][:24] + "…", language=None)
        c[2].caption("Repris par le bloc suivant")
        c[2].code(
            chain[index + 1]["prev_hash"][:24] + "…" if index + 1 < n else "dernier bloc",
            language=None,
        )
        c = st.columns(4)
        c[0].caption("Racine de Merkle")
        c[0].markdown(f"`{str(block.get('merkle_root', '–'))[:16]}…`")
        c[1].caption("Preuve de travail")
        c[1].markdown(f"{block.get('difficulty', '–')} zéros · nonce {block.get('nonce', '–')}")
        c[2].caption("Signature Ed25519")
        c[2].markdown(f"`{(block.get('signature') or '–')[8:24]}…`")
        c[3].caption("Événements")
        c[3].markdown(str(len(block["events"])))

    section("Contenu du bloc")
    events = pd.DataFrame(
        [
            {
                "Événement": data.EVENT_LABELS.get(str(e.get("type")), str(e.get("type"))),
                "Détail": data.event_summary(e),
            }
            for e in block["events"]
        ]
    )
    st.dataframe(events, hide_index=True, width="stretch")
    with st.expander("Données brutes du bloc"):
        st.json(block["events"], expanded=1 if index == 0 else 2)

    section("Blocs voisins")
    lo = max(0, index - 5)
    near = pd.DataFrame(
        [
            {
                "Bloc": b["index"],
                "Date": b.get("sim_date") or "genèse",
                "Contenu": ", ".join(
                    data.EVENT_LABELS.get(str(e.get("type")), "?") for e in b["events"]
                ),
                "Hash précédent": b["prev_hash"][:14] + "…",
                "Hash": b["hash"][:14] + "…",
            }
            for b in chain[lo : lo + 11]
        ]
    )
    st.dataframe(near, hide_index=True, width="stretch")

# --- Vérification -------------------------------------------------------------------------
with tab_verify:
    st.markdown(
        "Pour chaque bloc, la vérification recalcule le lien avec le bloc précédent, la "
        "racine de Merkle des événements, le hash, la preuve de travail, et contrôle la "
        "signature avec la clé publique publiée."
    )
    if st.button("Vérifier tous les registres", type="primary", icon=":material/verified:"):
        rows = []
        bar = st.progress(0.0)
        for i, p in enumerate(files):
            v = verify(p)
            rows.append(
                {
                    "Registre": file_label(p),
                    "Blocs": v["n_blocks"],
                    "État": "Intact" if v["valid"] else "Altéré",
                    "Signatures": "Vérifiées" if v["valid"] else "–",
                    "Détail": "" if v["valid"] else f"bloc {v['bad_index']} : {v['reason']}",
                }
            )
            bar.progress((i + 1) / len(files))
        bar.empty()
        st.session_state["verify_rows"] = rows
    rows = st.session_state.get("verify_rows")
    if rows:
        bad = [r for r in rows if r["État"] != "Intact"]
        if bad:
            st.error(
                f"{len(bad)} registre(s) altéré(s) : " + ", ".join(r["Registre"] for r in bad),
                icon=":material/error:",
            )
        else:
            total = f"{sum(r['Blocs'] for r in rows):,}".replace(",", " ")
            st.success(
                f"Les {len(rows)} registres sont intacts : {total} blocs vérifiés, "
                "signatures comprises.",
                icon=":material/verified:",
            )
        table = pd.DataFrame(rows)
        if not bad:
            table = table.drop(columns=["Détail"])
        st.dataframe(table, hide_index=True, width="stretch")
    st.caption(
        "La même vérification existe en script autonome, livré avec le dépôt, pour qu'un "
        "tiers la refasse sans installer le projet."
    )

# --- Falsification ------------------------------------------------------------------------
with tab_tamper:
    st.markdown(
        "On falsifie une **copie en mémoire** du registre, jamais le fichier lui-même, puis "
        "on relance la vérification. Exemple : gonfler après coup la valeur du portefeuille "
        "un jour donné."
    )
    path = st.selectbox(
        "Registre à falsifier", strategy_files, format_func=file_label, key="tamper_file"
    )
    chain = blocks(path)
    c = st.columns(3, vertical_alignment="bottom")
    index = int(c[0].number_input("Bloc", 1, len(chain) - 1, len(chain) // 2, key="tamper_i"))
    current = next(
        (e["value"] for e in chain[index]["events"] if e.get("type") == "PORTFOLIO"), 10000.0
    )
    new_value = c[1].number_input(
        "Nouvelle valeur du portefeuille ($)",
        value=round(float(current) * 1.5, 2),
        step=1000.0,
        key=f"tamper_v_{path.stem}_{index}",
    )
    c[2].metric(
        f"Valeur enregistrée le {chain[index].get('sim_date')}",
        data.fmt_money(float(current)),
        border=True,
    )
    mode = st.radio(
        "Ce que tente le fraudeur",
        list(data.TAMPER_MODES),
        format_func=lambda m: data.TAMPER_MODES[m][:1].upper() + data.TAMPER_MODES[m][1:],
        key="tamper_mode",
    )
    if st.button("Falsifier la copie et vérifier", type="primary", icon=":material/gpp_bad:"):
        with st.spinner("Falsification et vérification…"):
            result = data.tamper_value(path, index, float(new_value), mode, PUBLIC_KEY)
        v = result.verification
        if v.valid:
            st.warning(f"Non détecté : {v.describe()}", icon=":material/warning:")
        else:
            st.error(
                f"Falsification détectée au bloc {v.bad_index} : {v.reason}.",
                icon=":material/gpp_bad:",
            )
        explain = {
            "edit": "Le contenu ne correspond plus à la racine de Merkle enregistrée.",
            "rehash": "Le bloc re-haché perd sa signature (seul l'auteur peut signer) et le "
            "bloc suivant pointe toujours vers l'ancien hash.",
            "rewrite": "Le fraudeur a tout re-haché et re-signé, mais avec sa propre clé : les "
            "signatures ne correspondent pas à la clé publique publiée.",
        }
        intact = verify(path)["valid"]
        st.caption(explain[mode] + (" Le registre d'origine, lui, est intact." if intact else ""))

    with st.expander("Rejouer les cinq scénarios de falsification"):
        if st.button("Lancer les scénarios", key="tamper_all"):
            with st.spinner("Cinq copies falsifiées, cinq vérifications…"):
                st.session_state["tamper_table"] = data.tamper_scenarios(path, PUBLIC_KEY)
        table = st.session_state.get("tamper_table")
        if table is not None:
            shown = table.rename(
                columns={
                    "falsification": "Falsification",
                    "détectée": "Détectée",
                    "bloc fautif": "Bloc fautif",
                    "motif": "Motif",
                }
            )
            st.dataframe(shown, hide_index=True, width="stretch")

# --- Ancrage ------------------------------------------------------------------------------
with tab_anchor:
    st.markdown(
        "L'empreinte des registres est publiée dans une transaction du réseau de test "
        "Ethereum **Sepolia**. La date de la transaction prouve que les registres existaient "
        "déjà, tels quels, à ce moment-là."
    )
    if not anchors:
        st.info("Aucun ancrage public pour le moment.")
    for a in reversed(anchors):
        ok = a["status"] == "conforme"
        badge = {
            "conforme": ":green-badge[:material/verified: Conforme]",
            "DIFFÉRENT": ":red-badge[:material/error: Différent]",
        }.get(a["status"], ":gray-badge[Partiel : registres non présents]")
        with st.container(border=True):
            left, right = st.columns([4, 1], vertical_alignment="center")
            left.markdown(
                f"**{a['anchored_at'][:16].replace('T', ' à ')} UTC** · bloc Ethereum "
                f"{a['block_number']:,} · {badge}".replace(",", " ")
            )
            right.link_button("Etherscan", a["url"], icon=":material/open_in_new:", width="stretch")
            files_table = a["files"].copy()
            files_table["fichier"] = files_table["fichier"].map(data.ledger_file_label)
            files_table["hash de tête ancré"] = files_table["hash de tête ancré"].str[:16] + "…"
            files_table["état"] = (
                files_table["état"]
                .map({"identique": "Identique", "DIFFÉRENT": "Différent"})
                .fillna("Non présent")
            )
            files_table.columns = ["Registre", "Blocs", "Hash de tête ancré", "État"]
            with st.expander(f"{len(files_table)} registre(s) couverts", expanded=ok):
                st.dataframe(files_table, hide_index=True, width="stretch")
    st.caption(
        "« Non présent » : registre de travail conservé hors du dépôt (par exemple sur la "
        "validation) ; seuls les registres présents peuvent être comparés."
    )

section("Ce que la blockchain garantit")
c = st.columns(2)
with c[0].container(border=True):
    st.markdown(
        "**Garanti**\n\n- Intégrité : aucune décision modifiée, ajoutée ou supprimée.\n"
        "- Antériorité : les décisions existaient à la date de l'ancrage."
    )
with c[1].container(border=True):
    st.markdown(
        "**Non garanti**\n\n- Que les prévisions soient bonnes ou la stratégie rentable.\n"
        "- Que le code qui a produit les décisions soit correct."
    )
