"""Données du tableau de bord : tout vient des registres et des résultats déjà calculés.

Les registres du test final (`outputs/ledger/test/`) sont versionnés et contiennent tout :
chaque bloc d'un jour porte la décision, la prévision, l'ordre, le trade et la valeur du
portefeuille. Le tableau de bord marche donc sur un simple clone du dépôt. Les CSV et
figures de `outputs/results/` (non versionnés) et la base MLflow (`mlruns/`) ne servent
qu'en complément, s'ils sont présents.
"""

from __future__ import annotations

import copy
import json
import math
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from gold_agents.blockchain.anchor import LedgerHead, anchor_digest, etherscan_url
from gold_agents.blockchain.archive import read_moves, resolve
from gold_agents.blockchain.chain import Chain, Verification, rewrite_from, tamper_demo
from gold_agents.blockchain.signing import Signer, fingerprint, load_public_key
from gold_agents.blockchain.storage import read_jsonl
from gold_agents.evaluation import financial_metrics as fm

# Registres qui ne sont pas une stratégie : plan du test et résultats scellés.
SPECIAL_LEDGERS = ("plan", "resultats")
# Ordre d'affichage : références simples, agents, puis hasard et oracle.
STRATEGY_ORDER = (
    "buy_and_hold",
    "sma_20_50",
    "agents_lstm",
    "agents_gru",
    "agents_ensemble",
    "agents_ensemble_mc",
    "agents_lightgbm",
    "aleatoire",
    "oracle",
)
LABELS = {
    "buy_and_hold": "Buy & Hold",
    "sma_20_50": "SMA 20/50",
    "agents_lstm": "Agents (LSTM)",
    "agents_gru": "Agents (GRU)",
    "agents_ensemble": "Agents (ensemble)",
    "agents_ensemble_mc": "Agents (ensemble + incertitude)",
    "agents_lightgbm": "Agents (LightGBM)",
    "aleatoire": "Hasard",
    "oracle": "Oracle (connaît le futur)",
}


def period_title(name: str) -> str:
    """Nom lisible d'une période ; `test_tentative1` : « Test final, tentative 1 »."""
    base, _, n = name.partition("_tentative")
    if n:
        return f"{PERIOD_LABELS.get(base, base).split(' (')[0]}, tentative {n} (archivée)"
    return PERIOD_LABELS.get(name, name)


PERIOD_LABELS = {
    "test": "Test final (2022-2026)",
    "final_val": "Répétition sur la validation",
    "agents_val": "Agents sur la validation",
}


def label(name: str) -> str:
    return LABELS.get(name, name)


def strategy_order(name: str) -> tuple[int, str]:
    return (STRATEGY_ORDER.index(name) if name in STRATEGY_ORDER else len(STRATEGY_ORDER), name)


# --- Jeux de registres ------------------------------------------------------------------


@dataclass(frozen=True)
class RunSet:
    """Un dossier de registres (une période) et ses résultats éventuels."""

    name: str
    ledger_dir: Path
    strategies: tuple[str, ...]
    results_dir: Path | None

    @property
    def title(self) -> str:
        return period_title(self.name)

    def ledger(self, strategy: str) -> Path:
        return self.ledger_dir / f"{strategy}.jsonl"


def run_sets(root: Path) -> list[RunSet]:
    """Les dossiers de `outputs/ledger/` qui contiennent des registres, le test en premier."""
    base = root / "outputs" / "ledger"
    found = []
    for d in sorted(p for p in base.iterdir() if p.is_dir()) if base.is_dir() else []:
        stems = [p.stem for p in d.glob("*.jsonl") if p.stem not in SPECIAL_LEDGERS]
        if not stems:
            continue
        results = None
        for candidate in (f"final_{d.name}", d.name):
            if (root / "outputs" / "results" / candidate).is_dir():
                results = root / "outputs" / "results" / candidate
                break
        found.append(RunSet(d.name, d, tuple(sorted(stems, key=strategy_order)), results))
    return sorted(found, key=lambda r: (r.name != "test", r.name))


# --- Lecture des blocs --------------------------------------------------------------------


def read_blocks(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def genesis(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """Les événements du bloc genèse, fusionnés en un seul dictionnaire."""
    out: dict[str, Any] = {}
    if blocks:
        for event in blocks[0]["events"]:
            out.update(event)
    return out


def daily_frame(blocks: list[dict[str, Any]]) -> pd.DataFrame:
    """Un jour par ligne : valeur du portefeuille, décision et son motif, prévision, ordre."""
    rows = []
    for block in blocks:
        if block.get("sim_date") is None:
            continue
        row: dict[str, Any] = {"date": block["sim_date"], "block": block["index"]}
        for event in block["events"]:
            kind = event.get("type")
            if kind == "PORTFOLIO":
                row.update(value=event["value"], cash=event["cash"], qty=event["qty"])
            elif kind == "DECISION":
                row["action"] = event.get("action")
                row["reason"] = event.get("reason")
                if "close" in event:
                    row["close"] = event["close"]
            elif kind == "FORECAST":
                row.update(
                    model_id=event.get("model_id"),
                    r_hat=event.get("r_hat"),
                    sigma_hat=event.get("sigma_hat"),
                    z=event.get("z"),
                )
            elif kind == "SIGNAL":
                row.update(
                    signal=event.get("action"),
                    signal_reason=event.get("reason"),
                    strength=event.get("strength"),
                )
            elif kind == "ORDER":
                # L'ordre validé par l'agent risque est la décision finale du jour.
                row.update(
                    action=event.get("action"),
                    reason=event.get("reason"),
                    approved=event.get("approved"),
                    rule=event.get("rule"),
                    fraction=event.get("fraction"),
                )
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"])
    for col in ("r_hat", "sigma_hat", "z", "strength", "close", "fraction"):
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def trades_frame(blocks: list[dict[str, Any]]) -> pd.DataFrame:
    """Les ordres exécutés (prix d'exécution, frais, glissement)."""
    rows = [
        {"block": b["index"], **e} for b in blocks for e in b["events"] if e.get("type") == "TRADE"
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=["block", "decision_date", "exec_date", "side", "qty", "price", "fees"]
        )
    frame = frame.drop(columns=["type"])
    for col in ("decision_date", "exec_date"):
        frame[col] = pd.to_datetime(frame[col])
    return frame


def prices(frames: dict[str, pd.DataFrame]) -> pd.Series:
    """Clôture de l'or jour par jour, lue dans un registre qui la contient (Buy & Hold)."""
    for name in sorted(frames, key=lambda n: (n != "buy_and_hold", strategy_order(n))):
        frame = frames[name]
        if "close" in frame and frame["close"].notna().sum() > 1:
            return frame.set_index("date")["close"].dropna().rename("close")
    return pd.Series(dtype="float64", name="close")


def realized_returns(close: pd.Series) -> pd.Series:
    """Rendement log du jour t à t+1 : ce que la prévision du jour t cherche à prédire."""
    ratio = (close.shift(-1) / close).astype("float64")
    return pd.Series(np.log(ratio.to_numpy()), index=close.index, name="target_return")


# --- Résultats -----------------------------------------------------------------------------


def results_events(ledger_dir: Path) -> dict[str, Any]:
    """Les tableaux scellés dans `resultats.jsonl` (métriques, frais, hasard, prévisions)."""
    path = ledger_dir / "resultats.jsonl"
    if not path.is_file():
        return {}
    out: dict[str, Any] = {}
    for block in read_blocks(path):
        for event in block["events"]:
            kind = event.get("type")
            if kind == "METRICS":
                out["metrics"] = event["metrics"]
            elif kind in ("COSTS_SENSITIVITY", "RANDOM_REFERENCES", "FORECAST_METRICS"):
                out[kind.lower()] = event["rows"]
            elif kind == "ALERTS":
                out["alerts"] = event["alerts"]
            elif kind == "GENESIS":
                out["genesis"] = event
    return out


def computed_metrics(name: str, frame: pd.DataFrame, trades: pd.DataFrame) -> dict[str, Any]:
    """Métriques recalculées depuis la courbe de valeur (quand rien n'est scellé)."""
    values = frame["value"].astype("float64").reset_index(drop=True)
    initial = float(values.iloc[0])
    return {
        "strategy": name,
        "initial_capital": initial,
        "final_value": float(values.iloc[-1]),
        "net_profit": float(values.iloc[-1] - initial),
        "total_return": float(values.iloc[-1] / initial - 1),
        "cagr": fm.cagr(values, frame["date"].reset_index(drop=True)),
        "sharpe": fm.sharpe(values),
        "max_drawdown": fm.max_drawdown(values),
        "n_trades": len(trades),
        "exposure": float((frame["qty"] > 0).mean()),
        "total_costs": _costs(trades),
    }


def _costs(trades: pd.DataFrame) -> float:
    cols = [c for c in ("fees", "slippage_cost") if c in trades]
    return float(trades[cols].to_numpy("float64").sum()) if len(trades) and cols else 0.0


METRIC_COLUMNS = [
    "strategy",
    "final_value",
    "net_profit",
    "total_return",
    "cagr",
    "sharpe",
    "max_drawdown",
    "n_trades",
    "exposure",
    "total_costs",
]


def metrics_table(
    sealed: list[dict[str, Any]] | None,
    frames: dict[str, pd.DataFrame],
    trades: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Une ligne par stratégie : les métriques scellées si elles existent, sinon recalculées."""
    by_name = {m["strategy"]: m for m in sealed or []}
    rows = []
    for name in sorted(frames, key=strategy_order):
        if name in by_name:
            rows.append({c: by_name[name].get(c) for c in METRIC_COLUMNS})
        elif not frames[name].empty:
            rows.append(computed_metrics(name, frames[name], trades[name]))
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def rule_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """Combien de fois chaque règle a décidé, par action (qui a bloqué quoi)."""
    if "rule" not in frame:
        return pd.DataFrame(columns=["rule", "action", "jours"])
    counts = frame.dropna(subset=["rule"]).groupby(["rule", "action"]).size()
    return counts.rename("jours").reset_index()


# --- Blockchain ---------------------------------------------------------------------------


def _key(public_key_path: Path | None) -> Ed25519PublicKey | None:
    if public_key_path is None or not public_key_path.is_file():
        return None
    return load_public_key(public_key_path)


def verify(path: Path, public_key_path: Path | None) -> Verification:
    key = _key(public_key_path)
    return read_jsonl(path).verify(key)


def key_fingerprint(public_key_path: Path) -> str | None:
    if not public_key_path.is_file():
        return None
    return fingerprint(load_public_key(public_key_path))


TAMPER_MODES = {
    "edit": "modifier la valeur seulement",
    "rehash": "modifier puis recalculer le hash du bloc (preuve de travail refaite)",
    "rewrite": "modifier puis réécrire toute la suite de la chaîne, signée avec une autre clé",
}


@dataclass(frozen=True)
class TamperResult:
    before: float | None
    after: float
    verification: Verification


def tamper_value(
    path: Path, index: int, new_value: float, mode: str, public_key_path: Path | None
) -> TamperResult:
    """Falsifie une COPIE en mémoire : la valeur du portefeuille du bloc `index`.

    Le fichier sur le disque n'est jamais modifié. `mode` : voir `TAMPER_MODES`.
    """
    chain = read_jsonl(path)
    if not 0 <= index < len(chain):
        raise ValueError(f"bloc {index} hors de la chaîne (0 à {len(chain) - 1})")
    blocks = list(chain.blocks)
    events = copy.deepcopy(blocks[index].events)
    target = next((e for e in events if e.get("type") == "PORTFOLIO"), None)
    if target is None:  # bloc sans portefeuille (genèse) : on ajoute un événement
        before = None
        events.append({"type": "PORTFOLIO", "value": new_value, "note": "falsification"})
    else:
        before = float(target["value"])
        target["value"] = new_value
    blocks[index] = replace(blocks[index], events=events)
    if mode == "rehash":
        blocks[index] = blocks[index].sealed()
    elif mode == "rewrite":
        blocks = rewrite_from(blocks, index, Signer.generate())
    elif mode != "edit":
        raise ValueError(f"mode inconnu : {mode}")
    key = _key(public_key_path)
    return TamperResult(before, new_value, Chain(blocks).verify(key))


def tamper_scenarios(path: Path, public_key_path: Path | None) -> pd.DataFrame:
    """Les cinq falsifications de la phase 7 sur une copie, et ce que la vérification dit."""
    key = _key(public_key_path)
    rows = [
        {
            "falsification": what,
            "détectée": not v.valid,
            "bloc fautif": v.bad_index,
            "motif": v.reason or v.describe(),
        }
        for what, v in tamper_demo(read_jsonl(path), key)
    ]
    return pd.DataFrame(rows)


def ledger_head(path: Path, root: Path) -> LedgerHead:
    """Nombre de blocs et hash de tête, lus à la fin du fichier."""
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    file = path.relative_to(root).as_posix()
    return LedgerHead(file, len(lines), str(json.loads(lines[-1])["hash"]))


def anchor_checks(root: Path) -> list[dict[str, Any]]:
    """Chaque ancrage Sepolia comparé aux registres présents sur le disque."""
    path = root / "ledger" / "anchors.json"
    if not path.is_file():
        return []
    moves = read_moves(root / "ledger" / "moves.json")
    out = []
    for anchor in json.loads(path.read_text(encoding="utf-8")):
        files = []
        current: list[LedgerHead] = []
        for entry in anchor["ledgers"]:
            # Un registre archivé est relu à sa nouvelle place, sous son nom ancré.
            ledger = root / resolve(entry["file"], moves)
            if ledger.is_file():
                head = ledger_head(ledger, root)
                head = LedgerHead(entry["file"], head.blocks, head.head)
                current.append(head)
                same = head.head == entry["head"] and head.blocks == entry["blocks"]
                state = "identique" if same else "DIFFÉRENT"
            else:
                state = "absent de ce PC"
            files.append(
                {
                    "fichier": entry["file"],
                    "blocs": entry["blocks"],
                    "hash de tête ancré": entry["head"],
                    "état": state,
                }
            )
        complete = len(current) == len(anchor["ledgers"])
        if complete:
            status = "conforme" if anchor_digest(current) == anchor["digest"] else "DIFFÉRENT"
        else:
            status = "incomplet (registres absents)"
        out.append(
            {
                "anchored_at": anchor["anchored_at"],
                "network": anchor["network"],
                "block_number": anchor["block_number"],
                "tx_hash": anchor["tx_hash"],
                "url": etherscan_url(anchor["tx_hash"]),
                "digest": anchor["digest"],
                "status": status,
                "files": pd.DataFrame(files),
            }
        )
    return out


# --- Expériences --------------------------------------------------------------------------


def mlflow_runs(db_path: Path) -> pd.DataFrame:
    """Les runs MLflow lus directement dans la base SQLite (sans installer MLflow)."""
    if not db_path.is_file():
        return pd.DataFrame()
    with sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True) as con:
        runs = pd.read_sql_query(
            "SELECT r.run_uuid, e.name AS experiment, r.name AS run, r.status, r.start_time "
            "FROM runs r JOIN experiments e ON r.experiment_id = e.experiment_id "
            "WHERE r.lifecycle_stage = 'active'",
            con,
        )
        params = pd.read_sql_query("SELECT run_uuid, key, value FROM params", con)
        metrics = pd.read_sql_query(
            "SELECT run_uuid, key, value FROM latest_metrics "
            "WHERE key NOT IN ('train_loss', 'val_loss')",
            con,
        )
    if runs.empty:
        return runs
    runs["start_time"] = pd.to_datetime(runs["start_time"], unit="ms")
    for table, prefix in ((params, "param."), (metrics, "")):
        if not table.empty:
            wide = table.pivot_table(
                index="run_uuid", columns="key", values="value", aggfunc="first"
            )
            wide.columns = [f"{prefix}{c}" for c in wide.columns]
            runs = runs.merge(wide, left_on="run_uuid", right_index=True, how="left")
    return runs.sort_values("start_time", ascending=False).reset_index(drop=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """{"a": {"b": 1}} -> {"a.b": 1}, pour afficher des réglages imbriqués en tableau."""
    out: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, f"{name}."))
        else:
            out[name] = value
    return out


def fmt_pct(x: float | None, digits: int = 1, signed: bool = True) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    value = 0.0 if abs(x) < 0.5 * 10 ** -(digits + 2) else x  # pas de « -0,0 % »
    sign = "+" if signed else ""
    return f"{value * 100:{sign}.{digits}f} %".replace(".", ",")


def fmt_money(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:,.0f} $".replace(",", " ")


def fmt_num(x: float | None, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:.{digits}f}".replace(".", ",")


# --- Libellés pour l'interface ------------------------------------------------------------

LEDGER_LABELS = {"plan": "Plan du test (scellé avant le run)", "resultats": "Résultats scellés"}
MODEL_LABELS = {
    "lstm": "LSTM",
    "gru": "GRU",
    "ensemble_lstm_gru": "Ensemble LSTM + GRU",
    "lightgbm": "LightGBM",
    "b0_zero": "Référence : rendement nul",
    "always_up": "Référence : toujours en hausse",
    "ensemble": "Ensemble LSTM + GRU",
}
ACTION_LABELS = {"BUY": "Achat", "SELL": "Vente", "HOLD": "Attente", "END": "Clôture"}
RULE_LABELS = {
    "signal": "Signal de l'agent stratégie",
    "stop_loss": "Stop-loss",
    "max_drawdown": "Coupe-circuit (perte maximale)",
    "rejet": "Ordre refusé par l'agent risque",
}
EVENT_LABELS = {
    "GENESIS": "Configuration scellée",
    "FORECAST": "Prévision",
    "SIGNAL": "Signal",
    "ORDER": "Ordre (agent risque)",
    "DECISION": "Décision",
    "TRADE": "Exécution",
    "PORTFOLIO": "Portefeuille",
    "METRICS": "Métriques",
    "COSTS_SENSITIVITY": "Sensibilité aux frais",
    "RANDOM_REFERENCES": "Comparaison au hasard",
    "FORECAST_METRICS": "Erreurs de prévision",
    "ALERTS": "Alertes",
}
PERIOD_SUFFIX = {"test": "test", "agents_val": "validation", "final_val": "répétition"}


def ledger_label(stem: str) -> str:
    return LEDGER_LABELS.get(stem, label(stem))


def ledger_file_label(file: str) -> str:
    """« outputs/ledger/agents_val/agents_gru.jsonl » -> « Agents (GRU) · validation »."""
    path = Path(file)
    name = ledger_label(path.stem)
    if path.stem.startswith("reglages_figes"):
        version = path.stem.partition("_v")[2]
        return f"Réglages figés, version {version}" if version else "Réglages figés"
    base, _, n = path.parent.name.partition("_tentative")
    period = PERIOD_SUFFIX.get(base, base) + (f", tentative {n}" if n else "")
    return f"{name} · {period}" if path.parent.name != "ledger" else name


def model_label(model_id: str) -> str:
    return MODEL_LABELS.get(model_id, model_id)


def action_label(action: Any) -> Any:
    return ACTION_LABELS.get(action, action) if isinstance(action, str) else action


def rule_label(rule: Any) -> Any:
    return RULE_LABELS.get(rule, rule) if isinstance(rule, str) else rule


def event_summary(event: dict[str, Any]) -> str:
    """Une phrase lisible pour un événement de bloc."""
    kind = event.get("type")
    if kind == "FORECAST":
        sigma = event.get("sigma_hat")
        extra = f", incertitude σ̂ {sigma:.4%}" if sigma else ""
        return f"{model_label(str(event.get('model_id')))} prévoit {event['r_hat']:+.4%}{extra}"
    if kind in ("SIGNAL", "DECISION"):
        return f"{action_label(event.get('action'))} : {event.get('reason', '')}"
    if kind == "ORDER":
        state = "accepté" if event.get("approved") else "refusé"
        return (
            f"{action_label(event.get('action'))} {state} ({rule_label(event.get('rule'))}) : "
            f"{event.get('reason', '')}"
        )
    if kind == "TRADE":
        return (
            f"{action_label(event.get('side'))} de {event['qty']:.4f} oz à "
            f"{event['price']:,.2f} $ le {event.get('exec_date')}, frais {event['fees']:.2f} $"
        ).replace(",", " ")
    if kind == "PORTFOLIO":
        return (
            f"Valeur {fmt_money(event['value'])}, liquidités {fmt_money(event['cash'])}, "
            f"{event['qty']:.4f} oz d'or"
        )
    if kind == "GENESIS":
        return "Configuration, empreinte des données et réglages, fixés avant le premier jour"
    return ", ".join(f"{k} : {v}" for k, v in event.items() if k != "type")[:200]


HYPERPARAM_LABELS = {
    "net.dropout": "Dropout",
    "net.hidden": "Neurones cachés",
    "net.layers": "Couches",
    "net.n_features": "Variables d'entrée",
    "train.batch_size": "Taille de lot",
    "train.clip": "Écrêtage du gradient",
    "train.loss": "Fonction de perte",
    "train.lr": "Taux d'apprentissage",
    "train.max_epochs": "Époques max",
    "train.patience": "Patience (arrêt anticipé)",
    "train.weight_decay": "Régularisation L2",
    "colsample_bytree": "Part des variables par arbre",
    "learning_rate": "Taux d'apprentissage",
    "min_child_samples": "Exemples min par feuille",
    "num_leaves": "Feuilles par arbre",
    "reg_lambda": "Régularisation L2",
    "subsample": "Part des exemples par arbre",
}


def hyperparam_label(key: str) -> str:
    return HYPERPARAM_LABELS.get(key, key)
