"""Simulation en direct (paper trading) : un jour de cotation de plus à chaque lancement.

Le principe : juger le système sur des prix qui n'existaient pas encore quand il a décidé.

1. `live-init` (une fois) : les données sont mises à jour, LSTM et GRU sont réentraînés sur
   tout l'historique connu (hyperparamètres de la phase 4, mêmes règles que le walk-forward)
   puis figés. Le bloc genèse du registre en direct scelle la règle du dernier
   `freeze` (la même que pour le test, vérifiée par `live-init`), l'empreinte SHA-256 de chaque poids,
   les données et le commit. Il est signé, puis ancré sur Sepolia AVANT le premier jour.
2. `live` (chaque jour de cotation) : nouvelles barres quotidiennes (Twelve Data), puis,
   pour chaque nouveau jour D : l'ordre décidé la veille est exécuté à l'ouverture de D,
   le portefeuille est valorisé à la clôture de D, et les agents décident pour le jour
   suivant avec les prix jusqu'à D et rien d'autre. Le bloc de D est signé et ancré.

Une décision est « à temps » si, au moment où elle est écrite, la barre du jour suivant
n'existe pas encore dans les données (la barre du jour en cours, pas close, n'est jamais
lue). Si un jour a été manqué, sa décision est quand même calculée (le système est
déterministe : mêmes poids, mêmes prix, même décision), mais elle est marquée « en retard ».

L'état (portefeuille, agent de risque, prévisions récentes, ordre en attente, Buy & Hold
de référence) est écrit à la fin de chaque bloc : le lancement suivant repart de là, sans
rien recalculer du passé. Le registre est la seule source de vérité des jours écoulés.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gold_agents.agents.execution import ExecutionAgent
from gold_agents.agents.messages import Forecast, MarketState, Order
from gold_agents.agents.risk import RiskAgent
from gold_agents.agents.strategy import StrategyAgent
from gold_agents.blockchain.chain import Chain
from gold_agents.config import Config, LiveConfig
from gold_agents.data.clean import clean
from gold_agents.data.features import RobustScaler, build_features, feature_columns
from gold_agents.data.load import load_combined, read_update_csv
from gold_agents.simulation.costs import CostModel
from gold_agents.simulation.engine import Trade, execute_order, jsonable
from gold_agents.simulation.portfolio import Portfolio
from gold_agents.simulation.strategies import Action, Decision

LEDGER_DIR = Path("outputs/ledger/live")
LEDGER_FILE = "agents_live.jsonl"
MODELS_DIR = Path("outputs/models/live")
SCALER_FILE = "scaler.json"
MODEL_ID = "ensemble_lstm_gru"
SYSTEM = "agents_live"
KINDS = ("lstm", "gru")


def _day(ts: Any) -> date:
    return pd.Timestamp(ts).date()


# --- Données ---------------------------------------------------------------------------


def live_frame(cfg: Config) -> pd.DataFrame:
    """Kaggle figé + trous comblés + jours ajoutés par l'API : features et cible.

    Même nettoyage et mêmes features que le reste du projet. Chaque feature du jour t ne
    lit que les jours ≤ t ; la cible (le lendemain) n'est lue que pour l'entraînement.
    """
    raw = load_combined(cfg.data.raw_path, cfg.data.update_path)
    gap_path = cfg.data.gapfill_path
    gapfill = read_update_csv(gap_path) if gap_path.exists() else None
    cleaned = clean(raw, cfg.splits, gapfill)
    return build_features(cleaned, cfg.features).reset_index(drop=True)


def data_digests(cfg: Config) -> dict[str, str | None]:
    from gold_agents.data.update import sha256_file

    paths = {
        "raw": cfg.data.raw_path,
        "gapfill": cfg.data.gapfill_path,
        "update": cfg.data.update_path,
    }
    return {k: sha256_file(p) if p.exists() else None for k, p in paths.items()}


# --- Modèles figés ---------------------------------------------------------------------


@dataclass(frozen=True)
class LiveModels:
    networks: list[Any]  # TrainedNetwork (LSTM, GRU), toutes les graines
    scaler: RobustScaler
    window: int

    @property
    def columns(self) -> list[str]:
        return list(self.scaler.columns)

    def windows(self, frame: pd.DataFrame, positions: list[int]) -> np.ndarray | None:
        """Fenêtres des W jours qui finissent à chaque position ; None si incomplètes."""
        valid = frame["features_valid"].astype(bool).to_numpy()
        w = self.window
        if any(i < w - 1 or not valid[i - w + 1 : i + 1].all() for i in positions):
            return None
        values = self.scaler.transform(frame)[self.columns].to_numpy(dtype="float32")
        offsets = np.arange(-w + 1, 1)
        return values[np.asarray(positions)[:, None] + offsets[None, :]]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """r̂ de l'ensemble (moyenne LSTM et GRU, chacun moyenné sur ses graines)."""
        out: np.ndarray = np.mean([net.predict(x) for net in self.networks], axis=0)
        return out.astype("float64")

    def forecast(
        self, frame: pd.DataFrame, i: int, mc_samples: int, seed: int
    ) -> tuple[float | None, float | None]:
        from gold_agents.agents.forecast import network_table

        x = self.windows(frame, [i])
        if x is None:
            return None, None
        dates = pd.DatetimeIndex([frame["date"].iloc[i]])
        table = network_table(MODEL_ID, self.networks, x, dates, mc_samples, seed)
        sigma = None if table.sigma_hat is None else float(table.sigma_hat[0])
        return float(table.r_hat[0]), sigma


def train_models(
    frame: pd.DataFrame,
    cfg: Config,
    live: LiveConfig,
    net_meta: dict[str, dict[str, Any]],
    out_dir: Path,
    max_epochs: int,
    patience: int,
    log: Any = print,
) -> dict[str, Any]:
    """LSTM et GRU appris sur tout l'historique dont la cible est connue, puis figés.

    Comme une période du walk-forward : les `early_stop_days` derniers jours servent à
    l'arrêt anticipé (après une purge de W jours), le scaler est ajusté sur le passé.
    """
    from gold_agents.data.windows import make_windows
    from gold_agents.models import walk_forward as wf
    from gold_agents.models.predict import save_network
    from gold_agents.models.recurrent import NetConfig
    from gold_agents.models.train import TrainConfig, fit

    window = cfg.features.window
    columns = feature_columns(cfg.features)
    days = frame["date"].map(_day)
    after = days.iloc[-1] + timedelta(days=1)
    rows = wf.fold_rows(frame, window, after, after, live.early_stop_days)
    scaler = RobustScaler.fit(frame[rows.fit | rows.early_stop], columns)
    scaled = scaler.transform(frame)
    fit_w = make_windows(scaled, columns, window, rows.fit)
    es_w = make_windows(scaled, columns, window, rows.early_stop)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / SCALER_FILE).write_text(json.dumps(scaler.to_dict(), indent=2), "utf-8")
    info: dict[str, Any] = {
        "n_fit": len(fit_w.y),
        "n_early_stop": len(es_w.y),
        "fit": [str(_day(fit_w.dates[0])), str(_day(fit_w.dates[-1]))],
        "early_stop": [str(_day(es_w.dates[0])), str(_day(es_w.dates[-1]))],
        "seeds": list(live.seeds),
        "max_epochs": max_epochs,
        "patience": patience,
    }
    for kind in KINDS:
        meta = net_meta[kind]
        net = NetConfig(**meta["net"])
        train_cfg = TrainConfig(**{**meta["train"], "max_epochs": max_epochs, "patience": patience})
        y_scale = float(np.std(fit_w.y))
        models, epochs = {}, []
        for seed in live.seeds:
            result = fit(
                kind, net, train_cfg, fit_w.x, fit_w.y / y_scale, es_w.x, es_w.y / y_scale, seed
            )
            models[seed] = result.model
            epochs.append(result.best_epoch)
            log(f"  {kind.upper()} graine {seed} : meilleure époque {result.best_epoch}")
        save_network(
            out_dir / kind,
            kind,
            models,
            {"net": meta["net"], "train": meta["train"], "y_scale": y_scale, "best_epochs": epochs},
        )
        info[f"{kind}_best_epochs"] = epochs
    es_positions = np.flatnonzero(rows.early_stop.to_numpy())
    keep = max(cfg.strategy.quantile_window, 1)  # prévisions passées utiles au seuil relatif
    info["warmup_positions"] = [int(i) for i in es_positions[-keep:]]
    return info


def load_models(models_dir: Path, window: int) -> LiveModels:
    from gold_agents.models.predict import load_network

    scaler = RobustScaler.from_dict(json.loads((models_dir / SCALER_FILE).read_text("utf-8")))
    return LiveModels([load_network(models_dir / k) for k in KINDS], scaler, window)


def models_digest(models_dir: Path) -> dict[str, str]:
    """SHA-256 de chaque fichier des modèles figés (poids, hyperparamètres, scaler)."""
    files = sorted(p for p in models_dir.rglob("*") if p.is_file())
    return {
        p.relative_to(models_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }


def check_models(models_dir: Path, expected: dict[str, str]) -> list[str]:
    """Différences entre les modèles sur le disque et ceux scellés dans le genèse."""
    if not models_dir.is_dir():
        return [f"{models_dir} absent"]
    current = models_digest(models_dir)
    out = [f"{name} modifié" for name, h in expected.items() if current.get(name) != h]
    out += [f"{name} ajouté après le genèse" for name in current if name not in expected]
    return out


# --- Réglages et état -------------------------------------------------------------------


def live_settings(cfg: Config, live: LiveConfig) -> dict[str, Any]:
    """La règle figée par `freeze` (stratégie et risque), plus les frais et le MC Dropout."""
    st, rk, sim = cfg.strategy, cfg.risk, cfg.simulation
    return {
        "model": st.model,
        "model_id": MODEL_ID,
        "buy_rule": st.buy_rule,
        "theta_buy": st.theta_buy,
        "theta_sell": st.theta_sell,
        "min_confidence": st.min_confidence,
        "quantile_window": st.quantile_window,
        "quantile_min_days": st.quantile_min_days,
        "max_position": rk.max_position,
        "stop_loss": rk.stop_loss,
        "max_drawdown": rk.max_drawdown,
        "allow_short": rk.allow_short,
        "cooldown_days": rk.cooldown_days,
        "fee_rate": sim.fee_rate,
        "slippage": sim.slippage,
        "initial_capital": sim.initial_capital,
        "window": cfg.features.window,
        "mc_samples": cfg.models.mc_samples,
        "mc_seed": live.mc_seed,
    }


def _order_dict(order: Order) -> dict[str, Any]:
    return {
        k: (v.value if isinstance(v, Action) else v) for k, v in jsonable(asdict(order)).items()
    }


def _order(data: dict[str, Any]) -> Order:
    return Order(
        date.fromisoformat(data["date"]),
        Action(data["action"]),
        float(data["fraction"]),
        bool(data["approved"]),
        str(data["rule"]),
        str(data["reason"]),
    )


@dataclass
class LiveState:
    """Tout ce qu'il faut pour continuer le lendemain, écrit à la fin de chaque bloc."""

    cash: float
    qty: float = 0.0
    entry_price: float | None = None
    peak_value: float = 0.0
    halted: bool = False
    days_in_cash: int = 0
    pending: dict[str, Any] | None = None  # ordre à exécuter à la prochaine ouverture
    history: list[float] = field(default_factory=list)  # dernières prévisions r̂
    bh_cash: float = 0.0
    bh_qty: float = 0.0
    bh_pending: bool = False  # Buy & Hold : achat à la prochaine ouverture
    last_date: str | None = None

    def event(self) -> dict[str, Any]:
        return {"type": "STATE", **asdict(self)}

    @classmethod
    def from_event(cls, event: dict[str, Any]) -> LiveState:
        data = {k: v for k, v in event.items() if k != "type"}
        return cls(**data)


def last_state(chain: Chain) -> LiveState:
    for event in reversed(chain.head.events):
        if event.get("type") == "STATE":
            return LiveState.from_event(event)
    raise ValueError("le dernier bloc du registre en direct n'a pas d'état (STATE)")


def genesis(chain: Chain) -> dict[str, Any]:
    event = chain.blocks[0].events[0]
    if event.get("kind") != "LIVE_PLAN":
        raise ValueError("le premier bloc n'est pas le plan de la simulation en direct")
    return dict(event)


# --- Un jour de cotation ------------------------------------------------------------------


@dataclass(frozen=True)
class DayReport:
    day: date
    on_time: bool
    trade: Trade | None
    r_hat: float | None
    threshold: float | None  # seuil d'achat du jour (None : historique trop court)
    order: Order
    value: float
    bh_value: float


def step(
    chain: Chain,
    frame: pd.DataFrame,
    i: int,
    models: LiveModels,
    settings: dict[str, Any],
    state: LiveState,
    now: datetime,
) -> tuple[LiveState, DayReport]:
    """Ajoute au registre le bloc du jour `frame[i]` et renvoie le nouvel état."""
    row = frame.iloc[i]
    today = _day(row["date"])
    latest = _day(frame["date"].iloc[-1])
    costs = CostModel(settings["fee_rate"], settings["slippage"])
    events: list[dict[str, Any]] = [
        {
            "type": "MARKET",
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "source": str(row.get("source", "")),
        }
    ]

    # 1. Ouverture : l'ordre décidé la veille, et l'achat de Buy & Hold le premier jour.
    portfolio = Portfolio(state.cash, state.qty)
    trade = None
    if state.pending is not None:
        trade = ExecutionAgent(costs).act(_order(state.pending), row, portfolio)
        if trade is not None:
            events.append({"type": "TRADE", **jsonable(asdict(trade))})
            state.entry_price = (
                None if trade.side == "SELL" else (trade.price * trade.qty + trade.fees) / trade.qty
            )
    bench = Portfolio(state.bh_cash, state.bh_qty)
    bh_trade = None
    if state.bh_pending:
        decided = date.fromisoformat(state.last_date) if state.last_date else today
        buy = Decision(Action.BUY, "Buy & Hold : achat le premier jour")
        bh_trade = execute_order(buy, decided, row, bench, costs, settings["max_position"])
        state.bh_pending = False

    # 2. Clôture : valeur du portefeuille.
    close = float(row["close"])
    value = portfolio.value(close)

    # 3. Décision pour le jour suivant, avec les prix jusqu'à aujourd'hui seulement.
    r_hat, sigma = models.forecast(frame, i, settings["mc_samples"], settings["mc_seed"])
    forecast = Forecast(today, r_hat, sigma, MODEL_ID)
    strategy = StrategyAgent(
        settings["theta_buy"],
        settings["theta_sell"],
        settings["min_confidence"],
        settings["buy_rule"],
        settings["quantile_window"],
        settings["quantile_min_days"],
    )
    strategy._history = list(state.history)
    threshold = strategy.buy_threshold()
    signal = strategy.act(forecast, portfolio.qty > 0)
    risk = RiskAgent(
        settings["max_position"],
        settings["stop_loss"],
        settings["max_drawdown"],
        settings["allow_short"],
        settings["cooldown_days"],
    )
    risk.peak_value, risk.halted, risk.days_in_cash = (
        state.peak_value,
        state.halted,
        state.days_in_cash,
    )
    market = MarketState(today, close, portfolio.cash, portfolio.qty, value, state.entry_price)
    order = risk.act(signal, market)
    if bench.qty == 0 and bh_trade is None and state.bh_qty == 0:
        state.bh_pending = True  # Buy & Hold décide d'acheter dès le premier jour
    bh_value = bench.value(close)

    on_time = today == latest
    events += [
        {**forecast.event(), "buy_threshold": None if math.isnan(threshold) else threshold},
        signal.event(),
        order.event(),
        {"type": "PORTFOLIO", "cash": portfolio.cash, "qty": portfolio.qty, "value": value},
        {
            "type": "BENCHMARK",
            "strategy": "buy_and_hold",
            "cash": bench.cash,
            "qty": bench.qty,
            "value": bh_value,
            "trade": None if bh_trade is None else jsonable(asdict(bh_trade)),
        },
        {
            "type": "TIMING",
            "decided_at": now.astimezone(UTC).isoformat(timespec="seconds"),
            "data_last_date": latest.isoformat(),
            "on_time": on_time,
            "note": (
                "décision écrite avant que la barre du jour suivant existe"
                if on_time
                else "jour rattrapé : la barre suivante existait déjà (décision en retard)"
            ),
        },
    ]
    new = LiveState(
        cash=portfolio.cash,
        qty=portfolio.qty,
        entry_price=state.entry_price if portfolio.qty > 0 else None,
        peak_value=risk.peak_value,
        halted=risk.halted,
        days_in_cash=risk.days_in_cash,
        pending=_order_dict(order) if order.executable else None,
        history=strategy._history[-max(settings["quantile_window"], 1) :],
        bh_cash=bench.cash,
        bh_qty=bench.qty,
        bh_pending=state.bh_pending,
        last_date=today.isoformat(),
    )
    events.append(new.event())
    chain.append(today.isoformat(), events)
    shown = None if math.isnan(threshold) else threshold
    report = DayReport(today, on_time, trade, r_hat, shown, order, value, bh_value)
    return new, report


def new_days(frame: pd.DataFrame, after: str | None, first: str) -> list[int]:
    """Positions des jours à ajouter : après le dernier bloc, à partir du premier jour."""
    days = frame["date"].map(_day)
    keep = days >= date.fromisoformat(first)
    if after is not None:
        keep &= days > date.fromisoformat(after)
    return [int(i) for i in np.flatnonzero(keep.to_numpy())]


def advance(
    chain: Chain,
    frame: pd.DataFrame,
    models: LiveModels,
    now: datetime | None = None,
) -> list[DayReport]:
    """Ajoute un bloc par nouveau jour de cotation ; renvoie le résumé de chaque jour."""
    plan = genesis(chain)
    settings = plan["settings"]
    state = last_state(chain)
    now = now or datetime.now(UTC)
    reports = []
    for i in new_days(frame, state.last_date, plan["first_day"]):
        state, report = step(chain, frame, i, models, settings, state, now)
        reports.append(report)
    return reports


def revised_bars(chain: Chain, frame: pd.DataFrame, tolerance: float = 1e-6) -> list[str]:
    """Jours déjà inscrits dont la barre a changé depuis dans les données (révision API)."""
    closes = dict(
        zip(frame["date"].map(lambda d: _day(d).isoformat()), frame["close"], strict=True)
    )
    out = []
    for block in chain.blocks[1:]:
        market = next((e for e in block.events if e.get("type") == "MARKET"), None)
        now = closes.get(str(block.sim_date))
        if market and now is not None and abs(float(now) / market["close"] - 1) > tolerance:
            out.append(str(block.sim_date))
    return out


def plan_event(
    cfg: Config,
    live: LiveConfig,
    frozen: dict[str, Any],
    frame: pd.DataFrame,
    models: LiveModels,
    models_dir: Path,
    training: dict[str, Any],
    code: dict[str, Any],
    frozen_head: str,
    now: datetime,
) -> tuple[dict[str, Any], LiveState]:
    """Bloc genèse : tout ce qui est figé avant le premier jour, et l'état de départ."""
    settings = live_settings(cfg, live)
    positions = training.pop("warmup_positions")
    x = models.windows(frame, positions)
    warm = [] if x is None else [float(v) for v in models.predict(x)]
    warm_dates = [_day(frame["date"].iloc[i]).isoformat() for i in positions]
    first = _day(frame["date"].iloc[-1])
    capital = settings["initial_capital"]
    event = {
        "type": "GENESIS",
        "kind": "LIVE_PLAN",
        "strategy": SYSTEM,
        "created_at": now.astimezone(UTC).isoformat(timespec="seconds"),
        "first_day": first.isoformat(),
        "settings": settings,
        "frozen_settings": {**frozen, "head": frozen_head},
        "config": json.loads(cfg.model_dump_json()),
        "data": {"last_date": first.isoformat(), "sha256": data_digests(cfg)},
        "models": {
            "dir": models_dir.as_posix(),
            "files": models_digest(models_dir),
            "training": training,
        },
        "warmup": [{"date": d, "r_hat": r} for d, r in zip(warm_dates, warm, strict=True)],
        "code": code,
        "rules": [
            "la décision du jour D ne lit que les prix jusqu'à la clôture de D",
            "elle est exécutée à l'ouverture du jour de cotation suivant (frais et slippage)",
            "la barre du jour en cours, pas close, n'est jamais lue",
            "modèles, seuils et risque sont figés dans ce bloc ; poids vérifiés à chaque jour",
        ],
    }
    state = LiveState(cash=capital, history=warm, bh_cash=capital)
    return event, state
