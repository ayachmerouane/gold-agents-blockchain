"""Stratégie ICT à règles et exécution par ordre conditionnel (achat seulement).

Concepts ICT traduits en règles objectives, calculées le soir du jour t avec les bougies
jusqu'à t inclus, et seulement elles :

- **sommet / creux (swing)** : plus haut (plus bas) des `swing` bougies de chaque côté.
  Il n'est CONFIRMÉ que `swing` jours plus tard : avant, la stratégie ne le connaît pas ;
- **sweep de liquidité** : une bougie passe sous le dernier creux confirmé puis clôture
  au-dessus (les stops sous le creux ont été pris) ;
- **changement de structure (MSS)** : après le sweep, une clôture au-dessus du dernier
  sommet confirmé avant le sweep ;
- **Fair Value Gap (FVG)** haussier : bas de la bougie i > haut de la bougie i-2, formé
  après le sweep et pas encore comblé ;
- **zone discount** : moitié basse (`discount`) du range qui va du creux du sweep au plus
  haut atteint depuis.

Le soir, la stratégie écrit un ordre valable UN jour, avant l'ouverture du lendemain :

- `limit_fvg` : achat à cours limité au haut du FVG, s'il est en zone discount ;
- `stop_pdh` : achat si le prix casse le plus haut de la veille (réaction haussière
  confirmée après l'ouverture), si la clôture est en zone discount.

Stop sous le creux du sweep, objectif au plus haut du range (liquidité au-dessus).
Le lendemain, le plus haut et le plus bas du jour disent si l'ordre est exécuté, puis si
le stop ou l'objectif est touché. On ignore l'ordre des prix dans la journée : quand les
deux sont possibles, on suppose le PIRE (le stop d'abord).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Literal

import numpy as np
import pandas as pd

from gold_agents.blockchain.chain import Chain
from gold_agents.simulation.costs import CostModel
from gold_agents.simulation.engine import BacktestResult, Trade, jsonable
from gold_agents.simulation.portfolio import Portfolio

Entry = Literal["limit_fvg", "stop_pdh"]


@dataclass(frozen=True)
class IctParams:
    entry: Entry = "limit_fvg"
    swing: int = 2
    sweep_lookback: int = 10
    discount: float = 0.5
    max_hold_days: int = 20
    min_reward_risk: float = 1.0

    @property
    def name(self) -> str:
        return f"ict_{self.entry}"

    @property
    def window(self) -> int:
        """Bougies utiles pour décider : le sweep et le sommet confirmé avant lui."""
        return self.sweep_lookback + 4 * self.swing + 60


@dataclass(frozen=True)
class Order:
    """Ordre d'achat conditionnel écrit le soir pour le lendemain seulement."""

    kind: Entry  # limit_fvg : achat si le prix DESCEND au niveau ; stop_pdh : s'il MONTE
    level: float
    stop: float
    target: float
    reason: str


def swing_points(high: np.ndarray, low: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices des sommets et des creux (fractales de `n` bougies de chaque côté).

    Un point d'indice j n'est connu qu'au jour j + n : c'est à l'appelant de ne garder
    que les points confirmés à la date où il décide.
    """
    if len(high) < 2 * n + 1:
        empty = np.asarray([], dtype=int)
        return empty, empty
    win_h = np.lib.stride_tricks.sliding_window_view(high, 2 * n + 1)
    win_l = np.lib.stride_tricks.sliding_window_view(low, 2 * n + 1)
    mid_h, mid_l = high[n : len(high) - n], low[n : len(low) - n]
    is_high = (mid_h == win_h.max(axis=1)) & ((win_h == mid_h[:, None]).sum(axis=1) == 1)
    is_low = (mid_l == win_l.min(axis=1)) & ((win_l == mid_l[:, None]).sum(axis=1) == 1)
    return np.flatnonzero(is_high) + n, np.flatnonzero(is_low) + n


def _last_confirmed(points: np.ndarray, known_on: int, n: int) -> int | None:
    """Dernier point confirmé au jour `known_on` (j + n ≤ known_on)."""
    ok = points[points + n <= known_on]
    return int(ok[-1]) if len(ok) else None


def ict_order(history: pd.DataFrame, params: IctParams) -> Order | None:
    """Ordre conditionnel pour le lendemain, ou `None`. N'utilise que `history`."""
    tail = history.iloc[-params.window :]
    high = tail["high"].to_numpy(dtype="float64")
    low = tail["low"].to_numpy(dtype="float64")
    close = tail["close"].to_numpy(dtype="float64")
    t, n = len(close) - 1, params.swing
    if t < 4 * n + 3:
        return None
    swing_highs, swing_lows = swing_points(high, low, n)

    # 1. Sweep le plus récent : sous le dernier creux confirmé la veille, clôture au-dessus.
    sweep = None
    for k in range(t, max(t - params.sweep_lookback, 0), -1):
        j = _last_confirmed(swing_lows, k - 1, n)
        if j is not None and low[k] < low[j] < close[k]:
            sweep = k
            break
    if sweep is None:
        return None
    stop = float(low[sweep])
    if low[sweep + 1 : t + 1].min(initial=np.inf) <= stop:
        return None  # creux du sweep déjà cassé : le scénario a échoué

    # 2. Changement de structure : clôture au-dessus du dernier sommet confirmé au sweep.
    h = _last_confirmed(swing_highs, sweep, n)
    if h is None or close[sweep : t + 1].max() <= high[h]:
        return None

    range_high = float(high[sweep : t + 1].max())
    discount_level = stop + params.discount * (range_high - stop)

    if params.entry == "limit_fvg":
        # 3. FVG haussier né après le sweep, dont le bas n'a pas été touché depuis.
        level = None
        for i in range(t, sweep + 1, -1):
            gap_bottom, gap_top = high[i - 2], low[i]
            if i - 2 >= sweep and gap_top > gap_bottom:
                if low[i + 1 : t + 1].min(initial=np.inf) > gap_bottom:
                    level = float(gap_top)
                break
        if level is None or level > discount_level:
            return None
        reason = f"sweep {stop:.2f}, MSS, FVG en discount : achat limite {level:.2f}"
    else:
        # 3'. Prix en zone discount : achat si le lendemain casse le plus haut du jour.
        if close[t] > discount_level:
            return None
        level = float(high[t])
        reason = f"sweep {stop:.2f}, MSS, discount : achat si > plus haut veille {level:.2f}"

    risk, reward = level - stop, range_high - level
    if risk <= 0 or reward < params.min_reward_risk * risk:
        return None
    return Order(params.entry, level, stop, range_high, f"{reason}, objectif {range_high:.2f}")


def entry_fill(order: Order, bar: Mapping[str, float]) -> float | None:
    """Prix (avant slippage) auquel l'ordre est exécuté pendant `bar`, ou `None`.

    Si l'ouverture dépasse déjà le niveau, on est servi à l'ouverture (écart de cotation).
    """
    o, h, lo = float(bar["open"]), float(bar["high"]), float(bar["low"])
    if order.kind == "limit_fvg":
        if o <= order.stop:
            return None  # ouvre sous le stop : scénario invalidé, pas d'achat
        if o <= order.level:
            return o
        return order.level if lo <= order.level else None
    if o >= order.target:
        return None  # objectif déjà atteint à l'ouverture : plus rien à gagner
    if o >= order.level:
        return o
    return order.level if h >= order.level else None


def exit_fill(
    stop: float, target: float, bar: Mapping[str, float], entry_day: bool
) -> tuple[float, str] | None:
    """Sortie pendant `bar` : (prix avant slippage, règle), ou `None`.

    Stop et objectif touchés le même jour : le stop (pire cas). Le jour de l'entrée, on
    ignore l'objectif (on ne sait pas s'il a été atteint après l'achat).
    """
    o, h, lo = float(bar["open"]), float(bar["high"]), float(bar["low"])
    if not entry_day:
        if o <= stop:
            return o, "stop (ouverture sous le stop)"
        if o >= target:
            return o, "objectif (ouverture au-dessus)"
    if lo <= stop:
        return stop, "stop"
    if not entry_day and h >= target:
        return target, "objectif"
    return None


def run_ict_backtest(
    frame: pd.DataFrame,
    params: IctParams,
    costs: CostModel,
    initial_capital: float,
    start: date,
    end: date | None = None,
    ledger: Chain | None = None,
) -> BacktestResult:
    """Simule la stratégie ICT journalière de `start` à `end` (date, open, high, low, close)."""
    return run_order_backtest(
        frame,
        params.name,
        lambda history: ict_order(history, params),
        costs,
        initial_capital,
        start,
        end,
        max_hold=params.max_hold_days,
        ledger=ledger,
    )


def run_order_backtest(
    frame: pd.DataFrame,
    name: str,
    decide: Callable[[pd.DataFrame], Order | None],
    costs: CostModel,
    initial_capital: float,
    start: date,
    end: date | None = None,
    *,
    max_hold: int,
    exit_at_close: Callable[[pd.Timestamp], bool] | None = None,
    may_decide: np.ndarray | None = None,
    decide_at: Callable[[int], Order | None] | None = None,
    intraday: bool = False,
    ledger: Chain | None = None,
) -> BacktestResult:
    """Moteur des ordres conditionnels, en barres journalières ou intraday.

    Barre t : (1) sortie à l'ouverture si `max_hold` barres sont passées ; (2) exécution
    éventuelle de l'ordre écrit à la barre précédente, puis stop / objectif avec le plus
    haut et le plus bas ; (3) sortie à la clôture si `exit_at_close(t)` (fin de séance) ;
    (4) valorisation à la clôture ; (5) si on n'a pas d'or, `decide` reçoit les barres
    jusqu'à t incluse, et seulement elles, et écrit l'ordre de la barre suivante.
    L'ordre de la dernière barre n'est pas exécuté. `may_decide[t]` (facultatif, calculé
    d'après la seule barre t, par exemple « la barre suivante est en kill zone ») évite
    d'appeler `decide` quand aucun ordre n'est possible. `decide_at(t)` (facultatif) fait
    la même chose que `decide(frame.iloc[: t + 1])`, sans copier le tableau : il ne doit
    lire que les barres 0 à t.
    """
    frame = frame.reset_index(drop=True)
    cols = {c: frame[c].to_numpy(dtype="float64") for c in ("open", "high", "low", "close")}
    stamps = frame["date"].map(pd.Timestamp)
    days = stamps.map(lambda ts: ts.date())
    labels = stamps if intraday else days
    in_period = (days >= start) & ((days <= end) if end is not None else True)
    positions = frame.index[in_period]
    if len(positions) < 2:
        raise ValueError("période de backtest trop courte (au moins 2 barres)")

    book = Portfolio(cash=initial_capital)
    trades: list[Trade] = []
    rows: list[dict[str, Any]] = []
    pending: tuple[Order, Any] | None = None
    held: tuple[Order, Any, int] | None = None  # ordre d'entrée, décidé à, barres tenues
    last = positions[-1]

    def trade(side: str, when: Any, raw: float, decided: Any, reason: str) -> Trade:
        if side == "BUY":
            price = costs.buy_price(raw)
            qty = book.cash / (price * (1 + costs.fee_rate))
            fee = costs.fee(qty * price)
            book.buy(qty, price, fee)
        else:
            price, qty = costs.sell_price(raw), book.qty
            fee = costs.fee(qty * price)
            book.sell(qty, price, fee)
        t = Trade(decided, when, side, raw, price, qty, fee, abs(price - raw) * qty, reason)
        trades.append(t)
        return t

    label_list = list(labels)
    stamp_list = list(stamps)
    for i in positions:
        row = {c: float(v[i]) for c, v in cols.items()}
        now = label_list[i]
        done: list[Trade] = []
        if held is not None:
            order, decided, n_bars = held
            if n_bars >= max_hold:
                done.append(trade("SELL", now, float(row["open"]), decided, "durée maximale"))
                held = None
            else:
                out = exit_fill(order.stop, order.target, row, entry_day=False)
                if out is not None:
                    done.append(trade("SELL", now, out[0], decided, out[1]))
                    held = None
                else:
                    held = (order, decided, n_bars + 1)
        elif pending is not None:
            order, decided = pending
            fill = entry_fill(order, row)
            if fill is not None:
                done.append(trade("BUY", now, fill, decided, order.reason))
                held = (order, decided, 1)
                out = exit_fill(order.stop, order.target, row, entry_day=True)
                if out is not None:
                    done.append(trade("SELL", now, out[0], decided, out[1] + " (barre d'achat)"))
                    held = None
        pending = None
        if held is not None and exit_at_close is not None and exit_at_close(stamp_list[i]):
            done.append(trade("SELL", now, float(row["close"]), held[1], "fin de séance"))
            held = None

        close = float(row["close"])
        value = book.value(close)
        action = "END"
        new_order = None
        if i != last:
            action = "HOLD"
            if book.qty == 0 and (may_decide is None or may_decide[i]):
                new_order = decide(frame.iloc[: i + 1]) if decide_at is None else decide_at(i)
                if new_order is not None:
                    pending, action = (new_order, now), "ORDER"
        rows.append(
            {
                "date": now,
                "close": close,
                "cash": book.cash,
                "qty": book.qty,
                "value": value,
                "action": action,
            }
        )
        if ledger is not None:
            events: list[dict[str, Any]] = [{"type": "TRADE", **jsonable(asdict(t))} for t in done]
            if new_order is not None:
                events.append({"type": "ORDER", "strategy": name, **asdict(new_order)})
            events.append({"type": "PORTFOLIO", "cash": book.cash, "qty": book.qty, "value": value})
            ledger.append(now.isoformat(), events)

    return BacktestResult(name, initial_capital, pd.DataFrame(rows), trades)
