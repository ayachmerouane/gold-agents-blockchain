"""Agent ICT intraday : il suit le marché barre par barre et décide pendant la séance.

Modèle ICT « sweep du range asiatique » (achat seulement), sur des barres d'une heure,
heures de New York :

1. **Range asiatique** : plus haut et plus bas des barres de 20 h à minuit du jour de
   cotation (qui commence à 18 h).
2. **Sweep** : après minuit, le prix passe sous le bas asiatique (les stops sont pris).
   Le creux de ce mouvement est le stop.
3. **Changement de structure (MSS)** : une clôture au-dessus du dernier sommet confirmé
   avant le sweep (sommet connu seulement `swing` barres après sa formation).
4. **Entrée pendant une kill zone** (Londres 2 h-5 h, New York 7 h-10 h par défaut),
   par ordre valable une barre, écrit à la clôture de la barre précédente :
   `limit_fvg` (retour dans le dernier FVG haussier) ou `stop_pdh` (cassure du plus haut
   de la barre précédente : la réaction est confirmée).
5. **Objectif** : le haut asiatique (`asian_high`) ou le plus haut de la veille
   (`prev_day_high`), s'il paie au moins `min_reward_risk` fois le risque ; ou bien
   exactement `min_reward_risk` fois le risque (`fixed_r`).
6. **Sortie** au stop, à l'objectif (pire cas si les deux dans la même barre), ou à la
   clôture de la séance de New York (`flat_hour`) : jamais de position la nuit.

L'agent ne voit que les barres closes : la décision de la barre t est exécutée à
partir de la barre t+1. Les séances sont lues dans les colonnes `ny_hour` et
`trading_day` (voir `data.intraday`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd

from gold_agents.blockchain.chain import Chain
from gold_agents.simulation.costs import CostModel
from gold_agents.simulation.engine import BacktestResult
from gold_agents.simulation.ict import Entry, Order, run_order_backtest, swing_points

Target = Literal["asian_high", "prev_day_high", "fixed_r"]

ASIA_START, ASIA_END = 20, 24  # heures de New York, la veille au soir


@dataclass(frozen=True)
class IntradayParams:
    entry: Entry = "limit_fvg"
    swing: int = 1
    target: Target = "asian_high"
    min_reward_risk: float = 1.0
    kill_zones: tuple[tuple[int, int], ...] = field(default=((2, 5), (7, 10)))
    flat_hour: int = 16  # tout est vendu à la clôture de la barre qui finit à 16 h
    max_hold_hours: int = 24
    bar_minutes: int = 60  # durée d'une barre : 60, 15, 5…

    @property
    def name(self) -> str:
        return f"ict_{self.bar_minutes}m_{self.entry}_{self.target}"

    @property
    def max_hold_bars(self) -> int:
        return self.max_hold_hours * 60 // self.bar_minutes

    @property
    def window(self) -> int:
        """Barres relues à chaque décision : la veille, le jour en cours et une marge."""
        return 3 * 24 * 60 // self.bar_minutes + 10 * self.swing


def in_kill_zone(hour: int, zones: tuple[tuple[int, int], ...]) -> bool:
    return any(a <= hour < b for a, b in zones)


def next_bar_hours(hours: np.ndarray, minutes: np.ndarray, bar_minutes: int) -> np.ndarray:
    """Heure (New York) du début de la barre suivante, d'après la barre courante seule."""
    return ((hours * 60 + minutes + bar_minutes) // 60) % 24


def intraday_order(history: pd.DataFrame, params: IntradayParams) -> Order | None:
    """Ordre pour la barre suivante, ou `None`. N'utilise que `history` (barres closes)."""
    history = history.iloc[-params.window :]
    last_h = history["ny_hour"].to_numpy()[-1:]
    last_m = history["ny_minute"].to_numpy()[-1:]
    next_hour = int(next_bar_hours(last_h, last_m, params.bar_minutes)[0])
    if not in_kill_zone(next_hour, params.kill_zones):
        return None
    day = history["trading_day"].iat[-1]
    days = history["trading_day"].to_numpy()
    # Barres utiles : la veille (plus haut) et le jour de cotation en cours.
    start = len(history) - 1
    while start > 0 and days[start - 1] == day:
        start -= 1
    today = history.iloc[start:]
    hours = today["ny_hour"].to_numpy()
    asia = (hours >= ASIA_START) & (hours < ASIA_END)
    if not asia.any():
        return None
    asian_low = float(today["low"].to_numpy()[asia].min())
    asian_high = float(today["high"].to_numpy()[asia].max())
    after = np.flatnonzero(hours < ASIA_START)  # de minuit à maintenant (18-20 h exclus)
    after = after[after > np.flatnonzero(asia).max()]
    if len(after) == 0:
        return None

    lookback = max(0, start - 4 * params.swing - 6)
    bars = history.iloc[lookback:]
    high = bars["high"].to_numpy(dtype="float64")
    low = bars["low"].to_numpy(dtype="float64")
    close = bars["close"].to_numpy(dtype="float64")
    t = len(close) - 1
    offset = start - lookback
    after_idx = after + offset

    sweep = int(after_idx[np.argmin(low[after_idx])])
    stop = float(low[sweep])
    if stop >= asian_low:
        return None  # pas de sweep sous le bas asiatique
    n = params.swing
    swing_highs, _ = swing_points(high, low, n)
    known = swing_highs[(swing_highs + n <= sweep) & (swing_highs < sweep)]
    if len(known) == 0 or close[sweep : t + 1].max() <= high[known[-1]]:
        return None  # pas de changement de structure depuis le sweep

    if params.entry == "limit_fvg":
        level = None
        for i in range(t, sweep + 1, -1):
            if i - 2 >= sweep and low[i] > high[i - 2]:
                if low[i + 1 : t + 1].min(initial=np.inf) > high[i - 2]:
                    level = float(low[i])
                break
        if level is None:
            return None
        how = f"achat limite {level:.2f} (FVG)"
    else:
        level = float(high[t])
        how = f"achat si > {level:.2f} (plus haut de la barre)"

    if params.target == "fixed_r":
        target = level + params.min_reward_risk * (level - stop)
    elif params.target == "asian_high":
        target = asian_high
    else:
        prev = history.iloc[:start]
        prev = prev[prev["trading_day"] == prev["trading_day"].iat[-1]] if len(prev) else prev
        if prev.empty:
            return None
        target = float(prev["high"].max())
    risk, reward = level - stop, target - level
    if risk <= 0 or reward < params.min_reward_risk * risk:
        return None
    return Order(
        params.entry,
        level,
        stop,
        target,
        f"sweep du bas asiatique {asian_low:.2f} -> {stop:.2f}, MSS, {how}, objectif {target:.2f}",
    )


def run_intraday_backtest(
    frame: pd.DataFrame,
    params: IntradayParams,
    costs: CostModel,
    initial_capital: float,
    start: date,
    end: date | None = None,
    ledger: Chain | None = None,
) -> BacktestResult:
    """Simule l'agent intraday ; `frame` vient de `data.intraday` (UTC + colonnes de séance)."""
    flat = params.flat_hour * 60
    ny = pd.to_datetime(frame["date"], utc=True).dt.tz_convert("America/New_York")
    nxt = next_bar_hours(ny.dt.hour.to_numpy(), ny.dt.minute.to_numpy(), params.bar_minutes)
    may_decide = np.array([in_kill_zone(int(h), params.kill_zones) for h in range(24)])[nxt]

    def exit_at_close(ts: pd.Timestamp) -> bool:
        ny_ts = ts.tz_convert("America/New_York")
        begin = ny_ts.hour * 60 + ny_ts.minute
        return begin < flat <= begin + params.bar_minutes

    return run_order_backtest(
        frame,
        params.name,
        lambda history: intraday_order(history, params),
        costs,
        initial_capital,
        start,
        end,
        max_hold=params.max_hold_bars,
        exit_at_close=exit_at_close,
        may_decide=may_decide,
        intraday=True,
        ledger=ledger,
    )


def daily_equity(result: BacktestResult) -> BacktestResult:
    """Courbe de valeur ramenée à une valeur par jour (dernière barre), pour les métriques.

    Le Sharpe annualisé suppose des rendements journaliers : on ne le calcule pas sur des
    rendements horaires.
    """
    eq = result.equity.copy()
    eq["day"] = pd.to_datetime(eq["date"], utc=True).dt.tz_convert("America/New_York").dt.date
    daily = eq.drop(columns="date").groupby("day", sort=True).last()
    daily = daily.rename_axis("date").reset_index()
    return BacktestResult(result.strategy, result.initial_capital, daily, result.trades)
