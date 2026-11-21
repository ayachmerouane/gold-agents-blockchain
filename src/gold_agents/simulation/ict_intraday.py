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
    out: np.ndarray = ((hours * 60 + minutes + bar_minutes) // 60) % 24
    return out


def bar_arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Colonnes utiles en tableaux numpy (le jour de cotation en numéro de jour)."""
    out = {c: frame[c].to_numpy(dtype="float64") for c in ("high", "low", "close")}
    out["ny_hour"] = frame["ny_hour"].to_numpy(dtype="int64")
    out["ny_minute"] = frame["ny_minute"].to_numpy(dtype="int64")
    days = pd.to_datetime(pd.Series(frame["trading_day"])).to_numpy()
    out["trading_day"] = days.astype("datetime64[D]").astype("int64")
    return out


def intraday_order(history: pd.DataFrame, params: IntradayParams) -> Order | None:
    """Ordre pour la barre suivante, ou `None`. N'utilise que `history` (barres closes)."""
    tail = history.iloc[-params.window :]
    return order_at(bar_arrays(tail), len(tail) - 1, params)


def order_at(bars: dict[str, np.ndarray], t: int, params: IntradayParams) -> Order | None:
    """Ordre écrit à la clôture de la barre `t`, avec les seules barres `t - window + 1 … t`."""
    first = max(0, t - params.window + 1)
    a = {c: v[first : t + 1] for c, v in bars.items()}
    last_h, last_m = a["ny_hour"][-1:], a["ny_minute"][-1:]
    next_hour = int(next_bar_hours(last_h, last_m, params.bar_minutes)[0])
    if not in_kill_zone(next_hour, params.kill_zones):
        return None
    days = a["trading_day"]
    n_bars = len(days)
    day = days[-1]
    # Barres utiles : la veille (plus haut) et le jour de cotation en cours.
    start = int(np.searchsorted(days, day, side="left"))
    hours = a["ny_hour"][start:]
    asia = (hours >= ASIA_START) & (hours < ASIA_END)
    if not asia.any():
        return None
    today_low, today_high = a["low"][start:], a["high"][start:]
    asian_low = float(today_low[asia].min())
    asian_high = float(today_high[asia].max())
    after = np.flatnonzero(hours < ASIA_START)  # de minuit à maintenant (18-20 h exclus)
    after = after[after > np.flatnonzero(asia).max()]
    if len(after) == 0:
        return None

    lookback = max(0, start - 4 * params.swing - 6)
    high = a["high"][lookback:]
    low = a["low"][lookback:]
    close = a["close"][lookback:]
    t_rel = n_bars - 1 - lookback
    after_idx = after + (start - lookback)

    sweep = int(after_idx[np.argmin(low[after_idx])])
    stop = float(low[sweep])
    if stop >= asian_low:
        return None  # pas de sweep sous le bas asiatique
    n = params.swing
    swing_highs, _ = swing_points(high, low, n)
    known = swing_highs[(swing_highs + n <= sweep) & (swing_highs < sweep)]
    if len(known) == 0 or close[sweep : t_rel + 1].max() <= high[known[-1]]:
        return None  # pas de changement de structure depuis le sweep

    if params.entry == "limit_fvg":
        level = None
        for i in range(t_rel, sweep + 1, -1):
            if i - 2 >= sweep and low[i] > high[i - 2]:
                if low[i + 1 : t_rel + 1].min(initial=np.inf) > high[i - 2]:
                    level = float(low[i])
                break
        if level is None:
            return None
        how = f"achat limite {level:.2f} (FVG)"
    else:
        level = float(high[t_rel])
        how = f"achat si > {level:.2f} (plus haut de la barre)"

    if params.target == "fixed_r":
        target = level + params.min_reward_risk * (level - stop)
    elif params.target == "asian_high":
        target = asian_high
    else:
        if start == 0:
            return None
        prev_day = days[start - 1]
        p0 = int(np.searchsorted(days, prev_day, side="left"))
        target = float(a["high"][p0:start].max())
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
    bars = bar_arrays(frame.reset_index(drop=True))
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
        decide_at=lambda t: order_at(bars, t, params),
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
