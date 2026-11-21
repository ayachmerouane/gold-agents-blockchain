"""Stratégie ICT (§7.3) : points pivots, ordre conditionnel, exécution au pire cas."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from gold_agents.simulation.costs import CostModel
from gold_agents.simulation.ict import (
    IctParams,
    Order,
    entry_fill,
    exit_fill,
    ict_order,
    run_ict_backtest,
    swing_points,
)

# (high, low, close) : creux, sommet, creux confirmé, sweep (7), MSS + FVG (9), repli.
BARS = [
    (10.0, 9.0, 9.5),
    (9.5, 8.0, 8.5),
    (11.0, 9.0, 10.5),
    (12.0, 10.0, 11.5),
    (11.5, 9.5, 10.0),
    (10.0, 8.5, 9.0),
    (10.5, 9.0, 10.0),
    (10.0, 8.2, 9.5),
    (11.0, 9.6, 10.8),
    (12.8, 11.2, 12.5),
    (12.6, 10.9, 11.2),
]


def _setup() -> pd.DataFrame:
    h, lo, c = (np.array(x) for x in zip(*BARS, strict=True))
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=len(BARS), tz="UTC"),
            "open": c,  # sans importance pour la décision
            "high": h,
            "low": lo,
            "close": c,
        }
    )


def _bar(o: float, h: float, lo: float, c: float) -> pd.Series:
    return pd.Series({"open": o, "high": h, "low": lo, "close": c})


def test_swing_points() -> None:
    df = _setup()
    highs, lows = swing_points(df["high"].to_numpy(), df["low"].to_numpy(), 1)
    assert list(lows) == [1, 5, 7]
    assert 3 in highs and 6 in highs and 9 in highs


def test_limit_order_on_textbook_setup() -> None:
    params = IctParams(entry="limit_fvg", swing=1, discount=1.0, min_reward_risk=0.5)
    order = ict_order(_setup(), params)
    assert order is not None
    assert (order.level, order.stop, order.target) == (11.2, 8.2, 12.8)


def test_filters_reject_setup() -> None:
    df = _setup()
    # Haut du FVG (11,2) au-dessus de la moitié du range (10,5) : pas en zone discount.
    assert ict_order(df, IctParams(swing=1, discount=0.5, min_reward_risk=0.5)) is None
    # Gain possible 1,6 pour un risque de 3 : refusé si l'on exige R/R ≥ 1.
    assert ict_order(df, IctParams(swing=1, discount=1.0, min_reward_risk=1.0)) is None
    # Sans sweep récent, pas d'ordre.
    assert ict_order(df, IctParams(swing=1, discount=1.0, sweep_lookback=2)) is None


def test_swing_not_known_before_confirmation() -> None:
    """Le creux du jour 7 n'est confirmé qu'au jour 8 : coupé au jour 7, pas de sweep."""
    df = _setup()
    params = IctParams(entry="limit_fvg", swing=1, discount=1.0, min_reward_risk=0.5)
    assert ict_order(df.iloc[:8], params) is None


ORDER_L = Order("limit_fvg", level=100.0, stop=95.0, target=110.0, reason="")
ORDER_S = Order("stop_pdh", level=100.0, stop=95.0, target=110.0, reason="")


@pytest.mark.parametrize(
    ("order", "bar", "expected"),
    [
        (ORDER_L, _bar(102, 104, 99, 103), 100.0),  # le prix descend jusqu'au niveau
        (ORDER_L, _bar(98, 101, 97, 100), 98.0),  # ouvre sous le niveau : servi à l'ouverture
        (ORDER_L, _bar(102, 104, 100.5, 103), None),  # niveau jamais atteint
        (ORDER_L, _bar(94, 99, 93, 98), None),  # ouvre sous le stop : scénario invalidé
        (ORDER_S, _bar(99, 101, 98, 100), 100.0),  # casse le plus haut de la veille
        (ORDER_S, _bar(102, 103, 101, 102), 102.0),  # ouvre au-dessus : à l'ouverture
        (ORDER_S, _bar(98, 99.5, 97, 99), None),  # pas de cassure
        (ORDER_S, _bar(111, 112, 109, 111), None),  # objectif déjà dépassé
    ],
)
def test_entry_fill(order: Order, bar: pd.Series, expected: float | None) -> None:
    assert entry_fill(order, bar) == expected


def test_exit_worst_case() -> None:
    assert exit_fill(95, 110, _bar(100, 111, 94, 105), entry_day=False) == (95, "stop")
    assert exit_fill(95, 110, _bar(100, 111, 96, 105), entry_day=False) == (110, "objectif")
    assert exit_fill(95, 110, _bar(94, 96, 93, 95), entry_day=False)[0] == 94  # trou
    assert exit_fill(95, 110, _bar(100, 111, 96, 105), entry_day=True) is None
    assert exit_fill(95, 110, _bar(100, 104, 94, 101), entry_day=True) == (95, "stop")


def test_backtest_trades_and_costs() -> None:
    """Après l'ordre du jour 10 : achat à 11,2 le jour 11, objectif 12,8 le jour 12."""
    df = _setup()
    extra = pd.DataFrame(
        {
            "date": pd.bdate_range(df["date"].iloc[-1], periods=4, tz="UTC")[1:],
            "open": [11.5, 11.8, 13.0],
            "high": [11.9, 13.0, 13.5],
            "low": [11.0, 11.7, 12.5],
            "close": [11.6, 12.9, 13.2],
        }
    )
    df = pd.concat([df, extra], ignore_index=True)
    params = IctParams(entry="limit_fvg", swing=1, discount=1.0, min_reward_risk=0.5)
    costs = CostModel(0.001, 0.0002)
    res = run_ict_backtest(df, params, costs, 1000.0, date(2024, 1, 1))
    sides = [(t.side, t.market_open) for t in res.trades]
    assert sides[:2] == [("BUY", 11.2), ("SELL", 12.8)]
    assert res.trades[0].price == pytest.approx(11.2 * 1.0002)
    assert res.equity["cash"].min() >= 0 and res.equity["qty"].min() >= 0
