"""Features et cible à partir de la série nettoyée (§4.2, étapes `add_returns`, `add_indicators`).

Règle d'or : la ligne du jour t ne dépend que des jours ≤ t. Pas de `center=True`, pas de
`shift(-k)` ; la seule exception est la cible, `target_return` = rendement de t+1, qui est
justement ce que le modèle doit prédire et n'entre jamais dans les features.
Les tests `tests/leakage/` vérifient cette règle par troncature et par remplacement du futur.

Choix issus de l'EDA (docs/constats_eda.md) :

- le prix n'est pas stationnaire : les moyennes mobiles, l'ATR et le MACD sont exprimés
  relativement au prix (écart en %), jamais en dollars ;
- le volume MT4 est un volume de ticks propre au courtier : il n'est pas utilisé ;
- un rendement qui traverse un trou non comblé (`valid_return = False`) n'est pas un
  rendement d'un jour : il est retiré des features de rendement et de la cible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gold_agents.config import FeaturesConfig

TRADING_DAYS = 252
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
BOLLINGER_K = 2.0

TARGET_COLUMNS = ["target_return", "target_up", "target_valid"]


# --- Indicateurs élémentaires (séries -> séries, passé seulement) ----------


def sma(x: pd.Series, n: int) -> pd.Series:
    """Moyenne des `n` dernières valeurs, jour t inclus."""
    return x.rolling(n, min_periods=n).mean()


def ema(x: pd.Series, n: int) -> pd.Series:
    """Moyenne exponentielle (alpha = 2 / (n + 1)), définie à partir de `n` valeurs."""
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def wilder(x: pd.Series, n: int) -> pd.Series:
    """Lissage de Wilder : moyenne simple des `n` premières valeurs, puis
    `m_t = (m_{t-1} * (n - 1) + x_t) / n`. Les NaN de tête sont ignorés."""
    values = x.to_numpy(dtype="float64")
    out = np.full(len(values), np.nan)
    valid = np.flatnonzero(~np.isnan(values))
    if len(valid) < n:
        return pd.Series(out, index=x.index)
    start = valid[0]
    seed = start + n - 1
    out[seed] = values[start : seed + 1].mean()
    for t in range(seed + 1, len(values)):
        out[t] = (out[t - 1] * (n - 1) + values[t]) / n
    return pd.Series(out, index=x.index)


def rsi(close: pd.Series, n: int) -> pd.Series:
    """RSI de Wilder sur `n` jours, entre 0 et 100."""
    delta = close.diff()
    gain = wilder(delta.clip(lower=0), n)
    loss = wilder((-delta).clip(lower=0), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100 - 100 / (1 + gain / loss)
    # Aucune baisse sur la période : RSI = 100 ; prix plat (ni hausse ni baisse) : 50.
    out = out.where(loss != 0, 100.0).where((gain != 0) | (loss != 0), 50.0)
    return out.where(gain.notna())


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Plus grand de : amplitude du jour, écarts du haut et du bas à la clôture de la veille."""
    prev = close.shift(1)
    ranges = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1)
    return ranges.max(axis=1)  # premier jour : high - low (pas de veille)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    """Average True Range de Wilder sur `n` jours, en dollars."""
    return wilder(true_range(high, low, close), n)


# --- Rendements et cible -----------------------------------------------------


def day_returns(df: pd.DataFrame) -> pd.Series:
    """Rendement log d'un jour, NaN quand il traverse un trou (`valid_return = False`)."""
    r: pd.Series = np.log(df["close"]).diff()
    return r.where(df["valid_return"]) if "valid_return" in df else r


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Cible du jour t : rendement log de t à t+1 (`target_return`) et son signe (`target_up`).

    `target_valid` est faux pour le dernier jour (pas de lendemain) et quand le lendemain
    suit un trou non comblé. C'est le seul endroit du module qui regarde le futur.
    """
    out = df.copy()
    r = day_returns(df)
    out["target_return"] = r.shift(-1)
    out["target_valid"] = out["target_return"].notna()
    out["target_up"] = (out["target_return"] > 0).astype("int8")
    return out


# --- Features -----------------------------------------------------------------


def feature_columns(cfg: FeaturesConfig) -> list[str]:
    """Noms des features, dans l'ordre des colonnes du tenseur `(N, W, F)`."""
    return [
        *(f"ret_{k}d" for k in cfg.return_lags),
        *(f"dist_sma_{p}" for p in cfg.sma_periods),
        *(f"dist_ema_{p}" for p in cfg.ema_periods),
        f"rsi_{cfg.rsi_period}",
        "macd",
        "macd_signal",
        "macd_hist",
        f"atr_{cfg.atr_period}_pct",
        "range_pct",
        "bollinger_width",
        f"volatility_{cfg.volatility_window}d",
        "dow_sin",
        "dow_cos",
    ]


def add_features(df: pd.DataFrame, cfg: FeaturesConfig) -> pd.DataFrame:
    """Ajoute toutes les features (colonnes de `feature_columns`) et `features_valid`.

    `df` est la série nettoyée (`clean.clean`), triée par date. Chaque feature du jour t
    n'utilise que les lignes ≤ t ; les premières lignes valent NaN le temps que les
    fenêtres se remplissent (50 jours pour la SMA 50, par exemple).
    """
    out = df.copy()
    close, high, low = out["close"], out["high"], out["low"]
    r = day_returns(out)

    # Rendements passés : somme des rendements log des k derniers jours.
    for k in cfg.return_lags:
        out[f"ret_{k}d"] = r.rolling(k, min_periods=k).sum()

    # Tendance : écart relatif du prix à sa moyenne mobile.
    for p in cfg.sma_periods:
        out[f"dist_sma_{p}"] = close / sma(close, p) - 1
    for p in cfg.ema_periods:
        out[f"dist_ema_{p}"] = close / ema(close, p) - 1

    # Élan : RSI et MACD (rapporté au prix).
    out[f"rsi_{cfg.rsi_period}"] = rsi(close, cfg.rsi_period)
    macd = (ema(close, MACD_FAST) - ema(close, MACD_SLOW)) / close
    signal = ema(macd, MACD_SIGNAL)
    out["macd"], out["macd_signal"], out["macd_hist"] = macd, signal, macd - signal

    # Risque : ATR, amplitude du jour, largeur de Bollinger, volatilité réalisée.
    out[f"atr_{cfg.atr_period}_pct"] = atr(high, low, close, cfg.atr_period) / close
    out["range_pct"] = (high - low) / close
    w = cfg.volatility_window
    mid = sma(close, w)
    out["bollinger_width"] = 2 * BOLLINGER_K * close.rolling(w, min_periods=w).std() / mid
    out[f"volatility_{w}d"] = r.rolling(w, min_periods=w).std() * np.sqrt(TRADING_DAYS)

    # Calendrier : jour de la semaine sur un cercle (lundi et vendredi restent voisins).
    dow = out["date"].dt.day_of_week
    out["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 5)

    cols = feature_columns(cfg)
    out["features_valid"] = out[cols].notna().all(axis=1)
    return out


def build_features(clean: pd.DataFrame, cfg: FeaturesConfig) -> pd.DataFrame:
    """Série nettoyée -> features + cible, une ligne par jour."""
    return add_target(add_features(clean, cfg))


# --- Mise à l'échelle (ajustée sur le train seulement) --------------------------


@dataclass(frozen=True)
class RobustScaler:
    """Centre par la médiane et réduit par l'écart interquartile, colonne par colonne.

    Plus robuste que la moyenne et l'écart-type aux jours extrêmes (kurtosis de 6 sur le
    train, docs/constats_eda.md §3). Ajusté sur les lignes de train uniquement.
    """

    columns: tuple[str, ...]
    center: tuple[float, ...]
    scale: tuple[float, ...]

    @classmethod
    def fit(cls, df: pd.DataFrame, columns: list[str]) -> RobustScaler:
        values = df[columns].to_numpy(dtype="float64")
        if np.isnan(values).any():
            raise ValueError("le scaler s'ajuste sur des lignes sans NaN")
        q1, med, q3 = np.percentile(values, [25, 50, 75], axis=0)
        iqr = q3 - q1
        iqr[iqr == 0] = 1.0  # colonne constante : on se contente de centrer
        return cls(tuple(columns), tuple(map(float, med)), tuple(map(float, iqr)))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        cols = list(self.columns)
        out[cols] = (df[cols].to_numpy(dtype="float64") - self.center) / self.scale
        return out

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {
            c: {"center": m, "scale": s}
            for c, m, s in zip(self.columns, self.center, self.scale, strict=True)
        }

    @classmethod
    def from_dict(cls, data: dict[str, dict[str, float]]) -> RobustScaler:
        cols = tuple(data)
        return cls(
            cols, tuple(data[c]["center"] for c in cols), tuple(data[c]["scale"] for c in cols)
        )
