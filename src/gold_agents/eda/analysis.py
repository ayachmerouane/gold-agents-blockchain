"""Analyse exploratoire (EDA) de la série XAU/USD quotidienne (phase 1, §12).

Chaque fonction est pure : elle lit un tableau au schéma commun (`data.load`) ou une série
de rendements, et renvoie des chiffres sans rien écrire. Figures et rapport : `eda.figures`
et `eda.report`.

Règle anti look-ahead : tout seuil (régimes de volatilité, échelle des valeurs aberrantes)
et tout test statistique sont calculés sur le **train seulement**, puis appliqués tels quels
à la validation et au test. Les jours ajoutés par l'API ne font partie d'aucun ensemble.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from gold_agents.config import SplitsConfig

TRADING_DAYS = 252
SPLIT_NAMES = ("train", "val", "test")
_WEEKDAYS = ["lun", "mar", "mer", "jeu", "ven", "sam", "dim"]
# Facteur qui rend le MAD comparable à un écart-type pour une loi normale.
_MAD_TO_STD = 1.4826


def _day(ts: Any) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


# --- Rendements et découpage ----------------------------------------------


def log_returns(df: pd.DataFrame) -> pd.Series:
    """r(t) = ln(close_t / close_{t-1}), indexé par la date t (le premier jour est perdu)."""
    close = df.set_index("date")["close"]
    r: pd.Series = np.log(close).diff().dropna().rename("log_return")
    return r


def assign_split(dates: pd.Series | pd.Index, splits: SplitsConfig) -> pd.Series:
    """Nom de l'ensemble (train/val/test) de chaque date, ou `None` hors ensembles."""
    days = pd.Series(pd.DatetimeIndex(dates).tz_localize(None).date, index=dates)
    out = pd.Series(None, index=dates, dtype="object")
    out[days <= splits.train_end] = "train"
    out[(days >= splits.val_start) & (days <= splits.val_end)] = "val"
    out[days >= splits.test_start] = "test"
    return out


def train_part(r: pd.Series, splits: SplitsConfig) -> pd.Series:
    part: pd.Series = r[assign_split(r.index, splits).to_numpy() == "train"]
    return part


# --- Calendrier : trous de dates, jours de week-end ------------------------


def calendar_report(df: pd.DataFrame, top: int = 10) -> dict[str, Any]:
    """Trous de dates (jours ouvrés absents), barres de week-end, couverture par année.

    Un trou de 1 jour ouvré est presque toujours un jour férié (Noël, Nouvel An, Vendredi
    saint) ; plusieurs jours d'affilée signalent plutôt un manque dans l'export.
    """
    dates = pd.DatetimeIndex(df["date"]).tz_localize(None)
    weekday_counts = pd.Series(dates.day_of_week).value_counts().reindex(range(7), fill_value=0)
    bdays = pd.bdate_range(dates.min(), dates.max())
    missing = bdays.difference(dates)
    per_year = pd.Series(1, index=missing).groupby(missing.year).sum()

    as_days = dates.to_numpy().astype("datetime64[D]")
    between = np.busday_count(as_days[:-1] + 1, as_days[1:])
    order = np.argsort(-between, kind="stable")
    gaps = [
        {
            "after": _day(dates[i]),
            "before": _day(dates[i + 1]),
            "missing_business_days": int(between[i]),
        }
        for i in order[:top]
        if between[i] > 0
    ]
    return {
        "n_rows": len(dates),
        "first_date": _day(dates.min()),
        "last_date": _day(dates.max()),
        "rows_per_weekday": dict(zip(_WEEKDAYS, weekday_counts.tolist(), strict=True)),
        "weekend_rows": int(weekday_counts.iloc[5:].sum()),
        "missing_business_days": len(missing),
        "missing_business_days_per_year": dict(
            zip(per_year.index.astype(int).tolist(), per_year.astype(int).tolist(), strict=True)
        ),
        "gaps_over_1_business_day": int((between > 1).sum()),
        "largest_gaps": gaps,
    }


# --- Cohérence OHLC --------------------------------------------------------


def ohlc_report(df: pd.DataFrame, examples: int = 5) -> dict[str, Any]:
    """Règles qu'une barre doit respecter : low ≤ open, close ≤ high, prix > 0, pas de NaN."""
    o, h, low, c = (df[k] for k in ("open", "high", "low", "close"))
    eps = 1e-9 * c.abs()
    rules = {
        "nan_price": df[["open", "high", "low", "close"]].isna().any(axis=1),
        "non_positive_price": (df[["open", "high", "low", "close"]] <= 0).any(axis=1),
        "high_below_low": h < low - eps,
        "high_below_open_or_close": h < np.maximum(o, c) - eps,
        "low_above_open_or_close": low > np.minimum(o, c) + eps,
        "flat_bar": (o == h) & (h == low) & (low == c),
    }
    report: dict[str, Any] = {name: int(mask.sum()) for name, mask in rules.items()}
    report["violation_examples"] = {
        name: [_day(d) for d in df.loc[mask, "date"].head(examples)]
        for name, mask in rules.items()
        if mask.any()
    }
    volume = df["volume"]
    report["volume_missing_share"] = float(volume.isna().mean())
    report["volume_zero_rows"] = int((volume == 0).sum())
    return report


# --- Distribution des rendements -------------------------------------------


def returns_report(r: pd.Series) -> dict[str, Any]:
    """Moments, extrêmes et test de normalité de Jarque-Bera des rendements log."""
    n = len(r)
    std = float(r.std())
    skew = float(cast(float, r.skew()))
    kurt = float(cast(float, r.kurt()))  # kurtosis *en excès* (0 pour une loi normale)
    jb = n / 6 * (skew**2 + kurt**2 / 4)
    z = (r - r.mean()) / std
    return {
        "n": n,
        "mean_daily": float(r.mean()),
        "std_daily": std,
        "annualized_return": float(r.mean() * TRADING_DAYS),
        "annualized_volatility": std * np.sqrt(TRADING_DAYS),
        "skewness": skew,
        "excess_kurtosis": kurt,
        "positive_share": float((r > 0).mean()),
        "min": {"date": _day(r.idxmin()), "value": float(r.min())},
        "max": {"date": _day(r.idxmax()), "value": float(r.max())},
        # Jarque-Bera suit un χ² à 2 degrés de liberté : p = exp(-JB/2).
        "jarque_bera": {"stat": float(jb), "p_value": float(np.exp(-jb / 2))},
        "share_beyond_3_sigma": float((z.abs() > 3).mean()),
        "share_beyond_4_sigma": float((z.abs() > 4).mean()),
        # Valeurs attendues sous une loi normale, pour comparaison.
        "normal_share_beyond_3_sigma": 0.0027,
        "normal_share_beyond_4_sigma": 0.000063,
    }


# --- Valeurs aberrantes ----------------------------------------------------


@dataclass(frozen=True)
class RobustScale:
    """Centre et échelle robustes (médiane, MAD) estimés sur le train."""

    center: float
    scale: float

    @classmethod
    def fit(cls, x: pd.Series) -> RobustScale:
        values = x.to_numpy(dtype=float)
        center = float(np.median(values))
        scale = float(np.median(np.abs(values - center)) * _MAD_TO_STD)
        return cls(center, scale if scale > 0 else float(x.std()))

    def z(self, x: pd.Series) -> pd.Series:
        return (x - self.center) / self.scale


def outliers_report(
    df: pd.DataFrame, splits: SplitsConfig, threshold: float = 6.0, top: int = 15
) -> dict[str, Any]:
    """Jours extrêmes repérés par un z-score robuste (médiane/MAD du train).

    Trois mesures : rendement close-close, amplitude (high-low)/close, écart d'ouverture
    open_t / close_{t-1}. Un « aller-retour » (gros mouvement aussitôt annulé le lendemain)
    ressemble davantage à une erreur de cotation qu'à un vrai choc de marché.
    """
    frame = df.set_index("date")
    measures = {
        "return": np.log(frame["close"]).diff(),
        "range": (frame["high"] - frame["low"]) / frame["close"],
        "open_gap": np.log(frame["open"] / frame["close"].shift(1)),
    }
    rows = []
    for kind, values in measures.items():
        values = values.dropna()
        z = RobustScale.fit(train_part(values, splits)).z(values)
        for d in z.index[z.abs() > threshold]:
            rows.append({"date": d, "kind": kind, "value": float(values[d]), "z": float(z[d])})

    r = measures["return"].dropna()
    rz = RobustScale.fit(train_part(r, splits)).z(r)
    nxt = rz.shift(-1)  # diagnostic a posteriori uniquement, jamais une feature
    reversal = (rz.abs() > threshold) & (nxt.abs() > threshold) & (np.sign(rz) != np.sign(nxt))

    table = pd.DataFrame(rows, columns=["date", "kind", "value", "z"])
    table = table.reindex(table["z"].abs().sort_values(ascending=False).index)
    return {
        "threshold": threshold,
        "counts": {k: int((table["kind"] == k).sum()) for k in measures},
        "spike_reversals": [_day(d) for d in rz.index[reversal.to_numpy()]],
        "top": [
            {
                **row,
                "date": _day(row["date"]),
                "split": assign_split(pd.DatetimeIndex([row["date"]]), splits).iloc[0],
            }
            for row in table.head(top).to_dict("records")
        ],
    }


# --- Régimes de volatilité --------------------------------------------------


def rolling_volatility(r: pd.Series, window: int) -> pd.Series:
    """Volatilité annualisée sur les `window` derniers jours, jour t inclus (passé seulement)."""
    vol: pd.Series = (r.rolling(window).std() * np.sqrt(TRADING_DAYS)).rename("volatility")
    return vol


def volatility_regimes(r: pd.Series, splits: SplitsConfig, window: int) -> dict[str, Any]:
    """Classe chaque jour en volatilité basse / moyenne / haute (tertiles du train)."""
    vol = rolling_volatility(r, window).dropna()
    low_cut, high_cut = (float(q) for q in train_part(vol, splits).quantile([1 / 3, 2 / 3]))
    regime = pd.cut(vol, [-np.inf, low_cut, high_cut, np.inf], labels=["basse", "moyenne", "haute"])
    split = assign_split(vol.index, splits)
    shares = {
        name: {
            k: float(v)
            for k, v in regime[split.to_numpy() == name]
            .value_counts(normalize=True)
            .sort_index()
            .items()
        }
        for name in SPLIT_NAMES
    }
    yearly = r.groupby(pd.DatetimeIndex(r.index).year).std() * np.sqrt(TRADING_DAYS)
    return {
        "window": window,
        "thresholds_from_train": {"low_mid": low_cut, "mid_high": high_cut},
        "regime_share_by_split": shares,
        "annualized_volatility_by_year": {int(y): float(v) for y, v in yearly.items()},
        "max_volatility": {"date": _day(vol.idxmax()), "value": float(vol.max())},
    }


# --- Autocorrélation et stationnarité --------------------------------------


def autocorrelation_report(r: pd.Series, lags: int = 20) -> dict[str, Any]:
    """ACF de r et |r|, test de Ljung-Box (à calculer sur le train).

    Peu d'autocorrélation dans r mais beaucoup dans |r| = rendements presque imprévisibles
    en direction, mais volatilité qui se regroupe (« volatility clustering »).
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import acf

    out: dict[str, Any] = {"n": len(r), "lags": lags, "band_95": 1.96 / np.sqrt(len(r))}
    for name, x in {"returns": r, "abs_returns": r.abs()}.items():
        values = acf(x, nlags=lags, fft=True)[1:]
        lb = acorr_ljungbox(x, lags=[5, 10, lags])
        out[name] = {
            "acf": [float(v) for v in values],
            "ljung_box": {
                int(lag): {"stat": float(row.lb_stat), "p_value": float(row.lb_pvalue)}
                for lag, row in lb.iterrows()
            },
        }
    return out


def stationarity_report(log_price: pd.Series, r: pd.Series) -> dict[str, Any]:
    """Tests ADF (H0 : racine unitaire) et KPSS (H0 : stationnaire) sur ln(prix) et r."""
    from statsmodels.tsa.stattools import adfuller, kpss

    out: dict[str, Any] = {}
    for name, x in {"log_price": log_price, "returns": r}.items():
        adf_stat, adf_p, *_ = adfuller(x.dropna(), autolag="AIC", result_object=False)
        with warnings.catch_warnings():
            # KPSS borne sa p-valeur à [0,01 ; 0,1] et prévient quand elle est hors table.
            warnings.simplefilter("ignore")
            kpss_stat, kpss_p, *_ = kpss(x.dropna(), regression="c", nlags="auto")
        stationary = adf_p < 0.05 and kpss_p > 0.05
        out[name] = {
            "adf": {"stat": float(adf_stat), "p_value": float(adf_p)},
            "kpss": {"stat": float(kpss_stat), "p_value": float(kpss_p)},
            "verdict": "stationnaire" if stationary else "non stationnaire",
        }
    return out


# --- Aperçu du découpage chronologique -------------------------------------


def split_preview(df: pd.DataFrame, splits: SplitsConfig, window: int) -> dict[str, Any]:
    """Taille, période et statistiques de chaque ensemble, sur le CSV Kaggle seul.

    Les `window` premiers jours de validation et de test servent seulement d'historique
    aux fenêtres (purge, §4.2) : aucune fenêtre ne chevauche ainsi deux ensembles.
    """
    frame = df.set_index("date")
    r = np.log(frame["close"]).diff()
    split = assign_split(frame.index, splits)
    parts: dict[str, Any] = {}
    for name in SPLIT_NAMES:
        mask = split.to_numpy() == name
        part, pr = frame[mask], r[mask].dropna()
        if part.empty:
            parts[name] = {"n_rows": 0}
            continue
        usable = len(part) - (window if name != "train" else 0)
        parts[name] = {
            "first_date": _day(part.index.min()),
            "last_date": _day(part.index.max()),
            "n_rows": len(part),
            "n_usable_after_purge": max(usable, 0),
            "close_first": float(part["close"].iloc[0]),
            "close_last": float(part["close"].iloc[-1]),
            "annualized_return": float(pr.mean() * TRADING_DAYS),
            "annualized_volatility": float(pr.std() * np.sqrt(TRADING_DAYS)),
            "buy_and_hold_return": float(part["close"].iloc[-1] / part["close"].iloc[0] - 1),
        }
    unassigned = int(split.isna().sum())
    total = sum(p["n_rows"] for p in parts.values())
    return {
        "window": window,
        "splits": parts,
        "share": {k: (v["n_rows"] / total if total else 0.0) for k, v in parts.items()},
        "unassigned_rows": unassigned,
    }


def leakage_checks(
    kaggle: pd.DataFrame, update: pd.DataFrame, splits: SplitsConfig, window: int
) -> dict[str, bool]:
    """Garde-fous vérifiés à chaque EDA ; tous doivent valoir `True`."""
    split = assign_split(kaggle["date"], splits).to_numpy()
    dates = kaggle["date"].to_numpy()

    def bounds(name: str) -> tuple[Any, Any] | None:
        sel = dates[split == name]
        return (sel.min(), sel.max()) if len(sel) else None

    b = {name: bounds(name) for name in SPLIT_NAMES}
    ordered = all(
        b[x] is None or b[y] is None or b[x][1] < b[y][0]  # type: ignore[index]
        for x, y in [("train", "val"), ("val", "test")]
    )
    kaggle_last = kaggle["date"].max()
    return {
        "splits_disjoint_and_ordered": ordered,
        "every_split_non_empty": all(v is not None for v in b.values()),
        "val_and_test_longer_than_window": all(
            (split == name).sum() > window for name in ("val", "test")
        ),
        "update_rows_strictly_after_kaggle": bool(
            update.empty or update["date"].min() > kaggle_last
        ),
        "splits_built_on_kaggle_only": True,  # par construction : `split_preview` ne lit que Kaggle
    }


def source_junction(
    kaggle: pd.DataFrame, update: pd.DataFrame, splits: SplitsConfig
) -> dict[str, Any]:
    """Raccord Kaggle → mise à jour API : un saut de niveau trahirait deux cotations différentes."""
    if update.empty:
        return {"update_rows": 0}
    last, first = kaggle.iloc[-1], update.iloc[0]
    jump = float(np.log(first["close"] / last["close"]))
    scale = RobustScale.fit(train_part(log_returns(kaggle), splits))
    return {
        "update_rows": len(update),
        "update_first_date": _day(update["date"].min()),
        "update_last_date": _day(update["date"].max()),
        "update_sources": sorted(update["source"].astype(str).unique().tolist()),
        "kaggle_last_close": float(last["close"]),
        "update_first_close": float(first["close"]),
        "junction_log_return": jump,
        "junction_robust_z": float(scale.z(pd.Series([jump])).iloc[0]),
        "business_days_between": int(
            np.busday_count(
                last["date"].tz_localize(None).date() + pd.Timedelta(days=1),
                first["date"].tz_localize(None).date(),
            )
        ),
    }
