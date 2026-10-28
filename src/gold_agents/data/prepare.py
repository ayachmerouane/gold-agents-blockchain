"""Étape `prepare` : série nettoyée -> features, cible, découpage, scaler (§4.2).

Écrit dans `data.processed_dir` :

- `xau_1d_features.parquet` : une ligne par jour, features brutes (non mises à l'échelle),
  cible, ensemble (`split`) et colonne `sample` (jour utilisable comme exemple) ;
- `scaler.json` : médiane et écart interquartile de chaque feature, calculés sur les
  exemples du train seulement ;
- `xau_1d_features.json` : fiche de l'étape (colonnes, paramètres, comptes, empreinte
  du fichier nettoyé d'origine).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from gold_agents.config import Config
from gold_agents.data.features import (
    TARGET_COLUMNS,
    RobustScaler,
    build_features,
    feature_columns,
)
from gold_agents.data.splits import check_splits, sample_mask, split_bounds
from gold_agents.data.update import sha256_file

KEPT_COLUMNS = ["date", "split", "open", "high", "low", "close", "source", "filled"]
FEATURES_FILE = "xau_1d_features.parquet"
SCALER_FILE = "scaler.json"


@dataclass(frozen=True)
class Prepared:
    frame: pd.DataFrame
    columns: list[str]
    scaler: RobustScaler


def prepare(clean: pd.DataFrame, cfg: Config) -> Prepared:
    """Features + cible + exemples utilisables + scaler ajusté sur le train."""
    columns = feature_columns(cfg.features)
    full = build_features(clean, cfg.features)
    full["sample"] = sample_mask(full, cfg.features.window)
    frame = full[[*KEPT_COLUMNS, *columns, "features_valid", *TARGET_COLUMNS, "sample"]]
    frame = frame.reset_index(drop=True)
    train = frame[frame["sample"] & (frame["split"] == "train")]
    return Prepared(frame, columns, RobustScaler.fit(train, columns))


def check_prepared(prepared: Prepared, window: int) -> list[str]:
    """Règles violées par le résultat de `prepare` (liste vide si tout va bien)."""
    frame = prepared.frame
    problems = check_splits(frame, frame["sample"], window)
    used = frame[frame["sample"]]
    if used[prepared.columns].isna().any().any():
        problems.append("exemple avec des features manquantes")
    if used["target_return"].isna().any():
        problems.append("exemple sans cible")
    return problems


def write_prepared(prepared: Prepared, cfg: Config, clean_path: Path) -> Path:
    out_dir = cfg.data.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / FEATURES_FILE
    prepared.frame.to_parquet(path, index=False)
    (out_dir / SCALER_FILE).write_text(
        json.dumps(prepared.scaler.to_dict(), indent=2), encoding="utf-8"
    )
    frame = prepared.frame
    meta = {
        "source": {"path": str(clean_path), "sha256": sha256_file(clean_path)},
        "features": prepared.columns,
        "target": "target_return (rendement log de t à t+1) ; target_up = 1 si hausse",
        "params": cfg.features.model_dump(),
        "splits": {k: str(v) for k, v in cfg.splits.model_dump().items()},
        "rows": len(frame),
        "samples": split_bounds(frame, frame["sample"]),
        "share_up_by_split": {
            name: round(float(part["target_up"].mean()), 4)
            for name, part in frame[frame["sample"]].groupby("split")
        },
        "scaler": {"type": "robust (médiane, IQR)", "fit_on": "train", "file": SCALER_FILE},
    }
    (out_dir / "xau_1d_features.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path


def load_prepared(processed_dir: Path) -> tuple[pd.DataFrame, RobustScaler]:
    """Relit le fichier de features et son scaler (pour l'entraînement, phase 3)."""
    frame = pd.read_parquet(processed_dir / FEATURES_FILE)
    scaler = RobustScaler.from_dict(
        json.loads((processed_dir / SCALER_FILE).read_text(encoding="utf-8"))
    )
    return frame, scaler
