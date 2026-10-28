"""Relecture des réseaux entraînés (`outputs/models/<lstm|gru>/`) et prévision
moyennée sur les graines, dans l'échelle des rendements. Sert à l'agent de prévision."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gold_agents.models.recurrent import NetConfig, RecurrentRegressor
from gold_agents.models.train import build_network, predict

META_FILE = "model.json"


@dataclass(frozen=True)
class TrainedNetwork:
    kind: str
    models: list[RecurrentRegressor]  # un réseau par graine
    y_scale: float
    meta: dict[str, Any]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Rendement prévu de t+1 : moyenne des réseaux des différentes graines."""
        preds = np.stack([predict(m, x) for m in self.models])
        out: np.ndarray = preds.mean(axis=0).astype("float64") * self.y_scale
        return out


def save_network(
    out_dir: Path, kind: str, models: dict[int, RecurrentRegressor], meta: dict[str, Any]
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed, model in models.items():
        torch.save(model.state_dict(), out_dir / f"seed_{seed}.pt")
    meta = {**meta, "kind": kind, "seeds": sorted(models)}
    (out_dir / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_network(model_dir: Path) -> TrainedNetwork:
    meta = json.loads((model_dir / META_FILE).read_text(encoding="utf-8"))
    net = NetConfig(**meta["net"])
    models = []
    for seed in meta["seeds"]:
        model = build_network(meta["kind"], net)
        model.load_state_dict(torch.load(model_dir / f"seed_{seed}.pt", weights_only=True))
        model.eval()
        models.append(model)
    return TrainedNetwork(meta["kind"], models, float(meta["y_scale"]), meta)
