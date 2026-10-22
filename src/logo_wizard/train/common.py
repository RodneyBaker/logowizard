"""Shared training helpers: device, seeding, resumable checkpoints, CSV logs."""

import csv
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from .. import paths

RUNS_DIR = paths.RUNS_DIR


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cosine_lr(step: int, base: float, warmup: int, total: int, floor: float = 0.05) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))


def save_checkpoint(path: Path, **state) -> None:
    """Write to a temp file then rename, so a power cut never leaves a half-written checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, device: torch.device) -> dict | None:
    if not path.exists():
        return None
    return torch.load(path, map_location=device, weights_only=False)


class CsvLog:
    def __init__(self, path: Path, fields: list[str]):
        self.path = path
        self.fields = fields
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(["time"] + fields)

    def write(self, **values) -> None:
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([time.strftime("%Y-%m-%d %H:%M:%S")] + [values.get(k, "") for k in self.fields])


class Deadline:
    """Stop cleanly after --max-minutes, for evening and overnight sessions."""

    def __init__(self, minutes: float | None):
        self.end = time.monotonic() + minutes * 60 if minutes else None

    def passed(self) -> bool:
        return self.end is not None and time.monotonic() > self.end
