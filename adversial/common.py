from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def read_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_micro: float
    recall_micro: float
    f1_micro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float


