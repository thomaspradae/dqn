from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class RenderConfig:
    term_width: int = 60
    term_height: int = 30
    supersample: int = 4
    cell_aspect: float = 0.55
    margin: float = 0.06
    pca_align: bool = True
    blur_passes: int = 1
    pool_max_weight: float = 0.72
    gamma: float = 0.82
    ink_threshold: float = 0.22
    hard_threshold: float = 0.52

    @property
    def high_width(self) -> int:
        return self.term_width * self.supersample

    @property
    def high_height(self) -> int:
        return self.term_height * self.supersample


@dataclass(slots=True)
class SearchConfig:
    seed: str = "poormans-hpc"
    per_family: int = 64
    count: int = 24
    iteration_scale: float = 1.0
    family_balance: float = 0.35
    max_similarity: float = 0.89
    max_per_family: int = 0
    seed_each_family: bool = True


@dataclass(slots=True, eq=False)
class Candidate:
    family: str
    seed: str
    params: dict[str, Any]
    term: np.ndarray
    raw_score: float
    metrics: dict[str, float]
    family_z: float = 0.0
    family_percentile: float = 0.0
    selection_score: float = 0.0
    rank: int = 0
    id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "rank": self.rank,
            "family": self.family,
            "seed": self.seed,
            "params": _jsonable(self.params),
            "raw_score": round(float(self.raw_score), 6),
            "family_z": round(float(self.family_z), 6),
            "family_percentile": round(float(self.family_percentile), 6),
            "selection_score": round(float(self.selection_score), 6),
            "metrics": {k: round(float(v), 6) for k, v in self.metrics.items()},
            "metadata": _jsonable(self.metadata),
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return float(value)
    return value
