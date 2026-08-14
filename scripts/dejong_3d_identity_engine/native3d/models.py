from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np


VectorField = Callable[[np.ndarray, dict[str, float]], np.ndarray]
OrbitGenerator = Callable[[dict[str, float], tuple[float, float, float], int, int], np.ndarray]


@dataclass(frozen=True, slots=True)
class AttractorFamily:
    name: str
    field: VectorField | None
    defaults: dict[str, float]
    ranges: dict[str, tuple[float, float]]
    initial: tuple[float, float, float]
    dt: float
    burn_steps: int
    topology: str = "volume"
    generator: OrbitGenerator | None = None


@dataclass(frozen=True, slots=True)
class CameraPose:
    pitch_deg: float
    yaw_deg: float
    roll_deg: float
    perspective: float = 0.0


@dataclass(slots=True)
class NativeCandidate:
    id: str
    family: str
    seed: str
    params: dict[str, float]
    initial: tuple[float, float, float]
    points: np.ndarray
    object_metrics: dict[str, float]
    object_score: float
    canonical_pose: CameraPose
    terminal: np.ndarray
    projection_metrics: dict[str, float]
    projection_score: float
    total_score: float
    target_pose: CameraPose
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "seed": self.seed,
            "params": self.params,
            "initial": list(self.initial),
            "point_count": int(len(self.points)),
            "object_metrics": self.object_metrics,
            "object_score": float(self.object_score),
            "canonical_pose": asdict(self.canonical_pose),
            "projection_metrics": self.projection_metrics,
            "projection_score": float(self.projection_score),
            "total_score": float(self.total_score),
            "target_pose": asdict(self.target_pose),
            "metadata": self.metadata,
        }
