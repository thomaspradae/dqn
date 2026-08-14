from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class RenderConfig:
    term_width: int = 60
    term_height: int = 30
    supersample: int = 4
    cell_aspect: float = 0.55
    margin: float = 0.06
    blur_passes: int = 1
    pool_max_weight: float = 0.72
    gamma: float = 0.82
    ink_threshold: float = 0.22
    hard_threshold: float = 0.52
    renderer: str = "lit"
    ambient: float = 0.34
    diffuse: float = 0.78
    specular: float = 0.10
    specular_power: float = 18.0
    normal_strength: float = 4.2
    depth_fog: float = 0.32
    z_softness: float = 0.035
    light_dir: tuple[float, float, float] = (-0.42, -0.58, 0.70)

    @property
    def high_width(self) -> int:
        return self.term_width * self.supersample

    @property
    def high_height(self) -> int:
        return self.term_height * self.supersample


@dataclass(slots=True)
class VariantSearchConfig:
    master_seed: str = "poormans-hpc"
    count: int = 16
    candidates_per_code: int = 28
    similarity_floor: float = 0.34
    silhouette_target: float = 0.58
    perspective_strength: float = 0.55
    expressive_weight: float = 0.38
    modes: tuple[str, ...] = ("surface", "contour_stack", "ribbon_skeleton", "shell_relief")


@dataclass(slots=True)
class CanonicalSpec:
    a: float = 1.1504140582765
    b: float = -2.3598824892870325
    c: float = 2.223958296969601
    d: float = -1.9009456070297253
    seed: str = "poormans-hpc:dejong:00012"
    rank: int = 9
    raw_score: float = 8.470892
    selection_score: float = 9.039588
    iterations: int = 42_000
    burn_in: int = 700


@dataclass(slots=True)
class VariantSpec:
    code: str
    mode: str
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    depth: float
    gamma: float
    thickness: float
    ripple_x_amp: float
    ripple_y_amp: float
    ripple_r_amp: float
    ripple_x_freq: int
    ripple_y_freq: int
    ripple_r_freq: int
    ripple_x_phase: float
    ripple_y_phase: float
    ripple_r_phase: float
    twist: float
    perspective: float
    focal: float
    camera_z: float
    shell_layers: int
    search_index: int = 0


@dataclass(slots=True, eq=False)
class VariantRecord:
    code: str
    spec: VariantSpec
    term: np.ndarray
    raw_score: float
    identity_score: float
    expressive_score: float
    metrics: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = ""
    rank: int = 0

    def manifest_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "rank": self.rank,
            "code": self.code,
            "raw_score": round(float(self.raw_score), 6),
            "identity_score": round(float(self.identity_score), 6),
            "expressive_score": round(float(self.expressive_score), 6),
            "metrics": {k: round(float(v), 6) for k, v in self.metrics.items()},
            "spec": _jsonable(asdict(self.spec)),
            "metadata": _jsonable(self.metadata),
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return float(value)
    return value
