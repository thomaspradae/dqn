from __future__ import annotations

import numpy as np

from .models import AttractorFamily


class IntegrationError(RuntimeError):
    pass


def rk4_orbit(
    family: AttractorFamily,
    params: dict[str, float],
    initial: tuple[float, float, float],
    samples: int,
    stride: int = 1,
) -> np.ndarray:
    """Integrate one continuous-time system and return its ordered 3D orbit."""
    if family.field is None:
        raise IntegrationError(f"{family.name} has no continuous vector field")
    samples = max(512, int(samples))
    stride = max(1, int(stride))
    state = np.asarray(initial, dtype=np.float64)
    dt = float(family.dt)
    output = np.empty((samples, 3), dtype=np.float64)
    out_index = 0
    total_steps = family.burn_steps + samples * stride

    for step in range(total_steps):
        field = family.field
        k1 = field(state, params)
        k2 = field(state + 0.5 * dt * k1, params)
        k3 = field(state + 0.5 * dt * k2, params)
        k4 = field(state + dt * k3, params)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        if not np.isfinite(state).all() or float(np.max(np.abs(state))) > 1e6:
            raise IntegrationError(f"{family.name} orbit diverged at integration step {step}")
        if step >= family.burn_steps and (step - family.burn_steps) % stride == 0:
            output[out_index] = state
            out_index += 1

    return output[:out_index]


def generate_orbit(
    family: AttractorFamily,
    params: dict[str, float],
    initial: tuple[float, float, float],
    samples: int,
) -> np.ndarray:
    if family.generator is not None:
        return family.generator(params, initial, max(512, int(samples)), family.burn_steps)
    return rk4_orbit(family, params, initial, samples=samples)


def normalize_cloud(points: np.ndarray) -> np.ndarray:
    """Robustly center and uniformly scale a 3D cloud without faking volume."""
    pts = np.asarray(points, dtype=np.float64)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if len(pts) < 512:
        raise IntegrationError("orbit produced too few finite points")

    lo = np.quantile(pts, 0.002, axis=0)
    hi = np.quantile(pts, 0.998, axis=0)
    keep = ((pts >= lo) & (pts <= hi)).all(axis=1)
    trimmed = pts[keep]
    if len(trimmed) >= 512:
        pts = trimmed

    center = np.median(pts, axis=0)
    centered = pts - center
    radial = np.linalg.norm(centered, axis=1)
    scale = float(np.quantile(radial, 0.995))
    if scale <= 1e-12:
        raise IntegrationError("orbit collapsed to a point")
    return np.clip(centered / scale, -1.5, 1.5)


def _occupied_voxels(points: np.ndarray, resolution: int) -> int:
    mins = points.min(axis=0)
    spans = np.maximum(points.max(axis=0) - mins, 1e-12)
    cells = np.floor((points - mins) / spans * (resolution - 1)).astype(np.int64)
    keys = cells[:, 0] + resolution * (cells[:, 1] + resolution * cells[:, 2])
    return int(np.unique(keys).size)


def cloud_metrics(points: np.ndarray) -> dict[str, float]:
    centered = points - points.mean(axis=0, keepdims=True)
    covariance = np.cov(centered.T)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    largest = max(float(eigenvalues[0]), 1e-12)
    lambda2_ratio = float(eigenvalues[1] / largest)
    lambda3_ratio = float(eigenvalues[2] / largest)

    resolutions = np.array([8.0, 12.0, 18.0, 26.0])
    occupied = np.array([_occupied_voxels(points, int(r)) for r in resolutions], dtype=np.float64)
    box_dimension = float(np.polyfit(np.log(resolutions), np.log(np.maximum(occupied, 1.0)), 1)[0])

    octants = (points >= 0.0).astype(np.int64)
    octant_keys = octants[:, 0] + 2 * octants[:, 1] + 4 * octants[:, 2]
    octant_occupancy = float(np.unique(octant_keys).size / 8.0)

    radius = np.linalg.norm(centered, axis=1)
    radial_cv = float(np.std(radius) / max(np.mean(radius), 1e-12))

    # Global covariance can mistake a bent sheet or tube for a volume. Measure
    # covariance again inside spatial neighborhoods: a true local sheet has a
    # near-zero third eigenvalue even when it curls through all three axes.
    rng = np.random.default_rng(0x504F4F52)
    anchor_count = min(72, len(points))
    neighborhood = min(256, max(32, len(points) // 12))
    anchors = rng.choice(len(points), size=anchor_count, replace=False)
    local_l2: list[float] = []
    local_l3: list[float] = []
    for anchor in anchors:
        distances = np.sum((points - points[anchor]) ** 2, axis=1)
        indices = np.argpartition(distances, neighborhood - 1)[:neighborhood]
        local_values = np.sort(np.linalg.eigvalsh(np.cov(points[indices].T)))[::-1]
        local_largest = max(float(local_values[0]), 1e-12)
        local_l2.append(float(local_values[1] / local_largest))
        local_l3.append(float(local_values[2] / local_largest))
    return {
        "lambda2_ratio": lambda2_ratio,
        "lambda3_ratio": lambda3_ratio,
        "box_dimension": box_dimension,
        "octant_occupancy": octant_occupancy,
        "radial_cv": radial_cv,
        "voxels_18": float(occupied[2]),
        "local_lambda2_ratio": float(np.median(local_l2)),
        "local_lambda3_ratio": float(np.median(local_l3)),
    }


def object_quality(metrics: dict[str, float], topology: str = "volume") -> tuple[float, bool]:
    l2 = metrics["lambda2_ratio"]
    l3 = metrics["lambda3_ratio"]
    dimension = metrics["box_dimension"]
    octants = metrics["octant_occupancy"]
    local_l2 = metrics["local_lambda2_ratio"]
    local_l3 = metrics["local_lambda3_ratio"]
    score = 1.15 * min(l2 / 0.55, 1.4) + 1.85 * min(l3 / 0.30, 1.5)
    score += 0.90 * min(max(dimension - 1.0, 0.0) / 1.2, 1.3) + 0.45 * octants
    score += 0.55 * min(local_l2 / 0.45, 1.2) + 0.95 * min(local_l3 / 0.05, 1.2)
    if topology == "curve":
        # A Lissajous knot is intentionally one-dimensional locally. Demand
        # non-planarity and broad 3D usage, but do not pretend it is a volume.
        score = 1.25 * min(l2 / 0.55, 1.4) + 2.0 * min(l3 / 0.35, 1.5)
        score += 0.55 * octants + 0.50 * min(max(dimension - 0.70, 0.0) / 0.45, 1.2)
        accepted = l2 >= 0.16 and l3 >= 0.055 and octants >= 0.625 and 0.65 <= dimension <= 1.65
    else:
        accepted = (
            l2 >= 0.16
            and l3 >= 0.045
            and dimension >= 1.28
            and octants >= 0.625
            and local_l2 >= 0.10
            and local_l3 >= 0.010
        )
    return float(score), bool(accepted)
