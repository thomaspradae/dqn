from __future__ import annotations

import hashlib
import math
import random

import numpy as np

from .models import VariantSpec
from .render import rotation_matrix_xyz, project_points, project_points_with_depth

TAU = math.tau
MODES = ("orbit_embedding", "surface", "contour_stack", "ribbon_skeleton", "shell_relief")


def rng_for_code(code: str) -> random.Random:
    seed_bytes = hashlib.sha256(code.encode("utf-8")).digest()
    return random.Random(int.from_bytes(seed_bytes[:16], "big"))


def sample_variant_spec(rng: random.Random, code: str, search_index: int, modes: tuple[str, ...] = MODES) -> VariantSpec:
    mode = rng.choice(modes)
    # Mild mode conditioning so each geometry family lives in a productive regime.
    if mode == "orbit_embedding":
        # One Z value per De Jong orbit point. Unlike the relief modes, this
        # geometry has no duplicated XY sheets to reveal when it rotates.
        depth = rng.uniform(0.42, 0.82)
        twist = rng.uniform(-0.70, 0.70)
        perspective = rng.uniform(0.16, 0.68)
        shell_layers = 1
    elif mode == "ribbon_skeleton":
        depth = rng.uniform(0.14, 0.42)
        twist = rng.uniform(-0.55, 0.55)
        perspective = rng.uniform(0.22, 0.82)
        shell_layers = rng.randint(2, 4)
    elif mode == "contour_stack":
        depth = rng.uniform(0.22, 0.58)
        twist = rng.uniform(-0.20, 0.25)
        perspective = rng.uniform(0.12, 0.64)
        shell_layers = rng.randint(4, 7)
    elif mode == "shell_relief":
        depth = rng.uniform(0.24, 0.72)
        twist = rng.uniform(-0.22, 0.22)
        perspective = rng.uniform(0.08, 0.55)
        shell_layers = rng.randint(5, 8)
    else:
        depth = rng.uniform(0.18, 0.62)
        twist = rng.uniform(-0.35, 0.35)
        perspective = rng.uniform(0.18, 0.85)
        shell_layers = rng.randint(3, 6)

    return VariantSpec(
        code=code,
        mode=mode,
        yaw_deg=rng.uniform(-30.0, 30.0),
        pitch_deg=rng.uniform(-20.0, 20.0),
        roll_deg=rng.uniform(-12.0, 12.0),
        depth=depth,
        gamma=math.exp(rng.uniform(math.log(0.68), math.log(1.65))),
        thickness=rng.uniform(0.12, 0.32),
        ripple_x_amp=rng.uniform(0.00, 0.10),
        ripple_y_amp=rng.uniform(0.00, 0.09),
        ripple_r_amp=rng.uniform(0.00, 0.07),
        ripple_x_freq=rng.randint(1, 5),
        ripple_y_freq=rng.randint(1, 5),
        ripple_r_freq=rng.randint(2, 7),
        ripple_x_phase=rng.uniform(0.0, TAU),
        ripple_y_phase=rng.uniform(0.0, TAU),
        ripple_r_phase=rng.uniform(0.0, TAU),
        twist=twist,
        perspective=perspective,
        focal=rng.uniform(2.3, 4.5),
        camera_z=rng.uniform(2.2, 3.8),
        shell_layers=shell_layers,
        search_index=search_index,
    )


def canonical_front_spec(code: str = "canonical") -> VariantSpec:
    return VariantSpec(
        code=code,
        mode="surface",
        yaw_deg=0.0,
        pitch_deg=0.0,
        roll_deg=0.0,
        depth=0.34,
        gamma=1.0,
        thickness=0.18,
        ripple_x_amp=0.015,
        ripple_y_amp=0.012,
        ripple_r_amp=0.008,
        ripple_x_freq=2,
        ripple_y_freq=3,
        ripple_r_freq=4,
        ripple_x_phase=0.0,
        ripple_y_phase=1.2,
        ripple_r_phase=0.7,
        twist=0.0,
        perspective=0.0,
        focal=3.0,
        camera_z=3.0,
        shell_layers=4,
        search_index=0,
    )


def _ripple(x: np.ndarray, y: np.ndarray, r: np.ndarray, spec: VariantSpec) -> np.ndarray:
    return (
        spec.ripple_x_amp * np.sin(spec.ripple_x_freq * math.pi * x + spec.ripple_x_phase)
        + spec.ripple_y_amp * np.sin(spec.ripple_y_freq * math.pi * y + spec.ripple_y_phase)
        + spec.ripple_r_amp * np.sin(spec.ripple_r_freq * math.pi * r + spec.ripple_r_phase)
    )


def _twist_xy(x: np.ndarray, y: np.ndarray, z: np.ndarray, spec: VariantSpec) -> tuple[np.ndarray, np.ndarray]:
    theta = spec.twist * z * math.pi
    ct = np.cos(theta)
    st = np.sin(theta)
    xx = x * ct - y * st
    yy = x * st + y * ct
    return xx, yy


def _normalize_embedding(values: np.ndarray) -> np.ndarray:
    """Robustly center and scale an orbit-derived coordinate to [-1, 1]."""
    centered = np.asarray(values, dtype=np.float64) - float(np.median(values))
    scale = float(np.quantile(np.abs(centered), 0.995))
    if scale <= 1e-12:
        return np.zeros_like(centered)
    return np.clip(centered / scale, -1.0, 1.0)


def _orbit_embedding_geometry(
    base_points: np.ndarray,
    densities: np.ndarray,
    spec: VariantSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Lift the ordered De Jong orbit into 3D without cloning its XY image.

    Each source point receives exactly one Z coordinate derived from delayed
    states of the same orbit. Orthographic projection along Z therefore
    recovers the original canonical XY point cloud exactly.
    """
    x = base_points[:, 0]
    y = base_points[:, 1]

    # The hash-derived frequencies and phases choose a deterministic delay
    # embedding while staying within a stable, visually productive regime.
    tau_x = 7 + 5 * int(spec.ripple_x_freq)
    tau_y = 19 + 7 * int(spec.ripple_y_freq)
    delayed_x = np.roll(x, tau_x)
    delayed_y = np.roll(y, tau_y)

    mix = 0.5 + 0.28 * math.tanh(spec.twist)
    harmonic = np.sin(
        math.pi * (spec.ripple_r_freq * 0.28 * x + spec.ripple_x_freq * 0.22 * y)
        + spec.ripple_r_phase
    )
    cross = np.tanh(1.8 * delayed_x * delayed_y)
    z_raw = mix * delayed_x + (1.0 - mix) * delayed_y + 0.24 * harmonic + 0.14 * cross
    z_unit = _normalize_embedding(z_raw)
    z_shaped = np.sign(z_unit) * np.power(np.abs(z_unit), np.clip(spec.gamma, 0.55, 1.8))
    z = spec.depth * z_shaped

    # Do not twist X/Y in this mode: preserving them is what guarantees that
    # the front projection remains the exact canonical De Jong geometry.
    points = np.column_stack((x, y, z))
    weights = 0.58 + 0.92 * np.clip(densities, 0.0, 1.0)
    return points, weights


def _surface_geometry(base_points: np.ndarray, densities: np.ndarray, spec: VariantSpec) -> tuple[np.ndarray, np.ndarray]:
    x = base_points[:, 0]
    y = base_points[:, 1]
    r = np.sqrt(x * x + y * y)
    height = spec.depth * np.power(np.clip(densities, 0.0, 1.0), spec.gamma)
    ripple = _ripple(x, y, r, spec)
    top = height + ripple
    bottom = -spec.thickness * height + 0.35 * ripple
    layer_fracs = np.linspace(0.0, 1.0, spec.shell_layers, dtype=np.float64)
    layers = []
    weights = []
    for frac in layer_fracs:
        z = bottom + (top - bottom) * frac
        xx, yy = _twist_xy(x, y, z, spec)
        layers.append(np.column_stack((xx, yy, z)))
        weights.append(0.55 + 0.90 * densities)
    return np.concatenate(layers, axis=0), np.concatenate(weights, axis=0)


def _shell_relief_geometry(base_points: np.ndarray, densities: np.ndarray, spec: VariantSpec) -> tuple[np.ndarray, np.ndarray]:
    x = base_points[:, 0]
    y = base_points[:, 1]
    r = np.sqrt(x * x + y * y)
    base = np.power(np.clip(densities, 0.0, 1.0), spec.gamma)
    ripple = _ripple(x, y, r, spec)
    layer_fracs = np.linspace(-1.0, 1.0, spec.shell_layers, dtype=np.float64)
    layers = []
    weights = []
    inflate = 0.18 + 0.12 * spec.thickness
    for frac in layer_fracs:
        shell = np.sign(frac) * np.abs(frac) ** 0.8
        z = shell * (0.15 + spec.depth * base) + 0.22 * ripple
        scale = 1.0 + inflate * shell * base
        xx = x * scale
        yy = y * scale
        xx, yy = _twist_xy(xx, yy, z, spec)
        layers.append(np.column_stack((xx, yy, z)))
        weights.append((0.45 + 1.10 * base) * (1.0 - 0.12 * np.abs(frac)))
    return np.concatenate(layers, axis=0), np.concatenate(weights, axis=0)


def _contour_stack_geometry(base_points: np.ndarray, densities: np.ndarray, spec: VariantSpec) -> tuple[np.ndarray, np.ndarray]:
    x = base_points[:, 0]
    y = base_points[:, 1]
    r = np.sqrt(x * x + y * y)
    ripple = _ripple(x, y, r, spec)
    n_levels = max(4, spec.shell_layers)
    thresholds = np.linspace(0.18, 0.90, n_levels)
    layers = []
    weights = []
    for i, t in enumerate(thresholds, start=1):
        mask = densities >= t
        if mask.sum() < 128:
            continue
        xx = x[mask]
        yy = y[mask]
        rr = r[mask]
        dd = densities[mask]
        rip = ripple[mask]
        z = -0.22 * spec.depth + spec.depth * (i / n_levels) + 0.16 * rip
        expand = 1.0 + 0.05 * (i / n_levels) + 0.08 * dd
        xx = xx * expand
        yy = yy * expand
        xx, yy = _twist_xy(xx, yy, np.full_like(xx, z), spec)
        layers.append(np.column_stack((xx, yy, np.full_like(xx, z))))
        weights.append(0.60 + 0.80 * dd + 0.15 * (i / n_levels))
    if not layers:
        return _surface_geometry(base_points, densities, spec)
    return np.concatenate(layers, axis=0), np.concatenate(weights, axis=0)


def _ribbon_skeleton_geometry(base_points: np.ndarray, densities: np.ndarray, spec: VariantSpec) -> tuple[np.ndarray, np.ndarray]:
    # Use the ordered trajectory itself as a backbone, then build narrow ribbons around it.
    step = max(1, len(base_points) // 6000)
    pts = base_points[::step]
    dens = densities[::step]
    if len(pts) < 256:
        pts = base_points
        dens = densities
    prev = np.roll(pts, 1, axis=0)
    nxt = np.roll(pts, -1, axis=0)
    tangent = nxt - prev
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-9)
    tangent = tangent / norm
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    x = pts[:, 0]
    y = pts[:, 1]
    r = np.sqrt(x * x + y * y)
    ripple = _ripple(x, y, r, spec)
    width = 0.018 + 0.090 * np.power(np.clip(dens, 0.0, 1.0), 0.8)
    offsets = np.array([-1.0, -0.45, 0.0, 0.45, 1.0], dtype=np.float64)
    z_layers = np.linspace(-0.35, 0.35, max(2, spec.shell_layers - 1))
    clouds = []
    weights = []
    for zi in z_layers:
        z = spec.depth * (0.65 * dens + 0.25 * ripple) + zi * spec.thickness
        for off in offsets:
            p2 = pts + normal * (off * width)[:, None]
            xx, yy = _twist_xy(p2[:, 0], p2[:, 1], z, spec)
            clouds.append(np.column_stack((xx, yy, z)))
            weights.append(0.65 + 0.70 * dens - 0.10 * np.abs(off))
    return np.concatenate(clouds, axis=0), np.concatenate(weights, axis=0)


def build_geometry(base_points: np.ndarray, densities: np.ndarray, spec: VariantSpec) -> tuple[np.ndarray, np.ndarray]:
    mode = spec.mode
    if mode == "orbit_embedding":
        return _orbit_embedding_geometry(base_points, densities, spec)
    if mode == "surface":
        return _surface_geometry(base_points, densities, spec)
    if mode == "shell_relief":
        return _shell_relief_geometry(base_points, densities, spec)
    if mode == "contour_stack":
        return _contour_stack_geometry(base_points, densities, spec)
    if mode == "ribbon_skeleton":
        return _ribbon_skeleton_geometry(base_points, densities, spec)
    raise ValueError(f"unknown mode: {mode}")


def transform_and_project(points3d: np.ndarray, spec: VariantSpec) -> tuple[np.ndarray, np.ndarray]:
    pts2, weights, _ = transform_and_project_with_depth(points3d, spec)
    return pts2, weights


def transform_and_project_with_depth(points3d: np.ndarray, spec: VariantSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = rotation_matrix_xyz(spec.pitch_deg, spec.yaw_deg, spec.roll_deg)
    rotated = points3d @ R.T
    z = rotated[:, 2]
    z = z - (z.min() + z.max()) / 2.0
    rotated = np.column_stack((rotated[:, 0], rotated[:, 1], z))
    return project_points_with_depth(rotated, spec.perspective, spec.focal, spec.camera_z)
