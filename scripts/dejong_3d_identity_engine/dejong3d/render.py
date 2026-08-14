from __future__ import annotations

import math

import numpy as np

from .models import RenderConfig


SHADE_RAMP = " ░▒▓█"
ASCII_RAMP = " .:-=+*#%@"


def fit_points_to_grid(points: np.ndarray, width: int, height: int, cell_aspect: float, margin: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(points) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=bool)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = np.maximum(maxs - mins, 1e-12)
    centered = points - (mins + maxs) / 2.0
    usable_w = max(1.0, width * (1.0 - 2.0 * margin))
    usable_h = max(1.0, height * (1.0 - 2.0 * margin))
    physical_w = usable_w * cell_aspect
    physical_h = usable_h
    scale = min(physical_w / span[0], physical_h / span[1])

    x_phys = centered[:, 0] * scale
    y_phys = centered[:, 1] * scale
    x = x_phys / cell_aspect + (width - 1) / 2.0
    y = -y_phys + (height - 1) / 2.0
    xi = np.rint(x).astype(np.int64)
    yi = np.rint(y).astype(np.int64)
    keep = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    return xi[keep], yi[keep], keep


def box_blur3(values: np.ndarray, passes: int) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64)
    for _ in range(max(0, passes)):
        padded = np.pad(out, 1, mode="edge")
        accum = np.zeros_like(out)
        for dy in range(3):
            for dx in range(3):
                accum += padded[dy : dy + out.shape[0], dx : dx + out.shape[1]]
        out = accum / 9.0
    return out


def rasterize_weighted(points2d: np.ndarray, weights: np.ndarray | None, cfg: RenderConfig) -> np.ndarray:
    points2d = np.asarray(points2d, dtype=np.float64)
    if points2d.ndim != 2 or points2d.shape[1] != 2 or len(points2d) < 64:
        return np.zeros((cfg.high_height, cfg.high_width), dtype=np.float64)
    xs, ys, keep = fit_points_to_grid(points2d, cfg.high_width, cfg.high_height, cfg.cell_aspect, cfg.margin)
    if len(xs) == 0:
        return np.zeros((cfg.high_height, cfg.high_width), dtype=np.float64)
    grid = np.zeros((cfg.high_height, cfg.high_width), dtype=np.float64)
    if weights is None:
        np.add.at(grid, (ys, xs), 1.0)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if len(weights) != len(points2d):
            raise ValueError("weights length must match points2d")
        weights = weights[keep]
        np.add.at(grid, (ys, xs), weights)
    grid = np.log1p(grid)
    if cfg.blur_passes:
        grid = box_blur3(grid, cfg.blur_passes)
    positive = grid[grid > 0]
    if positive.size:
        cap = float(np.quantile(positive, 0.995)) or 1.0
        grid = np.clip(grid / cap, 0.0, 1.0)
    return grid


def downsample(high: np.ndarray, cfg: RenderConfig) -> np.ndarray:
    s = cfg.supersample
    h, w = cfg.term_height, cfg.term_width
    blocks = high.reshape(h, s, w, s).transpose(0, 2, 1, 3)
    block_max = blocks.max(axis=(2, 3))
    block_mean = blocks.mean(axis=(2, 3))
    term = cfg.pool_max_weight * block_max + (1.0 - cfg.pool_max_weight) * block_mean
    positive = term[term > 0]
    if positive.size:
        cap = float(np.quantile(positive, 0.99)) or 1.0
        term = np.clip(term / cap, 0.0, 1.0)
    return term


def render_terminal(points2d: np.ndarray, weights: np.ndarray | None, cfg: RenderConfig) -> np.ndarray:
    return downsample(rasterize_weighted(points2d, weights, cfg), cfg)


def _unit_vector(values: tuple[float, float, float]) -> np.ndarray:
    vec = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return vec / norm


def _screen_space_lighting(depth: np.ndarray, visible: np.ndarray, cfg: RenderConfig) -> np.ndarray:
    mask = visible.astype(np.float64)
    if not visible.any():
        return np.zeros_like(depth)

    # Smooth only for the normal estimate. The density pass remains sharp.
    numerator = box_blur3(np.where(visible, depth, 0.0), 2)
    denominator = np.maximum(box_blur3(mask, 2), 1e-9)
    smooth_depth = numerator / denominator
    smooth_depth = np.where(visible | (denominator > 0.05), smooth_depth, 0.0)

    gy, gx = np.gradient(smooth_depth)
    nx = -gx * cfg.normal_strength / max(cfg.cell_aspect, 1e-6)
    ny = -gy * cfg.normal_strength
    nz = np.ones_like(smooth_depth)
    n_norm = np.maximum(np.sqrt(nx * nx + ny * ny + nz * nz), 1e-9)
    nx, ny, nz = nx / n_norm, ny / n_norm, nz / n_norm

    light = _unit_vector(cfg.light_dir)
    lambert = np.clip(nx * light[0] + ny * light[1] + nz * light[2], 0.0, 1.0)

    half_vec = _unit_vector((light[0], light[1], light[2] + 1.0))
    spec_angle = np.clip(nx * half_vec[0] + ny * half_vec[1] + nz * half_vec[2], 0.0, 1.0)
    specular = np.power(spec_angle, max(1.0, cfg.specular_power))

    fog = (1.0 - cfg.depth_fog) + cfg.depth_fog * np.clip(smooth_depth, 0.0, 1.0)
    return (cfg.ambient + cfg.diffuse * lambert + cfg.specular * specular) * fog


def rasterize_lit_weighted(
    points2d: np.ndarray,
    depths: np.ndarray,
    weights: np.ndarray,
    cfg: RenderConfig,
) -> np.ndarray:
    points2d = np.asarray(points2d, dtype=np.float64)
    depths = np.asarray(depths, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if points2d.ndim != 2 or points2d.shape[1] != 2 or len(points2d) < 64:
        return np.zeros((cfg.high_height, cfg.high_width), dtype=np.float64)
    if len(depths) != len(points2d) or len(weights) != len(points2d):
        raise ValueError("depths and weights must match points2d")

    xs, ys, keep = fit_points_to_grid(points2d, cfg.high_width, cfg.high_height, cfg.cell_aspect, cfg.margin)
    if len(xs) == 0:
        return np.zeros((cfg.high_height, cfg.high_width), dtype=np.float64)

    depths = np.clip(depths[keep], 0.0, 1.0)
    weights = np.maximum(weights[keep], 0.0)

    depth_grid = np.full((cfg.high_height, cfg.high_width), -np.inf, dtype=np.float64)
    np.maximum.at(depth_grid, (ys, xs), depths)
    front_depth = depth_grid[ys, xs]
    visible_points = depths >= (front_depth - cfg.z_softness)

    grid = np.zeros((cfg.high_height, cfg.high_width), dtype=np.float64)
    if visible_points.any():
        np.add.at(grid, (ys[visible_points], xs[visible_points]), weights[visible_points])
    grid = np.log1p(grid)

    visible = np.isfinite(depth_grid)
    if cfg.blur_passes:
        grid = box_blur3(grid, cfg.blur_passes)
        visible = box_blur3(visible.astype(np.float64), cfg.blur_passes) > 0.02

    positive = grid[grid > 0]
    if positive.size:
        cap = float(np.quantile(positive, 0.995)) or 1.0
        grid = np.clip(grid / cap, 0.0, 1.0)

    lighting = _screen_space_lighting(np.where(np.isfinite(depth_grid), depth_grid, 0.0), visible, cfg)
    return np.clip(grid * lighting, 0.0, 1.0)


def render_terminal_lit(points2d: np.ndarray, depths: np.ndarray, weights: np.ndarray, cfg: RenderConfig) -> np.ndarray:
    return downsample(rasterize_lit_weighted(points2d, depths, weights, cfg), cfg)


def terminal_to_text(term: np.ndarray, ramp: str = SHADE_RAMP, gamma: float = 0.82) -> list[str]:
    values = np.clip(term, 0.0, 1.0) ** gamma
    n = len(ramp) - 1
    levels = np.rint(values * n).astype(np.int64)
    lines: list[str] = []
    for row in levels:
        line = "".join(ramp[int(v)] for v in row).rstrip()
        lines.append(line)
    return lines


def terminal_to_binary(term: np.ndarray, threshold: float = 0.22, ink: str = "█") -> list[str]:
    mask = term >= threshold
    return ["".join(ink if v else " " for v in row).rstrip() for row in mask]


def text_block(lines: list[str], width: int | None = None) -> str:
    if width is None:
        width = max((len(line) for line in lines), default=0)
    return "\n".join(line.ljust(width) for line in lines)


def rotation_matrix_xyz(pitch_deg: float, yaw_deg: float, roll_deg: float) -> np.ndarray:
    rx = math.radians(pitch_deg)
    ry = math.radians(yaw_deg)
    rz = math.radians(roll_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def project_points(points3d: np.ndarray, perspective: float, focal: float, camera_z: float) -> tuple[np.ndarray, np.ndarray]:
    pts2, weights, _ = project_points_with_depth(points3d, perspective, focal, camera_z)
    return pts2, weights


def project_points_with_depth(
    points3d: np.ndarray, perspective: float, focal: float, camera_z: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # perspective in [0,1]: 0 = orthographic, 1 = perspective.
    x, y, z = points3d[:, 0], points3d[:, 1], points3d[:, 2]
    denom = np.maximum(camera_z - z, 0.35)
    persp_scale = focal / denom
    ortho = np.column_stack((x, y))
    persp = np.column_stack((x * persp_scale, y * persp_scale))
    pts2 = (1.0 - perspective) * ortho + perspective * persp
    # Keep depth on a stable world-space scale. Per-frame min/max
    # normalization made an arbitrarily small amount of emerging depth occupy
    # the full lighting range, producing a visible pop at animation start.
    depth_norm = np.clip(0.5 + 0.5 * z / 1.6, 0.0, 1.0)
    weights = 0.70 + 0.65 * depth_norm
    return pts2, weights, depth_norm
