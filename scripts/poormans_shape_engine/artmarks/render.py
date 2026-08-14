from __future__ import annotations

import math

import numpy as np

from .models import RenderConfig


SHADE_RAMP = " ░▒▓█"
ASCII_RAMP = " .:-=+*#%@"


def sanitize_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        return np.empty((0, 2), dtype=np.float64)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if len(points) < 64:
        return np.empty((0, 2), dtype=np.float64)

    # Robustly discard runaway tails without clipping the silhouette itself.
    lo = np.quantile(points, 0.0025, axis=0)
    hi = np.quantile(points, 0.9975, axis=0)
    keep = ((points >= lo) & (points <= hi)).all(axis=1)
    trimmed = points[keep]
    return trimmed if len(trimmed) >= 64 else points


def pca_align(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    if len(centered) < 3:
        return centered
    cov = np.cov(centered.T)
    if not np.isfinite(cov).all():
        return centered
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    rotated = centered @ vecs[:, order]

    # Canonicalize reflections so the same geometry doesn't flip because of eigenvector sign.
    for axis in (0, 1):
        v = rotated[:, axis]
        moment = float(np.mean(v ** 3))
        if abs(moment) < 1e-12:
            moment = float(v[np.argmax(np.abs(v))])
        if moment < 0:
            rotated[:, axis] *= -1.0
    return rotated


def fit_points_to_grid(points: np.ndarray, width: int, height: int, cell_aspect: float, margin: float) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = maxs - mins
    if span[0] <= 1e-12 or span[1] <= 1e-12:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

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
    return xi[keep], yi[keep]


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


def rasterize(points: np.ndarray, cfg: RenderConfig) -> np.ndarray:
    pts = sanitize_points(points)
    if len(pts) < 64:
        return np.zeros((cfg.high_height, cfg.high_width), dtype=np.float64)
    if cfg.pca_align:
        pts = pca_align(pts)

    xs, ys = fit_points_to_grid(pts, cfg.high_width, cfg.high_height, cfg.cell_aspect, cfg.margin)
    grid = np.zeros((cfg.high_height, cfg.high_width), dtype=np.float64)
    if len(xs):
        np.add.at(grid, (ys, xs), 1.0)

    # Log density preserves both thin structure and dense attractor cores.
    grid = np.log1p(grid)
    if cfg.blur_passes:
        grid = box_blur3(grid, cfg.blur_passes)
    positive = grid[grid > 0]
    if positive.size:
        scale = float(np.quantile(positive, 0.995)) or 1.0
        grid = np.clip(grid / scale, 0.0, 1.0)
    return grid


def downsample(high: np.ndarray, cfg: RenderConfig) -> np.ndarray:
    s = cfg.supersample
    h, w = cfg.term_height, cfg.term_width
    expected = (h * s, w * s)
    if high.shape != expected:
        raise ValueError(f"expected high-res grid {expected}, got {high.shape}")

    blocks = high.reshape(h, s, w, s).transpose(0, 2, 1, 3)
    block_max = blocks.max(axis=(2, 3))
    block_mean = blocks.mean(axis=(2, 3))
    term = cfg.pool_max_weight * block_max + (1.0 - cfg.pool_max_weight) * block_mean
    positive = term[term > 0]
    if positive.size:
        scale = float(np.quantile(positive, 0.99)) or 1.0
        term = np.clip(term / scale, 0.0, 1.0)
    return term


def render_terminal(points: np.ndarray, cfg: RenderConfig) -> np.ndarray:
    return downsample(rasterize(points, cfg), cfg)


def terminal_to_text(term: np.ndarray, ramp: str = SHADE_RAMP, gamma: float = 0.82) -> list[str]:
    values = np.clip(term, 0.0, 1.0) ** gamma
    n = len(ramp) - 1
    levels = np.rint(values * n).astype(np.int64)
    lines: list[str] = []
    for row in levels:
        line = "".join(ramp[int(i)] for i in row).rstrip()
        lines.append(line)
    return lines


def terminal_to_binary(term: np.ndarray, threshold: float = 0.22, ink: str = "█") -> list[str]:
    mask = term >= threshold
    return ["".join(ink if v else " " for v in row).rstrip() for row in mask]


def text_block(lines: list[str], width: int | None = None) -> str:
    if width is None:
        width = max((len(line) for line in lines), default=0)
    return "\n".join(line.ljust(width) for line in lines)
