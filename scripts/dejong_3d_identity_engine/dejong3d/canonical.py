from __future__ import annotations

import math
import hashlib

import numpy as np

from .models import CanonicalSpec, RenderConfig


def generate_dejong_points(spec: CanonicalSpec) -> np.ndarray:
    total = spec.iterations + spec.burn_in
    out = np.empty((spec.iterations, 2), dtype=np.float64)
    x = y = 0.1
    j = 0
    a, b, c, d = spec.a, spec.b, spec.c, spec.d
    for i in range(total):
        x, y = math.sin(a * y) - math.cos(b * x), math.sin(c * x) - math.cos(d * y)
        if i >= spec.burn_in:
            out[j] = (x, y)
            j += 1
    return out


def robust_trim(points: np.ndarray, lo_q: float = 0.0025, hi_q: float = 0.9975) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if len(pts) < 256:
        return pts
    lo = np.quantile(pts, lo_q, axis=0)
    hi = np.quantile(pts, hi_q, axis=0)
    keep = ((pts >= lo) & (pts <= hi)).all(axis=1)
    trimmed = pts[keep]
    return trimmed if len(trimmed) >= 256 else pts


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

    for axis in (0, 1):
        v = rotated[:, axis]
        moment = float(np.mean(v**3))
        if abs(moment) < 1e-12:
            moment = float(v[np.argmax(np.abs(v))])
        if moment < 0:
            rotated[:, axis] *= -1.0
    return rotated


def normalize_points(points: np.ndarray) -> np.ndarray:
    pts = pca_align(robust_trim(points))
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2.0
    span = np.maximum(maxs - mins, 1e-9)
    scale = 2.0 / max(span[0], span[1])
    out = (pts - center) * scale
    return out


def density_field(points: np.ndarray, width: int, height: int, blur_passes: int = 1) -> np.ndarray:
    pts = normalize_points(points)
    x = (pts[:, 0] + 1.0) * 0.5 * (width - 1)
    y = (1.0 - (pts[:, 1] + 1.0) * 0.5) * (height - 1)
    xi = np.clip(np.rint(x).astype(np.int64), 0, width - 1)
    yi = np.clip(np.rint(y).astype(np.int64), 0, height - 1)
    grid = np.zeros((height, width), dtype=np.float64)
    np.add.at(grid, (yi, xi), 1.0)
    grid = np.log1p(grid)
    if blur_passes:
        grid = box_blur3(grid, blur_passes)
    positive = grid[grid > 0]
    if positive.size:
        cap = float(np.quantile(positive, 0.995)) or 1.0
        grid = np.clip(grid / cap, 0.0, 1.0)
    return grid


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


def sample_density(field: np.ndarray, points_norm: np.ndarray) -> np.ndarray:
    h, w = field.shape
    x = (points_norm[:, 0] + 1.0) * 0.5 * (w - 1)
    y = (1.0 - (points_norm[:, 1] + 1.0) * 0.5) * (h - 1)
    xi = np.clip(np.rint(x).astype(np.int64), 0, w - 1)
    yi = np.clip(np.rint(y).astype(np.int64), 0, h - 1)
    return field[yi, xi]


def canonical_signature(points_norm: np.ndarray, cfg: RenderConfig) -> str:
    digest = hashlib.sha256()
    sample = np.rint((points_norm[:: max(1, len(points_norm) // 4096)] + 1.0) * 8192).astype(np.int64)
    digest.update(sample.tobytes())
    digest.update(str((cfg.term_width, cfg.term_height, cfg.supersample)).encode("utf-8"))
    return digest.hexdigest()[:16]
