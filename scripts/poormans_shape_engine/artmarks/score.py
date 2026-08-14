from __future__ import annotations

from collections import deque

import numpy as np

from .models import RenderConfig


REJECT_SCORE = -1_000_000.0


def _components(mask: np.ndarray) -> list[int]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    sizes: list[int] = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            q = [(y, x)]
            seen[y, x] = True
            size = 0
            while q:
                cy, cx = q.pop()
                size += 1
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            sizes.append(size)
    return sizes


def _holes(mask: np.ndarray) -> tuple[int, float]:
    h, w = mask.shape
    background = ~mask
    exterior = np.zeros_like(mask, dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for x in range(w):
        for y in (0, h - 1):
            if background[y, x] and not exterior[y, x]:
                exterior[y, x] = True
                q.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if background[y, x] and not exterior[y, x]:
                exterior[y, x] = True
                q.append((y, x))

    while q:
        cy, cx = q.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and background[ny, nx] and not exterior[ny, nx]:
                exterior[ny, nx] = True
                q.append((ny, nx))

    hole_mask = background & ~exterior
    sizes = _components(hole_mask)
    meaningful = [s for s in sizes if s >= 2]
    return len(meaningful), float(sum(meaningful) / mask.size)


def _entropy(term: np.ndarray) -> float:
    vals = term[term > 0.02]
    if vals.size < 2:
        return 0.0
    hist, _ = np.histogram(vals, bins=8, range=(0.0, 1.0))
    p = hist[hist > 0].astype(np.float64)
    p /= p.sum()
    ent = -float(np.sum(p * np.log2(p)))
    return ent / 3.0  # log2(8)


def _target(value: float, center: float, width: float) -> float:
    # Smooth 0..1 preference around a target, still tolerant of alternatives.
    z = (value - center) / max(width, 1e-9)
    return float(np.exp(-(z * z)))


def terminal_metrics(term: np.ndarray, cfg: RenderConfig) -> dict[str, float]:
    mask = term >= cfg.ink_threshold
    hard = term >= cfg.hard_threshold
    ink = int(mask.sum())
    total = mask.size
    coverage = ink / total
    hard_coverage = float(hard.mean())

    sizes = _components(mask)
    components = len(sizes)
    largest = max(sizes, default=0)
    largest_ratio = largest / max(ink, 1)
    speckle_ink = sum(s for s in sizes if s <= 2)
    speckle_ratio = speckle_ink / max(ink, 1)

    ys, xs = np.nonzero(mask)
    if ink:
        bbox_w = (xs.max() - xs.min() + 1) / mask.shape[1]
        bbox_h = (ys.max() - ys.min() + 1) / mask.shape[0]
        cx = float(xs.mean() / max(mask.shape[1] - 1, 1))
        cy = float(ys.mean() / max(mask.shape[0] - 1, 1))
        center_offset = min(1.0, ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5 / 0.7071)
    else:
        bbox_w = bbox_h = 0.0
        center_offset = 1.0

    border_count = int(mask[0, :].sum() + mask[-1, :].sum() + mask[:, 0].sum() + mask[:, -1].sum())
    border_ratio = border_count / max(ink, 1)

    transitions = int(np.count_nonzero(mask[:, 1:] != mask[:, :-1]) + np.count_nonzero(mask[1:, :] != mask[:-1, :]))
    perimeter_ratio = transitions / max(ink, 1)

    holes, hole_area = _holes(mask) if ink else (0, 0.0)
    intensity_entropy = _entropy(term)

    # How much of the occupied bounding rectangle is filled. High values are blob-like.
    bbox_cells = max(1.0, bbox_w * mask.shape[1] * bbox_h * mask.shape[0])
    bbox_fill = ink / bbox_cells

    return {
        "coverage": coverage,
        "hard_coverage": hard_coverage,
        "components": float(components),
        "largest_ratio": largest_ratio,
        "speckle_ratio": speckle_ratio,
        "bbox_w": bbox_w,
        "bbox_h": bbox_h,
        "bbox_fill": bbox_fill,
        "border_ratio": border_ratio,
        "perimeter_ratio": perimeter_ratio,
        "holes": float(holes),
        "hole_area": hole_area,
        "center_offset": center_offset,
        "entropy": intensity_entropy,
    }


def score_terminal(term: np.ndarray, cfg: RenderConfig) -> tuple[float, dict[str, float]]:
    m = terminal_metrics(term, cfg)

    # Hard garbage gates. These are deliberately broad; the continuous score does the taste work.
    if not 0.045 <= m["coverage"] <= 0.62:
        return REJECT_SCORE, m
    if m["bbox_w"] < 0.34 or m["bbox_h"] < 0.34:
        return REJECT_SCORE, m
    if m["components"] > 90 or m["speckle_ratio"] > 0.34:
        return REJECT_SCORE, m
    if m["border_ratio"] > 0.42:
        return REJECT_SCORE, m

    coverage = _target(m["coverage"], 0.27, 0.19)
    hard_density = _target(m["hard_coverage"], 0.105, 0.105)
    bbox_usage = (m["bbox_w"] * m["bbox_h"]) ** 0.5
    center = 1.0 - m["center_offset"]

    # 1-8 structures is useful; many fragments are not. One component is fine if it has holes/edges.
    comp = m["components"]
    component_score = 1.0 if 1.0 <= comp <= 8.0 else max(0.0, 1.0 - (comp - 8.0) / 28.0)
    connectedness = _target(m["largest_ratio"], 0.72, 0.36)

    # Thin structured silhouettes tend to have more perimeter per cell; solid stains have very little.
    edge_score = _target(m["perimeter_ratio"], 0.88, 0.62)
    hole_score = min(1.0, m["holes"] / 4.0) * 0.65 + min(1.0, m["hole_area"] / 0.16) * 0.35

    blob_penalty = max(0.0, (m["bbox_fill"] - 0.52) / 0.32)
    border_penalty = max(0.0, (m["border_ratio"] - 0.10) / 0.25)
    speckle_penalty = min(1.0, m["speckle_ratio"] / 0.18)

    score = (
        1.65 * coverage
        + 1.10 * hard_density
        + 1.15 * bbox_usage
        + 0.55 * center
        + 1.05 * component_score
        + 0.90 * connectedness
        + 1.35 * edge_score
        + 0.95 * hole_score
        + 0.80 * m["entropy"]
        - 1.65 * blob_penalty
        - 1.10 * border_penalty
        - 1.35 * speckle_penalty
    )
    return float(score), m
