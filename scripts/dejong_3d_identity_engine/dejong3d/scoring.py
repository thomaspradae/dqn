from __future__ import annotations

from collections import deque

import numpy as np


def binary_metrics(term: np.ndarray, threshold: float) -> dict[str, float]:
    mask = np.asarray(term >= threshold, dtype=np.uint8)
    h, w = mask.shape
    coverage = float(mask.mean())
    hard_coverage = float(np.mean(term >= max(threshold, 0.52)))

    visited = np.zeros_like(mask, dtype=bool)
    component_sizes: list[int] = []
    holes = 0

    def neighbors(y: int, x: int):
        if y > 0:
            yield y - 1, x
        if y + 1 < h:
            yield y + 1, x
        if x > 0:
            yield y, x - 1
        if x + 1 < w:
            yield y, x + 1

    for y in range(h):
        for x in range(w):
            if mask[y, x] and not visited[y, x]:
                q = deque([(y, x)])
                visited[y, x] = True
                size = 0
                while q:
                    cy, cx = q.popleft()
                    size += 1
                    for ny, nx in neighbors(cy, cx):
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            q.append((ny, nx))
                component_sizes.append(size)

    if component_sizes:
        component_sizes.sort(reverse=True)
        largest_ratio = component_sizes[0] / max(sum(component_sizes), 1)
    else:
        largest_ratio = 0.0

    # Hole counting via flood-fill on inverse mask.
    inv = 1 - mask
    seen = np.zeros_like(inv, dtype=bool)
    for y in range(h):
        for x in range(w):
            if inv[y, x] and not seen[y, x]:
                q = deque([(y, x)])
                seen[y, x] = True
                touches_border = y == 0 or x == 0 or y == h - 1 or x == w - 1
                while q:
                    cy, cx = q.popleft()
                    for ny, nx in neighbors(cy, cx):
                        if inv[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            if ny == 0 or nx == 0 or ny == h - 1 or nx == w - 1:
                                touches_border = True
                            q.append((ny, nx))
                if not touches_border:
                    holes += 1

    ys, xs = np.where(mask)
    if len(xs):
        bbox_area = float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
        bbox_fill = float(mask.sum() / max(bbox_area, 1.0))
    else:
        bbox_fill = 0.0

    # 4-neighbor perimeter count.
    # Cast before differencing. uint8 would wrap -1 to 255 and make ordinary
    # silhouettes report perimeter ratios above 100.
    pad = np.pad(mask, 1).astype(np.int16)
    perimeter = np.abs(np.diff(pad, axis=0)).sum() + np.abs(np.diff(pad, axis=1)).sum()
    perimeter_ratio = float(perimeter / max(mask.sum(), 1))

    row_mass = mask.sum(axis=1).astype(np.int64)
    col_mass = mask.sum(axis=0).astype(np.int64)
    balance_x = 1.0 - abs(float(col_mass[: w // 2].sum()) - float(col_mass[w // 2 :].sum())) / max(float(mask.sum()), 1.0)
    balance_y = 1.0 - abs(float(row_mass[: h // 2].sum()) - float(row_mass[h // 2 :].sum())) / max(float(mask.sum()), 1.0)
    balance = 0.5 * (balance_x + balance_y)

    edge_touches = float(mask[0].sum() + mask[-1].sum() + mask[:, 0].sum() + mask[:, -1].sum()) / max(float(mask.sum()), 1.0)

    positive = term[term > 0]
    entropy = 0.0
    if positive.size:
        hist, _ = np.histogram(positive, bins=12, range=(0.0, 1.0), density=False)
        probs = hist / max(hist.sum(), 1)
        probs = probs[probs > 0]
        entropy = float(-(probs * np.log2(probs)).sum())

    speckle = float(sum(size <= 2 for size in component_sizes) / max(len(component_sizes), 1))
    return {
        "coverage": coverage,
        "hard_coverage": hard_coverage,
        "components": float(len(component_sizes)),
        "largest_ratio": largest_ratio,
        "holes": float(holes),
        "bbox_fill": bbox_fill,
        "perimeter_ratio": perimeter_ratio,
        "balance": balance,
        "edge_touches": edge_touches,
        "entropy": entropy,
        "speckle": speckle,
    }


def silhouette_iou(term_a: np.ndarray, term_b: np.ndarray, threshold: float) -> float:
    a = term_a >= threshold
    b = term_b >= threshold
    union = float(np.logical_or(a, b).sum())
    if union == 0:
        return 0.0
    inter = float(np.logical_and(a, b).sum())
    return inter / union


def intensity_correlation(term_a: np.ndarray, term_b: np.ndarray) -> float:
    a = term_a.ravel().astype(np.float64)
    b = term_b.ravel().astype(np.float64)
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def identity_score(term: np.ndarray, canonical_term: np.ndarray, threshold: float) -> float:
    iou = silhouette_iou(term, canonical_term, threshold)
    corr = intensity_correlation(term, canonical_term)
    metrics = binary_metrics(term, threshold)
    canon_metrics = binary_metrics(canonical_term, threshold)
    cov_pen = abs(metrics["coverage"] - canon_metrics["coverage"])
    fill_pen = abs(metrics["bbox_fill"] - canon_metrics["bbox_fill"])
    edge_pen = abs(metrics["edge_touches"] - canon_metrics["edge_touches"])
    return 2.25 * iou + 0.65 * max(corr, 0.0) - 0.75 * cov_pen - 0.35 * fill_pen - 0.25 * edge_pen


def expressive_score(metrics: dict[str, float], view_energy: float) -> float:
    cov = metrics["coverage"]
    cov_score = 1.0 - abs(cov - 0.22) / 0.22
    hard_score = 1.0 - abs(metrics["hard_coverage"] - 0.08) / 0.08
    comp_score = min(metrics["components"], 6.0) / 6.0
    hole_score = min(metrics["holes"], 4.0) / 4.0
    edge_score = min(metrics["perimeter_ratio"], 5.0) / 5.0
    entropy_score = min(metrics["entropy"], 3.5) / 3.5
    balance_score = metrics["balance"]
    penalties = 0.9 * metrics["speckle"] + 0.7 * metrics["edge_touches"] + 0.8 * metrics["largest_ratio"]
    return (
        0.80 * cov_score
        + 0.30 * hard_score
        + 0.45 * comp_score
        + 0.40 * hole_score
        + 0.65 * edge_score
        + 0.35 * entropy_score
        + 0.28 * balance_score
        + 0.35 * view_energy
        - penalties
    )
