#!/usr/bin/env python3
"""Curate two-stage terminal marks for poormans hpc.

This generates a local HTML gallery of candidate master marks. Each candidate
is a strange-attractor density field rendered through a block-character ramp.
The candidate seed and rendering parameters are saved beside the preview so a
chosen mark can later be frozen as the permanent logo skeleton.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from cluster_art_taste import (
    attractor_params,
    attractor_quality,
    bin_points,
    digest,
    iterate_attractor,
)


RAMP = " \u2591\u2592\u2593\u2588"
DEFAULT_OUT = Path("/tmp/poormans_mark_gallery")


@dataclass
class Candidate:
    id: str
    seed: str
    family: str
    params: tuple[float, ...]
    nonce: int
    score: float
    selection_score: float
    art: list[str]
    art_variants: dict[str, list[str]]


def normalize_grid(grid: list[list[int]]) -> list[list[float]]:
    counts = [value for row in grid for value in row if value > 0]
    if not counts:
        return [[0.0 for _ in row] for row in grid]

    lo = math.log(min(counts) + 1)
    hi = math.log(max(counts) + 1)
    span = (hi - lo) or 1.0
    return [
        [(math.log(value + 1) - lo) / span if value > 0 else 0.0 for value in row]
        for row in grid
    ]


def dilate_grid(values: list[list[float]], passes: int) -> list[list[float]]:
    rows = len(values)
    cols = len(values[0])
    current = values
    for _ in range(passes):
        output = [row[:] for row in current]
        for y in range(rows):
            for x in range(cols):
                best = current[y][x]
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny = y + dy
                        nx = x + dx
                        if 0 <= ny < rows and 0 <= nx < cols:
                            best = max(best, current[ny][nx] * 0.76)
                output[y][x] = best
        current = output
    return current


def blur_grid(values: list[list[float]], passes: int) -> list[list[float]]:
    rows = len(values)
    cols = len(values[0])
    current = values
    for _ in range(passes):
        output = [[0.0 for _ in range(cols)] for _ in range(rows)]
        for y in range(rows):
            for x in range(cols):
                total = 0.0
                count = 0
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny = y + dy
                        nx = x + dx
                        if 0 <= ny < rows and 0 <= nx < cols:
                            total += current[ny][nx]
                            count += 1
                output[y][x] = total / count
        current = output

    max_value = max(value for row in current for value in row) or 1.0
    return [[value / max_value for value in row] for row in current]


def normalize_values(values: list[list[float]]) -> list[list[float]]:
    flat = [value for row in values for value in row]
    low = min(flat)
    high = max(flat)
    span = (high - low) or 1.0
    return [[(value - low) / span for value in row] for row in values]


def downsample_values(values: list[list[float]], width: int, height: int) -> list[list[float]]:
    source_h = len(values)
    source_w = len(values[0])
    if source_w == width and source_h == height:
        return values

    output = [[0.0 for _ in range(width)] for _ in range(height)]
    for y in range(height):
        y0 = int(y * source_h / height)
        y1 = max(y0 + 1, int((y + 1) * source_h / height))
        for x in range(width):
            x0 = int(x * source_w / width)
            x1 = max(x0 + 1, int((x + 1) * source_w / width))
            total = 0.0
            count = 0
            for sy in range(y0, min(y1, source_h)):
                for sx in range(x0, min(x1, source_w)):
                    total += values[sy][sx]
                    count += 1
            output[y][x] = total / max(count, 1)
    return normalize_values(output)


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge0 == edge1:
        return 1.0 if value >= edge1 else 0.0
    t = max(0.0, min(1.0, (value - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def field_quality(values: list[list[float]]) -> float:
    rows = len(values)
    cols = len(values[0])
    low = sum(1 for row in values for value in row if value > 0.14)
    high = sum(1 for row in values for value in row if value > 0.55)
    low_coverage = low / (rows * cols)
    high_coverage = high / (rows * cols)
    if low_coverage < 0.04 or low_coverage > 0.72:
        return -1.0

    # Count hard-ink components. Big directional marks should have only a few
    # components; speckle and stain fields fragment into many islands.
    mask = [[value > 0.42 for value in row] for row in values]
    seen = [[False for _ in range(cols)] for _ in range(rows)]
    components = 0
    largest = 0
    for y in range(rows):
        for x in range(cols):
            if not mask[y][x] or seen[y][x]:
                continue
            components += 1
            stack = [(y, x)]
            seen[y][x] = True
            size = 0
            while stack:
                cy, cx = stack.pop()
                size += 1
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny = cy + dy
                    nx = cx + dx
                    if 0 <= ny < rows and 0 <= nx < cols and mask[ny][nx] and not seen[ny][nx]:
                        seen[ny][nx] = True
                        stack.append((ny, nx))
            largest = max(largest, size)

    largest_ratio = largest / max(high, 1)
    component_penalty = min(components, 40) * 0.028
    coverage_score = 1.4 * min(low_coverage, 0.42) + 1.1 * min(high_coverage, 0.28)
    return coverage_score + largest_ratio * 0.8 - component_penalty


def seeded_params(seed: str, family: str, count: int) -> tuple[float, ...]:
    raw = digest(f"{family}:{seed}:params")
    values = []
    for index in range(count):
        byte = raw[index % len(raw)]
        values.append(byte / 255.0)
    return tuple(values)


def ribbon_values(seed: str, width: int, height: int) -> tuple[list[list[float]], tuple[float, ...]]:
    """Seeded continuous bands.

    This is intentionally less chaotic than a strange attractor. The seed
    chooses direction, spacing, waviness, and thickness, but the primitive is a
    long ribbon. That pushes the output away from blobs and toward structure.
    """

    raw = digest(f"ribbon:{seed}")
    rng = random.Random(int.from_bytes(raw, "big"))
    angle = rng.choice([-0.55, -0.35, -0.18, 0.0, 0.18, 0.35, 0.55, 1.57])
    band_count = rng.choice([3, 4, 4, 5])
    spacing = 2.15 / (band_count + 1)
    thickness = rng.uniform(0.075, 0.135)
    waviness = rng.uniform(0.07, 0.18)
    freq1 = rng.choice([1.0, 1.5, 2.0, 2.5])
    freq2 = rng.choice([2.5, 3.0, 3.5, 4.0])
    phase1 = rng.uniform(0, math.tau)
    phase2 = rng.uniform(0, math.tau)
    shear = rng.uniform(-0.24, 0.24)
    rib_freq = rng.choice([5, 6, 7, 8])
    rib_phase = rng.uniform(0, math.tau)
    ca = math.cos(angle)
    sa = math.sin(angle)
    aspect = width / max(height, 1) / 2.0
    values = [[0.0 for _ in range(width)] for _ in range(height)]

    for y in range(height):
        yy = ((y + 0.5) / height - 0.5) * 2.0
        for x in range(width):
            xx = ((x + 0.5) / width - 0.5) * 2.0 * aspect
            u = xx * ca + yy * sa
            v = -xx * sa + yy * ca + shear * u
            best = 0.0
            for band in range(band_count):
                offset = -1.05 + spacing * (band + 1)
                center = (
                    offset
                    + waviness * math.sin(freq1 * math.pi * u + phase1 + band * 0.65)
                    + waviness * 0.45 * math.sin(freq2 * math.pi * u + phase2 - band * 0.45)
                )
                dist = abs(v - center)
                band_value = 1.0 - smoothstep(thickness * 0.45, thickness * 1.65, dist)
                rib = 0.86 + 0.14 * math.sin(rib_freq * math.pi * u + rib_phase + band)
                best = max(best, band_value * rib)
            values[y][x] = best
    params = (angle, band_count, spacing, thickness, waviness, freq1, freq2, shear)
    return values, params


def guilloche_points(seed: str, iterations: int) -> tuple[list[tuple[float, float]], tuple[float, ...]]:
    raw = digest(f"guilloche:{seed}")
    rng = random.Random(int.from_bytes(raw, "big"))
    loops = rng.choice([5, 6, 7, 8, 9])
    a = rng.choice([2, 3, 4, 5])
    b = rng.choice([3, 4, 5, 6, 7])
    c = rng.choice([4, 5, 6, 7, 8])
    d = rng.choice([5, 6, 7, 8, 9])
    phase = rng.uniform(0, math.tau)
    wobble = rng.uniform(0.08, 0.22)
    points = []
    for index in range(iterations):
        t = math.tau * loops * index / max(iterations - 1, 1)
        r = 0.78 + wobble * math.sin(a * t + phase) + wobble * 0.55 * math.sin(b * t - phase)
        x = r * math.cos(t) + 0.22 * math.sin(c * t + phase * 0.7)
        y = r * math.sin(t) + 0.22 * math.cos(d * t - phase * 0.4)
        points.append((x, y))
    params = (loops, a, b, c, d, phase, wobble)
    return points, params


def source_field(
    seed: str,
    family: str,
    width: int,
    height: int,
    iterations: int,
    nonce_tries: int,
) -> tuple[list[list[float]], tuple[float, ...], int, float]:
    if family in {"clifford", "dejong"}:
        best = None
        for nonce in range(max(1, nonce_tries)):
            params = attractor_params(seed, nonce, family)
            points = iterate_attractor(params, family, iterations)
            values = normalize_grid(bin_points(points, width, height))
            score = attractor_quality(points, width, height) + field_quality(values) * 0.65
            if best is None or score > best[0]:
                best = (score, values, tuple(params), nonce)
        assert best is not None
        score, values, params, nonce = best
        return values, params, nonce, score
    if family == "ribbon":
        values, params = ribbon_values(seed, width, height)
        return values, params, 0, field_quality(values)
    if family == "guilloche":
        points, params = guilloche_points(seed, iterations)
        values = normalize_grid(bin_points(points, width, height))
        return values, params, 0, field_quality(values)
    raise ValueError(f"unknown family: {family}")


def threshold_mask(x: int, y: int) -> float:
    return digest(f"poormans-mark-mask-v1:{x}:{y}")[0] / 255.0


def render_shaded(
    values: list[list[float]],
    remix_seed: str,
    jitter: float,
    gamma_wobble: float,
    row_shift: int,
) -> list[str]:
    rows = len(values)
    cols = len(values[0])
    gamma_hash = digest(f"{remix_seed}:gamma")
    gamma = 1.0 + ((gamma_hash[0] / 255.0) - 0.5) * 2.0 * gamma_wobble
    lines = []

    for y in range(rows):
        shift_hash = digest(f"{remix_seed}:row:{y}")
        shift = (shift_hash[0] % (row_shift * 2 + 1)) - row_shift if row_shift else 0
        chars = []
        for x in range(cols):
            sx = x - shift
            value = values[y][sx] if 0 <= sx < cols else 0.0
            if value <= 0.018:
                chars.append(" ")
                continue
            value = value**gamma
            jitter_hash = digest(f"{remix_seed}:jitter:{x}:{y}")
            value += ((jitter_hash[0] / 255.0) - 0.5) * jitter * min(1.0, value * 2.0)
            value = max(0.0, min(1.0, value))

            level = value * (len(RAMP) - 1)
            base = int(level)
            frac = level - base
            if frac > threshold_mask(x, y) and base < len(RAMP) - 1:
                base += 1
            chars.append(RAMP[base])
        lines.append("".join(chars).rstrip())
    return [line.ljust(cols) for line in lines]


def render_bw(
    values: list[list[float]],
    remix_seed: str,
    threshold: float,
    dither: float,
    row_shift: int,
) -> list[str]:
    rows = len(values)
    cols = len(values[0])
    lines = []

    for y in range(rows):
        shift_hash = digest(f"{remix_seed}:row:{y}")
        shift = (shift_hash[0] % (row_shift * 2 + 1)) - row_shift if row_shift else 0
        chars = []
        for x in range(cols):
            sx = x - shift
            value = values[y][sx] if 0 <= sx < cols else 0.0
            local_threshold = threshold + (threshold_mask(x, y) - 0.5) * dither
            chars.append("\u2588" if value >= local_threshold else " ")
        lines.append("".join(chars).rstrip())
    return [line.ljust(cols) for line in lines]


def render_palette(
    values: list[list[float]],
    remix_seed: str,
    palette: str,
    jitter: float,
    gamma_wobble: float,
    row_shift: int,
    bw_threshold: float,
    bw_dither: float,
) -> list[str]:
    if palette == "bw":
        return render_bw(values, remix_seed, bw_threshold, bw_dither, row_shift)
    shaded = render_shaded(values, remix_seed, jitter, gamma_wobble, row_shift)
    if palette == "bw-top2":
        return ["".join("\u2588" if char in {"\u2593", "\u2588"} else " " for char in line) for line in shaded]
    if palette.startswith("bw-ge"):
        threshold = int(palette.removeprefix("bw-ge"))
        return [
            "".join("\u2588" if RAMP.find(char) >= threshold else " " for char in line)
            for line in shaded
        ]
    if palette.startswith("bw-only"):
        target = int(palette.removeprefix("bw-only"))
        return [
            "".join("\u2588" if RAMP.find(char) == target else " " for char in line)
            for line in shaded
        ]
    return shaded


def is_bw_palette(palette: str) -> bool:
    return palette in {"bw", "bw-top2"} or palette.startswith("bw-ge") or palette.startswith("bw-only")


def palette_label(palette: str) -> str:
    labels = {
        "shade": "\u2591\u2592\u2593\u2588 ladder",
        "bw": "\u2588 threshold",
        "bw-top2": "\u2588 top2",
        "bw-ge1": "\u2265\u2591",
        "bw-ge2": "\u2265\u2592",
        "bw-ge3": "\u2265\u2593",
        "bw-ge4": "\u2588 only",
        "bw-only1": "\u2591 only",
        "bw-only2": "\u2592 only",
        "bw-only3": "\u2593 only",
        "bw-only4": "\u2588 only",
    }
    return labels.get(palette, palette)


def levels_from_shaded(lines: list[str]) -> list[list[int]]:
    return [[max(0, RAMP.find(char)) for char in line] for line in lines]


def component_sizes(mask: list[list[bool]]) -> list[int]:
    rows = len(mask)
    cols = len(mask[0]) if rows else 0
    seen = [[False for _ in range(cols)] for _ in range(rows)]
    sizes = []
    for y in range(rows):
        for x in range(cols):
            if not mask[y][x] or seen[y][x]:
                continue
            stack = [(y, x)]
            seen[y][x] = True
            size = 0
            while stack:
                cy, cx = stack.pop()
                size += 1
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny = cy + dy
                    nx = cx + dx
                    if 0 <= ny < rows and 0 <= nx < cols and mask[ny][nx] and not seen[ny][nx]:
                        seen[ny][nx] = True
                        stack.append((ny, nx))
            sizes.append(size)
    return sizes


def enclosed_holes(mask: list[list[bool]]) -> int:
    rows = len(mask)
    cols = len(mask[0]) if rows else 0
    seen = [[False for _ in range(cols)] for _ in range(rows)]
    holes = 0
    for y in range(rows):
        for x in range(cols):
            if mask[y][x] or seen[y][x]:
                continue
            stack = [(y, x)]
            seen[y][x] = True
            touches_edge = False
            while stack:
                cy, cx = stack.pop()
                if cy in {0, rows - 1} or cx in {0, cols - 1}:
                    touches_edge = True
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny = cy + dy
                    nx = cx + dx
                    if 0 <= ny < rows and 0 <= nx < cols and not mask[ny][nx] and not seen[ny][nx]:
                        seen[ny][nx] = True
                        stack.append((ny, nx))
            if not touches_edge:
                holes += 1
    return holes


def mask_bbox(mask: list[list[bool]]) -> tuple[int, int, int, int] | None:
    coords = [(y, x) for y, row in enumerate(mask) for x, value in enumerate(row) if value]
    if not coords:
        return None
    ys = [coord[0] for coord in coords]
    xs = [coord[1] for coord in coords]
    return min(xs), min(ys), max(xs), max(ys)


def final_art_score(shaded_lines: list[str]) -> float:
    levels = levels_from_shaded(shaded_lines)
    rows = len(levels)
    cols = len(levels[0]) if rows else 0
    if not rows or not cols:
        return -100.0

    soft_mask = [[level >= 1 for level in row] for row in levels]
    hard_mask = [[level >= 3 for level in row] for row in levels]
    soft_count = sum(1 for row in soft_mask for value in row if value)
    hard_count = sum(1 for row in hard_mask for value in row if value)
    total = rows * cols
    soft_coverage = soft_count / total
    hard_coverage = hard_count / total
    if soft_coverage < 0.05 or soft_coverage > 0.72 or hard_coverage < 0.025 or hard_coverage > 0.46:
        return -100.0

    hard_sizes = sorted(component_sizes(hard_mask), reverse=True)
    if not hard_sizes:
        return -100.0
    components = len(hard_sizes)
    largest_ratio = hard_sizes[0] / hard_count
    small_components = sum(1 for size in hard_sizes if size <= 2)
    bbox = mask_bbox(soft_mask)
    assert bbox is not None
    x0, y0, x1, y1 = bbox
    bbox_w = (x1 - x0 + 1) / cols
    bbox_h = (y1 - y0 + 1) / rows

    perimeter = 0
    for y, row in enumerate(hard_mask):
        for x, value in enumerate(row):
            if not value:
                continue
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny = y + dy
                nx = x + dx
                if not (0 <= ny < rows and 0 <= nx < cols) or not hard_mask[ny][nx]:
                    perimeter += 1
    outline = perimeter / max(hard_count, 1)
    holes = enclosed_holes(hard_mask)
    max_row_density = max(sum(row) / cols for row in hard_mask)
    max_col_density = max(sum(hard_mask[y][x] for y in range(rows)) / rows for x in range(cols))

    def band(value: float, low: float, high: float) -> float:
        center = (low + high) / 2
        radius = (high - low) / 2
        return max(0.0, 1.0 - abs(value - center) / max(radius, 1e-9))

    score = 0.0
    score += band(soft_coverage, 0.18, 0.58) * 1.1
    score += band(hard_coverage, 0.06, 0.30) * 1.2
    score += band(bbox_w, 0.50, 0.96) * 0.9
    score += band(bbox_h, 0.45, 0.96) * 0.9
    score += band(outline, 0.28, 1.45) * 1.0
    score += min(holes, 5) * 0.12
    score += max(0.0, 1.0 - abs(components - 3) / 5.0) * 0.7
    score += max(0.0, 1.0 - abs(largest_ratio - 0.58) / 0.40) * 0.8
    score -= max(0, components - 9) * 0.12
    score -= small_components * 0.05
    score -= max(0.0, max_row_density - 0.68) * 1.6
    score -= max(0.0, max_col_density - 0.72) * 1.2
    score -= max(0.0, largest_ratio - 0.88) * 2.0
    return score


def art_signature(lines: list[str], width: int = 24, height: int = 12) -> tuple[int, ...]:
    levels = levels_from_shaded(lines)
    rows = len(levels)
    cols = len(levels[0]) if rows else 0
    signature = []
    for y in range(height):
        y0 = int(y * rows / height)
        y1 = max(y0 + 1, int((y + 1) * rows / height))
        for x in range(width):
            x0 = int(x * cols / width)
            x1 = max(x0 + 1, int((x + 1) * cols / width))
            total = 0
            count = 0
            for sy in range(y0, min(y1, rows)):
                for sx in range(x0, min(x1, cols)):
                    total += 1 if levels[sy][sx] >= 3 else 0
                    count += 1
            signature.append(1 if total / max(count, 1) > 0.22 else 0)
    return tuple(signature)


def signature_distance(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    union = sum(1 for a, b in zip(left, right) if a or b)
    if union == 0:
        return 0.0
    intersection = sum(1 for a, b in zip(left, right) if a and b)
    return 1.0 - intersection / union


def add_border(lines: list[str], border: str = "\u2591") -> list[str]:
    width = len(lines[0])
    edge = border * (width + 2)
    return [edge, *[f"{border}{line}{border}" for line in lines], edge]


def build_candidate(
    seed: str,
    family: str,
    width: int,
    height: int,
    iterations: int,
    nonce_tries: int,
    highres_scale: int,
    dilate: int,
    blur: int,
    jitter: float,
    gamma_wobble: float,
    row_shift: int,
    palette: str,
    bw_threshold: float,
    bw_dither: float,
    framed: bool,
) -> Candidate:
    work_width = width * max(1, highres_scale)
    work_height = height * max(1, highres_scale)
    values, params, nonce, _source_score = source_field(
        seed,
        family,
        work_width,
        work_height,
        iterations,
        nonce_tries,
    )
    values = dilate_grid(values, dilate)
    values = blur_grid(values, blur)
    values = downsample_values(values, width, height)
    shaded_for_score = render_shaded(values, f"{seed}:preview", jitter, gamma_wobble, row_shift)
    final_score = final_art_score(shaded_for_score)
    if palette == "compare":
        palette_names = ["shade", "bw-top2"]
    elif palette == "compare-levels":
        palette_names = [
            "shade",
            "bw-ge1",
            "bw-ge2",
            "bw-ge3",
            "bw-ge4",
            "bw-only1",
            "bw-only2",
            "bw-only3",
            "bw-only4",
        ]
    else:
        palette_names = [palette]
    art_variants = {}
    for palette_name in palette_names:
        art = render_palette(
            values=values,
            remix_seed=f"{seed}:preview",
            palette=palette_name,
            jitter=jitter,
            gamma_wobble=gamma_wobble,
            row_shift=row_shift,
            bw_threshold=bw_threshold,
            bw_dither=bw_dither,
        )
        if framed:
            border = " " if is_bw_palette(palette_name) else "\u2591"
            art = add_border(art, border=border)
        art_variants[palette_name] = art
    primary = art_variants[palette_names[0]]
    return Candidate("", seed, family, params, nonce, final_score, final_score, primary, art_variants)


def family_for_index(index: int, mode: str, seed: str) -> str:
    if mode in {"clifford", "dejong", "ribbon", "guilloche"}:
        return mode
    if mode == "auto":
        return "clifford" if digest(f"family:{seed}")[0] % 2 else "dejong"
    if mode == "stiff":
        return "ribbon" if index % 2 == 0 else "guilloche"
    if mode == "all":
        return ["clifford", "dejong", "ribbon", "guilloche"][index % 4]
    return "clifford" if index % 2 == 0 else "dejong"


def generate_candidates(args: argparse.Namespace) -> list[Candidate]:
    pool = max(args.pool, args.count)
    candidates = []
    for index in range(pool):
        seed = f"{args.base_seed}:candidate:{index:04d}"
        family = family_for_index(index, args.family, seed)
        candidate = build_candidate(
            seed=seed,
            family=family,
            width=args.width,
            height=args.height,
            iterations=args.iterations,
            nonce_tries=args.nonce_tries,
            highres_scale=args.highres_scale,
            dilate=args.dilate,
            blur=args.blur,
            jitter=args.jitter,
            gamma_wobble=args.gamma_wobble,
            row_shift=args.row_shift,
            palette=args.palette,
            bw_threshold=args.bw_threshold,
            bw_dither=args.bw_dither,
            framed=not args.no_frame,
        )
        candidates.append(candidate)

    selected = select_candidates(candidates, args.count)
    for index, candidate in enumerate(selected, start=1):
        candidate.id = f"{index:03d}"
    return selected


def normalize_candidate_scores(candidates: list[Candidate]) -> None:
    by_family: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        by_family.setdefault(candidate.family, []).append(candidate)

    for group in by_family.values():
        finite = [candidate.score for candidate in group if candidate.score > -50]
        if not finite:
            for candidate in group:
                candidate.selection_score = 0.0
            continue
        low = min(finite)
        high = max(finite)
        span = (high - low) or 1.0
        for candidate in group:
            if candidate.score <= -50:
                candidate.selection_score = 0.0
            else:
                candidate.selection_score = (candidate.score - low) / span


def select_candidates(candidates: list[Candidate], count: int) -> list[Candidate]:
    normalize_candidate_scores(candidates)
    viable = [candidate for candidate in candidates if candidate.score > -50]
    if len(viable) < count:
        viable = candidates[:]
    signatures = {id(candidate): art_signature(candidate.art_variants["shade"]) for candidate in viable}
    selected: list[Candidate] = []
    remaining = viable[:]

    while remaining and len(selected) < count:
        best_candidate = None
        best_objective = -1e9
        for candidate in remaining:
            if not selected:
                diversity = 1.0
            else:
                diversity = min(
                    signature_distance(signatures[id(candidate)], signatures[id(other)])
                    for other in selected
                )
            same_family = sum(1 for other in selected if other.family == candidate.family)
            objective = candidate.selection_score * 0.68 + diversity * 0.42 - same_family * 0.055
            if objective > best_objective:
                best_objective = objective
                best_candidate = candidate
        assert best_candidate is not None
        selected.append(best_candidate)
        remaining.remove(best_candidate)

    selected.sort(key=lambda candidate: candidate.selection_score, reverse=True)
    return selected


def write_candidate_files(out_dir: Path, candidates: list[Candidate], args: argparse.Namespace) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "base_seed": args.base_seed,
        "width": args.width,
        "height": args.height,
        "highres_scale": args.highres_scale,
        "iterations": args.iterations,
        "nonce_tries": args.nonce_tries,
        "dilate": args.dilate,
        "blur": args.blur,
        "jitter": args.jitter,
        "gamma_wobble": args.gamma_wobble,
        "row_shift": args.row_shift,
        "palette": args.palette,
        "bw_threshold": args.bw_threshold,
        "bw_dither": args.bw_dither,
        "framed": not args.no_frame,
        "ramp": RAMP,
        "candidates": [],
    }

    for candidate in candidates:
        text_files = {}
        for palette_name, art in candidate.art_variants.items():
            suffix = "" if len(candidate.art_variants) == 1 else f"_{palette_name}"
            text_name = f"candidate_{candidate.id}{suffix}.txt"
            text_path = out_dir / text_name
            text_path.write_text("\n".join(art) + "\n", encoding="utf-8")
            text_files[palette_name] = text_name

        json_name = f"candidate_{candidate.id}.json"
        json_path = out_dir / json_name
        record = {
            "id": candidate.id,
            "seed": candidate.seed,
            "family": candidate.family,
            "params": [round(value, 8) for value in candidate.params],
            "nonce": candidate.nonce,
            "score": candidate.score,
            "selection_score": candidate.selection_score,
            "text_file": next(iter(text_files.values())),
            "text_files": text_files,
            "json_file": json_name,
        }
        json_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest["candidates"].append(record)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def render_html(out_dir: Path, manifest_path: Path, candidates: list[Candidate]) -> Path:
    cards = []
    save_script = Path(__file__).name
    for candidate in candidates:
        escaped_seed = html.escape(candidate.seed)
        variant_blocks = []
        variant_tabs = []
        variant_links = []
        for index, (palette_name, art) in enumerate(candidate.art_variants.items()):
            hidden = "" if index == 0 else " hidden"
            active = " active" if index == 0 else ""
            escaped_art = html.escape("\n".join(art))
            label = palette_label(palette_name)
            suffix = "" if len(candidate.art_variants) == 1 else f"_{palette_name}"
            text_name = f"candidate_{candidate.id}{suffix}.txt"
            variant_tabs.append(
                f"""<button type="button" class="tab{active}" data-palette="{palette_name}" onclick="switchVariant('{candidate.id}', '{palette_name}')">{html.escape(label)}</button>"""
            )
            variant_blocks.append(
                f"""<pre class="variant{hidden}" data-palette="{palette_name}">{escaped_art}</pre>"""
            )
            variant_links.append(f"""<a href="{text_name}" download>{html.escape(label)} txt</a>""")
        save_command = (
            f"python3 scripts/{save_script} save "
            f"--manifest {manifest_path} --id {candidate.id} --out selected_poormans_mark.json"
        )
        cards.append(
            f"""
      <article class="card" id="candidate-{candidate.id}" data-id="{candidate.id}">
        <div class="meta">
          <strong>#{candidate.id}</strong>
          <span>{html.escape(candidate.family)}</span>
          <span>score {candidate.score:.3f}</span>
        </div>
        <div class="tabs">
          {''.join(variant_tabs)}
        </div>
        {''.join(variant_blocks)}
        <div class="actions">
          <button type="button" onclick="selectCandidate('{candidate.id}')">select</button>
          <button type="button" onclick="copyText('{escaped_seed}')">copy seed</button>
          {''.join(variant_links)}
          <a href="candidate_{candidate.id}.json" download>download json</a>
        </div>
        <code>{html.escape(save_command)}</code>
      </article>
"""
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>poormans mark gallery</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0b0f19;
      --panel: #111827;
      --panel2: #172033;
      --ink: #e5e7eb;
      --muted: #8b95a7;
      --line: #263244;
      --accent: #7dd3fc;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 ui-sans-serif, system-ui, sans-serif;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 2;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: rgba(11, 15, 25, 0.94);
      backdrop-filter: blur(8px);
    }}
    h1 {{
      margin: 0 0 4px;
      font-size: 18px;
      font-weight: 650;
    }}
    header p {{
      margin: 0;
      color: var(--muted);
    }}
    main {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(520px, 1fr));
      gap: 14px;
      padding: 14px;
    }}
    .card {{
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 12px;
    }}
    .card.selected {{
      border-color: var(--accent);
      background: var(--panel2);
      box-shadow: 0 0 0 1px var(--accent);
    }}
    .meta {{
      display: flex;
      gap: 12px;
      align-items: center;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .meta strong {{
      color: var(--ink);
      font-size: 15px;
    }}
    pre {{
      overflow: auto;
      margin: 0;
      padding: 12px;
      border: 1px solid #1f2937;
      border-radius: 6px;
      background: #050816;
      color: #eef2f7;
      font: 13px/1.05 "DejaVu Sans Mono", "Cascadia Mono", "Menlo", monospace;
      letter-spacing: 0;
    }}
    .tabs {{
      display: flex;
      gap: 8px;
      margin: 8px 0;
    }}
    .tab.active {{
      border-color: var(--accent);
      color: #ffffff;
      background: #172554;
    }}
    .hidden {{
      display: none;
    }}
    .actions {{
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
      margin-top: 10px;
    }}
    button, a {{
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #0f172a;
      color: var(--ink);
      padding: 6px 9px;
      text-decoration: none;
      font: inherit;
      cursor: pointer;
    }}
    button:hover, a:hover {{
      border-color: var(--accent);
    }}
    code {{
      display: block;
      margin-top: 9px;
      color: var(--muted);
      overflow-wrap: anywhere;
    }}
  </style>
</head>
<body>
  <header>
    <h1>poormans mark gallery</h1>
    <p>Click select to shortlist visually. Use the command under a card to save that candidate as JSON.</p>
  </header>
  <main>
    {''.join(cards)}
  </main>
  <script>
    const storageKey = "poormans-mark-selected";
    function applySelection(id) {{
      document.querySelectorAll(".card").forEach(card => {{
        card.classList.toggle("selected", card.dataset.id === id);
      }});
    }}
    function selectCandidate(id) {{
      localStorage.setItem(storageKey, id);
      applySelection(id);
    }}
    async function copyText(text) {{
      await navigator.clipboard.writeText(text);
    }}
    function switchVariant(id, palette) {{
      const card = document.querySelector(`#candidate-${{id}}`);
      card.querySelectorAll(".variant").forEach(pre => {{
        pre.classList.toggle("hidden", pre.dataset.palette !== palette);
      }});
      card.querySelectorAll(".tab").forEach(tab => {{
        tab.classList.toggle("active", tab.dataset.palette === palette);
      }});
    }}
    applySelection(localStorage.getItem(storageKey));
  </script>
</body>
</html>
"""
    html_path = out_dir / "index.html"
    html_path.write_text(page, encoding="utf-8")
    return html_path


def save_candidate(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = {record["id"]: record for record in manifest["candidates"]}
    if args.id not in records:
        raise SystemExit(f"candidate id {args.id!r} not found in {manifest_path}")

    record = records[args.id]
    selected = {
        "schema_version": 1,
        "selected_candidate": record,
        "gallery_manifest": str(manifest_path),
        "rendering": {
            key: manifest[key]
            for key in (
                "base_seed",
                "width",
                "height",
                "highres_scale",
                "iterations",
                "nonce_tries",
                "dilate",
                "blur",
                "jitter",
                "gamma_wobble",
                "row_shift",
                "palette",
                "bw_threshold",
                "bw_dither",
                "framed",
                "ramp",
            )
        },
    }
    out = Path(args.out)
    out.write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved {record['id']} {record['family']} {record['seed']} -> {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    gallery = sub.add_parser("gallery", help="generate a clickable local HTML gallery")
    gallery.add_argument("--out", type=Path, default=DEFAULT_OUT)
    gallery.add_argument("--base-seed", default="poormans-hpc")
    gallery.add_argument("--count", type=int, default=24)
    gallery.add_argument("--pool", type=int, default=72)
    gallery.add_argument(
        "--family",
        choices=["both", "auto", "stiff", "all", "clifford", "dejong", "ribbon", "guilloche"],
        default="both",
    )
    gallery.add_argument("--width", type=int, default=60)
    gallery.add_argument("--height", type=int, default=30)
    gallery.add_argument("--highres-scale", type=int, default=4)
    gallery.add_argument("--iterations", type=int, default=35000)
    gallery.add_argument("--nonce-tries", type=int, default=1)
    gallery.add_argument("--dilate", type=int, default=1)
    gallery.add_argument("--blur", type=int, default=1)
    gallery.add_argument("--jitter", type=float, default=0.12)
    gallery.add_argument("--gamma-wobble", type=float, default=0.12)
    gallery.add_argument("--row-shift", type=int, default=1)
    gallery.add_argument(
        "--palette",
        choices=[
            "shade",
            "bw",
            "bw-top2",
            "bw-ge1",
            "bw-ge2",
            "bw-ge3",
            "bw-ge4",
            "bw-only1",
            "bw-only2",
            "bw-only3",
            "bw-only4",
            "compare",
            "compare-levels",
        ],
        default="shade",
    )
    gallery.add_argument("--bw-threshold", type=float, default=0.23)
    gallery.add_argument("--bw-dither", type=float, default=0.20)
    gallery.add_argument("--no-frame", action="store_true")
    gallery.add_argument("--clean", action="store_true", help="delete the output directory before regenerating")

    save = sub.add_parser("save", help="save one candidate record from a gallery manifest")
    save.add_argument("--manifest", required=True)
    save.add_argument("--id", required=True)
    save.add_argument("--out", default="selected_poormans_mark.json")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        args = parser.parse_args(["gallery"])

    if args.command == "save":
        return save_candidate(args)

    if args.clean and args.out.exists():
        shutil.rmtree(args.out)

    candidates = generate_candidates(args)
    manifest_path = write_candidate_files(args.out, candidates, args)
    html_path = render_html(args.out, manifest_path, candidates)
    print(f"gallery: {html_path}")
    print(f"manifest: {manifest_path}")
    print(f"open: file://{html_path}")
    print("save a chosen candidate with the command printed under its card")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
