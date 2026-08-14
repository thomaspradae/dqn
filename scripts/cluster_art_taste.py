#!/usr/bin/env python3
"""Generate deterministic terminal art taste tests for cluster identities."""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import re


WALK_GLYPHS = " .:-=+*#%@"
IDENT_GLYPHS = " .:+#"
HYBRID_GLYPHS = " .:-=+*#%@"
CHAOS_DENSITY = ".123456789"


def digest(seed: str) -> bytes:
    return hashlib.sha256(seed.encode("utf-8")).digest()


def bit_pairs(data: bytes):
    for byte in data:
        yield byte & 0b11
        yield (byte >> 2) & 0b11
        yield (byte >> 4) & 0b11
        yield (byte >> 6) & 0b11


def nibbles(data: bytes):
    for byte in data:
        yield byte & 0xF
        yield (byte >> 4) & 0xF


def clean_word(value: str) -> str:
    word = re.sub(r"[^A-Za-z0-9]", "", value).upper()
    return word or "POORMANS"


def frame(lines: list[str], title: str) -> str:
    width = max(len(line) for line in lines + [title])
    top = f"+-- {title} " + "-" * max(0, width - len(title) - 3) + "+"
    body = [f"| {line.ljust(width)} |" for line in lines]
    bottom = "+" + "-" * (width + 2) + "+"
    return "\n".join([top, *body, bottom])


def ssh_walk(seed: str, width: int = 17, height: int = 9) -> list[str]:
    counts = [[0 for _ in range(width)] for _ in range(height)]
    x = width // 2
    y = height // 2
    start = (x, y)
    moves = {
        0: (-1, -1),
        1: (1, -1),
        2: (-1, 1),
        3: (1, 1),
    }

    for step in bit_pairs(digest(seed)):
        dx, dy = moves[step]
        x = max(0, min(width - 1, x + dx))
        y = max(0, min(height - 1, y + dy))
        counts[y][x] += 1

    end = (x, y)
    lines = []
    for row_idx, row in enumerate(counts):
        chars = []
        for col_idx, visits in enumerate(row):
            if (col_idx, row_idx) == start:
                chars.append("S")
            elif (col_idx, row_idx) == end:
                chars.append("E")
            else:
                chars.append(WALK_GLYPHS[min(visits, len(WALK_GLYPHS) - 1)])
        lines.append("".join(chars))
    return lines


def identicon(seed: str, width: int = 11, height: int = 9) -> list[str]:
    if width % 2 == 0:
        raise ValueError("identicon width must be odd")

    data = list(nibbles(digest(seed)))
    half = width // 2 + 1
    lines = []
    i = 0
    for _ in range(height):
        left = []
        for _ in range(half):
            value = data[i % len(data)]
            i += 1
            left.append(IDENT_GLYPHS[value % len(IDENT_GLYPHS)] if value > 3 else " ")
        right = left[:-1][::-1]
        lines.append("".join(left + right))
    return lines


def hybrid(seed: str, width: int = 17, height: int = 9) -> list[str]:
    counts = [[0 for _ in range(width)] for _ in range(height)]
    rng = random.Random(int.from_bytes(digest(seed), "big"))
    center = width // 2
    x = center
    y = height // 2

    for _ in range(width * height // 2):
        x += rng.choice([-1, 0, 1])
        y += rng.choice([-1, 0, 1])
        x = max(0, min(center, x))
        y = max(0, min(height - 1, y))
        strength = 1 + (1 if rng.random() > 0.72 else 0)
        counts[y][x] += strength
        mirror_x = width - 1 - x
        if mirror_x != x:
            counts[y][mirror_x] += strength

    lines = []
    for row in counts:
        lines.append("".join(HYBRID_GLYPHS[min(value, len(HYBRID_GLYPHS) - 1)] for value in row))
    return lines


def cubic(point0, point1, point2, point3, t: float) -> tuple[float, float]:
    u = 1.0 - t
    x = (
        u * u * u * point0[0]
        + 3 * u * u * t * point1[0]
        + 3 * u * t * t * point2[0]
        + t * t * t * point3[0]
    )
    y = (
        u * u * u * point0[1]
        + 3 * u * u * t * point1[1]
        + 3 * u * t * t * point2[1]
        + t * t * t * point3[1]
    )
    return x, y


def curve_points(rng: random.Random, width: int, height: int, band: int, bands: int) -> list[tuple[float, float]]:
    lane = (band + 0.5) * height / bands
    drift = rng.uniform(-height * 0.22, height * 0.22)
    start_y = max(0.0, min(height - 1.0, lane + drift))
    end_y = max(0.0, min(height - 1.0, lane - drift * 0.75 + rng.uniform(-1.1, 1.1)))
    controls = (
        (-2.0, start_y),
        (width * rng.uniform(0.15, 0.38), rng.uniform(-0.5, height - 0.5)),
        (width * rng.uniform(0.62, 0.85), rng.uniform(-0.5, height - 0.5)),
        (width + 1.0, end_y),
    )
    return [cubic(*controls, i / 80) for i in range(81)]


def distance_to_curve(x: float, y: float, points: list[tuple[float, float]]) -> float:
    return min(math.hypot(x - px, y - py) for px, py in points)


def flow_field(seed: str, word: str = "POORMANS", width: int = 25, height: int = 11) -> list[str]:
    """Poster-like seeded character bands.

    The bands are Bezier ribbons, not random pixels. That keeps the result
    recognizable while the seed still changes the exact shape.
    """

    raw = digest(f"flow:{seed}:{word}")
    rng = random.Random(int.from_bytes(raw, "big"))
    word = clean_word(word)
    band_count = 4
    offset = raw[0] % len(word)
    curves = []
    for band in range(band_count):
        curves.append(
            {
                "points": curve_points(rng, width, height, band, band_count),
                "glyph": word[(offset + band) % len(word)],
                "thickness": rng.uniform(0.72, 1.34),
                "phase": rng.uniform(0, math.tau),
            }
        )

    lines = []
    for row in range(height):
        chars = []
        for col in range(width):
            x = col + 0.5
            y = row + 0.5
            best = None
            for band, curve in enumerate(curves):
                dist = distance_to_curve(x, y, curve["points"])
                ripple = 0.15 * math.sin(col * 0.62 + row * 0.37 + curve["phase"])
                score = curve["thickness"] - dist + ripple
                if best is None or score > best[0]:
                    best = (score, band, curve)

            assert best is not None
            score, _, curve = best
            if score <= 0.02:
                chars.append(" ")
            elif score < 0.34:
                chars.append(".")
            else:
                chars.append(curve["glyph"])
        lines.append("".join(chars).rstrip())
    return [line.ljust(width) for line in lines]


def flow_lockup(seed: str, word: str = "POORMANS") -> list[str]:
    art = flow_field(seed, word=word, width=21, height=9)
    info = [
        ("cluster", seed.split(":", 1)[-1]),
        ("primary", "ofi1"),
        ("network", "tailscale"),
        ("nodes", "2/4 online"),
        ("exec", "slurm"),
    ]
    lines = []
    for index, line in enumerate(art):
        if index < len(info):
            key, value = info[index]
            lines.append(f"{line}   {key:<8} {value}")
        else:
            lines.append(line)
    return lines


def attractor_params(seed: str, nonce: int, family: str) -> tuple[float, float, float, float]:
    raw = digest(f"{family}:{seed}:{nonce}")
    values = []
    for index in range(4):
        chunk = int.from_bytes(raw[index * 4 : (index + 1) * 4], "big")
        values.append((chunk / 0xFFFFFFFF) * 6.0 - 3.0)
    return tuple(values)


def iterate_attractor(
    params: tuple[float, float, float, float],
    family: str,
    iterations: int,
    burn_in: int = 500,
) -> list[tuple[float, float]]:
    a, b, c, d = params
    x = y = 0.1
    points = []
    for index in range(iterations + burn_in):
        if family == "dejong":
            x, y = (
                math.sin(a * y) - math.cos(b * x),
                math.sin(c * x) - math.cos(d * y),
            )
        elif family == "clifford":
            x, y = (
                math.sin(a * y) + c * math.cos(a * x),
                math.sin(b * x) + d * math.cos(b * y),
            )
        else:
            raise ValueError(f"unknown attractor family: {family}")
        if index >= burn_in:
            points.append((x, y))
    return points


def bin_points(points: list[tuple[float, float]], width: int, height: int) -> list[list[int]]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xr = (xmax - xmin) or 1.0
    yr = (ymax - ymin) or 1.0
    grid = [[0 for _ in range(width)] for _ in range(height)]

    for x, y in points:
        col = min(width - 1, max(0, int((x - xmin) / xr * width)))
        row = min(height - 1, max(0, int((y - ymin) / yr * height)))
        grid[row][col] += 1
    return grid


def attractor_quality(points: list[tuple[float, float]], width: int, height: int) -> float:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    if max(xs) - min(xs) < 0.45 or max(ys) - min(ys) < 0.45:
        return -1.0

    grid = bin_points(points, width, height)
    hits = [value for row in grid for value in row if value > 0]
    coverage = len(hits) / (width * height)
    if not 0.08 <= coverage <= 0.62:
        return -1.0

    max_count = max(hits)
    mean_count = sum(hits) / len(hits)
    contrast = max_count / (mean_count or 1.0)
    edge_hits = 0
    for row_index, row in enumerate(grid):
        for col_index, value in enumerate(row):
            if value and (row_index in {0, height - 1} or col_index in {0, width - 1}):
                edge_hits += 1
    edge_ratio = edge_hits / len(hits)

    # Favor distinct silhouettes with some density contrast, without selecting
    # attractors that just smear across the entire box border.
    return coverage * 2.2 + min(contrast, 16.0) * 0.08 - edge_ratio * 0.9


def find_attractor(
    seed: str,
    family: str,
    width: int,
    height: int,
    iterations: int,
    tries: int,
) -> tuple[tuple[float, float, float, float], list[tuple[float, float]], int]:
    best = None
    for nonce in range(tries):
        params = attractor_params(seed, nonce, family)
        points = iterate_attractor(params, family, iterations)
        score = attractor_quality(points, width, height)
        if best is None or score > best[0]:
            best = (score, params, points, nonce)
        if score > 1.25:
            break
    assert best is not None
    _, params, points, nonce = best
    return params, points, nonce


def render_attractor_grid(points: list[tuple[float, float]], seed: str, width: int, height: int) -> list[str]:
    grid = bin_points(points, width, height)
    counts = [value for row in grid for value in row if value > 0]
    if not counts:
        return ["." * width for _ in range(height)]

    lo = math.log(min(counts) + 1)
    hi = math.log(max(counts) + 1)
    rng = random.Random(int.from_bytes(digest(f"ink:{seed}")[:8], "big"))
    lines = []
    for row in grid:
        chars = []
        for value in row:
            if value == 0:
                chars.append(".")
                continue
            t = (math.log(value + 1) - lo) / ((hi - lo) or 1.0)
            # Keep the thin ink a little irregular. Dense cells earn heavier
            # digits; sparse cells mostly remain 1s.
            jitter = rng.uniform(-0.08, 0.08)
            level = max(1, min(9, 1 + int((t + jitter) * 8)))
            if t < 0.28 and rng.random() < 0.7:
                level = 1
            chars.append(CHAOS_DENSITY[level])
        lines.append("".join(chars))
    return lines


def attractor_art(
    seed: str,
    family: str = "clifford",
    width: int = 42,
    height: int = 19,
    iterations: int = 90000,
    tries: int = 32,
) -> tuple[list[str], tuple[float, float, float, float], int]:
    params, points, nonce = find_attractor(seed, family, width, height, iterations, tries)
    return render_attractor_grid(points, f"{family}:{seed}:{nonce}", width, height), params, nonce


def attractor_lockup(seed: str, family: str = "clifford") -> list[str]:
    art, _, _ = attractor_art(seed, family=family, width=28, height=13, iterations=55000, tries=24)
    info = [
        ("cluster", seed.split(":", 1)[-1]),
        ("primary", "ofi1"),
        ("network", "tailscale"),
        ("nodes", "2/4 online"),
        ("exec", "slurm"),
    ]
    pad_top = (len(art) - len(info)) // 2
    lines = []
    for index, line in enumerate(art):
        info_index = index - pad_top
        if 0 <= info_index < len(info):
            key, value = info[info_index]
            lines.append(f"{line}   {key:<8} {value}")
        else:
            lines.append(line)
    return lines


def seeded_family(seed: str) -> str:
    return "clifford" if digest(f"family:{seed}")[0] % 2 else "dejong"


def render_seed(seed: str, word: str, styles: set[str]) -> str:
    panels = []
    if "ssh" in styles:
        panels.append(frame(ssh_walk(seed), f"ssh-walk {seed}"))
    if "identicon" in styles:
        panels.append(frame(identicon(seed), f"identicon {seed}"))
    if "hybrid" in styles:
        panels.append(frame(hybrid(seed), f"hybrid {seed}"))
    if "flow" in styles:
        panels.append(frame(flow_field(seed, word=word), f"flow {seed}"))
    if "lockup" in styles:
        panels.append(frame(flow_lockup(seed, word=word), f"lockup {seed}"))
    if "clifford" in styles:
        art, params, nonce = attractor_art(seed, family="clifford")
        _ = params
        title = f"clifford {seed} nonce={nonce}"
        panels.append(frame(art, title))
    if "dejong" in styles:
        art, params, nonce = attractor_art(seed, family="dejong")
        _ = params
        title = f"dejong {seed} nonce={nonce}"
        panels.append(frame(art, title))
    if "clifford-lockup" in styles:
        panels.append(frame(attractor_lockup(seed, family="clifford"), f"clifford-lockup {seed}"))
    if "dejong-lockup" in styles:
        panels.append(frame(attractor_lockup(seed, family="dejong"), f"dejong-lockup {seed}"))
    if "chaos-lockup" in styles:
        family = seeded_family(seed)
        panels.append(frame(attractor_lockup(seed, family=family), f"{family}-chaos-lockup {seed}"))
    return "\n\n".join(panels)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seeds", nargs="*", default=["cluster:gas", "node:ofi1", "node:old1"])
    parser.add_argument("--salt", default="", help="optional seed suffix for date/time/session variants")
    parser.add_argument("--word", default="POORMANS", help="character alphabet for flow-field bands")
    parser.add_argument(
        "--styles",
        default="flow,lockup,clifford,dejong,chaos-lockup",
        help="comma-separated: ssh,identicon,hybrid,flow,lockup,clifford,dejong,clifford-lockup,dejong-lockup,chaos-lockup,all",
    )
    args = parser.parse_args()
    styles = {item.strip() for item in args.styles.split(",") if item.strip()}
    if "all" in styles:
        styles = {
            "ssh",
            "identicon",
            "hybrid",
            "flow",
            "lockup",
            "clifford",
            "dejong",
            "clifford-lockup",
            "dejong-lockup",
            "chaos-lockup",
        }

    for index, seed in enumerate(args.seeds):
        if index:
            print()
        effective_seed = f"{seed}|{args.salt}" if args.salt else seed
        print(render_seed(effective_seed, args.word, styles))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
