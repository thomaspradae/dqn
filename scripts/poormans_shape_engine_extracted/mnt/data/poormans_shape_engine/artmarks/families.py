from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


TAU = math.tau


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _jitter_anchor(rng: random.Random, anchor: tuple[float, ...], amount: float, lo: float, hi: float) -> tuple[float, ...]:
    return tuple(_clamp(v + rng.uniform(-amount, amount), lo, hi) for v in anchor)


@dataclass(slots=True)
class Family(ABC):
    name: str
    default_iterations: int
    burn_in: int = 500

    @abstractmethod
    def sample_params(self, rng: random.Random) -> dict[str, float | int]:
        raise NotImplementedError

    @abstractmethod
    def generate_points(self, params: dict[str, float | int], iterations: int) -> np.ndarray:
        raise NotImplementedError


class Clifford(Family):
    # Known productive neighborhoods plus broad exploration.
    anchors = (
        (-1.40, 1.60, 1.00, 0.70),
        (-1.70, 1.30, -0.10, -1.21),
        (1.70, 1.70, 0.60, 1.20),
        (-1.30, -1.30, -1.80, -1.90),
        (1.50, -1.80, 1.60, 0.90),
    )

    def __init__(self) -> None:
        super().__init__("clifford", 42_000, 700)

    def sample_params(self, rng: random.Random) -> dict[str, float]:
        if rng.random() < 0.68:
            a, b, c, d = _jitter_anchor(rng, rng.choice(self.anchors), 0.42, -2.8, 2.8)
        else:
            while True:
                a, b, c, d = (rng.uniform(-2.8, 2.8) for _ in range(4))
                if sum(abs(v) for v in (a, b, c, d)) > 2.8 and min(abs(a), abs(b)) > 0.25:
                    break
        return {"a": a, "b": b, "c": c, "d": d}

    def generate_points(self, p: dict[str, float], iterations: int) -> np.ndarray:
        a, b, c, d = p["a"], p["b"], p["c"], p["d"]
        total = iterations + self.burn_in
        out = np.empty((iterations, 2), dtype=np.float64)
        x = y = 0.1
        j = 0
        for i in range(total):
            x, y = math.sin(a * y) + c * math.cos(a * x), math.sin(b * x) + d * math.cos(b * y)
            if i >= self.burn_in:
                out[j] = (x, y)
                j += 1
        return out


class DeJong(Family):
    anchors = (
        (1.40, -2.30, 2.40, -2.10),
        (-2.24, 0.43, -0.65, -2.43),
        (2.01, -2.53, 1.61, -0.33),
        (-2.00, -2.00, -1.20, 2.00),
        (1.65, 1.20, -2.10, -1.35),
    )

    def __init__(self) -> None:
        super().__init__("dejong", 42_000, 700)

    def sample_params(self, rng: random.Random) -> dict[str, float]:
        if rng.random() < 0.72:
            a, b, c, d = _jitter_anchor(rng, rng.choice(self.anchors), 0.45, -2.9, 2.9)
        else:
            while True:
                a, b, c, d = (rng.uniform(-2.9, 2.9) for _ in range(4))
                if sum(abs(v) for v in (a, b, c, d)) > 3.0:
                    break
        return {"a": a, "b": b, "c": c, "d": d}

    def generate_points(self, p: dict[str, float], iterations: int) -> np.ndarray:
        a, b, c, d = p["a"], p["b"], p["c"], p["d"]
        total = iterations + self.burn_in
        out = np.empty((iterations, 2), dtype=np.float64)
        x = y = 0.1
        j = 0
        for i in range(total):
            x, y = math.sin(a * y) - math.cos(b * x), math.sin(c * x) - math.cos(d * y)
            if i >= self.burn_in:
                out[j] = (x, y)
                j += 1
        return out


class Hopalong(Family):
    def __init__(self) -> None:
        super().__init__("hopalong", 36_000, 150)

    def sample_params(self, rng: random.Random) -> dict[str, float]:
        # Smaller magnitudes usually survive low-resolution compression better.
        if rng.random() < 0.72:
            a = rng.uniform(-4.5, 4.5)
            b = rng.uniform(-3.0, 3.0)
            c = rng.uniform(-5.5, 5.5)
        else:
            a = rng.uniform(-10.0, 10.0)
            b = rng.uniform(-6.0, 6.0)
            c = rng.uniform(-10.0, 10.0)
        if abs(b) < 0.12:
            b = math.copysign(0.12, b or 1.0)
        return {"a": a, "b": b, "c": c}

    def generate_points(self, p: dict[str, float], iterations: int) -> np.ndarray:
        a, b, c = p["a"], p["b"], p["c"]
        total = iterations + self.burn_in
        out = np.empty((iterations, 2), dtype=np.float64)
        x = y = 0.1
        j = 0
        for i in range(total):
            sign_x = -1.0 if x < 0.0 else 1.0
            nx = y - sign_x * math.sqrt(abs(b * x - c))
            ny = a - x
            x, y = nx, ny
            if abs(x) > 1e9 or abs(y) > 1e9 or not (math.isfinite(x) and math.isfinite(y)):
                return out[:j]
            if i >= self.burn_in:
                out[j] = (x, y)
                j += 1
        return out


class GumowskiMira(Family):
    def __init__(self) -> None:
        super().__init__("gumowski_mira", 34_000, 600)

    def sample_params(self, rng: random.Random) -> dict[str, float]:
        # Classic productive region: a~0.008, b~0.05, vary mu heavily.
        mu = rng.uniform(-0.95, 0.92)
        if rng.random() < 0.65:
            mu = rng.choice((-0.92, -0.80, -0.70, -0.55, -0.496, -0.35, 0.35, 0.55, 0.72)) + rng.uniform(-0.07, 0.07)
        return {
            "a": rng.uniform(0.0055, 0.0115),
            "b": rng.uniform(0.035, 0.072),
            "mu": _clamp(mu, -0.98, 0.95),
        }

    @staticmethod
    def _f(x: float, mu: float) -> float:
        return mu * x + 2.0 * (1.0 - mu) * x * x / (1.0 + x * x)

    def generate_points(self, p: dict[str, float], iterations: int) -> np.ndarray:
        a, b, mu = p["a"], p["b"], p["mu"]
        total = iterations + self.burn_in
        out = np.empty((iterations, 2), dtype=np.float64)
        x, y = 0.1, 0.0
        j = 0
        for i in range(total):
            nx = y + a * (1.0 - b * y * y) * y + self._f(x, mu)
            ny = -x + self._f(nx, mu)
            x, y = nx, ny
            if abs(x) > 1e7 or abs(y) > 1e7 or not (math.isfinite(x) and math.isfinite(y)):
                return out[:j]
            if i >= self.burn_in:
                out[j] = (x, y)
                j += 1
        return out


class Ikeda(Family):
    def __init__(self) -> None:
        super().__init__("ikeda", 38_000, 700)

    def sample_params(self, rng: random.Random) -> dict[str, float]:
        return {
            "u": rng.uniform(0.76, 0.945),
            "phase": rng.uniform(0.34, 0.46),
            "strength": rng.uniform(5.55, 6.45),
        }

    def generate_points(self, p: dict[str, float], iterations: int) -> np.ndarray:
        u, phase, strength = p["u"], p["phase"], p["strength"]
        total = iterations + self.burn_in
        out = np.empty((iterations, 2), dtype=np.float64)
        x = y = 0.1
        j = 0
        for i in range(total):
            t = phase - strength / (1.0 + x * x + y * y)
            ct, st = math.cos(t), math.sin(t)
            nx = 1.0 + u * (x * ct - y * st)
            ny = u * (x * st + y * ct)
            x, y = nx, ny
            if i >= self.burn_in:
                out[j] = (x, y)
                j += 1
        return out


class Lissajous(Family):
    """Two-harmonic closed Fourier curves, intentionally more useful than a single Lissajous pair."""

    def __init__(self) -> None:
        super().__init__("lissajous", 20_000, 0)

    def sample_params(self, rng: random.Random) -> dict[str, float | int]:
        freqs = rng.sample(range(1, 10), 4)
        return {
            "ax": freqs[0],
            "bx": freqs[1],
            "ay": freqs[2],
            "by": freqs[3],
            "A": rng.uniform(0.65, 1.15),
            "B": rng.uniform(0.18, 0.72),
            "C": rng.uniform(0.65, 1.15),
            "D": rng.uniform(0.18, 0.72),
            "p1": rng.uniform(0.0, TAU),
            "p2": rng.uniform(0.0, TAU),
            "p3": rng.uniform(0.0, TAU),
            "p4": rng.uniform(0.0, TAU),
        }

    def generate_points(self, p: dict[str, float | int], iterations: int) -> np.ndarray:
        n = max(5_000, iterations)
        t = np.linspace(0.0, TAU, n, endpoint=False, dtype=np.float64)
        x = p["A"] * np.sin(p["ax"] * t + p["p1"]) + p["B"] * np.sin(p["bx"] * t + p["p2"])
        y = p["C"] * np.sin(p["ay"] * t + p["p3"]) + p["D"] * np.sin(p["by"] * t + p["p4"])
        return np.column_stack((x, y))


class Hypotrochoid(Family):
    def __init__(self) -> None:
        super().__init__("hypotrochoid", 22_000, 0)

    def sample_params(self, rng: random.Random) -> dict[str, float | int]:
        # Integer radii make closed curves; prefer coprime ratios for richer silhouettes.
        for _ in range(30):
            R = rng.randint(4, 13)
            r = rng.randint(1, R - 1)
            if math.gcd(R, r) == 1:
                break
        d = r * rng.uniform(0.45, 2.15)
        return {"R": R, "r": r, "d": d, "phase": rng.uniform(0.0, TAU)}

    def generate_points(self, p: dict[str, float | int], iterations: int) -> np.ndarray:
        R, r, d = float(p["R"]), float(p["r"]), float(p["d"])
        # Closure period for integer R/r is 2*pi*r/gcd(R,r); gcd is normally 1 here.
        loops = int(r / math.gcd(int(R), int(r)))
        t = np.linspace(0.0, TAU * loops, max(6_000, iterations), endpoint=False, dtype=np.float64) + float(p["phase"])
        q = (R - r) / r
        x = (R - r) * np.cos(t) + d * np.cos(q * t)
        y = (R - r) * np.sin(t) - d * np.sin(q * t)
        return np.column_stack((x, y))


class Superformula(Family):
    def __init__(self) -> None:
        super().__init__("superformula", 18_000, 0)

    def sample_params(self, rng: random.Random) -> dict[str, float | int]:
        # Log sampling gives both rounded and spiky regimes without spending everything on huge exponents.
        def log_uniform(lo: float, hi: float) -> float:
            return math.exp(rng.uniform(math.log(lo), math.log(hi)))

        return {
            "m": rng.randint(2, 14),
            "n1": log_uniform(0.25, 7.0),
            "n2": log_uniform(0.25, 7.0),
            "n3": log_uniform(0.25, 7.0),
            "phase": rng.uniform(0.0, TAU),
            "warp": rng.uniform(-0.17, 0.17),
            "warp_freq": rng.randint(2, 7),
        }

    def generate_points(self, p: dict[str, float | int], iterations: int) -> np.ndarray:
        n = max(5_000, iterations)
        phi = np.linspace(0.0, TAU, n, endpoint=False, dtype=np.float64) + float(p["phase"])
        m = float(p["m"])
        n1, n2, n3 = float(p["n1"]), float(p["n2"]), float(p["n3"])
        c = np.abs(np.cos(m * phi / 4.0)) ** n2
        s = np.abs(np.sin(m * phi / 4.0)) ** n3
        base = np.maximum(c + s, 1e-12)
        r = base ** (-1.0 / max(n1, 1e-6))
        r *= 1.0 + float(p["warp"]) * np.sin(float(p["warp_freq"]) * phi)
        # Avoid rare parameter sets with astronomical spikes.
        cap = np.quantile(r[np.isfinite(r)], 0.995) if np.isfinite(r).any() else 1.0
        r = np.clip(r, 0.0, max(cap, 1e-6))
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        return np.column_stack((x, y))


FAMILY_REGISTRY = {
    family.name: family
    for family in (
        Clifford(),
        DeJong(),
        Hopalong(),
        GumowskiMira(),
        Ikeda(),
        Lissajous(),
        Hypotrochoid(),
        Superformula(),
    )
}


def get_families(names: list[str] | tuple[str, ...] | None = None) -> list[Family]:
    if not names:
        return list(FAMILY_REGISTRY.values())
    missing = [name for name in names if name not in FAMILY_REGISTRY]
    if missing:
        raise ValueError(f"unknown families: {', '.join(missing)}")
    return [FAMILY_REGISTRY[name] for name in names]
