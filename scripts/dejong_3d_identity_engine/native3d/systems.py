from __future__ import annotations

import math
import random

import numpy as np

from .models import AttractorFamily


TAU = math.tau


def _iterate_cyclic_map(
    equation: str,
    params: dict[str, float],
    initial: tuple[float, float, float],
    samples: int,
    burn_steps: int,
) -> np.ndarray:
    a, b, c, d, e, f = (params[k] for k in ("a", "b", "c", "d", "e", "f"))
    x, y, z = initial
    output = np.empty((samples, 3), dtype=np.float64)
    j = 0
    for step in range(samples + burn_steps):
        if equation == "dejong3d_cyclic":
            x, y, z = (
                math.sin(a * y) - math.cos(b * z),
                math.sin(c * z) - math.cos(d * x),
                math.sin(e * x) - math.cos(f * y),
            )
        elif equation == "clifford3d_cyclic":
            x, y, z = (
                math.sin(a * y) + c * math.cos(a * z),
                math.sin(b * z) + d * math.cos(b * x),
                math.sin(e * x) + f * math.cos(e * y),
            )
        else:
            x, y, z = (
                math.sin(a * y) + math.cos(b * z),
                math.sin(c * z) + math.cos(d * x),
                math.sin(e * x) + math.cos(f * y),
            )
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            return output[:j]
        if step >= burn_steps:
            output[j] = (x, y, z)
            j += 1
    return output


def _dejong3d_generator(params, initial, samples, burn_steps):
    return _iterate_cyclic_map("dejong3d_cyclic", params, initial, samples, burn_steps)


def _clifford3d_generator(params, initial, samples, burn_steps):
    return _iterate_cyclic_map("clifford3d_cyclic", params, initial, samples, burn_steps)


def _trig3d_generator(params, initial, samples, burn_steps):
    return _iterate_cyclic_map("trig3d_symmetric", params, initial, samples, burn_steps)


def _lissajous3d_generator(params, initial, samples, burn_steps):
    del initial, burn_steps
    t = np.linspace(0.0, TAU, samples, endpoint=False, dtype=np.float64)
    x = params["ax_amp"] * np.sin(round(params["ax_freq"]) * t + params["ax_phase"])
    y = params["ay_amp"] * np.sin(round(params["ay_freq"]) * t + params["ay_phase"])
    z = params["az_amp"] * np.sin(round(params["az_freq"]) * t + params["az_phase"])
    return np.column_stack((x, y, z))


def _fourier3d_generator(params, initial, samples, burn_steps):
    del initial, burn_steps
    t = np.linspace(0.0, TAU, samples, endpoint=False, dtype=np.float64)
    axes = []
    for axis in ("x", "y", "z"):
        values = params[f"{axis}1_amp"] * np.sin(
            round(params[f"{axis}1_freq"]) * t + params[f"{axis}1_phase"]
        )
        values += params[f"{axis}2_amp"] * np.sin(
            round(params[f"{axis}2_freq"]) * t + params[f"{axis}2_phase"]
        )
        axes.append(values)
    return np.column_stack(axes)


def _aizawa(p: np.ndarray, q: dict[str, float]) -> np.ndarray:
    x, y, z = p
    a, b, c, d, e, f = (q[k] for k in ("a", "b", "c", "d", "e", "f"))
    return np.array(
        [
            (z - b) * x - d * y,
            d * x + (z - b) * y,
            c + a * z - z**3 / 3.0 - (x * x + y * y) * (1.0 + e * z) + f * z * x**3,
        ],
        dtype=np.float64,
    )


def _thomas(p: np.ndarray, q: dict[str, float]) -> np.ndarray:
    x, y, z = p
    b = q["b"]
    return np.array([math.sin(y) - b * x, math.sin(z) - b * y, math.sin(x) - b * z], dtype=np.float64)


def _halvorsen(p: np.ndarray, q: dict[str, float]) -> np.ndarray:
    x, y, z = p
    a = q["a"]
    return np.array(
        [
            -a * x - 4.0 * y - 4.0 * z - y * y,
            -a * y - 4.0 * z - 4.0 * x - z * z,
            -a * z - 4.0 * x - 4.0 * y - x * x,
        ],
        dtype=np.float64,
    )


def _dadras(p: np.ndarray, q: dict[str, float]) -> np.ndarray:
    x, y, z = p
    a, b, c, d, e = (q[k] for k in ("a", "b", "c", "d", "e"))
    return np.array([y - a * x + b * y * z, c * y - x * z + z, d * x * y - e * z], dtype=np.float64)


def _arneodo(p: np.ndarray, q: dict[str, float]) -> np.ndarray:
    x, y, z = p
    a, b, c = (q[k] for k in ("a", "b", "c"))
    return np.array([y, z, -a * x - b * y - z + c * x**3], dtype=np.float64)


FAMILIES: dict[str, AttractorFamily] = {
    "dejong3d_cyclic": AttractorFamily(
        "dejong3d_cyclic",
        None,
        {"a": 1.150414, "b": -2.359882, "c": 2.223958, "d": -1.900946, "e": 1.65, "f": -2.10},
        {"a": (0.85, 2.65), "b": (-2.75, -1.10), "c": (1.05, 2.75), "d": (-2.70, -0.45), "e": (-2.70, 2.70), "f": (-2.70, 2.70)},
        (0.11, -0.07, 0.13),
        1.0,
        1200,
        "volume",
        _dejong3d_generator,
    ),
    "clifford3d_cyclic": AttractorFamily(
        "clifford3d_cyclic",
        None,
        {"a": 1.70, "b": 1.70, "c": 0.60, "d": 1.20, "e": -1.30, "f": -1.30},
        {"a": (-2.60, 2.60), "b": (-2.60, 2.60), "c": (-1.85, 1.85), "d": (-1.85, 1.85), "e": (-2.60, 2.60), "f": (-1.85, 1.85)},
        (0.11, -0.07, 0.13),
        1.0,
        1200,
        "volume",
        _clifford3d_generator,
    ),
    "trig3d_symmetric": AttractorFamily(
        "trig3d_symmetric",
        None,
        {"a": 2.10, "b": 1.70, "c": -2.30, "d": 1.40, "e": 2.70, "f": -1.80},
        {"a": (-3.40, 3.40), "b": (-3.40, 3.40), "c": (-3.40, 3.40), "d": (-3.40, 3.40), "e": (-3.40, 3.40), "f": (-3.40, 3.40)},
        (0.11, -0.07, 0.13),
        1.0,
        1200,
        "volume",
        _trig3d_generator,
    ),
    "lissajous3d": AttractorFamily(
        "lissajous3d",
        None,
        {"ax_freq": 3.0, "ay_freq": 4.0, "az_freq": 7.0, "ax_amp": 1.0, "ay_amp": 1.0, "az_amp": 1.0, "ax_phase": 0.0, "ay_phase": math.pi / 3.0, "az_phase": math.pi / 5.0},
        {"ax_freq": (1.0, 12.0), "ay_freq": (1.0, 12.0), "az_freq": (1.0, 12.0), "ax_amp": (0.72, 1.18), "ay_amp": (0.72, 1.18), "az_amp": (0.72, 1.18), "ax_phase": (0.0, TAU), "ay_phase": (0.0, TAU), "az_phase": (0.0, TAU)},
        (0.0, 0.0, 0.0),
        1.0,
        0,
        "curve",
        _lissajous3d_generator,
    ),
    "fourier3d": AttractorFamily(
        "fourier3d",
        None,
        {
            "x1_freq": 2.0, "x2_freq": 7.0, "x1_amp": 1.0, "x2_amp": 0.42, "x1_phase": 0.0, "x2_phase": 0.7,
            "y1_freq": 3.0, "y2_freq": 8.0, "y1_amp": 1.0, "y2_amp": 0.38, "y1_phase": 1.1, "y2_phase": 2.2,
            "z1_freq": 5.0, "z2_freq": 9.0, "z1_amp": 1.0, "z2_amp": 0.46, "z1_phase": 0.4, "z2_phase": 1.8,
        },
        {
            **{f"{axis}{harmonic}_freq": (1.0, 12.0) for axis in "xyz" for harmonic in (1, 2)},
            **{f"{axis}1_amp": (0.72, 1.15) for axis in "xyz"},
            **{f"{axis}2_amp": (0.18, 0.62) for axis in "xyz"},
            **{f"{axis}{harmonic}_phase": (0.0, TAU) for axis in "xyz" for harmonic in (1, 2)},
        },
        (0.0, 0.0, 0.0),
        1.0,
        0,
        "curve",
        _fourier3d_generator,
    ),
    "aizawa": AttractorFamily(
        "aizawa",
        _aizawa,
        {"a": 0.95, "b": 0.70, "c": 0.60, "d": 3.50, "e": 0.25, "f": 0.10},
        {"a": (0.86, 1.04), "b": (0.62, 0.78), "c": (0.52, 0.68), "d": (3.15, 3.85), "e": (0.16, 0.34), "f": (0.055, 0.145)},
        (0.10, 0.0, 0.0),
        0.01,
        4500,
    ),
    "thomas": AttractorFamily(
        "thomas",
        _thomas,
        {"b": 0.185},
        {"b": (0.165, 0.205)},
        (0.10, 0.0, -0.10),
        0.045,
        3000,
    ),
    "halvorsen": AttractorFamily(
        "halvorsen",
        _halvorsen,
        {"a": 1.40},
        {"a": (1.30, 1.50)},
        (1.0, 0.0, 0.0),
        0.004,
        5000,
    ),
    "dadras": AttractorFamily(
        "dadras",
        _dadras,
        {"a": 3.0, "b": 2.7, "c": 1.7, "d": 2.0, "e": 9.0},
        {"a": (2.75, 3.25), "b": (2.45, 2.95), "c": (1.48, 1.92), "d": (1.75, 2.25), "e": (8.2, 9.8)},
        (1.10, 2.10, -2.0),
        0.004,
        5000,
    ),
    "arneodo": AttractorFamily(
        "arneodo",
        _arneodo,
        {"a": -5.5, "b": 3.5, "c": -1.0},
        {"a": (-5.75, -5.25), "b": (3.25, 3.75), "c": (-1.10, -0.90)},
        (0.10, 0.0, 0.0),
        0.008,
        5000,
    ),
}


def sample_parameters(family: AttractorFamily, rng: random.Random, index: int) -> dict[str, float]:
    if index == 0:
        return dict(family.defaults)
    sampled = {name: rng.uniform(lo, hi) for name, (lo, hi) in family.ranges.items()}
    if family.name == "lissajous3d":
        frequencies = rng.sample(range(1, 13), 3)
        for axis, frequency in zip(("x", "y", "z"), frequencies):
            sampled[f"a{axis}_freq"] = float(frequency)
    elif family.name == "fourier3d":
        frequencies = rng.sample(range(1, 13), 6)
        for key, frequency in zip(
            (f"{axis}{harmonic}_freq" for axis in "xyz" for harmonic in (1, 2)), frequencies
        ):
            sampled[key] = float(frequency)
    return sampled


def sample_initial(family: AttractorFamily, rng: random.Random, index: int) -> tuple[float, float, float]:
    if index == 0:
        return family.initial
    return tuple(float(v + rng.uniform(-0.08, 0.08)) for v in family.initial)
