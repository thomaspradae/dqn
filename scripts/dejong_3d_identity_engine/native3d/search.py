from __future__ import annotations

from dataclasses import asdict
import hashlib
import math
import random

import numpy as np

from dejong3d.models import CanonicalSpec, RenderConfig
from dejong3d.render import render_terminal, rotation_matrix_xyz
from dejong3d.scoring import binary_metrics
from dejong3d.search import build_canonical_assets

from .integrate import IntegrationError, cloud_metrics, generate_orbit, normalize_cloud, object_quality
from .models import CameraPose, NativeCandidate
from .systems import FAMILIES, sample_initial, sample_parameters


def _rng(seed: str) -> random.Random:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


def rotate_cloud(points: np.ndarray, pose: CameraPose) -> np.ndarray:
    rotation = rotation_matrix_xyz(pose.pitch_deg, pose.yaw_deg, pose.roll_deg)
    return points @ rotation.T


def project_cloud(points: np.ndarray, pose: CameraPose) -> tuple[np.ndarray, np.ndarray]:
    rotated = rotate_cloud(points, pose)
    if pose.perspective <= 0.0:
        return rotated[:, :2], rotated[:, 2]
    camera_z = 3.2
    scale = 3.0 / np.maximum(camera_z - rotated[:, 2], 0.5)
    perspective_xy = rotated[:, :2] * scale[:, None]
    xy = (1.0 - pose.perspective) * rotated[:, :2] + pose.perspective * perspective_xy
    return xy, rotated[:, 2]


def _camera_poses(seed: str, count: int) -> list[CameraPose]:
    rng = _rng(seed)
    poses = [
        CameraPose(0.0, 0.0, 0.0),
        CameraPose(0.0, 90.0, 0.0),
        CameraPose(90.0, 0.0, 0.0),
    ]
    for _ in range(max(0, count - len(poses))):
        poses.append(
            CameraPose(
                pitch_deg=math.degrees(math.asin(rng.uniform(-1.0, 1.0))),
                yaw_deg=rng.uniform(-180.0, 180.0),
                roll_deg=rng.uniform(-180.0, 180.0),
                perspective=rng.uniform(0.0, 0.18),
            )
        )
    return poses


def _depth_complexity(points: np.ndarray, pose: CameraPose, width: int = 32, height: int = 20) -> float:
    xy, depth = project_cloud(points, pose)
    mins = xy.min(axis=0)
    spans = np.maximum(xy.max(axis=0) - mins, 1e-9)
    cells = np.floor((xy - mins) / spans * np.array([width - 1, height - 1])).astype(np.int64)
    dmin, dspan = float(depth.min()), max(float(depth.max() - depth.min()), 1e-9)
    dbin = np.clip(np.floor((depth - dmin) / dspan * 7.0).astype(np.int64), 0, 7)
    pixel = cells[:, 0] + width * cells[:, 1]
    occupied_pixels = max(int(np.unique(pixel).size), 1)
    layers = np.unique(pixel * 8 + dbin).size / occupied_pixels
    return float(min(layers / 4.0, 1.5))


def _target_similarity(metrics: dict[str, float], reference: dict[str, float]) -> float:
    scales = {
        "coverage": 0.20,
        "hard_coverage": 0.12,
        "bbox_fill": 0.30,
        "perimeter_ratio": 1.8,
        "entropy": 2.0,
    }
    error = 0.0
    for key, scale in scales.items():
        error += min(abs(metrics[key] - reference[key]) / scale, 2.0)
    return math.exp(-error / len(scales))


def projection_quality(
    term: np.ndarray,
    depth_complexity: float,
    reference_metrics: dict[str, float],
    threshold: float,
) -> tuple[float, dict[str, float]]:
    metrics = binary_metrics(term, threshold)
    coverage = metrics["coverage"]
    coverage_score = math.exp(-((coverage - 0.25) / 0.16) ** 2)
    fill_score = math.exp(-((metrics["bbox_fill"] - 0.42) / 0.30) ** 2)
    component_score = math.exp(-((min(metrics["components"], 16.0) - 4.0) / 6.0) ** 2)
    holes_score = min(metrics["holes"], 7.0) / 7.0
    perimeter_score = min(metrics["perimeter_ratio"] / 2.8, 1.3)
    entropy_score = min(metrics["entropy"] / 3.2, 1.2)
    reference_score = _target_similarity(metrics, reference_metrics)
    penalties = 0.8 * metrics["edge_touches"] + 0.6 * metrics["speckle"]
    if coverage < 0.06 or coverage > 0.62:
        penalties += 2.0
    score = (
        1.15 * coverage_score
        + 0.65 * fill_score
        + 0.35 * component_score
        + 0.52 * holes_score
        + 0.62 * perimeter_score
        + 0.42 * entropy_score
        + 0.58 * reference_score
        + 0.48 * depth_complexity
        - penalties
    )
    metrics = dict(metrics)
    metrics["depth_complexity"] = depth_complexity
    metrics["dejong_spirit"] = reference_score
    return float(score), metrics


def _target_pose(canonical: CameraPose, seed: str) -> CameraPose:
    rng = _rng(f"target|{seed}")
    yaw_delta = rng.choice((-1.0, 1.0)) * rng.uniform(48.0, 82.0)
    pitch_delta = rng.choice((-1.0, 1.0)) * rng.uniform(18.0, 38.0)
    roll_delta = rng.uniform(-18.0, 18.0)
    return CameraPose(
        pitch_deg=canonical.pitch_deg + pitch_delta,
        yaw_deg=canonical.yaw_deg + yaw_delta,
        roll_deg=canonical.roll_deg + roll_delta,
        perspective=min(0.20, canonical.perspective + 0.06),
    )


def search_native_identities(
    seed: str = "poormans-hpc-native3d",
    families: tuple[str, ...] | None = None,
    per_family: int = 3,
    samples: int = 14_000,
    cameras: int = 48,
    keep: int = 12,
    max_per_family: int = 3,
    render_cfg: RenderConfig | None = None,
) -> tuple[list[NativeCandidate], list[dict[str, object]]]:
    render_cfg = render_cfg or RenderConfig(renderer="density", blur_passes=1)
    family_names = families or tuple(FAMILIES)
    canonical = build_canonical_assets(render_cfg, CanonicalSpec())
    reference_metrics = binary_metrics(canonical["canonical_term"], render_cfg.ink_threshold)
    candidates: list[NativeCandidate] = []
    rejected: list[dict[str, object]] = []

    for family_name in family_names:
        family = FAMILIES[family_name]
        family_rng = _rng(f"{seed}|{family_name}")
        for index in range(max(1, per_family)):
            candidate_seed = f"{seed}:{family_name}:{index:04d}"
            params = sample_parameters(family, family_rng, index)
            initial = sample_initial(family, family_rng, index)
            try:
                points = normalize_cloud(generate_orbit(family, params, initial, samples=samples))
            except IntegrationError as exc:
                rejected.append({"seed": candidate_seed, "family": family_name, "reason": str(exc)})
                continue

            metrics3d = cloud_metrics(points)
            object_score, accepted = object_quality(metrics3d, family.topology)
            if not accepted:
                rejected.append(
                    {
                        "seed": candidate_seed,
                        "family": family_name,
                        "reason": "insufficiently volumetric",
                        "object_metrics": metrics3d,
                    }
                )
                continue

            best: tuple[float, CameraPose, np.ndarray, dict[str, float]] | None = None
            for pose in _camera_poses(f"camera|{candidate_seed}", cameras):
                xy, _ = project_cloud(points, pose)
                term = render_terminal(xy, None, render_cfg)
                depth_score = _depth_complexity(points[::2], pose)
                score, projection_metrics = projection_quality(
                    term, depth_score, reference_metrics, render_cfg.ink_threshold
                )
                if best is None or score > best[0]:
                    best = (score, pose, term, projection_metrics)
            assert best is not None
            projection_score, canonical_pose, term, projection_metrics = best
            total_score = projection_score + 0.42 * object_score
            candidates.append(
                NativeCandidate(
                    id="",
                    family=family_name,
                    seed=candidate_seed,
                    params=params,
                    initial=initial,
                    points=points,
                    object_metrics=metrics3d,
                    object_score=object_score,
                    canonical_pose=canonical_pose,
                    terminal=term,
                    projection_metrics=projection_metrics,
                    projection_score=projection_score,
                    total_score=total_score,
                    target_pose=_target_pose(canonical_pose, candidate_seed),
                    metadata={
                        "dt": family.dt,
                        "burn_steps": family.burn_steps,
                        "search_cameras": cameras,
                        "topology": family.topology,
                        "generator": "parametric_or_map" if family.generator is not None else "rk4",
                    },
                )
            )

    candidates.sort(key=lambda item: item.total_score, reverse=True)
    selected: list[NativeCandidate] = []
    family_counts: dict[str, int] = {}
    for candidate in candidates:
        if family_counts.get(candidate.family, 0) >= max(1, max_per_family):
            continue
        candidate.id = f"{len(selected) + 1:03d}"
        selected.append(candidate)
        family_counts[candidate.family] = family_counts.get(candidate.family, 0) + 1
        if len(selected) >= keep:
            break
    return selected, rejected
