from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict

import numpy as np

from .families import Family
from .models import Candidate, RenderConfig, SearchConfig
from .render import render_terminal
from .score import REJECT_SCORE, score_terminal


def _seed_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def generate_pool(families: list[Family], render_cfg: RenderConfig, search_cfg: SearchConfig) -> list[Candidate]:
    candidates: list[Candidate] = []
    for family in families:
        family_rng = random.Random(_seed_int(f"{search_cfg.seed}:{family.name}"))
        iterations = max(2_000, int(round(family.default_iterations * search_cfg.iteration_scale)))
        for index in range(search_cfg.per_family):
            candidate_seed = f"{search_cfg.seed}:{family.name}:{index:05d}"
            # Fork the RNG per candidate, so a rejected/changed parameter branch doesn't shift every later candidate.
            rng = random.Random(_seed_int(candidate_seed) ^ family_rng.getrandbits(64))
            params = family.sample_params(rng)
            try:
                points = family.generate_points(params, iterations)
                term = render_terminal(points, render_cfg)
                raw_score, metrics = score_terminal(term, render_cfg)
            except (FloatingPointError, OverflowError, ValueError, ZeroDivisionError):
                continue
            if raw_score <= REJECT_SCORE / 2:
                continue
            candidates.append(
                Candidate(
                    family=family.name,
                    seed=candidate_seed,
                    params=params,
                    term=term,
                    raw_score=raw_score,
                    metrics=metrics,
                    metadata={"iterations": iterations},
                )
            )
    return candidates


def normalize_family_scores(candidates: list[Candidate], family_balance: float) -> None:
    groups: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.family].append(candidate)

    for group in groups.values():
        raw = np.array([c.raw_score for c in group], dtype=np.float64)
        median = float(np.median(raw))
        mad = float(np.median(np.abs(raw - median)))
        robust_sd = max(1.4826 * mad, float(np.std(raw)), 0.15)
        order = np.argsort(raw)
        percentiles = np.empty(len(raw), dtype=np.float64)
        if len(raw) == 1:
            percentiles[:] = 1.0
        else:
            percentiles[order] = np.linspace(0.0, 1.0, len(raw))

        for i, candidate in enumerate(group):
            candidate.family_z = float(np.clip((candidate.raw_score - median) / robust_sd, -3.0, 3.0))
            candidate.family_percentile = float(percentiles[i])
            # Raw score remains the dominant quality signal. Family-relative score prevents one family
            # from monopolizing the gallery because its natural score distribution is shifted upward.
            candidate.selection_score = float(
                candidate.raw_score
                + family_balance * (0.72 * candidate.family_z + 0.55 * candidate.family_percentile)
            )


def bitmap_similarity(a: Candidate, b: Candidate) -> float:
    va = a.term.ravel().astype(np.float64)
    vb = b.term.ravel().astype(np.float64)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    cosine = float(np.dot(va, vb) / denom) if denom > 1e-12 else 0.0

    ma = va >= 0.22
    mb = vb >= 0.22
    union = int(np.count_nonzero(ma | mb))
    jaccard = float(np.count_nonzero(ma & mb) / union) if union else 0.0
    return 0.68 * cosine + 0.32 * jaccard


def _can_add(candidate: Candidate, selected: list[Candidate], threshold: float, counts: Counter[str], max_per_family: int) -> bool:
    if max_per_family > 0 and counts[candidate.family] >= max_per_family:
        return False
    return all(bitmap_similarity(candidate, existing) <= threshold for existing in selected)


def diversity_select(candidates: list[Candidate], families: list[Family], cfg: SearchConfig) -> list[Candidate]:
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda c: (c.selection_score, c.raw_score), reverse=True)
    selected: list[Candidate] = []
    counts: Counter[str] = Counter()

    # Seed the gallery with one strong result per family when possible, but still obey diversity.
    if cfg.seed_each_family and cfg.count >= len(families):
        for family in families:
            options = [c for c in ordered if c.family == family.name]
            for candidate in options:
                if _can_add(candidate, selected, min(cfg.max_similarity + 0.035, 0.97), counts, cfg.max_per_family):
                    selected.append(candidate)
                    counts[candidate.family] += 1
                    break

    # Start strict, then relax only if strict diversity leaves empty slots.
    thresholds = [cfg.max_similarity, min(0.94, cfg.max_similarity + 0.035), 0.965, 0.985, 1.001]
    for threshold in thresholds:
        for candidate in ordered:
            if candidate in selected:
                continue
            if len(selected) >= cfg.count:
                break
            if _can_add(candidate, selected, threshold, counts, cfg.max_per_family):
                selected.append(candidate)
                counts[candidate.family] += 1
        if len(selected) >= cfg.count:
            break

    # If a hard family cap prevented filling the requested count, fill remaining slots by quality.
    if len(selected) < cfg.count:
        for candidate in ordered:
            if candidate not in selected:
                selected.append(candidate)
                if len(selected) >= cfg.count:
                    break

    selected.sort(key=lambda c: (c.selection_score, c.raw_score), reverse=True)
    for rank, candidate in enumerate(selected[: cfg.count], start=1):
        candidate.rank = rank
        candidate.id = f"{rank:03d}"
    return selected[: cfg.count]


def search_marks(families: list[Family], render_cfg: RenderConfig, search_cfg: SearchConfig) -> tuple[list[Candidate], list[Candidate]]:
    pool = generate_pool(families, render_cfg, search_cfg)
    normalize_family_scores(pool, search_cfg.family_balance)
    selected = diversity_select(pool, families, search_cfg)
    return selected, pool
