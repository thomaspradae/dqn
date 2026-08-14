from __future__ import annotations

from typing import Iterable

from .canonical import generate_dejong_points, normalize_points, density_field, sample_density, canonical_signature
from .geometry import canonical_front_spec, build_geometry, rng_for_code, sample_variant_spec
from .models import CanonicalSpec, RenderConfig, VariantRecord, VariantSearchConfig
from .render import render_terminal, render_terminal_lit
from .scoring import binary_metrics, expressive_score, identity_score


def default_codes(master_seed: str, count: int) -> list[str]:
    return [f"{master_seed}:variant:{i:04d}" for i in range(1, count + 1)]


def build_canonical_assets(render_cfg: RenderConfig, spec: CanonicalSpec | None = None) -> dict[str, object]:
    spec = spec or CanonicalSpec()
    raw_points = generate_dejong_points(spec)
    base_points = normalize_points(raw_points)
    field = density_field(raw_points, render_cfg.high_width, render_cfg.high_height, blur_passes=render_cfg.blur_passes)
    densities = sample_density(field, base_points)
    canonical_term = render_terminal(base_points, None, render_cfg)
    return {
        "spec": spec,
        "base_points": base_points,
        "field": field,
        "densities": densities,
        "canonical_term": canonical_term,
        "signature": canonical_signature(base_points, render_cfg),
    }


def render_variant(spec, base_points, densities, render_cfg: RenderConfig):
    cloud3d, geom_weights = build_geometry(base_points, densities, spec)
    from .geometry import transform_and_project_with_depth

    pts2, proj_weights, depths = transform_and_project_with_depth(cloud3d, spec)
    if len(proj_weights) != len(cloud3d):
        raise AssertionError("projected weights must match cloud size")
    total_weights = proj_weights * geom_weights
    if render_cfg.renderer == "density":
        term = render_terminal(pts2, total_weights, render_cfg)
    else:
        term = render_terminal_lit(pts2, depths, total_weights, render_cfg)
    return term


def _view_energy(spec) -> float:
    camera = min((abs(spec.yaw_deg) / 30.0 + abs(spec.pitch_deg) / 20.0 + abs(spec.roll_deg) / 12.0) / 3.0, 1.0)
    depth = min(spec.depth / 0.72, 1.0)
    persp = spec.perspective
    return 0.45 * camera + 0.30 * depth + 0.25 * persp


def choose_best_variant(
    code: str,
    assets: dict[str, object],
    render_cfg: RenderConfig,
    search_cfg: VariantSearchConfig,
) -> VariantRecord:
    base_points = assets["base_points"]
    densities = assets["densities"]
    canonical_term = assets["canonical_term"]

    if code == "canonical":
        spec = canonical_front_spec(code)
        term = render_variant(spec, base_points, densities, render_cfg)
        metrics = binary_metrics(term, render_cfg.ink_threshold)
        ident = identity_score(term, canonical_term, render_cfg.ink_threshold)
        expr = expressive_score(metrics, _view_energy(spec))
        return VariantRecord(
            code=code,
            spec=spec,
            term=term,
            raw_score=ident + search_cfg.expressive_weight * expr,
            identity_score=ident,
            expressive_score=expr,
            metrics=metrics,
            metadata={"signature": assets["signature"], "canonical": True, "mode": spec.mode},
        )

    rng = rng_for_code(f"{search_cfg.master_seed}|{code}")
    best: VariantRecord | None = None
    for idx in range(search_cfg.candidates_per_code):
        spec = sample_variant_spec(rng, code, idx, search_cfg.modes)
        term = render_variant(spec, base_points, densities, render_cfg)
        metrics = binary_metrics(term, render_cfg.ink_threshold)
        ident = identity_score(term, canonical_term, render_cfg.ink_threshold)
        expr = expressive_score(metrics, _view_energy(spec))
        score = ident + search_cfg.expressive_weight * expr
        if metrics["coverage"] < 0.06 or metrics["coverage"] > 0.55:
            score -= 1.6
        if ident < search_cfg.similarity_floor:
            score -= (search_cfg.similarity_floor - ident) * 2.0
        record = VariantRecord(
            code=code,
            spec=spec,
            term=term,
            raw_score=score,
            identity_score=ident,
            expressive_score=expr,
            metrics=metrics,
            metadata={
                "signature": assets["signature"],
                "canonical": False,
                "search_index": idx,
                "view_energy": _view_energy(spec),
                "mode": spec.mode,
            },
        )
        if best is None or record.raw_score > best.raw_score:
            best = record
    assert best is not None
    return best


def generate_variants(
    codes: Iterable[str] | None = None,
    render_cfg: RenderConfig | None = None,
    search_cfg: VariantSearchConfig | None = None,
    canonical_spec: CanonicalSpec | None = None,
) -> tuple[dict[str, object], list[VariantRecord]]:
    render_cfg = render_cfg or RenderConfig()
    search_cfg = search_cfg or VariantSearchConfig()
    assets = build_canonical_assets(render_cfg, canonical_spec)
    if codes is None:
        codes = default_codes(search_cfg.master_seed, search_cfg.count)
    code_list = ["canonical", *list(codes)]
    out = [choose_best_variant(code, assets, render_cfg, search_cfg) for code in code_list]
    for rank, record in enumerate(out, start=1):
        record.rank = rank
        record.id = f"{rank:03d}"
    return assets, out
