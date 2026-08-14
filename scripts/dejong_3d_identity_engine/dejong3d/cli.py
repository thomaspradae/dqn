from __future__ import annotations

import argparse
from pathlib import Path

from .export import export_gallery
from .geometry import MODES
from .models import CanonicalSpec, RenderConfig, VariantSearchConfig
from .search import default_codes, generate_variants


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate deterministic 3D identity variants from a canonical De Jong logo.")
    p.add_argument("--out", type=Path, default=Path("/tmp/dejong3d_gallery"))
    p.add_argument("--seed", default="poormans-hpc")
    p.add_argument("--count", type=int, default=16)
    p.add_argument("--codes", nargs="*", default=None, help="explicit variant codes; otherwise uses seed:variant:0001...")
    p.add_argument("--candidates-per-code", type=int, default=28)
    p.add_argument("--term-width", type=int, default=60)
    p.add_argument("--term-height", type=int, default=30)
    p.add_argument("--supersample", type=int, default=5)
    p.add_argument("--similarity-floor", type=float, default=0.34)
    p.add_argument("--expressive-weight", type=float, default=0.38)
    p.add_argument("--modes", nargs="*", choices=MODES, default=list(MODES), help="subset of 3D derivation modes to search")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    render_cfg = RenderConfig(term_width=args.term_width, term_height=args.term_height, supersample=args.supersample)
    search_cfg = VariantSearchConfig(
        master_seed=args.seed,
        count=args.count,
        candidates_per_code=args.candidates_per_code,
        similarity_floor=args.similarity_floor,
        expressive_weight=args.expressive_weight,
        modes=tuple(args.modes),
    )
    codes = args.codes if args.codes else default_codes(args.seed, args.count)
    canonical_spec = CanonicalSpec()
    assets, variants = generate_variants(codes=codes, render_cfg=render_cfg, search_cfg=search_cfg, canonical_spec=canonical_spec)
    out = export_gallery(args.out, assets, variants, render_cfg, search_cfg, canonical_spec)
    print(f"saved gallery -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
