from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from .export import export_gallery
from .families import FAMILY_REGISTRY, get_families
from .models import RenderConfig, SearchConfig
from .search import search_marks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search mathematical shape families for compact terminal marks.")
    parser.add_argument("--out", type=Path, default=Path("./poormans_mark_gallery"))
    parser.add_argument("--seed", default="poormans-hpc")
    parser.add_argument("--families", default="all", help=f"comma-separated or all: {','.join(FAMILY_REGISTRY)}")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--per-family", type=int, default=64)
    parser.add_argument("--width", type=int, default=60)
    parser.add_argument("--height", type=int, default=30)
    parser.add_argument("--supersample", type=int, default=4)
    parser.add_argument("--cell-aspect", type=float, default=0.55)
    parser.add_argument("--iteration-scale", type=float, default=1.0, help="0.35 fast, 1 normal, 2 dense")
    parser.add_argument("--family-balance", type=float, default=0.35)
    parser.add_argument("--max-similarity", type=float, default=0.89)
    parser.add_argument("--max-per-family", type=int, default=0, help="0 disables the cap")
    parser.add_argument("--no-family-seeding", action="store_true")
    parser.add_argument("--no-pca", action="store_true")
    parser.add_argument("--blur", type=int, default=1)
    parser.add_argument("--ink-threshold", type=float, default=0.22)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    names = list(FAMILY_REGISTRY) if args.families == "all" else [x.strip() for x in args.families.split(",") if x.strip()]
    families = get_families(names)

    render_cfg = RenderConfig(
        term_width=max(12, args.width),
        term_height=max(8, args.height),
        supersample=max(2, args.supersample),
        cell_aspect=max(0.2, args.cell_aspect),
        pca_align=not args.no_pca,
        blur_passes=max(0, args.blur),
        ink_threshold=min(0.9, max(0.02, args.ink_threshold)),
    )
    search_cfg = SearchConfig(
        seed=args.seed,
        per_family=max(1, args.per_family),
        count=max(1, args.count),
        iteration_scale=max(0.1, args.iteration_scale),
        family_balance=max(0.0, args.family_balance),
        max_similarity=min(0.999, max(0.1, args.max_similarity)),
        max_per_family=max(0, args.max_per_family),
        seed_each_family=not args.no_family_seeding,
    )

    selected, pool = search_marks(families, render_cfg, search_cfg)
    if not selected:
        print("No candidates survived the quality gates. Increase --per-family or lower --ink-threshold.", file=sys.stderr)
        return 2

    html_path = export_gallery(args.out, selected, pool, render_cfg, search_cfg)
    accepted = Counter(c.family for c in pool)
    chosen = Counter(c.family for c in selected)
    print(f"accepted {len(pool)} candidates: {dict(sorted(accepted.items()))}")
    print(f"selected {len(selected)} candidates: {dict(sorted(chosen.items()))}")
    print(f"gallery: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
