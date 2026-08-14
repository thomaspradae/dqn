#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import time

from dejong3d.models import RenderConfig
from native3d.export import export_gallery
from native3d.search import search_native_identities
from native3d.systems import FAMILIES


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Search genuine 3D attractors and their best canonical camera views.")
    p.add_argument("--out", type=Path, default=Path("examples/native3d_search"))
    p.add_argument("--seed", default="poormans-hpc-native3d")
    p.add_argument("--families", nargs="+", choices=tuple(FAMILIES), default=list(FAMILIES))
    p.add_argument("--per-family", type=int, default=3)
    p.add_argument("--points", type=int, default=14_000)
    p.add_argument("--cameras", type=int, default=48)
    p.add_argument("--keep", type=int, default=12)
    p.add_argument("--max-per-family", type=int, default=3)
    p.add_argument("--preview-points", type=int, default=6500)
    return p


def main() -> int:
    args = parser().parse_args()
    started = time.monotonic()
    render_cfg = RenderConfig(renderer="density", blur_passes=1)
    candidates, rejected = search_native_identities(
        seed=args.seed,
        families=tuple(args.families),
        per_family=args.per_family,
        samples=args.points,
        cameras=args.cameras,
        keep=args.keep,
        max_per_family=args.max_per_family,
        render_cfg=render_cfg,
    )
    path = export_gallery(args.out, candidates, rejected, render_cfg, max_points=args.preview_points)
    elapsed = time.monotonic() - started
    print(f"searched {len(args.families)} families; kept {len(candidates)}; rejected {len(rejected)}")
    for candidate in candidates:
        m = candidate.object_metrics
        print(
            f"#{candidate.id} {candidate.family:<10} total={candidate.total_score:.3f} "
            f"l2={m['lambda2_ratio']:.3f} l3={m['lambda3_ratio']:.3f} dim={m['box_dimension']:.3f}"
        )
    print(f"saved -> {path}")
    print(f"elapsed {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
