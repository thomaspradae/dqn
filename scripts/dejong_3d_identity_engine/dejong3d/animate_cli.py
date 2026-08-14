from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

from dataclasses import replace

from .animation import EXAMPLES, PROFILES, build_animation, export_frames, playback_terminal, profile_config
from .geometry import MODES
from .models import CanonicalSpec, RenderConfig, VariantSearchConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Animate canonical De Jong logo into a deterministic 3D descendant.")
    p.add_argument("code", nargs="?", default="node:ofi1", help="node/hash/id/string that defines the final descendant")
    p.add_argument("--profile", choices=tuple(PROFILES), default="clean", help="animation/rendering preset")
    p.add_argument("--example", choices=tuple(EXAMPLES), default=None, help="load one polished showcase preset")
    p.add_argument("--mode", choices=MODES, default=None, help="single continuous 3D derivation mode")
    p.add_argument("--seed", default="poormans-hpc")
    p.add_argument("--frames", type=int, default=None)
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--seconds", type=float, default=None, help="override frame count from fps*seconds")
    p.add_argument("--candidates", type=int, default=None, help="deterministic target-search candidates")
    p.add_argument("--style", choices=("shade", "binary", "ascii"), default="shade")
    p.add_argument("--axis-gizmo", action="store_true", help="overlay a colored X/Y/Z rotation diagnostic")
    p.add_argument("--term-width", type=int, default=60)
    p.add_argument("--term-height", type=int, default=30)
    p.add_argument("--supersample", type=int, default=4)
    p.add_argument("--hold", type=float, default=None)
    p.add_argument("--loop", type=int, default=1)
    p.add_argument("--pingpong", action="store_true")
    p.add_argument("--no-alt-screen", action="store_true")
    p.add_argument("--no-auto-fit", action="store_true", help="do not shrink the render to fit the current terminal")
    p.add_argument("--intro-hold", type=float, default=None, help="seconds to hold the exact canonical mark before motion")
    p.add_argument("--export", type=Path, default=None, help="write frame txt files + self-playing HTML preview")
    p.add_argument("--no-play", action="store_true", help="generate/export but do not play in terminal")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code = args.code
    profile_name = args.profile
    mode = args.mode
    if args.example is not None:
        example = EXAMPLES[args.example]
        profile_name = str(example["profile"])
        mode = str(example["mode"])
        code = str(example["code"])

    profile = profile_config(profile_name)
    frames = int(args.frames or profile["frames"])
    fps = float(args.fps or profile["fps"])
    candidates = int(args.candidates or profile["candidates"])
    hold = float(args.hold if args.hold is not None else profile["hold"])
    intro_hold = float(args.intro_hold if args.intro_hold is not None else profile["intro_hold"])
    mode = mode or str(profile["mode"])

    if args.seconds is not None:
        frames = max(2, round(args.seconds * fps) + 1)
    term_width, term_height = args.term_width, args.term_height
    if not args.no_play and not args.no_auto_fit and sys.stdout.isatty():
        size = shutil.get_terminal_size(fallback=(80, 24))
        term_width = min(term_width, max(20, size.columns - 2))
        term_height = min(term_height, max(10, size.lines - 2))
    render_cfg = RenderConfig(term_width=term_width, term_height=term_height, supersample=args.supersample)
    render_cfg = replace(render_cfg, **profile["render"])
    search_cfg = VariantSearchConfig(master_seed=args.seed, candidates_per_code=candidates, modes=(mode,))
    assets, target, terms, specs = build_animation(
        code=code,
        render_cfg=render_cfg,
        search_cfg=search_cfg,
        canonical_spec=CanonicalSpec(),
        frames=frames,
        mode=mode,
        profile=profile_name,
    )
    print(
        f"target code={target.code} profile={profile_name} mode={target.spec.mode} "
        f"yaw={target.spec.yaw_deg:.1f} pitch={target.spec.pitch_deg:.1f} roll={target.spec.roll_deg:.1f} "
        f"identity={target.identity_score:.3f} expressive={target.expressive_score:.3f}",
        flush=True,
    )
    if args.export is not None:
        path = export_frames(
            args.export,
            terms,
            specs,
            target,
            render_cfg,
            fps,
            args.style,
            profile_name,
            args.axis_gizmo,
            intro_hold,
            hold,
        )
        print(f"saved animation preview -> {path}", flush=True)
    if not args.no_play:
        playback_terminal(
            terms,
            render_cfg,
            specs=specs,
            fps=fps,
            style=args.style,
            axis_gizmo=args.axis_gizmo,
            hold=hold,
            intro_hold=intro_hold,
            loop=args.loop,
            pingpong=args.pingpong,
            alt_screen=not args.no_alt_screen,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
