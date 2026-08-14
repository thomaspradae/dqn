#!/usr/bin/env python3
"""Prototype terminal layout for the public poormans/join flow."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
MARK = ROOT / "poormans_shape_engine" / "selected_main" / "poormans_hpc_main_shade.txt"
ENGINE = ROOT / "dejong_3d_identity_engine"


def load_mark() -> list[str]:
    return MARK.read_text(encoding="utf-8").splitlines()


def pair(left: list[str], right: list[str], gap: int = 4) -> list[str]:
    width = max(len(line) for line in left)
    rows = max(len(left), len(right))
    output = []
    for index in range(rows):
        left_line = left[index] if index < len(left) else ""
        right_line = right[index] if index < len(right) else ""
        output.append(f"{left_line.ljust(width)}{' ' * gap}{right_line}")
    return output


def divider(width: int = 92) -> str:
    return "─" * width


def animated_mark(args: argparse.Namespace) -> list[str]:
    sys.path.insert(0, str(ENGINE))
    from dejong3d.animation import build_animation, frame_to_text, playback_terminal, profile_config
    from dejong3d.models import CanonicalSpec, RenderConfig, VariantSearchConfig

    profile = profile_config(args.profile)
    render_cfg = RenderConfig(term_width=args.term_width, term_height=args.term_height, supersample=args.supersample)
    render_cfg = replace(render_cfg, **profile["render"])
    search_cfg = VariantSearchConfig(
        master_seed=args.seed,
        candidates_per_code=args.candidates,
        modes=(args.mode,),
    )
    _, target, terms, specs = build_animation(
        code=args.code,
        render_cfg=render_cfg,
        search_cfg=search_cfg,
        canonical_spec=CanonicalSpec(),
        frames=args.frames,
        mode=args.mode,
        profile=args.profile,
    )
    if args.animate:
        playback_terminal(
            terms,
            render_cfg,
            specs=specs,
            fps=args.fps,
            style=args.style,
            axis_gizmo=args.axis_gizmo,
            hold=args.hold,
            intro_hold=args.intro_hold,
            alt_screen=not args.no_alt_screen,
        )
    print(
        f"identity target {target.code} | profile {args.profile} | mode {target.spec.mode} | "
        f"yaw {target.spec.yaw_deg:.1f} pitch {target.spec.pitch_deg:.1f}",
        file=sys.stderr,
    )
    return frame_to_text(terms[-1], render_cfg, args.style, specs[-1], args.axis_gizmo).splitlines()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prototype terminal layout for poormans/join.")
    parser.add_argument("--animate", action="store_true", help="play canonical-to-descendant terminal animation first")
    parser.add_argument("--code", default="join:uace-new-node", help="deterministic identity string for the descendant")
    parser.add_argument("--seed", default="poormans-hpc")
    parser.add_argument("--profile", choices=("clean", "hero", "loop"), default="clean")
    parser.add_argument(
        "--mode",
        choices=("surface", "contour_stack", "ribbon_skeleton", "shell_relief"),
        default="surface",
    )
    parser.add_argument("--style", choices=("shade", "binary", "ascii"), default="shade")
    parser.add_argument("--axis-gizmo", action="store_true", help="overlay a colored X/Y/Z rotation diagnostic")
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--candidates", type=int, default=1)
    parser.add_argument("--intro-hold", type=float, default=1.0)
    parser.add_argument("--hold", type=float, default=1.4)
    parser.add_argument("--term-width", type=int, default=60)
    parser.add_argument("--term-height", type=int, default=30)
    parser.add_argument("--supersample", type=int, default=4)
    parser.add_argument("--no-alt-screen", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mark = animated_mark(args) if args.animate else load_mark()
    info = [
        "poormans hpc",
        "public node join",
        "",
        "This machine is being prepared as a worker.",
        "No private inventory is written here.",
        "No Slurm or Prometheus config is applied here.",
        "",
        "bootstrap",
        "  ssh       openssh-server",
        "  service   ssh enabled",
        "  network   local IP discovery",
        "  output    controller adopt command",
        "",
        "controller-side next step",
        "  poormans adopt <user>@<ip>",
        "  poormans admit <alias>",
    ]

    print()
    print("\n".join(pair(mark, info)))
    print()
    print(divider())
    print("poormans-node-bootstrap")
    print(divider())
    print("[1/5] checking Ubuntu host")
    print("[2/5] installing OpenSSH server")
    print("[3/5] enabling ssh.service")
    print("[4/5] collecting local addresses")
    print("[5/5] ready for controller adoption")
    print()
    print("hostname")
    print("  uace-new-node")
    print()
    print("users")
    print("  t")
    print()
    print("addresses")
    print("  192.168.1.42")
    print("  fe80::demo")
    print()
    print("Run this from the controller:")
    print()
    print("  poormans adopt t@192.168.1.42")
    print()
    print("After adoption succeeds, continue on the controller with:")
    print()
    print("  poormans admit <alias>")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
