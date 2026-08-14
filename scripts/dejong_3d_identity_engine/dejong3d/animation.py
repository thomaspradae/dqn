from __future__ import annotations

from dataclasses import replace
import html
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from .models import CanonicalSpec, RenderConfig, VariantRecord, VariantSearchConfig, VariantSpec
from .render import ASCII_RAMP, SHADE_RAMP, rotation_matrix_xyz, terminal_to_binary, terminal_to_text, text_block
from .search import build_canonical_assets, choose_best_variant, render_variant


PROFILES: dict[str, dict[str, object]] = {
    "clean": {
        "mode": "surface",
        "frames": 72,
        "fps": 20.0,
        "intro_hold": 1.00,
        "hold": 1.40,
        "candidates": 24,
        "pose_scale": 2.60,
        "depth_scale": 1.14,
        "perspective_scale": 1.22,
        "schedule": {
            "canonical_until": 0.00,
            "renderer": (0.00, 0.24),
            "depth": (0.00, 0.34),
            "camera": (0.14, 0.86),
            "detail": (0.36, 0.92),
        },
        "render": {
            "ambient": 0.36,
            "diffuse": 0.74,
            "specular": 0.07,
            "normal_strength": 3.8,
            "depth_fog": 0.26,
        },
    },
    "hero": {
        "mode": "contour_stack",
        "frames": 72,
        "fps": 24.0,
        "intro_hold": 0.36,
        "hold": 0.80,
        "candidates": 36,
        "pose_scale": 2.40,
        "depth_scale": 1.18,
        "perspective_scale": 1.26,
        "schedule": {
            "canonical_until": 0.00,
            "renderer": (0.00, 0.22),
            "depth": (0.00, 0.34),
            "camera": (0.14, 0.88),
            "detail": (0.36, 0.94),
        },
        "render": {
            "ambient": 0.30,
            "diffuse": 0.88,
            "specular": 0.14,
            "normal_strength": 5.0,
            "depth_fog": 0.38,
        },
    },
    "loop": {
        "mode": "surface",
        "frames": 56,
        "fps": 20.0,
        "intro_hold": 0.20,
        "hold": 0.20,
        "candidates": 18,
        "pose_scale": 0.85,
        "depth_scale": 0.92,
        "perspective_scale": 0.90,
        "schedule": {
            "canonical_until": 0.00,
            "renderer": (0.00, 0.20),
            "depth": (0.00, 0.30),
            "camera": (0.14, 0.86),
            "detail": (0.34, 0.88),
        },
        "render": {
            "ambient": 0.38,
            "diffuse": 0.68,
            "specular": 0.05,
            "normal_strength": 3.4,
            "depth_fog": 0.22,
        },
    },
}


EXAMPLES: dict[str, dict[str, object]] = {
    "surface_clean": {"profile": "clean", "mode": "surface", "code": "node:ofi1"},
    "contour_stack_hero": {"profile": "hero", "mode": "contour_stack", "code": "node:ofi2"},
    "ribbon_skeleton_expressive": {"profile": "hero", "mode": "ribbon_skeleton", "code": "lease:vast-h100:demo"},
}


def smootherstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def surge_ease(t: float) -> float:
    """Sharper ease-in/ease-out with a visibly faster middle section."""
    t = max(0.0, min(1.0, t))
    k = 2.6
    return 0.5 + 0.5 * math.tanh(k * (2.0 * t - 1.0)) / math.tanh(k)


def ease_in_out_sine(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return -(math.cos(math.pi * t) - 1.0) / 2.0


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _shortest_angle_lerp(a: float, b: float, t: float) -> float:
    delta = (b - a + 180.0) % 360.0 - 180.0
    return a + delta * t


def _window(t: float, start: float, end: float) -> float:
    if end <= start:
        return 1.0 if t >= end else 0.0
    return smootherstep((t - start) / (end - start))


def profile_config(name: str) -> dict[str, object]:
    if name not in PROFILES:
        raise ValueError(f"unknown animation profile: {name}")
    return PROFILES[name]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _scale_target_pose(target: VariantRecord, assets: dict[str, object], render_cfg: RenderConfig, profile: dict[str, object], expressive_weight: float) -> VariantRecord:
    pose_scale = float(profile.get("pose_scale", 1.0))
    depth_scale = float(profile.get("depth_scale", 1.0))
    perspective_scale = float(profile.get("perspective_scale", 1.0))
    if pose_scale == 1.0 and depth_scale == 1.0 and perspective_scale == 1.0:
        return target

    spec = replace(
        target.spec,
        yaw_deg=_clamp(target.spec.yaw_deg * pose_scale, -76.0, 76.0),
        pitch_deg=_clamp(target.spec.pitch_deg * pose_scale, -46.0, 46.0),
        roll_deg=_clamp(target.spec.roll_deg * pose_scale, -32.0, 32.0),
        depth=_clamp(target.spec.depth * depth_scale, 0.0, 0.82),
        perspective=_clamp(target.spec.perspective * perspective_scale, 0.0, 0.88),
    )
    term = render_variant(spec, assets["base_points"], assets["densities"], render_cfg)

    from .scoring import binary_metrics, expressive_score, identity_score
    from .search import _view_energy

    metrics = binary_metrics(term, render_cfg.ink_threshold)
    ident = identity_score(term, assets["canonical_term"], render_cfg.ink_threshold)
    expr = expressive_score(metrics, _view_energy(spec))
    metadata = dict(target.metadata)
    metadata["pose_scaled"] = True
    metadata["pose_scale"] = pose_scale
    metadata["depth_scale"] = depth_scale
    metadata["perspective_scale"] = perspective_scale
    return replace(
        target,
        spec=spec,
        term=term,
        metrics=metrics,
        identity_score=ident,
        expressive_score=expr,
        raw_score=ident + expressive_weight * expr,
        metadata=metadata,
    )


def neutral_spec(target: VariantSpec) -> VariantSpec:
    """A flat front-facing state using the target's discrete topology.

    Discrete choices (mode/frequencies/layer count) are fixed from frame zero.
    Their amplitudes begin at zero, so they do not pop into view midway.
    """
    return replace(
        target,
        yaw_deg=0.0,
        pitch_deg=0.0,
        roll_deg=0.0,
        depth=0.0,
        gamma=1.0,
        thickness=0.0,
        ripple_x_amp=0.0,
        ripple_y_amp=0.0,
        ripple_r_amp=0.0,
        twist=0.0,
        perspective=0.0,
        focal=3.0,
        camera_z=3.0,
        search_index=target.search_index,
    )


def interpolate_spec(start: VariantSpec, end: VariantSpec, t: float, schedule: dict[str, object] | None = None) -> VariantSpec:
    """Interpolate only continuous state; discrete geometry remains the target's."""
    schedule = schedule or profile_config("clean")["schedule"]
    depth_start, depth_end = schedule["depth"]
    camera_start, camera_end = schedule["camera"]
    detail_start, detail_end = schedule["detail"]
    e = _window(t, float(depth_start), float(depth_end))
    camera_e = surge_ease((t - float(camera_start)) / max(float(camera_end) - float(camera_start), 1e-9))
    detail_e = _window(t, float(detail_start), float(detail_end))
    return replace(
        end,
        yaw_deg=_shortest_angle_lerp(start.yaw_deg, end.yaw_deg, camera_e),
        pitch_deg=_shortest_angle_lerp(start.pitch_deg, end.pitch_deg, camera_e),
        roll_deg=_shortest_angle_lerp(start.roll_deg, end.roll_deg, camera_e),
        depth=_lerp(start.depth, end.depth, e),
        gamma=_lerp(start.gamma, end.gamma, e),
        thickness=_lerp(start.thickness, end.thickness, e),
        ripple_x_amp=_lerp(start.ripple_x_amp, end.ripple_x_amp, detail_e),
        ripple_y_amp=_lerp(start.ripple_y_amp, end.ripple_y_amp, detail_e),
        ripple_r_amp=_lerp(start.ripple_r_amp, end.ripple_r_amp, detail_e),
        twist=_lerp(start.twist, end.twist, detail_e),
        perspective=_lerp(start.perspective, end.perspective, camera_e),
        focal=_lerp(start.focal, end.focal, camera_e),
        camera_z=_lerp(start.camera_z, end.camera_z, camera_e),
    )


def build_animation(
    code: str,
    render_cfg: RenderConfig | None = None,
    search_cfg: VariantSearchConfig | None = None,
    canonical_spec: CanonicalSpec | None = None,
    frames: int = 48,
    mode: str = "surface",
    profile: str = "clean",
) -> tuple[dict[str, object], VariantRecord, list[np.ndarray], list[VariantSpec]]:
    render_cfg = render_cfg or RenderConfig()
    base_search = search_cfg or VariantSearchConfig()
    profile_data = profile_config(profile)
    schedule = profile_data["schedule"]
    # Animation defaults to one geometry mode so there is no topology switch mid-motion.
    animation_search = replace(base_search, modes=(mode,))
    canonical_spec = canonical_spec or CanonicalSpec()
    assets = build_canonical_assets(render_cfg, canonical_spec)
    target = choose_best_variant(code, assets, render_cfg, animation_search)
    target = _scale_target_pose(target, assets, render_cfg, profile_data, animation_search.expressive_weight)
    start = neutral_spec(target.spec)

    frames = max(2, int(frames))
    terms: list[np.ndarray] = []
    specs: list[VariantSpec] = []
    for i in range(frames):
        t = i / (frames - 1)
        if i == 0:
            # Frame zero is bit-for-bit the canonical 2D renderer.
            term = np.array(assets["canonical_term"], copy=True)
            spec = start
        elif t <= float(schedule["canonical_until"]):
            term = np.array(assets["canonical_term"], copy=True)
            spec = start
        elif i == frames - 1:
            # Last frame is exactly the chosen deterministic descendant.
            term = np.array(target.term, copy=True)
            spec = target.spec
        else:
            spec = interpolate_spec(start, target.spec, t, schedule)
            rendered = render_variant(spec, assets["base_points"], assets["densities"], render_cfg)
            blend_start, blend_end = schedule.get("renderer", (0.0, 0.0))
            renderer_e = _window(t, float(blend_start), float(blend_end))
            term = (1.0 - renderer_e) * assets["canonical_term"] + renderer_e * rendered
        terms.append(term)
        specs.append(spec)
    return assets, target, terms, specs


AXIS_COLORS = {
    "x": "\x1b[1;31m",
    "y": "\x1b[1;32m",
    "z": "\x1b[1;34m",
    "o": "\x1b[1;37m",
}
ANSI_RESET = "\x1b[0m"


def _axis_line_cells(cx: int, cy: int, ex: float, ey: float) -> list[tuple[int, int]]:
    dx = ex - cx
    dy = ey - cy
    steps = max(1, int(round(max(abs(dx), abs(dy)))))
    cells = []
    for step in range(1, steps + 1):
        x = int(round(cx + dx * step / steps))
        y = int(round(cy + dy * step / steps))
        cells.append((x, y))
    return cells


def _colored(ch: str, axis: str, enabled: bool) -> str:
    if not enabled:
        return ch
    return f"{AXIS_COLORS[axis]}{ch}{ANSI_RESET}"


def axis_gizmo_endpoints(spec: VariantSpec, render_cfg: RenderConfig) -> dict[str, tuple[float, float]]:
    width = render_cfg.term_width
    height = render_cfg.term_height
    cx = width // 2
    cy = height // 2
    # Keep the diagnostic compact. Horizontal columns are compensated only
    # enough to read as roughly square in common terminal/browser fonts.
    radius = max(2.8, min(3.8, min(width * render_cfg.cell_aspect, height) * 0.13))
    x_comp = 1.0 / max(math.sqrt(render_cfg.cell_aspect), 1e-6)
    basis = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
    }
    rotation = rotation_matrix_xyz(spec.pitch_deg, spec.yaw_deg, spec.roll_deg)
    endpoints: dict[str, tuple[float, float]] = {"o": (float(cx), float(cy))}
    for axis, vec in basis.items():
        rotated = vec @ rotation.T
        endpoints[axis] = (cx + rotated[0] * radius * x_comp, cy - rotated[1] * radius)
    return endpoints


def overlay_axis_gizmo(
    lines: list[str],
    spec: VariantSpec,
    render_cfg: RenderConfig,
    color: bool = True,
) -> list[str]:
    """Overlay a small colored camera-axis diagnostic on rendered text.

    X is red, Y is green, and Z is blue. At the canonical front-on orientation
    the Z axis points out of the screen, so it collapses near the origin; as the
    object rotates, the blue axis should visibly separate from the center.
    """
    width = render_cfg.term_width
    height = len(lines)
    if width < 12 or height < 8:
        return lines

    cells: list[list[str]] = [list(line.ljust(width)) for line in lines]
    endpoints = axis_gizmo_endpoints(spec, render_cfg)
    cx = int(round(endpoints["o"][0]))
    cy = int(round(endpoints["o"][1]))
    labels = {"x": "X", "y": "Y", "z": "Z"}
    line_chars = {"x": "x", "y": "y", "z": "z"}

    for axis in ("x", "y", "z"):
        ex, ey = endpoints[axis]
        cells_on_axis = _axis_line_cells(cx, cy, ex, ey)
        for x, y in cells_on_axis[:-1]:
            if 0 <= x < width and 0 <= y < height:
                cells[y][x] = _colored(line_chars[axis], axis, color)
        if cells_on_axis:
            x, y = cells_on_axis[-1]
            if 0 <= x < width and 0 <= y < height:
                cells[y][x] = _colored(labels[axis], axis, color)

    cells[cy][cx] = _colored("+", "o", color)
    return ["".join(row).rstrip() for row in cells]


def frame_to_text(
    term: np.ndarray,
    render_cfg: RenderConfig,
    style: str = "shade",
    spec: VariantSpec | None = None,
    axis_gizmo: bool = False,
    axis_color: bool = True,
) -> str:
    if style == "binary":
        lines = terminal_to_binary(term, render_cfg.ink_threshold)
    elif style == "ascii":
        lines = terminal_to_text(term, ASCII_RAMP, render_cfg.gamma)
    else:
        lines = terminal_to_text(term, SHADE_RAMP, render_cfg.gamma)
    if axis_gizmo and spec is not None:
        lines = overlay_axis_gizmo(lines, spec, render_cfg, axis_color)
    return text_block(lines, render_cfg.term_width)


def playback_terminal(
    terms: list[np.ndarray],
    render_cfg: RenderConfig,
    specs: list[VariantSpec] | None = None,
    fps: float = 24.0,
    style: str = "shade",
    axis_gizmo: bool = False,
    hold: float = 0.8,
    intro_hold: float = 0.30,
    loop: int = 1,
    pingpong: bool = False,
    alt_screen: bool = True,
) -> None:
    fps = max(1.0, float(fps))
    interval = 1.0 / fps
    spec_list = specs if specs is not None else [None] * len(terms)
    frames = list(zip(terms, spec_list))
    if pingpong and len(frames) > 2:
        frames = frames + frames[-2:0:-1]
    enter = "\x1b[?1049h" if alt_screen else ""
    leave = "\x1b[?1049l" if alt_screen else ""
    sys.stdout.write(enter + "\x1b[?25l\x1b[2J\x1b[H")
    sys.stdout.flush()
    try:
        if intro_hold > 0 and frames:
            term, spec = frames[0]
            sys.stdout.write("\x1b[H" + frame_to_text(term, render_cfg, style, spec, axis_gizmo))
            sys.stdout.flush()
            time.sleep(intro_hold)
        loops = max(1, int(loop))
        for _ in range(loops):
            deadline = time.monotonic()
            for term, spec in frames:
                payload = frame_to_text(term, render_cfg, style, spec, axis_gizmo)
                # One write per frame prevents line-by-line tearing/flicker.
                sys.stdout.write("\x1b[H" + payload)
                sys.stdout.flush()
                deadline += interval
                delay = deadline - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
        if hold > 0:
            time.sleep(hold)
    finally:
        sys.stdout.write("\x1b[?25h" + leave)
        sys.stdout.flush()


def export_frames(
    out_dir: Path,
    terms: list[np.ndarray],
    specs: list[VariantSpec],
    target: VariantRecord,
    render_cfg: RenderConfig,
    fps: float,
    style: str = "shade",
    profile: str = "clean",
    axis_gizmo: bool = False,
    intro_hold: float = 0.0,
    hold: float = 0.0,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    texts: list[str] = []
    for i, term in enumerate(terms):
        text = frame_to_text(term, render_cfg, style, specs[i], axis_gizmo, axis_color=False)
        texts.append(text)
        (out_dir / f"frame_{i:04d}.txt").write_text(text + "\n", encoding="utf-8")
    manifest = {
        "engine": "dejong-3d-identity-animation",
        "code": target.code,
        "mode": target.spec.mode,
        "profile": profile,
        "axis_gizmo": axis_gizmo,
        "fps": fps,
        "frame_count": len(terms),
        "preview_intro_hold": intro_hold,
        "preview_final_hold": hold,
        "target": target.manifest_record(),
        "frames": [
            {
                "index": i,
                "yaw_deg": specs[i].yaw_deg,
                "pitch_deg": specs[i].pitch_deg,
                "roll_deg": specs[i].roll_deg,
                "depth": specs[i].depth,
                "twist": specs[i].twist,
                "perspective": specs[i].perspective,
            }
            for i in range(len(specs))
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    intro_repeat = max(0, int(round(float(intro_hold) * float(fps))))
    final_repeat = max(0, int(round(float(hold) * float(fps))))
    preview_texts = [texts[0]] * intro_repeat + texts + [texts[-1]] * final_repeat if texts else []
    js_frames = json.dumps(preview_texts)
    if axis_gizmo:
        def preview_html(text: str) -> str:
            spans = {
                "X": '<span class="axis-x">X</span>',
                "x": '<span class="axis-x">x</span>',
                "Y": '<span class="axis-y">Y</span>',
                "y": '<span class="axis-y">y</span>',
                "Z": '<span class="axis-z">Z</span>',
                "z": '<span class="axis-z">z</span>',
                "+": '<span class="axis-o">+</span>',
            }
            return "".join(spans.get(ch, html.escape(ch)) for ch in text)

        js_frames = json.dumps([preview_html(text) for text in preview_texts])
    duration_ms = max(16, round(1000.0 / max(fps, 1.0)))
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>De Jong identity animation</title>
<style>
:root{{color-scheme:dark}}body{{margin:0;background:#080c12;color:#e6edf3;font:14px system-ui,sans-serif;display:grid;min-height:100vh;place-items:center}}
.wrap{{width:min(94vw,900px)}}.meta{{color:#8b98aa;margin-bottom:12px}}pre{{margin:0;background:#030609;border:1px solid #263143;border-radius:10px;padding:18px;overflow:auto;font:14px/1 monospace;white-space:pre}}
button{{margin-top:12px;background:#111a27;color:#e6edf3;border:1px solid #263143;border-radius:6px;padding:6px 10px}}
.axis-x{{color:#ff6b6b;font-weight:700}}.axis-y{{color:#51cf66;font-weight:700}}.axis-z{{color:#74c0fc;font-weight:700}}.axis-o{{color:#f8f9fa;font-weight:700}}
</style>
</head><body><div class="wrap"><div class="meta">{html.escape(target.code)} · {html.escape(target.spec.mode)} · {len(terms)} motion frames · {len(preview_texts)} preview frames · {fps:g} fps</div><pre id="screen"></pre><button id="toggle">pause</button></div>
<script>
const frames={js_frames}; const htmlMode={str(axis_gizmo).lower()}; let i=0, playing=true; const screen=document.getElementById('screen');
function show(frame){{if(htmlMode){{screen.innerHTML=frame}}else{{screen.textContent=frame}}}}
function tick(){{if(playing){{show(frames[i]);i=(i+1)%frames.length}}}} show(frames[0]); setInterval(tick,{duration_ms});
document.getElementById('toggle').onclick=()=>{{playing=!playing;document.getElementById('toggle').textContent=playing?'pause':'play'}};
</script></body></html>"""
    path = out_dir / "index.html"
    path.write_text(page, encoding="utf-8")
    return path
