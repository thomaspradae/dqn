#!/usr/bin/env python3
"""Export a non-ASCII canvas preview of the De Jong 3D identity rotation."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from dejong3d.animation import build_animation, profile_config
from dejong3d.geometry import MODES, build_geometry, transform_and_project_with_depth
from dejong3d.models import CanonicalSpec, RenderConfig, VariantSearchConfig
from dejong3d.render import rotation_matrix_xyz


def axis_endpoints(spec, scale: float = 0.75) -> dict[str, list[float]]:
    rotation = rotation_matrix_xyz(spec.pitch_deg, spec.yaw_deg, spec.roll_deg)
    axes = {
        "x": np.array([scale, 0.0, 0.0], dtype=np.float64),
        "y": np.array([0.0, scale, 0.0], dtype=np.float64),
        "z": np.array([0.0, 0.0, scale], dtype=np.float64),
    }
    out: dict[str, list[float]] = {"o": [0.0, 0.0]}
    for name, vec in axes.items():
        rotated = vec @ rotation.T
        out[name] = [float(rotated[0]), float(rotated[1])]
    return out


def frame_points(assets: dict[str, object], spec, max_points: int) -> dict[str, object]:
    cloud3d, weights = build_geometry(assets["base_points"], assets["densities"], spec)
    pts2, _, depths = transform_and_project_with_depth(cloud3d, spec)
    if len(pts2) > max_points:
        step = max(1, len(pts2) // max_points)
        pts2 = pts2[::step]
        depths = depths[::step]
        weights = weights[::step]

    # Global canonical normalization is already roughly [-1, 1]. Keep this stable
    # so rotation reads as object motion rather than per-frame recentering.
    pts = np.column_stack((pts2[:, 0], pts2[:, 1], depths, weights))
    return {
        "points": np.round(pts, 4).tolist(),
        "axis": axis_endpoints(spec),
        "yaw": round(float(spec.yaw_deg), 3),
        "pitch": round(float(spec.pitch_deg), 3),
        "roll": round(float(spec.roll_deg), 3),
    }


def write_html(out: Path, payload: dict[str, object]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, separators=(",", ":"))
    (out / "frames.json").write_text(data, encoding="utf-8")
    page = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>De Jong real rotation preview</title>
<style>
:root{color-scheme:dark}
body{margin:0;background:#05070a;color:#dce6f1;font:14px system-ui,sans-serif;display:grid;min-height:100vh;place-items:center}
.wrap{width:min(96vw,980px)}
.meta{display:flex;gap:14px;flex-wrap:wrap;color:#8fa1b7;margin:0 0 10px}
canvas{display:block;width:100%;aspect-ratio:1.45;background:#020407;border:1px solid #243041;border-radius:8px}
button{margin-top:10px;background:#101826;color:#dce6f1;border:1px solid #2a384c;border-radius:6px;padding:6px 10px}
.x{color:#ff6b6b}.y{color:#51cf66}.z{color:#74c0fc}
</style>
</head>
<body>
<div class="wrap">
  <div class="meta">
    <span id="title"></span>
    <span><b class="x">X</b> red</span>
    <span><b class="y">Y</b> green</span>
    <span><b class="z">Z</b> blue</span>
    <span id="pose"></span>
  </div>
  <canvas id="c" width="1160" height="800"></canvas>
  <button id="toggle">pause</button>
</div>
<script>
let frames=[], i=0, playing=true;
const data=__DATA__;
const c=document.getElementById('c'), ctx=c.getContext('2d');
const title=document.getElementById('title'), pose=document.getElementById('pose');
const toggle=document.getElementById('toggle');
function sx(x){ return c.width*0.5 + x*c.width*0.31; }
function sy(y){ return c.height*0.5 - y*c.height*0.31; }
function drawAxis(f){
  const o=f.axis.o;
  const axes=[['x','#ff6b6b'],['y','#51cf66'],['z','#74c0fc']];
  ctx.lineWidth=4; ctx.font='700 18px ui-monospace, monospace'; ctx.textBaseline='middle';
  for(const [name,color] of axes){
    const p=f.axis[name];
    ctx.strokeStyle=color; ctx.fillStyle=color;
    ctx.beginPath(); ctx.moveTo(sx(o[0]), sy(o[1])); ctx.lineTo(sx(p[0]), sy(p[1])); ctx.stroke();
    ctx.beginPath(); ctx.arc(sx(p[0]), sy(p[1]), 5, 0, Math.PI*2); ctx.fill();
    ctx.fillText(name.toUpperCase(), sx(p[0])+8, sy(p[1]));
  }
  ctx.fillStyle='#f8f9fa'; ctx.beginPath(); ctx.arc(sx(o[0]), sy(o[1]), 5, 0, Math.PI*2); ctx.fill();
}
function draw(f){
  ctx.clearRect(0,0,c.width,c.height);
  ctx.fillStyle='#020407'; ctx.fillRect(0,0,c.width,c.height);
  const pts=f.points.slice().sort((a,b)=>a[2]-b[2]);
  for(const p of pts){
    const z=p[2], w=p[3];
    const b=Math.max(45, Math.min(245, 65 + z*155 + w*40));
    const r=1.0 + z*1.7;
    ctx.fillStyle=`rgba(${b},${b},${b},${0.22 + z*0.62})`;
    ctx.beginPath(); ctx.arc(sx(p[0]), sy(p[1]), r, 0, Math.PI*2); ctx.fill();
  }
  drawAxis(f);
  pose.textContent=`yaw ${f.yaw.toFixed(1)} / pitch ${f.pitch.toFixed(1)} / roll ${f.roll.toFixed(1)}`;
}
function tick(){ if(playing){ draw(frames[i]); i=(i+1)%frames.length; } }
frames=data.frames; title.textContent=`${data.code} · ${data.mode} · ${data.frames.length} frames @ ${data.fps}fps`;
draw(frames[0]); setInterval(tick, 1000/data.fps);
toggle.onclick=()=>{ playing=!playing; toggle.textContent=playing?'pause':'play'; };
</script>
</body>
</html>
"""
    page = page.replace("__DATA__", data)
    (out / "index.html").write_text(page, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export a browser/canvas preview of the actual 3D rotation.")
    p.add_argument("code", nargs="?", default="node:ofi1")
    p.add_argument("--profile", choices=("clean", "hero", "loop"), default="clean")
    p.add_argument("--mode", choices=MODES, default=None)
    p.add_argument("--seed", default="poormans-hpc")
    p.add_argument("--frames", type=int, default=72)
    p.add_argument("--fps", type=float, default=20.0)
    p.add_argument("--candidates", type=int, default=1)
    p.add_argument("--max-points", type=int, default=6500)
    p.add_argument("--out", type=Path, default=Path("examples/real_rotation_surface"))
    return p


def main() -> int:
    args = build_parser().parse_args()
    profile = profile_config(args.profile)
    mode = args.mode or str(profile["mode"])
    render_cfg = RenderConfig()
    render_cfg = replace(render_cfg, **profile["render"])
    search_cfg = VariantSearchConfig(master_seed=args.seed, candidates_per_code=args.candidates, modes=(mode,))
    assets, target, _, specs = build_animation(
        code=args.code,
        render_cfg=render_cfg,
        search_cfg=search_cfg,
        canonical_spec=CanonicalSpec(),
        frames=args.frames,
        mode=mode,
        profile=args.profile,
    )
    frames = [frame_points(assets, spec, args.max_points) for spec in specs]
    payload = {
        "code": args.code,
        "mode": mode,
        "profile": args.profile,
        "fps": args.fps,
        "frames": frames,
        "target": target.manifest_record(),
    }
    write_html(args.out, payload)
    print(f"saved real rotation preview -> {args.out / 'index.html'}")
    print(
        f"target yaw={target.spec.yaw_deg:.1f} pitch={target.spec.pitch_deg:.1f} roll={target.spec.roll_deg:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
