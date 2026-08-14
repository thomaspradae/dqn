from __future__ import annotations

import html
import json
from pathlib import Path

from .models import CanonicalSpec, RenderConfig, VariantRecord, VariantSearchConfig
from .render import SHADE_RAMP, terminal_to_binary, terminal_to_text, text_block


def export_gallery(
    out_dir: Path,
    assets: dict[str, object],
    variants: list[VariantRecord],
    render_cfg: RenderConfig,
    search_cfg: VariantSearchConfig,
    canonical_spec: CanonicalSpec,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "engine": "dejong-3d-identity-engine",
        "canonical": {
            "seed": canonical_spec.seed,
            "rank": canonical_spec.rank,
            "raw_score": canonical_spec.raw_score,
            "selection_score": canonical_spec.selection_score,
            "params": {"a": canonical_spec.a, "b": canonical_spec.b, "c": canonical_spec.c, "d": canonical_spec.d},
            "iterations": canonical_spec.iterations,
            "burn_in": canonical_spec.burn_in,
            "signature": assets["signature"],
        },
        "search": {
            "master_seed": search_cfg.master_seed,
            "count": search_cfg.count,
            "candidates_per_code": search_cfg.candidates_per_code,
            "similarity_floor": search_cfg.similarity_floor,
            "silhouette_target": search_cfg.silhouette_target,
            "perspective_strength": search_cfg.perspective_strength,
            "expressive_weight": search_cfg.expressive_weight,
        },
        "render": {
            "term_width": render_cfg.term_width,
            "term_height": render_cfg.term_height,
            "supersample": render_cfg.supersample,
            "cell_aspect": render_cfg.cell_aspect,
            "margin": render_cfg.margin,
            "blur_passes": render_cfg.blur_passes,
            "pool_max_weight": render_cfg.pool_max_weight,
            "gamma": render_cfg.gamma,
            "ink_threshold": render_cfg.ink_threshold,
            "hard_threshold": render_cfg.hard_threshold,
            "ramp": SHADE_RAMP,
        },
        "modes": list(search_cfg.modes),
        "variants": [record.manifest_record() for record in variants],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    cards: list[str] = []
    for record in variants:
        shaded = terminal_to_text(record.term, SHADE_RAMP, render_cfg.gamma)
        binary = terminal_to_binary(record.term, render_cfg.ink_threshold)
        (out_dir / f"variant_{record.id}.txt").write_text(text_block(shaded, render_cfg.term_width) + "\n", encoding="utf-8")
        (out_dir / f"variant_{record.id}_binary.txt").write_text(text_block(binary, render_cfg.term_width) + "\n", encoding="utf-8")
        (out_dir / f"variant_{record.id}.json").write_text(json.dumps(record.manifest_record(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        spec = record.spec
        spec_line = (
            f"mode={spec.mode} yaw={spec.yaw_deg:.1f} pitch={spec.pitch_deg:.1f} roll={spec.roll_deg:.1f} "
            f"depth={spec.depth:.3f} gamma={spec.gamma:.3f} twist={spec.twist:.3f} "
            f"persp={spec.perspective:.3f} focal={spec.focal:.3f} camera_z={spec.camera_z:.3f}"
        )
        cards.append(
            f"""
<article class=\"card\">
  <div class=\"meta\"><strong>#{record.id}</strong><span>{html.escape(record.code)}</span><span>{html.escape(record.spec.mode)}</span><span>score {record.raw_score:.3f}</span><span>identity {record.identity_score:.3f}</span><span>expressive {record.expressive_score:.3f}</span></div>
  <div class=\"tabs\"><button class=\"active\" onclick=\"showTab(this,'shade')\">shade</button><button onclick=\"showTab(this,'binary')\">binary</button></div>
  <pre data-tab=\"shade\">{html.escape(text_block(shaded, render_cfg.term_width))}</pre>
  <pre data-tab=\"binary\" hidden>{html.escape(text_block(binary, render_cfg.term_width))}</pre>
  <div class=\"metrics\">coverage {record.metrics['coverage']:.2f} · components {record.metrics['components']:.0f} · holes {record.metrics['holes']:.0f} · fill {record.metrics['bbox_fill']:.2f} · perimeter {record.metrics['perimeter_ratio']:.2f}</div>
  <code>{html.escape(spec_line)}</code>
  <div class=\"links\"><a href=\"variant_{record.id}.txt\">shade txt</a><a href=\"variant_{record.id}_binary.txt\">binary txt</a><a href=\"variant_{record.id}.json\">json</a></div>
</article>
"""
        )

    page = f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>De Jong 3D identity engine</title>
<style>
:root {{ color-scheme: dark; --bg:#090d14; --panel:#101722; --line:#263143; --ink:#e6edf3; --muted:#8b98aa; --accent:#7dd3fc; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 ui-sans-serif,system-ui,sans-serif; }}
header {{ position:sticky; top:0; z-index:3; padding:14px 18px; background:rgba(9,13,20,.96); border-bottom:1px solid var(--line); }}
header strong {{ font-size:16px; }} header span {{ color:var(--muted); margin-left:14px; display:inline-block; }}
main {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(560px,1fr)); gap:14px; padding:14px; }}
.card {{ min-width:0; background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px; overflow:auto; }}
.meta,.tabs,.links {{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; }}
.meta span,.metrics,code {{ color:var(--muted); }}
.tabs {{ margin-top:8px; }}
button {{ background:#172131; color:var(--ink); border:1px solid var(--line); border-radius:5px; padding:3px 8px; cursor:pointer; }}
button.active {{ border-color:var(--accent); color:var(--accent); }}
pre {{ margin:10px 0; padding:12px; background:#05080d; border:1px solid #1d2735; border-radius:7px; overflow:auto; font:12px/1 monospace; letter-spacing:0; white-space:pre; }}
code {{ display:block; margin-top:7px; font:11px/1.4 monospace; white-space:normal; }}
a {{ color:var(--accent); text-decoration:none; }} .links {{ margin-top:8px; }}
</style>
<script>
function showTab(btn, tab) {{ const card=btn.closest('.card'); card.querySelectorAll('button').forEach(b=>b.classList.remove('active')); btn.classList.add('active'); card.querySelectorAll('pre[data-tab]').forEach(p=>p.hidden=p.dataset.tab!==tab); }}
</script>
</head>
<body>
<header>
  <strong>De Jong 3D identity engine</strong>
  <span>canonical seed {html.escape(canonical_spec.seed)}</span>
  <span>a={canonical_spec.a:.6f} b={canonical_spec.b:.6f} c={canonical_spec.c:.6f} d={canonical_spec.d:.6f}</span>
  <span>{len(variants)} variants</span>
</header>
<main>{''.join(cards)}</main>
</body>
</html>
"""
    html_path = out_dir / "index.html"
    html_path.write_text(page, encoding="utf-8")
    return html_path
