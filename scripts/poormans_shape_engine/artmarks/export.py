from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path

from .models import Candidate, RenderConfig, SearchConfig
from .render import SHADE_RAMP, terminal_to_binary, terminal_to_text, text_block


def export_gallery(out_dir: Path, selected: list[Candidate], pool: list[Candidate], render_cfg: RenderConfig, search_cfg: SearchConfig) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 2,
        "engine": "poormans-shape-engine",
        "search": {
            "seed": search_cfg.seed,
            "per_family": search_cfg.per_family,
            "count": search_cfg.count,
            "iteration_scale": search_cfg.iteration_scale,
            "family_balance": search_cfg.family_balance,
            "max_similarity": search_cfg.max_similarity,
            "max_per_family": search_cfg.max_per_family,
            "seed_each_family": search_cfg.seed_each_family,
        },
        "render": {
            "term_width": render_cfg.term_width,
            "term_height": render_cfg.term_height,
            "supersample": render_cfg.supersample,
            "cell_aspect": render_cfg.cell_aspect,
            "margin": render_cfg.margin,
            "pca_align": render_cfg.pca_align,
            "blur_passes": render_cfg.blur_passes,
            "pool_max_weight": render_cfg.pool_max_weight,
            "gamma": render_cfg.gamma,
            "ink_threshold": render_cfg.ink_threshold,
            "hard_threshold": render_cfg.hard_threshold,
            "ramp": SHADE_RAMP,
        },
        "pool_size": len(pool),
        "pool_by_family": dict(Counter(c.family for c in pool)),
        "selected_by_family": dict(Counter(c.family for c in selected)),
        "candidates": [c.manifest_record() for c in selected],
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    cards: list[str] = []
    for candidate in selected:
        shaded = terminal_to_text(candidate.term, SHADE_RAMP, render_cfg.gamma)
        binary = terminal_to_binary(candidate.term, render_cfg.ink_threshold)
        (out_dir / f"candidate_{candidate.id}.txt").write_text(text_block(shaded, render_cfg.term_width) + "\n", encoding="utf-8")
        (out_dir / f"candidate_{candidate.id}_binary.txt").write_text(text_block(binary, render_cfg.term_width) + "\n", encoding="utf-8")
        (out_dir / f"candidate_{candidate.id}.json").write_text(
            json.dumps(candidate.manifest_record(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        metrics = candidate.metrics
        params = " ".join(f"{k}={_short(v)}" for k, v in candidate.params.items())
        cards.append(
            f"""
<article class="card">
  <div class="meta"><strong>#{candidate.id}</strong><span>{html.escape(candidate.family)}</span><span>score {candidate.raw_score:.3f}</span><span>select {candidate.selection_score:.3f}</span></div>
  <div class="tabs"><button class="active" onclick="showTab(this,'shade')">shade</button><button onclick="showTab(this,'binary')">binary</button></div>
  <pre data-tab="shade">{html.escape(text_block(shaded, render_cfg.term_width))}</pre>
  <pre data-tab="binary" hidden>{html.escape(text_block(binary, render_cfg.term_width))}</pre>
  <div class="metrics">coverage {metrics['coverage']:.2f} · components {metrics['components']:.0f} · holes {metrics['holes']:.0f} · fill {metrics['bbox_fill']:.2f} · perimeter {metrics['perimeter_ratio']:.2f}</div>
  <code>{html.escape(params)}</code>
  <div class="links"><a href="candidate_{candidate.id}.txt">shade txt</a><a href="candidate_{candidate.id}_binary.txt">binary txt</a><a href="candidate_{candidate.id}.json">json</a></div>
</article>
"""
        )

    counts = Counter(c.family for c in selected)
    family_summary = " · ".join(f"{name} {count}" for name, count in sorted(counts.items()))
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>poormans shape engine</title>
<style>
:root {{ color-scheme: dark; --bg:#090d14; --panel:#101722; --line:#263143; --ink:#e6edf3; --muted:#8b98aa; --accent:#7dd3fc; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 ui-sans-serif,system-ui,sans-serif; }}
header {{ position:sticky; top:0; z-index:3; padding:14px 18px; background:rgba(9,13,20,.96); border-bottom:1px solid var(--line); }}
header strong {{ font-size:16px; }} header span {{ color:var(--muted); margin-left:14px; }}
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
<header><strong>poormans shape engine</strong><span>{len(selected)} selected from {len(pool)} accepted</span><span>{html.escape(family_summary)}</span></header>
<main>{''.join(cards)}</main>
</body>
</html>
"""
    html_path = out_dir / "index.html"
    html_path.write_text(page, encoding="utf-8")
    return html_path


def _short(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)
