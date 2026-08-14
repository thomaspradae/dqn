from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dejong3d.models import RenderConfig
from dejong3d.render import SHADE_RAMP, terminal_to_text, text_block

from .models import NativeCandidate


def _round_points(points: np.ndarray, max_points: int) -> list[list[float]]:
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points = points[indices]
    return np.round(points, 5).tolist()


def _candidate_payload(candidate: NativeCandidate, render_cfg: RenderConfig, max_points: int) -> dict[str, object]:
    record = candidate.manifest_record()
    record["points"] = _round_points(candidate.points, max_points)
    record["terminal_text"] = text_block(
        terminal_to_text(candidate.terminal, SHADE_RAMP, render_cfg.gamma), render_cfg.term_width
    )
    return record


def export_gallery(
    out: Path,
    candidates: list[NativeCandidate],
    rejected: list[dict[str, object]],
    render_cfg: RenderConfig,
    max_points: int = 6500,
) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    # Output directories are reproducible build artifacts. Remove only files
    # owned by this exporter so a smaller rerun cannot leave stale candidates.
    for pattern in ("candidate_*.json", "candidate_*.txt"):
        for stale in out.glob(pattern):
            stale.unlink()
    manifest = {
        "engine": "native-3d-object-first-v1",
        "rule": "the object exists completely from frame zero; only the camera moves",
        "candidate_count": len(candidates),
        "candidates": [candidate.manifest_record() for candidate in candidates],
        "rejected": rejected,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    payloads = []
    for candidate in candidates:
        payload = _candidate_payload(candidate, render_cfg, max_points)
        payloads.append(payload)
        stem = f"candidate_{candidate.id}_{candidate.family}"
        metadata = candidate.manifest_record()
        (out / f"{stem}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (out / f"{stem}.txt").write_text(str(payload["terminal_text"]) + "\n", encoding="utf-8")

    data = json.dumps({"candidates": payloads, "rejected": rejected}, separators=(",", ":"))
    page = _PAGE.replace("__DATA__", data)
    path = out / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


_PAGE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Native 3D identity search</title>
<style>
:root{color-scheme:dark;--bg:#07090c;--panel:#0d1117;--line:#252b34;--text:#e6edf3;--muted:#8b949e;--accent:#d7ff4f}
*{box-sizing:border-box;scrollbar-color:#303842 #0d1117;scrollbar-width:thin}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:0}
button{font:inherit}.app{height:100vh;display:grid;grid-template-columns:250px minmax(0,1fr) 300px;overflow:hidden}
.rail,.info{background:var(--panel);overflow:auto}.rail{border-right:1px solid var(--line)}.info{border-left:1px solid var(--line);padding:18px}
.rail-head{padding:16px 14px 12px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--panel);z-index:2}.rail-head strong{display:block;font-size:15px}.rail-head span{color:var(--muted);font-size:12px}
.candidate{display:grid;grid-template-columns:32px 1fr;gap:10px;width:100%;padding:11px 14px;border:0;border-bottom:1px solid #1c222a;background:transparent;color:var(--text);text-align:left;cursor:pointer}
.candidate:hover,.candidate.active{background:#161b22}.candidate.active{box-shadow:inset 3px 0 var(--accent)}.rank{color:var(--muted)}.family{font-weight:700}.score{display:block;color:var(--muted);font-size:11px;margin-top:2px}
.stage{min-width:0;display:grid;grid-template-rows:auto 1fr auto;background:#030507}.toolbar{display:flex;align-items:center;gap:8px;padding:10px 14px;border-bottom:1px solid var(--line);background:#090c10;overflow-x:auto}
.toolbar button{border:1px solid #303842;background:#111720;color:var(--text);border-radius:5px;padding:5px 9px;cursor:pointer;white-space:nowrap}.toolbar button:hover{border-color:#566171}.toolbar .active{background:#e6edf3;color:#090c10;border-color:#e6edf3}.spacer{flex:1}.pose{color:var(--muted);font-size:12px;white-space:nowrap}
.viewport{position:relative;min-height:0;display:grid;place-items:center;overflow:hidden;cursor:grab;touch-action:none;user-select:none}.viewport.dragging{cursor:grabbing}.viewport canvas{width:100%;height:100%;display:block}.terminal{display:none;width:min(96%,980px);max-height:92%;overflow:auto;white-space:pre;color:#f0f3f6;font:clamp(8px,1.12vw,16px)/1 ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:0;pointer-events:none}
.viewport.terminal-mode canvas{display:none}.viewport.terminal-mode .terminal{display:block}.timeline{height:4px;background:#171c23}.timeline i{display:block;height:100%;width:0;background:var(--accent)}
.info h1{font-size:18px;margin:0 0 3px}.sub{color:var(--muted);margin-bottom:18px}.group{border-top:1px solid var(--line);padding-top:14px;margin-top:14px}.group h2{font-size:11px;text-transform:uppercase;color:var(--muted);margin:0 0 9px}.metric{display:flex;justify-content:space-between;gap:16px;padding:3px 0}.metric span:last-child{color:#fff}.rule{color:var(--accent);font-size:12px;margin-top:16px}.axis-key{display:flex;gap:12px;color:var(--muted)}.x{color:#ff6b6b}.y{color:#69db7c}.z{color:#74c0fc}
.snapshot-panel{position:fixed;z-index:10;top:0;right:0;width:min(430px,100vw);height:100vh;background:#0b0f14;border-left:1px solid var(--line);transform:translateX(100%);transition:transform .18s ease;display:grid;grid-template-rows:auto 1fr;box-shadow:-14px 0 36px #0008}.snapshot-panel.open{transform:translateX(0)}.snapshot-head{display:flex;align-items:center;gap:10px;padding:13px 15px;border-bottom:1px solid var(--line)}.snapshot-head strong{flex:1}.snapshot-head button,.snap-actions button{border:1px solid #303842;background:#111720;color:var(--text);border-radius:5px;padding:5px 8px;cursor:pointer}.snapshot-list{overflow:auto;padding-bottom:20px}.empty{padding:20px;color:var(--muted)}.snap{border-bottom:1px solid var(--line);padding:14px}.snap-title{display:flex;justify-content:space-between;gap:10px}.snap-title strong{color:var(--accent)}.snap-meta{color:var(--muted);font-size:11px;margin:5px 0 9px}.snap pre{margin:0;max-height:220px;overflow:auto;white-space:pre;font:7px/1 ui-monospace,monospace;color:#dce3ea;background:#030507;padding:8px;border:1px solid #1c222a}.snap-actions{display:flex;gap:6px;flex-wrap:wrap;margin-top:9px}.hint{position:absolute;left:12px;bottom:10px;color:#7d8997;font-size:11px;pointer-events:none;background:#030507cc;padding:3px 6px}
@media(max-width:900px){.app{grid-template-columns:190px minmax(0,1fr)}.info{display:none}}@media(max-width:620px){.app{grid-template-columns:1fr;grid-template-rows:150px 1fr}.rail{border-right:0;border-bottom:1px solid var(--line)}.candidate{padding:8px 12px}.pose{display:none}}
</style>
</head>
<body>
<main class="app">
  <aside class="rail"><div class="rail-head"><strong>native 3D search</strong><span id="count"></span></div><div id="list"></div></aside>
  <section class="stage">
    <div class="toolbar">
      <button id="play" title="Play or pause">Pause</button>
      <button id="restart" title="Restart from canonical view">Restart</button>
      <button id="view3d" class="active">3D</button><button id="viewTerm">Terminal</button>
      <button id="saveView" title="Save this terminal projection and camera pose">Save view</button>
      <button id="openSnapshots" title="Open saved terminal views">Snapshots <span id="snapCount">0</span></button>
      <label><input id="axes" type="checkbox" checked> axes</label>
      <span class="spacer"></span><span class="pose" id="pose"></span>
    </div>
    <div class="viewport" id="viewport"><canvas id="canvas"></canvas><pre class="terminal" id="terminal"></pre><span class="hint">drag: orbit · Shift+drag: roll · wheel: zoom</span></div>
    <div class="timeline"><i id="progress"></i></div>
  </section>
  <aside class="info" id="info"></aside>
</main>
<aside class="snapshot-panel" id="snapshotPanel"><div class="snapshot-head"><strong>Terminal snapshots</strong><button id="closeSnapshots" title="Close">Close</button></div><div class="snapshot-list" id="snapshotList"></div></aside>
<script>
const DATA=__DATA__, candidates=DATA.candidates;
const list=document.getElementById('list'), count=document.getElementById('count'), info=document.getElementById('info');
const canvas=document.getElementById('canvas'), ctx=canvas.getContext('2d'), viewport=document.getElementById('viewport');
const terminal=document.getElementById('terminal'), poseLabel=document.getElementById('pose'), progress=document.getElementById('progress');
const playButton=document.getElementById('play'),axesToggle=document.getElementById('axes'),snapshotPanel=document.getElementById('snapshotPanel'),snapshotList=document.getElementById('snapshotList'),snapCount=document.getElementById('snapCount');
const query=new URLSearchParams(location.search);
let selected=clamp(Number(query.get('candidate')||0),0,Math.max(candidates.length-1,0)),playing=query.get('still')!=='1',elapsed=0,previous=performance.now(),currentPose=null,zoom=1,viewMode=query.get('mode')==='terminal'?'terminal':'3d',dragging=false,lastPointer=null,renderPending=false,snapshots=[];
const HOLD_START=1.1,MOVE=3.4,HOLD_END=1.5,TOTAL=HOLD_START+MOVE+HOLD_END;
count.textContent=`${candidates.length} survivors · ${DATA.rejected.length} rejected`;
function metric(label,value,digits=3){return `<div class="metric"><span>${label}</span><span>${Number(value).toFixed(digits)}</span></div>`}
function clonePose(p){return {pitch_deg:p.pitch_deg,yaw_deg:p.yaw_deg,roll_deg:p.roll_deg,perspective:p.perspective}}
function select(index){selected=index;elapsed=0;zoom=1;currentPose=clonePose(candidates[index].canonical_pose);[...list.children].forEach((b,i)=>b.classList.toggle('active',i===index));renderInfo();renderCurrent()}
for(const [i,c] of candidates.entries()){
  const button=document.createElement('button');button.className='candidate';button.innerHTML=`<span class="rank">#${c.id}</span><span><span class="family">${c.family}</span><span class="score">score ${c.total_score.toFixed(3)}</span></span>`;button.onclick=()=>select(i);list.appendChild(button)
}
function renderInfo(){const c=candidates[selected],m=c.object_metrics,p=c.projection_metrics;info.innerHTML=`
  <h1>#${c.id} ${c.family}</h1><div class="sub">${c.seed}<br>${c.metadata.topology} · ${c.metadata.generator}</div>
  <div class="axis-key"><span class="x">X red</span><span class="y">Y green</span><span class="z">Z blue</span></div>
  <div class="group"><h2>Object</h2>${metric('lambda 2 / 1',m.lambda2_ratio)}${metric('lambda 3 / 1',m.lambda3_ratio)}${metric('local lambda 2 / 1',m.local_lambda2_ratio)}${metric('local lambda 3 / 1',m.local_lambda3_ratio)}${metric('box dimension',m.box_dimension)}${metric('octants',m.octant_occupancy)}${metric('object score',c.object_score)}</div>
  <div class="group"><h2>Canonical projection</h2>${metric('coverage',p.coverage)}${metric('holes',p.holes,0)}${metric('perimeter',p.perimeter_ratio)}${metric('depth complexity',p.depth_complexity)}${metric('De Jong spirit',p.dejong_spirit)}${metric('projection score',c.projection_score)}</div>
  <div class="group"><h2>Parameters</h2>${Object.entries(c.params).map(([k,v])=>metric(k,v,4)).join('')}</div>
  <div class="rule">Object complete at frame zero. Camera motion only.</div>`}
function resize(){const r=canvas.getBoundingClientRect(),d=Math.min(devicePixelRatio||1,2);canvas.width=Math.max(1,Math.floor(r.width*d));canvas.height=Math.max(1,Math.floor(r.height*d));ctx.setTransform(d,0,0,d,0,0)}
new ResizeObserver(()=>{resize();scheduleRender()}).observe(canvas);
function clamp(v,a,b){return Math.max(a,Math.min(b,v))}function smooth(t){t=clamp(t,0,1);return t*t*t*(t*(t*6-15)+10)}
function angleLerp(a,b,t){let d=((b-a+180)%360+360)%360-180;return a+d*t}function lerp(a,b,t){return a+(b-a)*t}
function poseAt(t){const c=candidates[selected],a=c.canonical_pose,b=c.target_pose,e=smooth(t);return {pitch_deg:angleLerp(a.pitch_deg,b.pitch_deg,e),yaw_deg:angleLerp(a.yaw_deg,b.yaw_deg,e),roll_deg:angleLerp(a.roll_deg,b.roll_deg,e),perspective:lerp(a.perspective,b.perspective,e)}}
function matrix(p){const x=p.pitch_deg*Math.PI/180,y=p.yaw_deg*Math.PI/180,z=p.roll_deg*Math.PI/180,cx=Math.cos(x),sx=Math.sin(x),cy=Math.cos(y),sy=Math.sin(y),cz=Math.cos(z),sz=Math.sin(z);return [[cz*cy,cz*sy*sx-sz*cx,cz*sy*cx+sz*sx],[sz*cy,sz*sy*sx+cz*cx,sz*sy*cx-cz*sx],[-sy,cy*sx,cy*cx]]}
function transform(v,m){return [m[0][0]*v[0]+m[0][1]*v[1]+m[0][2]*v[2],m[1][0]*v[0]+m[1][1]*v[1]+m[1][2]*v[2],m[2][0]*v[0]+m[2][1]*v[1]+m[2][2]*v[2]]}
function currentT(){return clamp((elapsed-HOLD_START)/MOVE,0,1)}
function projectedPoints(c,p){const m=matrix(p),points=[];for(const v of c.points){let q=transform(v,m);if(p.perspective>0){const s=3/Math.max(3.2-q[2],.5),k=1-p.perspective+p.perspective*s;q=[q[0]*k,q[1]*k,q[2]]}points.push(q)}return {points,m}}
function drawAxes(m,cx,cy,scale){if(!axesToggle.checked)return;const axes=[[[.48,0,0],'#ff6b6b','X'],[[0,.48,0],'#69db7c','Y'],[[0,0,.48],'#74c0fc','Z']];ctx.lineWidth=3;ctx.font='700 14px ui-monospace,monospace';for(const [v,color,label] of axes){const q=transform(v,m);ctx.strokeStyle=color;ctx.fillStyle=color;ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(cx+q[0]*scale,cy-q[1]*scale);ctx.stroke();ctx.fillText(label,cx+q[0]*scale+5,cy-q[1]*scale)}ctx.fillStyle='#fff';ctx.beginPath();ctx.arc(cx,cy,3,0,Math.PI*2);ctx.fill()}
function quantile(values,q){if(!values.length)return 1;values.sort((a,b)=>a-b);return values[Math.min(values.length-1,Math.floor((values.length-1)*q))]||1}
function terminalProjection(c,p,includeHeader=true){const W=60,H=30,S=2,HW=W*S,HH=H*S,aspect=.55,{points}=projectedPoints(c,p);let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;for(const q of points){minX=Math.min(minX,q[0]);maxX=Math.max(maxX,q[0]);minY=Math.min(minY,q[1]);maxY=Math.max(maxY,q[1])}const spanX=Math.max(maxX-minX,1e-9),spanY=Math.max(maxY-minY,1e-9),cx=(minX+maxX)/2,cy=(minY+maxY)/2,scale=Math.min(HW*.88*aspect/spanX,HH*.88/spanY)*zoom,grid=new Float64Array(HW*HH);for(const q of points){const x=Math.round((q[0]-cx)*scale/aspect+(HW-1)/2),y=Math.round(-(q[1]-cy)*scale+(HH-1)/2);if(x>=0&&x<HW&&y>=0&&y<HH){const depth=clamp(.5+.5*q[2]/1.6,0,1);grid[y*HW+x]+=.7+.65*depth}}for(let i=0;i<grid.length;i++)grid[i]=Math.log1p(grid[i]);let positives=Array.from(grid).filter(v=>v>0),cap=quantile(positives,.995);for(let i=0;i<grid.length;i++)grid[i]=Math.min(grid[i]/cap,1);const term=new Float64Array(W*H);for(let y=0;y<H;y++)for(let x=0;x<W;x++){let mx=0,sum=0;for(let dy=0;dy<S;dy++)for(let dx=0;dx<S;dx++){const v=grid[(y*S+dy)*HW+x*S+dx];mx=Math.max(mx,v);sum+=v}term[y*W+x]=.72*mx+.28*(sum/(S*S))}positives=Array.from(term).filter(v=>v>0);cap=quantile(positives,.99);const ramp=' ░▒▓█',lines=[];for(let y=0;y<H;y++){let line='';for(let x=0;x<W;x++){const v=Math.pow(clamp(term[y*W+x]/cap,0,1),.82);line+=ramp[Math.round(v*(ramp.length-1))]}lines.push(line.replace(/\s+$/,''))}const header=`${c.family} #${c.id}\npitch ${p.pitch_deg.toFixed(2)}  yaw ${p.yaw_deg.toFixed(2)}  roll ${p.roll_deg.toFixed(2)}  perspective ${p.perspective.toFixed(3)}  zoom ${zoom.toFixed(2)}`;return includeHeader?header+'\n\n'+lines.join('\n'):lines.join('\n')}
function renderCurrent(){if(!candidates.length||!currentPose)return;const c=candidates[selected],p=currentPose,{points,m}=projectedPoints(c,p),w=canvas.clientWidth,h=canvas.clientHeight,cx=w/2,cy=h/2,scale=Math.min(w,h)*.37*zoom;ctx.clearRect(0,0,w,h);ctx.fillStyle='#030507';ctx.fillRect(0,0,w,h);points.sort((a,b)=>a[2]-b[2]);for(const q of points){const zn=clamp((q[2]+1.3)/2.6,0,1),b=Math.round(44+196*zn),alpha=.18+.60*zn,r=.55+1.15*zn;ctx.fillStyle=`rgba(${b},${b},${b},${alpha})`;ctx.beginPath();ctx.arc(cx+q[0]*scale,cy-q[1]*scale,r,0,Math.PI*2);ctx.fill()}drawAxes(m,cx,cy,scale);poseLabel.textContent=`pitch ${p.pitch_deg.toFixed(1)} · yaw ${p.yaw_deg.toFixed(1)} · roll ${p.roll_deg.toFixed(1)} · zoom ${zoom.toFixed(2)}`;if(viewMode==='terminal')terminal.textContent=terminalProjection(c,p,true)}
function scheduleRender(){if(renderPending)return;renderPending=true;requestAnimationFrame(()=>{renderPending=false;renderCurrent()})}
function tick(now){const dt=Math.min((now-previous)/1000,.1);previous=now;if(playing){elapsed+=dt;if(elapsed>TOTAL)elapsed=0;currentPose=poseAt(currentT());renderCurrent()}requestAnimationFrame(tick)}
function pauseForInteraction(){playing=false;playButton.textContent='Play'}
viewport.addEventListener('pointerdown',e=>{dragging=true;lastPointer={x:e.clientX,y:e.clientY};viewport.classList.add('dragging');viewport.setPointerCapture(e.pointerId);pauseForInteraction()});
viewport.addEventListener('pointermove',e=>{if(!dragging||!currentPose)return;const dx=e.clientX-lastPointer.x,dy=e.clientY-lastPointer.y;lastPointer={x:e.clientX,y:e.clientY};if(e.shiftKey)currentPose.roll_deg+=dx*.38;else{currentPose.yaw_deg+=dx*.38;currentPose.pitch_deg=clamp(currentPose.pitch_deg-dy*.38,-89,89)}scheduleRender()});
function stopDrag(e){dragging=false;lastPointer=null;viewport.classList.remove('dragging');if(e.pointerId!==undefined&&viewport.hasPointerCapture(e.pointerId))viewport.releasePointerCapture(e.pointerId)}viewport.addEventListener('pointerup',stopDrag);viewport.addEventListener('pointercancel',stopDrag);
viewport.addEventListener('wheel',e=>{e.preventDefault();pauseForInteraction();zoom=clamp(zoom*Math.exp(-e.deltaY*.0012),.45,2.6);scheduleRender()},{passive:false});
playButton.onclick=()=>{playing=!playing;if(playing){elapsed=HOLD_START+currentT()*MOVE}playButton.textContent=playing?'Pause':'Play'};document.getElementById('restart').onclick=()=>{pauseForInteraction();elapsed=0;zoom=1;currentPose=clonePose(candidates[selected].canonical_pose);renderCurrent()};axesToggle.onchange=scheduleRender;
document.getElementById('view3d').onclick=()=>{viewMode='3d';viewport.classList.remove('terminal-mode');document.getElementById('view3d').classList.add('active');document.getElementById('viewTerm').classList.remove('active');renderCurrent()};document.getElementById('viewTerm').onclick=()=>{viewMode='terminal';viewport.classList.add('terminal-mode');document.getElementById('viewTerm').classList.add('active');document.getElementById('view3d').classList.remove('active');pauseForInteraction();renderCurrent()};
function safeName(value){return value.replace(/[^a-z0-9_-]+/gi,'-').replace(/^-|-$/g,'').toLowerCase()}
function download(name,text,type='text/plain'){const blob=new Blob([text],{type}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
async function copyText(text){try{await navigator.clipboard.writeText(text)}catch{const area=document.createElement('textarea');area.value=text;document.body.appendChild(area);area.select();document.execCommand('copy');area.remove()}}
function renderSnapshots(){snapCount.textContent=String(snapshots.length);if(!snapshots.length){snapshotList.innerHTML='<div class="empty">No saved views yet.</div>';return}snapshotList.innerHTML='';snapshots.forEach((s,i)=>{const section=document.createElement('section');section.className='snap';section.innerHTML=`<div class="snap-title"><strong>Snapshot ${i+1}</strong><span>${s.family} #${s.candidateId}</span></div><div class="snap-meta">pitch ${s.pose.pitch_deg.toFixed(2)} · yaw ${s.pose.yaw_deg.toFixed(2)} · roll ${s.pose.roll_deg.toFixed(2)} · perspective ${s.pose.perspective.toFixed(3)} · zoom ${s.zoom.toFixed(2)}</div><pre></pre><div class="snap-actions"><button data-action="restore">Restore</button><button data-action="copy">Copy TXT</button><button data-action="txt">Download TXT</button><button data-action="json">Download JSON</button><button data-action="delete">Delete</button></div>`;section.querySelector('pre').textContent=s.text;section.querySelectorAll('button').forEach(b=>b.onclick=()=>snapshotAction(i,b.dataset.action));snapshotList.appendChild(section)})}
function snapshotAction(index,action){const s=snapshots[index],base=`${safeName(s.family)}-${s.candidateId}-view-${index+1}`;if(action==='restore'){select(s.candidateIndex);currentPose=clonePose(s.pose);zoom=s.zoom;pauseForInteraction();renderCurrent();snapshotPanel.classList.remove('open')}else if(action==='copy')copyText(s.text);else if(action==='txt')download(base+'.txt',s.text+'\n');else if(action==='json')download(base+'.json',JSON.stringify({family:s.family,candidate_id:s.candidateId,seed:s.seed,camera:s.pose,zoom:s.zoom,terminal:s.text},null,2),'application/json');else if(action==='delete'){snapshots.splice(index,1);renderSnapshots()}}
document.getElementById('saveView').onclick=()=>{pauseForInteraction();const c=candidates[selected],text=terminalProjection(c,currentPose,true);snapshots.push({candidateIndex:selected,candidateId:c.id,family:c.family,seed:c.seed,pose:clonePose(currentPose),zoom,text});renderSnapshots();snapshotPanel.classList.add('open')};document.getElementById('openSnapshots').onclick=()=>snapshotPanel.classList.add('open');document.getElementById('closeSnapshots').onclick=()=>snapshotPanel.classList.remove('open');
renderSnapshots();
if(viewMode==='terminal'){viewport.classList.add('terminal-mode');document.getElementById('viewTerm').classList.add('active');document.getElementById('view3d').classList.remove('active');playing=false}
if(candidates.length){select(selected);playButton.textContent=playing?'Pause':'Play';resize();renderCurrent();requestAnimationFrame(tick)}else{viewport.innerHTML='<p>No candidates survived. Inspect manifest.json.</p>'}
</script>
</body>
</html>'''
