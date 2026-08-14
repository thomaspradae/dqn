#!/usr/bin/env python3
"""Build a browser laboratory for ensemble transients and curve construction."""

from __future__ import annotations

import json
from pathlib import Path

from native3d.systems import FAMILIES


def catalog() -> list[dict[str, object]]:
    systems: list[dict[str, object]] = [
        {
            "id": "dejong2d_classic",
            "label": "De Jong 2D - classic",
            "kind": "map2d",
            "protocol": "ensemble transient",
            "params": {"a": 1.4, "b": -2.3, "c": 2.4, "d": -2.1},
            "extent": 2.0,
            "view_scale": 2.15,
            "recurrent_at": 22,
        },
        {
            "id": "dejong2d_poormans",
            "label": "De Jong 2D - poormans #009",
            "kind": "map2d",
            "protocol": "ensemble transient",
            "params": {"a": 1.150414, "b": -2.359882, "c": 2.223958, "d": -1.900946},
            "extent": 2.0,
            "view_scale": 2.15,
            "recurrent_at": 24,
        },
        {
            "id": "clifford2d",
            "label": "Clifford 2D",
            "kind": "map2d",
            "protocol": "ensemble transient",
            "params": {"a": -1.4, "b": 1.6, "c": 1.0, "d": 0.7},
            "extent": 2.0,
            "view_scale": 2.4,
            "recurrent_at": 24,
        },
    ]

    settings = {
        "dejong3d_cyclic": ("map3d", 2.1, 2.35, 24),
        "clifford3d_cyclic": ("map3d", 2.1, 2.6, 24),
        "trig3d_symmetric": ("map3d", 2.1, 2.35, 24),
        "aizawa": ("ode3d", 0.85, 1.55, 90),
        "thomas": ("ode3d", 1.5, 5.0, 130),
        "halvorsen": ("ode3d", 2.5, 14.0, 100),
        "dadras": ("ode3d", 1.2, 6.5, 110),
        "arneodo": ("ode3d", 0.8, 5.5, 120),
        "lissajous3d": ("curve3d", 1.0, 1.35, 100),
        "fourier3d": ("curve3d", 1.0, 1.8, 100),
    }
    for name, (kind, extent, view_scale, recurrent_at) in settings.items():
        family = FAMILIES[name]
        systems.append(
            {
                "id": name,
                "label": name.replace("_", " "),
                "kind": kind,
                "protocol": "parametric construction" if kind == "curve3d" else "ensemble transient",
                "params": family.defaults,
                "initial": list(family.initial),
                "dt": family.dt,
                "extent": extent,
                "view_scale": view_scale,
                "recurrent_at": recurrent_at,
            }
        )
    return systems


def write_lab(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"systems": catalog()}, separators=(",", ":"))
    page = PAGE.replace("__DATA__", payload)
    path = out / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


PAGE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Attractor transient laboratory</title>
<style>
:root{color-scheme:dark;--bg:#030405;--panel:#0b0e12;--line:#272d35;--text:#e8edf2;--muted:#8793a1;--accent:#d8ff4f}
*{box-sizing:border-box;scrollbar-color:#303842 #0b0e12;scrollbar-width:thin}body{margin:0;background:var(--bg);color:var(--text);font:13px/1.4 ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:0;overflow:hidden}
button,select,input{font:inherit}.app{height:100vh;display:grid;grid-template-rows:auto 1fr}.toolbar{display:flex;align-items:center;gap:7px;padding:9px 12px;background:var(--panel);border-bottom:1px solid var(--line);overflow-x:auto}.toolbar button,.toolbar select{border:1px solid #313a45;background:#111720;color:var(--text);border-radius:5px;padding:5px 8px;white-space:nowrap}.toolbar button{cursor:pointer}.toolbar button:hover{border-color:#687585}.toolbar label{display:flex;align-items:center;gap:6px;color:var(--muted);white-space:nowrap}.toolbar input[type=range]{width:90px}.spacer{flex:1}.stage{position:relative;min-height:0;cursor:grab;touch-action:none;user-select:none}.stage.dragging{cursor:grabbing}canvas{display:block;width:100%;height:100%;background:#000}.readout{position:absolute;left:14px;top:12px;pointer-events:none}.readout strong{display:block;color:#fff;font-size:14px}.readout span{display:block;color:var(--muted);margin-top:2px}.state{color:var(--accent)!important}.help{position:absolute;left:14px;bottom:12px;color:#7f8a96;background:#020304cc;padding:4px 6px;pointer-events:none}.equation{position:absolute;right:14px;bottom:12px;max-width:min(440px,55vw);text-align:right;color:#7f8a96;pointer-events:none}.recording{color:#ff6b6b!important;border-color:#ff6b6b!important}
@media(max-width:720px){.readout{top:8px;left:8px}.equation{display:none}.help{left:8px;bottom:8px}.toolbar input[type=range]{width:70px}}
</style>
</head>
<body>
<main class="app">
  <div class="toolbar">
    <select id="system" title="Mathematical system"></select>
    <select id="seed" title="Initial ensemble"><option value="sheet">sheet</option><option value="line">line</option><option value="cloud">cloud</option></select>
    <select id="density" title="Particle count"><option value="20000">20k points</option><option value="50000" selected>50k points</option><option value="100000">100k points</option></select>
    <button id="play">Pause</button><button id="step">Step</button><button id="restart">Restart</button>
    <label>speed <input id="speed" type="range" min="1" max="20" value="10"><span id="speedValue">10</span>/s</label>
    <label>exposure <input id="exposure" type="range" min="5" max="60" value="32"><span id="exposureValue">32</span>%</label>
    <label><input id="fit" type="checkbox" checked> auto fit</label>
    <span class="spacer"></span>
    <button id="camera">Reset camera</button><button id="png">PNG</button><button id="record">Record WebM</button>
  </div>
  <section class="stage" id="stage">
    <canvas id="canvas"></canvas>
    <div class="readout"><strong id="title"></strong><span id="protocol"></span><span id="counter"></span><span class="state" id="state"></span></div>
    <div class="help">drag: orbit · Shift+drag: roll · wheel: zoom</div>
    <div class="equation" id="equation"></div>
  </section>
</main>
<script>
const DATA=__DATA__,systems=DATA.systems,byId=Object.fromEntries(systems.map(s=>[s.id,s]));
const canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d',{alpha:false}),stage=document.getElementById('stage'),systemSelect=document.getElementById('system'),seedSelect=document.getElementById('seed'),densitySelect=document.getElementById('density'),playButton=document.getElementById('play'),speed=document.getElementById('speed'),exposure=document.getElementById('exposure');
const query=new URLSearchParams(location.search);
let system=null,x=null,y=null,z=null,count=0,iteration=0,playing=query.get('still')!=='1',lastTime=performance.now(),accumulator=0,pitch=-18,yaw=24,roll=0,zoom=1,dragging=false,lastPointer=null,curveProgress=0,recorder=null,recordChunks=[];
const equations={
dejong2d_classic:'x′ = sin(a y) - cos(b x) · y′ = sin(c x) - cos(d y)',dejong2d_poormans:'x′ = sin(a y) - cos(b x) · y′ = sin(c x) - cos(d y)',clifford2d:'x′ = sin(a y) + c cos(a x) · y′ = sin(b x) + d cos(b y)',
dejong3d_cyclic:'cyclic De Jong map: x←(y,z), y←(z,x), z←(x,y)',clifford3d_cyclic:'cyclic Clifford map: x←(y,z), y←(z,x), z←(x,y)',trig3d_symmetric:'symmetric cyclic trigonometric map',
aizawa:'Aizawa continuous flow · RK4 ensemble',thomas:'Thomas cyclic flow · RK4 ensemble',halvorsen:'Halvorsen three-lobed flow · RK4 ensemble',dadras:'Dadras flow · RK4 ensemble',arneodo:'Arneodo flow · RK4 ensemble',lissajous3d:'3D Lissajous parametric construction',fourier3d:'two-harmonic 3D Fourier construction'};
for(const s of systems){const o=document.createElement('option');o.value=s.id;o.textContent=s.label;systemSelect.appendChild(o)}
systemSelect.value=byId[query.get('system')]?query.get('system'):'dejong2d_classic';
if(['line','sheet','cloud'].includes(query.get('seed')))seedSelect.value=query.get('seed');
if(['20000','50000','100000'].includes(query.get('density')))densitySelect.value=query.get('density');
function rngFactory(seed){let a=seed>>>0;return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
function initialize(){system=byId[systemSelect.value];count=Number(densitySelect.value);x=new Float32Array(count);y=new Float32Array(count);z=new Float32Array(count);iteration=0;curveProgress=0;const mode=seedSelect.value,rng=rngFactory(0x504f4f52),extent=system.extent,center=system.initial||[0,0,0],side=Math.ceil(Math.sqrt(count));for(let i=0;i<count;i++){const u=(i%side)/(side-1||1),v=Math.floor(i/side)/(side-1||1);if(mode==='line'){x[i]=center[0]+extent*(2*u-1);y[i]=center[1];z[i]=center[2]}else if(mode==='cloud'){x[i]=center[0]+extent*(2*rng()-1);y[i]=center[1]+extent*(2*rng()-1);z[i]=(system.kind==='map2d'?0:center[2]+extent*(2*rng()-1))}else{x[i]=center[0]+extent*(2*u-1);y[i]=center[1]+extent*(2*v-1);z[i]=center[2]}}if(system.kind==='curve3d')constructCurve(0);updateReadout();render()}
function mapStep(){const p=system.params,id=system.id;for(let i=0;i<count;i++){const X=x[i],Y=y[i],Z=z[i];let nx,ny,nz=0;if(id.startsWith('dejong2d')){nx=Math.sin(p.a*Y)-Math.cos(p.b*X);ny=Math.sin(p.c*X)-Math.cos(p.d*Y)}else if(id==='clifford2d'){nx=Math.sin(p.a*Y)+p.c*Math.cos(p.a*X);ny=Math.sin(p.b*X)+p.d*Math.cos(p.b*Y)}else if(id==='dejong3d_cyclic'){nx=Math.sin(p.a*Y)-Math.cos(p.b*Z);ny=Math.sin(p.c*Z)-Math.cos(p.d*X);nz=Math.sin(p.e*X)-Math.cos(p.f*Y)}else if(id==='clifford3d_cyclic'){nx=Math.sin(p.a*Y)+p.c*Math.cos(p.a*Z);ny=Math.sin(p.b*Z)+p.d*Math.cos(p.b*X);nz=Math.sin(p.e*X)+p.f*Math.cos(p.e*Y)}else{nx=Math.sin(p.a*Y)+Math.cos(p.b*Z);ny=Math.sin(p.c*Z)+Math.cos(p.d*X);nz=Math.sin(p.e*X)+Math.cos(p.f*Y)}x[i]=nx;y[i]=ny;z[i]=nz}iteration++}
function derivative(id,X,Y,Z,p){if(id==='aizawa')return[(Z-p.b)*X-p.d*Y,p.d*X+(Z-p.b)*Y,p.c+p.a*Z-Z*Z*Z/3-(X*X+Y*Y)*(1+p.e*Z)+p.f*Z*X*X*X];if(id==='thomas')return[Math.sin(Y)-p.b*X,Math.sin(Z)-p.b*Y,Math.sin(X)-p.b*Z];if(id==='halvorsen')return[-p.a*X-4*Y-4*Z-Y*Y,-p.a*Y-4*Z-4*X-Z*Z,-p.a*Z-4*X-4*Y-X*X];if(id==='dadras')return[Y-p.a*X+p.b*Y*Z,p.c*Y-X*Z+Z,p.d*X*Y-p.e*Z];return[Y,Z,-p.a*X-p.b*Y-Z+p.c*X*X*X]}
function odeStep(){const p=system.params,dt=system.dt,substeps=system.id==='thomas'?4:2;for(let sub=0;sub<substeps;sub++)for(let i=0;i<count;i++){const X=x[i],Y=y[i],Z=z[i],k1=derivative(system.id,X,Y,Z,p),k2=derivative(system.id,X+dt*k1[0]/2,Y+dt*k1[1]/2,Z+dt*k1[2]/2,p),k3=derivative(system.id,X+dt*k2[0]/2,Y+dt*k2[1]/2,Z+dt*k2[2]/2,p),k4=derivative(system.id,X+dt*k3[0],Y+dt*k3[1],Z+dt*k3[2],p);x[i]=X+dt*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6;y[i]=Y+dt*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6;z[i]=Z+dt*(k1[2]+2*k2[2]+2*k3[2]+k4[2])/6}iteration++}
function constructCurve(progress){const p=system.params,e=progress*progress*(3-2*progress);for(let i=0;i<count;i++){const t=Math.PI*2*i/count,line=2*i/(count-1)-1;let tx,ty,tz;if(system.id==='lissajous3d'){tx=p.ax_amp*Math.sin(Math.round(p.ax_freq)*t+p.ax_phase);ty=p.ay_amp*Math.sin(Math.round(p.ay_freq)*t+p.ay_phase);tz=p.az_amp*Math.sin(Math.round(p.az_freq)*t+p.az_phase)}else{const axis=a=>p[`${a}1_amp`]*Math.sin(Math.round(p[`${a}1_freq`])*t+p[`${a}1_phase`])+p[`${a}2_amp`]*Math.sin(Math.round(p[`${a}2_freq`])*t+p[`${a}2_phase`]);tx=axis('x');ty=axis('y');tz=axis('z')}x[i]=(1-e)*line+e*tx;y[i]=e*ty;z[i]=e*tz}iteration=Math.round(progress*system.recurrent_at)}
function advance(){if(system.kind.startsWith('map'))mapStep();else if(system.kind==='ode3d')odeStep();else{curveProgress=Math.min(1,curveProgress+.0125);constructCurve(curveProgress)}updateReadout();render()}
function matrix(){const rx=pitch*Math.PI/180,ry=yaw*Math.PI/180,rz=roll*Math.PI/180,cx=Math.cos(rx),sx=Math.sin(rx),cy=Math.cos(ry),sy=Math.sin(ry),cz=Math.cos(rz),sz=Math.sin(rz);return[[cz*cy,cz*sy*sx-sz*cx,cz*sy*cx+sz*sx],[sz*cy,sz*sy*sx+cz*cx,sz*sy*cx-cz*sx],[-sy,cy*sx,cy*cx]]}
function resize(){const r=canvas.getBoundingClientRect(),d=Math.min(devicePixelRatio||1,1.5);canvas.width=Math.max(1,Math.floor(r.width*d));canvas.height=Math.max(1,Math.floor(r.height*d));ctx.setTransform(d,0,0,d,0,0)}
new ResizeObserver(()=>{resize();render()}).observe(canvas);
function bounds(){if(!document.getElementById('fit').checked)return{cx:0,cy:0,scale:system.view_scale};let xs=[],ys=[],step=Math.max(1,Math.floor(count/3000)),m=matrix(),is2d=system.kind==='map2d';for(let i=0;i<count;i+=step){const X=x[i],Y=y[i],Z=z[i],px=is2d?X:m[0][0]*X+m[0][1]*Y+m[0][2]*Z,py=is2d?Y:m[1][0]*X+m[1][1]*Y+m[1][2]*Z;if(Number.isFinite(px)&&Number.isFinite(py)){xs.push(px);ys.push(py)}}xs.sort((a,b)=>a-b);ys.sort((a,b)=>a-b);if(!xs.length)return{cx:0,cy:0,scale:system.view_scale};const lo=Math.floor(xs.length*.005),hi=Math.ceil(xs.length*.995)-1,minX=xs[lo],maxX=xs[hi],minY=ys[lo],maxY=ys[hi];return{cx:(minX+maxX)/2,cy:(minY+maxY)/2,scale:Math.max(maxX-minX,maxY-minY)/2/.88}}
function render(){if(!system||!canvas.clientWidth)return;const w=canvas.clientWidth,h=canvas.clientHeight,m=matrix(),b=bounds(),scale=Math.min(w,h)*.45/Math.max(b.scale,1e-6)*zoom,alpha=Math.min(.8,Number(exposure.value)/100*Math.sqrt(50000/count));ctx.clearRect(0,0,w,h);ctx.fillStyle='#000';ctx.fillRect(0,0,w,h);ctx.globalCompositeOperation='lighter';ctx.fillStyle=`rgba(235,238,242,${alpha})`;const is2d=system.kind==='map2d';for(let i=0;i<count;i++){const X=x[i],Y=y[i],Z=z[i];if(!Number.isFinite(X)||!Number.isFinite(Y)||!Number.isFinite(Z))continue;const px=is2d?X:m[0][0]*X+m[0][1]*Y+m[0][2]*Z,py=is2d?Y:m[1][0]*X+m[1][1]*Y+m[1][2]*Z;const sx=w/2+(px-b.cx)*scale,sy=h/2-(py-b.cy)*scale;if(sx>=0&&sx<w&&sy>=0&&sy<h)ctx.fillRect(sx,sy,1,1)}ctx.globalCompositeOperation='source-over'}
function updateReadout(){document.getElementById('title').textContent=system.label;document.getElementById('protocol').textContent=system.protocol;document.getElementById('counter').textContent=`step ${iteration} · ${count.toLocaleString()} particles`;const recurrent=iteration>=system.recurrent_at||system.kind==='curve3d'&&curveProgress>=1;document.getElementById('state').textContent=recurrent?(system.kind==='curve3d'?'constructed':'recurrent regime'):'transient regime';document.getElementById('equation').textContent=equations[system.id]||system.id}
function tick(now){const dt=Math.min((now-lastTime)/1000,.1);lastTime=now;if(playing){accumulator+=dt;const interval=1/Number(speed.value);while(accumulator>=interval){advance();accumulator-=interval}}requestAnimationFrame(tick)}
function pause(){playing=false;playButton.textContent='Play'}playButton.onclick=()=>{playing=!playing;playButton.textContent=playing?'Pause':'Play'};document.getElementById('step').onclick=()=>{pause();advance()};document.getElementById('restart').onclick=()=>initialize();systemSelect.onchange=initialize;seedSelect.onchange=initialize;densitySelect.onchange=initialize;speed.oninput=()=>document.getElementById('speedValue').textContent=speed.value;exposure.oninput=()=>{document.getElementById('exposureValue').textContent=exposure.value;render()};document.getElementById('fit').onchange=render;
stage.addEventListener('pointerdown',e=>{if(system.kind==='map2d')return;dragging=true;lastPointer={x:e.clientX,y:e.clientY};stage.classList.add('dragging');stage.setPointerCapture(e.pointerId);pause()});stage.addEventListener('pointermove',e=>{if(!dragging)return;const dx=e.clientX-lastPointer.x,dy=e.clientY-lastPointer.y;lastPointer={x:e.clientX,y:e.clientY};if(e.shiftKey)roll+=dx*.35;else{yaw+=dx*.35;pitch=Math.max(-89,Math.min(89,pitch-dy*.35))}render()});function endDrag(e){dragging=false;stage.classList.remove('dragging');if(stage.hasPointerCapture(e.pointerId))stage.releasePointerCapture(e.pointerId)}stage.addEventListener('pointerup',endDrag);stage.addEventListener('pointercancel',endDrag);stage.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.35,Math.min(3,zoom*Math.exp(-e.deltaY*.0012)));pause();render()},{passive:false});
document.getElementById('camera').onclick=()=>{pitch=-18;yaw=24;roll=0;zoom=1;render()};document.getElementById('png').onclick=()=>{const a=document.createElement('a');a.download=`${system.id}-step-${iteration}.png`;a.href=canvas.toDataURL('image/png');a.click()};document.getElementById('record').onclick=()=>{const button=document.getElementById('record');if(recorder&&recorder.state==='recording'){recorder.stop();return}recordChunks=[];const stream=canvas.captureStream(30),mime=MediaRecorder.isTypeSupported('video/webm;codecs=vp9')?'video/webm;codecs=vp9':'video/webm';recorder=new MediaRecorder(stream,{mimeType:mime});recorder.ondataavailable=e=>{if(e.data.size)recordChunks.push(e.data)};recorder.onstop=()=>{const blob=new Blob(recordChunks,{type:'video/webm'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`${system.id}-transient.webm`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);button.textContent='Record WebM';button.classList.remove('recording')};recorder.start();button.textContent='Stop recording';button.classList.add('recording')};
initialize();const requestedStep=Math.max(0,Math.min(300,Number(query.get('step')||0)));for(let i=0;i<requestedStep;i++)advance();playButton.textContent=playing?'Pause':'Play';resize();render();requestAnimationFrame(tick);
</script>
</body>
</html>'''


def main() -> int:
    out = Path("examples/transient_lab")
    path = write_lab(out)
    print(f"saved transient laboratory -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
