#!/usr/bin/env python3
"""Build a self-contained, dependency-free wrist-motion dashboard.

Usage:
    python3 visualize_wrist_motion.py
    python3 visualize_wrist_motion.py path/to/recording.csv -o dashboard.html

The generated HTML embeds one recording for immediate viewing and also accepts
any other compatible CSV through the browser's file picker.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


SENSOR_COLUMNS = [
    "wristMotion_rotationRateX", "wristMotion_rotationRateY", "wristMotion_rotationRateZ",
    "wristMotion_gravityX", "wristMotion_gravityY", "wristMotion_gravityZ",
    "wristMotion_accelerationX", "wristMotion_accelerationY", "wristMotion_accelerationZ",
    "wristMotion_quaternionW", "wristMotion_quaternionX", "wristMotion_quaternionY",
    "wristMotion_quaternionZ",
]
REQUIRED_COLUMNS = ["secondsElapsed", *SENSOR_COLUMNS, "weight", "set", "reps", "activity", "activityEncoded"]


def number(value: str | None) -> float | None:
    try:
        result = float(value) if value not in (None, "") else None
        return result if result is None or math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def load_recording(path: Path, max_points: int) -> dict:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no data rows")
    missing = [column for column in REQUIRED_COLUMNS if column not in rows[0]]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    valid = [row for row in rows if number(row.get("secondsElapsed")) is not None]
    stride = max(1, math.ceil(len(valid) / max_points))
    sampled = valid[::stride]
    last = valid[-1]
    if sampled and sampled[-1] is not last:
        sampled.append(last)

    columns = ["secondsElapsed", *SENSOR_COLUMNS]
    data = {column: [number(row.get(column)) for row in sampled] for column in columns}
    meta_row = next((row for row in valid if row.get("activity")), valid[0])
    return {
        "source": path.name,
        "originalRows": len(rows),
        "plottedRows": len(sampled),
        "sampleStride": stride,
        "meta": {key: meta_row.get(key, "") for key in ("activity", "activityEncoded", "weight", "set", "reps")},
        "data": data,
    }


HTML_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Wrist Motion Explorer</title>
<style>
:root{--bg:#08111f;--panel:#101d2e;--panel2:#14243a;--text:#e8f0fb;--muted:#8fa3bc;--grid:#29405b;--cyan:#45d7da;--orange:#ffad5a;--violet:#a98bff;--green:#67d391;--red:#ff7082}
*{box-sizing:border-box} body{margin:0;background:radial-gradient(circle at top right,#17314d 0,#08111f 38%);color:var(--text);font:14px/1.45 Inter,ui-sans-serif,system-ui,sans-serif;min-height:100vh}
.shell{max-width:1440px;margin:auto;padding:28px}.top{display:flex;justify-content:space-between;gap:24px;align-items:flex-end;margin-bottom:22px}.eyebrow{color:var(--cyan);font-size:12px;font-weight:700;letter-spacing:.15em;text-transform:uppercase}.title{font-size:clamp(28px,4vw,46px);margin:4px 0 3px;letter-spacing:-.04em}.sub{color:var(--muted);margin:0}.picker{display:flex;align-items:center;gap:10px;background:#0d1929;border:1px solid #28405c;border-radius:12px;padding:10px 13px;white-space:nowrap}.picker input{max-width:220px}.cards{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin-bottom:18px}.card,.chart-card,.note{background:linear-gradient(145deg,rgba(20,36,58,.94),rgba(13,25,41,.94));border:1px solid #253b55;border-radius:16px;box-shadow:0 14px 36px rgba(0,0,0,.2)}.card{padding:14px 16px}.label{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.09em}.value{font-size:21px;font-weight:750;margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.chart-card{padding:15px}.chart-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px}.chart-head h2{font-size:16px;margin:0}.unit{color:var(--muted);font-size:12px}.plot{width:100%;height:260px;display:block}.legend{display:flex;gap:16px;margin-top:8px;color:var(--muted);font-size:12px}.legend span:before{content:'';display:inline-block;width:10px;height:3px;border-radius:2px;background:var(--c);vertical-align:middle;margin-right:6px}.note{margin-top:16px;padding:14px 17px;color:var(--muted)}.note strong{color:var(--text)}
@media(max-width:900px){.top{align-items:flex-start;flex-direction:column}.cards{grid-template-columns:repeat(3,1fr)}.grid{grid-template-columns:1fr}}@media(max-width:540px){.shell{padding:18px}.cards{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body><main class="shell">
<header class="top"><div><div class="eyebrow">Sensor analytics</div><h1 class="title">Wrist Motion Explorer</h1><p class="sub" id="sourceLine"></p></div><label class="picker">Open another CSV <input type="file" id="fileInput" accept=".csv,text/csv"></label></header>
<section class="cards" id="cards"></section>
<section class="grid">
  <article class="chart-card"><div class="chart-head"><h2>Rotation rate</h2><span class="unit">x · y · z</span></div><canvas class="plot" id="rotation"></canvas><div class="legend"><span style="--c:var(--cyan)">X</span><span style="--c:var(--orange)">Y</span><span style="--c:var(--violet)">Z</span></div></article>
  <article class="chart-card"><div class="chart-head"><h2>Gravity vector</h2><span class="unit">x · y · z</span></div><canvas class="plot" id="gravity"></canvas><div class="legend"><span style="--c:var(--cyan)">X</span><span style="--c:var(--orange)">Y</span><span style="--c:var(--violet)">Z</span></div></article>
  <article class="chart-card"><div class="chart-head"><h2>User acceleration</h2><span class="unit">x · y · z</span></div><canvas class="plot" id="acceleration"></canvas><div class="legend"><span style="--c:var(--cyan)">X</span><span style="--c:var(--orange)">Y</span><span style="--c:var(--violet)">Z</span></div></article>
  <article class="chart-card"><div class="chart-head"><h2>Orientation quaternion</h2><span class="unit">w · x · y · z</span></div><canvas class="plot" id="quaternion"></canvas><div class="legend"><span style="--c:var(--green)">W</span><span style="--c:var(--cyan)">X</span><span style="--c:var(--orange)">Y</span><span style="--c:var(--violet)">Z</span></div></article>
</section>
<div class="note"><strong>Reading the plots:</strong> all panels share <code>secondsElapsed</code> on the horizontal axis. Gaps represent missing sensor values. Large files are evenly downsampled for display; the first and final observations are retained.</div>
</main>
<script>
const embedded = __DATA__;
const groups={
 rotation:['wristMotion_rotationRateX','wristMotion_rotationRateY','wristMotion_rotationRateZ'],
 gravity:['wristMotion_gravityX','wristMotion_gravityY','wristMotion_gravityZ'],
 acceleration:['wristMotion_accelerationX','wristMotion_accelerationY','wristMotion_accelerationZ'],
 quaternion:['wristMotion_quaternionW','wristMotion_quaternionX','wristMotion_quaternionY','wristMotion_quaternionZ']};
const colors=['#45d7da','#ffad5a','#a98bff'], qcolors=['#67d391','#45d7da','#ffad5a','#a98bff'];
let current=embedded;
const fmt=v=>v===''||v==null?'—':String(v).replace(/\.0$/,'');
function cards(d){const m=d.meta||{};const duration=(d.data.secondsElapsed.filter(Number.isFinite).at(-1)||0).toFixed(2)+' s';const items=[['Activity',fmt(m.activity)],['Encoded',fmt(m.activityEncoded)],['Weight',fmt(m.weight)],['Set',fmt(m.set)],['Reps',fmt(m.reps)],['Duration',duration]],root=document.querySelector('#cards');root.replaceChildren();for(const [label,value] of items){const card=document.createElement('div'),a=document.createElement('div'),b=document.createElement('div');card.className='card';a.className='label';b.className='value';a.textContent=label;b.textContent=value;card.append(a,b);root.append(card)}document.querySelector('#sourceLine').textContent=`${d.source} · ${d.originalRows.toLocaleString()} rows · ${d.plottedRows.toLocaleString()} plotted`;}
function draw(canvas, time, series, palette){const ratio=devicePixelRatio||1,rect=canvas.getBoundingClientRect();canvas.width=rect.width*ratio;canvas.height=rect.height*ratio;const c=canvas.getContext('2d');c.scale(ratio,ratio);const W=rect.width,H=rect.height,p={l:52,r:16,t:12,b:31};let vals=series.flat().filter(Number.isFinite);if(!vals.length){c.fillStyle='#8fa3bc';c.fillText('No numeric sensor data',p.l,p.t+20);return}let lo=Math.min(...vals),hi=Math.max(...vals);if(lo===hi){lo-=1;hi+=1}const pad=(hi-lo)*.08;lo-=pad;hi+=pad;const t0=time.find(Number.isFinite)||0,t1=[...time].reverse().find(Number.isFinite)||1;const x=v=>p.l+(v-t0)/(t1-t0||1)*(W-p.l-p.r),y=v=>p.t+(hi-v)/(hi-lo)*(H-p.t-p.b);c.strokeStyle='#29405b';c.fillStyle='#8fa3bc';c.font='11px system-ui';c.lineWidth=1;for(let i=0;i<=4;i++){let yy=p.t+i*(H-p.t-p.b)/4;c.beginPath();c.moveTo(p.l,yy);c.lineTo(W-p.r,yy);c.stroke();let val=hi-i*(hi-lo)/4;c.fillText(val.toFixed(Math.abs(hi-lo)<2?2:1),4,yy+4)}for(let i=0;i<=5;i++){let v=t0+i*(t1-t0)/5,xx=x(v);c.beginPath();c.moveTo(xx,p.t);c.lineTo(xx,H-p.b);c.stroke();c.fillText(v.toFixed(1),xx-9,H-9)}series.forEach((arr,j)=>{c.strokeStyle=palette[j];c.lineWidth=1.5;c.beginPath();let open=false;for(let i=0;i<arr.length;i++){const v=arr[i],t=time[i];if(!Number.isFinite(v)||!Number.isFinite(t)){open=false;continue}if(!open){c.moveTo(x(t),y(v));open=true}else c.lineTo(x(t),y(v))}c.stroke()});}
function render(d){current=d;cards(d);for(const [id,cols] of Object.entries(groups)){draw(document.querySelector('#'+id),d.data.secondsElapsed,cols.map(k=>d.data[k]||[]),id==='quaternion'?qcolors:colors)}}
function parseCSV(text,name){const lines=text.trim().split(/\r?\n/);const split=line=>{let out=[],s='',q=false;for(let i=0;i<line.length;i++){let ch=line[i];if(ch==='"'){if(q&&line[i+1]==='"'){s+='"';i++}else q=!q}else if(ch===','&&!q){out.push(s);s=''}else s+=ch}out.push(s);return out};const header=split(lines[0]),idx=Object.fromEntries(header.map((h,i)=>[h,i]));const needed=['secondsElapsed',...Object.values(groups).flat()];for(const col of needed)if(!(col in idx))throw new Error('Missing column: '+col);const raw=lines.slice(1).map(split),stride=Math.max(1,Math.ceil(raw.length/4000)),rows=raw.filter((_,i)=>i%stride===0);const data={};for(const col of needed)data[col]=rows.map(r=>{const v=Number(r[idx[col]]);return r[idx[col]]===''||!Number.isFinite(v)?null:v});const first=raw.find(r=>r[idx.activity])||raw[0];return{source:name,originalRows:raw.length,plottedRows:rows.length,sampleStride:stride,meta:Object.fromEntries(['activity','activityEncoded','weight','set','reps'].map(k=>[k,idx[k]==null?'':first[idx[k]]])),data};}
document.querySelector('#fileInput').addEventListener('change',async e=>{const f=e.target.files[0];if(!f)return;try{render(parseCSV(await f.text(),f.name))}catch(err){alert(err.message)}});addEventListener('resize',()=>render(current));render(embedded);
</script></body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path, help="CSV recording to embed")
    parser.add_argument("-o", "--output", type=Path, default=Path("wrist_motion_dashboard.html"))
    parser.add_argument("--max-points", type=int, default=4000, help="maximum embedded plot points")
    args = parser.parse_args()

    csv_path = args.csv
    if csv_path is None:
        candidates = sorted(Path("Whales1and2_Raw_Labelled").glob("*.csv"))
        if not candidates:
            parser.error("no CSV supplied and no CSVs found in Whales1and2_Raw_Labelled")
        csv_path = candidates[0]
    if args.max_points < 100:
        parser.error("--max-points must be at least 100")

    payload = load_recording(csv_path, args.max_points)
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(payload, separators=(",", ":"), allow_nan=False))
    args.output.write_text(html, encoding="utf-8")
    print(f"Created {args.output} from {csv_path} ({payload['plottedRows']:,} plotted points)")


if __name__ == "__main__":
    main()
