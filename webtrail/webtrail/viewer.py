"""`view` subcommand: build a self-contained HTML trajectory browser.

Scans a run's ``trajectories/`` (or a single trajectory) and writes one
``index.html`` you open with ``file://`` — no server, no dependencies. The left
pane lists every trajectory (coloured by status / judge verdict); the right pane
steps through screenshots with the action, the agent's analysis, and the judge's
thought for that run. Screenshots are referenced by relative path, so the file
stays small even for large runs; all step/result text is embedded inline so the
page works offline.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _rel(path: Path, root: Path) -> str | None:
    """Path relative to the run root (where index.html lives), or None if absent."""
    return path.relative_to(root).as_posix() if path.exists() else None


def _steps(traj_dir: Path, root: Path) -> list[dict]:
    steps: list[dict] = []
    for agent_path in sorted((traj_dir / "agent").glob("step_*.json")):
        stem = agent_path.stem                      # step_000
        agent = _read_json(agent_path)
        state = _read_json(traj_dir / "states" / f"{stem}.json")
        args = {k: v for k, v in (agent.get("args") or {}).items() if k != "analysis"}
        steps.append({
            "idx": int(stem.split("_")[1]),
            "url": state.get("url") or "",
            "title": state.get("title") or "",
            "action": agent.get("action"),
            "args": args,
            "analysis": (agent.get("analysis") or "")[:1200],
            "guard": (state.get("guard") or {}).get("kind"),
            "shot": _rel(traj_dir / "screenshots" / f"{stem}.png", root),
            "annotated": _rel(traj_dir / "annotated" / f"{stem}.png", root),
            "model_view": _rel(traj_dir / "model_views" / f"{stem}.png", root),
        })
    return steps


def _trajectory(traj_dir: Path, root: Path) -> dict:
    task = _read_json(traj_dir / "task.json")
    result = _read_json(traj_dir / "result.json")
    judge = _read_json(traj_dir / "judge.json")
    return {
        "id": traj_dir.name,
        "instruction": task.get("instruction") or "",
        "criteria": task.get("criteria") or [],
        "status": result.get("status") or "unknown",
        "block": result.get("block"),
        "steps_taken": result.get("steps_taken"),
        "final_url": result.get("final_url") or "",
        "answer": result.get("stop_answer") or "",
        "judge": judge.get("judge"),
        "judge_thought": judge.get("thought") or "",
        "steps": _steps(traj_dir, root),
    }


def collect_trajectories(run_dir: Path) -> list[dict]:
    if (run_dir / "result.json").exists():          # run_dir is itself a trajectory
        return [_trajectory(run_dir, run_dir)]
    traj_root = run_dir / "trajectories"
    if not traj_root.exists():
        return []
    dirs = sorted(d for d in traj_root.iterdir() if (d / "result.json").exists())
    return [_trajectory(d, run_dir) for d in dirs]


def render_html(run_dir: Path, trajectories: list[dict]) -> str:
    payload = json.dumps(trajectories, ensure_ascii=False).replace("</", "<\\/")
    title = html.escape(run_dir.name or "run")
    return _TEMPLATE.replace("__TITLE__", title).replace("__DATA__", payload)


def build(run_dir: Path, out: Path | None = None) -> Path:
    trajectories = collect_trajectories(run_dir)
    if not trajectories:
        raise SystemExit(f"no trajectories with result.json found under {run_dir}")
    # index.html sits at the run root so the relative image paths resolve; a
    # single-trajectory dir is its own root
    target = out or run_dir / "index.html"
    target.write_text(render_html(run_dir, trajectories), encoding="utf-8")
    return target


# ---------------------------------------------------------------------------

_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>webtrail — __TITLE__</title>
<style>
  :root {
    --bg: #0f1115; --panel: #171a21; --panel2: #1e222b; --line: #2a2f3a;
    --fg: #e6e9ef; --muted: #9aa3b2; --accent: #5b9dff;
    --ok: #35c07d; --fail: #e0524a; --warn: #e0a83a;
  }
  @media (prefers-color-scheme: light) {
    :root { --bg:#f5f6f8; --panel:#fff; --panel2:#eef0f4; --line:#dfe3ea;
            --fg:#1b1f27; --muted:#5a6472; --accent:#2563eb; }
  }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
         background:var(--bg); color:var(--fg); height:100vh; display:flex; overflow:hidden; }
  #side { width:320px; flex:none; border-right:1px solid var(--line);
          background:var(--panel); display:flex; flex-direction:column; }
  #side header { padding:12px 14px; border-bottom:1px solid var(--line); }
  #side h1 { font-size:14px; margin:0 0 8px; font-weight:600; }
  #q { width:100%; padding:6px 8px; border:1px solid var(--line); border-radius:6px;
       background:var(--panel2); color:var(--fg); }
  #list { overflow-y:auto; flex:1; }
  .row { padding:9px 14px; border-bottom:1px solid var(--line); cursor:pointer; }
  .row:hover { background:var(--panel2); }
  .row.sel { background:var(--panel2); border-left:3px solid var(--accent); padding-left:11px; }
  .row .t { font-size:12.5px; margin-bottom:3px; overflow:hidden; text-overflow:ellipsis;
            display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; }
  .row .m { font-size:11px; color:var(--muted); display:flex; gap:6px; align-items:center; }
  .pill { font-size:10px; padding:1px 6px; border-radius:10px; background:var(--panel2);
          border:1px solid var(--line); white-space:nowrap; }
  .pill.ok { color:var(--ok); border-color:var(--ok); }
  .pill.fail { color:#e0524a; border-color:#e0524a; }
  .dot { width:8px; height:8px; border-radius:50%; flex:none; background:var(--muted); }
  .dot.completed { background:var(--ok); } .dot.blocked, .dot.max_steps,
  .dot.stale_loop { background:var(--warn); }
  .dot.internal_error, .dot.env_error, .dot.error { background:#e0524a; }
  #main { flex:1; display:flex; flex-direction:column; overflow:hidden; }
  #head { padding:12px 18px; border-bottom:1px solid var(--line); background:var(--panel); }
  #head .instr { font-weight:600; margin-bottom:6px; }
  #head .meta { font-size:12px; color:var(--muted); display:flex; flex-wrap:wrap; gap:6px 14px; }
  #head a { color:var(--accent); text-decoration:none; }
  #stage { flex:1; display:flex; overflow:hidden; }
  #imgwrap { flex:1; display:flex; align-items:center; justify-content:center;
             background:var(--bg); overflow:auto; padding:14px; }
  #imgwrap img { max-width:100%; max-height:100%; border:1px solid var(--line);
                 border-radius:6px; box-shadow:0 4px 24px rgba(0,0,0,.3); }
  #detail { width:340px; flex:none; border-left:1px solid var(--line); background:var(--panel);
            overflow-y:auto; padding:14px 16px; font-size:13px; }
  #detail h3 { font-size:11px; text-transform:uppercase; letter-spacing:.05em;
               color:var(--muted); margin:16px 0 6px; }
  #detail h3:first-child { margin-top:0; }
  code, pre { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; }
  pre { white-space:pre-wrap; word-break:break-word; margin:0; background:var(--panel2);
        padding:8px 10px; border-radius:6px; }
  .act { font-weight:600; color:var(--accent); }
  #nav { padding:10px 18px; border-top:1px solid var(--line); background:var(--panel);
         display:flex; align-items:center; gap:12px; }
  #nav button { background:var(--panel2); color:var(--fg); border:1px solid var(--line);
                border-radius:6px; padding:5px 12px; cursor:pointer; font-size:13px; }
  #nav button:disabled { opacity:.4; cursor:default; }
  #scrub { flex:1; }
  #stepno { font-variant-numeric:tabular-nums; color:var(--muted); min-width:96px; }
  .toggle { margin-left:auto; font-size:12px; color:var(--muted); }
  .toggle label { cursor:pointer; margin-left:10px; }
  .empty { margin:auto; color:var(--muted); }
</style>
</head>
<body>
<div id="side">
  <header>
    <h1>webtrail · __TITLE__</h1>
    <input id="q" placeholder="filter by text / status / SUCCESS / FAIL">
  </header>
  <div id="list"></div>
</div>
<div id="main">
  <div id="head"><div class="empty">Select a trajectory</div></div>
  <div id="stage">
    <div id="imgwrap"><div class="empty">—</div></div>
    <div id="detail"></div>
  </div>
  <div id="nav">
    <button id="prev">◀ Prev</button>
    <span id="stepno">—</span>
    <input id="scrub" type="range" min="0" max="0" value="0">
    <button id="next">Next ▶</button>
    <span class="toggle">
      <label><input type="radio" name="img" value="annotated" checked> annotated</label>
      <label><input type="radio" name="img" value="shot"> raw</label>
      <label><input type="radio" name="img" value="model_view"> model view</label>
    </span>
  </div>
</div>
<script>
const DATA = __DATA__;
let cur = null, step = 0, imgKind = "annotated";
const esc = s => (s==null?"":String(s)).replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));

function renderList(filter="") {
  const f = filter.toLowerCase();
  const el = document.getElementById("list");
  el.innerHTML = "";
  DATA.forEach((t, i) => {
    const hay = (t.instruction+" "+t.status+" "+(t.judge||"")+" "+t.id).toLowerCase();
    if (f && !hay.includes(f)) return;
    const row = document.createElement("div");
    row.className = "row" + (cur===i ? " sel" : "");
    const judge = t.judge ? `<span class="pill ${t.judge==='SUCCESS'?'ok':'fail'}">${t.judge}</span>` : "";
    row.innerHTML = `<div class="t">${esc(t.instruction)||"(no instruction)"}</div>
      <div class="m"><span class="dot ${esc(t.status)}"></span>${esc(t.status)}
      <span>· ${t.steps.length} steps</span>${judge}</div>`;
    row.onclick = () => select(i);
    el.appendChild(row);
  });
}

function select(i) { cur = i; step = 0; renderList(document.getElementById("q").value); renderHead(); renderStep(); }

function renderHead() {
  const t = DATA[cur];
  const crit = t.criteria.length ? `<span>criteria: ${esc(t.criteria.join("; "))}</span>` : "";
  const ans = t.answer ? `<span>answer: <b>${esc(t.answer)}</b></span>` : "";
  const judge = t.judge ? `<span>judge: <b class="${t.judge==='SUCCESS'?'act':''}">${t.judge}</b></span>` : "";
  document.getElementById("head").innerHTML =
    `<div class="instr">${esc(t.instruction)}</div>
     <div class="meta"><span>${esc(t.id)}</span><span>status: ${esc(t.status)}</span>
     <span>${t.steps.length} steps</span>${judge}${ans}${crit}
     ${t.final_url?`<a href="${esc(t.final_url)}" target="_blank">${esc(t.final_url)}</a>`:""}</div>`;
}

function renderStep() {
  const t = DATA[cur];
  if (!t || !t.steps.length) { document.getElementById("imgwrap").innerHTML='<div class="empty">no steps</div>';
    document.getElementById("detail").innerHTML=""; return; }
  step = Math.max(0, Math.min(step, t.steps.length-1));
  const s = t.steps[step];
  const src = s[imgKind] || s.annotated || s.shot;
  document.getElementById("imgwrap").innerHTML = src ? `<img src="${esc(src)}">`
    : '<div class="empty">no screenshot</div>';
  const args = Object.keys(s.args||{}).length ? `<pre>${esc(JSON.stringify(s.args,null,1))}</pre>` : "";
  const guard = s.guard ? `<h3>guard</h3><pre>${esc(s.guard)}</pre>` : "";
  const jt = (step===t.steps.length-1 && t.judge_thought)
    ? `<h3>judge · ${esc(t.judge)}</h3><pre>${esc(t.judge_thought)}</pre>` : "";
  document.getElementById("detail").innerHTML =
    `<h3>action</h3><div><span class="act">${esc(s.action)||"—"}</span></div>${args}
     <h3>analysis</h3><pre>${esc(s.analysis)||"—"}</pre>
     <h3>page</h3><pre>${esc(s.title)}\n${esc(s.url)}</pre>${guard}${jt}`;
  const sc = document.getElementById("scrub");
  sc.max = t.steps.length-1; sc.value = step;
  document.getElementById("stepno").textContent = `step ${step+1} / ${t.steps.length}`;
  document.getElementById("prev").disabled = step===0;
  document.getElementById("next").disabled = step===t.steps.length-1;
}

document.getElementById("q").oninput = e => renderList(e.target.value);
document.getElementById("prev").onclick = () => { step--; renderStep(); };
document.getElementById("next").onclick = () => { step++; renderStep(); };
document.getElementById("scrub").oninput = e => { step = +e.target.value; renderStep(); };
document.querySelectorAll('input[name=img]').forEach(r =>
  r.onchange = () => { imgKind = r.value; renderStep(); });
document.addEventListener("keydown", e => {
  if (e.target.id === "q") return;
  if (e.key === "ArrowLeft" && cur!==null) { step--; renderStep(); }
  if (e.key === "ArrowRight" && cur!==null) { step++; renderStep(); }
  if (e.key === "ArrowDown" && cur!==null && cur<DATA.length-1) select(cur+1);
  if (e.key === "ArrowUp" && cur!==null && cur>0) select(cur-1);
});
renderList();
if (DATA.length) select(0);
</script>
</body>
</html>
"""


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "view", help="build a self-contained HTML trajectory browser for a run"
    )
    parser.add_argument("--run", required=True,
                        help="run directory (or a single trajectory directory)")
    parser.add_argument("--out", help="output HTML path (default <run>/index.html)")
    parser.set_defaults(handler=main)


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_dir = Path(args.run)
    target = build(run_dir, Path(args.out) if args.out else None)
    trajectories = collect_trajectories(run_dir)
    print(f"wrote {target}  ({len(trajectories)} trajectories)")
    print(f"open it with:  file://{target.resolve()}")
