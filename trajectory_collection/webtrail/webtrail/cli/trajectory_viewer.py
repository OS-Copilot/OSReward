"""Local, read-only web UI for inspecting WebTrail trajectories.

The static ``webtrail view`` export is useful for sharing one finished run.  This
module serves every run below a directory and reads files on demand, which makes
it suitable for following an active collection as new trajectories and steps
arrive.  It intentionally uses only the Python standard library.

Run it from the repository root::

    python -m webtrail.cli.trajectory_viewer --runs-root runs --port 8765
"""

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import re
import time
from collections import Counter
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

logger = logging.getLogger(__name__)

STEP_RE = re.compile(r"^step_(\d+)\.(?:json|png)$")
MEDIA_KINDS = {"screenshots", "annotated", "model_views"}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _step_indexes(traj_dir: Path) -> list[int]:
    indexes: set[int] = set()
    for folder in ("agent", "states", "screenshots", "annotated", "model_views"):
        for path in (traj_dir / folder).glob("step_*.*"):
            match = STEP_RE.match(path.name)
            if match:
                indexes.add(int(match.group(1)))
    return sorted(indexes)


def _status(task: dict, result: dict, step_count: int) -> str:
    if result.get("status"):
        return str(result["status"])
    if step_count or task:
        return "running"
    return "pending"


def _trajectory_summary(traj_dir: Path) -> dict[str, Any]:
    task = _read_json(traj_dir / "task.json")
    result = _read_json(traj_dir / "result.json")
    judge = _read_json(traj_dir / "judge.json")
    indexes = _step_indexes(traj_dir)
    timing = result.get("timing") or {}
    urls = task.get("urls") or []
    updated = max(
        [_mtime(traj_dir / name) for name in ("task.json", "result.json", "judge.json")]
        + [_mtime(p) for folder in ("agent", "states", "screenshots")
           for p in (traj_dir / folder).glob("step_*.*")]
        + [0.0]
    )
    return {
        "id": traj_dir.name,
        "instruction": task.get("instruction") or "",
        "domain": result.get("domain") or task.get("domain") or "",
        "start_url": result.get("start_url") or (urls[0] if urls else ""),
        "final_url": result.get("final_url") or "",
        "status": _status(task, result, len(indexes)),
        "judge": judge.get("judge"),
        "judge_success": judge.get("success"),
        "steps": len(indexes),
        "steps_taken": result.get("steps_taken"),
        "duration_s": timing.get("duration_s"),
        "answer": result.get("stop_answer") or "",
        "error": result.get("error"),
        "updated_at": updated,
    }


def _safe_run(runs_root: Path, run_id: str) -> Path | None:
    if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
        return None
    root = runs_root.resolve()
    candidate = (root / run_id).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    if not candidate.is_dir() or not (candidate / "trajectories").is_dir():
        return None
    return candidate


def _safe_trajectory(run_dir: Path, trajectory_id: str) -> Path | None:
    if not trajectory_id or "/" in trajectory_id or "\\" in trajectory_id:
        return None
    root = (run_dir / "trajectories").resolve()
    candidate = (root / trajectory_id).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate if candidate.is_dir() else None


def list_runs(runs_root: Path) -> list[dict[str, Any]]:
    if not runs_root.is_dir():
        return []
    runs: list[dict[str, Any]] = []
    for run_dir in runs_root.iterdir():
        trajectory_root = run_dir / "trajectories"
        if not run_dir.is_dir() or not trajectory_root.is_dir():
            continue
        trajectories = [
            _trajectory_summary(path)
            for path in trajectory_root.iterdir()
            if path.is_dir()
        ]
        if not trajectories:
            continue
        statuses = Counter(item["status"] for item in trajectories)
        verdicts = Counter((item["judge"] or "unjudged") for item in trajectories)
        config = _read_json(run_dir / "run_config.json")
        model = config.get("model") or {}
        runs.append({
            "id": run_dir.name,
            "total": len(trajectories),
            "statuses": dict(statuses),
            "judges": dict(verdicts),
            "model": model.get("model") or "",
            "provider": model.get("provider") or "auto",
            "updated_at": max((item["updated_at"] for item in trajectories), default=0),
        })
    runs.sort(key=lambda item: (item["updated_at"], item["id"]), reverse=True)
    return runs


def list_trajectories(run_dir: Path) -> dict[str, Any]:
    trajectories = [
        _trajectory_summary(path)
        for path in (run_dir / "trajectories").iterdir()
        if path.is_dir()
    ]
    trajectories.sort(key=lambda item: item["id"])
    config = _read_json(run_dir / "run_config.json")
    return {
        "run": run_dir.name,
        "config": {
            "model": (config.get("model") or {}).get("model") or "",
            "provider": (config.get("model") or {}).get("provider") or "auto",
            "max_steps": (config.get("run") or {}).get("max_steps"),
        },
        "trajectories": trajectories,
    }


def _media_url(run_id: str, trajectory_id: str, folder: str, index: int,
               traj_dir: Path) -> str | None:
    filename = f"step_{index:03d}.png"
    if not (traj_dir / folder / filename).is_file():
        return None
    return "/media/{}/{}/{}/{}".format(
        quote(run_id, safe=""), quote(trajectory_id, safe=""), folder, filename
    )


def trajectory_detail(run_dir: Path, traj_dir: Path) -> dict[str, Any]:
    task = _read_json(traj_dir / "task.json")
    result = _read_json(traj_dir / "result.json")
    judge = _read_json(traj_dir / "judge.json")
    steps = []
    for index in _step_indexes(traj_dir):
        stem = f"step_{index:03d}"
        agent = _read_json(traj_dir / "agent" / f"{stem}.json")
        state = _read_json(traj_dir / "states" / f"{stem}.json")
        steps.append({
            "index": index,
            "images": {
                folder: _media_url(run_dir.name, traj_dir.name, folder, index, traj_dir)
                for folder in MEDIA_KINDS
            },
            "state": state,
            "agent": {
                key: agent.get(key) for key in (
                    "reply", "analysis", "action", "args", "resolved",
                    "commands", "command_results", "usage", "latency_s",
                    "parse_attempts", "sent_image_size", "notices", "error",
                ) if key in agent
            },
        })
    return {
        "run": run_dir.name,
        "id": traj_dir.name,
        "task": task,
        "result": result,
        "judge": judge,
        "steps": steps,
    }


class TrajectoryHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, runs_root: Path):
        super().__init__(address, handler)
        self.runs_root = runs_root.resolve()


class ViewerHandler(BaseHTTPRequestHandler):
    server: TrajectoryHTTPServer

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.info("%s - %s", self.client_address[0], fmt % args)

    def _json(self, value: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, message: str, status: int = HTTPStatus.NOT_FOUND) -> None:
        self._json({"error": message}, status)

    def _file(self, path: Path) -> None:
        try:
            stat = path.stat()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(stat.st_size))
            self.send_header("Cache-Control", "private, max-age=60")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            with path.open("rb") as handle:
                while chunk := handle.read(256 * 1024):
                    self.wfile.write(chunk)
        except (OSError, BrokenPipeError):
            return

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            body = HTML.encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; style-src 'unsafe-inline'; script-src 'unsafe-inline';")
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/health":
            self._json({"ok": True, "runs_root": str(self.server.runs_root), "time": time.time()})
            return
        if path == "/api/runs":
            self._json({"runs": list_runs(self.server.runs_root)})
            return

        parts = [unquote(part) for part in path.split("/") if part]
        if len(parts) == 3 and parts[:2] == ["api", "run"]:
            run_dir = _safe_run(self.server.runs_root, parts[2])
            if not run_dir:
                self._error("run not found")
                return
            self._json(list_trajectories(run_dir))
            return
        if len(parts) == 5 and parts[:2] == ["api", "run"] and parts[3] == "trajectory":
            run_dir = _safe_run(self.server.runs_root, parts[2])
            traj_dir = _safe_trajectory(run_dir, parts[4]) if run_dir else None
            if not run_dir or not traj_dir:
                self._error("trajectory not found")
                return
            self._json(trajectory_detail(run_dir, traj_dir))
            return
        if len(parts) == 5 and parts[0] == "media" and parts[3] in MEDIA_KINDS:
            run_dir = _safe_run(self.server.runs_root, parts[1])
            traj_dir = _safe_trajectory(run_dir, parts[2]) if run_dir else None
            filename = parts[4]
            if (not traj_dir or not STEP_RE.match(filename) or
                    not filename.endswith(".png")):
                self._error("image not found")
                return
            image_path = traj_dir / parts[3] / filename
            if not image_path.is_file():
                self._error("image not found")
                return
            self._file(image_path)
            return
        self._error("not found")


def serve(runs_root: Path, host: str, port: int) -> None:
    server = TrajectoryHTTPServer((host, port), ViewerHandler, runs_root)
    logger.info("Trajectory viewer: http://%s:%d", host, port)
    logger.info("Runs root: %s", server.runs_root)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a local WebTrail trajectory viewer")
    parser.add_argument("--runs-root", default="runs", help="directory containing run folders")
    parser.add_argument("--host", default="127.0.0.1", help="listen host (default: localhost only)")
    parser.add_argument("--port", type=int, default=8765, help="listen port")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = Path(args.runs_root)
    if not root.is_dir():
        raise SystemExit(f"runs root does not exist: {root}")
    serve(root, args.host, args.port)


HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WebTrail Observatory</title>
<style>
:root{--bg:#0a0d12;--panel:#11161e;--panel2:#171e28;--line:#273141;--text:#e9eef7;--muted:#8e9aae;--blue:#79a8ff;--green:#48d597;--red:#ff6b6b;--amber:#f1b958;--purple:#b896ff;--shadow:0 18px 50px #0008}*{box-sizing:border-box}html,body{margin:0;height:100%;overflow:hidden;background:var(--bg);color:var(--text);font:13px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}button,select,input{font:inherit;color:inherit}button,select,input{background:var(--panel2);border:1px solid var(--line);border-radius:8px}button{cursor:pointer;padding:7px 11px}button:hover{border-color:#53657d}button:disabled{opacity:.35;cursor:default}.app{height:100%;display:grid;grid-template-rows:58px 1fr}.top{display:flex;align-items:center;gap:14px;padding:0 18px;border-bottom:1px solid var(--line);background:#0e131bcc;backdrop-filter:blur(14px)}.brand{display:flex;align-items:center;gap:10px;font-weight:720;font-size:15px;white-space:nowrap}.brandmark{width:28px;height:28px;border-radius:8px;background:linear-gradient(135deg,#76a7ff,#9b6cff);box-shadow:0 0 24px #7198ff55;display:grid;place-items:center;color:white}.top select{height:34px;min-width:280px;padding:0 34px 0 10px}.runmeta{display:flex;gap:7px;align-items:center;color:var(--muted);min-width:0}.spacer{flex:1}.refreshing{animation:spin .8s linear infinite}@keyframes spin{to{transform:rotate(360deg)}}.layout{display:grid;grid-template-columns:355px minmax(420px,1fr) 390px;min-height:0}.sidebar,.inspector{background:var(--panel);min-height:0;display:flex;flex-direction:column}.sidebar{border-right:1px solid var(--line)}.inspector{border-left:1px solid var(--line)}.filters{padding:12px;border-bottom:1px solid var(--line);display:grid;grid-template-columns:1fr 110px 100px;gap:8px}.filters input{grid-column:1/-1;padding:8px 10px;width:100%}.filters select{padding:7px}.stats{padding:10px 12px;display:flex;gap:6px;flex-wrap:wrap;border-bottom:1px solid var(--line)}.badge{display:inline-flex;align-items:center;gap:5px;padding:2px 7px;border-radius:999px;background:#ffffff08;border:1px solid var(--line);font-size:11px;white-space:nowrap}.badge.completed,.badge.SUCCESS{color:var(--green);border-color:#48d59766}.badge.FAIL,.badge.env_error,.badge.error{color:var(--red);border-color:#ff6b6b66}.badge.running{color:var(--blue);border-color:#79a8ff66}.badge.stale_loop,.badge.blocked,.badge.max_steps{color:var(--amber);border-color:#f1b95866}.badge.unjudged{color:var(--muted)}.tasklist{overflow:auto;flex:1}.task{padding:11px 13px;border-bottom:1px solid var(--line);cursor:pointer;position:relative}.task:hover{background:#ffffff05}.task.selected{background:#7198ff12}.task.selected:before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--blue)}.task-title{display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;font-size:12.5px;margin-bottom:7px}.task-foot{display:flex;align-items:center;gap:6px;color:var(--muted);font-size:11px}.task-foot .domain{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1}.empty{height:100%;display:grid;place-items:center;color:var(--muted);padding:30px;text-align:center}.workspace{min-width:0;min-height:0;display:grid;grid-template-rows:auto minmax(260px,1fr) 128px 48px}.taskhead{padding:12px 16px;border-bottom:1px solid var(--line);background:var(--panel)}.instruction{font-size:14px;font-weight:620;margin-bottom:8px}.meta{display:flex;gap:7px;align-items:center;flex-wrap:wrap;color:var(--muted);font-size:11px}.meta a{color:var(--blue);text-decoration:none;max-width:45%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.stage{position:relative;overflow:auto;display:grid;place-items:center;padding:14px;background-color:#090c11;background-image:linear-gradient(45deg,#ffffff04 25%,transparent 25%),linear-gradient(-45deg,#ffffff04 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#ffffff04 75%),linear-gradient(-45deg,transparent 75%,#ffffff04 75%);background-size:24px 24px;background-position:0 0,0 12px,12px -12px,-12px 0}.stage.actual{display:block}.image-frame{position:relative;flex:none;line-height:0}.stage.actual .image-frame{margin:auto}.image-frame img{display:block;width:100%;height:100%;object-fit:fill;border:1px solid #354154;border-radius:7px;box-shadow:var(--shadow);background:#fff}.action-overlay{position:absolute;inset:0;width:100%;height:100%;pointer-events:none;overflow:visible}.stage-tools{position:absolute;z-index:2;right:12px;top:12px;display:flex;gap:6px;padding:5px;background:#0c1119dd;border:1px solid var(--line);border-radius:10px}.stage-tools button{padding:4px 8px;font-size:11px}.stage-tools button.active{color:var(--green);border-color:#48d59788}.thumbs{display:flex;gap:8px;overflow-x:auto;padding:10px 12px;background:var(--panel);border-top:1px solid var(--line)}.thumb{flex:0 0 142px;height:106px;border:1px solid var(--line);border-radius:8px;overflow:hidden;position:relative;background:#080b10;cursor:pointer}.thumb:hover,.thumb.selected{border-color:var(--blue)}.thumb img{width:100%;height:100%;object-fit:cover}.thumblabel{position:absolute;left:0;right:0;bottom:0;background:#07101ad9;padding:4px 6px;font-size:10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.nav{display:flex;align-items:center;gap:10px;padding:7px 12px;border-top:1px solid var(--line);background:var(--panel)}.nav input{flex:1}.stepno{min-width:84px;text-align:center;color:var(--muted);font-variant-numeric:tabular-nums}.tabs{display:flex;border-bottom:1px solid var(--line);padding:0 10px}.tab{border:0;border-radius:0;background:none;padding:12px 10px;color:var(--muted);border-bottom:2px solid transparent}.tab.active{color:var(--text);border-color:var(--blue)}.details{padding:14px 15px;overflow:auto;flex:1}.section{margin-bottom:18px}.label{font-size:10px;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);margin:0 0 6px}.value{white-space:pre-wrap;word-break:break-word}.thought{border-left:2px solid var(--purple);padding-left:10px}.action{font:650 15px ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--blue)}pre{margin:0;padding:9px 10px;border:1px solid var(--line);border-radius:8px;background:#0b1017;white-space:pre-wrap;word-break:break-word;font:11px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;color:#cdd7e6}.kv{display:grid;grid-template-columns:auto 1fr;gap:5px 12px}.kv span:nth-child(odd){color:var(--muted)}.notice{border:1px solid #f1b95855;background:#f1b9580c;padding:8px;border-radius:7px}.loading:after{content:"";width:22px;height:22px;border:2px solid var(--line);border-top-color:var(--blue);border-radius:50%;animation:spin .8s linear infinite}.toast{position:fixed;bottom:16px;left:50%;transform:translateX(-50%);background:#202936;border:1px solid #40506a;border-radius:8px;padding:9px 14px;box-shadow:var(--shadow);z-index:9}.hidden{display:none!important}@media(max-width:1100px){.layout{grid-template-columns:300px minmax(400px,1fr)}.inspector{position:fixed;right:0;top:58px;bottom:0;width:min(420px,90vw);z-index:5;box-shadow:var(--shadow)}.inspector.closed{display:none}.top select{min-width:200px}}@media(max-width:720px){.layout{grid-template-columns:1fr}.sidebar{position:fixed;left:0;top:58px;bottom:0;width:min(355px,92vw);z-index:5;box-shadow:var(--shadow)}.sidebar.closed{display:none}.runmeta{display:none}.workspace{grid-template-rows:auto minmax(220px,1fr) 110px 48px}}
</style>
</head>
<body>
<div class="app">
 <header class="top">
  <div class="brand"><span class="brandmark">W</span>Trajectory Observatory</div>
  <button id="toggleList" title="显示/隐藏任务列表">☰</button>
  <select id="runSelect" aria-label="选择 run"></select>
  <div class="runmeta" id="runMeta"></div>
  <div class="spacer"></div>
  <button id="refresh" title="刷新磁盘数据">↻ 刷新</button>
  <button id="toggleInspector" title="显示/隐藏详情">详情</button>
 </header>
 <main class="layout">
  <aside class="sidebar" id="sidebar">
   <div class="filters">
    <input id="search" placeholder="搜索任务、域名、ID、答案…">
    <select id="statusFilter"><option value="">全部状态</option></select>
    <select id="judgeFilter"><option value="">全部判定</option></select>
   </div>
   <div class="stats" id="stats"></div>
   <div class="tasklist" id="taskList"></div>
  </aside>
  <section class="workspace">
   <div class="taskhead" id="taskHead"><div class="empty">选择一条轨迹</div></div>
   <div class="stage fit" id="stage"><div class="empty">暂无截图</div><div class="stage-tools"><button id="overlayBtn" class="active">坐标/方向：开</button><button id="fitBtn">适应窗口</button><button id="actualBtn">原始尺寸</button><button id="openImageBtn">打开图片</button></div></div>
   <div class="thumbs" id="thumbs"></div>
   <div class="nav"><button id="prev">←</button><span class="stepno" id="stepNo">—</span><input id="scrub" type="range" min="0" max="0" value="0"><button id="next">→</button><button id="play">播放</button></div>
  </section>
  <aside class="inspector" id="inspector">
   <div class="tabs"><button class="tab active" data-tab="step">当前步骤</button><button class="tab" data-tab="outcome">最终结果</button><button class="tab" data-tab="raw">JSON</button></div>
   <div class="details" id="details"></div>
  </aside>
 </main>
</div>
<div id="toast" class="toast hidden"></div>
<script>
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const esc=v=>String(v??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmt=n=>n==null?"—":Number(n).toLocaleString(undefined,{maximumFractionDigits:2});
const fmtTime=t=>t?new Date(t*1000).toLocaleString():"—";
const json=v=>esc(JSON.stringify(v??{},null,2));
const badge=(v)=>`<span class="badge ${esc(v||'unjudged')}">${esc(v||'unjudged')}</span>`;
let runs=[], runData=null, filtered=[], selectedId=null, detail=null, step=0, imageKind="screenshots", fit=true, showOverlay=true, tab="step", timer=null;
async function api(path){const r=await fetch(path,{cache:"no-store"});if(!r.ok)throw new Error(`${r.status} ${await r.text()}`);return r.json()}
function toast(message){const el=$("#toast");el.textContent=message;el.classList.remove("hidden");setTimeout(()=>el.classList.add("hidden"),2400)}
async function loadRuns(preserve=true){$("#refresh").classList.add("refreshing");try{const old=preserve?$("#runSelect").value:"";runs=(await api('/api/runs')).runs;$("#runSelect").innerHTML=runs.map(r=>`<option value="${esc(r.id)}">${esc(r.id)} · ${r.total}</option>`).join('');if(old&&runs.some(r=>r.id===old))$("#runSelect").value=old;if(runs.length)await loadRun($("#runSelect").value);else $("#taskList").innerHTML='<div class="empty">没有发现轨迹 run</div>'}catch(e){toast(`加载失败：${e.message}`)}finally{$("#refresh").classList.remove("refreshing")}}
async function loadRun(id){if(!id)return;const keep=selectedId;runData=await api(`/api/run/${encodeURIComponent(id)}`);selectedId=runData.trajectories.some(t=>t.id===keep)?keep:null;const r=runs.find(x=>x.id===id);$("#runMeta").innerHTML=r?`${badge(r.model||'unknown model')}<span>${fmtTime(r.updated_at)}</span>`:'';populateFilters();applyFilters();if(selectedId)await selectTrajectory(selectedId);else if(filtered.length)await selectTrajectory(filtered[0].id)}
function populateFilters(){const statuses=[...new Set(runData.trajectories.map(t=>t.status))].sort(),judges=[...new Set(runData.trajectories.map(t=>t.judge||'unjudged'))].sort();const sv=$("#statusFilter").value,jv=$("#judgeFilter").value;$("#statusFilter").innerHTML='<option value="">全部状态</option>'+statuses.map(x=>`<option>${esc(x)}</option>`).join('');$("#judgeFilter").innerHTML='<option value="">全部判定</option>'+judges.map(x=>`<option>${esc(x)}</option>`).join('');$("#statusFilter").value=statuses.includes(sv)?sv:'';$("#judgeFilter").value=judges.includes(jv)?jv:''}
function applyFilters(){if(!runData)return;const q=$("#search").value.trim().toLowerCase(),s=$("#statusFilter").value,j=$("#judgeFilter").value;filtered=runData.trajectories.filter(t=>(!s||t.status===s)&&(!j||(t.judge||'unjudged')===j)&&(!q||[t.id,t.instruction,t.domain,t.answer,t.error].join(' ').toLowerCase().includes(q)));renderStats();renderTasks()}
function renderStats(){const all=runData.trajectories,counts={};all.forEach(t=>counts[t.status]=(counts[t.status]||0)+1);$("#stats").innerHTML=`<span class="badge">显示 ${filtered.length}/${all.length}</span>`+Object.entries(counts).map(([k,v])=>`<span class="badge ${esc(k)}">${esc(k)} ${v}</span>`).join('')}
function renderTasks(){const el=$("#taskList");if(!filtered.length){el.innerHTML='<div class="empty">没有匹配的轨迹</div>';return}el.innerHTML=filtered.map(t=>`<article class="task ${t.id===selectedId?'selected':''}" data-id="${esc(t.id)}"><div class="task-title">${esc(t.instruction||'(no instruction)')}</div><div class="task-foot">${badge(t.status)}${t.judge?badge(t.judge):''}<span>${t.steps} steps</span><span class="domain">${esc(t.domain||t.id)}</span></div></article>`).join('');$$('.task').forEach(el=>el.onclick=()=>{step=0;selectTrajectory(el.dataset.id)})}
async function selectTrajectory(id){selectedId=id;renderTasks();$("#details").innerHTML='<div class="empty loading"></div>';try{detail=await api(`/api/run/${encodeURIComponent(runData.run)}/trajectory/${encodeURIComponent(id)}`);step=Math.max(0,Math.min(step,detail.steps.length-1));renderAll();location.hash=`${encodeURIComponent(runData.run)}/${encodeURIComponent(id)}/${step}`}catch(e){toast(`轨迹加载失败：${e.message}`)}}
function renderAll(){renderHead();renderThumbs();renderStep();renderDetails()}
function renderHead(){const t=detail.task||{},r=detail.result||{},j=detail.judge||{},urls=t.urls||[],url=r.final_url||r.start_url||urls[0]||'';$("#taskHead").innerHTML=`<div class="instruction">${esc(t.instruction||'(no instruction)')}</div><div class="meta"><span>${esc(detail.id)}</span>${badge(r.status||(detail.steps.length?'running':'pending'))}${j.judge?badge(j.judge):badge('unjudged')}<span>${detail.steps.length} steps</span>${r.timing?.duration_s!=null?`<span>${fmt(r.timing.duration_s)}s</span>`:''}${url?`<a href="${esc(url)}" target="_blank" rel="noreferrer">${esc(url)}</a>`:''}</div>`}
function preferredImage(s){return s?.images?.[imageKind]||s?.images?.screenshots||s?.images?.annotated||s?.images?.model_views||null}
function finite(v){return Number.isFinite(Number(v))}
function actionOverlay(s,w,h){if(!showOverlay||!s)return'';const a=s.agent||{},resolved=a.resolved||{},commands=a.commands||[],action=a.action||'',point=Array.isArray(resolved.point)&&resolved.point.length>=2?resolved.point:null,cmdPoint=commands.find(c=>finite(c.x)&&finite(c.y)),p=point||((cmdPoint)?[Number(cmdPoint.x),Number(cmdPoint.y)]:null),stroke=action==='scroll'?'#43d9ff':'#ff4d67';let shapes='';const defs=`<defs><filter id="olShadow"><feDropShadow dx="0" dy="1" stdDeviation="2" flood-opacity=".9"/></filter><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="${stroke}"/></marker></defs>`;const label=(x,y,text,color=stroke)=>{const safe=esc(text),width=Math.max(110,safe.length*13+20),lx=Math.max(8,Math.min(w-width-8,x+25)),ly=Math.max(34,Math.min(h-8,y-25));return `<g filter="url(#olShadow)"><rect x="${lx}" y="${ly-27}" width="${width}" height="32" rx="8" fill="#080c13df" stroke="${color}" stroke-width="2" vector-effect="non-scaling-stroke"/><text x="${lx+10}" y="${ly-6}" fill="#fff" font-size="18" font-family="ui-monospace,SFMono-Regular,monospace">${safe}</text></g>`};const drag=commands.find(c=>c.kind==='drag'&&finite(c.x1)&&finite(c.y1)&&finite(c.x2)&&finite(c.y2));const scroll=commands.find(c=>c.kind==='scroll')||(action==='scroll'?{dx:a.args?.dx||0,dy:a.args?.dy||0,x:p?.[0],y:p?.[1]}:null);if(drag){const x1=Number(drag.x1),y1=Number(drag.y1),x2=Number(drag.x2),y2=Number(drag.y2);shapes+=`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${stroke}" stroke-width="7" marker-end="url(#arrow)" vector-effect="non-scaling-stroke" filter="url(#olShadow)"/><circle cx="${x1}" cy="${y1}" r="17" fill="#080c1388" stroke="${stroke}" stroke-width="5" vector-effect="non-scaling-stroke"/>${label(x1,y1,`drag (${Math.round(x1)}, ${Math.round(y1)}) → (${Math.round(x2)}, ${Math.round(y2)})`)}`}else if(scroll){const dx=Number(scroll.dx)||0,dy=Number(scroll.dy)||0,vertical=Math.abs(dy)>=Math.abs(dx),sign=Math.sign(vertical?dy:dx)||1,magnitude=Math.max(Math.abs(dx),Math.abs(dy)),length=Math.min(vertical?h*.32:w*.32,Math.max(110,(vertical?h:w)*(.15+Math.min(magnitude,1200)/12000))),cx=finite(scroll.x)?Number(scroll.x):w/2,cy=finite(scroll.y)?Number(scroll.y):h/2,x1=vertical?cx:cx-sign*length/2,y1=vertical?cy-sign*length/2:cy,x2=vertical?cx:cx+sign*length/2,y2=vertical?cy+sign*length/2:cy,direction=vertical?(sign>0?'↓':'↑'):(sign>0?'→':'←');shapes+=`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${stroke}" stroke-width="9" marker-end="url(#arrow)" vector-effect="non-scaling-stroke" filter="url(#olShadow)"/><circle cx="${x1}" cy="${y1}" r="11" fill="${stroke}" stroke="#fff" stroke-width="3" vector-effect="non-scaling-stroke"/>${label(x1,y1,`scroll ${direction} ${Math.round(magnitude)} px`)}`}else if(p){const x=Number(p[0]),y=Number(p[1]),color=action==='hover'?'#f1b958':stroke;shapes+=`<g filter="url(#olShadow)"><circle cx="${x}" cy="${y}" r="28" fill="${color}22" stroke="${color}" stroke-width="5" vector-effect="non-scaling-stroke"/><circle cx="${x}" cy="${y}" r="8" fill="${color}" stroke="#fff" stroke-width="3" vector-effect="non-scaling-stroke"/><line x1="${x-38}" y1="${y}" x2="${x+38}" y2="${y}" stroke="${color}" stroke-width="2" vector-effect="non-scaling-stroke"/><line x1="${x}" y1="${y-38}" x2="${x}" y2="${y+38}" stroke="${color}" stroke-width="2" vector-effect="non-scaling-stroke"/></g>${label(x,y,`${action||'point'} (${Math.round(x)}, ${Math.round(y)})`,color)}`}return shapes?`<svg class="action-overlay" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" aria-label="action overlay">${defs}${shapes}</svg>`:''}
function layoutImageFrame(){const stage=$("#stage"),frame=$("#imageFrame"),img=$("#mainImage");if(!stage||!frame||!img||!img.naturalWidth)return;const nw=img.naturalWidth,nh=img.naturalHeight,scale=fit?Math.min(1,Math.max(0.05,(stage.clientWidth-28)/nw),Math.max(0.05,(stage.clientHeight-28)/nh)):1;frame.style.width=`${Math.round(nw*scale)}px`;frame.style.height=`${Math.round(nh*scale)}px`}
function renderThumbs(){const el=$("#thumbs");el.innerHTML=detail.steps.map((s,i)=>{const src=preferredImage(s),a=s.agent?.action||s.agent?.error||'observe';return `<div class="thumb ${i===step?'selected':''}" data-i="${i}">${src?`<img loading="lazy" src="${esc(src)}">`:''}<div class="thumblabel">${String(i+1).padStart(2,'0')} · ${esc(a)}</div></div>`}).join('');$$('.thumb').forEach(x=>x.onclick=()=>{step=+x.dataset.i;renderStep();renderDetails();renderThumbs()});setTimeout(()=>$('.thumb.selected')?.scrollIntoView({behavior:'smooth',block:'nearest',inline:'center'}),0)}
function renderStep(){const s=detail?.steps?.[step],stage=$("#stage"),tools=stage.querySelector('.stage-tools')?.outerHTML||'';stage.className=`stage ${fit?'fit':'actual'}`;const src=preferredImage(s),viewport=s?.state?.viewport||[1920,1080],w=Number(viewport[0])||1920,h=Number(viewport[1])||1080;stage.innerHTML=(src?`<div class="image-frame" id="imageFrame"><img id="mainImage" src="${esc(src)}" alt="step ${step+1}">${actionOverlay(s,w,h)}</div>`:'<div class="empty">该步骤没有截图</div>')+tools;bindStageTools();const img=$("#mainImage");if(img){img.onload=layoutImageFrame;if(img.complete)layoutImageFrame()}$("#stepNo").textContent=detail?.steps?.length?`${step+1} / ${detail.steps.length}`:'—';$("#scrub").max=Math.max(0,(detail?.steps?.length||1)-1);$("#scrub").value=step;$("#prev").disabled=step<=0;$("#next").disabled=!detail||step>=detail.steps.length-1;location.hash=detail?`${encodeURIComponent(runData.run)}/${encodeURIComponent(detail.id)}/${step}`:''}
function bindStageTools(){$("#overlayBtn").classList.toggle('active',showOverlay);$("#overlayBtn").textContent=`坐标/方向：${showOverlay?'开':'关'}`;$("#overlayBtn").onclick=()=>{showOverlay=!showOverlay;renderStep()};$("#fitBtn").onclick=()=>{fit=true;renderStep()};$("#actualBtn").onclick=()=>{fit=false;renderStep()};$("#openImageBtn").onclick=()=>{const src=preferredImage(detail?.steps?.[step]);if(src)window.open(src,'_blank')}}
function renderDetails(){if(!detail)return;const s=detail.steps[step]||{},a=s.agent||{},st=s.state||{},r=detail.result||{},j=detail.judge||{};if(tab==='outcome'){$("#details").innerHTML=`<div class="section"><div class="label">状态与答案</div><div class="value">${badge(r.status||'running')} ${j.judge?badge(j.judge):badge('unjudged')}</div></div><div class="section"><div class="label">Agent 最终答案</div><div class="value">${esc(r.stop_answer||'—')}</div></div>${r.error?`<div class="section"><div class="label">错误</div><div class="notice">${esc(r.error)}</div></div>`:''}${j.thought?`<div class="section"><div class="label">Judge rationale</div><div class="thought value">${esc(j.thought)}</div></div>`:''}<div class="section"><div class="label">指标</div><div class="kv"><span>Judge latency</span><span>${fmt(j.latency_s)}s</span><span>Alignment</span><span>${fmt(j.alignment_score)}</span><span>Efficiency</span><span>${fmt(j.efficiency)}</span><span>Self correction</span><span>${fmt(j.self_correction)}</span><span>总耗时</span><span>${fmt(r.timing?.duration_s)}s</span></div></div>`;return}if(tab==='raw'){$("#details").innerHTML=`<div class="section"><div class="label">Agent</div><pre>${json(a)}</pre></div><div class="section"><div class="label">State</div><pre>${json(st)}</pre></div><div class="section"><div class="label">Result</div><pre>${json(r)}</pre></div><div class="section"><div class="label">Judge</div><pre>${json(j)}</pre></div>`;return}const usage=a.usage||{},commands=a.commands||[],results=a.command_results||[],rawReply=a.reply&&a.reply!==a.analysis?`<div class="section"><div class="label">Raw model reply</div><pre>${esc(a.reply)}</pre></div>`:'';$("#details").innerHTML=`<div class="section"><div class="label">Action</div><div class="action">${esc(a.action||a.error||'—')}</div>${a.args&&Object.keys(a.args).length?`<pre>${json(a.args)}</pre>`:''}</div><div class="section"><div class="label">Model analysis</div><div class="value">${esc(a.analysis||a.reply||'—')}</div></div>${rawReply}${a.error?`<div class="section"><div class="label">Error</div><div class="notice">${esc(a.error)}</div></div>`:''}${a.notices?.length?`<div class="section"><div class="label">Notices</div><div class="notice">${esc(a.notices.join('\n'))}</div></div>`:''}<div class="section"><div class="label">调用指标</div><div class="kv"><span>Latency</span><span>${fmt(a.latency_s)}s</span><span>Input tokens</span><span>${fmt(usage.input_tokens??usage.prompt_tokens)}</span><span>Output tokens</span><span>${fmt(usage.output_tokens??usage.completion_tokens)}</span><span>Image</span><span>${esc((a.sent_image_size||[]).join(' × ')||'—')}</span></div></div><div class="section"><div class="label">页面</div><div class="value">${esc(st.title||'')}${st.url?`\n${esc(st.url)}`:''}</div></div>${commands.length?`<div class="section"><div class="label">执行命令</div><pre>${json(commands)}</pre></div>`:''}${results.length?`<div class="section"><div class="label">执行结果</div><pre>${json(results)}</pre></div>`:''}`}
function moveStep(delta){if(!detail?.steps?.length)return;const next=Math.max(0,Math.min(detail.steps.length-1,step+delta));if(next===step)return;step=next;renderStep();renderDetails();renderThumbs()}
function moveTask(delta){const i=filtered.findIndex(t=>t.id===selectedId),n=Math.max(0,Math.min(filtered.length-1,i+delta));if(filtered[n]&&n!==i){step=0;selectTrajectory(filtered[n].id)}}
$("#runSelect").onchange=e=>{selectedId=null;step=0;loadRun(e.target.value)};$("#refresh").onclick=()=>loadRuns(true);$("#search").oninput=applyFilters;$("#statusFilter").onchange=applyFilters;$("#judgeFilter").onchange=applyFilters;$("#prev").onclick=()=>moveStep(-1);$("#next").onclick=()=>moveStep(1);$("#scrub").oninput=e=>{step=+e.target.value;renderStep();renderDetails();renderThumbs()};$("#play").onclick=()=>{if(timer){clearInterval(timer);timer=null;$("#play").textContent='播放'}else{timer=setInterval(()=>{if(step>=detail.steps.length-1){clearInterval(timer);timer=null;$("#play").textContent='播放'}else moveStep(1)},1200);$("#play").textContent='暂停'}};$("#toggleList").onclick=()=>$("#sidebar").classList.toggle('closed');$("#toggleInspector").onclick=()=>$("#inspector").classList.toggle('closed');$$('.tab').forEach(x=>x.onclick=()=>{tab=x.dataset.tab;$$('.tab').forEach(y=>y.classList.toggle('active',y===x));renderDetails()});document.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName))return;if(e.key==='ArrowLeft')moveStep(-1);if(e.key==='ArrowRight')moveStep(1);if(e.key==='ArrowUp')moveTask(-1);if(e.key==='ArrowDown')moveTask(1)});bindStageTools();
const initialHash=location.hash.slice(1);loadRuns(false).then(async()=>{const [run,id,index]=initialHash.split('/').map(decodeURIComponent);if(run&&runs.some(r=>r.id===run)){$("#runSelect").value=run;selectedId=id||null;step=Number(index)||0;await loadRun(run)}});
</script>
</body></html>'''


if __name__ == "__main__":
    main()
