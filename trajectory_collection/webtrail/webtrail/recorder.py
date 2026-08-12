"""Trajectory persistence.

Layout (one directory per trajectory, one subdirectory per artifact type,
files named by step so each folder browses as a flat sequence)::

    <out_dir>/
      run_config.json
      api_calls.jsonl                  # model-call latency/usage log
      rejects.jsonl                    # tasks that never produced a trajectory
      trajectories/<task_id>/
        task.json
        result.json                    # final status summary for the episode
        judge.json                     # written later by `webtrail judge`
        screenshots/step_000.png       # what the browser rendered
        model_views/step_000.png       # optional resized copy sent to the model
        annotated/step_000.png         # optional executed-action visualization
        html/step_000.html             # optional raw page HTML
        axtree/step_000.json           # optional accessibility tree
        elements/step_000.json
        states/step_000.json           # url/title/scroll/hashes/guard verdict
        agent/step_000.json            # raw reply, parsed action, resolved
                                       # target, commands, usage, latency
        messages/step_000.json         # optional debug dump of the model input

`result.json` is the only file post-processing must read to triage an episode.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from PIL import ImageDraw

from . import imutil
from .types import CompiledAction, PageState, Task

FINAL_STATUSES = {"completed", "max_steps", "blocked", "env_error",
                  "stale_loop", "agent_error"}


def _write_json(path: Path, data) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=1, default=str)
    # live pages occasionally contain unpaired UTF-16 surrogates; never let
    # broken page text kill a trajectory write
    path.write_text(text, encoding="utf-8", errors="replace")


class RunRecorder:
    """Run-level files: config snapshot, rejects, api log path."""

    def __init__(self, out_dir: str | Path, *, save_html: bool = False,
                 save_axtree: bool = False, save_model_views: bool = False):
        self.root = Path(out_dir)
        self.trajectories = self.root / "trajectories"
        self.trajectories.mkdir(parents=True, exist_ok=True)
        self._reject_lock = asyncio.Lock()
        self._save_html = save_html
        self._save_axtree = save_axtree
        self._save_model_views = save_model_views

    @property
    def api_log_path(self) -> Path:
        return self.root / "api_calls.jsonl"

    def save_config(self, config_dict: dict) -> None:
        _write_json(self.root / "run_config.json", {
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            **config_dict,
        })

    async def reject(self, task: Task, reason: str, detail: str = "") -> None:
        record = {
            "task_id": task.task_id,
            "domain": task.domain,
            "start_url": task.start_url,
            "reason": reason,
            "detail": detail[:500],
            "ts": time.time(),
        }
        async with self._reject_lock:
            with (self.root / "rejects.jsonl").open(
                "a", encoding="utf-8", errors="replace"
            ) as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def is_finished(self, task_id: str) -> bool:
        result = self.trajectories / task_id / "result.json"
        if not result.exists():
            return False
        try:
            return json.loads(result.read_text()).get("status") in FINAL_STATUSES
        except (ValueError, OSError):
            return False

    def open_trajectory(self, task: Task) -> "TrajectoryRecorder":
        return TrajectoryRecorder(
            self.trajectories / task.task_id,
            task,
            save_html=self._save_html,
            save_axtree=self._save_axtree,
            save_model_views=self._save_model_views,
        )


def step_file(traj_dir: Path, kind: str, index: int, suffix: str) -> Path:
    """Canonical artifact path: <traj>/<kind>/step_NNN<suffix>."""
    return traj_dir / kind / f"step_{index:03d}{suffix}"


class TrajectoryRecorder:
    def __init__(self, root: Path, task: Task, *, save_html: bool = False,
                 save_axtree: bool = False, save_model_views: bool = False):
        self.root = root
        self._save_html = save_html
        self._save_axtree = save_axtree
        self._save_model_views = save_model_views
        root.mkdir(parents=True, exist_ok=True)
        self.started_at = time.time()
        _write_json(root / "task.json", {
            "task_id": task.task_id,
            "instruction": task.instruction,
            "urls": task.urls,
            "steps": task.steps,
            "criteria": task.criteria,
            "extras": task.extras,
        })

    def _file(self, kind: str, index: int, suffix: str) -> Path:
        path = step_file(self.root, kind, index, suffix)
        path.parent.mkdir(exist_ok=True)
        return path

    def save_observation(self, index: int, state: PageState, verdict,
                         model_view_png: bytes | None = None) -> None:
        if state.screenshot_png:
            self._file("screenshots", index, ".png").write_bytes(state.screenshot_png)
        if self._save_model_views and model_view_png is not None:
            self._file("model_views", index, ".png").write_bytes(model_view_png)
        if self._save_html and state.html:
            self._file("html", index, ".html").write_text(
                state.html, encoding="utf-8", errors="replace")
        if self._save_axtree and state.axtree is not None:
            _write_json(self._file("axtree", index, ".json"), state.axtree)
        if state.elements is not None:
            _write_json(self._file("elements", index, ".json"), state.elements)

        shot_hash = None
        if state.screenshot_png:
            shot_hash = f"{imutil.dhash(imutil.load_png(state.screenshot_png)):016x}"
        _write_json(self._file("states", index, ".json"), {
            "url": state.url,
            "title": state.title,
            "scroll": state.scroll,
            "viewport": list(state.viewport),
            "html_bytes": len(state.html or ""),
            "num_elements": len(state.elements or []),
            "screenshot_dhash": shot_hash,
            "observation_errors": state.errors,
            "guard": {
                "kind": verdict.kind,
                "scope": verdict.scope,
                "evidence": verdict.evidence,
            } if verdict else None,
            "observed_at": time.time(),
        })

    def save_decision(self, index: int, *, reply_text: str, parsed_key: str | None,
                      parsed_args: dict | None, analysis: str,
                      compiled: CompiledAction | None, command_results: list[dict],
                      usage: dict, latency_s: float, parse_attempts: int,
                      sent_size: tuple[int, int], notices: list[str],
                      messages: list[dict] | None = None) -> None:
        _write_json(self._file("agent", index, ".json"), {
            "reply": reply_text,
            "analysis": analysis,
            "action": parsed_key,
            "args": parsed_args,
            "resolved": {
                "point": list(compiled.point) if compiled and compiled.point else None,
                "box_px": list(compiled.box_px) if compiled and compiled.box_px else None,
                "drag_to": list(compiled.drag_to) if compiled and compiled.drag_to else None,
                "goto_url": compiled.goto_url if compiled else None,
            },
            "commands": compiled.commands if compiled else [],
            "command_results": command_results,
            "usage": usage,
            "latency_s": round(latency_s, 3),
            "parse_attempts": parse_attempts,
            "sent_image_size": list(sent_size),
            "notices": notices,
        })
        if messages is not None:
            _write_json(self._file("messages", index, ".json"), messages)

    def save_annotation(self, index: int, screenshot_png: bytes,
                        compiled: CompiledAction) -> None:
        """Draw the executed action (box, point, drag arrow) onto a copy."""
        if not (compiled.point or compiled.box_px):
            return
        image = imutil.load_png(screenshot_png)
        draw = ImageDraw.Draw(image)
        color = (255, 40, 40)
        if compiled.box_px:
            x1, y1, x2, y2 = compiled.box_px
            for inset in range(3):
                draw.rectangle([x1 - inset, y1 - inset, x2 + inset, y2 + inset],
                               outline=color)
        if compiled.point:
            x, y = compiled.point
            radius = 9
            draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                         outline=color, width=3)
            draw.line([x - radius - 6, y, x + radius + 6, y], fill=color, width=2)
            draw.line([x, y - radius - 6, x, y + radius + 6], fill=color, width=2)
        if compiled.point and compiled.drag_to:
            draw.line([compiled.point, compiled.drag_to], fill=color, width=3)
            dx, dy = compiled.drag_to
            draw.ellipse([dx - 5, dy - 5, dx + 5, dy + 5], fill=color)
        label = compiled.key + (f" {compiled.point}" if compiled.point else "")
        draw.text((8, 8), label, fill=color)
        self._file("annotated", index, ".png").write_bytes(imutil.to_png_bytes(image))

    def save_result(self, task: Task, *, status: str, steps_taken: int,
                    final_url: str | None, stop_answer: str | None,
                    block: dict | None, counters: dict,
                    missing_screenshot_steps: list[int],
                    action_keys: list[str], error: str | None = None) -> dict:
        result = {
            "task_id": task.task_id,
            "domain": task.domain,
            "start_url": task.start_url,
            "final_url": final_url,
            "status": status,
            "stop_answer": stop_answer,
            "block": block,
            "steps_taken": steps_taken,
            "action_keys": action_keys,
            "counters": counters,
            "missing_screenshot_steps": missing_screenshot_steps,
            "error": error,
            "timing": {
                "started_at": self.started_at,
                "ended_at": time.time(),
                "duration_s": round(time.time() - self.started_at, 1),
            },
        }
        _write_json(self.root / "result.json", result)
        return result
