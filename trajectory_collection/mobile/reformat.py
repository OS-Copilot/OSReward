#!/usr/bin/env python3
"""Reformat extracted episodes into the unified OSReward trajectory schema.

Input: the per-episode directories produced by extract_episodes.py
(`{task}_{instance}_{timestamp}/` holding `<dir>.json` plus per-step PNGs
named `{task}_{instance}_{step:04d}.png`), recorded by the tool-call
collection agent (android_world.agents.seeact.ToolCallAgent).

Output: one JSON per episode in the unified schema (trace_id, instruction,
trajectory steps with state / action / thought / prm_label, orm_label).

Usage:
  python reformat.py --root /path/to/extracted --out_root /path/to/reformatted
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional

AGENT_METADATA = {
    "producer": "",
    "model_name": "",
    "prompt_version": "",
}
SOURCE = "AndroidWorld-Ext"
PLATFORM = "Mobile"
IN_DOMAIN = "0"


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def mirror_output_path(in_path: str, in_root: str, out_root: str) -> str:
    return os.path.join(out_root, os.path.relpath(in_path, start=in_root))


def sanitize_name(name: str) -> str:
    """Same sanitization as extract_episodes.py, for filename reconstruction."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _tag(text: str, tag: str) -> Optional[str]:
    m = re.search(rf"<{tag}>\s*([\s\S]*?)\s*</{tag}>", text or "")
    return m.group(1) if m else None


def _coord(x: Any, y: Any, sw: Optional[int], sh: Optional[int]) -> Dict[str, Any]:
    if x is None or y is None:
        return {"absolute": [None, None], "relative": [None, None]}
    ax, ay = int(round(float(x))), int(round(float(y)))
    rx = round(float(x) / sw, 6) if sw else None
    ry = round(float(y) / sh, 6) if sh else None
    return {"absolute": [ax, ay], "relative": [rx, ry]}


def convert_action(parsed: Optional[Dict[str, Any]],
                   sw: Optional[int], sh: Optional[int]) -> Dict[str, Any]:
    """Maps a ToolCallAgent JSONAction dict onto the unified action schema."""
    if not isinstance(parsed, dict):
        return {"type": "unknown"}

    a = (parsed.get("action_type") or "").lower()

    if a == "click":
        return {"type": "click",
                "coordinates": [_coord(parsed.get("x"), parsed.get("y"), sw, sh)]}
    if a == "long_press":
        return {"type": "longpress",
                "coordinates": [_coord(parsed.get("x"), parsed.get("y"), sw, sh)]}
    if a == "scroll":
        return {"type": "scroll", "coordinates": [],
                "direction": (parsed.get("direction") or "").lower()}
    if a == "input_text":
        return {"type": "type", "content": parsed.get("text") or ""}
    if a == "keyboard_enter":
        return {"type": "hotkey", "keys": ["enter"]}
    if a == "navigate_back":
        return {"type": "hotkey", "keys": ["back"]}
    if a == "navigate_home":
        return {"type": "hotkey", "keys": ["home"]}
    if a == "open_app":
        return {"type": "open_app", "appname": parsed.get("app_name") or ""}
    if a == "answer":
        return {"type": "answer", "text": parsed.get("text") or ""}
    if a == "wait":
        return {"type": "wait"}
    if a == "status":
        gs = (parsed.get("goal_status") or "").lower()
        if gs == "complete":
            return {"type": "answer", "content": "COMPLETE"}
        if gs == "infeasible":
            return {"type": "answer", "content": "IMPOSSIBLE"}
        return {"type": "answer", "content": gs.upper()}

    return {"type": "unknown", "original": parsed}


def process_episode(path: str, agent_metadata: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Converts one extracted episode JSON to the unified schema."""
    payload = load_json(path)
    if not isinstance(payload, dict) or "episode_data" not in payload:
        return None

    episode_data = payload.get("episode_data", {}) or {}

    responses = episode_data.get("response") or []
    parsed_list = episode_data.get("parsed") or []
    histories = episode_data.get("step_history") or []
    summaries = episode_data.get("summary") or []
    widths = episode_data.get("screen_width") or []
    heights = episode_data.get("screen_height") or []

    n_steps = max(len(responses), len(parsed_list), len(histories)) or 0
    if n_steps == 0:
        return None

    sw = next((w for w in widths if w), None)
    sh = next((h for h in heights if h), None)

    trace_id = os.path.basename(os.path.dirname(path)) or os.path.splitext(
        os.path.basename(path))[0]

    task_template = payload.get("task_template") or ""
    instance_id = payload.get("instance_id")
    instance_id_str = str(instance_id) if instance_id is not None else "0"
    goal = payload.get("goal") or ""
    png_prefix = f"{sanitize_name(task_template)}_{instance_id_str}"

    trajectory: List[Dict[str, Any]] = []
    for idx in range(n_steps):
        response = responses[idx] if idx < len(responses) else ""
        response = response if isinstance(response, str) else str(response)
        parsed = parsed_list[idx] if idx < len(parsed_list) else None

        thought = {
            "thinking": _tag(response, "thinking"),
            "conclusion": _tag(response, "conclusion"),
        }
        summary = summaries[idx] if idx < len(summaries) else None

        trajectory.append({
            "step_index": idx,
            "state": {
                "screenshot_path": f"{trace_id}/{png_prefix}_{idx:04d}.png",
            },
            "raw_response": response,
            "thought": thought,
            "summary": summary,
            "action": convert_action(parsed, sw, sh),
            "prm_label": {"is_error": False, "correction": None},
        })

    is_successful = payload.get("is_successful")

    return {
        "trace_id": trace_id,
        "task_id": task_template or goal or instance_id_str,
        "task_source": SOURCE,
        "in_domain": IN_DOMAIN,
        "platform": PLATFORM,
        "subdomain": "",
        "environment_details": {
            "screen_resolution": f"{sw}x{sh}" if sw and sh else "",
            "os_version": "",
            "browser_name": "",
            "browser_version": "",
        },
        "instruction": goal,
        "agent_metadata": agent_metadata,
        "held_out": 0,
        "trajectory": trajectory,
        "trajectory_length": len(trajectory),
        "orm_label": {
            # The task's own validator verdict; annotate_binary_reward.py can
            # overwrite binary_reward with a manually verified label.
            "score": is_successful,
            "binary_reward": None,
            "rationale": "",
        },
        "annotation_metadata": {
            "annotator_id": "",
            "annotation_tool_version": "",
            "timestamp": "",
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Reformat extracted episodes into the unified schema.")
    parser.add_argument("--root", required=True,
                        help="Input root (output of extract_episodes.py)")
    parser.add_argument("--out_root", required=True, help="Output root")
    parser.add_argument("--model_name", default="",
                        help="Model id for agent_metadata; defaults to the"
                             " episode's recorded agent_name")
    args = parser.parse_args()

    count = 0
    for dirpath, _, filenames in os.walk(args.root):
        if os.path.abspath(dirpath).startswith(os.path.abspath(args.out_root)):
            continue

        for fname in filenames:
            if not fname.lower().endswith(".json"):
                continue
            in_path = os.path.join(dirpath, fname)
            try:
                payload_model = args.model_name
                data = load_json(in_path)
                if not payload_model and isinstance(data, dict):
                    payload_model = data.get("agent_name") or ""
                agent_metadata = dict(AGENT_METADATA, model_name=payload_model)

                result = process_episode(in_path, agent_metadata)
                if result is None:
                    continue
                out_path = mirror_output_path(in_path, args.root, args.out_root)
                if os.path.exists(out_path):
                    continue
                save_json(out_path, result)
                count += 1
            except Exception as e:
                print(f"[ERROR] {fname}: {e}")

    print(f"Done. Processed {count} files.")


if __name__ == "__main__":
    main()
