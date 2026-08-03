#!/usr/bin/env python3
"""Reformat extracted AndroidWorld episodes into the unified trajectory schema.

Input: the per-episode directories produced by extract_episodes.py, each
holding {task_template}_{instance_id}.json plus per-step PNGs.
Output: one JSON per episode in the unified OSReward schema (trace_id,
instruction, trajectory steps with state / action / prm_label, orm_label).

Usage:
  python reformat.py --root /path/to/extracted --out_root /path/to/reformatted
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

AGENT_METADATA = {
    "producer": "Anthropic",
    "model_name": "claude-sonnet-4-5-20250929",
    "prompt_version": "",
}
SOURCE = "AndroidWorld"
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


def _coord_from_bbox_element(elem: Dict[str, Any],
                             sw: Optional[int],
                             sh: Optional[int]) -> Dict[str, Any]:
    """Returns the bbox-center coordinate of an a11y element.

    Output: {"absolute": [x, y], "relative": [rx, ry]} in pixels / fractions.
    """
    bbox = (elem or {}).get("bbox_pixels") or {}
    x_min = bbox.get("x_min")
    x_max = bbox.get("x_max")
    y_min = bbox.get("y_min")
    y_max = bbox.get("y_max")

    if None in (x_min, x_max, y_min, y_max):
        return {"absolute": [None, None], "relative": [None, None]}

    x = 0.5 * (float(x_min) + float(x_max))
    y = 0.5 * (float(y_min) + float(y_max))

    ax = int(round(x))
    ay = int(round(y))
    rx = round(x / float(sw), 6) if sw else None
    ry = round(y / float(sh), 6) if sh else None

    return {"absolute": [ax, ay], "relative": [rx, ry]}


def parse_action_from_raw_response(
    raw_response: str,
    a11y_tree: Optional[List[Dict[str, Any]]],
    sw: Optional[int],
    sh: Optional[int],
) -> Dict[str, Any]:
    """Extracts the AndroidWorld action from the raw model response.

    Expected format: 'Action: {"action_type": "status", "goal_status": ...}'.
    """
    if not raw_response:
        return {"type": "unknown"}

    m = re.search(r"Action:\s*(\{.*\})", raw_response)
    if not m:
        return {"type": "unknown"}

    try:
        action_json = json.loads(m.group(1))
    except Exception:
        return {"type": "unknown"}

    action_type = action_json.get("action_type", "").upper()

    if action_type == "STATUS":
        gs = (action_json.get("goal_status") or "").upper()
        if gs == "COMPLETE":
            return {"type": "answer", "content": "COMPLETE"}
        elif gs in ("FAILED", "FAIL", "INCOMPLETE"):
            return {"type": "answer", "content": "FAILED"}
        elif gs in ("IMPOSSIBLE", "UNACHIEVABLE"):
            return {"type": "answer", "content": "IMPOSSIBLE"}
        else:
            return {"type": "answer", "content": gs}

    if action_type == "CLICK":
        idx = action_json.get("index")
        coords: List[Dict[str, Any]] = []
        if isinstance(idx, int) and a11y_tree and 0 <= idx < len(a11y_tree):
            coords.append(_coord_from_bbox_element(a11y_tree[idx], sw, sh))
        return {"type": "click", "coordinates": coords}

    if action_type == "SCROLL":
        idx = action_json.get("index")
        direction = action_json.get("direction", "").lower()
        coords = []
        if isinstance(idx, int) and a11y_tree and 0 <= idx < len(a11y_tree):
            coords.append(_coord_from_bbox_element(a11y_tree[idx], sw, sh))
        return {"type": "scroll", "coordinates": coords, "direction": direction}

    if action_type == "INPUT_TEXT":
        text = action_json.get("text") or action_json.get("value") or ""
        return {"type": "type", "content": text}

    if action_type == "KEYBOARD_ENTER":
        return {"type": "hotkey", "keys": ["enter"]}

    if action_type == "NAVIGATE_BACK":
        return {"type": "hotkey", "keys": ["back"]}

    if action_type == "NAVIGATE_HOME":
        return {"type": "hotkey", "keys": ["home"]}

    if action_type == "LONG_PRESS":
        return {"type": "longpress", "coordinates": []}

    if action_type == "WAIT":
        return {"type": "wait"}

    if action_type == "ANSWER":
        content = action_json.get("content") or ""
        return {"type": "answer", "text": content}

    if action_type == "OPEN_APP":
        appname = action_json.get("app_name") or ""
        return {"type": "open_app", "appname": appname}

    return {"type": "unknown", "original": action_json}


def process_episode(path: str, agent_metadata: Dict[str, str]) -> List[Tuple[Optional[str], Dict]]:
    """Converts one extracted episode JSON to the unified schema.

    - trace_id = name of the directory containing the json.
    - screenshot_path = trace_id/trace_id_{step:04d}.png
    - state.a11y_tree = before_element_list[step]; after_elements are ignored.
    """
    payload = load_json(path)
    if not isinstance(payload, dict):
        return []

    episode_data = payload.get("episode_data", {}) or {}
    screen_config = payload.get("screen_config", {}) or {}

    before_elements = episode_data.get("before_element_list") or []
    action_prompt = episode_data.get("action_prompt") or []
    action_output = episode_data.get("action_output") or []
    action_raw_response = episode_data.get("action_raw_response") or []
    summary = episode_data.get("summary") or []

    n_steps = max(
        len(before_elements),
        len(action_prompt),
        len(action_output),
        len(action_raw_response),
        len(summary),
    ) or 0

    sw = screen_config.get("width")
    sh = screen_config.get("height")

    trace_id = os.path.basename(os.path.dirname(path)) or os.path.splitext(
        os.path.basename(path))[0]

    instance_id = payload.get("instance_id")
    instance_id_str = str(instance_id) if instance_id is not None else "0"

    goal = payload.get("goal") or ""
    task_template = payload.get("task_template")
    task_id_field = task_template or goal or instance_id_str

    trajectory: List[Dict[str, Any]] = []

    for idx in range(n_steps):
        screenshot_path = f"{trace_id}/{trace_id}_{idx:04d}.png"
        a11y_tree = before_elements[idx] if idx < len(before_elements) else None

        # Flatten action_raw_response[idx] (an Anthropic-shaped message) to text.
        rr = ""
        if idx < len(action_raw_response):
            rr_item = action_raw_response[idx]
            if isinstance(rr_item, dict):
                content = rr_item.get("content")
                if isinstance(content, list) and content and isinstance(content[0], dict):
                    text = content[0].get("text")
                    rr = text if isinstance(text, str) else ""
                else:
                    rr = json.dumps(rr_item, ensure_ascii=False)
            else:
                rr = str(rr_item)

        thought = ""
        if idx < len(summary) and summary[idx]:
            thought = str(summary[idx])
        elif idx < len(action_prompt) and action_prompt[idx]:
            thought = str(action_prompt[idx])

        trajectory.append({
            "step_index": idx,
            "state": {
                "screenshot_path": screenshot_path,
                "a11y_tree": a11y_tree,
            },
            "raw_response": rr,
            "thought": thought,
            "action": parse_action_from_raw_response(rr, a11y_tree, sw, sh),
            "prm_label": {"is_error": False, "correction": None},
        })

    transformed = {
        "trace_id": trace_id,
        "task_id": task_id_field,
        "task_source": SOURCE,
        "in_domain": IN_DOMAIN,
        "platform": PLATFORM,
        "subdomain": screen_config.get("config_name") or "",
        "environment_details": {
            "screen_resolution": f"{sw}x{sh}" if sw and sh else "",
            "os_version": "",
            "browser_name": "",
            "browser_version": "",
        },
        "instruction": goal or "",
        "agent_metadata": agent_metadata,
        "held_out": 0,
        "trajectory": trajectory,
        "trajectory_length": len(trajectory),
        "orm_label": {
            "score": None,
            "binary_reward": None,
            "rationale": "",
        },
        "annotation_metadata": {
            "annotator_id": "",
            "annotation_tool_version": "",
            "timestamp": "",
        },
    }

    return [(None, transformed)]


def main():
    parser = argparse.ArgumentParser(
        description="Reformat extracted AndroidWorld episodes.")
    parser.add_argument("--root", required=True,
                        help="Input root (output of extract_episodes.py)")
    parser.add_argument("--out_root", required=True, help="Output root")
    parser.add_argument("--model_name", default=AGENT_METADATA["model_name"],
                        help="Model id recorded in agent_metadata")
    args = parser.parse_args()

    agent_metadata = dict(AGENT_METADATA, model_name=args.model_name)

    count = 0
    for dirpath, _, filenames in os.walk(args.root):
        if os.path.abspath(dirpath).startswith(os.path.abspath(args.out_root)):
            continue

        for fname in filenames:
            if not fname.lower().endswith(".json"):
                continue

            in_path = os.path.join(dirpath, fname)
            try:
                for (out_name, data) in process_episode(in_path, agent_metadata):
                    target_name = out_name if out_name else fname
                    out_path = mirror_output_path(
                        os.path.join(dirpath, target_name), args.root, args.out_root)
                    if os.path.exists(out_path):
                        continue
                    save_json(out_path, data)
                    count += 1
            except Exception as e:
                print(f"[ERROR] {fname}: {e}")

    print(f"Done. Processed {count} files.")


if __name__ == "__main__":
    main()
