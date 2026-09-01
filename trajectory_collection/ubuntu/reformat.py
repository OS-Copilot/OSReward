"""Reformat collected episodes into the unified OSReward trajectory schema.

Input: a run.py result directory (<result_dir>/<domain>/meta_<task_id>.json plus
<result_dir>/<domain>/<task_id>/step_*.png).

Output: one <trace_id>.json per episode under --out_root, with screenshots
copied to <out_root>/<trace_id>/. Each trajectory step carries the screenshot
path and click coordinate both nested (state.screenshot_path, for the judged-SFT
exporter) and flat (screenshot_path / coordinate, for eval_pipeline/run_judge.py),
so either consumer reads the file directly. When the task validator produced a
score, it is written to orm_label.score and thresholded into human_label.

Usage:
    python reformat.py --result_dir results/demo --out_root results/demo_reformatted
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

SUCCESS_THRESHOLD = 0.99
PLATFORM = "Ubuntu"
SOURCE = "osreward_ubuntu"


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def process_episode(meta_path: str, out_root: str) -> str | None:
    meta = load_json(meta_path)
    trajectory_in = meta.get("trajectory") or []
    if not trajectory_in:
        return None

    trace_id = meta.get("trace_id") or os.path.splitext(
        os.path.basename(meta_path))[0].removeprefix("meta_")
    domain_dir = os.path.dirname(meta_path)
    out_trace_dir = os.path.join(out_root, trace_id)
    os.makedirs(out_trace_dir, exist_ok=True)

    trajectory_out = []
    for step in trajectory_in:
        idx = step.get("step_index", len(trajectory_out))
        src_png = os.path.join(domain_dir, step.get("screenshot_path") or "")
        rel_png = f"{trace_id}/step_{idx:04d}.png"
        if os.path.isfile(src_png):
            shutil.copyfile(src_png, os.path.join(out_root, rel_png))
        else:
            print(f"  [warn] missing screenshot: {src_png}")

        coordinate = step.get("coordinate")
        action: dict = {"type": "pyautogui", "command": step.get("action", "")}
        # Exporter convention: coordinates as {"relative": [x, y]} in 0-1.
        points = [coordinate] if coordinate else (step.get("coordinate2") or [])
        relative_points = [
            {"relative": [p[0] / 1000.0, p[1] / 1000.0]}
            for p in points
            if isinstance(p, (list, tuple)) and len(p) == 2
        ]
        if relative_points:
            action["coordinates"] = relative_points
        trajectory_out.append({
            "step_index": idx,
            # Nested form: canonical schema (export_judged_sft).
            "state": {"screenshot_path": rel_png},
            # Flat form: eval_pipeline/run_judge.py trace format.
            "screenshot_path": rel_png,
            "raw_response": step.get("raw_response", ""),
            "thought": step.get("thought", ""),
            "action": action,
            "coordinate": coordinate,          # [x, y] normalized 0-1000, or None
            "coordinate2": step.get("coordinate2"),
            "code_result": step.get("code_result", ""),
            "prm_label": {"is_error": False, "correction": None},
        })

    score = meta.get("score")
    rule_reward = (meta.get("rule_judge") or {}).get("reward")

    record = {
        "trace_id": trace_id,
        "task_id": meta.get("task_id", trace_id),
        "task_source": SOURCE,
        "platform": PLATFORM,
        "subdomain": meta.get("subdomain", ""),
        "environment_details": meta.get("environment_details") or {},
        "instruction": meta.get("instruction", ""),
        "agent": meta.get("agent", ""),
        "agent_metadata": {"model": meta.get("agent", "")},
        "trajectory": trajectory_out,
        "trajectory_length": len(trajectory_out),
        "orm_label": {
            # The task validator's verdict; annotate or judge downstream when absent.
            "score": score,
            "binary_reward": None,
            "rationale": "",
        },
        "annotation_metadata": meta.get("annotation_metadata") or {},
    }
    if isinstance(score, (int, float)) and score >= 0:
        record["human_label"] = "SUCCESS" if score >= SUCCESS_THRESHOLD else "FAIL"
    elif isinstance(rule_reward, (int, float)) and rule_reward >= 0:
        record["human_label"] = "SUCCESS" if rule_reward >= SUCCESS_THRESHOLD else "FAIL"

    out_json = os.path.join(out_root, f"{trace_id}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return out_json


def main():
    parser = argparse.ArgumentParser(
        description="Reformat collected episodes into the unified schema.")
    parser.add_argument("--result_dir", required=True,
                        help="run.py output directory (contains <domain>/meta_*.json).")
    parser.add_argument("--out_root", required=True,
                        help="Output directory for unified-schema episodes.")
    args = parser.parse_args()

    meta_paths = sorted(glob.glob(os.path.join(args.result_dir, "*", "meta_*.json")))
    if not meta_paths:
        raise SystemExit(f"No meta_*.json found under {args.result_dir}")
    os.makedirs(args.out_root, exist_ok=True)

    converted = 0
    for meta_path in meta_paths:
        out = process_episode(meta_path, args.out_root)
        if out:
            converted += 1
            print(f"[ok] {meta_path} -> {out}")
    print(f"Converted {converted}/{len(meta_paths)} episodes into {args.out_root}")


if __name__ == "__main__":
    main()
