"""Bucket assignment and rule-based quality scoring.

No judge model here: every signal comes from ``result.json`` plus a cheap scan
of the step artifacts. Trajectories land in exactly one bucket:

    valid_candidate     usable for downstream filtering / training
    blocked             ended on a captcha/challenge/denial page
    env_error           browser, network, or model infrastructure failure
    stale_loop          the page stopped responding to actions
    malformed_action    the model could not produce executable actions
    max_steps           ran out of budget without a `stop`
    missing_screenshot  too many steps lack the primary evidence
    empty               no steps were recorded at all
    duplicate           assigned later by the dedupe pass

The quality score is a 0-1 heuristic for ranking `valid_candidate` (and
`max_steps`) trajectories; weights are deliberately simple and documented
inline so they can be tuned without archaeology.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

MEANINGFUL_ACTIONS = {"click", "double_click", "type", "fill", "select_option",
                      "set_checked", "drag", "hotkey"}

STATUS_TO_BUCKET = {
    "blocked": "blocked",
    "env_error": "env_error",
    "stale_loop": "stale_loop",
    "agent_error": "malformed_action",
    "max_steps": "max_steps",
}


@dataclass
class Triage:
    trajectory_id: str
    bucket: str
    score: float
    flags: list[str] = field(default_factory=list)
    result: dict = field(default_factory=dict)


def _missing_screenshot_ratio(traj_dir: Path, steps_taken: int) -> float:
    if steps_taken <= 0:
        return 0.0
    missing = 0
    for index in range(steps_taken):
        if not (traj_dir / "screenshots" / f"step_{index:03d}.png").exists():
            missing += 1
    return missing / steps_taken


def triage_trajectory(traj_dir: Path) -> Triage | None:
    result_path = traj_dir / "result.json"
    if not result_path.exists():
        return Triage(traj_dir.name, "env_error", 0.0,
                      flags=["no_result_json"], result={})
    result = json.loads(result_path.read_text())

    status = result.get("status", "unknown")
    steps_taken = int(result.get("steps_taken") or 0)
    action_keys: list[str] = result.get("action_keys") or []
    counters: dict = result.get("counters") or {}
    flags: list[str] = []

    if steps_taken == 0 and not action_keys and status not in STATUS_TO_BUCKET:
        return Triage(traj_dir.name, "empty", 0.0, flags=["no_steps"], result=result)

    missing_ratio = _missing_screenshot_ratio(traj_dir, max(steps_taken, len(action_keys)))
    if missing_ratio > 0.34:
        return Triage(traj_dir.name, "missing_screenshot", 0.0,
                      flags=[f"missing_ratio={missing_ratio:.2f}"], result=result)

    bucket = STATUS_TO_BUCKET.get(status, "valid_candidate")
    if status == "completed":
        bucket = "valid_candidate"

    # ---- score ----
    score = 1.0
    if status != "completed":
        score -= 0.35
        flags.append(f"status={status}")
    if missing_ratio > 0:
        score -= 0.3 * missing_ratio
        flags.append(f"missing_screenshots={missing_ratio:.2f}")

    action_errors = int(counters.get("action_errors") or 0)
    if action_errors:
        score -= min(0.05 * action_errors, 0.25)
        flags.append(f"action_errors={action_errors}")

    stale = int(counters.get("stale_repeats") or 0)
    if action_keys and stale:
        score -= min(0.05 * stale, 0.25)
        flags.append(f"stale_repeats={stale}")

    parse_retries = int(counters.get("parse_retries") or 0)
    if parse_retries:
        score -= min(0.04 * parse_retries, 0.2)
        flags.append(f"parse_retries={parse_retries}")

    if action_keys:
        meaningful = sum(1 for key in action_keys if key in MEANINGFUL_ACTIONS)
        ratio = meaningful / len(action_keys)
        if ratio < 0.15:
            score -= 0.2                     # pure scroll/goto wandering
            flags.append(f"meaningful_ratio={ratio:.2f}")
        elif ratio > 0.4:
            score += 0.05

    if status == "completed" and result.get("stop_answer"):
        score += 0.05

    score = max(0.0, min(1.0, score))
    return Triage(traj_dir.name, bucket, round(score, 3), flags=flags, result=result)


def triage_run(run_dir: Path) -> list[Triage]:
    trajectories_dir = run_dir / "trajectories"
    if not trajectories_dir.is_dir():
        return []
    out = []
    for traj_dir in sorted(trajectories_dir.iterdir()):
        if traj_dir.is_dir():
            triage = triage_trajectory(traj_dir)
            if triage:
                out.append(triage)
    return out
