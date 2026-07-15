"""Lightweight trajectory deduplication — no embeddings required.

Two trajectories are considered duplicates when ALL of these agree:

* normalized instruction (lowercased, whitespace/punctuation collapsed)
* start domain
* normalized final URL (query string and fragment stripped)
* the action-key sequence
* the final screenshot's perceptual hash within a small Hamming distance

Exact-key grouping handles the first four; the perceptual hash breaks ties
inside a group so that two runs that genuinely ended on different pages are
both kept.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlparse

from .. import imutil
from ..types import domain_of
from .buckets import Triage

_NORM_RE = re.compile(r"[^a-z0-9 ]+")


def _normalize_instruction(text: str) -> str:
    return " ".join(_NORM_RE.sub(" ", text.lower()).split())


def _normalize_url(url: str | None) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    return f"{parsed.netloc}{parsed.path}".rstrip("/").lower()


def _final_screenshot_hash(traj_dir: Path, steps_taken: int) -> int | None:
    for index in range(max(steps_taken - 1, 0), -1, -1):
        path = traj_dir / "screenshots" / f"step_{index:03d}.png"
        if path.exists():
            try:
                return imutil.dhash(imutil.load_png(path.read_bytes()))
            except OSError:
                return None
    return None


def mark_duplicates(run_dir: Path, triages: list[Triage],
                    hamming_threshold: int = 4) -> int:
    """Rewrites `bucket` to ``duplicate`` in-place; returns how many were marked."""
    groups: dict[tuple, list[tuple[Triage, int | None]]] = {}

    for triage in triages:
        if triage.bucket != "valid_candidate":
            continue
        result = triage.result
        task_path = run_dir / "trajectories" / triage.trajectory_id / "task.json"
        try:
            instruction = json.loads(task_path.read_text()).get("instruction", "")
        except (OSError, ValueError):
            instruction = ""
        key = (
            _normalize_instruction(instruction),
            domain_of(result.get("start_url") or ""),
            _normalize_url(result.get("final_url")),
            tuple(result.get("action_keys") or []),
        )
        shot_hash = _final_screenshot_hash(
            run_dir / "trajectories" / triage.trajectory_id,
            int(result.get("steps_taken") or 0),
        )
        groups.setdefault(key, []).append((triage, shot_hash))

    marked = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        # keep the best-scored member of each visually-identical cluster
        members.sort(key=lambda pair: -pair[0].score)
        kept: list[tuple[Triage, int | None]] = []
        for triage, shot_hash in members:
            duplicate_of = None
            for kept_triage, kept_hash in kept:
                if shot_hash is None or kept_hash is None:
                    duplicate_of = kept_triage      # identical keys, no image to differ
                    break
                if imutil.hamming(shot_hash, kept_hash) <= hamming_threshold:
                    duplicate_of = kept_triage
                    break
            if duplicate_of is None:
                kept.append((triage, shot_hash))
            else:
                triage.bucket = "duplicate"
                triage.flags.append(f"duplicate_of={duplicate_of.trajectory_id}")
                marked += 1
    return marked
