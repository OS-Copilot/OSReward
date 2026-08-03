#!/usr/bin/env python3
"""
Extract screenshots for all episodes in an AndroidWorld run checkpoint directory
and save episode metadata to JSON.

For each episode saved in the IncrementalCheckpointer, this script creates a
directory named "{task_template}_{instance_id}" under the output directory.

It generates:
1. Per-step images: {task_template}_{instance_id}_{step_id}.png
2. Episode metadata: {task_template}_{instance_id}.json

Behavior:
- Uses episode_data["screenshot"] as the primary image source, falling back to
  "raw_screenshot" / "before_screenshot" for agents that record those instead.
- Preserves result fields like "response", "parsed", "step_history", "summary",
  and "repeat_time" in JSON output.

Usage:
  python extract_episodes.py --run_dir /path/to/run_dir --out_dir /path/to/output_dir
"""
from __future__ import annotations

import argparse
import base64
import copy
import io
import json
import os
import re
from typing import Any, Optional

import numpy as np
from PIL import Image

from android_world import checkpointer as aw_checkpointer
from android_world import constants as aw_constants


class AndroidWorldEncoder(json.JSONEncoder):
  """JSON encoder for NumPy types, binary blobs, and custom objects."""

  def default(self, obj):
    if isinstance(obj, np.integer):
      return int(obj)
    if isinstance(obj, np.floating):
      return float(obj)
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    if isinstance(obj, bytes):
      return "<binary_data_omitted>"
    if hasattr(obj, "to_dict"):
      return obj.to_dict()
    if hasattr(obj, "__dict__"):
      return obj.__dict__
    return str(obj)


def sanitize_name(name: str) -> str:
  """Sanitize directory/file name."""
  return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def stable_episode_timestamp(ep: dict[str, Any]) -> str:
  """Return a deterministic timestamp-like suffix for an episode identity."""
  finish_dtime = ep.get(aw_constants.EpisodeConstants.FINISH_DTIME)
  if hasattr(finish_dtime, "strftime"):
    return finish_dtime.strftime("%Y%m%dT%H%M%S")
  if finish_dtime is not None:
    return sanitize_name(str(finish_dtime))
  run_time = ep.get(aw_constants.EpisodeConstants.RUN_TIME)
  if run_time is not None:
    return sanitize_name(str(run_time))
  return "no_timestamp"


def decode_frame_to_numpy(frame: Any) -> Optional[np.ndarray]:
  """Decode a frame into a numpy uint8 HWC array."""
  if frame is None:
    return None
  if isinstance(frame, np.ndarray):
    return frame
  if isinstance(frame, (bytes, bytearray)):
    img = Image.open(io.BytesIO(frame)).convert("RGB")
    return np.array(img)
  if isinstance(frame, str):
    decoded = base64.b64decode(frame)
    img = Image.open(io.BytesIO(decoded)).convert("RGB")
    return np.array(img)
  if isinstance(frame, Image.Image):
    return np.array(frame.convert("RGB"))
  raise ValueError(f"Unsupported frame type: {type(frame)}")


def save_image(arr: np.ndarray, path: str) -> None:
  """Save a uint8 HWC numpy array as PNG."""
  if arr is None:
    return
  if not isinstance(arr, np.ndarray):
    raise ValueError("Expected numpy.ndarray")
  if arr.dtype != np.uint8:
    if np.issubdtype(arr.dtype, np.floating):
      arr = np.clip((arr * 255.0).round(), 0, 255).astype(np.uint8)
    else:
      arr = arr.astype(np.uint8)
  arr = np.ascontiguousarray(arr)
  Image.fromarray(arr).save(path)


def _placeholder_for_list(value: Any) -> str:
  if isinstance(value, list):
    return f"<Images extracted to .png files (count: {len(value)})>"
  return "<Images extracted to .png files>"


def save_episode_json(ep: dict[str, Any], out_dir: str, repo_name: str) -> None:
  """Save episode dictionary to JSON while stripping heavy image payloads."""
  json_path = os.path.join(out_dir, f"{repo_name}.json")

  ep_copy = copy.copy(ep)
  episode_data = ep_copy.get(aw_constants.EpisodeConstants.EPISODE_DATA, {})
  if isinstance(episode_data, dict):
    ep_data_clean = copy.copy(episode_data)
    image_keys = [
        "screenshot",
        "raw_screenshot",
        "before_screenshot",
        "after_screenshot",
    ]
    for key in image_keys:
      if key in ep_data_clean:
        ep_data_clean[key] = _placeholder_for_list(ep_data_clean[key])
    ep_copy[aw_constants.EpisodeConstants.EPISODE_DATA] = ep_data_clean

  try:
    with open(json_path, "w", encoding="utf-8") as f:
      json.dump(ep_copy, f, cls=AndroidWorldEncoder, indent=2)
  except Exception as e:
    print(f"  [!] Failed to save JSON {json_path}: {e}")


def _find_num_steps(episode_data: dict[str, Any]) -> int:
  n_steps = 0
  if aw_constants.STEP_NUMBER in episode_data and isinstance(
      episode_data[aw_constants.STEP_NUMBER], (list, tuple)
  ):
    return len(episode_data[aw_constants.STEP_NUMBER])
  for value in episode_data.values():
    if isinstance(value, (list, tuple)):
      n_steps = max(n_steps, len(value))
  return n_steps


def _get_screenshot_list(episode_data: dict[str, Any], n_steps: int) -> list[Any]:
  """Get the screenshot series, with fallbacks for other agents' field names."""
  for key in ("screenshot", "raw_screenshot", "before_screenshot"):
    value = episode_data.get(key)
    if isinstance(value, list):
      return value
  return [None] * n_steps


def extract_episode_artifacts(ep: dict[str, Any], out_root: str) -> None:
  """Extract per-step screenshots and save JSON data for one episode."""
  task_template = ep.get(aw_constants.EpisodeConstants.TASK_TEMPLATE, "unknown_task")
  instance_id = ep.get(aw_constants.EpisodeConstants.INSTANCE_ID, 0)

  episode_timestamp = stable_episode_timestamp(ep)
  dir_name = sanitize_name(f"{task_template}_{instance_id}_{episode_timestamp}")
  out_dir = os.path.join(out_root, dir_name)
  os.makedirs(out_dir, exist_ok=True)

  save_episode_json(ep, out_dir, dir_name)

  episode_data = ep.get(aw_constants.EpisodeConstants.EPISODE_DATA)
  if not isinstance(episode_data, dict):
    print(f"  [!] No episode_data for {dir_name}, skipping images.")
    return

  n_steps = _find_num_steps(episode_data)
  if n_steps == 0:
    return

  screenshot_list = _get_screenshot_list(episode_data, n_steps)
  for i in range(n_steps):
    frame_raw = screenshot_list[i] if i < len(screenshot_list) else None
    if frame_raw is None:
      continue
    try:
      arr = decode_frame_to_numpy(frame_raw)
    except Exception as e:
      print(f"  [!] Failed to decode screenshot for {dir_name} step {i}: {e}")
      continue
    if arr is None:
      continue
    fname = f"{sanitize_name(task_template)}_{instance_id}_{i:04d}.png"
    out_path = os.path.join(out_dir, fname)
    try:
      save_image(arr, out_path)
    except Exception as e:
      print(f"  [!] Failed to save image {out_path}: {e}")


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Extract screenshots and JSON data from run checkpoints."
  )
  parser.add_argument(
      "--run_dir",
      required=True,
      help="Path to the run directory (checkpoint_dir).",
  )
  parser.add_argument(
      "--out_dir",
      help="Where to write output. Defaults to run_dir if omitted.",
  )
  parser.add_argument(
      "--only_tasks",
      nargs="*",
      help="Optional list of task_template names to restrict extraction.",
  )
  args = parser.parse_args()

  run_dir = os.path.expanduser(args.run_dir)
  out_root = os.path.expanduser(args.out_dir) if args.out_dir else run_dir

  if not os.path.isdir(run_dir):
    raise FileNotFoundError(f"Run directory not found: {run_dir}")

  print(f"Loading episodes from {run_dir}...")
  cp = aw_checkpointer.IncrementalCheckpointer(run_dir)
  episodes = cp.load(fields=None)
  if not episodes:
    print("No episodes found in run directory.")
    return

  print(f"Found {len(episodes)} episodes. Extracting artifacts into {out_root} ...")
  for idx, ep in enumerate(episodes):
    task_template = ep.get(aw_constants.EpisodeConstants.TASK_TEMPLATE, "")
    if args.only_tasks and task_template not in args.only_tasks:
      continue
    instance_id = ep.get(aw_constants.EpisodeConstants.INSTANCE_ID, 0)
    print(f"[{idx + 1}/{len(episodes)}] Processing: {task_template} instance {instance_id}")
    try:
      extract_episode_artifacts(ep, out_root)
    except Exception as e:
      print(f"  [!] Error processing episode {task_template}_{instance_id}: {e}")

  print("Done.")


if __name__ == "__main__":
  main()
