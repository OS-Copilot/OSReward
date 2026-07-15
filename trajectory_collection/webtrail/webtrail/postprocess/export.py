"""`export` subcommand: turn collected trajectories into a training corpus.

Reads a run (optionally gated by its ``manifest.jsonl`` so only kept, deduped
trajectories are emitted) and writes one record per trajectory in a chat format
ready for SFT / reward-model pipelines:

* ``messages`` — OpenAI-style ``{"role", "content"}`` turns; screenshots are
  embedded as ``image_url`` data URIs so the file is self-contained.
* ``sharegpt`` — the same conversation in ShareGPT ``{"from", "value"}`` turns,
  with image placeholders and a parallel ``images`` list of file paths.

Each turn pairs the observation the agent saw with the action it took, so the
export is a faithful multi-turn transcript, not just the final answer.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _kept_ids(run_dir: Path, gate: str) -> set[str] | None:
    """Trajectory ids to keep per manifest.jsonl, or None to take everything."""
    manifest = run_dir / "manifest.jsonl"
    if gate == "all" or not manifest.exists():
        if gate != "all":
            logger.warning("no manifest.jsonl (run `webtrail filter` first); exporting all")
        return None
    ids: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if gate == "keep" and not rec.get("keep"):
            continue
        if gate == "success":
            judge = rec.get("judge") or {}
            if (judge.get("success") or 0.0) < 1.0:
                continue
        ids.add(rec["trajectory_id"])
    return ids


def _step_shot(traj_dir: Path, stem: str) -> Path | None:
    for sub in ("screenshots", "annotated"):
        p = traj_dir / sub / f"{stem}.png"
        if p.exists():
            return p
    return None


def _turns(traj_dir: Path) -> list[dict]:
    """Ordered (screenshot_path, user_text, assistant_text) triples for one run."""
    turns = []
    for agent_path in sorted((traj_dir / "agent").glob("step_*.json")):
        stem = agent_path.stem
        agent = _read_json(agent_path)
        state = _read_json(traj_dir / "states" / f"{stem}.json")
        action = {"action": agent.get("action"),
                  "args": {k: v for k, v in (agent.get("args") or {}).items()
                           if k != "analysis"}}
        assistant = agent.get("analysis") or ""
        assistant = (assistant + "\n\n```json\n"
                     + json.dumps(action, ensure_ascii=False) + "\n```").strip()
        turns.append({
            "shot": _step_shot(traj_dir, stem),
            "url": state.get("url") or "",
            "assistant": assistant,
        })
    return turns


def _data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def export_messages(traj_dir: Path, embed_images: bool) -> dict | None:
    task = _read_json(traj_dir / "task.json")
    result = _read_json(traj_dir / "result.json")
    turns = _turns(traj_dir)
    if not turns:
        return None
    messages: list[dict] = [
        {"role": "system",
         "content": "You operate a web browser to complete the user's task."}]
    for i, turn in enumerate(turns):
        content: list[dict] = []
        if i == 0:
            content.append({"type": "text", "text": task.get("instruction") or ""})
        if turn["shot"] is not None and embed_images:
            content.append({"type": "image_url",
                            "image_url": {"url": _data_uri(turn["shot"])}})
        elif turn["shot"] is not None:
            content.append({"type": "image_url",
                            "image_url": {"url": turn["shot"].as_posix()}})
        content.append({"type": "text", "text": f"(step {i + 1}) URL: {turn['url']}"})
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    return {
        "id": traj_dir.name,
        "instruction": task.get("instruction"),
        "status": result.get("status"),
        "answer": result.get("stop_answer"),
        "messages": messages,
    }


def export_sharegpt(traj_dir: Path) -> dict | None:
    task = _read_json(traj_dir / "task.json")
    result = _read_json(traj_dir / "result.json")
    turns = _turns(traj_dir)
    if not turns:
        return None
    conv, images = [], []
    for i, turn in enumerate(turns):
        parts = []
        if i == 0:
            parts.append(task.get("instruction") or "")
        if turn["shot"] is not None:
            parts.append("<image>")
            images.append(turn["shot"].as_posix())
        parts.append(f"(step {i + 1}) URL: {turn['url']}")
        conv.append({"from": "human", "value": "\n".join(parts)})
        conv.append({"from": "gpt", "value": turn["assistant"]})
    return {
        "id": traj_dir.name,
        "conversations": conv,
        "images": images,
        "status": result.get("status"),
        "answer": result.get("stop_answer"),
    }


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "export", help="export trajectories to a chat training format"
    )
    parser.add_argument("--run", required=True, help="run directory")
    parser.add_argument("--out", help="output JSONL (default <run>/export.<fmt>.jsonl)")
    parser.add_argument("--format", choices=["messages", "sharegpt"], default="messages")
    parser.add_argument("--gate", choices=["keep", "success", "all"], default="keep",
                        help="which trajectories to include per manifest.jsonl: "
                             "keep = filter's keep flag (default); success = judge "
                             "SUCCESS; all = every trajectory")
    parser.add_argument("--no-embed-images", action="store_true",
                        help="reference screenshots by path instead of embedding "
                             "them as data URIs (messages format only)")
    parser.set_defaults(handler=main)


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_dir = Path(args.run)
    traj_root = run_dir / "trajectories"
    if not traj_root.exists():
        raise SystemExit(f"no trajectories/ under {run_dir}")

    keep = _kept_ids(run_dir, args.gate)
    out = Path(args.out) if args.out else run_dir / f"export.{args.format}.jsonl"
    written = skipped = 0
    with out.open("w", encoding="utf-8", errors="replace") as handle:
        for traj_dir in sorted(d for d in traj_root.iterdir() if d.is_dir()):
            if keep is not None and traj_dir.name not in keep:
                skipped += 1
                continue
            if args.format == "sharegpt":
                record = export_sharegpt(traj_dir)
            else:
                record = export_messages(traj_dir, embed_images=not args.no_embed_images)
            if record is None:
                skipped += 1
                continue
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    print(f"wrote {written} trajectories -> {out}"
          + (f"  ({skipped} skipped)" if skipped else ""))
