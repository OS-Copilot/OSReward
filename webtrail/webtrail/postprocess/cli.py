"""`filter` subcommand: triage a finished run into a manifest.

Writes ``manifest.jsonl`` (one record per trajectory: bucket, score, flags)
next to the run and prints a bucket table. Downstream training-data selection
reads only the manifest.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from .buckets import triage_run
from .dedupe import mark_duplicates


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "filter", help="bucket, score, and dedupe a finished run"
    )
    parser.add_argument("--run", required=True, help="run directory (the --out of collect)")
    parser.add_argument("--min-score", type=float, default=0.5,
                        help="threshold for the keep recommendation (default 0.5)")
    parser.add_argument("--min-success", type=float, default=None,
                        help="additionally require judge success >= this "
                             "(applies only to trajectories that have judge.json)")
    parser.add_argument("--no-dedupe", action="store_true")
    parser.add_argument("--hamming", type=int, default=4,
                        help="screenshot-hash distance treated as identical")
    parser.set_defaults(handler=main)


def main(args: argparse.Namespace) -> None:
    run_dir = Path(args.run)
    triages = triage_run(run_dir)
    if not triages:
        raise SystemExit(f"no trajectories found under {run_dir}/trajectories")

    duplicates = 0
    if not args.no_dedupe:
        duplicates = mark_duplicates(run_dir, triages, hamming_threshold=args.hamming)

    manifest_path = run_dir / "manifest.jsonl"
    buckets = collections.Counter()
    kept = 0
    judged = 0
    with manifest_path.open("w", encoding="utf-8", errors="replace") as handle:
        for triage in triages:
            judge_path = run_dir / "trajectories" / triage.trajectory_id / "judge.json"
            judge = None
            if judge_path.exists():
                try:
                    raw = json.loads(judge_path.read_text())
                    judge = {k: raw.get(k) for k in
                             ("success", "efficiency", "self_correction")}
                    judged += 1
                except (ValueError, OSError):
                    pass

            keep = triage.bucket == "valid_candidate" and triage.score >= args.min_score
            if keep and args.min_success is not None and judge is not None:
                keep = (judge.get("success") or 0.0) >= args.min_success
            kept += keep
            buckets[triage.bucket] += 1
            result = triage.result
            handle.write(json.dumps({
                "trajectory_id": triage.trajectory_id,
                "bucket": triage.bucket,
                "quality_score": triage.score,
                "judge": judge,
                "keep": keep,
                "flags": triage.flags,
                "status": result.get("status"),
                "domain": result.get("domain"),
                "final_url": result.get("final_url"),
                "steps_taken": result.get("steps_taken"),
                "block": result.get("block"),
            }, ensure_ascii=False) + "\n")

    width = max(len(b) for b in buckets)
    print(f"\ntriaged {len(triages)} trajectories -> {manifest_path}")
    for bucket, count in buckets.most_common():
        print(f"  {bucket:<{width}}  {count}")
    print(f"\nduplicates marked: {duplicates}")
    if judged:
        print(f"trajectories with judge scores: {judged}")
    gate = f"score >= {args.min_score}"
    if args.min_success is not None:
        gate += f" & judge success >= {args.min_success}"
    print(f"keep (valid_candidate & {gate}): {kept}")
