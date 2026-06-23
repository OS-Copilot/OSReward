"""Prepare raw rollouts into judge-ready inputs.

Examples::

    # OSWorld / WindowsAgentArena (unpacked dir of <domain>/<task_id>/ tasks)
    python -m benchmark_analysis.cli.prepare --platform osworld --agent kimi \
        --kind dir --path data/raw/osworld/kimi-sample
    python -m benchmark_analysis.cli.prepare --platform windows --agent interngui \
        --kind dir --path data/raw/windows/interngui-sample

    # OSWorld zip
    python -m benchmark_analysis.cli.prepare --platform osworld --agent kimi \
        --kind zip --path data/raw/osworld/kimi.zip

    # WebArena (merged-JSONL rollout set directory)
    python -m benchmark_analysis.cli.prepare --platform webarena --agent gpt5gemini3f \
        --path data/raw/webarena/gpt5gemini3f-sample
"""
import argparse
import os

from .. import adapters, prepare


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--platform", required=True, help="osworld | windows | webarena | macos")
    ap.add_argument("--agent", required=True, help="rollout-agent label (output subdir)")
    ap.add_argument("--path", required=True, help="zip / dir (desktop) or rollout-set dir (webarena)")
    ap.add_argument("--kind", default="dir", choices=["dir", "zip"], help="desktop source kind")
    ap.add_argument("--last_n", type=int, default=5, help="screenshots kept (the last N)")
    ap.add_argument("--success_threshold", type=float, default=prepare.SUCCESS_THRESHOLD,
                    help="desktop: golden=SUCCESS iff verifier score >= this")
    ap.add_argument("--ossymphony_full", action="store_true",
                    help="fold ossymphony code/search sub-agent outputs into history")
    ap.add_argument("--judge_subdir", default=None,
                    help="webarena legacy: judgements/<subdir> to read (default: auto)")
    ap.add_argument("--limit", type=int, default=0, help="prepare only the first N tasks (0 = all)")
    args = ap.parse_args()

    path = os.path.abspath(args.path)
    if args.platform == "webarena":
        # auto-detect: top-level *.jsonl = merged rollout set; else judgements/ = legacy
        has_jsonl = os.path.isdir(path) and any(f.endswith(".jsonl") for f in os.listdir(path))
        if has_jsonl:
            stats = prepare.prepare_webarena(args.agent, path, args.last_n, limit=args.limit)
        elif os.path.isdir(os.path.join(path, "judgements")):
            print("(webarena legacy 'judgements' format)")
            stats = prepare.prepare_webarena_legacy(args.agent, path, args.judge_subdir,
                                                    limit=args.limit)
        else:
            raise SystemExit(f"webarena: no *.jsonl and no judgements/ under {path}")
    else:
        source = adapters.ZipSource(path) if args.kind == "zip" else adapters.DirSource(path)
        stats = prepare.prepare_desktop(args.platform, args.agent, source,
                                        last_n=args.last_n,
                                        success_threshold=args.success_threshold,
                                        ossymphony_full=args.ossymphony_full, limit=args.limit)
    print(f"platform={args.platform} agent={args.agent} "
          f"ok={stats['ok']} skipped={stats.get('skipped', 0)} "
          f"gold={stats['gold']} domains={stats['domains']}")
    if stats.get("formats"):
        print(f"formats={stats['formats']} missing_instruction={stats.get('no_instruction', 0)}")


if __name__ == "__main__":
    main()
