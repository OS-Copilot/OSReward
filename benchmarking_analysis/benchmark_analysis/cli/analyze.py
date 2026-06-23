"""Aggregate judge results into a metrics CSV + leaderboard / bias / error figures.

Example::

    python -m benchmark_analysis.cli.analyze --platform osworld --version v1 --setting last5

Pools rows per judge model across the platform's rollout agents and writes to
analysis/<platform>/{data,figures}/.
"""
import argparse

from .. import metrics, viz

# Optional in-domain (in-distribution) accuracy anchors per judge model, used to
# draw the OOD gap on the webarena leaderboard. Source: OSReward in-domain eval.
IN_DOMAIN = {
    "webarena": {
        "gemini-3-flash-preview": 85.6, "gemini-3.1-pro-preview": 86.0,
        "claude-sonnet-4-6": 86.0, "gpt-5-mini": 84.8,
        "qwen3.5-397b-a17b": 85.3, "qwen3-vl-30b-a3b-instruct": 69.2,
    },
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--platform", required=True)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--setting", default="last5")
    args = ap.parse_args()

    stats, per_agent = metrics.load_results(args.platform, args.version, args.setting)
    if not stats:
        print(f"[{args.platform}] no result files found for {args.version}/{args.setting}.")
        return
    in_domain = IN_DOMAIN.get(args.platform)
    csv = viz.write_metrics_csv(args.platform, stats, per_agent, args.version,
                                args.setting, in_domain)
    figs = viz.make_figures(args.platform, stats, args.version, args.setting, in_domain)

    print(f"[{args.platform}] models={len(stats)}")
    for m, s in sorted(stats.items(), key=lambda kv: -kv[1]["acc"]):
        print(f"  {m:32s} n={s['n']:4d} acc={s['acc'] * 100:5.1f}% "
              f"sRec={s['sRec']:.3f} fRec={s['fRec']:.3f} (FP={s['fp']} FN={s['fn']})")
    print("wrote:", csv)
    for p in figs:
        print("      ", p)


if __name__ == "__main__":
    main()
