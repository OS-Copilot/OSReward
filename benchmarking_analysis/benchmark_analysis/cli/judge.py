"""Run the VLM judge over a prepared (platform, agent).

Example::

    python -m benchmark_analysis.cli.judge --platform osworld --agent kimi \
        --model gemini-3-flash-preview --concurrency 12 --version v1

Reads analysis/<platform>/judge_ready/<agent>/ and writes
analysis/<platform>/results/judge_<version>_<setting>_<agent>_<model>.jsonl
(resume-safe: re-running continues and retries failed traces).
"""
import argparse

from .. import config, judge


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--platform", required=True)
    ap.add_argument("--agent", required=True)
    ap.add_argument("--model", default="gemini-3-flash-preview", help="judge (reward) model")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--setting", default="last5")
    ap.add_argument("--prompt", default=config.DEFAULT_PROMPT, help="system prompt file in prompts/")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--limit", type=int, default=0, help="judge only the first N (0 = all)")
    ap.add_argument("--sample", default="head", choices=["head", "random"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    judge.judge_platform(args.platform, args.agent, args.model, version=args.version,
                         setting=args.setting, prompt=args.prompt,
                         concurrency=args.concurrency, timeout=args.timeout,
                         limit=args.limit, sample=args.sample, seed=args.seed)


if __name__ == "__main__":
    main()
