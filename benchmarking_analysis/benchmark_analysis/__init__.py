"""benchmark_analysis — standalone VLM-as-a-judge for OOD agent-trajectory benchmarks.

Pipeline: ``prepare`` (raw rollouts -> judge-ready) -> ``judge`` (VLM scores
SUCCESS/FAIL vs the benchmark's golden label) -> ``analyze`` (metrics + figures).

Quick start (library)::

    from benchmark_analysis import adapters, prepare, judge
    stats = prepare.prepare_desktop("osworld", "kimi",
                                    adapters.DirSource("data/raw/osworld/kimi"))
    judge.judge_platform("osworld", "kimi", "gemini-3-flash-preview")

CLI::

    python -m benchmark_analysis.cli.prepare --platform osworld --agent kimi --kind dir --path ...
    python -m benchmark_analysis.cli.judge   --platform osworld --agent kimi --model gemini-3-flash-preview
    python -m benchmark_analysis.cli.analyze --platform osworld
"""
from . import adapters, config, judge, metrics, prepare, viz

__all__ = ["adapters", "config", "judge", "metrics", "prepare", "viz"]
