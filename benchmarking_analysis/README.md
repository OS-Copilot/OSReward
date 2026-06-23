# Benchmark Analysis — VLM-as-a-Judge for agent trajectories

Measures how well a reward model / LLM judge decides whether a GUI-agent
trajectory **succeeded**, on **out-of-distribution** benchmarks
(OSWorld · WindowsAgentArena · WebArena · AndroidWorld; macOS uses the same path).

Per trajectory, the judge sees the task instruction + the last *N* screenshots
(click points circled) + the agent's thought/action history, and outputs
`Judge: SUCCESS|FAIL`. That is scored against the benchmark's ground-truth label
(binary accuracy + strict/lenient recall).

Pipeline: **prepare** (raw → judge-ready) → **judge** (call the model) →
**analyze** (CSV + figures).

## Install

```bash
python3 -m pip install -r requirements.txt        # openai, Pillow, matplotlib, numpy
cp .env.example .env                              # then fill in your endpoint + key
```

`.env` (auto-loaded) sets the OpenAI-compatible endpoint:

```
MODEL_REQUEST_URL=https://your-endpoint/v1
API_KEY=sk-...
# API_KEYS=sk-a,sk-b      # optional: several keys -> more concurrency
# OOD_DATA_ROOT=/path      # optional: data/output root (default ./data)
```

## Use

Run the three stages per dataset. `--path` points at your raw rollouts; a tiny
`*-sample` per platform is bundled under `data/raw/` as a format reference.

```bash
# 1) prepare: raw rollouts -> judge-ready inputs
python3 -m benchmark_analysis.cli.prepare --platform osworld --agent kimi --kind dir \
    --path data/raw/osworld/kimi-sample

# 2) judge: score SUCCESS/FAIL with a model (resume-safe; --limit N for a subset)
python3 -m benchmark_analysis.cli.judge --platform osworld --agent kimi \
    --model gemini-3-flash-preview --concurrency 12

# 3) analyze: pool per model -> CSV + leaderboard/bias/error figures
python3 -m benchmark_analysis.cli.analyze --platform osworld
```

Outputs land in `data/analysis/<platform>/`: `data/*.csv` (metrics) and
`figures/*.png`. The CSV reports `acc`, `sRec` (= P(judge SUCCESS | gold SUCCESS),
lower = stricter), `fRec` (= P(judge FAIL | gold FAIL), lower = more lenient), and
confusion counts (`fp` = lenient errors, `fn` = strict errors).

### Input formats (per `--platform`)

| platform | raw form (`--path`) | `--kind` | golden label |
|---|---|---|---|
| osworld / windows | dir or zip of `<domain>/<task_id>/{result.txt, traj.jsonl, runtime.log, step_*.png}` | `dir` / `zip` | `result.txt` score ≥ `--success_threshold` (0.99) |
| webarena | a rollout-set dir — merged-JSONL (`judged_*.jsonl`+`images/`) **or** legacy `judgements/` | n/a | recorded `task_success` / stored label |
| androidworld | a normalized merged-JSONL **file** (e.g. `results/merged_*.jsonl`; screenshots resolve against the dir two levels up, or `--aw_root`) | n/a | recorded `task_success` |

Two gotchas:
- **OSWorld claude**: trajectories carry no instruction — prepare an agent that
  does (e.g. `kimi`) into the same data root **first**; claude reuses its
  instruction map.
- **WebArena**: the two raw forms are auto-detected; same command either way.

## As a library

```python
from benchmark_analysis import adapters, prepare, judge, metrics, viz
prepare.prepare_desktop("osworld", "kimi", adapters.DirSource(".../kimi"))
judge.judge_platform("osworld", "kimi", "gemini-3-flash-preview")
stats, per_agent = metrics.load_results("osworld", "v1", "last5")
viz.make_figures("osworld", stats, "v1", "last5")
```

## Layout

```
benchmark_analysis/   config · adapters · prepare · judge · metrics · viz · cli/ · prompts/
data/                 raw/ (inputs; ships small samples) · analysis/ (outputs)
```

System prompts live in `benchmark_analysis/prompts/` (default `multi_v4.txt`,
selectable with `--prompt`). No external services or logging dependencies.
