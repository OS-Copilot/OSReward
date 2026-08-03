# Mobile (Android) Trajectory Collection

AndroidWorld-based pipeline for collecting agent trajectories on a live
Android emulator. An LLM agent (T3A / M3A) runs tasks step by step; every
step's screenshot, accessibility tree, prompt, and raw model response are
persisted, then converted into the unified OSReward trajectory schema.

Pipeline stages:

| Stage | Script | Output |
|-------|--------|--------|
| collect | `run.py` | One `<task>_<instance>.pkl.gz` checkpoint per episode |
| extract | `extract_episodes.py` | Per-episode dir: step PNGs + full metadata JSON |
| reformat | `reformat.py` | Unified schema JSON (`trace_id`, `trajectory[]`, `orm_label`, ...) |
| annotate | `annotate_binary_reward.py` | Manual binary rewards filled into `orm_label` |

## Setup

1. **Android emulator.** Create an AVD named `AndroidWorldAvd` (Pixel 6,
   Tiramisu API 33 system image) via Android Studio, then launch it from the
   command line with gRPC enabled:

   ```bash
   ~/Library/Android/sdk/emulator/emulator -avd AndroidWorldAvd -no-snapshot -grpc 8554
   ```

2. **Python environment** (3.11+):

   ```bash
   conda create -n android_world python=3.11.8 && conda activate android_world
   pip install -r requirements.txt
   ```

3. **First run only:** pass `--perform_emulator_setup` to install the task
   apps and grant permissions (takes several minutes). Note that on every
   launch android_env downloads the accessibility-forwarder APK from
   `storage.googleapis.com`; a flaky connection to it fails the launch, so
   just retry.

## Collect

The default agent is `t3a_claude4` (`Claude4WrapperV2`), which speaks the
OpenAI chat-completion protocol to whatever endpoint `ANTHROPIC_BASE_URL`
points at (the official API or any OpenAI-compatible gateway):

```bash
export ANTHROPIC_API_KEY=<key>
export ANTHROPIC_BASE_URL=<https://gateway/ or https://api.anthropic.com/>

python run.py \
  --agent_name=t3a_claude4 \
  --model_name=claude-sonnet-4-5-20250929 \
  --tasks=ClockTimerEntryFiveMinutes,ExpenseAddBackdated \
  --checkpoint_dir=runs/demo
```

- `--model_name` overrides the agent's default model id.
- `--tasks` selects task templates; omit to run the whole registry.
- `--n_task_combinations=N` collects N random instances per template.
- Re-running with the same `--checkpoint_dir` resumes and skips finished
  episodes.
- GPT-family agents (`t3a_gpt4`, `m3a_gpt4v`) read `OPENAI_API_KEY` and
  `OPENAI_BASE_URL` instead.

Each API call is appended to `api_logs/api_calls.jsonl` (latency + token
usage, no payloads).

## Extract, reformat, annotate

```bash
python extract_episodes.py --run_dir runs/demo --out_dir runs/demo_extracted
python reformat.py --root runs/demo_extracted --out_root runs/demo_reformatted \
  --model_name claude-sonnet-4-5-20250929
python annotate_binary_reward.py --root runs/demo_reformatted [--dry_run]
```

- `extract_episodes.py` decodes each checkpoint into
  `<task>_<instance>/<task>_<instance>_<step:04d>.png` plus a JSON with the
  full episode record (a11y element lists, prompts, raw responses,
  `is_successful` from the task's own validator).
- `reformat.py` converts that into the unified schema: per-step
  `state.screenshot_path` / `state.a11y_tree`, the parsed `action`
  (click coordinates resolved from the a11y bbox), `thought`, `raw_response`,
  plus episode-level `orm_label` / `prm_label` placeholders.
- `annotate_binary_reward.py` writes manually verified 0/1 rewards into
  `orm_label.binary_reward` from its built-in per-task table; tasks not in
  the table are skipped.

## Tasks

The registry contains only the OSReward extension templates
(`android_world/task_evals/single/extensions/`, 148 registered) across Clock,
Calendar, Expense, Markor, Recipe, and Retro Music. Extension tasks come in
two generations: `extensions_<app>.py` subclass stock tasks with new
instructions; `extensions_<app>2.py` subclass validator-bearing base classes,
so `is_successful` is scored automatically. A few templates with broken
parameter generation are excluded in `registry.py` (see the comments there).

The stock AndroidWorld task classes for those six apps stay in the tree
solely as base classes; they are not registered. The unrelated AndroidWorld
task families (the other single-app tasks, MiniWoB, information retrieval,
composite) are removed.

## Provenance

Vendored from [google-research/android_world](https://github.com/google-research/android_world)
at upstream commit `c71a6b5` (2025-11-24). Local modifications:

- `registry.py`: registers only the extension tasks; unrelated task families
  (MiniWoB, information retrieval, composite, unused single-app tasks) are
  deleted from the tree, which also drops the protobuf build dependency.
- `agents/infer.py`: `OPENAI_BASE_URL` support for `Gpt4Wrapper`;
  `Claude4WrapperV1` (Anthropic Messages REST) and `Claude4WrapperV2`
  (OpenAI-protocol via `android_world/api/llm_api_utils.py`, records an
  Anthropic-shaped `raw_response` in trajectories).
- `task_evals/single/extensions/`: the custom task templates.
- `run.py`: `t3a_claude4` / `m3a_claude4` agents, `--model_name` flag,
  output defaults to `runs/`.
- Top-level `extract_episodes.py`, `reformat.py`, `annotate_binary_reward.py`.
- Removed upstream parts not needed for collection: `assets/` (demo media),
  `apps/` (Bazel sources for the companion APKs; prebuilt APKs are installed
  during emulator setup), and the Docker emulator stack (`Dockerfile`,
  `docker_setup/`, `server/`, `scripts/`). For those setups see the upstream
  [android_world](https://github.com/google-research/android_world) repo.
