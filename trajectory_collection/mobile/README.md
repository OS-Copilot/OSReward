# Mobile (Android) Trajectory Collection

AndroidWorld-based pipeline for collecting agent trajectories on a live
Android emulator. A screenshot-only tool-call agent
(`android_world/agents/seeact.py:ToolCallAgent`) runs tasks step by step
against any OpenAI-compatible model endpoint; every step's screenshot, raw
model response, and parsed action are persisted, then converted into the
unified OSReward trajectory schema.

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

3. **Model endpoint.** Copy `.env.example` to `.env` and fill in
   `MODEL_NAME` / `MODEL_BASE_URL` / `MODEL_API_KEY`. The model must be listed
   in `android_world/agents/model_profiles.py` (currently the qwen3-vl and
   gemini-3 families), or pass `--model_profile qwen3vl|gemini3` to force a
   prompt format.

4. **First run only:** pass `--perform_emulator_setup` to install the task
   apps and grant permissions (takes several minutes). Note that on every
   launch android_env downloads the accessibility-forwarder APK from
   `storage.googleapis.com`; a flaky connection to it fails the launch, so
   just retry.

## Collect

```bash
python run.py \
  --agent_name=toolcall \
  --tasks=ClockTimerEntryFiveMinutes,ExpenseAddBackdated \
  --checkpoint_dir=runs/demo
```

- The agent, model, and endpoint come from `.env` (or the `--model_name`,
  `--model_base_url`, `--model_api_key` flags).
- `--tasks` selects task templates; omit to run the whole active suite.
- `EXT_SUITE=aw|new_app` (env var) switches which task set the registry
  serves; see [Tasks](#tasks).
- `--n_task_combinations=N` collects N random instances per template.
- `--freeze_datetime` pins the device clock to the AndroidWorld benchmark
  time (October 2023); leave it off for the live-web `new_app` tasks.
- Re-running with the same `--checkpoint_dir` resumes and skips finished
  episodes.

Each API call is logged locally under `api_logs/` (latency + token usage) by
`android_world/api/local_api_logger`.

## Extract, reformat, annotate

```bash
python extract_episodes.py --run_dir runs/demo --out_dir runs/demo_extracted
python reformat.py --root runs/demo_extracted --out_root runs/demo_reformatted
python annotate_binary_reward.py --root runs/demo_reformatted [--dry_run]
```

- `extract_episodes.py` decodes each checkpoint into
  `<task>_<instance>_<timestamp>/` with per-step PNGs plus a JSON holding the
  full episode record (raw responses, parsed actions, step history, and
  `is_successful` from the task's validator).
- `reformat.py` converts that into the unified schema: per-step
  `state.screenshot_path`, the parsed `action` (pixel coordinates), the
  model's `<thinking>` / `<conclusion>` as `thought`, plus episode-level
  `orm_label` (validator verdict in `score`, manual label slot in
  `binary_reward`).
- `annotate_binary_reward.py` writes manually verified 0/1 rewards into
  `orm_label.binary_reward` from its built-in per-task table; tasks not in
  the table are skipped.

## Tasks

The registry serves only the OSReward extension tasks
(`--suite_family=android_world_extension`, the default and only family). Two
task sets, selected with `EXT_SUITE`:

- **`aw`** (default): 131 verified extension tasks over the AndroidWorld apps
  (Clock, Calendar, Expense, Markor, Recipe, Retro Music), defined in
  `android_world/task_evals/single/extensions/`. These inherit real
  `is_successful` validators.
- **`new_app`**: 115 tasks over newly added apps (Chrome, Gmail, Google Maps,
  YouTube, Yahoo Finance) plus cross-app flows, defined in
  `android_world/task_evals/single/{chrome,gmail,google_maps,youtube,yahoo_finance}.py`
  and freeform additions in the calendar/expense/retro_music modules. Most of
  their validators are placeholders (`is_successful` returns 1.0): the value
  of these trajectories is the recorded process, and rewards should come from
  the annotate stage or a judge.

`SettingsGoogleAppNotificationsAndData`
(`extensions/extensions_settings.py`) is registered in both sets; its
validator reads the notification-channel importance and background-data
policy over adb. The stock AndroidWorld task classes for the six base apps
stay in the tree solely as base classes; the unrelated AndroidWorld task
families (other single-app tasks, MiniWoB, information retrieval, composite)
are removed.

## Provenance

Vendored from the internal `android_world-publication-cleanup` branch of the
AndroidWorld fork (upstream base:
[google-research/android_world](https://github.com/google-research/android_world)
@ `c71a6b5`). Local modifications on top of that branch:

- Only the tool-call collection agent is kept; the upstream agents (M3A, T3A,
  SeeAct, human/random) and their wrappers are removed, along with the
  MiniWoB / information-retrieval / composite task families (this also drops
  the protobuf build dependency).
- `registry.py` registers only the extension suites; `EXT_SUITE` is
  env-configurable.
- `api/llm_api_utils.py` trimmed to the OpenAI-compatible path used by the
  agent; API keys are no longer written to local logs.
- `utils/file_utils.py` restored to upstream's concurrency-safe per-call temp
  directories.
- `pysqlite3` made an optional dependency (falls back to stdlib `sqlite3`).
- Top-level `extract_episodes.py` (renamed from `read_pkl.py`),
  `reformat.py`, `annotate_binary_reward.py`.
- Removed upstream parts not needed for collection: `assets/`, `apps/`, the
  Docker emulator stack (`Dockerfile`, `docker_setup/`, `server/`,
  `scripts/`), and docs. For those see the upstream repo.
