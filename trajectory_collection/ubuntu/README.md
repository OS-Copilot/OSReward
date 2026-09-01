# Ubuntu Trajectory Collection

OSWorld-based pipeline for collecting agent trajectories on an Ubuntu desktop
VM. A screenshot-only tool-call agent (`agents/toolcall_agent.py:ToolCallAgent`)
runs OSWorld-format tasks step by step against any OpenAI-compatible model
endpoint; every step's screenshot, raw model response, and parsed action are
persisted together with the task validator's verdict, then converted into the
unified OSReward trajectory schema.

Pipeline stages:

| Stage | Script | Output |
|-------|--------|--------|
| collect | `run.py` | Per episode: `step_<i>.png` + `traj.jsonl` + `result.txt` + `meta_<task>.json` |
| reformat | `reformat.py` | Unified schema JSON (`trace_id`, `trajectory[]`, `orm_label`, ...) |
| judge / export | root [`eval_pipeline/`](../../eval_pipeline/) and [`export_judged_sft.py`](../export_judged_sft.py) | Judge JSONL, SFT bundles |

## Setup

1. **Python environment** (3.10+):

   ```bash
   pip install -r requirements.txt
   playwright install chromium   # used by some task setup steps
   ```

2. **VM provider.** Pick one with `--provider_name`:
   - `docker` (default, most accessible): needs a running Docker daemon; the
     Ubuntu image (~10 GB) is downloaded automatically on first run into
     `./docker_vm_data` (override with `OSWORLD_DOCKER_VMS_DIR`).
   - `vmware`: install VMware Workstation Pro and pass `--path_to_vm` pointing
     at the `.vmx` of an OSWorld Ubuntu image.
   - `aws`: set AWS credentials; the AMI is resolved from the region
     automatically.

   Images and provider details follow upstream
   [OSWorld](https://github.com/xlang-ai/OSWorld); see its docs for
   provider-specific setup.

3. **Model endpoint.** Copy `.env.example` to `.env` and fill in
   `MODEL_NAME` / `MODEL_BASE_URL` / `MODEL_API_KEY`, or pass `--model`,
   `--base_url`, `--api_key` directly. Any OpenAI-compatible endpoint serving
   a vision model with tool calling works; the prompt format targets the
   qwen3-vl family.

4. **Tasks.** OSWorld-format JSON; see [`tasks/README.md`](tasks/README.md).

## Collect

```bash
python run.py \
  --provider_name docker \
  --rollout_test_all_meta_path tasks/test_all.json \
  --rollout_task_dir tasks/examples \
  --model qwen3-vl-235b-a22b-instruct \
  --result_dir results/demo
```

- `--num_envs N` runs N VMs in parallel (each worker process owns one VM).
- `--domain <name>` restricts collection to one domain of the task list.
- `--enable_code_tool` additionally exposes a bash/python code tool to the
  agent; executed code and its output are recorded per step.
- `--max_steps` caps the episode length (default 15).
- Re-running with the same `--result_dir` resumes: episodes whose
  `meta_<task_id>.json` exists are skipped, unfinished ones are re-collected.

Output layout is OSWorld-compatible
(`<result_dir>/<domain>/<task_id>/{result.txt, traj.jsonl, runtime.log, step_*.png}`),
so the episodes also feed `benchmarking_analysis/` directly. `result.txt`
holds the task validator's score; `meta_<task_id>.json` aggregates the full
trajectory (thoughts, actions, normalized click coordinates).

## Reformat

```bash
python reformat.py --result_dir results/demo --out_root results/demo_reformatted
```

Converts each episode into one unified-schema JSON with screenshots copied
alongside. Steps carry the screenshot path and click coordinate both nested
(`state.screenshot_path`, consumed by `export_judged_sft.py`) and flat
(`screenshot_path` / `coordinate`, consumed by `eval_pipeline/run_judge.py`).
When the validator produced a score it is thresholded (≥ 0.99) into
`human_label`; tasks without a validator leave `human_label` unset and get
their reward from annotation or a VLM judge downstream.

## Provenance

`desktop_env/` is adapted from
[OSWorld](https://github.com/xlang-ai/OSWorld) (via an internal collection
fork). Local modifications:

- Providers trimmed to docker / vmware / aws; OSWorld-V2 compatibility layer
  removed.
- Proxy support is environment-driven (`DESKTOP_ENV_PROXY`) instead of
  hardcoded; internal endpoints and credentials removed throughout.
- Only the tool-call collection agent is kept; benchmark-specific agents,
  task-generation and SFT-distillation code paths are removed.
- `run.py` collects a fixed task list (no online task synthesis) and records
  the trajectory in the unified OSReward schema; the self-judge path is
  removed (judging is done by the repo-level `eval_pipeline/`).
