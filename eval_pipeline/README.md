# OSReward binary judge

This directory contains the reference evaluator for the public
[OSReward benchmark](https://huggingface.co/datasets/OS-Copilot/OSReward).
It runs a multimodal judge through an OpenAI-compatible Chat Completions API or
the native Anthropic Messages API and reports strict binary metrics.

## Install

From the OSReward repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r eval_pipeline/requirements.txt
```

## Dataset loading

The evaluator downloads the selected Parquet configuration from Hugging Face.
Screenshots are embedded as an image column, so no TAR extraction is required.
Only Parquet metadata is read while traces are indexed; image bytes are loaded
from the selected trajectory when its request is built.

## Run Full or Hard

For an OpenAI or OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY=...

python eval_pipeline/run_judge.py \
  --dataset OS-Copilot/OSReward \
  --subset full \
  --models gpt-4o \
  --version full_gpt4o
```

To evaluate Hard, change only the subset:

```bash
python eval_pipeline/run_judge.py \
  --dataset OS-Copilot/OSReward \
  --subset hard \
  --models gpt-4o \
  --version hard_gpt4o
```

For a custom OpenAI-compatible service, prefer environment variables so the
credential does not appear in shell history or process arguments:

```bash
export JUDGE_API_KEY=...
export JUDGE_BASE_URL=https://your-endpoint.example/v1
```

For an Anthropic-native endpoint:

```bash
export ANTHROPIC_API_KEY=...

python eval_pipeline/run_judge.py \
  --dataset OS-Copilot/OSReward \
  --subset full \
  --api_style anthropic \
  --models claude-opus-4-6 \
  --version full_claude_opus46
```

Only send benchmark screenshots and trajectory text to an endpoint you trust.
Use HTTPS for any remote service.

## Evaluation protocol

The defaults implement the reference binary protocol:

- full thought and action history;
- the last five screenshots;
- red action-point markers when normalized coordinates are available;
- temperature 0;
- one parsed verdict per trajectory: `Judge: SUCCESS` or `Judge: FAIL`.

Important options:

```text
--dataset DATASET     Hugging Face dataset ID or local dataset repository
--traces PATH [...]   local Parquet shards or legacy JSON traces
--dataset_revision R  optional Hugging Face branch, tag, or commit
--cache_dir PATH      optional Hugging Face download cache
--first_n N|all       screenshots from the start (default: 0)
--last_n N|all        screenshots from the end (default: 5)
--history MODE        full, selected, or none (default: full)
--no_thought          remove thoughts but retain actions
--no_mark             disable normalized action-point markers
--api_style STYLE     openai or anthropic (default: openai)
--base_url URL        custom API endpoint
--max_workers N       concurrent requests per model
--limit N             evaluate only the first N trace IDs
```

Results are written incrementally to `eval_pipeline/results/`. Rerunning the
same model and version resumes completed trajectories. The metrics JSON reports
Accuracy, Balanced Accuracy, SUCCESS Recall, FAIL Recall, Coverage, and error
counts. Missing, API-error, and unparseable outputs remain in the denominator
and count as incorrect.

The public Parquet schema keeps one trajectory per row. `screenshots[i]` is the
image for `trajectory[i]`; unavailable images are represented by `None`. Local
Parquet shards can be evaluated without Hugging Face access:

```bash
python eval_pipeline/run_judge.py \
  --traces /path/to/parquet/hard \
  --subset hard \
  --models gpt-4o \
  --version local_hard_gpt4o
```

`--traces` continues to accept the legacy extracted JSON layout and custom JSON
records, so existing local experiments remain reproducible.

## Bundled smoke test

The synthetic example can verify installation and API connectivity without
downloading the benchmark:

```bash
python eval_pipeline/run_judge.py \
  --traces eval_pipeline/example/example_trace.json \
  --subset custom \
  --models gpt-4o \
  --version example
```
