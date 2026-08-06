# OSReward standalone judge

Self-contained evaluation code for [OSReward](https://arxiv.org/abs/2607.28609):
a VLM judges whether a GUI-agent trajectory completed the user's task, through
any OpenAI-compatible API. `--prompt_type` switches the output mode:

| `--prompt_type` | Output | Prompt file |
|---|---|---|
| `binary` | `Judge: SUCCESS \| FAIL` | `prompts/binary_v1.txt` |
| `multi` (default) | `Judge` + `Alignment` + `Efficiency` (each in {0, 0.5, 1.0}, N/A on FAIL) | `prompts/multi_v4.txt` |

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...            # or JUDGE_API_KEY
export OPENAI_BASE_URL=https://...      # optional; or JUDGE_BASE_URL / --base_url
```

## Quick start

```bash
python run_judge.py --traces example/example_trace.json --models gpt-4o                       # multi
python run_judge.py --traces example/example_trace.json --models gpt-4o --prompt_type binary  # binary
```

The bundled example is a synthetic 4-step trajectory ("Turn on Dark Mode in
the Settings app") with gold labels, so the report also prints accuracy.
Results land in `results/judge_<version>_<prompt_type>_<model>.jsonl`, one row
per trace; reruns skip traces that already have a `status="ok"` row.

## Trace format

One JSON per trajectory. `--traces` takes files and/or directories (scanned
recursively; any JSON with `trace_id` and `task_id` counts as a trace).

```jsonc
{
  "trace_id": "...",
  "task_id": "...",
  "platform": "Desktop",        // Desktop | Mobile | Web
  "instruction": "...",
  "trajectory": [
    {
      "step_index": 0,
      "screenshot_path": "screenshots/step_0000.png",  // relative to this JSON
      "thought": "...",
      "action": "click(coordinate=[150, 608])",
      "coordinate": [150, 608]  // optional; normalized 0-1000, drawn as a red circle
    }
  ],
  // Optional gold labels; enable accuracy scoring in the report:
  "human_label": "SUCCESS",     // SUCCESS | FAIL
  "human_alignment": 1.0,       // 0 | 0.5 | 1.0, only meaningful when SUCCESS
  "human_efficiency": 1.0
}
```

## What the judge sees

One API call per trace:

- Screenshots of the first `--first_n` and last `--last_n` steps (default:
  last 5; `all` includes every step), with a red circle at click coordinates
  (`--no_mark` disables).
- The instruction and platform.
- The action history (`--history full | selected | none`; `--no_thought`
  strips agent thoughts).

## Scoring (when gold labels are present)

- `binary_correct` = 1 if the predicted label equals `human_label`.
- Alignment / Efficiency are scored only on gold-SUCCESS traces as
  `1 - |gold - pred|`; a FAIL verdict or a missing score counts 0.0.

## Other options

`--version` tags output filenames; `--max_workers` sets concurrent API calls
per model; `--limit N` judges only the first N traces; `--temperature`,
`--max_tokens`, `--timeout`, `--max_retries` control the API call;
`--prompt_file` supplies a custom system prompt.
