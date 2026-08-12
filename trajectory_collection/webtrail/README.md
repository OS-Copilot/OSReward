# webtrail

Web trajectory collection for agent training data. `webtrail` drives
vision-language agents through live websites and records complete, replayable
evidence for every step — built for producing SFT / reward-model corpora at
scale, with the operational hardening that live-web collection actually needs.

```
 task file ─► domain governor ─► episode runner ──────────► trajectories/<id>/
               (per-domain        preflight → observe →        screenshots/  elements/
                pacing,            decide → act loop,           optional debug evidence
                cooldowns)         block guard, recovery,       states/  agent/
                                   stale detection              result.json  judge.json

 browser service (Node + Playwright)          model endpoint (OpenAI-compatible,
   stealth Chromium, typed actions,            or Anthropic native computer tool)
   snapshots: screenshot + HTML + a11y tree
```

Pipeline: **tasks import** → **collect** → **judge** → **filter** → **export**,
with **view** to inspect and **doctor** to check the stack. Each stage is a
subcommand and writes into the same run directory.

## Highlights

- **Isolated, stealthy browsing.** A small Node service owns Playwright
  Chromium with the stealth evasion set. Each episode gets its own browser (or
  a lighter per-context session), a fixed-size viewport (1080p by default, 2K
  supported), single-tab navigation (`target=_blank` and `window.open` are
  rewritten), an idle reaper so crashed runs never leak browsers, and global
  crash guards so one stray page error can't take down a worker mid-run.
- **Typed action execution.** The model's JSON is validated and compiled to a
  fixed action vocabulary — `click`, `double_click`, `hover`, `scroll`, `drag`,
  `type`, `fill`, `hotkey`, `wait`, `stop`, plus browser primitives `goto`,
  `go_back`, `go_forward`, `select_option`, `set_checked` in the `hybrid`
  profile. No model output is ever evaluated as code.
- **Model-aware grounding.** Coordinate conventions are per-model-family and
  auto-detected from the model id (overridable with `--grounding`):

  | model family | convention | scheme |
  |---|---|---|
  | Gemini | `box2d = [ymin,xmin,ymax,xmax]`, 0-1000 | `box1000` |
  | Qwen / UI-TARS | `point = [x,y]`, 0-1000 relative | `point1000` |
  | Claude / GPT / Kimi | `point = [x,y]` in screenshot pixels | `pixel` |

  Pixel-grounded models emit coordinates in the space of the screenshot they
  saw, so their screenshots are auto-capped at 1280px on the long side (larger
  images get downscaled inside the model and shift the coordinates); the
  resolver maps back to the real viewport.

  **Use a vision-grounding model, not a general chat model.** Coordinate
  accuracy depends entirely on the model actually localizing in the image. For
  Qwen this means the **`qwen3-vl-*`** vision models — the general
  `qwen3-max` / `qwen-plus` text models do not do visual grounding and return
  near-center guesses (measured ~3/1000 error for `qwen3-vl-235b` vs ~500/1000
  for `qwen3-max` on the same target).

- **Two agent backends.** `--backend prompt` (default) works for every model
  via a fenced-JSON action scheme. `--backend claude_cua` drives Claude through
  its native `computer` tool (Anthropic Messages API tool-use loop) — use it
  only against an endpoint that faithfully passes Anthropic tool calls through;
  proxy gateways that reformat tool calls will not work, and Claude runs well
  on the default prompt backend with `pixel` grounding regardless.
- **Compact evidence by default.** Every step keeps the browser screenshot, a
  compact role/name/bbox element map, parsed action, resolved pixel target,
  executed commands, and model usage. Raw HTML, accessibility trees, resized
  model-view copies, and action-annotated screenshots are opt-in via
  `--save-html`, `--save-axtree`, `--save-model-views`, and `--save-annotated`.
  `result.json` summarizes each episode in machine-readable form.
- **Robust on the live web.** Rule-based block detection, JS-challenge grace
  waits, search-engine fallback, block recovery with a per-episode blocked-site
  denylist, multi-URL preflight, per-domain pacing and cooldown. See
  [Robustness](#robustness-on-the-live-web).
- **Cheap failure detection.** Stability retries before every screenshot,
  page-signature staleness tracking (URL + HTML size + perceptual hash) that
  aborts do-nothing loops, corrective re-prompting for malformed output,
  automatic `max_tokens` escalation when a reasoning model spends its whole
  budget on hidden thinking, and resume that skips finished trajectories.
- **Judge + rule-based curation.** An optional VLM judge scores each
  trajectory; `webtrail filter` buckets, quality-scores, and deduplicates every
  trajectory into a `manifest.jsonl` for training-data selection.

## Setup

### Docker (recommended)

The whole stack comes up with one command — no Node, Playwright, or font setup
on the host:

```bash
docker compose up -d browser                      # start the browser workers
docker compose run --rm webtrail doctor --service http://browser:9300
docker compose run --rm webtrail collect \
  --tasks tasks/example.jsonl --out runs/demo \
  --model <id> --base-url <endpoint> --api-key $KEY \
  --service http://browser:9300 --service http://browser:9301
```

`WORKERS` sets how many browser workers start (default 4, on ports
9300–9303). `tasks/` and `runs/` are mounted, so task files and collected
trajectories live on the host.

### From source

```bash
# browser service
cd browser_service
npm install
npx playwright install chromium
WORKERS=4 BASE_PORT=9300 ./start.sh

# python package
pip install -e .          # add [data] for `tasks import` from HF/parquet
```

Run `webtrail doctor` any time to check that the browser workers (and, with
`--base-url`, the model endpoint) are reachable before starting a run.

### Linux servers

The service runs headless on Linux out of the box:

- The Chromium sandbox is disabled automatically on Linux (it cannot start in
  most containers or as root). Set `WEBTRAIL_SANDBOX=1` to force it on where the
  host supports it; append extra flags with `WEBTRAIL_CHROMIUM_ARGS`.
- Install Chromium's system libraries once: `npx playwright install-deps chromium`
  (or `playwright install --with-deps chromium`).
- Install fonts, or screenshots of many sites render as blank/tofu boxes:
  `apt-get install -y fonts-liberation fonts-noto-cjk fonts-noto-color-emoji`.
- `start.sh` cleans stale ports with whichever of `lsof` / `fuser` / `ss` is
  present, and skips cleanup if none are — nothing to install.

Node 18+ is required.

## Tasks

Collection reads a small JSONL whose acting-agent fields are `{url,
instruction}`. To pull tasks from
an existing dataset instead of hand-writing them, `tasks import` maps foreign
column names (`website`, `confirmed_task`, `ques`, `goal`, …) onto that schema,
drops records missing a URL or instruction, and de-duplicates:

```bash
# a local file
webtrail tasks import --source data/mind2web.parquet --out tasks/mind2web.jsonl
# a Hugging Face dataset (needs the [data] extra)
webtrail tasks import --source owner/web-agent-tasks --split train \
  --out tasks/imported.jsonl --limit 500
```

Columns are auto-detected; override with `--url-field` / `--instruction-field`.
For open-ended tasks with no fixed site, `--default-url https://duckduckgo.com`
gives every record a search-engine start URL.

## Collect

```bash
webtrail collect \
  --tasks tasks/example.jsonl \
  --out runs/demo \
  --model gemini-3-flash-preview \
  --base-url https://your-endpoint/v1 \
  --api-key $KEY \
  --service http://127.0.0.1:9300 --service http://127.0.0.1:9301 \
  --concurrency 8 --per-domain 1 \
  --max-steps 30 --profile hybrid
```

The acting agent receives only the task `instruction` and the live current
`url`. Gold `steps`, `criteria`, and other dataset metadata may remain attached
to the internal task record for judging and analysis, but are never inserted
into the agent prompt. The bundled `tasks/example.jsonl` therefore contains
only `url` and `instruction` fields.

Useful variants:

- `--base-url stub:scroll,stop` runs the full pipeline with a scripted fake
  model (no API key needed) — good for verifying the environment end to end.
- `--vision-only` drops the URL/title text from the prompt so the screenshot is
  the only page signal the agent gets (pure visual-GUI setting).
- `--backend claude_cua` uses Claude's native computer tool (needs a
  tool-passthrough endpoint); otherwise the default prompt backend fits all.
- `--grounding point1000 --history-mode text_full --image-max-side 1568`
  overrides the coordinate scheme, keeps the whole text history with only the
  latest screenshot, and downscales screenshots before sending.
- `--viewport 2560x1440` collects at 2K.
- `--rank N --world-size M` shards the task file across machines.
- Re-running the same command resumes: finished trajectories are skipped
  (`--no-resume` to force a full recollect).

## Judge

```bash
webtrail judge --run runs/demo \
  --model gemini-3-flash-preview --base-url https://your-endpoint/v1 --api-key $KEY \
  --concurrency 4 --votes 3
```

A reward-model-style VLM judge gives every trajectory a binary **SUCCESS** /
**FAIL** verdict from the task, the action log, the final answer, and the last
`--last-n` screenshots (default 5, with the click target circled where
available). It enforces grounding — an answer that reads as guessed or recalled
rather than read off the page is FAIL even if correct — and marks captcha /
login-wall / paywall / dead-page runs as FAIL. `--votes N` samples N times and
keeps the majority verdict (recommended for RM data — it suppresses single-sample
noise on borderline runs). Results land in each trajectory's `judge.json`
(`{judge, success, thought}`); already-judged trajectories are skipped unless
`--force`.

`--rubric multi` switches to a graded reward model: the same SUCCESS / FAIL
verdict plus four metric floats (0-1) — `alignment_score` (instruction
adherence), `success` (grounded task completion), `efficiency` (directness), and
`self_correction` (error recovery). Each metric is the median across `--votes`
samples. `filter` and `export` surface these extra axes when present, so you can
select or weight training data on more than a single bit.

## Curate

```bash
webtrail filter --run runs/demo --min-score 0.5 --min-success 0.7
```

Every trajectory is placed in exactly one bucket — `valid_candidate`,
`blocked`, `env_error`, `stale_loop`, `malformed_action`, `max_steps`,
`missing_screenshot`, `empty`, or `duplicate` — then given a rule-based quality
score and deduplicated by normalized instruction + domain + final URL + action
sequence + final-screenshot perceptual hash. Output is `manifest.jsonl`.

`--min-score` gates the rule score; `--min-success` additionally gates on the
judge's success score where a `judge.json` exists. Select training candidates
with `bucket == "valid_candidate" && keep == true`.

## Export

Turn kept trajectories into a chat training corpus. Each step becomes a
user turn (the screenshot the agent saw + its URL) paired with an assistant
turn (the agent's analysis + the action it took), so the export is a faithful
multi-turn transcript, not just the final answer.

```bash
webtrail export --run runs/demo --format messages   # OpenAI-style, images as data URIs
webtrail export --run runs/demo --format sharegpt    # ShareGPT turns + parallel images list
```

`--gate keep` (default) exports only the trajectories `filter` marked
`keep`; `--gate success` uses the judge's SUCCESS verdict; `--gate all` takes
everything. `--no-embed-images` references screenshots by path instead of
inlining them.

## View

Build a self-contained HTML browser for a run — no server, open it with
`file://`:

```bash
webtrail view --run runs/demo        # writes runs/demo/index.html
```

The left pane lists every trajectory (coloured by status and judge verdict, with
a text filter); the right pane steps through screenshots — raw, action-annotated,
or the exact downscaled model view — alongside the action, the agent's analysis,
and the judge's thought. Arrow keys navigate steps (←/→) and trajectories (↑/↓).
Pass a single trajectory directory to `--run` to inspect just that episode.

## Robustness on the live web

Live sites throttle and challenge automation. webtrail handles it in layers,
from cheapest to last-resort:

1. **Avoid.** Stealth Chromium + per-domain pacing: at most `--per-domain`
   concurrent episodes per registrable domain, jittered gaps between same-domain
   starts, and a cooldown after several consecutive blocks on a domain.
2. **Detect.** Every observation passes a rule-based guard for CAPTCHA,
   Cloudflare-style JS challenges, 403/denials, rate limits, login walls, geo
   blocks, and blank/error pages. Preflight rejects unreachable sites before any
   model call.
3. **Wait it out.** Non-interactive JS challenges ("Just a moment…") often clear
   on their own; the guard re-observes a few times before giving up.
4. **Reroute.** A blocked search engine falls back to another (DuckDuckGo /
   Bing). When the agent navigates *into* a blocked target site, it is sent back
   with a notice instead of ending the episode, and that domain is added to a
   per-episode denylist so the agent cannot loop back to it — it must find
   another site. Multi-URL tasks whose first site is blocked start from the next
   reachable URL rather than being dropped.
5. **Give up cleanly.** A hard-blocked target that can't be avoided ends the
   episode as `blocked` with the block type, scope, and evidence recorded — no
   wasted steps, and the reason is attributed (hard blocks note that a
   residential proxy is likely required). Set a proxy per session via the
   `proxy` browser setting to route around data-center-IP blocks.

Open-ended tasks (a search engine as the only start URL, no fixed target sites)
are the most robust: the agent picks whatever sites are reachable, so a single
blocked site never stops the task.

## Tests

- `python tests/action_fidelity.py` — **execution layer.** Drives every action
  in model format through grounding → compile → browser service against a local
  control-panel page and asserts the DOM effects (typed text, replaced values,
  selected options, checkbox state, slider position after drag, hover reveals,
  scroll offset, history navigation).
- `python tests/model_action_coverage.py --base-url … --api-key …` —
  **production layer.** For each model × action, sends the action catalogue (in
  that model's coordinate scheme) plus a screenshot and a single-purpose
  instruction, and checks the reply parses and compiles into the expected
  action. Produces a model × action coverage matrix.

## Output layout

```
runs/demo/
  run_config.json        exact configuration snapshot
  run_summary.json       status tally + per-domain stats
  api_calls.jsonl        model latency/usage log
  rejects.jsonl          tasks skipped at preflight (unreachable / blocked)
  manifest.jsonl         written by `webtrail filter`
  index.html             written by `webtrail view` (self-contained browser)
  export.*.jsonl         written by `webtrail export` (training corpus)
  trajectories/<task_id>/
    task.json
    result.json          status, block info, counters, action keys, timing
    judge.json           written by `webtrail judge`
    screenshots/step_000.png ...     browser captures, one per step
    annotated/step_001.png ...       optional (--save-annotated)
    model_views/step_000.png ...     optional (--save-model-views)
    html/step_000.html ...           optional (--save-html)
    axtree/step_000.json ...         optional (--save-axtree)
    elements/step_000.json ...       interactive element map
    states/step_000.json ...         url/title/scroll/hashes/guard verdict
    agent/step_000.json ...          reply, parsed action, resolved target
    messages/step_000.json ...       optional model-input dump (--save-messages)
```

`result.json` is the only file post-processing must read to triage an episode:
its `status`, `block`, and `counters` (action errors, stale repeats, block
recoveries, fallback switches) summarize the run without scanning the steps.
