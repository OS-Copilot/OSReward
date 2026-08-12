"""Collection run orchestration and the `collect` CLI subcommand.

Reads a task file (JSONL or a JSON list), fans episodes out across an asyncio
worker pool under the domain governor, and prints periodic progress. Finished
trajectories are skipped on resume, so re-running the same command continues
an interrupted run.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import logging
import time
from pathlib import Path

from .browser import ServicePool
from .config import Config
from .llm import ChatModel
from .recorder import RunRecorder
from .runner import run_episode
from .scheduler import DomainCoolingDown, DomainGovernor
from .types import Task

logger = logging.getLogger(__name__)


def load_tasks(path: str | Path) -> list[Task]:
    path = Path(path)
    text = path.read_text()
    records: list[dict]
    if path.suffix == ".jsonl" or "\n{" in text.strip():
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = json.loads(text)
        records = data if isinstance(data, list) else data.get("tasks", [])
    tasks = [Task.from_record(record, index) for index, record in enumerate(records)]

    seen: set[str] = set()
    unique: list[Task] = []
    for task in tasks:
        if task.task_id in seen:
            logger.warning("duplicate task id %s; keeping first occurrence", task.task_id)
            continue
        seen.add(task.task_id)
        unique.append(task)
    return unique


async def run_collection(config: Config, tasks: list[Task],
                         *, shard: tuple[int, int] = (0, 1)) -> dict:
    rank, world = shard
    tasks = [task for index, task in enumerate(tasks) if index % world == rank]

    recorder = RunRecorder(
        config.run.out_dir,
        save_html=config.run.save_html,
        save_axtree=config.run.save_axtree,
        save_model_views=config.run.save_model_views,
    )
    recorder.save_config(config.to_dict())

    if config.run.resume:
        pending = [t for t in tasks if not recorder.is_finished(t.task_id)]
        skipped = len(tasks) - len(pending)
        if skipped:
            logger.info("resume: skipping %d already-finished trajectories", skipped)
        tasks = pending

    pool = ServicePool(config.browser)
    health = await pool.health()
    down = [h["host"] for h in health if not h.get("ok")]
    if down:
        raise SystemExit(f"browser service worker(s) unreachable: {down} — "
                         "start them with browser_service/start.sh")

    model = ChatModel(
        config.model,
        api_log_path=recorder.api_log_path if config.run.api_log else None,
    )
    governor = DomainGovernor(config.pacing)
    queue: asyncio.Queue[Task] = asyncio.Queue()
    for task in tasks:
        queue.put_nowait(task)

    tally: collections.Counter[str] = collections.Counter()
    done_count = 0
    total = len(tasks)
    started = time.monotonic()

    async def worker(worker_id: int) -> None:
        nonlocal done_count
        while True:
            try:
                task = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                async with governor.slot(task.domain):
                    logger.info("[w%02d] start %s (%s)", worker_id,
                                task.task_id, task.domain)
                    result = await run_episode(task, config, pool, recorder, model)
                status = result.get("status", "unknown")
                governor.report(task.domain, blocked=status == "blocked")
            except DomainCoolingDown as err:
                await recorder.reject(task, "domain_cooldown", str(err))
                status = "domain_cooldown"
            except Exception as err:                      # keep the fleet alive
                logger.exception("[w%02d] %s crashed", worker_id, task.task_id)
                await recorder.reject(task, "internal_error", repr(err))
                status = "internal_error"
            tally[status] += 1
            done_count += 1
            queue.task_done()

    async def progress() -> None:
        while done_count < total:
            await asyncio.sleep(15)
            elapsed = time.monotonic() - started
            rate = done_count / elapsed * 3600 if elapsed else 0.0
            logger.info("progress: %d/%d done (%.0f/h) — %s",
                        done_count, total, rate, dict(tally))

    workers = [asyncio.create_task(worker(i))
               for i in range(config.pacing.max_concurrency)]
    reporter = asyncio.create_task(progress())
    try:
        await asyncio.gather(*workers)
    finally:
        reporter.cancel()
        await model.close()
        await pool.close()

    summary = {
        "total": total,
        "statuses": dict(tally),
        "duration_s": round(time.monotonic() - started, 1),
        "domains": governor.stats(),
    }
    (Path(config.run.out_dir) / "run_summary.json").write_text(
        json.dumps(summary, indent=1)
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("collect", help="run trajectory collection")
    parser.add_argument("--tasks", required=True, help="JSONL/JSON task file")
    parser.add_argument("--out", required=True, help="output run directory")
    parser.add_argument("--config", help="JSON config file (optional)")

    parser.add_argument("--model", help="model id for the agent")
    parser.add_argument("--base-url", help="OpenAI-compatible endpoint (or stub:...)")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--backend", choices=["prompt", "claude_cua"],
                        help="prompt = fenced-JSON scheme (any model); "
                             "claude_cua = Claude native computer tool")
    parser.add_argument("--grounding",
                        choices=["auto", "box1000", "point1000", "pixel"])
    parser.add_argument("--history-mode", choices=["windowed", "text_full"])
    parser.add_argument("--vision-only", action="store_true",
                        help="drop URL/title text; the screenshot is the only "
                             "page signal the agent gets")
    parser.add_argument("--image-max-side", type=int)

    parser.add_argument("--service", action="append",
                        help="browser service host, repeatable "
                             "(default http://127.0.0.1:9300)")
    parser.add_argument("--viewport", help="WIDTHxHEIGHT, e.g. 1920x1080 or 2560x1440")
    parser.add_argument("--isolation", choices=["browser", "context"])

    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--profile", choices=["gui", "hybrid"], dest="action_profile")
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--per-domain", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--save-messages", action="store_true")
    parser.add_argument("--save-html", action="store_true",
                        help="save raw page HTML for every step (off by default)")
    parser.add_argument("--save-axtree", action="store_true",
                        help="save accessibility trees for every step (off by default)")
    parser.add_argument("--save-model-views", action="store_true",
                        help="save resized images sent to the model (off by default)")
    parser.add_argument("--save-annotated", action="store_true",
                        help="save action-annotated screenshots (off by default)")

    parser.add_argument("--rank", type=int, default=0, help="shard index")
    parser.add_argument("--world-size", type=int, default=1, help="shard count")
    parser.set_defaults(handler=main)


def build_config(args: argparse.Namespace) -> Config:
    overrides: dict = {
        "run.out_dir": args.out,
        "model.model": args.model,
        "model.base_url": args.base_url,
        "model.api_key": args.api_key,
        "model.backend": args.backend,
        "model.grounding": args.grounding,
        "model.history_mode": args.history_mode,
        "model.vision_only": args.vision_only or None,
        "model.image_max_side": args.image_max_side,
        "browser.service_hosts": args.service,
        "browser.isolation": args.isolation,
        "run.max_steps": args.max_steps,
        "run.action_profile": args.action_profile,
        "pacing.max_concurrency": args.concurrency,
        "pacing.per_domain": args.per_domain,
    }
    if args.viewport:
        width, _, height = args.viewport.lower().partition("x")
        overrides["browser.viewport_width"] = int(width)
        overrides["browser.viewport_height"] = int(height)
    # pixel-grounded models (Claude/GPT/Kimi) emit coordinates in the space of
    # the screenshot they saw; large images get downscaled inside the model and
    # throw the coordinates off, so cap the sent image unless overridden.
    if args.image_max_side is None and args.model:
        from .grounding import scheme_for_model
        if scheme_for_model(args.model, args.grounding or "auto").id == "pixel":
            overrides["model.image_max_side"] = 1280
    if args.no_resume:
        overrides["run.resume"] = False
    if args.save_messages:
        overrides["run.save_messages"] = True
    if args.save_html:
        overrides["run.save_html"] = True
    if args.save_axtree:
        overrides["run.save_axtree"] = True
    if args.save_model_views:
        overrides["run.save_model_views"] = True
    if args.save_annotated:
        overrides["run.annotate_screenshots"] = True
    return Config.load(args.config, overrides)


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    config = build_config(args)
    if not config.model.base_url:
        raise SystemExit("--base-url (or model.base_url in --config) is required; "
                         "use 'stub:scroll,stop' for a dry run without a model")
    tasks = load_tasks(args.tasks)
    logger.info("loaded %d tasks from %s", len(tasks), args.tasks)
    summary = asyncio.run(
        run_collection(config, tasks, shard=(args.rank, args.world_size))
    )
    print(json.dumps(summary, indent=1))
