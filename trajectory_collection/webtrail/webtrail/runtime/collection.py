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

from ..agents.llm import ChatModel
from ..browser.client import ServicePool
from ..browser.vision import model_input_size, profile_for_model
from ..core.config import Config, JudgeSettings
from ..core.models import Task
from ..evaluation.judge import judge_trajectories
from .episode import run_episode
from .recorder import RunRecorder
from .resources import MemoryGovernor
from .scheduler import DomainCoolingDown, DomainGovernor

logger = logging.getLogger(__name__)


async def run_batch_judge(queue: asyncio.Queue[Path | None], run_dir: Path,
                          settings: JudgeSettings) -> dict:
    """Consume completed trajectories and score them in fixed-size batches."""
    batch_log = run_dir / "judge_batches.jsonl"
    try:
        batch_index = sum(1 for line in batch_log.read_text().splitlines()
                          if line.strip()) + 1
    except OSError:
        batch_index = 1

    pending: list[Path] = []
    pending_ids: set[str] = set()
    aggregate: collections.Counter[str] = collections.Counter()
    started = time.monotonic()

    async def flush(trigger: str) -> None:
        nonlocal batch_index
        batch = [path for path in pending if not (path / "judge.json").exists()]
        pending.clear()
        pending_ids.clear()
        if not batch:
            return

        batch_started = time.time()
        logger.info(
            "judge batch %d started: %d trajectories (trigger=%s, model=%s)",
            batch_index, len(batch), trigger, settings.model,
        )
        try:
            summary = await judge_trajectories(
                batch,
                settings.model_settings(),
                concurrency=settings.concurrency,
                last_n=settings.last_n,
                votes=settings.votes,
                style=settings.rubric,
            )
            error = None
        except Exception as err:  # isolate judge infrastructure from collectors
            logger.exception("judge batch %d crashed", batch_index)
            summary = {
                "judged": 0,
                "success": 0,
                "fail": 0,
                "failures": len(batch),
                "failed_trajectory_ids": [path.name for path in batch],
                "success_rate": None,
            }
            error = f"{type(err).__name__}: {err}"

        record = {
            "batch_index": batch_index,
            "trigger": trigger,
            "requested": len(batch),
            "trajectory_ids": [path.name for path in batch],
            "started_at": batch_started,
            "finished_at": time.time(),
            "duration_s": round(time.time() - batch_started, 2),
            "model": settings.model,
            "rubric": settings.rubric,
            "last_n": settings.last_n,
            "votes": settings.votes,
            "summary": summary,
            "error": error,
        }
        with batch_log.open("a", encoding="utf-8", errors="replace") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        aggregate["batches"] += 1
        aggregate["requested"] += len(batch)
        for key in ("judged", "success", "fail", "failures"):
            aggregate[key] += int(summary.get(key) or 0)
        logger.info(
            "judge batch %d finished: judged=%d success=%d fail=%d failures=%d",
            batch_index, summary["judged"], summary["success"],
            summary["fail"], summary["failures"],
        )
        batch_index += 1

    while True:
        path = await queue.get()
        try:
            if path is None:
                if pending and settings.flush_partial:
                    await flush("final_partial")
                break
            if (path / "judge.json").exists() or path.name in pending_ids:
                continue
            pending.append(path)
            pending_ids.add(path.name)
            if len(pending) >= settings.batch_size:
                await flush("batch_full")
        finally:
            queue.task_done()

    return {
        "enabled": True,
        "batch_size": settings.batch_size,
        "pending_unflushed": len(pending),
        "duration_s": round(time.monotonic() - started, 1),
        **dict(aggregate),
    }


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
        save_elements=config.run.save_elements,
        save_model_views=config.run.save_model_views,
    )
    recorder.save_config(config.to_dict())

    if config.judge.enabled:
        if config.judge.batch_size < 1:
            raise ValueError("judge.batch_size must be at least 1")
        if config.judge.concurrency < 1:
            raise ValueError("judge.concurrency must be at least 1")
        if config.judge.last_n < 1:
            raise ValueError("judge.last_n must be at least 1")
        if config.judge.votes < 1:
            raise ValueError("judge.votes must be at least 1")
        if config.judge.rubric not in {"binary", "multi"}:
            raise ValueError("judge.rubric must be binary or multi")
        if not config.judge.model:
            raise ValueError("judge.model is required when batch judging is enabled")

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

    model = ChatModel(config.model)
    viewport_size = (
        config.browser.viewport_width, config.browser.viewport_height
    )
    adapted_size = model_input_size(
        config.model.model, viewport_size, config.model.image_max_side
    )
    profile = profile_for_model(config.model.model)
    logger.info(
        "model image adapter: model=%s profile=%s viewport=%dx%d "
        "requested_max_side=%d sent=%dx%d",
        config.model.model, profile.name if profile else "none",
        *viewport_size, config.model.image_max_side, *adapted_size,
    )
    logger.info(
        "preflight recovery: attempts=%d attempt_timeout=%.0fs nav_timeout=%dms",
        config.browser.preflight_session_retries + 1,
        config.browser.preflight_attempt_timeout_s,
        config.browser.preflight_nav_timeout_ms,
    )
    judge_queue: asyncio.Queue[Path | None] | None = None
    judge_task: asyncio.Task | None = None
    if config.judge.enabled:
        judge_queue = asyncio.Queue()
        existing_unjudged = [
            path for path in sorted(recorder.trajectories.iterdir())
            if (path / "result.json").exists() and not (path / "judge.json").exists()
        ]
        for path in existing_unjudged:
            judge_queue.put_nowait(path)
        logger.info(
            "batch judge enabled: size=%d concurrency=%d model=%s; "
            "seeded %d existing unjudged trajectories",
            config.judge.batch_size, config.judge.concurrency,
            config.judge.model, len(existing_unjudged),
        )
        judge_task = asyncio.create_task(
            run_batch_judge(judge_queue, Path(config.run.out_dir), config.judge)
        )
    governor = DomainGovernor(config.pacing)
    memory = MemoryGovernor(config.resources, config.pacing.max_concurrency)
    if memory.effective_concurrency < config.pacing.max_concurrency:
        logger.warning(
            "memory gate reduced concurrency from %d to %d",
            config.pacing.max_concurrency, memory.effective_concurrency,
        )
    else:
        sample = memory.sample()
        logger.info(
            "resource gate: concurrency=%d, available=%d MiB, reserve=%d MiB, source=%s",
            memory.effective_concurrency, sample["available_mb"],
            memory.reserve_mb, sample["source"],
        )
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
                async with memory.slot(), governor.slot(task.domain):
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
            if judge_queue is not None:
                trajectory = recorder.trajectories / task.task_id
                if (trajectory / "result.json").exists():
                    judge_queue.put_nowait(trajectory)
            queue.task_done()

    async def progress() -> None:
        while done_count < total:
            await asyncio.sleep(15)
            elapsed = time.monotonic() - started
            rate = done_count / elapsed * 3600 if elapsed else 0.0
            resources = memory.sample()
            logger.info(
                "progress: %d/%d done (%.0f/h) — %s — workers=%d mem_available=%d MiB",
                done_count, total, rate, dict(tally), resources["active"],
                resources["available_mb"],
            )

    workers = [asyncio.create_task(worker(i))
               for i in range(memory.effective_concurrency)]
    reporter = asyncio.create_task(progress())
    judge_summary: dict = {"enabled": False}
    try:
        await asyncio.gather(*workers)
        if judge_queue is not None and judge_task is not None:
            judge_queue.put_nowait(None)
            judge_summary = await judge_task
    finally:
        reporter.cancel()
        await model.close()
        await pool.close()

    summary = {
        "total": total,
        "statuses": dict(tally),
        "duration_s": round(time.monotonic() - started, 1),
        "domains": governor.stats(),
        "resources": memory.stats(),
        "judge": judge_summary,
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

    parser.add_argument("--provider", choices=["auto", "openai", "anthropic"],
                        help="model provider; auto detects Claude model ids")
    parser.add_argument("--model", help="model id for the agent")
    parser.add_argument(
        "--base-url",
        help="optional compatible API base URL; official provider API by default "
             "(or stub:... for a dry run)",
    )
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
    parser.add_argument(
        "--step-timeout",
        type=float,
        help="wall-clock seconds for one observe/model/action/persist cycle "
             "(default 360; 0 disables)",
    )
    parser.add_argument("--profile", choices=["gui", "hybrid"], dest="action_profile")
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--per-domain", type=int)
    parser.add_argument("--memory-reserve-mb", type=int,
                        help="minimum free memory held back from browser workers")
    parser.add_argument("--estimated-episode-mb", type=int,
                        help="estimated RAM for one active browser episode")
    parser.add_argument("--no-memory-gate", action="store_true",
                        help="disable memory-aware concurrency admission")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--save-messages", action="store_true")
    parser.add_argument("--save-html", action="store_true",
                        help="save raw page HTML for every step (off by default)")
    parser.add_argument("--save-axtree", action="store_true",
                        help="save accessibility trees for every step (off by default)")
    parser.add_argument("--save-elements", action="store_true",
                        help="save DOM-derived interactive element maps (off by default)")
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
        "model.provider": args.provider,
        "model.model": args.model,
        "model.base_url": args.base_url,
        "model.grounding": args.grounding,
        "model.history_mode": args.history_mode,
        "model.vision_only": args.vision_only or None,
        "model.image_max_side": args.image_max_side,
        "browser.service_hosts": args.service,
        "browser.isolation": args.isolation,
        "run.max_steps": args.max_steps,
        "run.step_timeout_s": args.step_timeout,
        "run.action_profile": args.action_profile,
        "pacing.max_concurrency": args.concurrency,
        "pacing.per_domain": args.per_domain,
        "resources.memory_reserve_mb": args.memory_reserve_mb,
        "resources.estimated_episode_mb": args.estimated_episode_mb,
    }
    if args.viewport:
        width, _, height = args.viewport.lower().partition("x")
        overrides["browser.viewport_width"] = int(width)
        overrides["browser.viewport_height"] = int(height)
    # pixel-grounded models (Claude/GPT/Kimi) emit coordinates in the space of
    # the screenshot they saw; large images get downscaled inside the model and
    # throw the coordinates off, so cap the sent image unless overridden.
    if args.image_max_side is None and args.model:
        from ..browser.grounding import scheme_for_model
        if scheme_for_model(args.model, args.grounding or "auto").id == "pixel":
            overrides["model.image_max_side"] = 1280
    if args.no_resume:
        overrides["run.resume"] = False
    if args.no_memory_gate:
        overrides["resources.enabled"] = False
    if args.save_messages:
        overrides["run.save_messages"] = True
    if args.save_html:
        overrides["run.save_html"] = True
    if args.save_axtree:
        overrides["run.save_axtree"] = True
    if args.save_elements:
        overrides["run.save_elements"] = True
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
    if not config.model.model and not config.model.base_url.startswith("stub:"):
        raise SystemExit("--model (or model.model in --config) is required")
    tasks = load_tasks(args.tasks)
    logger.info("loaded %d tasks from %s", len(tasks), args.tasks)
    summary = asyncio.run(
        run_collection(config, tasks, shard=(args.rank, args.world_size))
    )
    print(json.dumps(summary, indent=1))
