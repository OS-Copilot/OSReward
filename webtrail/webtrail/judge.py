"""Post-collection trajectory judging with a reward-model-style VLM judge.

`webtrail judge --run <dir>` gives every trajectory a binary SUCCESS / FAIL
verdict, following reward-model conventions:

* an instruction that asks for an answer is SUCCESS only if the agent explicitly
  produced the correct answer;
* facts in the answer (numbers, names, dates, page ranges, titles, …) must be
  obtained by interacting with the page — an answer that reads as guessed or
  recalled from prior knowledge is FAIL even if it happens to be correct;
* a task blocked by captcha / login wall / paywall / dead page is FAIL.

The judge sees the last N states (screenshots, click point circled where
available) paired with the action taken at each, plus the task and the agent's
final answer. Verdicts land in each trajectory's ``judge.json``; `webtrail
filter` reads them. `--votes N` samples N times and takes the majority verdict.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import collections
import json
import logging
import re
import time
from pathlib import Path

from .config import ModelSettings
from .llm import ChatModel, LLMError

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
[ROLE]
You are a reward model evaluating a GUI agent that operates a live web browser.
Your job is to decide whether the agent's trajectory successfully completed the
user's task.

[INPUTS]
You receive: the user instruction; screenshots from the last states of the
trajectory (a click action's target may be circled in red); and the agent's
action history and final answer.

[EVALUATION GOAL]
Synthesize all the evidence and decide whether the agent completed the task. The
action history may wrongly claim success, and listed actions are not guaranteed
to have executed — judge by what the screenshots actually show.

[RULES]
1. Explicit answer requirement:
   - For navigational / action tasks with no required output, reaching the
     correct page or visual state is SUCCESS.
   - If the instruction asks for a text answer (a question, a value, a page
     range, a conclusion), it is SUCCESS only if the agent explicitly output the
     correct answer in its final answer.
2. Grounding rule:
   - Facts in the answer — numbers, names, dates, prices, rankings, titles, page
     ranges — must be obtained or verified through interaction with the page,
     not supplied from prior knowledge or guessing.
   - Even if the answer is correct, judge FAIL if the agent appears to have
     guessed or recalled it rather than reading it off the page. Minor
     paraphrasing of observed evidence is fine.
3. Blocked / impossible rule:
   - If the task is defeated by a captcha, login wall, paywall, region block,
     network failure, or unavailable page, the verdict is FAIL even if the agent
     behaved reasonably.
4. Long trajectory rule:
   - Intermediate mistakes, detours, or loops do not imply FAIL if the agent
     ultimately reached the correct result.

[OUTPUT FORMAT]
Thought: briefly explain why the task succeeded or failed.
Judge: SUCCESS or FAIL

Output exactly in that order: Thought, then Judge."""


def _extract_verdict(text: str) -> tuple[str, str]:
    """Return (verdict, thought). verdict is 'SUCCESS' or 'FAIL'."""
    thought = ""
    tm = re.search(r"thought\s*:\s*(.*?)(?:\n\s*judge\s*:|$)", text, re.IGNORECASE | re.DOTALL)
    if tm:
        thought = tm.group(1).strip()[:600]
    jm = re.search(r"judge\s*:\s*([a-zA-Z]+)", text, re.IGNORECASE)
    label = None
    if jm:
        token = jm.group(1).strip().upper()
        if token.startswith("SUCC"):
            label = "SUCCESS"
        elif token.startswith("FAIL"):
            label = "FAIL"
    if label is None:
        # fall back to a bare mention, preferring the last occurrence
        hits = re.findall(r"\b(success|fail)\b", text, re.IGNORECASE)
        if hits:
            label = "SUCCESS" if hits[-1].lower() == "success" else "FAIL"
    if label is None:
        raise ValueError("no SUCCESS/FAIL verdict found in judge reply")
    return label, thought


def _step_records(traj_dir: Path) -> list[dict]:
    records = []
    for agent_path in sorted((traj_dir / "agent").glob("step_*.json")):
        idx = agent_path.stem  # step_000
        agent = json.loads(agent_path.read_text())
        state_path = traj_dir / "states" / f"{idx}.json"
        url = ""
        if state_path.exists():
            url = json.loads(state_path.read_text()).get("url") or ""
        # prefer the annotated screenshot (click point drawn) for the judge
        annotated = traj_dir / "annotated" / f"{idx}.png"
        shot = annotated if annotated.exists() else traj_dir / "screenshots" / f"{idx}.png"
        records.append({
            "idx": int(idx.split("_")[1]),
            "url": url,
            "action": agent.get("action"),
            "args": {k: v for k, v in (agent.get("args") or {}).items()},
            "analysis": (agent.get("analysis") or "")[:300],
            "shot": shot if shot.exists() else None,
        })
    return records


def build_messages(traj_dir: Path, last_n: int) -> list[dict]:
    task = json.loads((traj_dir / "task.json").read_text())
    result = json.loads((traj_dir / "result.json").read_text())
    steps = _step_records(traj_dir)
    selected = steps[-last_n:] if last_n > 0 else steps

    content: list[dict] = []
    for rec in selected:
        if rec["shot"] is not None:
            encoded = base64.b64encode(rec["shot"].read_bytes()).decode()
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded}"}})

    history = "\n".join(
        f"Step {r['idx'] + 1} @ {r['url'][:90]}: {r['action']} "
        f"{json.dumps(r['args'], ensure_ascii=False)[:160]}"
        for r in selected
    ) or "(no actions recorded)"

    lines = [
        f"The screenshots of the last {len(selected)} states are provided above, "
        "oldest first; a click target may be circled in red.",
        "",
        f"User instruction: {task['instruction']}",
    ]
    if task.get("criteria"):
        lines += ["Success criteria: " + "; ".join(task["criteria"])]
    lines += [
        "",
        f"Final outcome status: {result.get('status')}",
        f"Final URL: {result.get('final_url')}",
        f"Agent's final answer: {result.get('stop_answer') or '(none)'}",
        "",
        "Action history for the shown states:",
        history,
    ]
    content.append({"type": "text", "text": "\n".join(lines)})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


async def judge_trajectory(model: ChatModel, traj_dir: Path, last_n: int,
                           retries: int = 1, votes: int = 1) -> dict:
    messages = build_messages(traj_dir, last_n)
    started = time.time()
    ballots: list[tuple[str, str]] = []
    for _ in range(max(1, votes)):
        last_error = None
        for _ in range(retries + 1):
            reply = await model.complete(messages)
            try:
                ballots.append(_extract_verdict(reply.text))
                break
            except ValueError as err:
                last_error = str(err)
        else:
            raise ValueError(f"judge reply unparseable: {last_error}")

    verdicts = [v for v, _ in ballots]
    winner = collections.Counter(verdicts).most_common(1)[0][0]
    thought = next((t for v, t in ballots if v == winner), "")
    record = {
        "judge": winner,
        "success": 1.0 if winner == "SUCCESS" else 0.0,
        "thought": thought,
        "votes": verdicts if len(verdicts) > 1 else None,
        "model": model.settings.model,
        "last_n": last_n,
        "latency_s": round(time.time() - started, 2),
        "judged_at": time.time(),
    }
    (traj_dir / "judge.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=1),
        encoding="utf-8", errors="replace",
    )
    return record


async def judge_run(run_dir: Path, settings: ModelSettings, *,
                    concurrency: int, last_n: int, force: bool,
                    votes: int = 1) -> dict:
    traj_dirs = [d for d in sorted((run_dir / "trajectories").iterdir())
                 if (d / "result.json").exists()]
    if not force:
        traj_dirs = [d for d in traj_dirs if not (d / "judge.json").exists()]

    model = ChatModel(settings, api_log_path=run_dir / "api_calls.jsonl")
    semaphore = asyncio.Semaphore(concurrency)
    tally: collections.Counter[str] = collections.Counter()
    failures = 0

    async def one(traj_dir: Path) -> None:
        nonlocal failures
        async with semaphore:
            try:
                record = await judge_trajectory(model, traj_dir, last_n, votes=votes)
                tally[record["judge"]] += 1
                logger.info("%s -> %s", traj_dir.name, record["judge"])
            except (LLMError, ValueError, OSError) as err:
                failures += 1
                logger.warning("judge failed for %s: %s", traj_dir.name, err)

    try:
        await asyncio.gather(*(one(d) for d in traj_dirs))
    finally:
        await model.close()

    judged = sum(tally.values())
    return {
        "judged": judged,
        "success": tally.get("SUCCESS", 0),
        "fail": tally.get("FAIL", 0),
        "failures": failures,
        "success_rate": round(tally.get("SUCCESS", 0) / judged, 3) if judged else None,
    }


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "judge", help="score collected trajectories with a reward-model VLM judge"
    )
    parser.add_argument("--run", required=True, help="run directory")
    parser.add_argument("--model", required=True, help="judge model id")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--last-n", type=int, default=5,
                        help="how many trailing states (screenshot + action) the "
                             "judge sees (default 5)")
    parser.add_argument("--votes", type=int, default=1,
                        help="independent judge samples per trajectory; the "
                             "majority verdict is kept")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="reasoning models may burn most of this on thinking")
    parser.add_argument("--force", action="store_true",
                        help="re-judge trajectories that already have judge.json")
    parser.set_defaults(handler=main)


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = ModelSettings(
        model=args.model, base_url=args.base_url, api_key=args.api_key,
        temperature=args.temperature, max_tokens=args.max_tokens,
    )
    summary = asyncio.run(judge_run(
        Path(args.run), settings,
        concurrency=args.concurrency, last_n=args.last_n,
        force=args.force, votes=args.votes,
    ))
    print(json.dumps(summary, indent=1))
