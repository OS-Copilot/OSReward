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

Two rubrics via ``--rubric``: ``binary`` (default) records just the SUCCESS /
FAIL verdict; ``multi`` records the same verdict plus graded metric floats
(``alignment_score`` / ``success`` / ``efficiency`` / ``self_correction``, each
0-1), with each metric taken as the median across votes.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import collections
import json
import logging
import re
import statistics
import time
from pathlib import Path

from ..agents.llm import ChatModel, LLMError
from ..browser import images
from ..core.config import ModelSettings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_MULTI = """\
[ROLE]
You are a reward model evaluating a GUI agent operating across diverse platforms
(e.g., Desktop, Web, Mobile). Your job is to determine whether the agent's
trajectory successfully completes the user's task.

[INPUTS]
You will receive varying combinations of the following evidence:
1. User Instruction: The task to be completed.
2. Visual States: Screenshots from selected steps of the trajectory (e.g., the
   final few states, or a mix of initial and final states).
3. Action Logs / History (Optional): The agent's action history, or internal
   thoughts in text format.

[EVALUATION GOAL & THINKING PATTERN]
Evaluate the overall task outcome holistically based on the provided visual and
textual evidence.
- Subtask Decomposition: Briefly break the instruction down into implied
  subtasks or rubrics as a reasoning aid.
- Evidence-Based: The agent operates purely on visual inputs. Your judgment must
  be grounded in visible UI changes, final screen states, and logged actions.
(1) General Tasks: For general tasks (e.g., navigational, action-oriented)
    without specific output requirements, reaching the correct destination page
    or achieving the intended visual state is sufficient for SUCCESS.
(2) Explicit Output Tasks: If the instruction explicitly requests a text-based
    response (e.g., answering a question, providing a filename, or stating a
    conclusion), the trajectory is SUCCESSFUL only if this specific answer is
    explicitly produced.
- Outcome-Focused: Do not require rigid one-to-one confirmation for every
  subtask. Judge whether the visible end state matches the user's intended goal.

[GROUNDING RULE]
Facts in an explicit answer (numbers, names, dates, prices, rankings, titles,
page ranges) must be obtained or verified through interaction with the page, not
supplied from prior knowledge. An answer that reads as guessed or recalled
rather than read off a visible state fails the grounding rule, even if correct.

[BLOCKED / IMPOSSIBLE RULE]
If the task fails because it is <persistently> blocked by external constraints,
the final judgment must be FAIL, even if the agent behaved logically. Examples:
- System/OS barriers (e.g., permission)
- Web/Account barriers (e.g., login walls, CAPTCHAs, paywalls, region restrictions)
- Environmental failures (e.g., network errors, unavailable pages/apps)

[OUTPUT FORMAT]
Write a detailed thought process, followed by the exact scoring structure.
Output strictly in this order:

Thought: <Identify subtasks/rubrics. Explain what visual or textual evidence
supports or weakens completion. Note any blocking conditions. Evaluate grounding
compliance. Conclude clearly why it is a SUCCESS or FAIL.>

Metrics:
{
  "alignment_score": float,
  "success": float,
  "efficiency": float,
  "self_correction": float
}

Judge: SUCCESS or FAIL

[SCORING RULES]
- All metric values must be floats between 0.0 and 1.0.
- alignment_score (Instruction Adherence): high when actions stay on-task, low
  with frequent irrelevant exploration or unnecessary steps.
- success (Task Completion): whether the final outcome matches the goal from
  visible evidence; must follow the [GROUNDING RULE]; if blocked, success = 0.0.
- efficiency (Execution Efficiency): high for minimal, direct steps; low for
  redundant actions, excessive navigation, or looping.
- self_correction (Error Recovery): high when the agent detects and fixes its
  own mistakes; if no error occurs, a neutral-to-high score based on stability.
- Judge must be EXACTLY ONE of: SUCCESS or FAIL.
- Final Judgment Rule: SUCCESS only if the task is completed AND grounded in
  observable evidence; FAIL if incomplete, blocked, or grounding is violated
  (even if the answer is correct).
- Be conservative: do not assume completion without explicit visual or logged
  evidence. If uncertain, prefer FAIL."""


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


METRIC_KEYS = ("alignment_score", "success", "efficiency", "self_correction")


def _clip01(value) -> float | None:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def _extract_metrics(text: str) -> dict:
    """Pull the Metrics JSON block; fall back to per-key float scraping."""
    parsed: dict = {}
    block = re.search(r"metrics\s*:\s*(\{.*?\})", text, re.IGNORECASE | re.DOTALL)
    if block:
        try:
            parsed = json.loads(block.group(1))
        except ValueError:
            parsed = {}
    out: dict = {}
    for key in METRIC_KEYS:
        value = _clip01(parsed.get(key))
        if value is None:                              # scrape "key": 0.8 anywhere
            hit = re.search(rf'"{key}"\s*:\s*([0-9]*\.?[0-9]+)', text)
            value = _clip01(hit.group(1)) if hit else None
        out[key] = value
    return out


def _extract_multi(text: str) -> tuple[str, str, dict]:
    """Return (verdict, thought, metrics) for the multi-metric rubric."""
    verdict, _ = _extract_verdict(text)                # robust label parsing
    tm = re.search(r"thought\s*:\s*(.*?)(?:\n\s*(?:metrics|judge)\s*:|$)",
                   text, re.IGNORECASE | re.DOTALL)
    thought = tm.group(1).strip()[:600] if tm else ""
    return verdict, thought, _extract_metrics(text)


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


def _image_data_url(path: Path, settings: ModelSettings | None) -> str:
    image_format = (settings.image_format if settings else "png").lower()
    max_side = settings.image_max_side if settings else 0
    jpeg_quality = settings.image_jpeg_quality if settings else 85

    if image_format == "png" and max_side <= 0:
        payload = path.read_bytes()
        media_type = "image/png"
    else:
        image = images.fit_max_side(images.load_png(path.read_bytes()), max_side)
        if image_format == "jpeg":
            payload = images.to_jpeg_bytes(image, jpeg_quality)
            media_type = "image/jpeg"
        elif image_format == "png":
            payload = images.to_png_bytes(image)
            media_type = "image/png"
        else:
            raise ValueError(f"unsupported judge image format: {image_format}")
    encoded = base64.b64encode(payload).decode()
    return f"data:{media_type};base64,{encoded}"


def build_messages(traj_dir: Path, last_n: int, style: str = "binary",
                   settings: ModelSettings | None = None) -> list[dict]:
    task = json.loads((traj_dir / "task.json").read_text())
    result = json.loads((traj_dir / "result.json").read_text())
    steps = _step_records(traj_dir)
    selected = steps[-last_n:] if last_n > 0 else steps

    content: list[dict] = []
    for rec in selected:
        if rec["shot"] is not None:
            content.append({"type": "image_url",
                            "image_url": {
                                "url": _image_data_url(rec["shot"], settings)
                            }})

    history = "\n".join(
        f"Step {r['idx'] + 1} @ {r['url'][:90]}: {r['action']} "
        f"{json.dumps(r['args'], ensure_ascii=False)[:160]}"
        for r in selected
    ) or "(no actions recorded)"

    lines = [
        (
            f"The screenshots of the last {len(selected)} states are provided above, "
            "oldest first; a click target may be circled in red."
        ),
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
    system = SYSTEM_PROMPT_MULTI if style == "multi" else SYSTEM_PROMPT
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]


async def judge_trajectory(model: ChatModel, traj_dir: Path, last_n: int,
                           retries: int = 1, votes: int = 1,
                           style: str = "binary") -> dict:
    messages = build_messages(traj_dir, last_n, style, model.settings)
    started = time.time()
    ballots: list[dict] = []
    for vote_index in range(max(1, votes)):
        last_error = None
        for parse_attempt in range(retries + 1):
            reply = await model.complete(messages)
            try:
                if style == "multi":
                    verdict, thought, metrics = _extract_multi(reply.text)
                    ballots.append({"verdict": verdict, "thought": thought,
                                    "metrics": metrics})
                else:
                    verdict, thought = _extract_verdict(reply.text)
                    ballots.append({"verdict": verdict, "thought": thought})
                break
            except ValueError as err:
                last_error = str(err)
        else:
            raise ValueError(f"judge reply unparseable: {last_error}")

    verdicts = [b["verdict"] for b in ballots]
    winner = collections.Counter(verdicts).most_common(1)[0][0]
    thought = next((b["thought"] for b in ballots if b["verdict"] == winner), "")
    record = {
        "judge": winner,
        "success": 1.0 if winner == "SUCCESS" else 0.0,
        "thought": thought,
        "votes": verdicts if len(verdicts) > 1 else None,
        "model": model.settings.model,
        "rubric": style,
        "last_n": last_n,
        "latency_s": round(time.time() - started, 2),
        "judged_at": time.time(),
    }
    if style == "multi":
        # majority verdict decides SUCCESS/FAIL; each metric is the median across
        # ballots so a single stray sample can't swing a graded score
        agg: dict = {}
        for key in METRIC_KEYS:
            vals = [b["metrics"][key] for b in ballots if b["metrics"].get(key) is not None]
            if vals:
                agg[key] = round(statistics.median(vals), 3)
            elif key == "success":
                agg[key] = 1.0 if winner == "SUCCESS" else 0.0
            else:
                agg[key] = 0.5                          # neutral when unparseable
        record.update(agg)                              # includes success (float)

    (traj_dir / "judge.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=1),
        encoding="utf-8", errors="replace",
    )
    return record


async def judge_trajectories(traj_dirs: list[Path], settings: ModelSettings, *,
                             concurrency: int, last_n: int, force: bool = False,
                             votes: int = 1, style: str = "binary") -> dict:
    """Judge an explicit, stable batch of trajectories.

    Taking an explicit list prevents trajectories completed while a batch is
    being scored from leaking into that batch. Per-trajectory ``judge.json``
    files remain the resume checkpoint.
    """
    traj_dirs = [Path(d) for d in traj_dirs if (Path(d) / "result.json").exists()]
    if not force:
        traj_dirs = [d for d in traj_dirs if not (d / "judge.json").exists()]

    model = ChatModel(settings)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    tally: collections.Counter[str] = collections.Counter()
    failed_trajectory_ids: list[str] = []

    async def one(traj_dir: Path) -> None:
        async with semaphore:
            try:
                record = await judge_trajectory(model, traj_dir, last_n,
                                                votes=votes, style=style)
                tally[record["judge"]] += 1
                logger.info("%s -> %s", traj_dir.name, record["judge"])
            except (LLMError, ValueError, OSError) as err:
                failed_trajectory_ids.append(traj_dir.name)
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
        "failures": len(failed_trajectory_ids),
        "failed_trajectory_ids": sorted(failed_trajectory_ids),
        "success_rate": round(tally.get("SUCCESS", 0) / judged, 3) if judged else None,
    }


async def judge_run(run_dir: Path, settings: ModelSettings, *,
                    concurrency: int, last_n: int, force: bool,
                    votes: int = 1, style: str = "binary") -> dict:
    trajectories_dir = run_dir / "trajectories"
    traj_dirs = [d for d in sorted(trajectories_dir.iterdir())
                 if (d / "result.json").exists()]
    return await judge_trajectories(
        traj_dirs,
        settings,
        concurrency=concurrency,
        last_n=last_n,
        force=force,
        votes=votes,
        style=style,
    )


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "judge", help="score collected trajectories with a reward-model VLM judge"
    )
    parser.add_argument("--run", required=True, help="run directory")
    parser.add_argument("--model", required=True, help="judge model id")
    parser.add_argument("--provider", choices=["auto", "openai", "anthropic"],
                        default="auto", help="auto detects Claude model ids")
    parser.add_argument(
        "--base-url",
        default="",
        help="optional compatible API base URL; official provider API by default",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--rubric", choices=["binary", "multi"], default="binary",
                        help="binary = SUCCESS/FAIL verdict (default); multi = the "
                             "same verdict plus alignment_score / success / "
                             "efficiency / self_correction floats (0-1)")
    parser.add_argument("--last-n", type=int, default=5,
                        help="how many trailing states (screenshot + action) the "
                             "judge sees (default 5)")
    parser.add_argument("--votes", type=int, default=1,
                        help="independent judge samples per trajectory; the "
                             "majority verdict is kept")
    parser.add_argument("--temperature", type=float,
                        help="optional sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--force", action="store_true",
                        help="re-judge trajectories that already have judge.json")
    parser.set_defaults(handler=main)


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = ModelSettings(
        provider=args.provider, model=args.model, base_url=args.base_url,
        temperature=args.temperature, max_tokens=args.max_tokens,
    )
    summary = asyncio.run(judge_run(
        Path(args.run), settings,
        concurrency=args.concurrency, last_n=args.last_n,
        force=args.force, votes=args.votes, style=args.rubric,
    ))
    print(json.dumps(summary, indent=1))
