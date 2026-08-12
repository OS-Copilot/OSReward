"""Prompt assembly for the browsing agent.

All text is generated from the action profile and grounding scheme so that a
single source of truth defines what the model may do and how it must localize
targets. Nothing here depends on a specific model family.
"""

from __future__ import annotations

from . import actions
from .grounding import GroundingContext
from .types import PageState, Task

RESPONSE_CONTRACT = """## How to respond

First write a compact analysis (about {analysis_words} words): what the current
screenshot shows, what your last action changed, how far the task has
progressed, and what to do next. Then output exactly ONE action as JSON in a
fenced block:

```json
{{"action": "<name>", "args": {{ ... }}}}
```

Never output more than one JSON block. Never invent action names or argument
fields that are not documented above."""

GROUND_RULES = """## Ground rules

- If you hit a CAPTCHA, "verify you are human" interstitial, or an access-denied
  page, do not try to solve or bypass it. Switch to another website or search
  engine that can also complete the task{switch_hint}; if the task cannot
  continue, `stop` and report what blocked you.
- Never sign in, register, subscribe, purchase, or submit orders. When a task
  involves filling a real form, fill the fields but stop before the final
  irreversible submission.
- A `type` action appends at the cursor; it does NOT erase what is already in
  the field. If a text field already holds wrong or leftover content, never
  `type` on top of it (that produces "oldtextnewtext"). Instead replace it in
  one step with `fill` (which clears then types), or `clear` the field first and
  then `type`. After entering a search query, remember to submit it (press Enter
  via the field, or click the search button).
- If the task lists several required URLs, visit every one of them before
  stopping.
- When clicking a link or button, aim at the visible text itself. For a link
  whose text wraps across two or more lines, click on the first line of the
  text, not the vertical center of the whole element — the center often lands in
  the blank gap between lines and the click does nothing.
- If the page did not change after a click, do NOT click the exact same spot
  again. Either aim closer to the text, or reach the same destination a
  different way — for a paper/result that also shows a short identifier or "PDF"
  link, click that single-line link instead of the long wrapped title.
- If the target is a list of results, an author's page or a dedicated listing is
  usually easier to operate than a crowded search-results page.
- Pages may render slowly; if a screenshot looks half-loaded, prefer `wait`
  over guessing.
- When the task asks for a specific fact (a page range, number, name, date,
  title, price, ranking, etc.), do not answer from memory or guess. Keep
  navigating until the exact answer is actually visible on the current page,
  then `stop` with that answer. Stopping before the evidence is on screen fails
  the task, even if the answer you give happens to be right.
- When the task is genuinely done and the evidence is in front of you, `stop`
  with a concise answer containing the information you gathered. Do not keep
  browsing after that."""


def system_prompt(profile: str, ctx: GroundingContext, analysis_words: int) -> str:
    switch_hint = (
        " (use `goto` with an alternative such as https://duckduckgo.com)"
        if profile == "hybrid" else ""
    )
    return "\n\n".join([
        "You operate a real desktop web browser to carry out the user's task on "
        "live websites. At each step you see a screenshot of the current viewport "
        "and reply with the single next action.",
        "## Locating targets\n\n" + ctx.scheme.doc_convention,
        "## Available actions\n\n" + actions.catalog(profile, ctx),
        GROUND_RULES.format(switch_hint=switch_hint),
        RESPONSE_CONTRACT.format(analysis_words=analysis_words),
    ])


def task_block(task: Task) -> str:
    # Only the instruction is task input.  `steps` and `criteria` are gold
    # annotations and may contain the literal answer or intended action path.
    return "\n".join(["## Task", "", task.instruction.strip()])


def step_block(task: Task, state: PageState, step_index: int, max_steps: int,
               notices: list[str], vision_only: bool = False) -> str:
    lines = [
        task_block(task),
        "",
        f"## Current state — step {step_index + 1} of {max_steps}",
    ]
    if not vision_only:
        # The live URL is the only non-visual page-state field given to the
        # agent.  Gold steps/criteria and other task metadata stay hidden.
        lines += ["", f"URL: {state.url}"]
    if notices:
        lines += ["", "Notices:"]
        lines += [f"- {notice}" for notice in notices]
    lines += ["", "The screenshot of the current viewport is attached. "
                  "Decide the next action."]
    return "\n".join(lines)


CORRECTIVE_TEMPLATE = (
    "Your previous reply could not be executed: {reason}\n"
    "Reply again for the same screenshot. Remember: a short analysis, then exactly "
    'one fenced JSON block of the form {{"action": "<name>", "args": {{...}}}}.'
)


def history_line(step_index: int, action_key: str | None, args: dict | None,
                 url: str) -> str:
    """One-line summary of an old step used when its screenshot is dropped."""
    rendered = "(no action)"
    if action_key:
        compact = {k: v for k, v in (args or {}).items() if k != "analysis"}
        rendered = f"{action_key} {compact}" if compact else action_key
    return f"step {step_index + 1} @ {url}: {rendered}"
