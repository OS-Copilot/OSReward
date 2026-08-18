"""Prompt assembly for the browsing agent.

All text is generated from the action profile and grounding scheme so that a
single source of truth defines what the model may do and how it must localize
targets. Nothing here depends on a specific model family.
"""

from __future__ import annotations

from ..browser import actions
from ..browser.grounding import GroundingContext
from ..core.models import PageState, Task

RESPONSE_CONTRACT = """## How to respond

First write a compact analysis (about {analysis_words} words): what the current
screenshot shows, what your last action changed, how far the task has
progressed, and what to do next. Then output exactly ONE action as JSON in a
fenced block:

```json
{{"action": "<name>", "args": {{ ... }}}}
```

Never output more than one JSON block. Never invent action names or argument
fields that are not documented above. Explicitly judge the previous action as
successful, failed, or uncertain from the current screenshot. An Action Result
only confirms whether the browser command executed; it does not by itself prove
that the intended page change happened."""

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

GOTO_RULES = """## URL navigation rules

- Prefer clicking visible links, menus, search results, and page controls. For
  filters and searches, operate the site's UI instead of constructing a URL.
- Never invent or guess a URL path, article slug, filename, query string, or
  filter parameter from a page title, task wording, or a site's apparent URL
  pattern. A plausible-looking URL is not evidence that the page exists.
- Use `goto` only with an exact full URL copied verbatim from the task, the
  current state, a notice or prior browser result, or text visibly displayed on
  the page. The only exception is opening one of these known search-engine
  homepages: start with `https://www.bing.com`; if Bing is blocked, explicitly
  switch to `https://duckduckgo.com`, then use `https://www.google.com` only as
  the last fallback. Enter the query through the homepage's search field.
- If a search engine shows a CAPTCHA, verification page, rate limit, or server
  error, do not reload, wait on, or repeatedly retry that engine. Use `goto` to
  switch explicitly to the next search-engine homepage listed above.
- On a search-results page, click the visible result. Do not convert its title
  into a guessed destination URL.
- If a URL produces a missing-page, access-denied, or server-error notice, do
  not try variations of its path or slug. Use the recovered page, visible links,
or a search engine to find a verified route."""

SCROLL_RULES = """## Scroll limit

- Never issue more than five `scroll` actions in consecutive turns. Scrolling
  up versus down, changing the distance, or changing the pointer position does
  not reset this count.
- After five consecutive scrolls, the next action MUST be a non-scroll action
  such as `click`, `type`, `select_option`, `goto`, `go_back`, or `stop`.
- Repeatedly scrolling up and down over the same page is not progress. If five
  scrolls have not revealed a useful next action or the required evidence,
  choose a different route or `stop` with the best supported answer available."""


def system_prompt(profile: str, ctx: GroundingContext, analysis_words: int) -> str:
    switch_hint = (
        " (try Bing first, then DuckDuckGo, with Google as the last fallback)"
        if profile == "hybrid" else ""
    )
    sections = [
        (
            "You operate a real desktop web browser to carry out the user's task on "
            "live websites. At each step you see a screenshot of the current viewport "
            "and reply with the single next action."
        ),
        "## Locating targets\n\n" + ctx.scheme.doc_convention,
        "## Available actions\n\n" + actions.catalog(profile, ctx),
    ]
    if profile == "hybrid":
        sections.append(GOTO_RULES)
    sections.extend([
        SCROLL_RULES,
        GROUND_RULES.format(switch_hint=switch_hint),
        RESPONSE_CONTRACT.format(analysis_words=analysis_words),
    ])
    return "\n\n".join(sections)


def task_block(task: Task) -> str:
    # The acting agent receives only the instruction.  Gold `steps`,
    # `criteria`, and all other task metadata stay out of its prompt.
    return f"## Task\n\n{task.instruction.strip()}"


def step_block(task: Task, state: PageState, step_index: int, max_steps: int,
               notices: list[str], vision_only: bool = False) -> str:
    lines = [
        task_block(task),
        "",
        f"## Current state — step {step_index + 1} of {max_steps}",
    ]
    if not vision_only:
        # The live URL is the only non-visual page-state field exposed.
        lines += ["", f"URL: {state.url}"]
    if notices:
        lines += ["", "Notices:"]
        lines += [f"- {notice}" for notice in notices]
    lines += [
        "",
        (
            "The screenshot of the current viewport is attached. "
            "Decide the next action."
        ),
    ]
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
