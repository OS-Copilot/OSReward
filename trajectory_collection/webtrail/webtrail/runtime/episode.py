"""One episode: preflight, the observe→decide→act loop, finalization.

Episode life cycle
------------------
1. **Preflight** – open a session, navigate to the start URL, take a cheap
   snapshot, run the block detector. Unreachable or blocked-at-arrival sites
   are recorded in ``rejects.jsonl`` without spending a single model call.
2. **Main loop** – observe (with stability retries), guard-check, ask the
   agent, execute, persist every artifact. Search-engine blocks trigger a
   fallback engine switch; target blocks end the episode. Repeated identical
   page states end the episode as ``stale_loop``.
3. **Finalize** – ``result.json`` gets a machine-readable summary either way.

Statuses: completed | blocked | env_error | stale_loop | agent_error
"""

from __future__ import annotations

import asyncio
import logging
import time

from ..agents.base import AgentFormatError, WebAgent
from ..agents.llm import ChatModel, LLMError
from ..browser import guard, images
from ..browser.actions import ActionError, compile_action
from ..browser.client import BrowserError, BrowserGone, BrowserSession, ServicePool
from ..browser.grounding import GroundingContext, scheme_for_model
from ..browser.vision import model_input_size
from ..core.config import Config
from ..core.models import CompiledAction, PageState, Task, domain_of
from .recorder import RunRecorder
from .url_policy import classify_url_policy

logger = logging.getLogger(__name__)


class _EpisodeEnd(Exception):
    def __init__(self, status: str, *, block: dict | None = None,
                 error: str | None = None, stop_answer: str | None = None):
        self.status = status
        self.block = block
        self.error = error
        self.stop_answer = stop_answer


class _StepTimeoutError(TimeoutError):
    """The wall-clock budget for one complete agent step was exhausted."""


class _StepDeadline:
    """Cancellation-based async deadline compatible with Python 3.10+.

    Component calls already have their own timeouts.  This outer deadline caps
    their combined retries and wait time.  It only converts cancellation fired
    by its own timer; an external task cancellation still propagates normally.
    """

    def __init__(self, timeout_s: float):
        self.timeout_s = max(0.0, float(timeout_s))
        self.started_at = time.monotonic()
        self._task: asyncio.Task | None = None
        self._handle: asyncio.TimerHandle | None = None
        self._expired = False
        self._cancelling_at_enter = 0

    async def __aenter__(self) -> _StepDeadline:  # noqa: PYI034 - Python 3.10
        self.started_at = time.monotonic()
        self._task = asyncio.current_task()
        if self._task is None:
            raise RuntimeError("step deadline requires an asyncio task")
        cancelling = getattr(self._task, "cancelling", None)
        if callable(cancelling):
            self._cancelling_at_enter = cancelling()
        if self.timeout_s > 0:
            self._handle = asyncio.get_running_loop().call_later(
                self.timeout_s, self._expire
            )
        return self

    def _expire(self) -> None:
        self._expired = True
        if self._task is not None:
            self._task.cancel()

    def check(self) -> None:
        """Enforce the deadline around short synchronous persistence work."""
        if self.timeout_s > 0 and self.elapsed_s >= self.timeout_s:
            self._expired = True
            raise _StepTimeoutError

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at

    async def __aexit__(self, exc_type, _exc, _tb) -> bool:
        if self._handle is not None:
            self._handle.cancel()
        if self._expired and exc_type is asyncio.CancelledError:
            # Python 3.11+ counts cancellation requests. Remove only our own;
            # preserve a simultaneous external cancellation if one exists.
            uncancel = getattr(self._task, "uncancel", None)
            if callable(uncancel):
                remaining = uncancel()
                if remaining > self._cancelling_at_enter:
                    return False
            raise _StepTimeoutError from None
        return False


async def _observe(session: BrowserSession, config: Config, *,
                   lite: bool = False) -> PageState:
    """Snapshot with retries for missing screenshots and unstable renders."""
    settings = config.browser
    state: PageState | None = None
    last_error: Exception | None = None
    for attempt in range(settings.snapshot_attempts):
        if attempt:
            await asyncio.sleep(settings.snapshot_retry_wait_s)
        try:
            state = await session.snapshot(lite=lite)
        except BrowserError as err:
            last_error = err
            continue
        if state.screenshot_png is None:
            continue
        final_attempt = attempt == settings.snapshot_attempts - 1
        looks_blank = (
            images.near_uniform(images.load_png(state.screenshot_png))
            and not state.elements
        )
        if looks_blank and not final_attempt:
            continue    # give async renders another chance; guard classifies later
        return state
    if state is not None:
        return state
    raise BrowserError(f"snapshot failed after {settings.snapshot_attempts} "
                       f"attempts: {last_error}")


# JS challenges that may clear on their own vs. blocks that never will
_WAITABLE_BLOCKS = {"challenge"}


async def _wait_out_challenge(session: BrowserSession, config: Config,
                              state: PageState, verdict, *, lite: bool):
    """If the page is a non-interactive JS challenge, wait and re-observe a few
    times; many Cloudflare-style checks clear within ~10-20s. Returns the last
    (state, verdict) seen — cleared if the challenge passed, still blocked if not.
    """
    if verdict.kind not in _WAITABLE_BLOCKS:
        return state, verdict
    for attempt in range(config.browser.challenge_wait_attempts):
        await asyncio.sleep(config.browser.challenge_wait_s)
        try:
            state = await session.snapshot(lite=lite)
        except BrowserError:
            continue
        verdict = guard.inspect_state(state)
        logger.info("challenge wait %d/%d on %s -> %s",
                    attempt + 1, config.browser.challenge_wait_attempts,
                    domain_of(state.url), verdict.kind or "cleared")
        if verdict.kind not in _WAITABLE_BLOCKS:
            break
    return state, verdict


async def _preflight_attempt(task: Task, session: BrowserSession,
                             config: Config, *, accept_blocked_search: bool = True):
    """Try all task URLs once on one browser session.

    Returns ``(ok, state, last_verdict, last_url)``. Session rotation and final
    classification are owned by :func:`run_episode`.
    """
    from ..core.models import Verdict

    last_verdict = Verdict()
    last_state_url = ""
    state: PageState | None = None
    nav_timeout_ms = min(
        config.browser.nav_timeout_ms,
        max(1, config.browser.preflight_nav_timeout_ms),
    )
    for idx, candidate in enumerate(task.urls):
        nav = await session.goto(candidate, timeout_ms=nav_timeout_ms)
        goto_error = None if nav.get("ok") else str(nav.get("error"))
        # Bare domains sometimes only resolve with a www. prefix.
        if goto_error and "ERR_NAME_NOT_RESOLVED" in goto_error:
            from urllib.parse import urlparse, urlunparse
            parts = urlparse(candidate)
            if parts.hostname and not parts.hostname.startswith("www."):
                nav = await session.goto(
                    urlunparse(parts._replace(netloc="www." + parts.netloc)),
                    timeout_ms=nav_timeout_ms,
                )
                if nav.get("ok"):
                    goto_error = None
        state = await _observe(session, config, lite=True)
        # Navigation errors (redirect races, interrupted goto) are moot if a
        # real page loaded anyway; judge the page we actually landed on.
        if (goto_error and state.url.startswith("http")
                and not state.url.startswith("chrome-error")):
            goto_error = None
        verdict = guard.inspect_state(state, goto_error=goto_error)
        if verdict.kind in _WAITABLE_BLOCKS:
            state, verdict = await _wait_out_challenge(
                session, config, state, verdict, lite=True
            )

        reachable = not verdict.blocked or (
            accept_blocked_search
            and verdict.scope == "search" and config.run.search_fallbacks
        )
        if reachable:
            if idx > 0:
                logger.info("%s: start URL blocked, starting from %s instead",
                            task.task_id, candidate)
            return True, state, verdict, state.url
        last_verdict, last_state_url = verdict, state.url

    return False, state, last_verdict, last_state_url


def _fallback_task(task: Task, config: Config) -> Task:
    return Task(
        task_id=task.task_id,
        instruction=task.instruction,
        urls=list(config.run.search_fallbacks),
        action_profile=task.action_profile,
    )


def _page_fingerprint(state: PageState) -> tuple:
    shot_hash = None
    if state.screenshot_png:
        shot_hash = images.dhash(images.load_png(state.screenshot_png))
    return (state.url, state.http_status, (len(state.html or "") // 512), shot_hash)


def _fingerprints_match(a: tuple | None, b: tuple) -> bool:
    if a is None:
        return False
    if a[:3] != b[:3]:
        return False
    if a[3] is None or b[3] is None:
        return a[3] == b[3]
    return images.hamming(a[3], b[3]) <= 2


_DOMAIN_WIDE_BLOCKS = {
    "captcha", "challenge", "rate_limit", "login_wall", "geo_blocked",
}
_ACCESS_CONTROL_BLOCKS = _DOMAIN_WIDE_BLOCKS | {"access_denied"}
_COORDINATE_KEYS = {"x", "y", "x1", "y1", "x2", "y2"}


def _canonical_url(url: str | None) -> str:
    """Normalize only the fragment away; paths and queries remain significant."""
    return (url or "").strip().split("#", 1)[0]


def _fingerprint_value(key: str, value):
    """Make action comparison tolerant of tiny model coordinate jitter."""
    if key in _COORDINATE_KEYS and isinstance(value, (int, float)) \
            and not isinstance(value, bool):
        return round(float(value) / 16.0)
    if isinstance(value, dict):
        return tuple(sorted((k, _fingerprint_value(k, v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_fingerprint_value(key, item) for item in value)
    return value


def _action_fingerprint(compiled: CompiledAction) -> tuple:
    commands = tuple(
        tuple(sorted((key, _fingerprint_value(key, value))
                     for key, value in command.items()))
        for command in compiled.commands
    )
    return compiled.key, _canonical_url(compiled.goto_url), commands


def _scroll_direction(compiled: CompiledAction) -> tuple[str, int] | None:
    """Return the dominant scroll axis and sign for no-progress enforcement."""
    command = next(
        (item for item in compiled.commands if item.get("kind") == "scroll"),
        None,
    )
    if command is None:
        return None
    dx = int(command.get("dx") or 0)
    dy = int(command.get("dy") or 0)
    if abs(dy) >= abs(dx) and dy:
        return "vertical", 1 if dy > 0 else -1
    if dx:
        return "horizontal", 1 if dx > 0 else -1
    return None


def _no_progress_action_error(
    stale_count: int,
    compiled: CompiledAction,
    last_action_key: str | None,
    last_scroll_direction: tuple[str, int] | None,
) -> str | None:
    """Reject actions that prolong a page known not to be changing."""
    if stale_count < 2:
        return None
    if compiled.key == "wait":
        return (
            "The page has remained unchanged for two consecutive observations. "
            "`wait` is now disabled; choose a different action category such as "
            "click, goto, go_back, type, or stop."
        )
    direction = _scroll_direction(compiled)
    if (compiled.key == "scroll" and last_action_key == "scroll"
            and direction is not None and direction == last_scroll_direction):
        return (
            "The page has remained unchanged for two consecutive observations. "
            "Another scroll in the same direction is disabled; reverse direction "
            "or choose a different action category."
        )
    return None


def _block_notice(verdict, url: str) -> str:
    if verdict.kind == "not_found":
        return (
            f"The page at {url} is missing (HTTP 404/410 or an equivalent error "
            "page). The browser remains on this page and the harness has not "
            "performed a recovery action. Explicitly choose `go_back`, a verified "
            "`goto`, or another available action; do not retry that exact URL."
        )
    if verdict.kind == "server_error":
        return (
            f"The page at {url} returned a server error. The browser remains on "
            "this page and the harness has not performed a recovery action. "
            "Explicitly choose the next action and avoid retrying that exact URL."
        )
    if verdict.kind in {"access_denied", "network_error"}:
        return (
            f"The page at {url} is unavailable because of {verdict.kind} "
            f"({verdict.evidence}). The browser remains on this page and the "
            "harness has not gone back, navigated elsewhere, or changed sessions. "
            "Explicitly choose the next browser action."
        )
    return (
        f"{domain_of(url)} is unavailable because of {verdict.kind}. The browser "
        "remains on this page and the harness has not performed a recovery action. "
        "Explicitly choose the next browser action and use another source."
    )


def _preflight_hint(verdict) -> str:
    if verdict.kind in _ACCESS_CONTROL_BLOCKS:
        return " (the site presented an access-control wall)"
    if verdict.kind == "network_error":
        return " (the site was unreachable or timed out; anti-bot blocking is unconfirmed)"
    return ""


def _model_view(state: PageState, max_side: int, model_id: str = ""
                ) -> tuple[bytes | None, tuple[int, int], bytes | None]:
    """Return a provider-adapted model image and its explicit coordinate size."""
    if state.screenshot_png is None:
        return None, state.viewport, None
    image = images.load_png(state.screenshot_png)
    adapted_size = model_input_size(model_id, image.size, max_side)
    resized = images.fit_size(image, adapted_size)
    if resized.size == image.size:
        return state.screenshot_png, state.viewport, None
    png = images.to_png_bytes(resized)
    return png, resized.size, png


async def run_episode(task: Task, config: Config, pool: ServicePool,
                      run_recorder: RunRecorder, model: ChatModel) -> dict:
    """Collect one trajectory. Always returns a result dict (also on rejects)."""
    profile = task.action_profile or config.run.action_profile
    max_steps = task.max_steps or config.run.max_steps
    scheme = scheme_for_model(config.model.model, config.model.grounding)
    url_policy = classify_url_policy(task)

    # ---------------- preflight ----------------
    # Retry transport/network failures once on the next round-robin worker.
    # CAPTCHA/access-control/dead-page verdicts are deterministic and are not
    # retried. Network errors are infrastructure outcomes, never "blocked".
    session: BrowserSession | None = None
    state: PageState | None = None
    failed_host: str | None = None
    fallback_only = False
    fallback_used = False
    original_failure_kind: str | None = None
    original_failure_detail = ""
    preflight_notices: list[str] = []
    preflight_blocked_domains: set[str] = set()
    preflight_blocked_urls: set[str] = set()
    fallback_task = _fallback_task(task, config)
    attempts = max(1, config.browser.preflight_session_retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            session = await asyncio.wait_for(
                pool.open_session(avoid_host=failed_host),
                timeout=min(20.0, config.browser.preflight_attempt_timeout_s),
            )
            check_task = fallback_task if fallback_only else task
            preflight_ok, state, last_verdict, last_state_url = await asyncio.wait_for(
                _preflight_attempt(
                    check_task, session, config,
                    accept_blocked_search=not fallback_only,
                ),
                timeout=config.browser.preflight_attempt_timeout_s,
            )
        except (BrowserError, BrowserGone, asyncio.TimeoutError) as err:
            if session is not None:
                failed_host = getattr(session, "service_host", None)
                await session.close()
                session = None
            detail = (
                f"preflight attempt exceeded "
                f"{config.browser.preflight_attempt_timeout_s:g}s"
                if isinstance(err, asyncio.TimeoutError) else str(err)
            )
            if (not url_policy.hard_required and not fallback_only
                    and config.run.search_fallbacks and attempt < attempts):
                original_failure_kind = "unreachable"
                original_failure_detail = detail
                fallback_only = True
                logger.warning(
                    "%s: flexible start URL preflight failed (%s); switching "
                    "worker and starting the agent from a search engine",
                    task.task_id, detail,
                )
                continue
            if attempt < attempts:
                logger.warning(
                    "%s: preflight session attempt %d/%d failed (%s); "
                    "switching browser worker",
                    task.task_id, attempt, attempts, detail,
                )
                continue
            await run_recorder.reject(
                task, "unreachable",
                f"preflight failed after {attempts} session attempt(s): {detail}",
            )
            return {"task_id": task.task_id, "status": "env_error",
                    "preflight": True, "preflight_attempts": attempt}

        if preflight_ok:
            if fallback_only:
                fallback_used = True
                logger.info(
                    "%s: flexible start URL unavailable (%s); agent starts at %s",
                    task.task_id, original_failure_kind, state.url,
                )
            break

        hint = _preflight_hint(last_verdict)
        detail = (
            f"preflight: all {len(check_task.urls)} url(s) failed; last "
            f"{last_verdict.evidence} @ {last_state_url}{hint}"
        )
        if not url_policy.hard_required and not fallback_only \
                and config.run.search_fallbacks:
            original_failure_kind = last_verdict.kind or "unavailable"
            original_failure_detail = detail
            if last_verdict.kind in _DOMAIN_WIDE_BLOCKS:
                preflight_blocked_domains.update(domain_of(url) for url in task.urls)
            else:
                preflight_blocked_urls.update(_canonical_url(url) for url in task.urls)
                if last_state_url:
                    preflight_blocked_urls.add(_canonical_url(last_state_url))
            # Never reuse a session that just rendered a CAPTCHA/error page or
            # suffered a failed navigation. Open the search page on a fresh,
            # different Chromium worker so target-page state cannot wedge the
            # fallback navigation.
            failed_host = getattr(session, "service_host", None)
            await session.close()
            session = None
            fallback_only = True
            if attempt < attempts:
                logger.info(
                    "%s: flexible start URL unavailable (%s); switching worker "
                    "before starting the agent from a search engine",
                    task.task_id, original_failure_kind,
                )
                continue
            await run_recorder.reject(task, "fallback_unavailable", detail)
            return {"task_id": task.task_id, "status": "env_error",
                    "preflight": True, "preflight_attempts": attempt,
                    "url_policy": url_policy.kind}

        if last_verdict.kind == "network_error":
            failed_host = getattr(session, "service_host", None)
            await session.close()
            session = None
            if attempt < attempts:
                logger.warning(
                    "%s: preflight network failure on attempt %d/%d; "
                    "switching browser worker",
                    task.task_id, attempt, attempts,
                )
                continue
            await run_recorder.reject(task, "network_error", detail)
            return {"task_id": task.task_id, "status": "env_error",
                    "block_kind": last_verdict.kind, "preflight": True,
                    "preflight_attempts": attempt}

        await session.close()
        await run_recorder.reject(task, last_verdict.kind or "blocked", detail)
        return {"task_id": task.task_id, "status": "blocked",
                "block_kind": last_verdict.kind, "preflight": True,
                "preflight_attempts": attempt}

    if session is None or state is None:
        raise RuntimeError("preflight succeeded without a live browser session")

    if fallback_used:
        for url in task.urls:
            preflight_blocked_urls.add(_canonical_url(url))
        preflight_notices.append(
            f"The provided start URL {task.start_url} was unavailable during "
            f"preflight ({original_failure_kind}: {original_failure_detail[:180]}). "
            "This is not a hard-required website task, so continue from the "
            "current search page and use another credible source. Do not retry "
            "the unavailable start URL."
        )

    # ---------------- main loop ----------------
    recorder = run_recorder.open_trajectory(task)
    ctx_base = GroundingContext(scheme=scheme, viewport=session.viewport,
                                sent_size=session.viewport)
    agent = WebAgent(model, config.model, config.run, ctx_base, profile)

    counters = {"action_errors": 0, "parse_retries": 0, "stale_repeats": 0,
                "same_action_repeats": 0,
                "fallback_switches": 0, "guard_flags": 0, "block_recoveries": 0,
                "navigation_recoveries": 0, "no_progress_action_refusals": 0,
                "step_timeouts": 0, "consecutive_step_timeouts": 0,
                "preflight_fallbacks": int(fallback_used),
                "max_steps_reached": 0}
    action_keys: list[str] = []
    missing_screenshots: list[int] = []
    blocked_domains: set[str] = set(preflight_blocked_domains)
    blocked_urls: set[str] = set(preflight_blocked_urls)
    notices: list[str] = list(preflight_notices)
    previous_fingerprint: tuple | None = None
    stale_count = 0
    last_action_fingerprint: tuple | None = None
    last_action_url: str | None = None
    same_action_repeats = 0
    last_action_key: str | None = None
    last_scroll_direction: tuple[str, int] | None = None
    pending_navigation_url: str | None = None
    step = 0
    final_url: str | None = state.url if state else task.start_url
    outcome: _EpisodeEnd

    try:
        iterations = 0
        while step < max_steps:
            iterations += 1
            if iterations > max_steps + 2:
                raise _EpisodeEnd("stale_loop", error="iteration guard tripped")

            step_phase = "observe"
            step_state: PageState | None = None
            step_verdict = None
            model_view_png: bytes | None = None
            sent_size = session.viewport
            step_notices: list[str] = []
            decision = None
            compiled: CompiledAction | None = None
            command_results: list[dict] = []
            step_deadline = _StepDeadline(config.run.step_timeout_s)

            try:
                async with step_deadline:
                    state = await _observe(session, config)
                    step_state = state
                    final_url = state.url
                    attempted_navigation_url = pending_navigation_url
                    pending_navigation_url = None
                    if state.screenshot_png is None:
                        missing_screenshots.append(step)
                        recorder.save_observation(step, state, None)
                        raise _EpisodeEnd(
                            "env_error",
                            error="screenshot unavailable after retries",
                        )

                    verdict = guard.inspect_state(state)
                    step_verdict = verdict
                    if verdict.kind in _WAITABLE_BLOCKS:
                        step_phase = "challenge_wait"
                        state, verdict = await _wait_out_challenge(
                            session, config, state, verdict, lite=False)
                        step_state, step_verdict = state, verdict
                        final_url = state.url
                    if verdict.blocked:
                        counters["guard_flags"] += 1
                        # Once the agent has started, every browser state change
                        # must come from a recorded model action.  Preserve the
                        # exact error page/session, annotate the observation, and
                        # let the model explicitly choose go_back/goto/stop.  In
                        # particular, never hide a back, fallback navigation, or
                        # session replacement inside the harness.
                        failed_url = attempted_navigation_url or state.url
                        if verdict.kind in _DOMAIN_WIDE_BLOCKS:
                            blocked_domains.add(domain_of(failed_url))
                        else:
                            blocked_urls.add(_canonical_url(failed_url))
                        notices.append(_block_notice(verdict, failed_url))

                    fingerprint = _page_fingerprint(state)
                    if _fingerprints_match(previous_fingerprint, fingerprint):
                        stale_count += 1
                        counters["stale_repeats"] += 1
                        notices.append(
                            "The page did not change after your last action. Reconsider "
                            "the target position or try a different approach."
                        )
                    else:
                        stale_count = 0
                    previous_fingerprint = fingerprint
                    if stale_count >= config.run.stale_limit:
                        recorder.save_observation(step, state, verdict)
                        step_deadline.check()
                        raise _EpisodeEnd("stale_loop", error=(
                            f"{stale_count} identical page states in a row"
                        ))

                    model_png, sent_size, model_view_png = _model_view(
                        state, config.model.image_max_side, config.model.model
                    )
                    ctx = GroundingContext(scheme=scheme, viewport=session.viewport,
                                           sent_size=sent_size)

                    def validate(parsed, _ctx=ctx):
                        try:
                            return compile_action(parsed, _ctx, profile)
                        except ActionError as err:
                            raise ValueError(str(err)) from None

                    step_notices, notices = notices, []
                    step_phase = "model"
                    policy_refusals = 0
                    decision_notices = list(step_notices)
                    while True:
                        try:
                            decision = await agent.decide(
                                task, state, step, max_steps, decision_notices,
                                model_png, validator=validate,
                            )
                        except AgentFormatError as err:
                            recorder.save_observation(
                                step, state, verdict, model_view_png
                            )
                            raise _EpisodeEnd("agent_error", error=str(err))
                        except LLMError as err:
                            recorder.save_observation(
                                step, state, verdict, model_view_png
                            )
                            raise _EpisodeEnd("env_error", error=f"llm: {err}")

                        compiled = decision.compiled  # type: ignore[assignment]
                        policy_error = _no_progress_action_error(
                            stale_count,
                            compiled,
                            last_action_key,
                            last_scroll_direction,
                        )
                        if policy_error is None:
                            break
                        policy_refusals += 1
                        counters["no_progress_action_refusals"] += 1
                        step_notices.append(policy_error)
                        record_action_result = getattr(
                            agent, "record_action_result", None
                        )
                        if callable(record_action_result):
                            record_action_result(
                                step,
                                [{
                                    "ok": False,
                                    "error": policy_error,
                                    "reason": "no_progress_action_policy",
                                }],
                                post_url=state.url,
                            )
                        if policy_refusals > config.run.parse_retries:
                            recorder.save_observation(
                                step, state, verdict, model_view_png
                            )
                            raise _EpisodeEnd(
                                "stale_loop",
                                error=(
                                    "model repeatedly selected an action disabled "
                                    "after two unchanged observations"
                                ),
                            )
                        decision_notices = [policy_error]

                    counters["parse_retries"] += decision.parse_attempts - 1
                    action_keys.append(compiled.key)

                    fingerprint = _action_fingerprint(compiled)
                    action_url = _canonical_url(state.url)
                    if (fingerprint == last_action_fingerprint
                            and action_url == last_action_url):
                        same_action_repeats += 1
                        counters["same_action_repeats"] += 1
                    else:
                        same_action_repeats = 0
                    last_action_fingerprint = fingerprint
                    last_action_url = action_url
                    repeated_action_refused = (
                        config.run.max_same_action_repeats > 0
                        and same_action_repeats >= config.run.max_same_action_repeats
                    )

                    # execute
                    step_phase = "action"
                    if repeated_action_refused:
                        command_results.append({
                            "ok": False,
                            "error": (
                                "refused repeated ineffective action: the same action was "
                                "already attempted on an unchanged page"
                            ),
                            "reason": "same_action_on_unchanged_page",
                        })
                    elif compiled.goto_url:
                        # refuse to re-enter a site we've already found to be blocked:
                        # models tend to loop back to the "official" site (e.g. the
                        # rail operator) even after being told it is unreachable
                        target_url = _canonical_url(compiled.goto_url)
                        if (domain_of(compiled.goto_url) in blocked_domains
                                or target_url in blocked_urls):
                            notices.append(
                                f"{compiled.goto_url} is known to be blocked or unavailable. "
                                "Do not retry it; use a different page or source."
                            )
                            command_results.append({"ok": False, "error": "navigation to "
                                                    "known-blocked destination refused"})
                        else:
                            pending_navigation_url = target_url
                            command_results.append(await session.goto(compiled.goto_url))
                    if not repeated_action_refused:
                        for command in compiled.commands:
                            command_results.append(await session.act(command))
                            if not command_results[-1].get("ok"):
                                break
                    failed = next((r for r in command_results if not r.get("ok")), None)
                    if failed is not None:
                        counters["action_errors"] += 1
                        notices.append(
                            f"Your `{compiled.key}` action reported an error: "
                            f"{str(failed.get('error'))[:200]}"
                        )
                    elif same_action_repeats > 0:
                        notices.append(
                            f"You repeated the same `{compiled.key}` action at the same "
                            "place on the same URL. Do not repeat it again; choose a "
                            "different target or approach if the task did not progress."
                        )

                    last_action_key = compiled.key
                    last_scroll_direction = _scroll_direction(compiled)

                    # Browser-use-style semantic history: attach the actual execution
                    # result to the model turn.  Only the small service response is
                    # replayed; observations such as DOM/HTML/elements never enter the
                    # model context.
                    record_action_result = getattr(agent, "record_action_result", None)
                    if callable(record_action_result):
                        post_url = next(
                            (str(result["final_url"]) for result in reversed(command_results)
                             if result.get("final_url")),
                            state.url,
                        )
                        record_action_result(step, command_results, post_url=post_url)

                    step_phase = "persistence"
                    step_deadline.check()
                    recorder.save_observation(step, state, verdict, model_view_png)
                    recorder.save_decision(
                        step,
                        reply_text=decision.reply.text,
                        parsed_key=decision.parsed.key,
                        parsed_args=decision.parsed.args,
                        analysis=decision.parsed.analysis,
                        compiled=compiled,
                        command_results=command_results,
                        usage=decision.reply.usage,
                        latency_s=decision.reply.latency_s,
                        parse_attempts=decision.parse_attempts,
                        sent_size=sent_size,
                        notices=step_notices,
                        messages=(agent.export_messages() if config.run.save_messages else None),
                    )
                    if config.run.annotate_screenshots and state.screenshot_png:
                        try:
                            recorder.save_annotation(step, state.screenshot_png, compiled)
                        except Exception:                  # never fail a step on drawing
                            logger.exception("annotation failed for %s step %d",
                                             task.task_id, step)
                    step_deadline.check()
                    counters["consecutive_step_timeouts"] = 0

                    if compiled.is_stop:
                        raise _EpisodeEnd("completed", stop_answer=compiled.stop_answer)
                    if repeated_action_refused:
                        step += 1
                        raise _EpisodeEnd(
                            "stale_loop",
                            error=(
                                f"same action repeated {same_action_repeats + 1} times "
                                "on the same URL"
                            ),
                        )
                    step += 1

            except _StepTimeoutError:
                timed_out_step = step
                elapsed_s = step_deadline.elapsed_s
                timeout_s = config.run.step_timeout_s
                timeout_error = (
                    f"step {timed_out_step + 1} exceeded the {timeout_s:g}s "
                    f"wall-clock timeout during {step_phase}"
                )
                logger.warning("%s: %s", task.task_id, timeout_error)
                counters["step_timeouts"] += 1
                counters["consecutive_step_timeouts"] += 1

                if step_state is not None:
                    final_url = step_state.url
                    recorder.save_observation(
                        timed_out_step, step_state, step_verdict, model_view_png
                    )
                elif timed_out_step not in missing_screenshots:
                    missing_screenshots.append(timed_out_step)

                timeout_result = {
                    "ok": False,
                    "error": timeout_error,
                    "phase": step_phase,
                    "outcome_uncertain": step_phase == "action",
                }
                command_results.append(timeout_result)
                record_action_result = getattr(agent, "record_action_result", None)
                if decision is not None and callable(record_action_result):
                    record_action_result(
                        timed_out_step,
                        command_results,
                        post_url=(step_state.url if step_state is not None else final_url),
                    )

                recorder.save_decision(
                    timed_out_step,
                    reply_text=decision.reply.text if decision is not None else "",
                    parsed_key=decision.parsed.key if decision is not None else None,
                    parsed_args=decision.parsed.args if decision is not None else None,
                    analysis=decision.parsed.analysis if decision is not None else "",
                    compiled=compiled,
                    command_results=command_results,
                    usage=decision.reply.usage if decision is not None else {},
                    latency_s=elapsed_s,
                    parse_attempts=(decision.parse_attempts
                                    if decision is not None else 0),
                    sent_size=sent_size,
                    notices=step_notices,
                    messages=(agent.export_messages()
                              if config.run.save_messages else None),
                )

                notices.append(
                    f"The previous step timed out during `{step_phase}`. Its "
                    "action outcome may be uncertain; inspect the fresh current "
                    "screenshot before choosing the next action."
                )
                previous_fingerprint = None
                stale_count = 0
                last_action_fingerprint = None
                last_action_url = None
                same_action_repeats = 0
                last_action_key = None
                last_scroll_direction = None
                step += 1
                if counters["consecutive_step_timeouts"] >= max(
                        1, config.run.max_consecutive_step_timeouts):
                    raise _EpisodeEnd("env_error", error=(
                        f"{counters['consecutive_step_timeouts']} consecutive "
                        f"step timeouts; last: {timeout_error}"
                    ))
                continue

        # Reaching the configured interaction budget is a valid completed
        # trajectory. Keep a counter so downstream users can still distinguish
        # explicit model stops from budget-complete episodes without treating
        # the latter as collection failures.
        counters["max_steps_reached"] = 1
        outcome = _EpisodeEnd("completed")
    except _EpisodeEnd as end:
        outcome = end
    except BrowserGone as err:
        outcome = _EpisodeEnd("env_error", error=f"browser session lost: {err}")
    except BrowserError as err:
        outcome = _EpisodeEnd("env_error", error=str(err))
    finally:
        await session.close()
        closer = getattr(agent, "close", None)
        if closer is not None:
            await closer()

    result = recorder.save_result(
        task,
        status=outcome.status,
        steps_taken=step + (
            1 if outcome.status == "completed"
            and not counters["max_steps_reached"] else 0
        ),
        final_url=final_url,
        stop_answer=outcome.stop_answer,
        block=outcome.block,
        counters=counters,
        missing_screenshot_steps=missing_screenshots,
        action_keys=action_keys,
        error=outcome.error,
    )
    return result
