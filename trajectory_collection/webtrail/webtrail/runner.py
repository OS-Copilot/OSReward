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

Statuses: completed | max_steps | blocked | env_error | stale_loop | agent_error
"""

from __future__ import annotations

import asyncio
import logging

from . import guard, imutil
from .actions import ActionError, compile_action
from .agent import AgentFormatError, WebAgent
from .browser import BrowserError, BrowserGone, BrowserSession, ServicePool
from .config import Config
from .grounding import GroundingContext, scheme_for_model
from .llm import ChatModel, LLMError
from .recorder import RunRecorder
from .types import CompiledAction, PageState, Task, domain_of

logger = logging.getLogger(__name__)


class _EpisodeEnd(Exception):
    def __init__(self, status: str, *, block: dict | None = None,
                 error: str | None = None, stop_answer: str | None = None):
        self.status = status
        self.block = block
        self.error = error
        self.stop_answer = stop_answer


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
            imutil.near_uniform(imutil.load_png(state.screenshot_png))
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


def _page_signature(state: PageState) -> tuple:
    shot_hash = None
    if state.screenshot_png:
        shot_hash = imutil.dhash(imutil.load_png(state.screenshot_png))
    return (state.url, (len(state.html or "") // 512), shot_hash)


def _signatures_match(a: tuple | None, b: tuple) -> bool:
    if a is None:
        return False
    if a[0] != b[0] or a[1] != b[1]:
        return False
    if a[2] is None or b[2] is None:
        return a[2] == b[2]
    return imutil.hamming(a[2], b[2]) <= 2


def _model_view(state: PageState, max_side: int) -> tuple[bytes | None, tuple[int, int], bytes | None]:
    """Return (png sent to the model, its size, resized copy for disk if any)."""
    if state.screenshot_png is None:
        return None, state.viewport, None
    if max_side <= 0:
        return state.screenshot_png, state.viewport, None
    image = imutil.load_png(state.screenshot_png)
    resized = imutil.fit_max_side(image, max_side)
    if resized.size == image.size:
        return state.screenshot_png, state.viewport, None
    png = imutil.to_png_bytes(resized)
    return png, resized.size, png


async def run_episode(task: Task, config: Config, pool: ServicePool,
                      run_recorder: RunRecorder, model: ChatModel) -> dict:
    """Collect one trajectory. Always returns a result dict (also on rejects)."""
    profile = task.action_profile or config.run.action_profile
    max_steps = task.max_steps or config.run.max_steps
    scheme = scheme_for_model(config.model.model, config.model.grounding)

    # ---------------- preflight ----------------
    try:
        session = await pool.open_session()
    except BrowserGone as err:
        await run_recorder.reject(task, "service_unavailable", str(err))
        return {"task_id": task.task_id, "status": "env_error", "preflight": True}

    # Preflight each of the task's URLs in order; the first one that opens and
    # is not blocked becomes the start. For a multi-URL task (e.g. "book a train
    # on site A, then a hotel on site B") a blocked first site must not kill the
    # whole task — the agent can still work the reachable half and reach the
    # blocked half's goal via a search engine.
    from .types import Verdict
    last_verdict = Verdict()
    last_state_url = ""
    preflight_ok = False
    state: PageState | None = None
    try:
        for idx, candidate in enumerate(task.urls):
            nav = await session.goto(candidate)
            goto_error = None if nav.get("ok") else str(nav.get("error"))
            # bare domains sometimes only resolve with a www. prefix
            if goto_error and "ERR_NAME_NOT_RESOLVED" in goto_error:
                from urllib.parse import urlparse, urlunparse
                parts = urlparse(candidate)
                if parts.hostname and not parts.hostname.startswith("www."):
                    nav = await session.goto(
                        urlunparse(parts._replace(netloc="www." + parts.netloc)))
                    if nav.get("ok"):
                        goto_error = None
            state = await _observe(session, config, lite=True)
            # navigation "errors" (redirect races, interrupted goto) are moot if
            # a real page loaded anyway; judge the page we actually landed on
            if goto_error and state.url.startswith("http") \
                    and not state.url.startswith("chrome-error"):
                goto_error = None
            verdict = guard.inspect_state(state, goto_error=goto_error)
            if verdict.kind in _WAITABLE_BLOCKS:
                state, verdict = await _wait_out_challenge(
                    session, config, state, verdict, lite=True)

            reachable = not verdict.blocked or (
                verdict.scope == "search" and config.run.search_fallbacks)
            if reachable:
                preflight_ok = True
                if idx > 0:
                    logger.info("%s: start URL blocked, starting from %s instead",
                                task.task_id, candidate)
                break
            last_verdict, last_state_url = verdict, state.url

        if not preflight_ok:
            hint = (" (target site hard-blocks automation; a residential proxy "
                    "is likely required)" if last_verdict.scope == "target" else "")
            await run_recorder.reject(
                task, last_verdict.kind or "blocked",
                f"preflight: all {len(task.urls)} url(s) blocked; last "
                f"{last_verdict.evidence} @ {last_state_url}{hint}",
            )
            return {"task_id": task.task_id, "status": "blocked",
                    "block_kind": last_verdict.kind, "preflight": True}
    except (BrowserError, BrowserGone) as err:
        await session.close()
        await run_recorder.reject(task, "unreachable", str(err))
        return {"task_id": task.task_id, "status": "env_error", "preflight": True}

    # ---------------- main loop ----------------
    recorder = run_recorder.open_trajectory(task)
    ctx_base = GroundingContext(scheme=scheme, viewport=session.viewport,
                                sent_size=session.viewport)
    if config.model.backend == "claude_cua":
        from .native_claude import ClaudeComputerAgent
        agent = ClaudeComputerAgent(
            config.model, config.run, session.viewport,
            api_log_path=run_recorder.api_log_path if config.run.api_log else None,
        )
    else:
        agent = WebAgent(model, config.model, config.run, ctx_base, profile)

    counters = {"action_errors": 0, "parse_retries": 0, "stale_repeats": 0,
                "fallback_switches": 0, "guard_flags": 0, "block_recoveries": 0}
    action_keys: list[str] = []
    missing_screenshots: list[int] = []
    blocked_domains: set[str] = set()
    notices: list[str] = []
    previous_signature: tuple | None = None
    stale_count = 0
    step = 0
    final_url: str | None = state.url if state else task.start_url
    outcome: _EpisodeEnd

    try:
        iterations = 0
        while step < max_steps:
            iterations += 1
            if iterations > max_steps + config.run.max_fallback_switches + 2:
                raise _EpisodeEnd("stale_loop", error="iteration guard tripped")

            state = await _observe(session, config)
            final_url = state.url
            if state.screenshot_png is None:
                missing_screenshots.append(step)
                recorder.save_observation(step, state, None)
                raise _EpisodeEnd(
                    "env_error",
                    error="screenshot unavailable after retries",
                )

            verdict = guard.inspect_state(state)
            if verdict.kind in _WAITABLE_BLOCKS:
                state, verdict = await _wait_out_challenge(
                    session, config, state, verdict, lite=False)
                final_url = state.url
            if verdict.blocked:
                counters["guard_flags"] += 1
                recorder.save_observation(step, state, verdict)
                if (verdict.scope == "search"
                        and counters["fallback_switches"] < config.run.max_fallback_switches
                        and config.run.search_fallbacks):
                    fallback = config.run.search_fallbacks[
                        counters["fallback_switches"] % len(config.run.search_fallbacks)
                    ]
                    counters["fallback_switches"] += 1
                    notices.append(
                        f"The search engine at {domain_of(state.url)} blocked "
                        f"automated access; you were moved to {fallback}. Continue "
                        "the task from there."
                    )
                    await session.goto(fallback)
                    continue
                # A target block the agent navigated INTO (not the start page):
                # don't end the episode — step back and tell it to take another
                # route (the other required site, or a search engine). Only give
                # up after repeated dead ends.
                if step > 0 and counters["block_recoveries"] < config.run.max_block_recoveries:
                    counters["block_recoveries"] += 1
                    blocked_domains.add(domain_of(state.url))
                    notices.append(
                        f"{domain_of(state.url)} blocks automated access "
                        f"(a {verdict.kind}); you cannot proceed there. You have "
                        "been sent back. Complete this part another way — use the "
                        "other required site, or a search engine for the "
                        "information — and do not return to that site."
                    )
                    await session.act({"kind": "back"})
                    continue
                raise _EpisodeEnd("blocked", block={
                    "kind": verdict.kind, "scope": verdict.scope,
                    "evidence": verdict.evidence, "step": step, "url": state.url,
                })

            signature = _page_signature(state)
            if _signatures_match(previous_signature, signature):
                stale_count += 1
                counters["stale_repeats"] += 1
                notices.append(
                    "The page did not change after your last action. Reconsider "
                    "the target position or try a different approach."
                )
            else:
                stale_count = 0
            previous_signature = signature
            if stale_count >= config.run.stale_limit:
                recorder.save_observation(step, state, verdict)
                raise _EpisodeEnd("stale_loop", error=(
                    f"{stale_count} identical page states in a row"
                ))

            model_png, sent_size, model_view_png = _model_view(
                state, config.model.image_max_side
            )
            ctx = GroundingContext(scheme=scheme, viewport=session.viewport,
                                   sent_size=sent_size)

            def validate(parsed, _ctx=ctx):
                try:
                    return compile_action(parsed, _ctx, profile)
                except ActionError as err:
                    raise ValueError(str(err)) from None

            step_notices, notices = notices, []
            try:
                decision = await agent.decide(
                    task, state, step, max_steps, step_notices, model_png,
                    validator=validate,
                )
            except AgentFormatError as err:
                recorder.save_observation(step, state, verdict, model_view_png)
                raise _EpisodeEnd("agent_error", error=str(err))
            except LLMError as err:
                recorder.save_observation(step, state, verdict, model_view_png)
                raise _EpisodeEnd("env_error", error=f"llm: {err}")

            compiled: CompiledAction = decision.compiled  # type: ignore[assignment]
            counters["parse_retries"] += decision.parse_attempts - 1
            action_keys.append(compiled.key)

            # execute
            command_results: list[dict] = []
            if compiled.goto_url:
                # refuse to re-enter a site we've already found to be blocked:
                # models tend to loop back to the "official" site (e.g. the
                # rail operator) even after being told it is unreachable
                if domain_of(compiled.goto_url) in blocked_domains:
                    notices.append(
                        f"{domain_of(compiled.goto_url)} is blocked for automated "
                        "access — you already tried it. Do not go there again; "
                        "use a different website to get this information."
                    )
                    command_results.append({"ok": False, "error": "navigation to "
                                            "known-blocked site refused"})
                else:
                    command_results.append(await session.goto(compiled.goto_url))
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
                except Exception:                          # never fail a step on drawing
                    logger.exception("annotation failed for %s step %d",
                                     task.task_id, step)

            if compiled.is_stop:
                raise _EpisodeEnd("completed", stop_answer=compiled.stop_answer)
            step += 1

        outcome = _EpisodeEnd("max_steps")
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
        steps_taken=step + (1 if outcome.status == "completed" else 0),
        final_url=final_url,
        stop_answer=outcome.stop_answer,
        block=outcome.block,
        counters=counters,
        missing_screenshot_steps=missing_screenshots,
        action_keys=action_keys,
        error=outcome.error,
    )
    return result
