"""Claude native computer-use backend.

An alternative agent backend that drives Claude through Anthropic's native
`computer` tool (the Messages API tool-use loop) instead of the prompt +
fenced-JSON scheme used by :class:`webtrail.agent.WebAgent`. Claude emits
`tool_use` blocks with pixel coordinates in the space of the screenshot it was
shown; those are translated back to the real browser viewport and compiled to
the same `CompiledAction` the browser service executes, so recording,
annotation, and post-processing are unchanged.

Conversation shape (managed here, not by the runner)::

    user      task text + screenshot
    assistant [thinking text] + tool_use(computer, {action, coordinate, ...})
    user      tool_result(screenshot after the action)
    assistant tool_use ...
    ...
    assistant text only, end_turn            -> treated as `stop`

The agent exposes the same ``decide(...)`` signature as ``WebAgent`` and
returns the same :class:`~webtrail.agent.Decision`, so the runner does not care
which backend is in use.
"""

from __future__ import annotations

import base64
import json
import time

import httpx

from .agent import Decision
from .config import ModelSettings, RunSettings
from .grounding import GroundingContext, Resolved, Scheme, resolve_target
from .imutil import fit_max_side, load_png, to_png_bytes
from .llm import LLMError, ChatReply
from .prompts import step_block, task_block
from .types import CompiledAction, PageState, ParsedAction, Task

TOOL_VERSION = "computer_20250124"
BETA_HEADER = "computer-use-2025-01-24"

# Anthropic recommends a display no larger than ~XGA/WXGA; larger screenshots
# get downscaled inside the model and throw the coordinates off.
DISPLAY_MAX_SIDE = 1280

_PIXEL_SCHEME = Scheme(id="pixel", doc_target="", doc_convention="", doc_example="")

SYSTEM_PROMPT = (
    "You operate a real desktop web browser to complete the user's task on live "
    "websites using the computer tool. You are shown a screenshot of the current "
    "viewport; act one step at a time.\n\n"
    "Rules:\n"
    "- Do not attempt to solve CAPTCHAs or bot-check interstitials. If a target "
    "page is blocked, explain that and stop.\n"
    "- Never sign in, register, purchase, or submit an order. Fill forms but stop "
    "before an irreversible submission.\n"
    "- If a field holds wrong text, clear it (select-all then delete) before typing.\n"
    "- When the task is complete, do not call the tool again: reply with a short "
    "final answer containing the information you gathered."
)

# xdotool-style key names (Claude) -> Playwright key names
_KEY_MAP = {
    "return": "Enter", "kp_enter": "Enter", "enter": "Enter",
    "tab": "Tab", "backspace": "Backspace", "delete": "Delete",
    "escape": "Escape", "esc": "Escape", "space": "Space",
    "page_up": "PageUp", "page_down": "PageDown", "prior": "PageUp", "next": "PageDown",
    "home": "Home", "end": "End", "insert": "Insert",
    "up": "ArrowUp", "down": "ArrowDown", "left": "ArrowLeft", "right": "ArrowRight",
    "ctrl": "Control", "control": "Control", "cmd": "Meta", "super": "Meta",
    "meta": "Meta", "win": "Meta", "alt": "Alt", "option": "Alt", "shift": "Shift",
}


def _convert_key(combo: str) -> str:
    """Translate an xdotool key spec ('ctrl+a', 'Page_Down') to Playwright form."""
    parts = combo.replace(" ", "").split("+")
    out = []
    for part in parts:
        lowered = part.lower()
        if lowered in _KEY_MAP:
            out.append(_KEY_MAP[lowered])
        elif len(part) == 1:
            out.append(part)
        else:
            out.append(part[:1].upper() + part[1:])
    return "+".join(out)


def _scroll_delta(direction, amount) -> tuple[int, int]:
    step = max(int(amount or 3), 1) * 120
    return {
        "down": (0, step), "up": (0, -step),
        "right": (step, 0), "left": (-step, 0),
    }.get(str(direction or "down").lower(), (0, step))


class ClaudeComputerAgent:
    """Prompt-free agent using Claude's native computer tool."""

    def __init__(self, settings: ModelSettings, run_settings: RunSettings,
                 viewport: tuple[int, int], api_log_path=None):
        self.settings = settings
        self.run_settings = run_settings
        self.viewport = viewport
        self._api_log_path = api_log_path
        base = settings.base_url.rstrip("/")
        # native Anthropic Messages API lives at /messages on the gateway base
        self._client = httpx.AsyncClient(
            base_url=base,
            timeout=httpx.Timeout(settings.request_timeout_s, connect=15.0),
            headers={
                "Authorization": f"Bearer {settings.api_key}",
                "x-api-key": settings.api_key,
                "anthropic-version": "2023-06-01",
                "anthropic-beta": BETA_HEADER,
                **settings.extra_headers,
            },
        )
        self.messages: list[dict] = []
        self._pending_tool_use_id: str | None = None
        self._display: tuple[int, int] = viewport

    # -- screenshot handling ---------------------------------------------

    def _encode_view(self, screenshot_png: bytes) -> tuple[str, tuple[int, int]]:
        image = load_png(screenshot_png)
        max_side = self.settings.image_max_side or DISPLAY_MAX_SIDE
        resized = fit_max_side(image, max_side)
        data = to_png_bytes(resized) if resized.size != image.size else screenshot_png
        return base64.b64encode(data).decode(), resized.size

    def _image_block(self, b64: str) -> dict:
        return {"type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": b64}}

    def _prune_old_images(self) -> None:
        """Keep only the most recent screenshots to bound context growth."""
        keep = max(1, self.settings.history_window)
        seen = 0
        for message in reversed(self.messages):
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                is_image = block.get("type") == "image" or (
                    block.get("type") == "tool_result"
                    and isinstance(block.get("content"), list)
                    and any(c.get("type") == "image" for c in block["content"])
                )
                if not is_image:
                    continue
                seen += 1
                if seen > keep:
                    if block.get("type") == "image":
                        block.clear()
                        block.update({"type": "text", "text": "[earlier screenshot omitted]"})
                    else:
                        block["content"] = [{"type": "text",
                                             "text": "[earlier screenshot omitted]"}]

    # -- action translation ----------------------------------------------

    def _pixel_target(self, coord, sent_size) -> Resolved:
        ctx = GroundingContext(_PIXEL_SCHEME, self.viewport, sent_size)
        return resolve_target({"point": list(coord)}, ctx)

    def _to_parsed_and_compiled(self, tool_input: dict, sent_size
                                ) -> tuple[ParsedAction, CompiledAction]:
        action = tool_input.get("action")
        args = dict(tool_input)

        def point(key="coordinate"):
            return self._pixel_target(tool_input[key], sent_size)

        if action in ("left_click", "left_mouse_down"):
            t = point()
            return (ParsedAction("click", args),
                    CompiledAction("click", [{"kind": "click", "x": t.x, "y": t.y,
                                              "button": "left", "count": 1}],
                                   point=(t.x, t.y)))
        if action == "right_click":
            t = point()
            return (ParsedAction("click", args),
                    CompiledAction("click", [{"kind": "click", "x": t.x, "y": t.y,
                                              "button": "right", "count": 1}],
                                   point=(t.x, t.y)))
        if action in ("double_click", "triple_click"):
            t = point()
            count = 2
            cmds = [{"kind": "click", "x": t.x, "y": t.y, "button": "left", "count": count}]
            if action == "triple_click":
                cmds.append({"kind": "click", "x": t.x, "y": t.y, "button": "left", "count": 1})
            return (ParsedAction("double_click", args),
                    CompiledAction("double_click", cmds, point=(t.x, t.y)))
        if action in ("mouse_move", "cursor_position"):
            t = point()
            return (ParsedAction("hover", args),
                    CompiledAction("hover", [{"kind": "hover", "x": t.x, "y": t.y}],
                                   point=(t.x, t.y)))
        if action == "left_click_drag":
            start = self._pixel_target(tool_input["start_coordinate"], sent_size)
            end = point()
            return (ParsedAction("drag", args),
                    CompiledAction("drag", [{"kind": "drag", "x1": start.x, "y1": start.y,
                                             "x2": end.x, "y2": end.y}],
                                   point=(start.x, start.y), drag_to=(end.x, end.y)))
        if action == "scroll":
            dx, dy = _scroll_delta(tool_input.get("scroll_direction"),
                                   tool_input.get("scroll_amount", 3))
            cmd = {"kind": "scroll", "dx": dx, "dy": dy}
            if "coordinate" in tool_input:
                t = point()
                cmd.update({"x": t.x, "y": t.y})
            return (ParsedAction("scroll", args), CompiledAction("scroll", [cmd]))
        if action == "type":
            text = tool_input.get("text", "")
            return (ParsedAction("type", args),
                    CompiledAction("type", [{"kind": "type", "text": text,
                                             "clear": False, "enter": False}]))
        if action in ("key", "hold_key"):
            combos = [_convert_key(k) for k in str(tool_input.get("text", "")).split()]
            combos = combos or ["Enter"]
            return (ParsedAction("hotkey", args),
                    CompiledAction("hotkey", [{"kind": "press", "keys": combos}]))
        if action == "wait":
            secs = min(max(float(tool_input.get("duration", 1)), 0.1), 8.0)
            return (ParsedAction("wait", args),
                    CompiledAction("wait", [{"kind": "wait", "ms": int(secs * 1000)}]))
        if action == "screenshot":
            # no page effect: re-observe on the next step
            return (ParsedAction("wait", args),
                    CompiledAction("wait", [{"kind": "wait", "ms": 200}]))
        raise LLMError(f"unsupported computer action: {action}")

    # -- API call --------------------------------------------------------

    async def _call(self) -> dict:
        display_w, display_h = self._display
        payload = {
            "model": self.settings.model,
            "max_tokens": self.settings.max_tokens,
            "system": SYSTEM_PROMPT,
            "tools": [{
                "type": TOOL_VERSION, "name": "computer",
                "display_width_px": display_w, "display_height_px": display_h,
            }],
            "messages": self.messages,
        }
        last_error = None
        for attempt in range(1, self.settings.max_retries + 1):
            try:
                response = await self._client.post("/messages", json=payload)
                if response.status_code in (408, 409, 425, 429, 500, 502, 503, 504):
                    last_error = LLMError(f"HTTP {response.status_code}: {response.text[:200]}")
                else:
                    response.raise_for_status()
                    return response.json()
            except httpx.HTTPError as err:
                last_error = err
            import asyncio
            await asyncio.sleep(min(2 ** attempt, 20))
        raise LLMError(f"anthropic call failed: {last_error}")

    async def _log(self, usage: dict, latency: float, ok: bool, error=None) -> None:
        if not self._api_log_path:
            return
        with open(self._api_log_path, "a", encoding="utf-8", errors="replace") as fh:
            fh.write(json.dumps({
                "ts": time.time(), "model": self.settings.model, "backend": "claude_cua",
                "ok": ok, "latency_s": round(latency, 3), "usage": usage, "error": error,
            }) + "\n")

    # -- decide ----------------------------------------------------------

    async def decide(self, task: Task, state: PageState, step_index: int,
                     max_steps: int, notices: list[str],
                     screenshot_png: bytes | None, validator=None) -> Decision:
        b64, sent_size = (self._encode_view(screenshot_png)
                          if screenshot_png else (None, self.viewport))
        self._display = sent_size

        # assemble this turn's user content
        text = (task_block(task) if step_index == 0
                else step_block(task, state, step_index, max_steps, notices,
                                vision_only=self.settings.vision_only))
        if step_index == 0 or self._pending_tool_use_id is None:
            content = [{"type": "text", "text": text}]
            if b64:
                content.append(self._image_block(b64))
            self.messages.append({"role": "user", "content": content})
        else:
            result_content = []
            if b64:
                result_content.append(self._image_block(b64))
            if notices:
                result_content.append({"type": "text",
                                       "text": "Notices: " + "; ".join(notices)})
            if not result_content:
                result_content.append({"type": "text", "text": "(screenshot unavailable)"})
            self.messages.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": self._pending_tool_use_id,
                "content": result_content,
            }]})

        self._prune_old_images()

        started = time.monotonic()
        try:
            body = await self._call()
        except LLMError as err:
            await self._log({}, time.monotonic() - started, ok=False, error=str(err))
            raise
        latency = time.monotonic() - started
        usage = body.get("usage") or {}
        await self._log(usage, latency, ok=True)

        blocks = body.get("content") or []
        self.messages.append({"role": "assistant", "content": blocks})

        analysis = " ".join(b.get("text", "") for b in blocks
                            if b.get("type") == "text").strip()
        tool_use = next((b for b in blocks if b.get("type") == "tool_use"), None)

        reply = ChatReply(
            text=json.dumps({"analysis": analysis,
                             "tool_use": tool_use.get("input") if tool_use else None},
                            ensure_ascii=False),
            usage={"prompt_tokens": usage.get("input_tokens", 0),
                   "completion_tokens": usage.get("output_tokens", 0),
                   "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)},
            latency_s=latency,
        )

        if tool_use is None:
            # end_turn: Claude is done, treat the final text as the answer
            self._pending_tool_use_id = None
            parsed = ParsedAction("stop", {"answer": analysis}, analysis=analysis)
            compiled = CompiledAction("stop", [], stop_answer=analysis or None)
            return Decision(parsed=parsed, reply=reply, parse_attempts=1,
                            compiled=compiled,
                            messages_meta={"backend": "claude_cua", "action": "stop"})

        self._pending_tool_use_id = tool_use.get("id")
        parsed, compiled = self._to_parsed_and_compiled(tool_use.get("input") or {}, sent_size)
        parsed.analysis = analysis
        return Decision(parsed=parsed, reply=reply, parse_attempts=1, compiled=compiled,
                        messages_meta={"backend": "claude_cua",
                                       "action": tool_use.get("input", {}).get("action")})

    def export_messages(self, strip_images: bool = True) -> list[dict]:
        if not strip_images:
            return self.messages
        cleaned = []
        for message in self.messages:
            content = message.get("content")
            if isinstance(content, list):
                new_content = []
                for block in content:
                    if block.get("type") == "image":
                        new_content.append({"type": "image", "source": "<png>"})
                    elif block.get("type") == "tool_result":
                        new_content.append({**block, "content": "<result>"})
                    else:
                        new_content.append(block)
                content = new_content
            cleaned.append({"role": message.get("role"), "content": content})
        return cleaned

    async def close(self) -> None:
        await self._client.aclose()
