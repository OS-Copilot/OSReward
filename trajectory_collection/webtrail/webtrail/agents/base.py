"""The browsing agent: context management, model calls, response parsing.

`WebAgent` is instantiated once per episode. It keeps a semantic turn history,
builds messages according to the configured history mode, calls the model, and
parses the reply into a `ParsedAction`. Malformed replies trigger a corrective
re-ask (bounded by ``run.parse_retries``).

History modes
-------------
* ``windowed``  – the last N steps keep the model reply and browser action
                  result; older steps collapse into one-line summaries.
* ``text_full`` – every past analysis/action is kept as plain text in a single
                  context message.

Both modes attach exactly one browser image: the current screenshot.  DOM,
HTML, accessibility trees, and element IDs are never serialized here.
"""

from __future__ import annotations

import base64
import copy
import json
import re
from dataclasses import dataclass, field

from ..browser.grounding import GroundingContext
from ..core.config import ModelSettings, RunSettings
from ..core.models import PageState, ParsedAction, Task
from . import prompts
from .llm import ChatModel, ChatReply


class AgentFormatError(Exception):
    """The model kept producing unparseable/invalid actions."""


_FENCED_JSON = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)


def extract_action(text: str) -> ParsedAction:
    """Pull the action JSON out of a model reply.

    Preference order: last fenced block that parses as JSON, then the last
    balanced ``{...}`` region in the raw text. Raises ValueError with a
    model-facing message when nothing parses.
    """
    candidates = [m.group(1).strip() for m in _FENCED_JSON.finditer(text)]

    # last-resort: balanced brace scan over the raw text
    if not candidates:
        depth, start = 0, -1
        for idx, char in enumerate(text):
            if char == "{":
                if depth == 0:
                    start = idx
                depth += 1
            elif char == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start >= 0:
                        candidates.append(text[start:idx + 1])
        if not candidates:
            raise ValueError("no JSON action found in the reply")

    last_error = "not valid JSON"
    for candidate in reversed(candidates):
        cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)  # drop trailing commas
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as err:
            last_error = f"JSON parse error: {err}"
            continue
        if not isinstance(data, dict):
            last_error = "the JSON block must be an object"
            continue
        key = data.get("action")
        if not isinstance(key, str) or not key:
            last_error = 'the JSON object must contain "action": "<name>"'
            continue
        args = data.get("args")
        if args is None:
            args = {k: v for k, v in data.items() if k != "action"}
        if not isinstance(args, dict):
            last_error = '"args" must be an object'
            continue
        analysis = text[: text.find("```")].strip() if "```" in text else ""
        return ParsedAction(key=key.strip(), args=args, analysis=analysis,
                            raw_block=candidate)
    raise ValueError(last_error)


@dataclass
class Turn:
    step_index: int
    url: str
    user_text: str
    image_data_url: str | None
    reply_text: str = ""
    action_key: str | None = None
    action_args: dict | None = None
    action_results: list[dict] = field(default_factory=list)
    post_url: str | None = None


@dataclass
class Decision:
    parsed: ParsedAction
    reply: ChatReply
    parse_attempts: int
    compiled: object | None = None      # validator's product (a CompiledAction)
    messages_meta: dict = field(default_factory=dict)


class WebAgent:
    def __init__(self, model: ChatModel, model_settings: ModelSettings,
                 run_settings: RunSettings, ctx: GroundingContext, profile: str):
        self.model = model
        self.model_settings = model_settings
        self.run_settings = run_settings
        self.ctx = ctx
        self.profile = profile
        self.system = prompts.system_prompt(
            profile, ctx, model_settings.analysis_words
        )
        self.turns: list[Turn] = []

    # -- message building -----------------------------------------------

    @staticmethod
    def _image_part(data_url: str) -> dict:
        return {"type": "image_url", "image_url": {"url": data_url}}

    @staticmethod
    def _safe_result(result: dict) -> str:
        """Render one browser result without leaking internal page structures."""
        if result.get("ok"):
            status = "success"
        else:
            status = "failure"

        details = []
        if result.get("error"):
            details.append(f"error={str(result['error'])[:300]}")
        if result.get("final_url"):
            details.append(f"final_url={result['final_url']}")
        if result.get("selected_label") is not None:
            details.append(f"selected_label={result['selected_label']!r}")
        if result.get("selected_value") is not None:
            details.append(f"selected_value={result['selected_value']!r}")
        if result.get("actual_value") is not None:
            details.append(f"actual_value={result['actual_value']!r}")
        if result.get("value_matches") is not None:
            details.append(f"value_matches={bool(result['value_matches'])}")
        if result.get("available_options"):
            options = [str(option)[:100] for option in result["available_options"][:30]]
            details.append(f"available_options={options!r}")
        rendered = "; ".join(details)
        return f"{status}: {rendered}" if rendered else status

    @classmethod
    def _turn_history(cls, turn: Turn, *, detailed: bool) -> str:
        header = f"### Step {turn.step_index + 1} @ {turn.url}"
        if detailed:
            action = turn.reply_text.strip() or "(no model reply recorded)"
        else:
            args = {k: v for k, v in (turn.action_args or {}).items()
                    if k != "analysis"}
            action = (f"Action: {turn.action_key} {args}" if turn.action_key
                      else "Action: (none)")

        if turn.action_results:
            results = "\n".join(
                f"Action Result {index + 1}: {cls._safe_result(result)}"
                for index, result in enumerate(turn.action_results)
            )
        elif turn.action_key == "stop":
            results = "Action Result: no browser command required"
        else:
            results = "Action Result: unavailable"
        post_url = f"\nPost-action URL: {turn.post_url}" if turn.post_url else ""
        return f"{header}\n{action}\n{results}{post_url}"

    def _history_message(self, turns: list[Turn], *, detailed_from: int) -> dict:
        blocks = [
            self._turn_history(turn, detailed=index >= detailed_from)
            for index, turn in enumerate(turns)
        ]
        return {
            "role": "user",
            "content": (
                "Agent history (oldest first). Action Result reports whether the "
                "browser command executed; use the current screenshot as ground truth "
                "for whether the task actually progressed.\n\n" + "\n\n".join(blocks)
            ),
        }

    def _windowed_messages(self, current: Turn) -> list[dict]:
        window = max(1, self.model_settings.history_window)
        messages: list[dict] = [{"role": "system", "content": self.system}]
        if self.turns:
            detailed_from = max(0, len(self.turns) - window)
            messages.append(self._history_message(self.turns, detailed_from=detailed_from))
            messages.append({"role": "assistant",
                             "content": "Understood. I will use the current screenshot as ground truth."})
        messages.append(self._turn_user_message(current))
        return messages

    def _text_full_messages(self, current: Turn) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": self.system}]
        if self.turns:
            messages.append(self._history_message(self.turns, detailed_from=0))
            messages.append({"role": "assistant",
                             "content": "Understood. I will use the current screenshot as ground truth."})
        messages.append(self._turn_user_message(current))
        return messages

    def _turn_user_message(self, turn: Turn) -> dict:
        if turn.image_data_url is None:
            return {"role": "user",
                    "content": turn.user_text + "\n\n(screenshot unavailable this step)"}
        return {"role": "user", "content": [
            {"type": "text", "text": turn.user_text},
            self._image_part(turn.image_data_url),
        ]}

    # -- deciding ---------------------------------------------------------

    async def decide(self, task: Task, state: PageState, step_index: int,
                     max_steps: int, notices: list[str],
                     screenshot_png: bytes | None,
                     validator=None) -> Decision:
        """Ask the model for the next action.

        `validator(parsed)` may raise ValueError to reject an action that
        parsed but cannot be executed (unknown key, unresolvable target); the
        rejection message is fed back to the model as a corrective turn.
        """
        image_url = None
        if screenshot_png is not None:
            encoded = base64.b64encode(screenshot_png).decode()
            image_url = f"data:image/png;base64,{encoded}"

        current = Turn(
            step_index=step_index,
            url=state.url,
            user_text=prompts.step_block(task, state, step_index, max_steps, notices,
                                         vision_only=self.model_settings.vision_only),
            image_data_url=image_url,
        )

        build = (self._text_full_messages
                 if self.model_settings.history_mode == "text_full"
                 else self._windowed_messages)

        messages = build(current)
        parse_error: str | None = None
        reply: ChatReply | None = None

        for attempt in range(1, self.run_settings.parse_retries + 2):
            if parse_error is not None and reply is not None:
                messages = messages + [
                    {"role": "assistant", "content": reply.text},
                    {"role": "user",
                     "content": prompts.CORRECTIVE_TEMPLATE.format(reason=parse_error)},
                ]
            reply = await self.model.complete(messages)
            try:
                parsed = extract_action(reply.text)
                compiled = validator(parsed) if validator is not None else None
            except ValueError as err:
                parse_error = str(err)
                continue

            current.reply_text = reply.text
            current.action_key = parsed.key
            current.action_args = parsed.args
            self.turns.append(current)
            return Decision(
                parsed=parsed,
                reply=reply,
                parse_attempts=attempt,
                compiled=compiled,
                messages_meta={
                    "num_messages": len(messages),
                    "num_images": sum(
                        1 for m in messages if isinstance(m.get("content"), list)
                    ),
                    "history_mode": self.model_settings.history_mode,
                },
            )

        raise AgentFormatError(f"unparseable model output after retries: {parse_error}")

    def record_action_result(self, step_index: int, results: list[dict],
                             post_url: str | None = None) -> None:
        """Attach browser-use-style action results to the completed model turn."""
        if not self.turns or self.turns[-1].step_index != step_index:
            raise RuntimeError(f"cannot attach action result for missing step {step_index}")
        turn = self.turns[-1]
        turn.action_results = copy.deepcopy(results)
        turn.post_url = post_url

    def record_notice_only(self) -> None:
        """Forget the pending turn (used when the episode aborts before acting)."""

    def export_messages(self, strip_images: bool = True) -> list[dict]:
        """Debug dump of what the next prompt would look like (images removed)."""
        fake = Turn(step_index=len(self.turns), url="(next)", user_text="(pending)",
                    image_data_url=None)
        messages = (self._text_full_messages(fake)
                    if self.model_settings.history_mode == "text_full"
                    else self._windowed_messages(fake))
        if not strip_images:
            return messages
        cleaned = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                content = [part if part.get("type") != "image_url"
                           else {"type": "image_url", "image_url": {"url": "<png>"}}
                           for part in content]
            cleaned.append({**message, "content": content})
        return cleaned
