"""The browsing agent: context management, model calls, response parsing.

`WebAgent` is instantiated once per episode. It keeps the turn history, builds
messages according to the configured history mode, calls the model, and parses
the reply into a `ParsedAction`. Malformed replies trigger a corrective
re-ask (bounded by ``run.parse_retries``).

History modes
-------------
* ``windowed``  – the last N steps are kept verbatim (screenshot + reply);
                  older steps collapse into one-line summaries.
* ``text_full`` – every past analysis/action is kept as plain text in a single
                  context message, and only the newest screenshot is attached.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field

from . import prompts
from .config import ModelSettings, RunSettings
from .grounding import GroundingContext
from .llm import ChatModel, ChatReply
from .types import PageState, ParsedAction, Task


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

    def _windowed_messages(self, current: Turn) -> list[dict]:
        window = max(1, self.model_settings.history_window)
        older = self.turns[:-(window - 1)] if window > 1 else self.turns
        recent = self.turns[len(older):]

        messages: list[dict] = [{"role": "system", "content": self.system}]
        if older:
            summary = "\n".join(
                prompts.history_line(t.step_index, t.action_key, t.action_args, t.url)
                for t in older
            )
            messages.append({
                "role": "user",
                "content": "Summary of earlier steps (screenshots omitted):\n" + summary,
            })
            messages.append({
                "role": "assistant",
                "content": "Understood. Continuing from there.",
            })
        for turn in recent:
            messages.append(self._turn_user_message(turn))
            messages.append({"role": "assistant", "content": turn.reply_text})
        messages.append(self._turn_user_message(current))
        return messages

    def _text_full_messages(self, current: Turn) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": self.system}]
        if self.turns:
            blocks = []
            for turn in self.turns:
                blocks.append(f"### Step {turn.step_index + 1} — {turn.url}\n"
                              f"{turn.reply_text.strip()}")
            messages.append({
                "role": "user",
                "content": ("Full history of your previous analyses and actions "
                            "(oldest first). Only the newest screenshot is "
                            "attached to the next message.\n\n" + "\n\n".join(blocks)),
            })
            messages.append({"role": "assistant",
                             "content": "Understood. Continuing from there."})
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
