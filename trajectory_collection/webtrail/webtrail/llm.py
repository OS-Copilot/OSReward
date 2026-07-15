"""Minimal async client for OpenAI-compatible chat endpoints.

Handles retry/backoff for transient failures and appends one JSONL record per
call when an api-log path is configured. A ``stub:`` base_url produces
scripted replies so the whole pipeline can be exercised without a real model.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .config import ModelSettings

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

# ceiling for the automatic max_tokens escalation used when a reasoning model
# burns its whole completion budget on hidden thinking
THINKING_MAX_TOKENS_CAP = 16_384


class LLMError(Exception):
    pass


@dataclass
class ChatReply:
    text: str
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0
    attempts: int = 1


class ChatModel:
    def __init__(self, settings: ModelSettings, api_log_path: str | Path | None = None):
        self.settings = settings
        self._api_log_path = Path(api_log_path) if api_log_path else None
        self._log_lock = asyncio.Lock()
        self._stub = settings.base_url.startswith("stub:")
        self._stub_script = [
            step for step in settings.base_url[len("stub:"):].split(",") if step
        ] or ["scroll", "stop"]
        self._client = None if self._stub else httpx.AsyncClient(
            base_url=settings.base_url.rstrip("/"),
            timeout=httpx.Timeout(settings.request_timeout_s, connect=15.0),
            headers={
                "Authorization": f"Bearer {settings.api_key}",
                **settings.extra_headers,
            },
        )

    async def complete(self, messages: list[dict]) -> ChatReply:
        if self._stub or self._client is None:
            return self._stub_reply(messages)

        max_tokens = self.settings.max_tokens
        last_error: Exception | None = None
        started = time.monotonic()
        for attempt in range(1, self.settings.max_retries + 1):
            payload = {
                "model": self.settings.model,
                "messages": messages,
                "temperature": self.settings.temperature,
                "top_p": self.settings.top_p,
                "max_tokens": max_tokens,
            }
            try:
                response = await self._client.post("/chat/completions", json=payload)
                if response.status_code in RETRYABLE_STATUS:
                    last_error = LLMError(
                        f"HTTP {response.status_code}: {response.text[:300]}"
                    )
                else:
                    response.raise_for_status()
                    body = response.json()
                    text = body["choices"][0]["message"]["content"] or ""
                    usage = body.get("usage") or {}

                    # reasoning models can spend the whole completion budget on
                    # hidden thinking and return empty content; give the next
                    # attempt a doubled budget instead of failing the call
                    completion_tokens = usage.get("completion_tokens") or 0
                    overflowed = (
                        not text.strip()
                        and completion_tokens >= 0.8 * max_tokens
                        and max_tokens < THINKING_MAX_TOKENS_CAP
                    )
                    if overflowed:
                        max_tokens = min(max_tokens * 2, THINKING_MAX_TOKENS_CAP)
                        last_error = LLMError(
                            f"empty content after {completion_tokens} completion "
                            f"tokens (reasoning overflow); retrying with "
                            f"max_tokens={max_tokens}"
                        )
                        continue

                    reply = ChatReply(
                        text=text,
                        usage=usage,
                        latency_s=time.monotonic() - started,
                        attempts=attempt,
                    )
                    await self._log(reply, error=None)
                    return reply
            except httpx.HTTPStatusError as err:
                await self._log(None, error=f"{err}: {err.response.text[:300]}")
                raise LLMError(
                    f"model call failed (HTTP {err.response.status_code}): "
                    f"{err.response.text[:300]}"
                ) from err
            except (httpx.HTTPError, KeyError, ValueError) as err:
                last_error = err
            await asyncio.sleep(min(2 ** attempt, 30) * (0.5 + random.random()))

        await self._log(None, error=str(last_error))
        raise LLMError(f"model call failed after {self.settings.max_retries} "
                       f"attempts: {last_error}")

    def _stub_reply(self, messages: list[dict]) -> ChatReply:
        turn = sum(1 for m in messages if m.get("role") == "assistant")
        step = self._stub_script[min(turn, len(self._stub_script) - 1)]
        canned = {
            "scroll": {"action": "scroll", "args": {"dy": 400}},
            "click": {"action": "click", "args": {"box2d": [450, 450, 550, 550]}},
            "wait": {"action": "wait", "args": {"seconds": 1}},
            "stop": {"action": "stop", "args": {"answer": "stub run complete"}},
        }.get(step, {"action": "stop", "args": {"answer": "stub run complete"}})
        text = (
            f"Stub reply for scripted step '{step}'.\n\n"
            "```json\n" + json.dumps(canned) + "\n```"
        )
        return ChatReply(text=text, usage={}, latency_s=0.0)

    async def _log(self, reply: ChatReply | None, error: str | None) -> None:
        if self._api_log_path is None:
            return
        record = {
            "ts": time.time(),
            "model": self.settings.model,
            "ok": error is None,
            "latency_s": round(reply.latency_s, 3) if reply else None,
            "attempts": reply.attempts if reply else self.settings.max_retries,
            "usage": reply.usage if reply else None,
            "error": error,
        }
        async with self._log_lock:
            with self._api_log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
