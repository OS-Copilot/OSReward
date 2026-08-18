"""Minimal model transport for the official OpenAI and Anthropic APIs.

Only standard environment variables are used for credentials:

* ``OPENAI_API_KEY`` for OpenAI Responses
* ``ANTHROPIC_API_KEY`` for Anthropic Messages

The transport returns only assistant text plus aggregate usage and latency;
provider-private metadata and full HTTP exchanges are never captured.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..core.config import ModelSettings

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class LLMError(Exception):
    """A model request failed after bounded retries."""


@dataclass
class ChatReply:
    text: str
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0


def resolve_provider(configured: str, model: str) -> str:
    """Resolve ``auto`` without contacting a provider."""
    provider = (configured or "auto").strip().lower()
    if provider == "auto":
        return "anthropic" if "claude" in model.lower() else "openai"
    if provider not in {"openai", "anthropic"}:
        raise ValueError(f"unknown model provider: {configured}")
    return provider


def credential_env(provider: str) -> str:
    return "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"


def _api_url(base_url: str, endpoint: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        return f"{root}/{endpoint}"
    return f"{root}/v1/{endpoint}"


class ChatModel:
    """Provider-neutral text/vision completion interface used by WebTrail."""

    def __init__(self, settings: ModelSettings):
        self.settings = settings
        self.provider = resolve_provider(settings.provider, settings.model)
        self._stub = settings.base_url.startswith("stub:")
        self._stub_script = [
            step for step in settings.base_url.removeprefix("stub:").split(",") if step
        ] or ["stop"]

        if self._stub:
            self._client = None
            self._url = ""
            return

        env_name = credential_env(self.provider)
        api_key = os.environ.get(env_name, "").strip()
        if not api_key:
            raise ValueError(f"{env_name} is required for provider={self.provider}")

        if self.provider == "anthropic":
            base = (
                settings.base_url
                or os.environ.get("ANTHROPIC_BASE_URL")
                or "https://api.anthropic.com"
            )
            self._url = _api_url(base, "messages")
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
        else:
            base = (
                settings.base_url
                or os.environ.get("OPENAI_BASE_URL")
                or "https://api.openai.com"
            )
            self._url = _api_url(base, "responses")
            headers = {
                "Authorization": f"Bearer {api_key}",
                "content-type": "application/json",
            }

        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=settings.request_timeout_s,
            # Never carry credential headers through an endpoint redirect.
            follow_redirects=False,
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    async def complete(
        self,
        messages: list[dict],
        *,
        max_tokens: int | None = None,
    ) -> ChatReply:
        """Return one assistant response."""
        if self._stub:
            return self._stub_reply(messages)

        payload = self._payload(messages, max_tokens=max_tokens)
        started = time.monotonic()
        last_error: Exception | None = None
        attempts = max(1, self.settings.max_retries)

        for attempt in range(1, attempts + 1):
            try:
                assert self._client is not None
                response = await self._client.post(self._url, json=payload)
                if response.status_code in RETRYABLE_STATUS and attempt < attempts:
                    await self._backoff(attempt, response)
                    continue
                response.raise_for_status()
                body = response.json()
                reply = self._parse(body)
                reply.latency_s = time.monotonic() - started
                return reply
            except (httpx.HTTPError, ValueError, KeyError, IndexError) as exc:
                last_error = exc
                if attempt >= attempts or (
                    isinstance(exc, httpx.HTTPStatusError)
                    and exc.response.status_code not in RETRYABLE_STATUS
                ):
                    break
                await self._backoff(attempt)

        raise LLMError(
            f"{self.provider} request failed after {attempts} attempt(s): {last_error}"
        )

    def _payload(self, messages: list[dict], *, max_tokens: int | None) -> dict:
        limit = max_tokens or self.settings.max_tokens
        if self.provider == "anthropic":
            system, converted = self._anthropic_messages(messages)
            payload: dict[str, Any] = {
                "model": self.settings.model,
                "messages": converted,
                "max_tokens": limit,
            }
            if system:
                payload["system"] = system
            if self.settings.temperature is not None:
                payload["temperature"] = self.settings.temperature
            return payload
        payload = {
            "model": self.settings.model,
            "input": self._openai_input(messages),
            "max_output_tokens": limit,
            "store": False,
        }
        if self.settings.temperature is not None:
            payload["temperature"] = self.settings.temperature
        return payload

    def _parse(self, body: dict) -> ChatReply:
        if self.provider == "anthropic":
            blocks = body.get("content") or []
            text = "\n".join(
                str(block.get("text") or "")
                for block in blocks
                if block.get("type") == "text"
            ).strip()
            usage = body.get("usage") or {}
            input_tokens = int(usage.get("input_tokens") or 0)
            output_tokens = int(usage.get("output_tokens") or 0)
            return ChatReply(
                text=text,
                usage={
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
                latency_s=0.0,
            )

        text_parts: list[str] = []
        for item in body.get("output") or []:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for part in item.get("content") or []:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "output_text" and part.get("text"):
                    text_parts.append(str(part["text"]))
                elif part.get("type") == "refusal" and part.get("refusal"):
                    text_parts.append(str(part["refusal"]))
        content = "\n".join(text_parts).strip()
        if not content and isinstance(body.get("output_text"), str):
            content = body["output_text"].strip()
        usage = body.get("usage") or {}
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        return ChatReply(
            text=str(content),
            usage={
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": int(
                    usage.get("total_tokens") or input_tokens + output_tokens
                ),
            },
            latency_s=0.0,
        )

    @staticmethod
    def _openai_input(messages: list[dict]) -> list[dict]:
        """Convert Chat-style vision messages to Responses API input items."""
        converted: list[dict] = []
        for message in messages:
            role = str(message.get("role") or "user")
            if role not in {"user", "assistant", "system", "developer"}:
                role = "user"
            content = message.get("content") or ""
            if isinstance(content, str):
                if content:
                    converted.append({"role": role, "content": content})
                continue
            if not isinstance(content, list):
                continue
            parts: list[dict] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    parts.append({
                        "type": "input_text",
                        "text": str(part.get("text") or ""),
                    })
                    continue
                if part.get("type") != "image_url":
                    continue
                image_url = part.get("image_url")
                url = image_url.get("url") if isinstance(image_url, dict) else image_url
                if isinstance(url, str):
                    parts.append({
                        "type": "input_image",
                        "image_url": url,
                        "detail": "auto",
                    })
            if parts:
                converted.append({"role": role, "content": parts})
        return converted

    @classmethod
    def _anthropic_messages(cls, messages: list[dict]) -> tuple[str, list[dict]]:
        system_parts: list[str] = []
        converted: list[dict] = []
        for message in messages:
            role = str(message.get("role") or "user")
            content = message.get("content") or ""
            if role == "system":
                system_parts.extend(cls._text_parts(content))
                continue
            role = "assistant" if role == "assistant" else "user"
            blocks = cls._anthropic_content(content)
            if not blocks:
                continue
            if converted and converted[-1]["role"] == role:
                converted[-1]["content"].extend(blocks)
            else:
                converted.append({"role": role, "content": blocks})
        return "\n\n".join(system_parts), converted

    @staticmethod
    def _text_parts(content: Any) -> list[str]:
        if isinstance(content, str):
            return [content] if content else []
        if not isinstance(content, list):
            return []
        return [
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]

    @classmethod
    def _anthropic_content(cls, content: Any) -> list[dict]:
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        if not isinstance(content, list):
            return []

        blocks: list[dict] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                blocks.append({"type": "text", "text": str(part.get("text") or "")})
                continue
            if part.get("type") != "image_url":
                continue
            image_url = part.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else image_url
            if not isinstance(url, str):
                continue
            if url.startswith("data:") and ";base64," in url:
                metadata, data = url.split(",", 1)
                media_type = metadata[5:].split(";", 1)[0]
                blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": data,
                    },
                })
            else:
                blocks.append({
                    "type": "image",
                    "source": {"type": "url", "url": url},
                })
        return blocks

    def _stub_reply(self, messages: list[dict]) -> ChatReply:
        turn = sum(1 for message in messages if message.get("role") == "assistant")
        step = self._stub_script[min(turn, len(self._stub_script) - 1)]
        canned = {
            "scroll": {"action": "scroll", "args": {"dy": 400}},
            "click": {"action": "click", "args": {"point": [500, 500]}},
            "wait": {"action": "wait", "args": {"seconds": 1}},
            "stop": {"action": "stop", "args": {"answer": "stub run complete"}},
        }.get(step, {"action": "stop", "args": {"answer": "stub run complete"}})
        text = (
            f"Stub reply for scripted step '{step}'.\n\n```json\n"
            + json.dumps(canned)
            + "\n```"
        )
        return ChatReply(text=text, usage={}, latency_s=0.0)

    @staticmethod
    async def _backoff(attempt: int, response: httpx.Response | None = None) -> None:
        retry_after = response.headers.get("retry-after") if response is not None else None
        try:
            delay = float(retry_after) if retry_after else 0.0
        except ValueError:
            delay = 0.0
        if delay <= 0:
            delay = min(20.0, (2 ** (attempt - 1)) + random.random())
        import asyncio

        await asyncio.sleep(delay)
