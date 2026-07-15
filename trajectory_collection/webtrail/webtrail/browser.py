"""Async client for the browser service.

`ServicePool` spreads sessions across the service worker ports; `BrowserSession`
wraps one live session and converts HTTP-level failures into `BrowserGone` /
`BrowserError` so the runner can tell "session died" apart from "action failed".
"""

from __future__ import annotations

import base64
import itertools
import logging

import httpx

from .config import BrowserSettings
from .types import PageState

logger = logging.getLogger(__name__)


class BrowserError(Exception):
    """The service answered, but the operation failed (recoverable per-step)."""


class BrowserGone(Exception):
    """The session or service is no longer reachable (fatal for the episode)."""


class ServicePool:
    """Round-robins new sessions across browser-service workers."""

    def __init__(self, settings: BrowserSettings):
        self.settings = settings
        self._hosts = itertools.cycle(settings.service_hosts)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.request_timeout_s, connect=10.0),
            limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
        )

    async def open_session(self) -> "BrowserSession":
        host = next(self._hosts)
        payload = {
            "width": self.settings.viewport_width,
            "height": self.settings.viewport_height,
            "isolation": self.settings.isolation,
            "locale": self.settings.locale,
            "timezone": self.settings.timezone,
            "userAgent": self.settings.user_agent,
            "proxy": self.settings.proxy,
            "navTimeoutMs": self.settings.nav_timeout_ms,
        }
        try:
            response = await self._client.post(f"{host}/session", json=payload)
        except httpx.HTTPError as err:
            raise BrowserGone(f"cannot reach browser service {host}: {err}") from err
        data = _expect_json(response)
        if not data.get("ok"):
            raise BrowserGone(f"session start failed: {data.get('error')}")
        return BrowserSession(self._client, host, data["session_id"], self.settings)

    async def health(self) -> list[dict]:
        results = []
        for host in self.settings.service_hosts:
            try:
                response = await self._client.get(f"{host}/healthz")
                results.append({"host": host, **response.json()})
            except httpx.HTTPError as err:
                results.append({"host": host, "ok": False, "error": str(err)})
        return results

    async def close(self) -> None:
        await self._client.aclose()


class BrowserSession:
    def __init__(self, client: httpx.AsyncClient, host: str, session_id: str,
                 settings: BrowserSettings):
        self._client = client
        self._host = host
        self._id = session_id
        self._settings = settings
        self.viewport = (settings.viewport_width, settings.viewport_height)

    async def _post(self, path: str, payload: dict | None = None) -> dict:
        url = f"{self._host}/session/{self._id}{path}"
        try:
            response = await self._client.post(url, json=payload or {})
        except httpx.HTTPError as err:
            raise BrowserGone(f"browser service unreachable: {err!r}") from err
        if response.status_code == 410:
            raise BrowserGone("session expired on the service")
        return _expect_json(response)

    async def goto(self, url: str, timeout_ms: int | None = None) -> dict:
        return await self._post("/goto", {
            "url": url,
            "timeoutMs": timeout_ms or self._settings.nav_timeout_ms,
        })

    async def snapshot(self, *, lite: bool = False) -> PageState:
        """Fetch one observation. `lite` skips html/axtree/elements (preflight checks)."""
        payload = {
            "settleMs": self._settings.settle_ms,
            "netIdleMs": self._settings.net_idle_ms,
            "htmlMaxBytes": self._settings.html_max_bytes,
        }
        if lite:
            payload.update({"axtree": False, "elements": False})
        data = await self._post("/snapshot", payload)
        if not data.get("ok"):
            raise BrowserError(f"snapshot failed: {data.get('error')}")

        errors = [
            str(data[key]) for key in
            ("screenshot_error", "collect_error", "axtree_error")
            if data.get(key)
        ]
        screenshot = base64.b64decode(data["screenshot"]) if data.get("screenshot") else None
        viewport = data.get("viewport") or {}
        return PageState(
            url=data.get("url") or "",
            title=data.get("title"),
            html=data.get("html"),
            screenshot_png=screenshot,
            elements=data.get("elements"),
            axtree=data.get("axtree"),
            scroll=data.get("scroll"),
            viewport=(
                viewport.get("width", self.viewport[0]),
                viewport.get("height", self.viewport[1]),
            ),
            errors=errors,
        )

    async def act(self, command: dict) -> dict:
        """Execute one typed action; returns the raw service response."""
        return await self._post("/act", command)

    async def close(self) -> None:
        url = f"{self._host}/session/{self._id}"
        try:
            await self._client.delete(url)
        except httpx.HTTPError:
            logger.debug("close for session %s failed; reaper will collect it", self._id[:8])


def _expect_json(response: httpx.Response) -> dict:
    try:
        return response.json()
    except ValueError as err:
        raise BrowserGone(
            f"non-JSON reply (HTTP {response.status_code}): {response.text[:200]}"
        ) from err
