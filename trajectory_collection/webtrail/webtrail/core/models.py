"""Shared value objects passed between pipeline stages."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from urllib.parse import urlparse


def domain_of(url: str) -> str:
    """Registrable-ish domain used for pacing keys ('scholar.google.com' -> 'google.com')."""
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    parts = host.split(".")
    if len(parts) > 2 and parts[-2] in {"co", "com", "org", "net", "ac", "gov", "edu"}:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def normalize_url(url: str) -> str:
    url = url.strip()
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url
    return url


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str, limit: int = 40) -> str:
    return _SLUG_RE.sub("-", text.lower()).strip("-")[:limit] or "task"


@dataclass
class Task:
    """One collection unit: a starting URL (or several) plus an instruction."""

    task_id: str
    instruction: str
    urls: list[str]
    steps: list[str] = field(default_factory=list)
    criteria: list[str] = field(default_factory=list)
    max_steps: int | None = None
    action_profile: str | None = None
    extras: dict = field(default_factory=dict)

    @property
    def start_url(self) -> str:
        return self.urls[0]

    @property
    def domain(self) -> str:
        return domain_of(self.start_url)

    @classmethod
    def from_record(cls, record: dict, index: int) -> Task:
        raw_urls = record.get("urls") or record.get("url") or record.get("website") or ""
        if isinstance(raw_urls, str):
            raw_urls = raw_urls.split()
        urls = [normalize_url(u) for u in raw_urls if u]
        if not urls:
            raise ValueError(f"task record {index} has no url")

        instruction = (record.get("instruction") or record.get("task") or "").strip()
        if not instruction:
            raise ValueError(f"task record {index} has no instruction")

        task_id = record.get("id") or record.get("task_id")
        if not task_id:
            digest = hashlib.sha1(
                (instruction + "|" + " ".join(urls)).encode()
            ).hexdigest()[:10]
            task_id = f"t{index:05d}.{slugify(domain_of(urls[0]))}.{digest}"

        steps = record.get("steps") or []
        criteria = record.get("criteria") or []
        if isinstance(steps, str):
            steps = [s for s in steps.splitlines() if s.strip()]
        if isinstance(criteria, str):
            criteria = [c for c in criteria.splitlines() if c.strip()]

        known = {"id", "task_id", "url", "urls", "website", "instruction", "task",
                 "steps", "criteria", "max_steps", "action_profile"}
        return cls(
            task_id=str(task_id),
            instruction=instruction,
            urls=urls,
            steps=list(steps),
            criteria=list(criteria),
            max_steps=record.get("max_steps"),
            action_profile=record.get("action_profile"),
            extras={k: v for k, v in record.items() if k not in known},
        )


@dataclass
class PageState:
    """One observed page: everything the agent and the recorder need."""

    url: str
    title: str | None
    html: str | None
    screenshot_png: bytes | None
    elements: list[dict] | None
    axtree: dict | None
    scroll: dict | None
    viewport: tuple[int, int]
    http_status: int | None = None
    errors: list[str] = field(default_factory=list)
    snapshot_meta: dict = field(default_factory=dict)

    @property
    def text_fingerprint(self) -> str:
        """Cheap page identity used for stale-state detection (screenshot hash is added separately)."""
        return f"{self.url}|{len(self.html or '')//512}"


@dataclass
class Verdict:
    """Block-detector output for one page state."""

    kind: str | None = None       # captcha | challenge | access_denied | rate_limit |
                                  # login_wall | geo_blocked | not_found | server_error |
                                  # network_error
    scope: str = "target"         # target | search
    evidence: str = ""

    @property
    def blocked(self) -> bool:
        return self.kind is not None


@dataclass
class ParsedAction:
    """Validated model action before grounding resolution."""

    key: str
    args: dict
    analysis: str = ""
    raw_block: str = ""


@dataclass
class CompiledAction:
    """A parsed action resolved to concrete service commands."""

    key: str
    commands: list[dict]                 # payloads for POST /act, in order
    point: tuple[int, int] | None = None
    box_px: tuple[int, int, int, int] | None = None
    drag_to: tuple[int, int] | None = None
    goto_url: str | None = None
    stop_answer: str | None = None

    @property
    def is_stop(self) -> bool:
        return self.key == "stop"
