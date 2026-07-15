"""Run configuration.

Everything the collector needs is grouped into four dataclasses and bundled
into :class:`Config`. A config can come from a JSON file, CLI flags, or both
(CLI wins). Keep defaults here conservative and production-ish.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BrowserSettings:
    """How to reach the browser service and shape each session."""

    service_hosts: list[str] = field(
        default_factory=lambda: ["http://127.0.0.1:9300"]
    )
    viewport_width: int = 1920
    viewport_height: int = 1080
    isolation: str = "browser"          # "browser" | "context"
    locale: str = "en-US"
    timezone: str | None = None
    user_agent: str | None = None
    proxy: dict | None = None           # {"server": ..., "username": ..., "password": ...}

    nav_timeout_ms: int = 45_000
    settle_ms: int = 800                # pause before observing, for async renders
    net_idle_ms: int = 2_500            # best-effort networkidle wait cap
    snapshot_attempts: int = 4          # retries when screenshot missing / page unstable
    snapshot_retry_wait_s: float = 1.5

    # non-interactive JS challenges (Cloudflare "Just a moment", etc.) often
    # clear on their own once the browser check finishes; wait and re-observe
    # before giving up. Interactive CAPTCHAs are never waited on.
    challenge_wait_attempts: int = 4
    challenge_wait_s: float = 5.0
    html_max_bytes: int = 3_000_000
    request_timeout_s: float = 90.0


@dataclass
class ModelSettings:
    """OpenAI-compatible chat endpoint plus how to present/ground actions."""

    model: str = ""
    base_url: str = ""                  # e.g. https://host/v1 ; "stub:" enables scripted replies
    api_key: str = ""
    temperature: float = 0.4
    max_tokens: int = 2048
    top_p: float = 1.0
    request_timeout_s: float = 180.0
    max_retries: int = 4
    extra_headers: dict = field(default_factory=dict)

    backend: str = "prompt"             # prompt | claude_cua (native computer tool)
    grounding: str = "auto"             # auto | box1000 | point1000 | pixel
    history_mode: str = "windowed"      # windowed | text_full
    history_window: int = 4             # screenshots kept in context (windowed mode)
    vision_only: bool = False           # drop URL/title text; screenshot is the only page signal
    image_max_side: int = 0             # 0 = send native resolution
    analysis_words: int = 150           # requested reasoning length in the prompt


@dataclass
class PacingSettings:
    """Global and per-domain concurrency plus cooldown behaviour."""

    max_concurrency: int = 8
    per_domain: int = 1
    domain_gap_s: tuple[float, float] = (4.0, 12.0)   # random pause between same-domain starts
    cooldown_after_blocks: int = 3
    cooldown_s: float = 900.0
    acquire_timeout_s: float = 1800.0   # give up waiting for a cooling domain after this


@dataclass
class RunSettings:
    """Collection-run shape and output layout."""

    out_dir: str = "runs/dev"
    max_steps: int = 30
    action_profile: str = "hybrid"      # gui | hybrid
    resume: bool = True                 # skip trajectories that already finished
    save_messages: bool = False         # dump full model input (images stripped) per step
    annotate_screenshots: bool = True   # draw the executed action on a copy
    stale_limit: int = 4                # identical page states in a row before aborting
    parse_retries: int = 2              # corrective re-asks for malformed model output
    search_fallbacks: list[str] = field(
        default_factory=lambda: ["https://duckduckgo.com", "https://www.bing.com"]
    )
    max_fallback_switches: int = 2
    max_block_recoveries: int = 2       # times the agent may be sent back off a blocked site
    api_log: bool = True                # append per-call latency/usage to api_calls.jsonl


@dataclass
class Config:
    browser: BrowserSettings = field(default_factory=BrowserSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    pacing: PacingSettings = field(default_factory=PacingSettings)
    run: RunSettings = field(default_factory=RunSettings)

    @classmethod
    def load(cls, path: str | Path | None = None, overrides: dict | None = None) -> "Config":
        """Build a config from an optional JSON file plus a flat/nested override dict.

        Override keys may be nested (``{"model": {"model": "..."}}``) or dotted
        (``{"model.temperature": 0.2}``).
        """
        data: dict[str, Any] = {}
        if path:
            data = json.loads(Path(path).read_text())
        for key, value in (overrides or {}).items():
            if value is None:
                continue
            if "." in key:
                section, leaf = key.split(".", 1)
                data.setdefault(section, {})[leaf] = value
            elif isinstance(value, dict):
                data.setdefault(key, {}).update(value)
            else:
                data[key] = value

        config = cls()
        for section_name in ("browser", "model", "pacing", "run"):
            section_data = data.get(section_name) or {}
            section = getattr(config, section_name)
            valid = {f.name for f in dataclasses.fields(section)}
            unknown = set(section_data) - valid
            if unknown:
                raise ValueError(f"unknown {section_name} config keys: {sorted(unknown)}")
            for key, value in section_data.items():
                if key == "domain_gap_s" and isinstance(value, list):
                    value = tuple(value)
                setattr(section, key, value)
        return config

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
