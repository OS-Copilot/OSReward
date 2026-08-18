"""Run configuration.

Everything the collector needs is grouped into dataclasses and bundled
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
    # Preflight is allowed one fresh-session retry on the next service worker.
    # The hard attempt deadline covers navigation + the lite snapshot together,
    # preventing their individual timeouts from stacking into multi-minute hangs.
    preflight_session_retries: int = 1
    preflight_attempt_timeout_s: float = 45.0
    preflight_nav_timeout_ms: int = 30_000
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
    """Model transport plus how to present and ground browser actions.

    Credentials are never part of configuration.  The transport reads the
    standard ``OPENAI_API_KEY`` or ``ANTHROPIC_API_KEY`` environment variable
    at runtime, selected by ``provider``.
    """

    provider: str = "auto"              # auto | openai | anthropic
    model: str = ""
    # Optional generic-compatible endpoint override. Empty uses the provider's
    # official API; ``stub:`` enables deterministic local smoke runs.
    base_url: str = ""
    temperature: float | None = None
    max_tokens: int = 2048
    request_timeout_s: float = 180.0
    max_retries: int = 4

    grounding: str = "auto"             # auto | box1000 | point1000 | pixel
    history_mode: str = "windowed"      # windowed | text_full
    history_window: int = 2             # current + immediately previous screenshot
    vision_only: bool = False           # optional stricter mode: also drop current URL text
    # Requested transport cap. The model adapter may reduce it further so the
    # provider cannot silently resize the image and invalidate pixel coordinates.
    image_max_side: int = 0
    image_format: str = "png"           # png | jpeg (model transport only)
    image_jpeg_quality: int = 85         # used when image_format = jpeg
    analysis_words: int = 150           # requested reasoning length in the prompt


@dataclass
class JudgeSettings:
    """Optional in-run VLM judging, triggered after a collection batch."""

    enabled: bool = False
    batch_size: int = 100
    concurrency: int = 4
    last_n: int = 5
    votes: int = 1
    rubric: str = "binary"              # binary | multi
    flush_partial: bool = True           # judge the final short batch on clean exit

    provider: str = "auto"
    model: str = ""
    base_url: str = ""
    temperature: float | None = None
    max_tokens: int = 4096
    request_timeout_s: float = 240.0
    max_retries: int = 2
    image_max_side: int = 1280
    image_format: str = "jpeg"
    image_jpeg_quality: int = 85

    def model_settings(self) -> ModelSettings:
        """Build the transport settings used by :class:`ChatModel`."""
        return ModelSettings(
            provider=self.provider,
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            request_timeout_s=self.request_timeout_s,
            max_retries=self.max_retries,
            image_max_side=self.image_max_side,
            image_format=self.image_format,
            image_jpeg_quality=self.image_jpeg_quality,
        )


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
class ResourceSettings:
    """Memory-aware admission control for parallel browser episodes."""

    enabled: bool = True
    # Keep both an absolute and proportional reserve; the larger one wins.
    memory_reserve_mb: int = 4096
    memory_reserve_fraction: float = 0.05
    # Includes the Chromium browser, its Node owner, and collector-side state.
    estimated_episode_mb: int = 1024
    memory_poll_s: float = 5.0


@dataclass
class RunSettings:
    """Collection-run shape and output layout."""

    out_dir: str = "runs/dev"
    max_steps: int = 15
    # Wall-clock budget for one complete observe -> model -> act -> persist
    # cycle.  This sits above the component-level HTTP/navigation/LLM timeouts
    # so retries in several layers cannot hold a worker forever.  Set to 0 to
    # disable.  A timed-out step is recorded and retried from a fresh
    # observation; repeated timeouts end the episode as an infrastructure error.
    step_timeout_s: float = 360.0
    max_consecutive_step_timeouts: int = 2
    action_profile: str = "hybrid"      # gui | hybrid
    resume: bool = True                 # skip trajectories that already finished
    save_messages: bool = False         # dump full model input (images stripped) per step
    save_html: bool = False             # raw page HTML is large; opt in when needed
    save_axtree: bool = False           # accessibility trees are large; opt in
    save_elements: bool = False         # DOM-derived interactive map; opt in for debugging
    save_model_views: bool = False      # optional copy of the resized model input
    annotate_screenshots: bool = False  # draw executed actions only when requested
    stale_limit: int = 4                # identical page states in a row before aborting
    max_same_action_repeats: int = 2    # refuse a third equivalent action on an unchanged page
    parse_retries: int = 2              # corrective re-asks for malformed model output
    search_fallbacks: list[str] = field(
        # Keep the runner's preflight order aligned with the acting prompt.
        default_factory=lambda: [
            "https://www.bing.com", "https://duckduckgo.com",
            "https://www.google.com",
        ]
    )
    # Kept for backward-compatible config loading. Main-loop recovery navigation
    # is model-owned; automatic fallback/session changes happen only in preflight.
    max_fallback_switches: int = 2
    max_block_recoveries: int = 2


@dataclass
class Config:
    browser: BrowserSettings = field(default_factory=BrowserSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    judge: JudgeSettings = field(default_factory=JudgeSettings)
    pacing: PacingSettings = field(default_factory=PacingSettings)
    resources: ResourceSettings = field(default_factory=ResourceSettings)
    run: RunSettings = field(default_factory=RunSettings)

    @classmethod
    def load(cls, path: str | Path | None = None, overrides: dict | None = None) -> Config:
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
        for section_name in (
            "browser", "model", "judge", "pacing", "resources", "run"
        ):
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
