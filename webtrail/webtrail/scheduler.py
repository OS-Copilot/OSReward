"""Domain-aware pacing for the collection run.

Live websites throttle and challenge aggressive automation, so episode starts
are shaped per registrable domain:

* at most ``per_domain`` concurrent episodes on the same domain,
* a randomized gap between consecutive starts on the same domain,
* a cooldown once a domain produces several consecutive blocked episodes.

Usage::

    governor = DomainGovernor(pacing)
    async with governor.slot(task.domain):        # may wait
        ... run the episode ...
    governor.report(task.domain, blocked=...)
"""

from __future__ import annotations

import asyncio
import random
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass

from .config import PacingSettings


class DomainCoolingDown(Exception):
    """Raised when a domain slot could not be acquired within the timeout."""


@dataclass
class _DomainState:
    semaphore: asyncio.Semaphore
    next_start_at: float = 0.0
    consecutive_blocks: int = 0
    cooling_until: float = 0.0
    starts: int = 0
    blocks: int = 0


class DomainGovernor:
    def __init__(self, settings: PacingSettings):
        self.settings = settings
        self._domains: dict[str, _DomainState] = {}
        self._lock = asyncio.Lock()

    def _state(self, domain: str) -> _DomainState:
        if domain not in self._domains:
            self._domains[domain] = _DomainState(
                semaphore=asyncio.Semaphore(self.settings.per_domain)
            )
        return self._domains[domain]

    @asynccontextmanager
    async def slot(self, domain: str):
        state = self._state(domain)
        deadline = time.monotonic() + self.settings.acquire_timeout_s

        try:
            await asyncio.wait_for(
                state.semaphore.acquire(),
                timeout=max(deadline - time.monotonic(), 0.1),
            )
        except asyncio.TimeoutError:
            raise DomainCoolingDown(f"{domain}: no slot within acquire timeout") from None

        try:
            # respect cooldown and inter-start gap while holding the slot so a
            # cooling domain does not admit a burst the moment it re-opens
            while True:
                async with self._lock:
                    now = time.monotonic()
                    wait_until = max(state.cooling_until, state.next_start_at)
                    if wait_until <= now:
                        low, high = self.settings.domain_gap_s
                        state.next_start_at = now + random.uniform(low, high)
                        state.starts += 1
                        break
                if wait_until > deadline:
                    raise DomainCoolingDown(
                        f"{domain}: cooling down for another "
                        f"{wait_until - time.monotonic():.0f}s"
                    )
                await asyncio.sleep(min(wait_until - time.monotonic() + 0.05, 5.0))
            yield
        finally:
            state.semaphore.release()

    def report(self, domain: str, blocked: bool) -> None:
        state = self._state(domain)
        if blocked:
            state.blocks += 1
            state.consecutive_blocks += 1
            if state.consecutive_blocks >= self.settings.cooldown_after_blocks:
                state.cooling_until = time.monotonic() + self.settings.cooldown_s
                state.consecutive_blocks = 0
        else:
            state.consecutive_blocks = 0

    def stats(self) -> dict:
        now = time.monotonic()
        return {
            domain: {
                "starts": s.starts,
                "blocks": s.blocks,
                "cooling_for_s": max(0, round(s.cooling_until - now)),
            }
            for domain, s in sorted(self._domains.items())
        }
