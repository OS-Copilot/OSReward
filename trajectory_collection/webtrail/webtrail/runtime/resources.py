"""Resource-aware admission control for browser-heavy collection workers."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from ..core.config import ResourceSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemorySnapshot:
    total_mb: int
    available_mb: int
    source: str


def _number(path: Path) -> int | None:
    try:
        value = path.read_text().strip()
        return None if value == "max" else int(value)
    except (OSError, ValueError):
        return None


def _inactive_file_bytes(path: Path) -> int:
    """Return reclaimable inactive file cache reported by a cgroup."""
    try:
        values = {
            key: int(value)
            for line in path.read_text().splitlines()
            if len(parts := line.split()) == 2
            for key, value in [parts]
        }
    except (OSError, ValueError):
        return 0
    # cgroup v1 can expose both local and hierarchical totals.  The usage
    # counter is hierarchical, so its matching total is the correct value.
    return max(0, values.get("total_inactive_file",
                             values.get("inactive_file", 0)))


def memory_snapshot() -> MemorySnapshot:
    """Return cgroup working-set headroom, otherwise host memory.

    Both cgroup v1 ``memory.usage_in_bytes`` and v2 ``memory.current`` include
    reclaimable page cache.  Treating that cache as resident memory can pin the
    admission gate after a large file scan even though the kernel can reclaim
    it.  Match Kubernetes' working-set calculation by subtracting inactive
    file cache before computing headroom.
    """
    proc: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, raw = line.partition(":")
            if key in {"MemTotal", "MemAvailable"}:
                proc[key] = int(raw.strip().split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        pass

    host_total = proc.get("MemTotal", 0)
    host_available = proc.get("MemAvailable", 0)
    candidates = [
        (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current"),
         Path("/sys/fs/cgroup/memory.stat")),
        (Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
         Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
         Path("/sys/fs/cgroup/memory/memory.stat")),
    ]
    for limit_path, used_path, stat_path in candidates:
        limit, used = _number(limit_path), _number(used_path)
        # Some cgroup-v1 hosts expose a sentinel close to 2**63 for "unlimited".
        if not limit or used is None or limit >= (1 << 60):
            continue
        working_set = max(0, used - _inactive_file_bytes(stat_path))
        available = max(0, limit - working_set)
        if host_total:
            available = min(available, host_available)
        return MemorySnapshot(
            total_mb=max(1, limit // (1024 * 1024)),
            available_mb=max(0, available // (1024 * 1024)),
            source="cgroup",
        )
    return MemorySnapshot(
        total_mb=max(1, host_total // (1024 * 1024)),
        available_mb=max(0, host_available // (1024 * 1024)),
        source="host",
    )


class MemoryGovernor:
    """Cap initial concurrency and pause new episodes under memory pressure.

    Existing episodes are never killed.  Instead, workers wait at the admission
    gate before opening another Chromium session.  This keeps results intact and
    makes a deliberately high requested concurrency safe on smaller machines.
    """

    def __init__(self, settings: ResourceSettings, requested_concurrency: int,
                 reader: Callable[[], MemorySnapshot] = memory_snapshot):
        self.settings = settings
        self.requested_concurrency = max(1, int(requested_concurrency))
        self._reader = reader
        initial = reader()
        self.reserve_mb = max(
            int(settings.memory_reserve_mb),
            int(initial.total_mb * float(settings.memory_reserve_fraction)),
        )
        estimate = max(1, int(settings.estimated_episode_mb))
        if settings.enabled:
            safe = max(1, (initial.available_mb - self.reserve_mb) // estimate)
            self.effective_concurrency = min(self.requested_concurrency, safe)
        else:
            self.effective_concurrency = self.requested_concurrency
        self._semaphore = asyncio.Semaphore(self.effective_concurrency)
        self._lock = asyncio.Lock()
        self._active = 0
        self._peak_active = 0
        self._min_available_mb = initial.available_mb
        self._wait_events = 0
        self._initial = initial

    async def _wait_for_headroom(self) -> MemorySnapshot:
        estimate = max(1, int(self.settings.estimated_episode_mb))
        while True:
            snapshot = self._reader()
            self._min_available_mb = min(self._min_available_mb, snapshot.available_mb)
            if (not self.settings.enabled or
                    snapshot.available_mb - self.reserve_mb >= estimate):
                return snapshot
            self._wait_events += 1
            logger.warning(
                "memory gate: %d MiB available, holding new episode until above %d MiB",
                snapshot.available_mb, self.reserve_mb + estimate,
            )
            await asyncio.sleep(max(0.1, float(self.settings.memory_poll_s)))

    @asynccontextmanager
    async def slot(self):
        await self._semaphore.acquire()
        admitted = False
        try:
            await self._wait_for_headroom()
            async with self._lock:
                self._active += 1
                self._peak_active = max(self._peak_active, self._active)
                admitted = True
            yield
        finally:
            if admitted:
                async with self._lock:
                    self._active -= 1
            self._semaphore.release()

    def sample(self) -> dict:
        snapshot = self._reader()
        self._min_available_mb = min(self._min_available_mb, snapshot.available_mb)
        return {**asdict(snapshot), "active": self._active}

    def stats(self) -> dict:
        return {
            "enabled": self.settings.enabled,
            "requested_concurrency": self.requested_concurrency,
            "effective_concurrency": self.effective_concurrency,
            "memory_reserve_mb": self.reserve_mb,
            "estimated_episode_mb": self.settings.estimated_episode_mb,
            "initial": asdict(self._initial),
            "min_available_mb": self._min_available_mb,
            "peak_active": self._peak_active,
            "wait_events": self._wait_events,
        }
