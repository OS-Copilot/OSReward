"""Model-aware screenshot sizing for coordinate-grounded collection.

Pixel coordinates are only valid in the image space a model actually sees.
Some providers silently downscale large images before inference, so merely
declaring the browser's 1920x1080 viewport is insufficient.  This module
pre-resizes screenshots to a known provider-native size; grounding can then
map that explicit size back onto the original browser viewport.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class VisionProfile:
    name: str
    max_edge: int
    max_visual_tokens: int
    patch_size: int = 28


# Claude Sonnet 4.6 and other standard-resolution Claude models use at most
# 1568 visual patches and a 1568-pixel edge.  Known newer high-resolution
# families have a larger budget.  Unknown Claude models intentionally use the
# conservative standard profile so their coordinates remain safe.
CLAUDE_STANDARD = VisionProfile("claude_standard", 1568, 1568)
CLAUDE_HIGH_RESOLUTION = VisionProfile("claude_high_resolution", 2576, 4784)

_CLAUDE_HIGH_RESOLUTION_HINTS = (
    "claude-sonnet-5",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-fable-5",
    "claude-mythos-5",
)


def profile_for_model(model_id: str) -> VisionProfile | None:
    normalized = (model_id or "").lower().replace("_", "-")
    if "claude" not in normalized:
        return None
    if any(hint in normalized for hint in _CLAUDE_HIGH_RESOLUTION_HINTS):
        return CLAUDE_HIGH_RESOLUTION
    return CLAUDE_STANDARD


def _fit_max_side(size: tuple[int, int], max_side: int) -> tuple[int, int]:
    width, height = size
    if max_side <= 0 or max(width, height) <= max_side:
        return width, height
    scale = max_side / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def _visual_tokens(size: tuple[int, int], profile: VisionProfile) -> int:
    width, height = size
    patch = profile.patch_size
    return math.ceil(width / patch) * math.ceil(height / patch)


def _fit_profile(size: tuple[int, int], profile: VisionProfile) -> tuple[int, int]:
    width, height = size
    if max(size) <= profile.max_edge and _visual_tokens(size, profile) <= \
            profile.max_visual_tokens:
        return size

    landscape = width >= height
    long_side = width if landscape else height
    short_side = height if landscape else width
    start = min(long_side, profile.max_edge)
    # Search the discrete 28x28 patch budget, preserving the source aspect
    # ratio.  For 1920x1080 under Claude's standard profile this deliberately
    # returns the provider-native 1456x819 instead of the edge-only 1568x882.
    for candidate_long in range(start, 0, -1):
        candidate_short = max(1, round(short_side * candidate_long / long_side))
        candidate = ((candidate_long, candidate_short) if landscape
                     else (candidate_short, candidate_long))
        if _visual_tokens(candidate, profile) <= profile.max_visual_tokens:
            return candidate
    return 1, 1


@lru_cache(maxsize=256)
def model_input_size(model_id: str, source_size: tuple[int, int],
                     requested_max_side: int = 0) -> tuple[int, int]:
    """Return the exact client-side size to send for a model and source image."""
    requested = _fit_max_side(source_size, requested_max_side)
    profile = profile_for_model(model_id)
    return _fit_profile(requested, profile) if profile else requested
