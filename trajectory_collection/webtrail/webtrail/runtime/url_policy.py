"""Classify whether a task must use its provided website.

The classifier is intentionally conservative: only explicit website
interactions are hard requirements. Read-only information gathering remains
flexible so a blocked start URL can fall back to another credible source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..core.models import Task


@dataclass(frozen=True)
class UrlPolicy:
    kind: str                    # hard_required | flexible
    reason: str

    @property
    def hard_required(self) -> bool:
        return self.kind == "hard_required"


_EXPLICIT_FALLBACK = re.compile(
    r"\b(?:if (?:the )?(?:site|website|page) is (?:inaccessible|unavailable|blocked)|"
    r"alternative (?:credible )?(?:website|site|source)|another (?:website|site|source)|"
    r"other (?:credible )?(?:website|site|source))\b",
    re.IGNORECASE,
)

# High-precision interaction requirements. Merely asking for information "on"
# or "from" a site is deliberately not enough: those answers may be obtained
# from another credible source when the provided URL is unavailable.
_HARD_INTERACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(
        r"^\s*(?:successfully\s+)?(?:subscribe|sign\s*up|register)\b",
        re.IGNORECASE,
    ),
     "the task requires an account or subscription interaction"),
    (re.compile(r"^\s*(?:successfully\s+)?create (?:an? )?account\b", re.IGNORECASE),
     "the task requires creating an account"),
    (re.compile(r"^\s*(?:successfully\s+)?upload\b", re.IGNORECASE),
     "the task requires uploading through the provided site"),
    (re.compile(r"^\s*convert\b.+\b(?:using|with)\b", re.IGNORECASE | re.DOTALL),
     "the task requires using the provided conversion tool"),
    (re.compile(r"\b(?:fill out|complete|submit)\b.+\bform\b", re.IGNORECASE | re.DOTALL),
     "the task requires operating a form on the provided site"),
    (re.compile(r"\bby (?:clicking|selecting|uploading)\b", re.IGNORECASE),
     "the task explicitly requires a website UI interaction"),
    (re.compile(
        r"\b(?:use|using)\b.+\b(?:calculator|converter|interactive map|online tool)\b",
        re.IGNORECASE | re.DOTALL,
    ), "the task requires a specific interactive website tool"),
)


def classify_url_policy(task: Task) -> UrlPolicy:
    """Return a high-precision hard/flexible URL dependency classification."""
    override = str(task.extras.get("url_policy") or "").strip().lower()
    if override in {"hard_required", "flexible"}:
        return UrlPolicy(override, "explicit task metadata override")

    instruction = task.instruction.strip()
    if _EXPLICIT_FALLBACK.search(instruction):
        return UrlPolicy(
            "flexible", "the instruction explicitly permits an alternative source"
        )
    for pattern, reason in _HARD_INTERACTIONS:
        if pattern.search(instruction):
            return UrlPolicy("hard_required", reason)
    return UrlPolicy(
        "flexible",
        "the task is read-only information gathering and can use another credible source",
    )
