"""Rule-based page health checks.

`inspect_state` looks at one observed page (URL, title, HTML, screenshot,
element map) and classifies hard failure modes so the runner can end an
episode early instead of burning model calls on a CAPTCHA or a 403 page.

Detected kinds:
    captcha, challenge, access_denied, rate_limit, login_wall, geo_blocked,
    network_error, empty_page

Each verdict also carries a scope: ``search`` when the blocked page belongs to
a search engine (the runner may fall back to another engine) and ``target``
otherwise (usually unrecoverable for this task).
"""

from __future__ import annotations

import re

from . import imutil
from .types import PageState, Verdict, domain_of

SEARCH_ENGINE_DOMAINS = {
    "google.com", "bing.com", "duckduckgo.com", "yahoo.com", "baidu.com",
    "yandex.com", "startpage.com", "ecosia.org", "brave.com",
}

# (kind, patterns) matched against lower-cased page title
_TITLE_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("challenge", ("just a moment", "checking your browser", "attention required",
                   "please verify", "verifying you are human", "security check",
                   "one more step")),
    ("captcha", ("captcha", "robot check", "are you a robot")),
    ("access_denied", ("access denied", "access to this page has been denied",
                       "403 forbidden", "forbidden", "not authorized",
                       "unauthorized")),
    ("rate_limit", ("too many requests", "rate limit", "429")),
    ("network_error", ("this site can", "server not found", "problem loading page",
                       "page not available", "err_", "dns_probe")),
]

# (kind, patterns) matched against lower-cased HTML
_HTML_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("captcha", ("g-recaptcha", "h-captcha", "hcaptcha.com", "cf-turnstile",
                 "arkose", "funcaptcha", "px-captcha", "captcha-delivery",
                 "geo.captcha", "recaptcha/api")),
    ("challenge", ("cf-challenge", "challenge-platform", "_cf_chl", "ddos-guard",
                   "datadome", "imperva", "incapsula", "akamai-bot",
                   "perimeterx", "verifying you are human",
                   "enable javascript and cookies to continue")),
    ("rate_limit", ("unusual traffic from your computer network",
                    "too many requests", "temporarily rate limited",
                    "sending automated queries")),
    ("access_denied", ("access denied", "you don't have permission to access",
                       "the owner of this website has banned",
                       "blocked by network security")),
    ("geo_blocked", ("not available in your country",
                     "not available in your region",
                     "unavailable in your location", "451 unavailable")),
]

_URL_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("captcha", ("/sorry/", "captcha",)),
    ("challenge", ("/cdn-cgi/challenge", "geo.captcha-delivery.com", "validate.perfdrive")),
    ("login_wall", ("accounts.google.com/v3/signin", "login.microsoftonline",
                    "/checkpoint/challenge")),
]

_LOGIN_HINTS = ("sign in to continue", "log in to continue", "login to continue",
                "you must be logged in", "please sign in to view")


def _match(rules: list[tuple[str, tuple[str, ...]]], haystack: str) -> tuple[str, str] | None:
    for kind, needles in rules:
        for needle in needles:
            if needle in haystack:
                return kind, needle
    return None


def is_search_domain(url: str) -> bool:
    return domain_of(url) in SEARCH_ENGINE_DOMAINS


def inspect_state(state: PageState, goto_error: str | None = None) -> Verdict:
    """Classify one observed page. Returns a non-blocking verdict when healthy."""
    url = (state.url or "").lower()
    title = (state.title or "").lower()
    html = (state.html or "").lower()
    scope = "search" if is_search_domain(state.url or "") else "target"

    if goto_error:
        lowered = goto_error.lower()
        if "timeout" in lowered:
            return Verdict("network_error", scope, f"navigation timeout: {goto_error[:160]}")
        if "net::" in lowered or "dns" in lowered:
            return Verdict("network_error", scope, f"navigation error: {goto_error[:160]}")

    hit = _match(_URL_RULES, url)
    if hit:
        return Verdict(hit[0], scope, f"url contains '{hit[1]}'")

    hit = _match(_TITLE_RULES, title)
    if hit:
        return Verdict(hit[0], scope, f"title contains '{hit[1]}'")

    text_len = len(re.sub(r"<[^>]+>", "", html or "").strip())
    interactive = len(state.elements or [])

    # HTML markers are ambiguous on their own: plenty of healthy pages embed a
    # reCAPTCHA widget or mention rate limits in copy. Only treat them as a
    # block when the page otherwise looks like an interstitial wall — barely
    # any text, or barely anything to interact with.
    wall_like = text_len < 2000 or (state.elements is not None and interactive <= 10)
    if wall_like:
        hit = _match(_HTML_RULES, html)
        if hit:
            return Verdict(hit[0], scope, f"html contains '{hit[1]}' on a wall-like page")
        # login wall: explicit gating text on an otherwise contentless page
        if any(hint in html for hint in _LOGIN_HINTS):
            return Verdict("login_wall", scope, "page demands login to continue")

    # empty / dead page checks
    if url in ("", "about:blank"):
        return Verdict("empty_page", scope, "url is about:blank")
    if state.html is not None and text_len < 80 and interactive == 0:
        return Verdict("empty_page", scope, f"page text is {text_len} chars with no elements")
    if state.screenshot_png is not None:
        image = imutil.load_png(state.screenshot_png)
        if imutil.near_uniform(image) and interactive == 0:
            return Verdict("empty_page", scope, "screenshot is a uniform frame")

    return Verdict(None, scope, "")
