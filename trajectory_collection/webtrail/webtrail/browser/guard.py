"""Rule-based page health checks.

`inspect_state` looks at one observed page (URL, title, HTML, screenshot,
element map) and classifies hard failure modes so the runner can end an
episode early instead of burning model calls on a CAPTCHA or a 403 page.

Detected kinds:
    captcha, challenge, access_denied, rate_limit, login_wall, geo_blocked,
    not_found, server_error, network_error

Each verdict also carries a scope: ``search`` when the blocked page belongs to
a search engine (the runner may fall back to another engine) and ``target``
otherwise (usually unrecoverable for this task).
"""

from __future__ import annotations

import re

from ..core.models import PageState, Verdict, domain_of

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
    ("not_found", ("page not found", "404 not found", "404 |", "404 -",
                   "error 404", "page does not exist", "page doesn't exist",
                   "page does not seem to exist")),
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
    ("not_found", ("page not found", "page you are looking for doesn't exist",
                   "page you were trying to access is not at this address",
                   "page does not seem to exist", "page has been moved or no longer exists")),
    ("geo_blocked", ("not available in your country",
                     "not available in your region",
                     "unavailable in your location", "451 unavailable")),
]

_URL_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("captcha", ("/sorry/", "captcha",)),
    ("challenge", ("/cdn-cgi/challenge", "geo.captcha-delivery.com", "validate.perfdrive")),
    ("login_wall", ("accounts.google.com/v3/signin", "login.microsoftonline",
                    "/checkpoint/challenge")),
    ("not_found", ("/errors/404", "/404.html", "/404/")),
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

    # Chromium's internal navigation failure page has no meaningful HTTP
    # metadata or content. Classify it for a model-visible diagnostic; after
    # preflight, the runner leaves this state/session intact so recovery remains
    # an explicit agent action.
    if url.startswith("chrome-error://"):
        return Verdict("network_error", scope, "Chromium rendered an internal navigation error page")

    # Browser response metadata is the strongest signal available. Keeping it
    # on PageState prevents a rendered 403/404 page from being mistaken for a
    # healthy page merely because it has working menus and a normal site title.
    status = state.http_status
    if status in {401, 403, 407}:
        return Verdict("access_denied", scope, f"main document returned HTTP {status}")
    if status in {404, 410}:
        return Verdict("not_found", scope, f"main document returned HTTP {status}")
    if status == 429:
        return Verdict("rate_limit", scope, "main document returned HTTP 429")
    if status == 451:
        return Verdict("geo_blocked", scope, "main document returned HTTP 451")
    if status == 406:
        return Verdict("network_error", scope, "main document returned HTTP 406")
    if status is not None and 500 <= status <= 599:
        return Verdict("server_error", scope, f"main document returned HTTP {status}")

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

    # Blank or nearly uniform pages are not terminal blocks. They may be an
    # intermediate render, a recoverable browser error, or a page the agent can
    # leave with reload/back/goto. The runner's stale-state guard still bounds
    # genuinely dead pages without misclassifying them as website blocking.

    return Verdict(None, scope, "")
