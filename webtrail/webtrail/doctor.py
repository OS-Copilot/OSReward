"""`doctor` subcommand: one-shot health check of the whole stack.

Verifies the pieces a collection run needs before you start one: the browser
service workers are up (and report a live Chromium), and — if you pass a model
endpoint — that the endpoint answers. Exits non-zero if anything essential is
down, so it drops into a deployment script.
"""

from __future__ import annotations

import argparse
import asyncio
import logging

import httpx

from .browser import ServicePool
from .config import BrowserSettings

logger = logging.getLogger(__name__)

OK, BAD, WARN = "\033[32m✓\033[0m", "\033[31m✗\033[0m", "\033[33m!\033[0m"


async def _check_services(hosts: list[str]) -> bool:
    pool = ServicePool(BrowserSettings(service_hosts=hosts))
    try:
        health = await pool.health()
    finally:
        await pool.close()
    all_ok = True
    for entry in health:
        if entry.get("ok"):
            extra = []
            if "sessions" in entry:
                extra.append(f"{entry['sessions']} live sessions")
            if "rss_mb" in entry:
                extra.append(f"{entry['rss_mb']} MB rss")
            if "uptime_s" in entry:
                extra.append(f"up {entry['uptime_s']}s")
            print(f"  {OK} browser service {entry['host']}"
                  + (f"  ({', '.join(extra)})" if extra else ""))
        else:
            all_ok = False
            print(f"  {BAD} browser service {entry['host']}: {entry.get('error', 'unhealthy')}")
    if not health:
        print(f"  {BAD} no browser service hosts configured")
        return False
    return all_ok


async def _check_model(base_url: str, api_key: str, model: str) -> bool:
    if base_url.startswith("stub:"):
        print(f"  {WARN} model endpoint is a stub ({base_url}); skipping live check")
        return True
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    payload = {"model": model or "gpt-4o-mini",
               "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, json=payload, headers=headers)
    except httpx.HTTPError as err:
        print(f"  {BAD} model endpoint {base_url}: {err}")
        return False
    if resp.status_code < 400:
        print(f"  {OK} model endpoint {base_url} ({model or '?'}) responded")
        return True
    # 400s on a 1-token ping are usually the model complaining, not the endpoint
    snippet = resp.text[:120].replace("\n", " ")
    ok = resp.status_code < 500 and resp.status_code != 401 and resp.status_code != 403
    print(f"  {OK if ok else BAD} model endpoint {base_url}: HTTP {resp.status_code} {snippet}")
    return ok


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "doctor", help="check the browser service (and optionally the model endpoint)"
    )
    parser.add_argument("--service", action="append",
                        help="browser service host, repeatable "
                             "(default http://127.0.0.1:9300)")
    parser.add_argument("--base-url", help="model endpoint to ping (optional)")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="", help="model id for the ping")
    parser.set_defaults(handler=main)


async def _run(args: argparse.Namespace) -> int:
    hosts = args.service or ["http://127.0.0.1:9300"]
    print("browser service")
    ok = await _check_services(hosts)
    if args.base_url:
        print("model endpoint")
        ok = await _check_model(args.base_url, args.api_key, args.model) and ok
    print()
    print("all good — ready to collect" if ok else "problems found — see above")
    return 0 if ok else 1


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    raise SystemExit(asyncio.run(_run(args)))
