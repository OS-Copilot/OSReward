"""`doctor` subcommand: one-shot health check of the whole stack.

Verifies the pieces a collection run needs before you start one: the browser
service workers are up (and report a live Chromium), and — if you pass a model
id — that the selected provider answers. Exits non-zero if anything essential is
down, so it drops into a deployment script.
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from ..agents.llm import ChatModel, LLMError
from ..browser.client import ServicePool
from ..core.config import BrowserSettings, ModelSettings

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


async def _check_model(settings: ModelSettings) -> bool:
    model: ChatModel | None = None
    try:
        model = ChatModel(settings)
        if settings.base_url.startswith("stub:"):
            print(f"  {WARN} model transport is a stub; skipping live check")
            return True
        await model.complete(
            [{"role": "user", "content": "Reply with OK."}], max_tokens=8
        )
    except (LLMError, ValueError) as err:
        print(f"  {BAD} {settings.provider} model {settings.model}: {err}")
        return False
    finally:
        if model is not None:
            await model.close()
    print(f"  {OK} {model.provider} model {settings.model} responded")
    return True


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "doctor", help="check the browser service (and optionally a model provider)"
    )
    parser.add_argument("--service", action="append",
                        help="browser service host, repeatable "
                             "(default http://127.0.0.1:9300)")
    parser.add_argument("--provider", choices=["auto", "openai", "anthropic"],
                        default="auto")
    parser.add_argument("--base-url", default="",
                        help="optional compatible API base URL")
    parser.add_argument("--model", default="", help="model id to ping (optional)")
    parser.set_defaults(handler=main)


async def _run(args: argparse.Namespace) -> int:
    hosts = args.service or ["http://127.0.0.1:9300"]
    print("browser service")
    ok = await _check_services(hosts)
    if args.model or args.base_url:
        print("model provider")
        settings = ModelSettings(
            provider=args.provider, model=args.model, base_url=args.base_url,
            max_tokens=8, request_timeout_s=30.0, max_retries=1,
        )
        ok = await _check_model(settings) and ok
    print()
    print("all good — ready to collect" if ok else "problems found — see above")
    return 0 if ok else 1


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    raise SystemExit(asyncio.run(_run(args)))
