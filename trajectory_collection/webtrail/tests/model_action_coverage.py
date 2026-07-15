"""Model × action coverage test.

Complements tests/action_fidelity.py (which proves the *execution* layer runs
every action correctly). This proves the *production* layer: that each model,
prompted for a specific action against a real screenshot, emits a reply that
parses and compiles into that action.

For each (model, action) it sends the model the action catalogue (in that
model's own coordinate scheme) plus one screenshot of the control-panel page
and a single-purpose instruction, then checks the parsed action key and that
it compiles to executable commands.

Usage:
    python tests/model_action_coverage.py --base-url URL --api-key KEY \
        [--models gemini-3-flash-preview,gpt-5.5,...]
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from webtrail import prompts
from webtrail.actions import ActionError, compile_action
from webtrail.agent import extract_action
from webtrail.browser import ServicePool
from webtrail.config import BrowserSettings, ModelSettings
from webtrail.grounding import GroundingContext, scheme_for_model
from webtrail.llm import ChatModel

PAGE = Path(__file__).parent / "fidelity_page.html"

# (expected_action, instruction). Acceptable equivalents are listed per action
# in EQUIVALENTS below (e.g. a model may fill a field with `type`+clear).
SCENARIOS = [
    ("click", "Click the button that currently shows the text 'clicked:0'."),
    ("double_click", "Double-click the button that shows the text 'dbl:no'."),
    ("hover", "Hover the mouse over the bordered text that says 'hover over me'."),
    ("scroll", "Scroll the page down by 400 pixels to reveal content lower down."),
    ("drag", "Drag the 'Volume' slider handle to the right to raise the value."),
    ("type", "Type the text 'Ada Lovelace' into the 'Name' input field."),
    ("fill", "The 'Search' field already contains wrong text. Clear it and fill "
             "it with 'correct query'."),
    ("hotkey", "Press the keyboard Escape key."),
    ("wait", "Wait for 2 seconds for the page to settle."),
    ("goto", "Navigate directly to the URL https://example.com."),
    ("go_back", "Go back to the previous page in the browser history."),
    ("go_forward", "Go forward to the next page in the browser history."),
    ("select_option", "In the 'Color' dropdown, select the option 'Blue'."),
    ("set_checked", "Tick / check the 'I agree' checkbox."),
    ("stop", "The task is finished. Stop and report the answer 'all done'."),
]

# actions a model may legitimately express a different but valid way
EQUIVALENTS = {
    # into an empty field, type and fill are interchangeable (both enter text)
    "type": {"type", "fill"},
    "fill": {"fill", "type"},
    # a custom widget may need a plain click instead of the specialized action
    "double_click": {"double_click", "click"},
    "select_option": {"select_option", "click"},
    "set_checked": {"set_checked", "click"},
    "hotkey": {"hotkey"},
}


async def test_model(model_id: str, settings_base: dict, screenshot_png: bytes,
                     viewport) -> dict:
    scheme = scheme_for_model(model_id)
    ctx = GroundingContext(scheme, viewport, viewport)
    system = prompts.system_prompt("hybrid", ctx, analysis_words=40)
    b64 = base64.b64encode(screenshot_png).decode()
    image_url = f"data:image/png;base64,{b64}"

    model = ChatModel(ModelSettings(model=model_id, max_tokens=4096, **settings_base))
    results = {}

    async def one(expected: str, instruction: str) -> None:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text":
                 f"## Task\n\n{instruction}\n\nThe screenshot of the page is "
                 "attached. Output exactly the single next action."},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]},
        ]
        try:
            reply = await model.complete(messages)
            parsed = extract_action(reply.text)
            compile_action(parsed, ctx, "hybrid")   # must compile to commands
            accepted = EQUIVALENTS.get(expected, {expected})
            status = "ok" if parsed.key in accepted else f"got:{parsed.key}"
        except (ValueError, ActionError) as err:
            status = f"FAIL:{str(err)[:40]}"
        except Exception as err:
            status = f"ERR:{str(err)[:40]}"
        results[expected] = status

    await asyncio.gather(*(one(e, i) for e, i in SCENARIOS))
    await model.close()
    return results


async def main(args) -> None:
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    settings_base = dict(base_url=args.base_url, api_key=args.api_key,
                         temperature=0.2)

    pool = ServicePool(BrowserSettings(service_hosts=[args.service],
                                       settle_ms=200, net_idle_ms=500))
    session = await pool.open_session()
    await session.goto(PAGE.resolve().as_uri())
    state = await session.snapshot()
    shot = state.screenshot_png
    if shot is None:
        raise SystemExit("failed to capture the test page screenshot")
    viewport = session.viewport
    await session.close()
    await pool.close()

    matrix = {}
    for m in models:
        print(f"testing {m} ...", flush=True)
        matrix[m] = await test_model(m, settings_base, shot, viewport)

    actions = [e for e, _ in SCENARIOS]
    w = max(len(a) for a in actions)
    print("\n" + " " * (w + 2) + "".join(f"{m.split('-')[0][:9]:>11}" for m in models))
    for a in actions:
        row = "".join(
            f"{'✓' if matrix[m][a] == 'ok' else matrix[m][a][:10]:>11}"
            for m in models)
        print(f"{a:{w}}  {row}")
    print()
    for m in models:
        oks = sum(1 for a in actions if matrix[m][a] == "ok")
        print(f"{m}: {oks}/{len(actions)} actions produced correctly")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--api-key", required=True)
    p.add_argument("--service", default="http://127.0.0.1:9300")
    p.add_argument("--models",
                   default="gemini-3-flash-preview,qwen3-max,gpt-5.5,kimi-k2.6,"
                           "claude-sonnet-4-6")
    asyncio.run(main(p.parse_args()))
