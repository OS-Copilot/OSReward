"""Action execution fidelity harness.

Proves that model-format actions run *exactly* as specified: every action is
expressed the way a model would emit it (0-1000 `box2d` targets), pushed
through the full grounding → compile → browser-service path, and then the
page's DOM state is asserted from a fresh element snapshot.

Run with the browser service up:

    python tests/action_fidelity.py [service_url]
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from webtrail.actions import compile_action
from webtrail.browser import ServicePool
from webtrail.config import BrowserSettings
from webtrail.grounding import SCHEMES, GroundingContext
from webtrail.types import ParsedAction

PAGE = Path(__file__).parent / "fidelity_page.html"


def bbox_of(state, name_fragment: str) -> list:
    for el in state.elements or []:
        if name_fragment.lower() in (el.get("name") or "").lower():
            return el["bbox"]
    raise AssertionError(f"element {name_fragment!r} not on page; have: "
                         f"{[(e.get('role'), e.get('name')) for e in state.elements or []][:20]}")


def value_of(state, name_fragment: str):
    for el in state.elements or []:
        if name_fragment.lower() in (el.get("name") or "").lower():
            return el["checked"] if "checked" in el else el.get("value")
    raise AssertionError(f"element {name_fragment!r} not found")


def to_box1000(bbox: list, viewport: tuple) -> list:
    """Convert a pixel bbox [x, y, w, h] into model-format [ymin, xmin, ymax, xmax] 0-1000."""
    x, y, w, h = bbox
    return [round(y / viewport[1] * 1000), round(x / viewport[0] * 1000),
            round((y + h) / viewport[1] * 1000), round((x + w) / viewport[0] * 1000)]


async def main(service: str) -> None:
    settings = BrowserSettings(service_hosts=[service], settle_ms=150, net_idle_ms=500)
    pool = ServicePool(settings)
    session = await pool.open_session()
    ctx = GroundingContext(SCHEMES["box1000"], session.viewport, session.viewport)
    passed = 0

    async def act(key: str, args: dict) -> None:
        compiled = compile_action(ParsedAction(key, args), ctx, "hybrid")
        if compiled.goto_url:
            result = await session.goto(compiled.goto_url)
            assert result.get("ok"), f"goto failed: {result}"
        for command in compiled.commands:
            result = await session.act(command)
            assert result.get("ok"), f"{key} failed: {result}"

    def check(label: str, actual, expected) -> None:
        nonlocal passed
        assert actual == expected, f"{label}: expected {expected!r}, got {actual!r}"
        passed += 1
        print(f"  ok  {label}: {actual!r}")

    try:
        nav = await session.goto(PAGE.resolve().as_uri())
        assert nav.get("ok"), nav
        state = await session.snapshot()

        # click — button label must change to clicked:1, exactly once
        await act("click", {"box2d": to_box1000(bbox_of(state, "clicked:0"), session.viewport)})
        state = await session.snapshot()
        check("click increments exactly once", bbox_of(state, "clicked:1") is not None, True)

        # double_click
        await act("double_click", {"box2d": to_box1000(bbox_of(state, "dbl:no"), session.viewport)})
        state = await session.snapshot()
        check("double_click fires dblclick", bbox_of(state, "dbl:yes") is not None, True)

        # type into an empty field
        await act("type", {"box2d": to_box1000(bbox_of(state, "Name"), session.viewport),
                           "text": "Ada Lovelace"})
        state = await session.snapshot()
        check("type writes exact text", value_of(state, "Name"), "Ada Lovelace")

        # fill must REPLACE existing wrong content, not append
        await act("fill", {"box2d": to_box1000(bbox_of(state, "Search"), session.viewport),
                           "text": "correct query"})
        state = await session.snapshot()
        check("fill replaces old value", value_of(state, "Search"), "correct query")

        # hotkey with repeat — trim chars off the end of the name field.
        # caret-to-end is platform-specific: macOS scrolls the page on "End"
        # (Cmd+Right is the caret command); Linux/Windows use "End".
        caret_end = "Meta+ArrowRight" if sys.platform == "darwin" else "End"
        await act("click", {"box2d": to_box1000(bbox_of(state, "Name"), session.viewport)})
        await act("hotkey", {"keys": caret_end})
        await act("hotkey", {"keys": "Backspace", "repeat": 9})
        state = await session.snapshot()
        check("hotkey repeat deletes chars", value_of(state, "Name"), "Ada")

        # select_option picks by visible label on a native <select>
        await act("select_option", {"box2d": to_box1000(bbox_of(state, "Color"), session.viewport),
                                    "label": "Blue"})
        state = await session.snapshot()
        check("select_option picks label", value_of(state, "Color"), "Blue")

        # set_checked
        await act("set_checked", {"box2d": to_box1000(bbox_of(state, "I agree"), session.viewport),
                                  "checked": True})
        state = await session.snapshot()
        check("set_checked checks the box", value_of(state, "I agree"), True)

        # hover reveals hidden content
        await act("hover", {"box2d": to_box1000(bbox_of(state, "hover over me"), session.viewport)})
        fresh = await session.snapshot()
        check("hover reveals secret", "SECRET-REVEALED" in (fresh.html or ""), True)

        # drag the slider from 0 to ~ mid-range
        slider_box = bbox_of(state, "Volume")
        x, y, w, h = slider_box
        start = [round((y + h / 2) / session.viewport[1] * 1000),
                 round((x + 8) / session.viewport[0] * 1000),
                 round((y + h / 2) / session.viewport[1] * 1000),
                 round((x + 8) / session.viewport[0] * 1000)]
        end = [start[0], round((x + w * 0.75) / session.viewport[0] * 1000),
               start[2], round((x + w * 0.75) / session.viewport[0] * 1000)]
        await act("drag", {"from": start, "to": end})
        state = await session.snapshot()
        slider_value = int(value_of(state, "Volume") or 0)
        check("drag moves slider past halfway", slider_value > 55, True)

        # scroll reaches page bottom content
        await act("scroll", {"dy": 4000})
        state = await session.snapshot()
        check("scroll moves the viewport", (state.scroll or {}).get("y", 0) > 2000, True)

        # go_back / go_forward round trip through goto
        await act("goto", {"url": "https://example.com"})
        await act("go_back", {})
        state = await session.snapshot()
        check("go_back returns to harness", "fidelity" in (state.title or "").lower(), True)

        print(f"\nACTION FIDELITY: all {passed} checks passed")
    finally:
        await session.close()
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:9300"))
