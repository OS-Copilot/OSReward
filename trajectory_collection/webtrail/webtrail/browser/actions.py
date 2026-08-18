"""Agent action space: validation, documentation, and compilation.

Two profiles are exposed:

* ``gui``    – pure visual operation: everything happens through mouse and
               keyboard on coordinates (plus ``wait`` and ``stop``).
* ``hybrid`` – the GUI set plus browser primitives (``goto``, ``go_back``,
               ``go_forward``, ``select_option``, ``set_checked``) that are
               closer to semi-programmatic control.

`compile_action` turns a parsed model action into the typed commands the
browser service executes, resolving visual coordinates via the grounding
context. Validation errors raise `ActionError` whose message is sent back to
the model as a corrective turn, so the wording is written for the model.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..core.models import CompiledAction, ParsedAction
from .grounding import (
    GroundingContext,
    GroundingError,
    resolve_optional_target,
    resolve_target,
)


class ActionError(ValueError):
    """Invalid action from the model; message is model-facing."""


@dataclass(frozen=True)
class ActionSpec:
    key: str
    describe: Callable[[GroundingContext], str]
    compile: Callable[[dict, GroundingContext], CompiledAction]


def _require_text(args: dict, key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value:
        raise ActionError(f"`{key}` (non-empty string) is required for this action")
    return value


# --- individual actions -----------------------------------------------------

def _click(args: dict, ctx: GroundingContext, *, count: int = 1) -> CompiledAction:
    target = resolve_target(args, ctx)
    button = "right" if args.get("button") == "right" else "left"
    return CompiledAction(
        key="click" if count == 1 else "double_click",
        commands=[{"kind": "click", "x": target.x, "y": target.y,
                   "button": button, "count": count}],
        point=(target.x, target.y),
        box_px=target.box_px,
    )


def _hover(args: dict, ctx: GroundingContext) -> CompiledAction:
    target = resolve_target(args, ctx)
    return CompiledAction(
        key="hover",
        commands=[{"kind": "hover", "x": target.x, "y": target.y}],
        point=(target.x, target.y),
        box_px=target.box_px,
    )


def _scroll(args: dict, ctx: GroundingContext) -> CompiledAction:
    try:
        dx = int(args.get("dx") or 0)
        dy = int(args.get("dy") or 0)
    except (TypeError, ValueError):
        raise ActionError("`dx`/`dy` must be integers (pixels)") from None
    if abs(dx) < 80 and abs(dy) < 80:
        raise ActionError("scroll too small: use |dy| or |dx| of at least 80 pixels")
    command: dict = {"kind": "scroll", "dx": dx, "dy": dy}
    anchor = resolve_optional_target(args, ctx)
    if anchor is not None:
        command.update({"x": anchor.x, "y": anchor.y})
    return CompiledAction(
        key="scroll",
        commands=[command],
        point=(anchor.x, anchor.y) if anchor else None,
    )


def _drag(args: dict, ctx: GroundingContext) -> CompiledAction:
    source = args.get("from") or args.get("start")
    dest = args.get("to") or args.get("end")
    if source is None or dest is None:
        raise ActionError('`drag` needs `from` and `to`, each a target in the standard format')

    def endpoint(value) -> dict:
        if isinstance(value, dict):
            return value
        if isinstance(value, (list, tuple)) and len(value) == 4:
            return {"box2d": value}
        return {"point": value}

    try:
        start = resolve_target(endpoint(source), ctx)
        finish = resolve_target(endpoint(dest), ctx)
    except GroundingError as err:
        raise ActionError(f"drag endpoints invalid: {err}") from None
    return CompiledAction(
        key="drag",
        commands=[{"kind": "drag", "x1": start.x, "y1": start.y,
                   "x2": finish.x, "y2": finish.y}],
        point=(start.x, start.y),
        drag_to=(finish.x, finish.y),
    )


def _type(args: dict, ctx: GroundingContext, *, clear: bool) -> CompiledAction:
    text = _require_text(args, "text" if "text" in args else "value")
    command: dict = {
        "kind": "type",
        "text": text,
        "clear": bool(args.get("clear", clear)),
        "enter": bool(args.get("press_enter") or args.get("enter")),
    }
    target = resolve_optional_target(args, ctx)
    if target is not None:
        command.update({"x": target.x, "y": target.y})
    return CompiledAction(
        key="fill" if clear else "type",
        commands=[command],
        point=(target.x, target.y) if target else None,
        box_px=target.box_px if target else None,
    )


def _clear(args: dict, ctx: GroundingContext) -> CompiledAction:
    # empty a field via the service's platform-correct select-all + delete
    # (reuses the type handler's clear path with no text to type)
    command: dict = {"kind": "type", "text": "", "clear": True, "enter": False}
    target = resolve_optional_target(args, ctx)
    if target is not None:
        command.update({"x": target.x, "y": target.y})
    return CompiledAction(
        key="clear", commands=[command],
        point=(target.x, target.y) if target else None,
        box_px=target.box_px if target else None,
    )


def _hotkey(args: dict, _ctx: GroundingContext) -> CompiledAction:
    keys = args.get("keys") or args.get("key")
    if isinstance(keys, str):
        keys = [keys]
    if not isinstance(keys, list) or not keys or not all(isinstance(k, str) for k in keys):
        raise ActionError('`keys` is required: a combo string like "Control+A" or a list of combos')
    if len(keys) > 24:
        raise ActionError("at most 24 key combos per hotkey action")
    command: dict = {"kind": "press", "keys": keys}
    if args.get("repeat") is not None:
        try:
            command["repeat"] = min(max(int(args["repeat"]), 1), 60)
        except (TypeError, ValueError):
            raise ActionError("`repeat` must be an integer") from None
    return CompiledAction(key="hotkey", commands=[command])


def _wait(args: dict, _ctx: GroundingContext) -> CompiledAction:
    try:
        seconds = float(args.get("seconds", 2))
    except (TypeError, ValueError):
        raise ActionError("`seconds` must be a number") from None
    ms = int(min(max(seconds, 0.1), 8.0) * 1000)
    return CompiledAction(key="wait", commands=[{"kind": "wait", "ms": ms}])


def _stop(args: dict, _ctx: GroundingContext) -> CompiledAction:
    answer = args.get("answer")
    return CompiledAction(
        key="stop", commands=[],
        stop_answer=str(answer) if answer is not None else None,
    )


def _goto(args: dict, _ctx: GroundingContext) -> CompiledAction:
    url = _require_text(args, "url")
    return CompiledAction(key="goto", commands=[], goto_url=url)


def _back(_args: dict, _ctx: GroundingContext) -> CompiledAction:
    return CompiledAction(key="go_back", commands=[{"kind": "back"}])


def _forward(_args: dict, _ctx: GroundingContext) -> CompiledAction:
    return CompiledAction(key="go_forward", commands=[{"kind": "forward"}])


def _select_option(args: dict, ctx: GroundingContext) -> CompiledAction:
    target = resolve_target(args, ctx)
    command: dict = {"kind": "select", "x": target.x, "y": target.y}
    if "label" in args:
        command["label"] = str(args["label"])
    elif "value" in args:
        command["value"] = str(args["value"])
    else:
        raise ActionError(
            "`select_option` needs `label` (visible option text); element IDs and "
            "option indices are not supported"
        )
    return CompiledAction(
        key="select_option", commands=[command],
        point=(target.x, target.y), box_px=target.box_px,
    )


def _set_checked(args: dict, ctx: GroundingContext) -> CompiledAction:
    target = resolve_target(args, ctx)
    return CompiledAction(
        key="set_checked",
        commands=[{"kind": "check", "x": target.x, "y": target.y,
                   "checked": bool(args.get("checked", True))}],
        point=(target.x, target.y), box_px=target.box_px,
    )


# --- documentation (rendered into the system prompt) ------------------------

def _doc(template: str) -> Callable[[GroundingContext], str]:
    def render(ctx: GroundingContext) -> str:
        return template.format(target=ctx.scheme.doc_target,
                               example=ctx.scheme.doc_example)
    return render


_SPECS: dict[str, ActionSpec] = {
    spec.key: spec for spec in [
        ActionSpec("click", _doc(
            '- `click` — click an element. args: {target}; optional "button": "right".'
        ), lambda a, c: _click(a, c, count=1)),
        ActionSpec("double_click", _doc(
            '- `double_click` — double-click an element. args: {target}.'
        ), lambda a, c: _click(a, c, count=2)),
        ActionSpec("hover", _doc(
            '- `hover` — move the mouse over an element (reveals menus/tooltips). args: {target}.'
        ), _hover),
        ActionSpec("scroll", _doc(
            '- `scroll` — wheel-scroll the page or a pane. args: "dy" pixels (positive = down), '
            'optional "dx"; optional {target} to scroll a specific pane. Use at least 80 px.'
        ), _scroll),
        ActionSpec("drag", _doc(
            '- `drag` — press at one target and release at another (sliders, range selects, '
            'spreadsheet cells). args: "from" and "to", each in the same format as {target}.'
        ), _drag),
        ActionSpec("type", _doc(
            '- `type` — type text at the current focus, or click a field first. args: "text"; '
            'optional {target}; optional "press_enter": true to submit; optional "clear": true '
            'to erase the existing value first.'
        ), lambda a, c: _type(a, c, clear=False)),
        ActionSpec("fill", _doc(
            '- `fill` — replace a field\'s value: clears it, then types. args: {target}, "text"; '
            'optional "press_enter": true. Use this when the field already has wrong content.'
        ), lambda a, c: _type(a, c, clear=True)),
        ActionSpec("clear", _doc(
            '- `clear` — empty a text field without typing anything. args: optional {target} '
            'to focus the field first. Use it when you need to remove existing content before '
            'a separate `type`.'
        ), _clear),
        ActionSpec("hotkey", _doc(
            '- `hotkey` — press keyboard shortcuts. args: "keys": one combo string such as '
            '"Control+A", "Backspace", "ArrowDown", "Enter", or a list of combos pressed in '
            'order; optional "repeat": press the sequence N times (e.g. Backspace x10).'
        ), _hotkey),
        ActionSpec("wait", _doc(
            '- `wait` — pause for slow-loading content. args: "seconds" (max 8).'
        ), _wait),
        ActionSpec("stop", _doc(
            '- `stop` — finish the episode. args: "answer": your final answer or a summary of '
            'what was accomplished. Use it as soon as the task is complete or clearly impossible.'
        ), _stop),
        ActionSpec("goto", _doc(
            '- `goto` — navigate only to an exact URL already provided or visibly observed. '
            'args: "url". Never guess paths, article slugs, filenames, query strings, or '
            'filter parameters; click visible links and search results instead.'
        ), _goto),
        ActionSpec("go_back", _doc(
            '- `go_back` — browser history back. args: none.'
        ), _back),
        ActionSpec("go_forward", _doc(
            '- `go_forward` — browser history forward. args: none.'
        ), _forward),
        ActionSpec("select_option", _doc(
            '- `select_option` — pick an option from a dropdown. args: {target} on the dropdown, '
            '"label": the visible text of the option to select.'
        ), _select_option),
        ActionSpec("set_checked", _doc(
            '- `set_checked` — set a checkbox or radio state. args: {target}, '
            '"checked": true or false.'
        ), _set_checked),
    ]
}

PROFILES: dict[str, tuple[str, ...]] = {
    "gui": ("click", "double_click", "hover", "scroll", "drag", "type", "fill",
            "clear", "hotkey", "wait", "stop"),
    "hybrid": ("click", "double_click", "hover", "scroll", "drag", "type", "fill",
               "clear", "hotkey", "wait", "goto", "go_back", "go_forward",
               "select_option", "set_checked", "stop"),
}


def catalog(profile: str, ctx: GroundingContext) -> str:
    """Render the action list for the system prompt."""
    try:
        keys = PROFILES[profile]
    except KeyError:
        raise ValueError(f"unknown action profile {profile!r}") from None
    return "\n".join(_SPECS[key].describe(ctx) for key in keys)


def compile_action(parsed: ParsedAction, ctx: GroundingContext, profile: str) -> CompiledAction:
    keys = PROFILES.get(profile) or PROFILES["hybrid"]
    if parsed.key not in keys:
        allowed = ", ".join(keys)
        raise ActionError(f"unknown action `{parsed.key}`; choose one of: {allowed}")
    try:
        return _SPECS[parsed.key].compile(parsed.args, ctx)
    except GroundingError as err:
        raise ActionError(str(err)) from None
