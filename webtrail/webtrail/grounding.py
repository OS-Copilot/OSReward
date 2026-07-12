"""Visual grounding: translate model-emitted coordinates into viewport pixels.

Different model families localize targets differently, so each supported
convention is a `Scheme`:

* ``box1000``  – bounding box ``[ymin, xmin, ymax, xmax]`` in a 0-1000 grid
                 (Gemini-style). The click point is the box center.
* ``point1000`` – point ``[x, y]`` in a 0-1000 grid (Qwen-style).
* ``pixel``    – point ``[x, y]`` in pixels *of the screenshot the model saw*
                 (Claude-style). If we downscaled the screenshot before sending,
                 the resolver maps coordinates back to the real viewport.

The resolver is deliberately forgiving about spellings (`box2d`/`box`/`bbox`,
`point`/`coordinate`, bare `x`/`y`) and about values given as 0-1 fractions.
"""

from __future__ import annotations

from dataclasses import dataclass


class GroundingError(ValueError):
    """Raised when an action's target cannot be resolved to a screen point."""


@dataclass(frozen=True)
class Scheme:
    id: str
    doc_target: str      # how the prompt names the target argument
    doc_convention: str  # coordinate convention paragraph for the system prompt
    doc_example: str     # example target argument in JSON


SCHEMES: dict[str, Scheme] = {
    "box1000": Scheme(
        id="box1000",
        doc_target='"box2d": [ymin, xmin, ymax, xmax]',
        doc_convention=(
            "Locate targets with a bounding box `box2d = [ymin, xmin, ymax, xmax]` "
            "of integers in [0, 1000], where (0, 0) is the top-left corner of the "
            "screenshot and (1000, 1000) the bottom-right. Draw the box tightly "
            "around the element; the click lands at the box center."
        ),
        doc_example='"box2d": [212, 80, 248, 310]',
    ),
    "point1000": Scheme(
        id="point1000",
        doc_target='"point": [x, y]',
        doc_convention=(
            "Locate targets with a point `point = [x, y]` of integers in [0, 1000], "
            "where x runs left-to-right and y top-to-bottom across the screenshot; "
            "(0, 0) is the top-left corner and (1000, 1000) the bottom-right. "
            "Aim at the center of the element."
        ),
        doc_example='"point": [145, 226]',
    ),
    "pixel": Scheme(
        id="pixel",
        doc_target='"point": [x, y]',
        doc_convention=(
            "Locate targets with a point `point = [x, y]` in pixels of the "
            "screenshot you are shown, x left-to-right and y top-to-bottom from "
            "the top-left corner. Aim at the center of the element."
        ),
        doc_example='"point": [640, 358]',
    ),
}

# Substring → default coordinate convention, matched in order against a
# lower-cased model id. Families that emit pixel coordinates of the screenshot
# they saw use "pixel"; the mapping falls back to box1000 (Gemini-style).
_MODEL_HINTS = (
    ("gemini", "box1000"),
    ("claude", "pixel"),
    ("gpt-5", "pixel"),
    ("gpt5", "pixel"),
    ("gpt-4o", "pixel"),
    ("computer-use", "pixel"),
    ("kimi", "pixel"),
    ("moonshot", "pixel"),
    ("k2", "pixel"),
    ("qwen", "point1000"),
    ("ui-tars", "point1000"),
    ("uitars", "point1000"),
)


def scheme_for_model(model_id: str, override: str = "auto") -> Scheme:
    if override and override != "auto":
        try:
            return SCHEMES[override]
        except KeyError:
            raise GroundingError(
                f"unknown grounding scheme {override!r}; pick one of {sorted(SCHEMES)}"
            ) from None
    lowered = (model_id or "").lower()
    for hint, scheme_id in _MODEL_HINTS:
        if hint in lowered:
            return SCHEMES[scheme_id]
    return SCHEMES["box1000"]


@dataclass(frozen=True)
class Resolved:
    x: int
    y: int
    box_px: tuple[int, int, int, int] | None = None   # viewport-pixel xyxy, if a box was given


@dataclass(frozen=True)
class GroundingContext:
    scheme: Scheme
    viewport: tuple[int, int]     # real browser viewport (w, h)
    sent_size: tuple[int, int]    # size of the screenshot actually sent to the model


def _floats(value, count: int) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != count:
        return None
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError):
        return None


def _clamp(value: float, upper: int) -> int:
    return int(min(max(round(value), 0), upper - 1))


def _axis_to_viewport(value: float, ctx: GroundingContext, axis: int) -> float:
    """Map one model coordinate onto the real viewport along axis 0=x / 1=y."""
    view = ctx.viewport[axis]
    if value <= 1.0:                       # fraction, scale-free
        return value * view
    if ctx.scheme.id == "pixel":           # pixels of the sent image
        sent = ctx.sent_size[axis] or view
        return value * view / sent
    return value / 1000.0 * view           # thousandths, scale-free


def resolve_target(args: dict, ctx: GroundingContext,
                   keys: tuple[str, ...] = ("box2d", "box", "bbox", "point", "coordinate")
                   ) -> Resolved:
    """Pull a target locator out of `args` and resolve it to viewport pixels."""
    for key in keys:
        if key not in args or args[key] is None:
            continue
        value = args[key]
        if key in ("box2d", "box", "bbox"):
            numbers = _floats(value, 4)
            if numbers is None:
                raise GroundingError(f"`{key}` must be [ymin, xmin, ymax, xmax] numbers")
            ymin, xmin, ymax, xmax = numbers
            x1 = _axis_to_viewport(min(xmin, xmax), ctx, 0)
            x2 = _axis_to_viewport(max(xmin, xmax), ctx, 0)
            y1 = _axis_to_viewport(min(ymin, ymax), ctx, 1)
            y2 = _axis_to_viewport(max(ymin, ymax), ctx, 1)
            width, height = ctx.viewport
            box = (_clamp(x1, width), _clamp(y1, height),
                   _clamp(x2, width), _clamp(y2, height))
            return Resolved(
                x=_clamp((x1 + x2) / 2, width),
                y=_clamp((y1 + y2) / 2, height),
                box_px=box,
            )
        numbers = _floats(value, 2)
        if numbers is None:
            raise GroundingError(f"`{key}` must be [x, y] numbers")
        x, y = numbers
        return Resolved(
            x=_clamp(_axis_to_viewport(x, ctx, 0), ctx.viewport[0]),
            y=_clamp(_axis_to_viewport(y, ctx, 1), ctx.viewport[1]),
        )

    if "x" in args and "y" in args:
        try:
            x, y = float(args["x"]), float(args["y"])
        except (TypeError, ValueError):
            raise GroundingError("`x`/`y` must be numbers") from None
        return Resolved(
            x=_clamp(_axis_to_viewport(x, ctx, 0), ctx.viewport[0]),
            y=_clamp(_axis_to_viewport(y, ctx, 1), ctx.viewport[1]),
        )

    # aliases some model families emit regardless of the prompted convention
    width, height = ctx.viewport
    for x_key, y_key, to_px in (
        ("x_rel", "y_rel", lambda v, size: v * size),          # 0-1 fraction
        ("x_norm", "y_norm", lambda v, size: v / 1000 * size if v > 1.5 else v * size),
        ("x_abs", "y_abs", lambda v, size: v),                 # viewport pixels
        ("x_px", "y_px", lambda v, size: v),
    ):
        if x_key in args and y_key in args:
            try:
                x = to_px(float(args[x_key]), width)
                y = to_px(float(args[y_key]), height)
            except (TypeError, ValueError):
                raise GroundingError(f"`{x_key}`/`{y_key}` must be numbers") from None
            return Resolved(x=_clamp(x, width), y=_clamp(y, height))

    raise GroundingError(
        f"no target given; provide {ctx.scheme.doc_target} (example: {ctx.scheme.doc_example})"
    )


def resolve_optional_target(args: dict, ctx: GroundingContext) -> Resolved | None:
    try:
        return resolve_target(args, ctx)
    except GroundingError:
        return None
