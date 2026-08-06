#!/usr/bin/env python3
"""Generate the mock GUI screenshots for the bundled example trace.

The example task is "Turn on Dark Mode in the Settings app" on a fictional
desktop. Four screens are drawn: desktop, Settings menu, Display page with
the toggle off, and the same page in dark colors with the toggle on.
The PNGs are committed with the repo; rerun this script only to regenerate.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).resolve().parent / "screenshots"
W, H = 1000, 625


def _font(size: int):
    for path in ("/System/Library/Fonts/Helvetica.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_TITLE, F_BODY, F_SMALL = _font(28), _font(22), _font(16)


def _window(draw: ImageDraw.ImageDraw, title: str, bg: str, bar: str, fg: str):
    draw.rectangle([0, 0, W, H], fill=bg)
    draw.rectangle([0, 0, W, 48], fill=bar)
    for i, c in enumerate(("#ff5f57", "#febc2e", "#28c840")):
        draw.ellipse([16 + i * 28, 16, 32 + i * 28, 32], fill=c)
    draw.text((W // 2, 24), title, font=F_BODY, fill=fg, anchor="mm")


def screen_desktop() -> Image.Image:
    img = Image.new("RGB", (W, H), "#3a6ea5")
    draw = ImageDraw.Draw(img)
    draw.text((W // 2, 60), "ExampleOS Desktop", font=F_TITLE, fill="white", anchor="mm")
    apps = ["Files", "Browser", "Mail", "Settings", "Music", "Photos"]
    for i, name in enumerate(apps):
        x = 150 + (i % 3) * 250
        y = 180 + (i // 3) * 200
        draw.rounded_rectangle([x - 45, y - 45, x + 45, y + 45], radius=16,
                               fill="#e8e8e8" if name != "Settings" else "#c0c8d8")
        draw.text((x, y), name[0], font=F_TITLE, fill="#444444", anchor="mm")
        draw.text((x, y + 70), name, font=F_BODY, fill="white", anchor="mm")
    return img


def screen_settings_menu() -> Image.Image:
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    _window(draw, "Settings", "white", "#dddddd", "#333333")
    items = ["Network", "Sound", "Display", "Notifications", "Privacy"]
    for i, name in enumerate(items):
        y = 100 + i * 90
        draw.rounded_rectangle([80, y, 920, y + 70], radius=10,
                               outline="#cccccc", width=2)
        draw.text((110, y + 35), name, font=F_BODY, fill="#333333", anchor="lm")
        draw.text((890, y + 35), ">", font=F_BODY, fill="#999999", anchor="rm")
    return img


def screen_display(dark: bool) -> Image.Image:
    bg = "#1e1e1e" if dark else "white"
    fg = "#eeeeee" if dark else "#333333"
    bar = "#111111" if dark else "#dddddd"
    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)
    _window(draw, "Settings — Display", bg, bar, fg)
    draw.text((80, 90), "< Back", font=F_BODY, fill="#4a90d9", anchor="lm")

    draw.text((80, 170), "Brightness", font=F_BODY, fill=fg, anchor="lm")
    draw.rounded_rectangle([400, 162, 900, 178], radius=8, fill="#bbbbbb")
    draw.ellipse([690, 155, 715, 180], fill="#4a90d9")

    draw.text((80, 280), "Dark Mode", font=F_BODY, fill=fg, anchor="lm")
    tx0, ty0, tx1, ty1 = 820, 262, 900, 298
    if dark:
        draw.rounded_rectangle([tx0, ty0, tx1, ty1], radius=18, fill="#28c840")
        draw.ellipse([tx1 - 34, ty0 + 2, tx1 - 2, ty1 - 2], fill="white")
        draw.text((80, 330), "On", font=F_SMALL, fill="#28c840", anchor="lm")
    else:
        draw.rounded_rectangle([tx0, ty0, tx1, ty1], radius=18, fill="#cccccc")
        draw.ellipse([tx0 + 2, ty0 + 2, tx0 + 34, ty1 - 2], fill="white")
        draw.text((80, 330), "Off", font=F_SMALL, fill="#999999", anchor="lm")

    draw.text((80, 420), "Resolution        1920 x 1080", font=F_BODY, fill=fg, anchor="lm")
    draw.text((80, 490), "Night Light       Off", font=F_BODY, fill=fg, anchor="lm")
    return img


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    screens = [
        screen_desktop(),          # step 0: click the Settings icon
        screen_settings_menu(),    # step 1: click the Display row
        screen_display(False),     # step 2: click the Dark Mode toggle
        screen_display(True),      # step 3: final state, toggle on
    ]
    for i, img in enumerate(screens):
        path = OUT_DIR / f"step_{i:04d}.png"
        img.save(path, optimize=True)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
