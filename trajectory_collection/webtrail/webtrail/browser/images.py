"""Small image helpers: perceptual hashing, blank-page detection, resizing.

Pure PIL, no numpy: the images involved are tiny (hash grids) or resized once.
"""

from __future__ import annotations

import io

from PIL import Image, ImageStat


def load_png(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def dhash(image: Image.Image, hash_size: int = 8) -> int:
    """64-bit difference hash; robust to mild rendering noise."""
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    # Pillow 14 removes getdata(); keep compatibility with the older versions
    # allowed by pyproject while preferring its replacement when available.
    data = (gray.get_flattened_data() if hasattr(gray, "get_flattened_data")
            else gray.getdata())
    pixels = list(data)
    bits = 0
    for row in range(hash_size):
        offset = row * (hash_size + 1)
        for col in range(hash_size):
            bits = (bits << 1) | (pixels[offset + col] > pixels[offset + col + 1])
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def near_uniform(image: Image.Image, stddev_threshold: float = 4.0) -> bool:
    """True for essentially blank captures (all-white, all-black, solid fills)."""
    stat = ImageStat.Stat(image.convert("L").resize((64, 64)))
    return stat.stddev[0] < stddev_threshold


def fit_max_side(image: Image.Image, max_side: int) -> Image.Image:
    """Downscale so the longer side is at most max_side; never upscale."""
    if max_side <= 0 or max(image.size) <= max_side:
        return image
    scale = max_side / max(image.size)
    new_size = (round(image.width * scale), round(image.height * scale))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def fit_size(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Downscale to an exact size selected by a model capability adapter."""
    if image.size == size:
        return image
    if size[0] > image.width or size[1] > image.height:
        raise ValueError(f"fit_size never upscales: {image.size} -> {size}")
    return image.resize(size, Image.Resampling.LANCZOS)


def to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def to_jpeg_bytes(image: Image.Image, quality: int = 85) -> bytes:
    """Encode an RGB screenshot as an optimized JPEG for model transport."""
    buffer = io.BytesIO()
    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=max(1, min(int(quality), 100)),
        optimize=True,
    )
    return buffer.getvalue()
