"""Upscale stylizer output to export size, keeping the letters sharp."""

import numpy as np
from PIL import Image, ImageFilter


def upscale(image: np.ndarray, size: tuple[int, int], sharp_mask: np.ndarray | None = None) -> np.ndarray:
    """Bicubic upscale with a light unsharp mask; letter interiors from a full-size mask stay fully opaque."""
    big = Image.fromarray(image, "RGBA").resize(size, Image.BICUBIC)
    rgb = big.convert("RGB").filter(ImageFilter.UnsharpMask(radius=2, percent=80, threshold=2))
    out = np.dstack([np.asarray(rgb), np.asarray(big)[..., 3]])
    if sharp_mask is not None:
        out[..., 3] = np.maximum(out[..., 3], sharp_mask)
    return out
