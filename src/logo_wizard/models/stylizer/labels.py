"""Stylizer conditioning labels and image conversions, without PyTorch.

Kept free of torch so the packaged app can run the ONNX stylizer.
"""

import numpy as np

from ...render.spec import FILL_MODES, STYLES, TEXTURES, DesignSpec

HALFTONE_BINS = [0.0, 0.4, 0.55, 0.7, 0.85, 1.0]
MAX_PALETTES = 63


def label_sizes() -> list[int]:
    """Number of ids per label field, counting 0 = unspecified."""
    return [len(STYLES) + 1, MAX_PALETTES + 1, len(FILL_MODES) + 1, len(TEXTURES) + 1, len(HALFTONE_BINS) + 1, 5, 3, 3]


def spec_labels(spec: DesignSpec, palettes: list[str]) -> list[int]:
    """Discrete look fields for conditioning. palettes is the ordered list saved with the model."""
    palette = palettes.index(spec.palette) + 1 if spec.palette in palettes else 0
    halftone = int(np.argmin([abs(b - spec.halftone) for b in HALFTONE_BINS])) + 1
    return [
        STYLES.index(spec.style) + 1 if spec.style in STYLES else 0,
        min(palette, MAX_PALETTES),
        FILL_MODES.index(spec.fill_mode) + 1,
        TEXTURES.index(spec.texture) + 1,
        halftone,
        min(len(spec.outlines), 3) + 1,
        int(spec.inline) + 1,
        int(spec.extrude_depth > 0) + 1,
    ]


def to_rgba_np(x: np.ndarray) -> np.ndarray:
    """(B,4,H,W) premultiplied [-1,1] -> (B,H,W,4) uint8 straight alpha."""
    x = ((np.clip(x, -1, 1) + 1) / 2).transpose(0, 2, 3, 1)
    alpha = x[..., 3:4]
    rgb = np.where(alpha > 1e-3, x[..., :3] / np.maximum(alpha, 1e-3), 0)
    return (np.concatenate([np.clip(rgb, 0, 1), alpha], axis=-1) * 255).round().astype(np.uint8)


def from_rgba(image: np.ndarray) -> np.ndarray:
    """(H,W,4) uint8 straight alpha -> (4,H,W) float32 premultiplied in [-1,1]."""
    x = image.astype(np.float32) / 255.0
    x[..., :3] *= x[..., 3:4]
    return (x * 2 - 1).transpose(2, 0, 1)
