"""Quality filters for generated samples: contrast, coverage, OCR, duplicates."""

import difflib
import os
import re
import shutil

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from ..render.draw import on_background

OCR_TIMEOUT = 3  # seconds per Tesseract call
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _linear(rgb: np.ndarray) -> np.ndarray:
    c = rgb / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def relative_luminance(rgb: np.ndarray) -> float:
    return float(_linear(np.asarray(rgb, dtype=np.float32)) @ _LUMA)


def contrast_ratio(a: np.ndarray, b: np.ndarray) -> float:
    la, lb = relative_luminance(a), relative_luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def edge_contrast(image: np.ndarray, mask: np.ndarray, rings: tuple[int, ...] = (2, 5, 9)) -> float:
    """Best contrast between the letter fill and an outline band around the letters.

    Bands at a few distances are tried, so a pale inner outline wrapped in a
    dark outer one still counts as readable. With nothing drawn around the
    letters (a plain novel title), the logo will sit on a cover, so the best
    of a black or white backdrop counts.
    """
    inside = mask > 200
    if inside.sum() < 20:
        return 0.0
    fill = image[..., :3][inside].astype(np.float32).mean(axis=0)
    drawn = image[..., 3] > 200
    best = 0.0
    previous = mask > 60
    for ring in rings:
        grown = np.asarray(Image.fromarray(mask).filter(ImageFilter.MaxFilter(2 * ring + 1))) > 60
        band = grown & ~previous & drawn
        previous = grown
        if band.sum() < inside.sum() * 0.03:
            continue
        best = max(best, contrast_ratio(fill, image[..., :3][band].astype(np.float32).mean(axis=0)))
    if best == 0.0:
        return max(contrast_ratio(fill, np.array([0, 0, 0])), contrast_ratio(fill, np.array([255, 255, 255])))
    return best


def coverage(image: np.ndarray, mask: np.ndarray) -> float:
    """Share of letter pixels that are still drawn (textures knock some out)."""
    inside = mask > 200
    if not inside.any():
        return 0.0
    return float((image[..., 3][inside] > 100).mean())


def dhash(image: np.ndarray, size: int = 8) -> int:
    """64-bit difference hash of the logo flattened on grey; equal hashes are near-duplicates."""
    grey = on_background(image, (128, 128, 128)).convert("L").resize((size + 1, size), Image.BILINEAR)
    px = np.asarray(grey, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def tesseract_available() -> bool:
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return False
    return shutil.which("tesseract") is not None or _windows_tesseract() is not None


def _windows_tesseract() -> str | None:
    for path in (r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
        if os.path.exists(path):
            return path
    return None


def _normalize(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def ocr_score(image: np.ndarray, expected: str) -> float:
    """How well Tesseract reads the logo back, 0 to 1 (character similarity)."""
    import pytesseract

    exe = _windows_tesseract()
    if exe and not shutil.which("tesseract"):
        pytesseract.pytesseract.tesseract_cmd = exe
    target = _normalize(expected)
    if not target:
        return 1.0
    best = 0.0
    # Tesseract wants dark text on light: try the logo on its preview backdrop,
    # then on black with the result inverted, which suits outlined comic titles.
    pictures = [on_background(image), ImageOps.invert(on_background(image, (0, 0, 0)).convert("L"))]
    for picture in pictures:
        scale = max(1, 320 // max(1, picture.height))
        if scale > 1:
            picture = picture.resize((picture.width * scale, picture.height * scale), Image.BICUBIC)
        for psm in (6, 11):
            try:
                got = _normalize(pytesseract.image_to_string(picture, config=f"--psm {psm}", timeout=OCR_TIMEOUT))
            except RuntimeError:  # Tesseract can stall on busy textures; count it as unread
                continue
            best = max(best, difflib.SequenceMatcher(None, target, got).ratio())
            if best > 0.85:
                return best
    return best
