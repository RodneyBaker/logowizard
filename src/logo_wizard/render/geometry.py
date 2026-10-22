"""Lay the title out in lines, then bend the letters.

Everything here works on polygons in layout units (font size 100, y down).
The warps map every outline point, which is why shaping subdivides edges.
"""

import math
from dataclasses import dataclass

import numpy as np

from .shaping import FONT_SIZE, Font
from .spec import DesignSpec

SMALL_WORDS = {"a", "an", "and", "at", "by", "for", "in", "of", "on", "or", "the", "to", "with", "from", "vs", "versus", "&"}
LINE_GAP = 0.18  # fraction of the font size between stacked lines
SMALL_SCALE = 0.42  # connector lines like "of the" in a stacked title
SUBTITLE_SCALE = 0.3


@dataclass
class Block:
    polys: list[np.ndarray]
    bounds: tuple[float, float, float, float]  # x0, y0, x1, y1


def apply_case(text: str, case: str, font: Font) -> str:
    if case == "upper" or not font.info.has_lower:
        return text.upper()
    return text


def split_lines(text: str, arrangement: str, max_chars: int = 14) -> list[tuple[str, float]]:
    """Split a title into (line, relative size). Connector words get small lines when stacked."""
    words = text.split()
    if arrangement != "stacked" or len(words) < 2:
        return [(text, 1.0)]
    runs: list[tuple[list[str], bool]] = []
    for word in words:
        small = word.lower().strip(",:") in SMALL_WORDS
        if runs and runs[-1][1] == small:
            runs[-1][0].append(word)
        else:
            runs.append(([word], small))
    lines: list[tuple[str, float]] = []
    for run, small in runs:
        if small:
            lines.append((" ".join(run), SMALL_SCALE))
            continue
        joined = " ".join(run)
        n_lines = max(math.ceil(len(joined) / max_chars), 2 if len(runs) == 1 and len(run) > 1 else 1)
        target = len(joined) / n_lines + 2
        current: list[str] = []
        for word in run:
            if current and len(" ".join(current + [word])) > target:
                lines.append((" ".join(current), 1.0))
                current = []
            current.append(word)
        lines.append((" ".join(current), 1.0))
    # A small run at the very end ("Tales of") reads better merged back.
    if len(lines) > 1 and lines[-1][1] < 1.0:
        tail = lines.pop()
        lines[-1] = (lines[-1][0] + " " + tail[0], lines[-1][1])
    if all(size < 1.0 for _, size in lines):
        return [(text, 1.0)]
    return lines


def _bounds(polys: list[np.ndarray]) -> tuple[float, float, float, float]:
    if not polys:
        return (0.0, 0.0, 1.0, 1.0)
    pts = np.concatenate(polys)
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())


def _rotate(polys: list[np.ndarray], degrees: float, cx: float, cy: float) -> list[np.ndarray]:
    a = math.radians(degrees)
    rot = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]], dtype=np.float32)
    c = np.array([cx, cy], dtype=np.float32)
    return [(p - c) @ rot + c for p in polys]


def layout_lines(font: Font, lines: list[tuple[str, float]], tracking: float, extra_gap: float = 0.0) -> tuple[list[list[np.ndarray]], float]:
    """Centered lines stacked top to bottom. Returns per-glyph polygons and the total height."""
    shaped = []
    for text, scale in lines:
        glyphs, width = font.shape(text, tracking)
        shaped.append((glyphs, width, scale))
    widest = max(w * s for _, w, s in shaped) or 1.0
    y = 0.0
    out: list[list[np.ndarray]] = []
    cap = font.cap_height
    for i, (glyphs, width, scale) in enumerate(shaped):
        y += cap * scale
        dx = (widest - width * scale) / 2
        for glyph in glyphs:
            if glyph:
                out.append([p * scale + np.array([dx, y], dtype=np.float32) for p in glyph])
        if i < len(shaped) - 1:
            y += LINE_GAP * FONT_SIZE * max(scale, shaped[i + 1][2]) + extra_gap
    return out, y


def stagger(glyphs: list[list[np.ndarray]], amount: float, rng: np.random.Generator) -> list[list[np.ndarray]]:
    out = []
    for i, glyph in enumerate(glyphs):
        x0, y0, x1, y1 = _bounds(glyph)
        angle = rng.uniform(-14, 14) * amount
        lift = (1 if i % 2 else -1) * amount * 0.08 * FONT_SIZE + rng.uniform(-3, 3) * amount
        moved = _rotate(glyph, angle, (x0 + x1) / 2, (y0 + y1) / 2)
        out.append([p + np.array([0, lift], dtype=np.float32) for p in moved])
    return out


def warp_points(pts: np.ndarray, kind: str, amount: float, bounds: tuple[float, float, float, float]) -> np.ndarray:
    x0, y0, x1, y1 = bounds
    w = max(x1 - x0, 1e-3)
    h = max(y1 - y0, 1e-3)
    x = pts[:, 0] - x0
    y = pts[:, 1] - y0
    u = x / w
    v = y / h
    if kind == "arch" or kind == "valley":
        arc = max(amount, 0.02) * math.radians(110)
        radius = w / arc
        theta = (x - w / 2) / radius
        if kind == "arch":
            r = radius + (h - y)
            nx = w / 2 + r * np.sin(theta)
            ny = h + radius - r * np.cos(theta)
        else:
            r = radius + y
            nx = w / 2 + r * np.sin(theta)
            ny = -radius + r * np.cos(theta)
    elif kind == "bulge":
        s = 1 + amount * 0.7 * np.sin(np.pi * np.clip(u, 0, 1))
        nx, ny = x, h / 2 + (y - h / 2) * s
    elif kind == "wave":
        nx, ny = x, y + amount * h * 0.22 * np.sin(2 * np.pi * u)
    elif kind == "perspective":
        # Recedes toward a vanishing point above the title.
        vc = np.clip(v, 0, 1)
        shrink = 1 - amount * 0.3 * (1 - vc)
        nx = w / 2 + (x - w / 2) * shrink
        ny = h * (vc + amount * 0.35 * vc * (vc - 1)) + (v - vc) * h
    elif kind == "fan":
        # Grows from left to right, like a title rushing toward the reader.
        a = amount * 0.7
        s = (1 - a) + a * u
        # Letters widen with their height: x follows the integral of s.
        nx = w * ((1 - a) * u + a * u * u / 2) / (1 - a / 2)
        ny = h / 2 + (y - h / 2) * s / (1 - a / 2)
    else:
        return pts
    return np.stack([nx + x0, ny + y0], axis=1).astype(np.float32)


def build_title(spec: DesignSpec, font: Font, rng: np.random.Generator) -> Block:
    """Lines leave room for the outlines, so stacked lines never collide."""
    text = apply_case(spec.text, spec.case, font)
    lines = split_lines(text, spec.arrangement)
    glyphs, _ = layout_lines(font, lines, spec.tracking, extra_gap=sum(spec.outlines) * 1.6)
    if spec.warp == "stagger":
        glyphs = stagger(glyphs, spec.warp_amount, rng)
    polys = [p for glyph in glyphs for p in glyph]
    if spec.warp not in ("none", "stagger"):
        bounds = _bounds(polys)
        polys = [warp_points(p, spec.warp, spec.warp_amount, bounds) for p in polys]
    if spec.tilt:
        x0, y0, x1, y1 = _bounds(polys)
        polys = _rotate(polys, spec.tilt, (x0 + x1) / 2, (y0 + y1) / 2)
    return Block(polys, _bounds(polys))


def build_subtitle(text: str, font: Font, title: Block, gap: float) -> Block:
    if not text:
        return Block([], title.bounds)
    glyphs, width = font.shape(apply_case(text, "upper", font), 0.12)
    scale = SUBTITLE_SCALE
    tx0, _, tx1, ty1 = title.bounds
    dx = (tx0 + tx1) / 2 - width * scale / 2
    dy = ty1 + gap + font.cap_height * scale
    polys = [p * scale + np.array([dx, dy], dtype=np.float32) for glyph in glyphs for p in glyph]
    return Block(polys, _bounds(polys))
