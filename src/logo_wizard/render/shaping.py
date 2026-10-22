"""Shape text with HarfBuzz and turn the glyphs into flattened outlines.

Outlines come back as lists of polygons (numpy arrays of x, y points, y down)
in layout units where the font size is 100. Curves and long straight edges
are both subdivided, so the warps can bend every edge smoothly.
"""

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import uharfbuzz as hb

from .. import paths

FONT_DIR = paths.FONT_DIR
MANIFEST = paths.FONT_MANIFEST

FONT_SIZE = 100.0
CURVE_STEPS = 8
MAX_SEGMENT = 4.0  # layout units; longer straight edges are split


@dataclass(frozen=True)
class FaceInfo:
    face_id: str
    family: str
    file: str
    weight: int
    variation: str
    license: str
    font_class: str
    moods: tuple[str, ...]
    has_lower: bool


def load_manifest(path: Path = MANIFEST, font_dir: Path = FONT_DIR, require_files: bool = True) -> dict[str, FaceInfo]:
    faces = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if require_files and not (font_dir / row["file"]).exists():
                continue
            faces[row["face_id"]] = FaceInfo(
                face_id=row["face_id"],
                family=row["family"],
                file=row["file"],
                weight=int(row["weight"]),
                variation=row["variation"],
                license=row["license"],
                font_class=row["font_class"],
                moods=tuple(row["moods"].split()),
                has_lower=row["has_lower"] == "True",
            )
    return faces


class _FlattenPen:
    """fontTools-style pen that collects flattened contours."""

    def __init__(self, scale: float):
        self.scale = scale
        self.contours: list[list[tuple[float, float]]] = []
        self.current: list[tuple[float, float]] = []

    def _pt(self, x, y):
        return (x * self.scale, -y * self.scale)

    def _line(self, p):
        x0, y0 = self.current[-1]
        steps = max(1, int(np.hypot(p[0] - x0, p[1] - y0) / MAX_SEGMENT))
        for i in range(1, steps + 1):
            t = i / steps
            self.current.append((x0 + (p[0] - x0) * t, y0 + (p[1] - y0) * t))

    def moveTo(self, p):
        self._close()
        self.current = [self._pt(*p)]

    def lineTo(self, p):
        self._line(self._pt(*p))

    def qCurveTo(self, *points):
        pts = [self._pt(*p) for p in points]
        # TrueType implied on-curve points between consecutive off-curve points.
        start = self.current[-1]
        offs, end = pts[:-1], pts[-1]
        for i, ctrl in enumerate(offs):
            nxt = end if i == len(offs) - 1 else ((ctrl[0] + offs[i + 1][0]) / 2, (ctrl[1] + offs[i + 1][1]) / 2)
            for s in range(1, CURVE_STEPS + 1):
                t = s / CURVE_STEPS
                a = (1 - t) ** 2
                b = 2 * (1 - t) * t
                c = t * t
                self.current.append((a * start[0] + b * ctrl[0] + c * nxt[0], a * start[1] + b * ctrl[1] + c * nxt[1]))
            start = nxt
        if not offs:
            self._line(end)

    def curveTo(self, *points):
        p1, p2, p3 = [self._pt(*p) for p in points[-3:]]
        p0 = self.current[-1]
        for s in range(1, CURVE_STEPS + 1):
            t = s / CURVE_STEPS
            mt = 1 - t
            self.current.append(
                (
                    mt**3 * p0[0] + 3 * mt * mt * t * p1[0] + 3 * mt * t * t * p2[0] + t**3 * p3[0],
                    mt**3 * p0[1] + 3 * mt * mt * t * p1[1] + 3 * mt * t * t * p2[1] + t**3 * p3[1],
                )
            )

    def closePath(self):
        self._close()

    endPath = closePath

    def _close(self):
        if len(self.current) > 2:
            self.contours.append(self.current)
        self.current = []


class Font:
    def __init__(self, info: FaceInfo, font_dir: Path = FONT_DIR):
        self.info = info
        blob = hb.Blob.from_file_path(str(font_dir / info.file))
        self.face = hb.Face(blob)
        self.font = hb.Font(self.face)
        if info.variation:
            tag, value = info.variation.split("=")
            self.font.set_variations({tag: float(value)})
        self.scale = FONT_SIZE / self.face.upem
        extents = self.font.get_font_extents("ltr")
        self.ascender = extents.ascender * self.scale
        self.descender = -extents.descender * self.scale
        self._glyphs: dict[int, list[np.ndarray]] = {}
        self.cap_height = self._cap_height()

    def _cap_height(self) -> float:
        polys = self.glyph(self.font.get_nominal_glyph(ord("H")) or 0)
        if not polys:
            return 0.7 * FONT_SIZE
        return float(-min(p[:, 1].min() for p in polys))

    def glyph(self, gid: int) -> list[np.ndarray]:
        if gid not in self._glyphs:
            pen = _FlattenPen(self.scale)
            self.font.draw_glyph_with_pen(gid, pen)
            pen._close()
            self._glyphs[gid] = [np.asarray(c, dtype=np.float32) for c in pen.contours]
        return self._glyphs[gid]

    def shape(self, text: str, tracking: float = 0.0, kerning: bool = True) -> tuple[list[list[np.ndarray]], float]:
        """Glyph outlines of one line placed on a baseline at y=0, and the line's advance width.

        Each entry is one glyph's polygons, so per-letter effects can move them.
        """
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        features = {"kern": kerning, "liga": True}
        hb.shape(self.font, buf, features)
        x = 0.0
        glyphs = []
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
            polys = self.glyph(info.codepoint)
            dx = x + pos.x_offset * self.scale
            dy = -pos.y_offset * self.scale
            glyphs.append([p + np.array([dx, dy], dtype=np.float32) for p in polys])
            x += pos.x_advance * self.scale + tracking * FONT_SIZE
        if glyphs:
            x -= tracking * FONT_SIZE
        return glyphs, x


@lru_cache(maxsize=64)
def _cached_font(info: FaceInfo, font_dir: str) -> Font:
    return Font(info, Path(font_dir))


def get_font(info: FaceInfo, font_dir: Path = FONT_DIR) -> Font:
    return _cached_font(info, str(font_dir))
