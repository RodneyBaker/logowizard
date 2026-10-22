"""Rasterize a design spec into an RGBA logo and its glyph mask.

Layers, bottom to top: drop shadow, 3D extrusion, outlines (outermost
first), letter fill, inline highlight, halftone dots, then texture. The
background is transparent. The glyph mask is the plain letter shapes, which
is what the stylizer is conditioned on.
"""

import math
from dataclasses import dataclass

import numpy as np
import skia
from PIL import Image

from .geometry import Block, build_subtitle, build_title
from .shaping import FONT_SIZE, FaceInfo, get_font, load_manifest
from .spec import DesignSpec, Palette, hex_to_rgb, load_palettes

MARGIN = 0.06


@dataclass
class RenderResult:
    image: np.ndarray  # H x W x 4 uint8, straight alpha
    mask: np.ndarray  # H x W uint8, plain letters
    spec: DesignSpec

    def to_pil(self) -> Image.Image:
        return Image.fromarray(self.image, "RGBA")

    def mask_pil(self) -> Image.Image:
        return Image.fromarray(self.mask, "L")


def _color(hex_color: str, alpha: int = 255) -> int:
    r, g, b = hex_to_rgb(hex_color)
    return skia.Color(r, g, b, alpha)


def _mix(a: str, b: str, t: float) -> str:
    ra, ga, ba = hex_to_rgb(a)
    rb, gb, bb = hex_to_rgb(b)
    return "#%02X%02X%02X" % (round(ra + (rb - ra) * t), round(ga + (gb - ga) * t), round(ba + (bb - ba) * t))


def _path(polys: list[np.ndarray]) -> skia.Path:
    path = skia.Path()
    path.setFillType(skia.PathFillType.kWinding)
    for poly in polys:
        path.addPoly([skia.Point(float(x), float(y)) for x, y in poly], True)
    return path


def _stroke_fill(color: int, radius: float, blur: float = 0.0) -> skia.Paint:
    paint = skia.Paint(AntiAlias=True, Color=color)
    if radius > 0:
        paint.setStyle(skia.Paint.kStrokeAndFill_Style)
        paint.setStrokeWidth(radius * 2)
        paint.setStrokeJoin(skia.Paint.kRound_Join)
    if blur > 0:
        paint.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, blur))
    return paint


def _fill_paint(spec: DesignSpec, palette: Palette, bounds: tuple[float, float, float, float]) -> skia.Paint:
    paint = skia.Paint(AntiAlias=True)
    colors = list(palette.fill)
    x0, y0, x1, y1 = bounds
    if spec.fill_mode == "flat" or len(colors) == 1:
        paint.setColor(_color(colors[0]))
        return paint
    if spec.fill_mode == "foil":
        light, mid = colors[0], colors[1]
        dark = colors[2] if len(colors) > 2 else _mix(mid, "#000000", 0.35)
        stops = [light, mid, light, dark, mid, light]
        positions = [0.0, 0.3, 0.48, 0.55, 0.8, 1.0]
    else:
        stops = colors
        positions = [i / (len(colors) - 1) for i in range(len(colors))]
    shader = skia.GradientShader.MakeLinear(
        [skia.Point(x0, y0), skia.Point(x0, y1)], [_color(c) for c in stops], positions
    )
    paint.setShader(shader)
    return paint


def _offset(angle: float, distance: float) -> tuple[float, float]:
    a = math.radians(angle)
    return math.cos(a) * distance, math.sin(a) * distance


def _block_bounds(blocks: list[Block]) -> tuple[float, float, float, float]:
    boxes = [b.bounds for b in blocks if b.polys]
    return min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)


def _snapshot(surface: skia.Surface) -> np.ndarray:
    return surface.makeImageSnapshot().toarray(colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kUnpremul_AlphaType)


def _smooth_noise(rng: np.random.Generator, h: int, w: int, cell: int) -> np.ndarray:
    small = rng.random((max(2, h // cell), max(2, w // cell))).astype(np.float32)
    img = Image.fromarray((small * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def _halftone(rgba: np.ndarray, letters: np.ndarray, color: str, strength: float, box: tuple[int, int], spacing: float) -> None:
    h, w = letters.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Dot grid rotated 45 degrees, dots growing toward the bottom of the title.
    u = (xx + yy) / math.sqrt(2) / spacing
    v = (xx - yy) / math.sqrt(2) / spacing
    d = np.hypot(u - np.round(u), v - np.round(v)) * spacing
    top, bottom = box
    t = np.clip((yy - top) / max(bottom - top, 1), 0, 1)
    radius = spacing * 0.5 * strength * (0.25 + 0.75 * t)
    dots = np.clip(radius - d + 0.5, 0, 1) * (letters / 255.0) * 0.55
    rgb = np.array(hex_to_rgb(color), dtype=np.float32)
    rgba[..., :3] = rgba[..., :3] * (1 - dots[..., None]) + rgb * dots[..., None]


def _texture(rgba: np.ndarray, kind: str, rng: np.random.Generator, scale: float) -> None:
    h, w = rgba.shape[:2]
    if kind == "distressed":
        fine = _smooth_noise(rng, h, w, max(2, int(scale * 1.5)))
        coarse = _smooth_noise(rng, h, w, max(4, int(scale * 9)))
        specks = np.clip((fine * 0.6 + coarse * 0.4 - 0.62) * 9, 0, 1)
        rgba[..., 3] = rgba[..., 3] * (1 - specks)
    elif kind == "aged":
        stain = _smooth_noise(rng, h, w, max(4, int(scale * 12)))
        grain = rng.random((h, w)).astype(np.float32)
        shade = 1 - 0.22 * stain - 0.08 * grain
        sepia = np.array([1.0, 0.93, 0.8], dtype=np.float32)
        rgba[..., :3] = rgba[..., :3] * shade[..., None] * sepia
        wear = np.clip((_smooth_noise(rng, h, w, max(2, int(scale * 2))) - 0.8) * 6, 0, 1)
        rgba[..., 3] = rgba[..., 3] * (1 - 0.7 * wear)


class Renderer:
    """Holds the font manifest and palettes; renders specs at any size."""

    def __init__(self, faces: dict[str, FaceInfo] | None = None, palettes: dict[str, Palette] | None = None):
        self.faces = faces if faces is not None else load_manifest()
        self.palettes = palettes if palettes is not None else load_palettes()
        if not self.faces:
            raise RuntimeError("No fonts found. Run logo-wizard-fonts first to download them into local/fonts.")

    def render(self, spec: DesignSpec, width: int = 512, height: int = 512) -> RenderResult:
        rng = np.random.default_rng(spec.seed)
        palette = self.palettes.get(spec.palette) or next(iter(self.palettes.values()))
        font = get_font(self.faces.get(spec.font) or next(iter(self.faces.values())))
        sub_font = get_font(self.faces.get(spec.subtitle_font) or font.info)

        title = build_title(spec, font, rng)
        outline_total = sum(spec.outlines)
        subtitle = build_subtitle(spec.subtitle, sub_font, title, gap=outline_total + spec.extrude_depth + 0.18 * FONT_SIZE)
        blocks = [title] + ([subtitle] if subtitle.polys else [])

        # Fit everything, including the reach of outlines, extrusion and shadow.
        x0, y0, x1, y1 = _block_bounds(blocks)
        ex, ey = _offset(spec.extrude_angle, spec.extrude_depth)
        sx, sy = _offset(spec.shadow_angle, spec.shadow)
        pad = outline_total + 2
        reach_x = (min(0, ex + sx) - pad, max(0, ex + sx) + pad)
        reach_y = (min(0, ey + sy) - pad, max(0, ey + sy) + pad)
        bx0, bx1 = x0 + reach_x[0], x1 + reach_x[1]
        by0, by1 = y0 + reach_y[0], y1 + reach_y[1]
        avail_w, avail_h = width * (1 - 2 * MARGIN), height * (1 - 2 * MARGIN)
        scale = min(avail_w / (bx1 - bx0), avail_h / (by1 - by0))
        tx = (width - (bx1 - bx0) * scale) / 2 - bx0 * scale
        ty = (height - (by1 - by0) * scale) / 2 - by0 * scale
        matrix = skia.Matrix.MakeAll(scale, 0, tx, 0, scale, ty, 0, 0, 1)

        letters = _path([p for b in blocks for p in b.polys])
        info = skia.ImageInfo.Make(width, height, skia.kRGBA_8888_ColorType, skia.kPremul_AlphaType)
        surface = skia.Surface.MakeRaster(info)
        canvas = surface.getCanvas()
        canvas.clear(skia.ColorTRANSPARENT)
        canvas.concat(matrix)

        outer = palette.outlines[min(len(spec.outlines), len(palette.outlines)) - 1] if spec.outlines else palette.extrude
        if spec.shadow > 0:
            blur = 0.0 if spec.style == "comic" else spec.shadow * 0.35
            canvas.save()
            canvas.translate(ex + sx, ey + sy)
            canvas.drawPath(letters, _stroke_fill(_color(palette.shadow, 200), outline_total, blur))
            canvas.restore()

        if spec.extrude_depth > 0:
            steps = max(2, int(spec.extrude_depth * scale))
            edge = max(1.2 / scale, outline_total * 0.15)
            for pass_color, extra in ((None, edge), (palette.extrude, 0.0)):
                for k in range(steps, 0, -1):
                    t = k / steps
                    dx, dy = ex * t, ey * t
                    color = outer if pass_color is None else _mix(pass_color, "#000000", 0.45 * t)
                    canvas.save()
                    canvas.translate(dx, dy)
                    canvas.drawPath(letters, _stroke_fill(_color(color), outline_total + extra))
                    canvas.restore()

        radius = outline_total
        for i in range(len(spec.outlines) - 1, -1, -1):
            color = palette.outlines[i % len(palette.outlines)]
            canvas.drawPath(letters, _stroke_fill(_color(color), radius))
            radius -= spec.outlines[i]

        canvas.drawPath(letters, _fill_paint(spec, palette, title.bounds))

        if spec.inline:
            width_units = max(0.9 / scale, 1.6)
            canvas.save()
            canvas.clipPath(letters, skia.ClipOp.kIntersect, True)
            canvas.translate(-width_units, -width_units)
            paint = skia.Paint(AntiAlias=True, Color=_color(palette.inline, 230), Style=skia.Paint.kStroke_Style, StrokeWidth=width_units)
            canvas.drawPath(letters, paint)
            canvas.restore()

        rgba = _snapshot(surface).astype(np.float32)

        mask_surface = skia.Surface.MakeRaster(info)
        mask_canvas = mask_surface.getCanvas()
        mask_canvas.clear(skia.ColorTRANSPARENT)
        mask_canvas.concat(matrix)
        mask_canvas.drawPath(letters, skia.Paint(AntiAlias=True, Color=skia.ColorWHITE))
        mask = _snapshot(mask_surface)[..., 3].copy()

        if spec.halftone > 0:
            top = title.bounds[1] * scale + ty
            bottom = title.bounds[3] * scale + ty
            _halftone(rgba, mask.astype(np.float32), palette.halftone, spec.halftone, (int(top), int(bottom)), spacing=max(3.0, 5.5 * scale))
        if spec.texture != "none":
            _texture(rgba, spec.texture, rng, scale * 3)

        image = np.clip(rgba, 0, 255).astype(np.uint8)
        return RenderResult(image=image, mask=mask, spec=spec)


def preview_background(image: np.ndarray) -> tuple[int, int, int]:
    """Light grey behind dark logos, dark slate behind light ones."""
    alpha = image[..., 3].astype(np.float32) / 255.0
    if alpha.sum() < 1:
        return (235, 237, 240)
    lum = (image[..., :3].astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)) / 255.0
    mean = float((lum * alpha).sum() / alpha.sum())
    return (235, 237, 240) if mean < 0.6 else (38, 42, 52)


def on_background(image: np.ndarray, color: tuple[int, int, int] | None = None) -> Image.Image:
    """Flatten an RGBA logo onto a solid color for previews and contact sheets."""
    color = color or preview_background(image)
    base = Image.new("RGBA", (image.shape[1], image.shape[0]), color + (255,))
    base.alpha_composite(Image.fromarray(image, "RGBA"))
    return base.convert("RGB")
