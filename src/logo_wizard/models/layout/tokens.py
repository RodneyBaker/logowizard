"""Turn design specs into a fixed sequence of discrete tokens and back.

The layout model writes one token per field, always in the same order, and
at each step it may only pick among that field's values. So every output
decodes to a valid spec. Continuous fields are snapped to small sets of
bins; the bins cover the ranges the title grammar samples.
"""

from dataclasses import dataclass

import numpy as np

from ...render.grammar import MOOD_WORDS, read_moods, read_style
from ...render.spec import ARRANGEMENTS, CASES, FILL_MODES, STYLES, TEXTURES, WARPS, DesignSpec

PAD, BOS, EOS, UNK, SEP = 0, 1, 2, 3, 4
CHARS = [chr(c) for c in range(32, 127)]
CHAR_OFFSET = 5
MAX_TEXT = 112

# Bins for the continuous fields: (field, values).
BINS = {
    "warp_amount": [0.0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1.0],
    "tracking": [-0.02, 0.0, 0.02, 0.04, 0.07, 0.1, 0.14, 0.18, 0.23, 0.3],
    "tilt": [-10.0, -7.0, -4.0, -2.0, 0.0, 2.0, 4.0, 7.0, 10.0],
    "outline": [0.0, 0.8, 1.5, 2.5, 3.5, 4.5, 5.5, 7.0],
    "extrude_depth": [0.0, 2.0, 4.0, 6.0, 9.0, 12.0, 16.0, 20.0, 24.0],
    "angle": [45.0, 60.0, 90.0, 120.0, 135.0],
    "shadow": [0.0, 2.0, 3.0, 4.5, 6.0, 8.0, 10.0],
    "halftone": [0.0, 0.4, 0.55, 0.7, 0.85, 1.0],
}


def _nearest(values: list[float], x: float) -> int:
    return int(np.argmin([abs(v - x) for v in values]))


@dataclass
class Field:
    name: str
    values: list  # the value each choice stands for


class SpecVocab:
    """Field order and value lists. Built from the font manifest and palettes, saved with the model.

    tags maps a font or palette name to its descriptive tags (moods, font class,
    palette family). The model shares what it learns about a tag across every
    font or palette that carries it, instead of learning each one alone.
    """

    def __init__(self, fonts: list[str], palettes: list[str], tags: dict[str, list[str]] | None = None):
        self.fonts = list(fonts)
        self.palettes = list(palettes)
        self.tags = {k: list(v) for k, v in (tags or {}).items()}
        self.fields = [
            Field("style", list(STYLES)),
            Field("font", self.fonts),
            Field("palette", self.palettes),
            Field("case", list(CASES)),
            Field("arrangement", list(ARRANGEMENTS)),
            Field("warp", list(WARPS)),
            Field("warp_amount", BINS["warp_amount"]),
            Field("tracking", BINS["tracking"]),
            Field("tilt", BINS["tilt"]),
            Field("fill_mode", list(FILL_MODES)),
            Field("outline1", BINS["outline"]),
            Field("outline2", BINS["outline"]),
            Field("outline3", BINS["outline"]),
            Field("extrude_depth", BINS["extrude_depth"]),
            Field("extrude_angle", BINS["angle"]),
            Field("shadow", BINS["shadow"]),
            Field("shadow_angle", BINS["angle"]),
            Field("halftone", BINS["halftone"]),
            Field("inline", [False, True]),
            Field("texture", list(TEXTURES)),
            Field("subtitle_font", ["none"] + self.fonts),
        ]
        # Output ids: 0..2 are PAD/BOS/EOS, then each field's values in a block.
        self.offsets = []
        offset = 3
        for f in self.fields:
            self.offsets.append(offset)
            offset += len(f.values)
        self.size = offset
        self.index = {f.name: i for i, f in enumerate(self.fields)}

    @property
    def length(self) -> int:
        return len(self.fields)

    def to_dict(self) -> dict:
        return {"fonts": self.fonts, "palettes": self.palettes, "tags": self.tags}

    @classmethod
    def from_dict(cls, data: dict) -> "SpecVocab":
        return cls(data["fonts"], data["palettes"], data.get("tags"))

    def tag_matrix(self) -> np.ndarray:
        """(vocab, tags) 0/1: which tags each font and palette output id carries."""
        names = sorted({t for v in self.tags.values() for t in v})
        column = {t: j for j, t in enumerate(names)}
        matrix = np.zeros((self.size, len(names)), dtype=np.float32)
        for field in ("font", "palette", "subtitle_font"):
            i = self.index[field]
            for k, value in enumerate(self.fields[i].values):
                for t in self.tags.get(value, []):
                    matrix[self.offsets[i] + k, column[t]] = 1.0
        return matrix

    def field_mask(self) -> np.ndarray:
        """(fields, vocab) boolean: which output ids each step may emit."""
        mask = np.zeros((self.length, self.size), dtype=bool)
        for i, f in enumerate(self.fields):
            mask[i, self.offsets[i] : self.offsets[i] + len(f.values)] = True
        return mask

    def encode(self, spec: DesignSpec) -> list[int]:
        outlines = list(spec.outlines) + [0.0, 0.0, 0.0]
        choice = {
            "style": STYLES.index(spec.style) if spec.style in STYLES else 0,
            "font": self.fonts.index(spec.font) if spec.font in self.fonts else 0,
            "palette": self.palettes.index(spec.palette) if spec.palette in self.palettes else 0,
            "case": CASES.index(spec.case),
            "arrangement": ARRANGEMENTS.index(spec.arrangement),
            "warp": WARPS.index(spec.warp),
            "warp_amount": _nearest(BINS["warp_amount"], spec.warp_amount if spec.warp != "none" else 0.0),
            "tracking": _nearest(BINS["tracking"], spec.tracking),
            "tilt": _nearest(BINS["tilt"], spec.tilt),
            "fill_mode": FILL_MODES.index(spec.fill_mode),
            "outline1": _nearest(BINS["outline"], outlines[0]),
            "outline2": _nearest(BINS["outline"], outlines[1]),
            "outline3": _nearest(BINS["outline"], outlines[2]),
            "extrude_depth": _nearest(BINS["extrude_depth"], spec.extrude_depth),
            "extrude_angle": _nearest(BINS["angle"], spec.extrude_angle),
            "shadow": _nearest(BINS["shadow"], spec.shadow),
            "shadow_angle": _nearest(BINS["angle"], spec.shadow_angle),
            "halftone": _nearest(BINS["halftone"], spec.halftone),
            "inline": int(bool(spec.inline)),
            "texture": TEXTURES.index(spec.texture),
            "subtitle_font": 1 + self.fonts.index(spec.subtitle_font) if spec.subtitle_font in self.fonts else 0,
        }
        return [self.offsets[i] + choice[f.name] for i, f in enumerate(self.fields)]

    def decode(self, ids: list[int], text: str, subtitle: str = "", seed: int = 0) -> DesignSpec:
        v = {}
        for i, f in enumerate(self.fields):
            k = int(ids[i]) - self.offsets[i]
            v[f.name] = f.values[min(max(k, 0), len(f.values) - 1)]
        outlines = []
        for name in ("outline1", "outline2", "outline3"):
            if v[name] <= 0:
                break
            outlines.append(v[name])
        sub_font = v["subtitle_font"] if subtitle and v["subtitle_font"] != "none" else ""
        return DesignSpec(
            text=text,
            style=v["style"],
            subtitle=subtitle,
            font=v["font"],
            subtitle_font=sub_font,
            palette=v["palette"],
            case=v["case"],
            arrangement=v["arrangement"],
            warp=v["warp"],
            warp_amount=v["warp_amount"] if v["warp"] != "none" else 0.0,
            tracking=v["tracking"],
            tilt=v["tilt"],
            fill_mode=v["fill_mode"],
            outlines=outlines,
            extrude_depth=v["extrude_depth"],
            extrude_angle=v["extrude_angle"],
            shadow=v["shadow"],
            shadow_angle=v["shadow_angle"],
            halftone=v["halftone"],
            inline=bool(v["inline"]),
            texture=v["texture"],
            seed=seed,
        )


# Tags the title grammar reads from the words (moods like horror or western, and
# comic or novel). Giving them to the model as tokens makes the link from words to
# fonts and palettes easy to learn; the characters still carry everything else.
TAGS = sorted(MOOD_WORDS) + list(STYLES)
TAG_OFFSET = CHAR_OFFSET + len(CHARS)


def text_tags(text: str, words: str = "") -> list[str]:
    tags = read_moods(text, words)
    style = read_style(words, default="")
    return tags + [style] if style else tags


def encode_text(text: str, words: str = "", max_len: int = MAX_TEXT) -> list[int]:
    """Tags, then the characters of 'style words | title', as input ids: BOS ... EOS."""
    def chars(s: str) -> list[int]:
        return [CHAR_OFFSET + CHARS.index(c) if c in CHARS else UNK for c in s]

    tags = [TAG_OFFSET + TAGS.index(t) for t in text_tags(text, words) if t in TAGS]
    ids = [BOS] + tags + chars(words.lower().strip()) + [SEP] + chars(text.strip())
    return ids[: max_len - 1] + [EOS]


TEXT_VOCAB = TAG_OFFSET + len(TAGS)
