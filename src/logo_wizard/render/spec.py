"""The design spec: everything the renderer needs to draw one title logo.

The layout model predicts these fields and the procedural grammar samples
them; both hand the same spec to the renderer. Lengths are layout units in
which the main font size is 100, so a spec looks the same at any pixel size.
Colors come from the named palette, so a spec stays small and discrete.
"""

import json
from dataclasses import asdict, dataclass, field, fields
from functools import lru_cache
from pathlib import Path

import yaml

from .. import paths

PALETTE_FILE = paths.PALETTES

STYLES = ("comic", "novel")
CASES = ("upper", "title")
ARRANGEMENTS = ("straight", "stacked")
WARPS = ("none", "arch", "valley", "bulge", "wave", "perspective", "fan", "stagger")
FILL_MODES = ("flat", "gradient", "foil")
TEXTURES = ("none", "distressed", "aged")


@dataclass
class DesignSpec:
    text: str
    style: str = "comic"
    subtitle: str = ""
    font: str = ""
    subtitle_font: str = ""
    palette: str = "classic-hero"
    case: str = "upper"
    arrangement: str = "straight"
    warp: str = "none"
    warp_amount: float = 0.0  # 0..1
    tracking: float = 0.0  # extra space between letters, as a fraction of the font size
    tilt: float = 0.0  # degrees, whole logo
    fill_mode: str = "gradient"
    outlines: list[float] = field(default_factory=list)  # stroke widths, innermost first
    extrude_depth: float = 0.0
    extrude_angle: float = 45.0  # degrees, 90 is straight down
    shadow: float = 0.0  # offset distance
    shadow_angle: float = 45.0
    halftone: float = 0.0  # 0..1 strength of the dot overlay
    inline: bool = False  # thin highlight line inside the letters
    texture: str = "none"
    seed: int = 0  # drives per-letter jitter and texture noise

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "DesignSpec":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_json(cls, text: str) -> "DesignSpec":
        return cls.from_dict(json.loads(text))


@dataclass(frozen=True)
class Palette:
    name: str
    family: str
    moods: tuple[str, ...]
    fill: tuple[str, ...]
    outlines: tuple[str, ...]
    extrude: str
    shadow: str
    halftone: str
    inline: str


@lru_cache(maxsize=4)
def load_palettes(path: Path = PALETTE_FILE) -> dict[str, Palette]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    out = {}
    for family, entries in data.items():
        for e in entries:
            out[e["name"]] = Palette(
                name=e["name"],
                family=family,
                moods=tuple(" ".join(e["moods"]).split()),
                fill=tuple(e["fill"]),
                outlines=tuple(e["outlines"]),
                extrude=e["extrude"],
                shadow=e["shadow"],
                halftone=e["halftone"],
                inline=e["inline"],
            )
    return out


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
