"""The procedural title grammar: hand-written rules that turn text into specs.

This is option A from the design plan. It is the app's fallback when no
model is trained, and it is the generator of the synthetic dataset the
layout model learns from. Moods are read from the title and any style
words, then steer font, palette, warp and effects, so the dataset carries a
real link between words and looks for the model to learn.
"""

import re

import numpy as np

from .shaping import FaceInfo
from .spec import DesignSpec, Palette

MOOD_WORDS = {
    "horror": """blood bloody curse cursed ghost ghosts grave graves dead death doom haunted haunting horror terror
        vampire witch witches zombie crypt tomb skull bones demon devil hell night nightmare fear dread creature
        monster monsters beast fiend shadow shadows gothic macabre dark darkness corpse coffin werewolf phantom
        evil wicked unholy scream""",
    "scifi": """star stars space galaxy galactic planet planets robot robots atomic atom rocket rockets comet cosmic
        future mars moon lunar alien aliens orbit nova quantum cyber laser ray rays astro astronaut android
        starship nebula meteor solar void time machine electric science sci-fi scifi tomorrow""",
    "western": """gun guns gunfighter outlaw outlaws ranch frontier sheriff trail desert canyon colt rider riders
        rustler cowboy cowboys saddle stagecoach mesa prairie bandit bandits gold rush wild west western
        dusty lone ranger mustang badlands bounty""",
    "fantasy": """dragon dragons king kings queen sword swords kingdom crown quest wizard magic realm throne legend
        legends enchanted sorcerer sorceress knight knights castle elf elves dwarf giant giants myth prophecy
        oracle temple babylon empire saga fable rune runes""",
    "elegant": """love heart hearts rose roses bride lady duchess secret secrets garden gardens romance beloved
        silk pearl diamond moonlight letters summer winter spring autumn waltz society manor""",
    "pulp": """detective murder crime case mystery mysteries danger spy spies agent agents gangster mob killer
        thrills thrilling adventure adventures action daring perilous racket city noir dame mask masked""",
    "playful": """fun funny kid kids silly tiny little puppy kitten candy bubble bubbles giggle jolly happy
        goofy comics cartoon""",
    "retro": """amazing astonishing incredible thrilling marvel wonder wonders golden atomic""",
}
STYLE_WORDS = {
    "comic": "comic comics superhero hero heroes cartoon pulp",
    "novel": "novel book gothic romance literary classic victorian",
}
_MOOD_LOOKUP = {w: mood for mood, words in MOOD_WORDS.items() for w in words.split() + [mood]}
# How much a font or palette tagged with the title's mood is favoured. Strong enough that
# most picks fit the mood (and the layout model has a clear link to learn), with room for surprises.
MOOD_BOOST = 12.0
_STYLE_LOOKUP = {w: style for style, words in STYLE_WORDS.items() for w in words.split()}


def read_moods(text: str, style_words: str = "") -> list[str]:
    words = re.findall(r"[a-z\-]+", (text + " " + style_words).lower())
    found = []
    for w in words:
        mood = _MOOD_LOOKUP.get(w) or _MOOD_LOOKUP.get(w.rstrip("s"))
        if mood and mood not in found:
            found.append(mood)
    if "gothic" in words and "horror" not in found:
        found.append("horror")
    return found


def read_style(style_words: str, default: str = "comic") -> str:
    for w in re.findall(r"[a-z\-]+", style_words.lower()):
        if w in _STYLE_LOOKUP:
            return _STYLE_LOOKUP[w]
    return default


def _choice(rng: np.random.Generator, options, weights=None):
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()
    return options[int(rng.choice(len(options), p=weights))]


def _score(tags: tuple[str, ...], style: str, moods: list[str]) -> float:
    score = 1.0 if style in tags else 0.15
    for m in moods:
        if m in tags:
            score *= MOOD_BOOST
    return score


def pick_font(faces: dict[str, FaceInfo], style: str, moods: list[str], rng: np.random.Generator, heavy: bool | None = None) -> FaceInfo:
    options = list(faces.values())
    weights = []
    for f in options:
        w = _score(f.moods, style, moods)
        if style == "comic" and f.font_class in ("script", "hand") and not moods:
            w *= 0.5
        if heavy is True:
            w *= 1.6 if f.weight >= 600 or f.font_class == "display" else 0.6
        weights.append(w)
    return _choice(rng, options, weights)


def pick_palette(palettes: dict[str, Palette], style: str, moods: list[str], rng: np.random.Generator) -> Palette:
    options = list(palettes.values())
    weights = [_score(p.moods + (p.family,), style, moods) * (1.0 if p.family == style else 0.1) for p in options]
    return _choice(rng, options, weights)


def _comic(spec: DesignSpec, rng: np.random.Generator, n_words: int, n_chars: int) -> None:
    long_title = n_chars > 16 or n_words > 3
    spec.arrangement = "stacked" if (long_title and rng.random() < 0.85) or rng.random() < 0.3 or n_chars > 22 else "straight"
    spec.warp = _choice(rng, ["none", "arch", "valley", "bulge", "wave", "perspective", "fan", "stagger"], [0.18, 0.22, 0.04, 0.18, 0.07, 0.12, 0.1, 0.09])
    spec.warp_amount = round(float(rng.uniform(0.3, 1.0)), 2)
    spec.tilt = round(float(rng.normal(0, 4)) if rng.random() < 0.5 else 0.0, 1)
    spec.case = "upper" if rng.random() < 0.85 else "title"
    spec.tracking = round(float(rng.uniform(-0.02, 0.06)), 3)
    spec.fill_mode = _choice(rng, ["flat", "gradient", "foil"], [0.2, 0.65, 0.15])
    n_outlines = _choice(rng, [1, 2, 3], [0.25, 0.45, 0.3])
    spec.outlines = [round(float(rng.uniform(2.5, 7)), 1) for _ in range(n_outlines)]
    if rng.random() < 0.6:
        spec.extrude_depth = round(float(rng.uniform(6, 24)), 1)
        spec.extrude_angle = float(_choice(rng, [45, 60, 90, 120, 135]))
    if rng.random() < 0.5:
        spec.shadow = round(float(rng.uniform(3, 10)), 1)
        spec.shadow_angle = spec.extrude_angle if spec.extrude_depth else float(_choice(rng, [45, 60, 90, 135]))
    spec.halftone = round(float(rng.uniform(0.4, 1.0)), 2) if rng.random() < 0.4 else 0.0
    spec.inline = bool(rng.random() < 0.35)
    spec.texture = _choice(rng, ["none", "distressed", "aged"], [0.85, 0.1, 0.05])


def _novel(spec: DesignSpec, rng: np.random.Generator, n_words: int, n_chars: int) -> None:
    long_title = n_chars > 14 or n_words > 2
    spec.arrangement = "stacked" if (long_title and rng.random() < 0.8) or rng.random() < 0.25 or n_chars > 22 else "straight"
    spec.warp = _choice(rng, ["none", "arch", "valley", "bulge", "wave", "perspective", "fan", "stagger"], [0.68, 0.16, 0.05, 0.03, 0.02, 0.04, 0.0, 0.02])
    spec.warp_amount = round(float(rng.uniform(0.15, 0.5)), 2)
    spec.tilt = 0.0
    spec.case = "upper" if rng.random() < 0.65 else "title"
    spec.tracking = round(float(rng.uniform(0.04, 0.3)) if spec.case == "upper" else float(rng.uniform(0.0, 0.06)), 3)
    spec.fill_mode = _choice(rng, ["flat", "gradient", "foil"], [0.35, 0.3, 0.35])
    spec.outlines = [round(float(rng.uniform(0.8, 2.5)), 1)] if rng.random() < 0.45 else []
    if rng.random() < 0.15:
        spec.extrude_depth = round(float(rng.uniform(2, 6)), 1)
        spec.extrude_angle = float(_choice(rng, [45, 90, 135]))
    if rng.random() < 0.55:
        spec.shadow = round(float(rng.uniform(2, 6)), 1)
        spec.shadow_angle = float(_choice(rng, [45, 90, 135]))
    spec.halftone = 0.0
    spec.inline = bool(rng.random() < 0.15)
    spec.texture = _choice(rng, ["none", "distressed", "aged"], [0.65, 0.15, 0.2])


def sample_spec(
    text: str,
    faces: dict[str, FaceInfo],
    palettes: dict[str, Palette],
    rng: np.random.Generator,
    style: str | None = None,
    style_words: str = "",
    subtitle: str = "",
) -> DesignSpec:
    style = style or read_style(style_words)
    moods = read_moods(text, style_words)
    words = text.split()
    spec = DesignSpec(text=text, style=style, subtitle=subtitle, seed=int(rng.integers(0, 2**31 - 1)))
    if style == "comic":
        _comic(spec, rng, len(words), len(text))
    else:
        _novel(spec, rng, len(words), len(text))
    font = pick_font(faces, style, moods, rng, heavy=True if style == "comic" else None)
    spec.font = font.face_id
    spec.palette = pick_palette(palettes, style, moods, rng).name
    if subtitle:
        plain = {k: v for k, v in faces.items() if v.font_class in ("serif", "sans", "slab")}
        spec.subtitle_font = pick_font(plain or faces, "novel", [], rng).face_id
    if font.font_class in ("script", "blackletter") or (font.font_class == "hand" and rng.random() < 0.6):
        # Capitals in a script or blackletter face are close to unreadable.
        spec.case = "title"
        spec.tracking = min(spec.tracking, 0.03)
    return spec
