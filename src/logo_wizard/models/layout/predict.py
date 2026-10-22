"""Use a trained layout model to propose design specs for a title."""

from pathlib import Path

import numpy as np
import torch

from ... import paths
from ...render.spec import DesignSpec
from .model import LayoutModel
from .tokens import encode_text

DEFAULT_WEIGHTS = paths.RUNS_DIR / "layout" / "best.pt"


class LayoutPredictor:
    def __init__(self, path: Path = DEFAULT_WEIGHTS, installed_fonts: set[str] | None = None, palettes: set[str] | None = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = LayoutModel.load(path, map_location=self.device).to(self.device)
        vocab = self.model.vocab
        allowed = np.ones(vocab.size, dtype=bool)
        # Never propose a font or palette this machine does not have.
        for name, names in (("font", installed_fonts), ("palette", palettes), ("subtitle_font", installed_fonts)):
            if names is None:
                continue
            i = vocab.index[name]
            for k, value in enumerate(vocab.fields[i].values):
                if value != "none" and value not in names:
                    allowed[vocab.offsets[i] + k] = False
        self.allowed = torch.from_numpy(allowed).to(self.device)

    def suggest(self, text: str, words: str = "", subtitle: str = "", n: int = 8, temperature: float = 1.0, top_p: float = 0.95, seed: int | None = None) -> list[DesignSpec]:
        src = torch.tensor([encode_text(text, words)] * n, dtype=torch.long, device=self.device)
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed if seed is not None else int(np.random.default_rng().integers(2**31)))
        ids = self.model.sample(src, temperature=temperature, top_p=top_p, allowed=self.allowed, generator=gen)
        seeds = np.random.default_rng(seed).integers(0, 2**31 - 1, size=n)
        return [self.model.vocab.decode(row.tolist(), text, subtitle, seed=int(s)) for row, s in zip(ids.cpu(), seeds)]
