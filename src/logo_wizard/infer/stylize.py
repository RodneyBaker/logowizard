"""Run the trained stylizer on a design spec, and upscale the result.

The renderer draws the plain glyph mask at the stylizer's size, the
diffusion model paints the finished logo around and inside it, and the
output is upscaled to the export size. The letter edges are then re-cut
from a full-size mask so the shapes stay sharp after upscaling.
"""

from pathlib import Path

import numpy as np
import torch

from .. import paths
from ..models.stylizer.diffusion import ddim_sample, spec_labels, to_rgba
from ..models.stylizer.unet import UNet, UNetConfig
from ..render.draw import Renderer
from ..render.spec import DesignSpec
from .upscale import upscale

DEFAULT_WEIGHTS = paths.RUNS_DIR / "stylizer" / "ema.pt"


class Stylizer:
    def __init__(self, path: Path = DEFAULT_WEIGHTS, renderer: Renderer | None = None, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        data = torch.load(path, map_location=self.device, weights_only=False)
        config = UNetConfig(**data["config"])
        config.checkpointing = False
        self.model = UNet(config).to(self.device).eval()
        self.model.load_state_dict(data["state"])
        self.palettes = data["palettes"]
        self.size = tuple(data.get("size", (128, 80)))
        self.renderer = renderer or Renderer()

    def stylize(self, specs: list[DesignSpec], out_size: tuple[int, int] | None = None, steps: int = 25, guidance: float = 2.0, seed: int = 0) -> list[np.ndarray]:
        w, h = self.size
        masks, labels = [], []
        for spec in specs:
            mask = self.renderer.render(spec, w, h).mask
            masks.append(mask.astype(np.float32)[None] / 127.5 - 1)
            labels.append(spec_labels(spec, self.palettes))
        m = torch.from_numpy(np.stack(masks)).to(self.device)
        lab = torch.tensor(labels, dtype=torch.long, device=self.device)
        gen = torch.Generator(device=self.device).manual_seed(seed)
        images = to_rgba(ddim_sample(self.model, m, lab, steps=steps, guidance=guidance, generator=gen))
        if out_size is None or out_size == (w, h):
            return list(images)
        out = []
        for spec, image in zip(specs, images):
            sharp = self.renderer.render(spec, *out_size).mask
            out.append(upscale(image, out_size, sharp))
        return out
