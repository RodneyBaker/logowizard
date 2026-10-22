"""Diffusion schedule, spec labels and the DDIM sampler for the stylizer."""

import math

import numpy as np
import torch

from .labels import from_rgba, label_sizes, spec_labels, to_rgba_np  # noqa: F401  (re-exported)

STEPS = 1000


def cosine_alphas(steps: int = STEPS) -> torch.Tensor:
    """Cumulative signal level alpha-bar for the cosine schedule (Nichol and Dhariwal)."""
    s = 0.008
    t = torch.linspace(0, steps, steps + 1, dtype=torch.float64) / steps
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = (f / f[0]).clamp(1e-5, 1.0)
    return alpha_bar[1:].float()


def add_noise(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
    a = alpha_bar[t].view(-1, 1, 1, 1)
    return a.sqrt() * x0 + (1 - a).sqrt() * noise


@torch.no_grad()
def ddim_sample(model, mask: torch.Tensor, labels: torch.Tensor, steps: int = 25, guidance: float = 2.0, generator=None, eta: float = 0.0) -> torch.Tensor:
    """Generate logos for a batch of glyph masks. Returns premultiplied RGBA in [-1, 1]."""
    device = mask.device
    alpha_bar = cosine_alphas().to(device)
    b, _, h, w = mask.shape
    x = torch.randn((b, 4, h, w), generator=generator, device=device)
    times = torch.linspace(STEPS - 1, 0, steps, device=device).long()
    null = torch.zeros_like(labels)
    for i, t in enumerate(times):
        tt = t.repeat(b)
        if guidance != 1.0:
            eps_c, eps_u = model(torch.cat([x, x]), torch.cat([tt, tt]), torch.cat([mask, mask]), torch.cat([labels, null])).chunk(2)
            eps = eps_u + guidance * (eps_c - eps_u)
        else:
            eps = model(x, tt, mask, labels)
        a = alpha_bar[t]
        a_prev = alpha_bar[times[i + 1]] if i + 1 < len(times) else torch.tensor(1.0, device=device)
        x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)
        sigma = eta * ((1 - a_prev) / (1 - a) * (1 - a / a_prev)).sqrt()
        direction = (1 - a_prev - sigma**2).clamp_min(0).sqrt() * eps
        x = a_prev.sqrt() * x0 + direction
        if eta > 0 and i + 1 < len(times):
            x = x + sigma * torch.randn(x.shape, generator=generator, device=device)
    return x


def to_rgba(x: torch.Tensor) -> np.ndarray:
    """(B,4,H,W) premultiplied [-1,1] -> (B,H,W,4) uint8 straight alpha."""
    return to_rgba_np(x.detach().cpu().numpy())
