"""Glyph-conditioned UNet for the stylizer.

Input: the noisy logo (4 channels, premultiplied RGBA in [-1, 1]) stacked
with the plain glyph mask (1 channel). The spec's discrete look fields
(style, palette, fill mode, texture, effects) are embedded and added to the
timestep embedding; index 0 of every field means "unspecified", which is
what classifier-free guidance drops to.
"""

import math
from dataclasses import asdict, dataclass, field

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .labels import label_sizes

PRESETS = {
    # about 36M parameters; 128x80 at batch 16 fits the GTX 1080 with checkpointing
    "base": dict(base=72, mults=(1, 2, 4, 4), res_blocks=2, attn_levels=(2, 3)),
    "small": dict(base=48, mults=(1, 2, 3, 4), res_blocks=1, attn_levels=(3,)),
    "tiny": dict(base=16, mults=(1, 2), res_blocks=1, attn_levels=()),
}


@dataclass
class UNetConfig:
    base: int = 64
    mults: tuple = (1, 2, 4, 4)
    res_blocks: int = 2
    attn_levels: tuple = (2, 3)
    in_channels: int = 5
    out_channels: int = 4
    dropout: float = 0.0
    cond_sizes: list = field(default_factory=label_sizes)
    checkpointing: bool = False

    @property
    def cond_fields(self) -> int:
        return len(self.cond_sizes)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["mults"] = list(self.mults)
        data["attn_levels"] = list(self.attn_levels)
        return data


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=1)


def norm(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(math.gcd(32, ch), ch)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, emb: int, dropout: float):
        super().__init__()
        self.n1 = norm(cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb, 2 * cout)
        self.n2 = norm(cout)
        self.drop = nn.Dropout(dropout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.emb(F.silu(e))[:, :, None, None].chunk(2, dim=1)
        h = self.n2(h) * (1 + scale) + shift
        h = self.c2(self.drop(F.silu(h)))
        return h + self.skip(x)


class Attention(nn.Module):
    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.n = norm(ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, e=None):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(b, 3, self.heads, c // self.heads, h * w).unbind(1)
        out = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        return x + self.proj(out.transpose(-1, -2).reshape(b, c, h, w))


class UNet(nn.Module):
    def __init__(self, config: UNetConfig):
        super().__init__()
        self.config = config
        ch = config.base
        emb = ch * 4
        self.time = nn.Sequential(nn.Linear(ch, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.cond = nn.ModuleList([nn.Embedding(n, emb) for n in config.cond_sizes])
        self.inp = nn.Conv2d(config.in_channels, ch, 3, padding=1)

        self.down = nn.ModuleList()
        skips = [ch]
        cur = ch
        for level, mult in enumerate(config.mults):
            for _ in range(config.res_blocks):
                blocks = [ResBlock(cur, ch * mult, emb, config.dropout)]
                cur = ch * mult
                if level in config.attn_levels:
                    blocks.append(Attention(cur))
                self.down.append(nn.ModuleList(blocks))
                skips.append(cur)
            if level < len(config.mults) - 1:
                self.down.append(nn.ModuleList([nn.Conv2d(cur, cur, 3, stride=2, padding=1)]))
                skips.append(cur)

        self.mid = nn.ModuleList([ResBlock(cur, cur, emb, config.dropout), Attention(cur), ResBlock(cur, cur, emb, config.dropout)])

        self.up = nn.ModuleList()
        for level, mult in reversed(list(enumerate(config.mults))):
            for i in range(config.res_blocks + 1):
                blocks = [ResBlock(cur + skips.pop(), ch * mult, emb, config.dropout)]
                cur = ch * mult
                if level in config.attn_levels:
                    blocks.append(Attention(cur))
                if level > 0 and i == config.res_blocks:
                    blocks.append(nn.Upsample(scale_factor=2, mode="nearest"))
                    blocks.append(nn.Conv2d(cur, cur, 3, padding=1))
                self.up.append(nn.ModuleList(blocks))

        self.out = nn.Sequential(norm(cur), nn.SiLU(), nn.Conv2d(cur, config.out_channels, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def _run(self, blocks, h, e):
        for block in blocks:
            if isinstance(block, (ResBlock, Attention)):
                if self.config.checkpointing and self.training:
                    h = checkpoint(block, h, e, use_reentrant=False)
                else:
                    h = block(h, e)
            else:
                h = block(h)
        return h

    def forward(self, x: torch.Tensor, t: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """x (B,4,H,W) noisy logo; t (B,) timesteps; mask (B,1,H,W); labels (B, cond_fields) ids. Returns predicted noise."""
        e = self.time(timestep_embedding(t, self.config.base))
        for i, embed in enumerate(self.cond):
            e = e + embed(labels[:, i])
        h = self.inp(torch.cat([x, mask], dim=1))
        hs = [h]
        for blocks in self.down:
            h = self._run(blocks, h, e)
            hs.append(h)
        h = self._run(self.mid, h, e)
        for blocks in self.up:
            h = torch.cat([h, hs.pop()], dim=1)
            h = self._run(blocks, h, e)
        return self.out(h)
