"""Text-to-spec transformer: a small encoder-decoder over characters and spec fields.

The encoder reads 'style words | title' as characters. The decoder writes
one token per spec field in a fixed order; logits outside the current
field's values are masked, so sampling can only produce valid specs.
"""

import math
import warnings
from dataclasses import asdict, dataclass

import torch
from torch import nn

from .tokens import BOS, PAD, TEXT_VOCAB, SpecVocab

# norm_first layers cannot use the nested-tensor fast path; nothing to fix.
warnings.filterwarnings("ignore", message="enable_nested_tensor is True")

PRESETS = {
    # about 17M parameters with the 356-face manifest; trains in under 2 GB of VRAM
    "base": dict(d_model=384, heads=8, enc_layers=4, dec_layers=4, ff=1536, dropout=0.1),
    "small": dict(d_model=256, heads=8, enc_layers=3, dec_layers=3, ff=1024, dropout=0.1),
    # CPU smoke tests
    "tiny": dict(d_model=64, heads=4, enc_layers=1, dec_layers=1, ff=128, dropout=0.0),
}


@dataclass
class LayoutConfig:
    d_model: int = 384
    heads: int = 8
    enc_layers: int = 4
    dec_layers: int = 4
    ff: int = 1536
    dropout: float = 0.1
    max_text: int = 128


class LayoutModel(nn.Module):
    def __init__(self, config: LayoutConfig, vocab: SpecVocab):
        super().__init__()
        self.config = config
        self.vocab = vocab
        d = config.d_model
        self.src_embed = nn.Embedding(TEXT_VOCAB, d, padding_idx=PAD)
        self.src_pos = nn.Embedding(config.max_text, d)
        self.tgt_embed = nn.Embedding(vocab.size, d)
        self.tgt_pos = nn.Embedding(vocab.length + 1, d)
        self.transformer = nn.Transformer(
            d_model=d,
            nhead=config.heads,
            num_encoder_layers=config.enc_layers,
            num_decoder_layers=config.dec_layers,
            dim_feedforward=config.ff,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.out = nn.Linear(d, vocab.size)
        # Font and palette logits also get a score per tag they carry (mood, class, family),
        # so seeing a few horror fonts teaches the model about all of them.
        tag_matrix = torch.from_numpy(vocab.tag_matrix())
        self.register_buffer("tag_matrix", tag_matrix, persistent=False)
        self.tag_out = nn.Linear(d, tag_matrix.shape[1], bias=False) if tag_matrix.shape[1] else None
        self.register_buffer("field_mask", torch.from_numpy(vocab.field_mask()), persistent=False)
        causal = torch.triu(torch.full((vocab.length, vocab.length), float("-inf")), diagonal=1)
        self.register_buffer("causal", causal, persistent=False)

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        scale = math.sqrt(self.config.d_model)
        pos = torch.arange(src.shape[1], device=src.device)
        return self.transformer.encoder(self.src_embed(src) * scale + self.src_pos(pos), src_key_padding_mask=src == PAD)

    def decode(self, memory: torch.Tensor, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        T = tgt_in.shape[1]
        pos = torch.arange(T, device=tgt_in.device)
        t = self.tgt_embed(tgt_in) * math.sqrt(self.config.d_model) + self.tgt_pos(pos)
        h = self.transformer.decoder(t, memory, tgt_mask=self.causal[:T, :T], memory_key_padding_mask=src == PAD)
        logits = self.out(h)
        if self.tag_out is not None:
            logits = logits + self.tag_out(h) @ self.tag_matrix.T
        return logits.masked_fill(~self.field_mask[:T], float("-inf"))

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        """src (B, S) char ids; tgt_in (B, T) = BOS + the first T-1 field tokens. Returns masked logits (B, T, V)."""
        return self.decode(self.encode(src), src, tgt_in)

    @torch.no_grad()
    def sample(
        self,
        src: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 1.0,
        allowed: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample one spec per row of src. allowed (V,) bool can forbid ids, e.g. fonts not installed.

        Nucleus sampling rather than top-k: the font field has hundreds of
        plausible values, and a fixed k would cut most of them off.
        """
        B = src.shape[0]
        tokens = torch.full((B, 1), BOS, dtype=torch.long, device=src.device)
        memory = self.encode(src)
        for _ in range(self.vocab.length):
            logits = self.decode(memory, src, tokens)[:, -1]
            if allowed is not None:
                logits = logits.masked_fill(~allowed, float("-inf"))
            if temperature <= 0:
                nxt = logits.argmax(-1)
            else:
                probs = torch.softmax(logits / temperature, -1)
                if top_p < 1.0:
                    # Nucleus sampling: keep the most likely values that cover top_p of the mass.
                    sorted_p, order = probs.sort(-1, descending=True)
                    drop = sorted_p.cumsum(-1) - sorted_p > top_p
                    probs = probs.scatter(-1, order, sorted_p.masked_fill(drop, 0.0))
                    probs = probs / probs.sum(-1, keepdim=True)
                nxt = torch.multinomial(probs, 1, generator=generator).squeeze(1)
            tokens = torch.cat([tokens, nxt[:, None]], dim=1)
        return tokens[:, 1:]

    def save(self, path, extra: dict | None = None) -> None:
        torch.save({"config": asdict(self.config), "vocab": self.vocab.to_dict(), "state": self.state_dict(), **(extra or {})}, path)

    @classmethod
    def load(cls, path, map_location="cpu") -> "LayoutModel":
        data = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(LayoutConfig(**data["config"]), SpecVocab.from_dict(data["vocab"]))
        model.load_state_dict(data["state"])
        model.eval()
        return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
