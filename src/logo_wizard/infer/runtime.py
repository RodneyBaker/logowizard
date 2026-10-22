"""Model backends for the app: ONNX Runtime first, PyTorch if installed, else none.

The packaged app ships ONNX Runtime but not PyTorch, so it uses the files
from logo-wizard-export-onnx in local/models/. A development install with
PyTorch can use the training checkpoints in local/runs/ directly.
"""

import json
import math
from pathlib import Path

import numpy as np

from .. import paths
from ..models.layout.tokens import BOS, MAX_TEXT, PAD, SpecVocab, encode_text
from ..render.spec import DesignSpec

MODELS_DIR = paths.MODELS_DIR
RUNS = paths.RUNS_DIR
STEPS = 1000


def _providers(gpu: bool) -> list[str]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    wanted = ["CUDAExecutionProvider", "DmlExecutionProvider"] if gpu else []
    return [p for p in wanted if p in available] + ["CPUExecutionProvider"]


def allowed_ids(vocab: SpecVocab, fonts: set[str] | None, palettes: set[str] | None) -> np.ndarray:
    """Output ids the sampler may use: fonts and palettes this machine lacks are switched off."""
    allowed = np.ones(vocab.size, dtype=bool)
    for name, names in (("font", fonts), ("palette", palettes), ("subtitle_font", fonts)):
        if names is None:
            continue
        i = vocab.index[name]
        for k, value in enumerate(vocab.fields[i].values):
            if value != "none" and value not in names:
                allowed[vocab.offsets[i] + k] = False
    return allowed


class OnnxLayout:
    name = "layout model (ONNX)"

    def __init__(self, model_dir: Path, fonts: set[str] | None = None, palettes: set[str] | None = None):
        import onnxruntime as ort

        meta = json.loads((model_dir / "layout.json").read_text(encoding="utf-8"))
        self.vocab = SpecVocab.from_dict(meta["vocab"])
        self.session = ort.InferenceSession(str(model_dir / "layout.onnx"), providers=["CPUExecutionProvider"])
        self.allowed = allowed_ids(self.vocab, fonts, palettes)

    def suggest(self, text: str, words: str = "", subtitle: str = "", n: int = 8, temperature: float = 1.0, top_p: float = 0.95, seed: int | None = None) -> list[DesignSpec]:
        rng = np.random.default_rng(seed)
        ids = encode_text(text, words)
        src = np.full((n, MAX_TEXT), PAD, dtype=np.int64)
        src[:, : len(ids)] = ids
        tgt = np.zeros((n, self.vocab.length), dtype=np.int64)
        tgt[:, 0] = BOS
        out = np.zeros((n, self.vocab.length), dtype=np.int64)
        for step in range(self.vocab.length):
            logits = self.session.run(None, {"src": src, "tgt": tgt})[0][:, step].astype(np.float64)
            logits[:, ~self.allowed] = -np.inf
            logits /= max(temperature, 1e-3)
            probs = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)
            if top_p < 1.0:
                order = np.argsort(-probs, axis=1)
                sorted_p = np.take_along_axis(probs, order, axis=1)
                sorted_p[np.cumsum(sorted_p, axis=1) - sorted_p > top_p] = 0.0
                probs = np.zeros_like(probs)
                np.put_along_axis(probs, order, sorted_p, axis=1)
                probs /= probs.sum(axis=1, keepdims=True)
            choice = np.array([rng.choice(len(p), p=p) for p in probs])
            out[:, step] = choice
            if step + 1 < self.vocab.length:
                tgt[:, step + 1] = choice
        seeds = rng.integers(0, 2**31 - 1, size=n)
        return [self.vocab.decode(row.tolist(), text, subtitle, seed=int(s)) for row, s in zip(out, seeds)]


class TorchLayout:
    name = "layout model (PyTorch)"

    def __init__(self, weights: Path, fonts: set[str] | None = None, palettes: set[str] | None = None):
        from ..models.layout.predict import LayoutPredictor

        self.predictor = LayoutPredictor(weights, fonts, palettes)

    def suggest(self, *args, **kwargs) -> list[DesignSpec]:
        return self.predictor.suggest(*args, **kwargs)


def cosine_alphas(steps: int = STEPS) -> np.ndarray:
    s = 0.008
    t = np.linspace(0, steps, steps + 1) / steps
    f = np.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    return np.clip(f / f[0], 1e-5, 1.0)[1:].astype(np.float32)


class OnnxStylizer:
    name = "stylizer (ONNX)"

    def __init__(self, model_dir: Path, gpu: bool = True):
        import onnxruntime as ort

        meta = json.loads((model_dir / "stylizer.json").read_text(encoding="utf-8"))
        self.palettes = meta["palettes"]
        self.size = tuple(meta["size"])
        self.session = ort.InferenceSession(str(model_dir / "stylizer.onnx"), providers=_providers(gpu))
        self.alpha_bar = cosine_alphas()

    def sample(self, mask: np.ndarray, labels: np.ndarray, steps: int = 25, guidance: float = 2.0, seed: int = 0) -> np.ndarray:
        """mask (B,1,H,W) in [-1,1], labels (B,F) int64 -> premultiplied RGBA (B,4,H,W) in [-1,1]."""
        rng = np.random.default_rng(seed)
        b, _, h, w = mask.shape
        x = rng.standard_normal((b, 4, h, w)).astype(np.float32)
        times = np.linspace(STEPS - 1, 0, steps).astype(np.int64)
        null = np.zeros_like(labels)
        for i, t in enumerate(times):
            feed = {
                "x": np.concatenate([x, x]),
                "t": np.full(2 * b, t, dtype=np.int64),
                "mask": np.concatenate([mask, mask]).astype(np.float32),
                "labels": np.concatenate([labels, null]).astype(np.int64),
            }
            eps_c, eps_u = np.split(self.session.run(None, feed)[0], 2)
            eps = eps_u + guidance * (eps_c - eps_u)
            a = self.alpha_bar[t]
            a_prev = self.alpha_bar[times[i + 1]] if i + 1 < len(times) else np.float32(1.0)
            x0 = np.clip((x - np.sqrt(1 - a) * eps) / np.sqrt(a), -1, 1)
            x = np.sqrt(a_prev) * x0 + np.sqrt(max(1 - a_prev, 0.0)) * eps
        return x


def load_layout(fonts: set[str] | None, palettes: set[str] | None):
    """The best layout backend available here, or None (the app then uses the grammar)."""
    if (MODELS_DIR / "layout.onnx").exists():
        try:
            return OnnxLayout(MODELS_DIR, fonts, palettes)
        except ImportError:
            pass
    weights = RUNS / "layout" / "best.pt"
    if weights.exists():
        try:
            return TorchLayout(weights, fonts, palettes)
        except ImportError:
            pass
    return None


def load_stylizer(renderer, gpu: bool = True):
    """A callable stylize(specs, out_size) or None when no stylizer is trained."""
    from ..models.stylizer.labels import spec_labels, to_rgba_np
    from .upscale import upscale

    if (MODELS_DIR / "stylizer.onnx").exists():
        try:
            backend = OnnxStylizer(MODELS_DIR, gpu)
        except ImportError:
            backend = None
        if backend is not None:
            def stylize(specs, out_size, steps=25, guidance=2.0, seed=0):
                w, h = backend.size
                masks = np.stack([renderer.render(s, w, h).mask.astype(np.float32)[None] / 127.5 - 1 for s in specs])
                labels = np.array([spec_labels(s, backend.palettes) for s in specs], dtype=np.int64)
                images = to_rgba_np(backend.sample(masks, labels, steps, guidance, seed))
                return [upscale(img, out_size, renderer.render(s, *out_size).mask) for s, img in zip(specs, images)]

            stylize.name = backend.name
            return stylize
    weights = RUNS / "stylizer" / "ema.pt"
    if weights.exists():
        try:
            from .stylize import Stylizer
        except ImportError:
            return None
        model = Stylizer(weights, renderer)

        def stylize(specs, out_size, steps=25, guidance=2.0, seed=0):
            return model.stylize(specs, out_size, steps=steps, guidance=guidance, seed=seed)

        stylize.name = "stylizer (PyTorch)"
        return stylize
    return None
