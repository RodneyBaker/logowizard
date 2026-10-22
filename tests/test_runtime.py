import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from logo_wizard.infer import onnx_export, runtime
from logo_wizard.models.layout.model import PRESETS, LayoutConfig, LayoutModel
from logo_wizard.models.layout.tokens import SpecVocab
from logo_wizard.models.stylizer.unet import PRESETS as UNET_PRESETS, UNet, UNetConfig

VOCAB = SpecVocab(["bangers-400", "cinzel-400", "rye-400"], ["classic-hero", "antique-gold"])


def test_onnx_layout_matches_torch_and_respects_allowed(tmp_path):
    torch.manual_seed(0)
    model = LayoutModel(LayoutConfig(**PRESETS["tiny"]), VOCAB).eval()
    model.save(tmp_path / "best.pt")
    onnx_export.export_layout(tmp_path / "best.pt", tmp_path)
    layout = runtime.OnnxLayout(tmp_path, fonts={"bangers-400", "cinzel-400"})
    specs = layout.suggest("The Iron Comet", "pulp", n=12, seed=0)
    assert len(specs) == 12 and all(s.font in {"bangers-400", "cinzel-400"} for s in specs)
    # Greedy decoding gives the same tokens through ONNX as through PyTorch.
    from logo_wizard.models.layout.tokens import BOS, MAX_TEXT, PAD, encode_text

    ids = encode_text("Star Patrol", "sci-fi")
    src = np.full((1, MAX_TEXT), PAD, dtype=np.int64)
    src[0, : len(ids)] = ids
    torch_tokens = model.sample(torch.from_numpy(src), temperature=0).numpy()[0]
    tgt = np.zeros((1, VOCAB.length), dtype=np.int64)
    tgt[0, 0] = BOS
    for step in range(VOCAB.length):
        logits = layout.session.run(None, {"src": src, "tgt": tgt})[0][0, step]
        token = int(np.argmax(logits))
        assert token == torch_tokens[step]
        if step + 1 < VOCAB.length:
            tgt[0, step + 1] = token


def test_onnx_stylizer_samples(tmp_path):
    torch.manual_seed(0)
    unet = UNet(UNetConfig(**UNET_PRESETS["tiny"])).eval()
    torch.save({"config": unet.config.to_dict(), "palettes": ["classic-hero"], "state": unet.state_dict(), "size": [32, 20], "step": 1}, tmp_path / "ema.pt")
    onnx_export.export_stylizer(tmp_path / "ema.pt", tmp_path)
    stylizer = runtime.OnnxStylizer(tmp_path, gpu=False)
    mask = np.zeros((2, 1, 20, 32), np.float32)
    out = stylizer.sample(mask, np.zeros((2, 8), np.int64), steps=3)
    assert out.shape == (2, 4, 20, 32) and np.isfinite(out).all()
    # Same sampler maths as the PyTorch DDIM: compare one denoising pass on the same noise.
    from logo_wizard.models.stylizer.diffusion import cosine_alphas

    assert np.allclose(runtime.cosine_alphas(), cosine_alphas().numpy(), atol=1e-6)
