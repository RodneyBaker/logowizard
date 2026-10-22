"""Export the trained models to ONNX for the app's PyTorch-free runtime.

Usage (needs the train extras):
    logo-wizard-export-onnx                    # local/runs/{layout/best.pt, stylizer/ema.pt} -> local/models/

Writes local/models/layout.onnx + layout.json and stylizer.onnx +
stylizer.json. The app prefers these when present: ONNX Runtime runs them
on the CPU, or on the GPU through DirectML on Windows, without PyTorch.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import torch

from .. import paths
from ..models.layout.model import LayoutModel
from ..models.layout.tokens import MAX_TEXT, PAD
from ..models.stylizer.unet import UNet, UNetConfig

RUNS = paths.RUNS_DIR
MODELS_DIR = paths.MODELS_DIR
OPSET = 17


class _LayoutFull(torch.nn.Module):
    """Fixed-length graph: src padded to MAX_TEXT, tgt always the full field count.

    The causal mask means logits at step i only see tokens before i, so the
    runtime fills the rest of tgt with padding and reads the step it needs.
    """

    def __init__(self, model: LayoutModel):
        super().__init__()
        self.model = model

    def forward(self, src, tgt):
        return self.model(src, tgt)


def export_layout(weights: Path, out_dir: Path) -> Path:
    model = LayoutModel.load(weights).eval()
    vocab = model.vocab
    src = torch.full((2, MAX_TEXT), PAD, dtype=torch.long)
    src[:, :4] = torch.tensor([1, 40, 41, 2])
    tgt = torch.ones((2, vocab.length), dtype=torch.long)
    target = out_dir / "layout.onnx"
    torch.onnx.export(
        _LayoutFull(model), (src, tgt), target, dynamo=False, opset_version=OPSET,
        input_names=["src", "tgt"], output_names=["logits"],
        dynamic_axes={"src": {0: "batch"}, "tgt": {0: "batch"}, "logits": {0: "batch"}},
    )
    (out_dir / "layout.json").write_text(json.dumps({"vocab": vocab.to_dict(), "max_text": MAX_TEXT}), encoding="utf-8")
    return target


def export_stylizer(weights: Path, out_dir: Path) -> Path:
    data = torch.load(weights, map_location="cpu", weights_only=False)
    config = UNetConfig(**data["config"])
    config.checkpointing = False
    model = UNet(config).eval()
    model.load_state_dict(data["state"])
    w, h = data.get("size", (128, 80))
    args = (torch.randn(2, 4, h, w), torch.tensor([10, 500]), torch.zeros(2, 1, h, w), torch.zeros(2, config.cond_fields, dtype=torch.long))
    target = out_dir / "stylizer.onnx"
    torch.onnx.export(
        model, args, target, dynamo=False, opset_version=OPSET,
        input_names=["x", "t", "mask", "labels"], output_names=["eps"],
        dynamic_axes={name: {0: "batch"} for name in ("x", "t", "mask", "labels", "eps")},
    )
    (out_dir / "stylizer.json").write_text(json.dumps({"palettes": data["palettes"], "size": [w, h], "step": data.get("step")}), encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layout", type=Path, default=RUNS / "layout" / "best.pt")
    parser.add_argument("--stylizer", type=Path, default=RUNS / "stylizer" / "ema.pt")
    parser.add_argument("--out", type=Path, default=MODELS_DIR)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    done = 0
    for name, path, export in (("layout", args.layout, export_layout), ("stylizer", args.stylizer, export_stylizer)):
        if path.exists():
            print(f"{name}: {export(path, args.out)}")
            done += 1
        else:
            print(f"{name}: no weights at {path}, skipped")
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
