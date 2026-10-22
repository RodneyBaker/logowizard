"""Fine-tune the layout model on your thumbs-up/down from the app.

Usage (needs the train extras):
    logo-wizard-finetune-layout                 # local/feedback/feedback.jsonl -> local/runs/layout/best.pt

Liked specs (thumbs-up, exports, blind-test wins) are trained toward with
the usual loss; disliked ones are pushed away with an unlikelihood loss.
A slice of the synthetic dataset is replayed alongside so the model keeps
what it learned. The previous weights are kept as best.before-feedback.pt,
and local/models/layout.onnx is refreshed if it exists.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from .. import paths
from ..models.layout.model import LayoutModel
from ..models.layout.tokens import BOS, encode_text
from ..render.spec import DesignSpec
from ..synth.generate import OUT_DIR
from .common import RUNS_DIR, pick_device, seed_everything
from .layout import batches, load_pairs, step_loss

FEEDBACK = paths.FEEDBACK_DIR / "feedback.jsonl"


def load_feedback(path: Path, model: LayoutModel) -> tuple[list[list[int]], np.ndarray, np.ndarray]:
    """Latest vote per (text, words, spec) wins, so changing your mind counts."""
    latest: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            key = json.dumps([record["spec"].get("text"), record.get("words", ""), record["spec"]], sort_keys=True)
            latest[key] = record
    sources, targets, votes = [], [], []
    for record in latest.values():
        spec = DesignSpec.from_dict(record["spec"])
        sources.append(encode_text(spec.text, record.get("words", "")))
        targets.append(model.vocab.encode(spec))
        votes.append(record["vote"])
    return sources, np.asarray(targets, dtype=np.int64).reshape(-1, model.vocab.length), np.asarray(votes)


def unlikelihood(model: LayoutModel, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    bos = torch.full((tgt.shape[0], 1), BOS, dtype=torch.long, device=tgt.device)
    logp = torch.log_softmax(model(src, torch.cat([bos, tgt[:, :-1]], dim=1)), -1)
    p = logp.gather(-1, tgt[..., None]).squeeze(-1).exp()
    return -torch.log((1 - p).clamp_min(1e-6)).mean()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--feedback", type=Path, default=FEEDBACK)
    parser.add_argument("--weights", type=Path, default=RUNS_DIR / "layout" / "best.pt")
    parser.add_argument("--data", type=Path, default=OUT_DIR, help="synthetic set to replay from")
    parser.add_argument("--replay", type=int, default=4000, help="synthetic samples mixed in")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--dislike-weight", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--min-votes", type=int, default=20)
    args = parser.parse_args(argv)

    if not args.feedback.exists():
        print(f"No feedback yet at {args.feedback}. Use the thumbs in the app first.", file=sys.stderr)
        return 1
    seed_everything(0)
    device = pick_device(args.device)
    model = LayoutModel.load(args.weights, map_location=device).to(device)
    src_fb, tgt_fb, votes = load_feedback(args.feedback, model)
    if len(votes) < args.min_votes:
        print(f"Only {len(votes)} votes; collect at least {args.min_votes} (or pass --min-votes).", file=sys.stderr)
        return 1
    liked = [i for i, v in enumerate(votes) if v > 0]
    disliked = [i for i, v in enumerate(votes) if v < 0]
    src_rp, tgt_rp = load_pairs(args.data, model.vocab) if args.data.exists() else ([], np.zeros((0, model.vocab.length), np.int64))
    rng = np.random.default_rng(0)
    replay = rng.permutation(len(src_rp))[: args.replay].tolist()
    print(f"{len(liked)} liked, {len(disliked)} disliked, {len(replay)} replayed synthetic samples on {device}")

    sources = src_fb + [src_rp[i] for i in replay]
    targets = np.concatenate([tgt_fb, tgt_rp[replay]]) if replay else tgt_fb
    positive = liked + list(range(len(src_fb), len(sources)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    model.train()
    for epoch in range(args.epochs):
        total, n = 0.0, 0
        for src, tgt in batches(sources, targets, positive, args.batch_size, np.random.default_rng(epoch)):
            loss, _ = step_loss(model, src.to(device), tgt.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total, n = total + loss.item(), n + 1
        for src, tgt in batches(sources, targets, disliked, args.batch_size, np.random.default_rng(epoch)):
            loss = args.dislike_weight * unlikelihood(model, src.to(device), tgt.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch}: liked/replay loss {total / max(n, 1):.3f}")

    backup = args.weights.with_name("best.before-feedback.pt")
    if not backup.exists():
        shutil.copy2(args.weights, backup)
    model.eval()
    model.save(args.weights, extra={"finetuned_on": len(votes)})
    print(f"Saved {args.weights} (previous weights: {backup})")
    onnx = paths.MODELS_DIR / "layout.onnx"
    if onnx.exists():
        from ..infer.onnx_export import export_layout

        export_layout(args.weights, onnx.parent)
        print(f"Refreshed {onnx}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
