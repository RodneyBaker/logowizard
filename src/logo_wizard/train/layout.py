"""Train the layout model on the synthetic dataset's (text, style words) -> spec pairs.

Usage (from the repo root, on the GTX 1080):
    logo-wizard-train-layout                           # base model, resumes local/runs/layout/last.pt
    logo-wizard-train-layout --max-minutes 180         # stop cleanly after 3 hours; run again to continue
    logo-wizard-train-layout --preset tiny --data local/dataset-small --epochs 1 --device cpu   # smoke test

Writes local/runs/layout/{last.pt, best.pt, log.csv}. best.pt is the
checkpoint the app loads.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..models.layout.model import PRESETS, LayoutConfig, LayoutModel, count_parameters
from ..models.layout.tokens import BOS, PAD, SpecVocab, encode_text
from ..render.shaping import load_manifest
from ..render.spec import DesignSpec, load_palettes
from ..synth.generate import OUT_DIR, iter_records
from .common import RUNS_DIR, CsvLog, Deadline, cosine_lr, load_checkpoint, pick_device, save_checkpoint, seed_everything


def build_vocab() -> SpecVocab:
    faces = load_manifest(require_files=False)
    palettes = load_palettes()
    tags = {k: [f"mood:{m}" for m in f.moods] + [f"class:{f.font_class}"] for k, f in faces.items()}
    tags.update({k: [f"mood:{m}" for m in p.moods] + [f"family:{p.family}"] for k, p in palettes.items()})
    return SpecVocab(sorted(faces), sorted(palettes), tags)


def load_pairs(root: Path, vocab: SpecVocab) -> tuple[list[list[int]], np.ndarray]:
    sources, targets = [], []
    for _, record in iter_records(root):
        spec = DesignSpec.from_dict(record["spec"])
        sources.append(encode_text(record["text"], record.get("words", "")))
        targets.append(vocab.encode(spec))
    return sources, np.asarray(targets, dtype=np.int64)


def batches(sources, targets, indices, batch_size, rng=None):
    order = np.array(indices)
    if rng is not None:
        rng.shuffle(order)
    for i in range(0, len(order), batch_size):
        idx = order[i : i + batch_size]
        width = max(len(sources[j]) for j in idx)
        src = np.full((len(idx), width), PAD, dtype=np.int64)
        for row, j in enumerate(idx):
            src[row, : len(sources[j])] = sources[j]
        yield torch.from_numpy(src), torch.from_numpy(targets[idx])


def step_loss(model: LayoutModel, src: torch.Tensor, tgt: torch.Tensor, weights: torch.Tensor | None = None):
    bos = torch.full((tgt.shape[0], 1), BOS, dtype=torch.long, device=tgt.device)
    logits = model(src, torch.cat([bos, tgt[:, :-1]], dim=1))
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none").view(tgt.shape)
    per_sample = loss.mean(1)
    if weights is not None:
        per_sample = per_sample * weights
    return per_sample.mean(), logits


@torch.no_grad()
def evaluate(model, sources, targets, indices, batch_size, device) -> tuple[float, np.ndarray]:
    model.eval()
    total, n = 0.0, 0
    correct = np.zeros(model.vocab.length)
    for src, tgt in batches(sources, targets, indices, batch_size):
        src, tgt = src.to(device), tgt.to(device)
        loss, logits = step_loss(model, src, tgt)
        total += loss.item() * len(src)
        n += len(src)
        correct += (logits.argmax(-1) == tgt).float().sum(0).cpu().numpy()
    model.train()
    return total / max(n, 1), correct / max(n, 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=OUT_DIR)
    parser.add_argument("--run", type=Path, default=RUNS_DIR / "layout")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="base")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-share", type=float, default=0.02)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--fresh", action="store_true", help="ignore last.pt and start over")
    args = parser.parse_args(argv)

    seed_everything(args.seed)
    device = pick_device(args.device)
    vocab = build_vocab()
    print(f"Loading samples from {args.data} ...")
    sources, targets = load_pairs(args.data, vocab)
    if len(sources) < 10:
        print(f"Only {len(sources)} samples under {args.data}; run logo-wizard-synth first.", file=sys.stderr)
        return 1
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(sources))
    n_val = max(1, int(len(order) * args.val_share))
    val_idx, train_idx = order[:n_val].tolist(), order[n_val:].tolist()

    last_path, best_path = args.run / "last.pt", args.run / "best.pt"
    ckpt = None if args.fresh else load_checkpoint(last_path, device)
    config = LayoutConfig(**ckpt["config"]) if ckpt else LayoutConfig(**PRESETS[args.preset])
    if ckpt:
        vocab = SpecVocab.from_dict(ckpt["vocab"])
    model = LayoutModel(config, vocab).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    start_epoch, step, best = 0, 0, float("inf")
    if ckpt:
        model.load_state_dict(ckpt["state"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch, step, best = ckpt["epoch"], ckpt["step"], ckpt["best"]
        print(f"Resuming at epoch {start_epoch}, step {step}")
    print(f"{count_parameters(model) / 1e6:.1f}M parameters on {device}; {len(train_idx):,} train / {len(val_idx):,} val samples")

    steps_per_epoch = (len(train_idx) + args.batch_size - 1) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    log = CsvLog(args.run / "log.csv", ["epoch", "step", "train_loss", "val_loss", "font_acc", "palette_acc", "warp_acc", "samples_per_s"])
    deadline = Deadline(args.max_minutes)

    def checkpoint(epoch: int) -> None:
        save_checkpoint(last_path, config=vars(config), vocab=vocab.to_dict(), state=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch, step=step, best=best)

    model.train()
    try:
        for epoch in range(start_epoch, args.epochs):
            t0, seen, running = time.perf_counter(), 0, 0.0
            for src, tgt in batches(sources, targets, train_idx, args.batch_size, np.random.default_rng([args.seed, epoch])):
                for group in optimizer.param_groups:
                    group["lr"] = cosine_lr(step, args.lr, warmup=min(500, total_steps // 10 + 1), total=total_steps)
                src, tgt = src.to(device), tgt.to(device)
                loss, _ = step_loss(model, src, tgt)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                step += 1
                seen += len(src)
                running = 0.98 * running + 0.02 * loss.item() if running else loss.item()
                if step % 200 == 0:
                    print(f"epoch {epoch} step {step}/{total_steps} loss {running:.3f} ({seen / (time.perf_counter() - t0):.0f} samples/s)", flush=True)
                if deadline.passed():
                    raise KeyboardInterrupt
            val_loss, acc = evaluate(model, sources, targets, val_idx, args.batch_size, device)
            rate = seen / (time.perf_counter() - t0)
            fields = vocab.index
            log.write(epoch=epoch, step=step, train_loss=round(running, 4), val_loss=round(val_loss, 4),
                      font_acc=round(acc[fields["font"]], 3), palette_acc=round(acc[fields["palette"]], 3),
                      warp_acc=round(acc[fields["warp"]], 3), samples_per_s=round(rate))
            print(f"epoch {epoch}: val loss {val_loss:.3f}, per-field accuracy {acc.mean():.3f}")
            if val_loss < best:
                best = val_loss
                model.save(best_path, extra={"val_loss": val_loss, "epoch": epoch})
            checkpoint(epoch + 1)
    except KeyboardInterrupt:
        checkpoint(epoch)
        print(f"Stopped; saved {last_path}. Run the same command to continue.")
        return 0
    print(f"Done. Best val loss {best:.3f}; the app uses {best_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
