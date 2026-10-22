"""Train the glyph-conditioned diffusion stylizer on the synthetic dataset.

Usage (from the repo root, on the GTX 1080):
    logo-wizard-train-stylizer                            # base model at 128x80, resumes local/runs/stylizer/last.pt
    logo-wizard-train-stylizer --max-minutes 480          # an overnight session; run again to continue
    logo-wizard-train-stylizer --size 256x160 --batch-size 6 --run local/runs/stylizer-256   # stretch goal
    logo-wizard-train-stylizer --preset tiny --data local/dataset-small --steps 20 --device cpu --batch-size 4   # smoke test

Trains in fp32 (Pascal has no fast fp16) with gradient checkpointing,
keeps an EMA copy of the weights for sampling, and writes a preview sheet
every --preview-every steps. Files land in local/runs/stylizer/.
"""

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..models.stylizer.diffusion import STEPS, add_noise, cosine_alphas, ddim_sample, from_rgba, spec_labels, to_rgba
from ..models.stylizer.unet import PRESETS, UNet, UNetConfig
from ..render.draw import on_background
from ..render.spec import DesignSpec, load_palettes
from ..synth.generate import OUT_DIR, iter_records
from .common import RUNS_DIR, CsvLog, Deadline, cosine_lr, load_checkpoint, pick_device, save_checkpoint, seed_everything


class LogoDataset(Dataset):
    """Reads (logo, mask, labels) from the dataset shards, resized to the training size."""

    def __init__(self, root: Path, size: tuple[int, int], palettes: list[str], indices: list[int] | None = None):
        self.items = [(shard, rec["id"], spec_labels(DesignSpec.from_dict(rec["spec"]), palettes)) for shard, rec in iter_records(root)]
        if indices is not None:
            self.items = [self.items[i] for i in indices]
        self.size = size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        shard, name, labels = self.items[i]
        logo = Image.open(shard / f"{name}.png").convert("RGBA").resize(self.size, Image.BOX)
        mask = Image.open(shard / f"{name}_mask.png").convert("L").resize(self.size, Image.BOX)
        x = torch.from_numpy(from_rgba(np.asarray(logo)))
        m = torch.from_numpy(np.asarray(mask, dtype=np.float32)[None] / 127.5 - 1)
        return x, m, torch.tensor(labels, dtype=torch.long)


def ema_update(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for e, p in zip(ema.parameters(), model.parameters()):
            e.mul_(decay).add_(p.detach(), alpha=1 - decay)


def preview(ema: UNet, batch, path: Path, steps: int = 25) -> None:
    x, m, labels = batch
    device = next(ema.parameters()).device
    gen = torch.Generator(device=device).manual_seed(0)
    out = to_rgba(ddim_sample(ema, m.to(device), labels.to(device), steps=steps, generator=gen))
    target = to_rgba(x)
    h, w = out.shape[1:3]
    sheet = Image.new("RGB", (3 * w, len(out) * h), (255, 255, 255))
    for i in range(len(out)):
        mask = ((m[i, 0].numpy() + 1) * 127.5).astype(np.uint8)
        sheet.paste(Image.fromarray(mask).convert("RGB"), (0, i * h))
        sheet.paste(on_background(out[i]), (w, i * h))
        sheet.paste(on_background(target[i]), (2 * w, i * h))
    sheet.save(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=OUT_DIR)
    parser.add_argument("--run", type=Path, default=RUNS_DIR / "stylizer")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="base")
    parser.add_argument("--size", default="128x80", help="training WIDTHxHEIGHT; both must divide by 8")
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ema", type=float, default=0.999)
    parser.add_argument("--label-dropout", type=float, default=0.1, help="share of steps trained without labels, for guidance")
    parser.add_argument("--no-checkpointing", action="store_true", help="faster, but needs more VRAM")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--preview-every", type=int, default=10_000)
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args(argv)

    seed_everything(args.seed)
    device = pick_device(args.device)
    width, _, height = args.size.partition("x")
    size = (int(width), int(height))
    last_path = args.run / "last.pt"
    ckpt = None if args.fresh else load_checkpoint(last_path, device)

    palettes = ckpt["palettes"] if ckpt else sorted(load_palettes())
    data = LogoDataset(args.data, size, palettes)
    if len(data) < args.batch_size:
        print(f"Only {len(data)} samples under {args.data}; run logo-wizard-synth first.", file=sys.stderr)
        return 1
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers,
                        persistent_workers=args.workers > 0, pin_memory=device.type == "cuda")

    if ckpt:
        config = UNetConfig(**ckpt["config"])
    else:
        config = UNetConfig(**PRESETS[args.preset])
    config.checkpointing = not args.no_checkpointing and device.type == "cuda"
    model = UNet(config).to(device)
    ema = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    step = 0
    if ckpt:
        model.load_state_dict(ckpt["state"])
        ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        print(f"Resuming at step {step}")
    for p in ema.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{n_params / 1e6:.1f}M parameters on {device}, {len(data):,} samples at {size[0]}x{size[1]}, batch {args.batch_size}")

    alpha_bar = cosine_alphas().to(device)
    log = CsvLog(args.run / "log.csv", ["step", "loss", "samples_per_s", "lr", "peak_vram_gb"])
    deadline = Deadline(args.max_minutes)
    preview_batch = next(iter(DataLoader(data, batch_size=8, shuffle=False)))
    gen = torch.Generator(device="cpu").manual_seed(args.seed + step)

    def save() -> None:
        state = dict(config=config.to_dict(), palettes=palettes, state=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(), step=step, size=list(size))
        save_checkpoint(last_path, **state)
        save_checkpoint(args.run / "ema.pt", config=config.to_dict(), palettes=palettes, state=ema.state_dict(), step=step, size=list(size))

    model.train()
    running, t0, seen = None, time.perf_counter(), 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    try:
        while step < args.steps:
            for x, m, labels in loader:
                x, m, labels = x.to(device, non_blocking=True), m.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                drop = torch.rand(labels.shape[0], generator=gen) < args.label_dropout
                labels[drop.to(device)] = 0
                t = torch.randint(0, STEPS, (x.shape[0],), generator=gen).to(device)
                noise = torch.randn(x.shape, generator=gen).to(device)
                pred = model(add_noise(x, t, noise, alpha_bar), t, m, labels)
                loss = torch.nn.functional.mse_loss(pred, noise)
                lr = cosine_lr(step, args.lr, warmup=1000, total=args.steps, floor=0.1)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ema_update(ema, model, args.ema if step > 1000 else 0.0)
                step += 1
                seen += x.shape[0]
                running = loss.item() if running is None else 0.99 * running + 0.01 * loss.item()
                if step % 100 == 0 or step == args.steps:
                    rate = seen / (time.perf_counter() - t0)
                    peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
                    print(f"step {step}/{args.steps} loss {running:.4f} ({rate:.1f} samples/s, peak VRAM {peak:.2f} GB)", flush=True)
                    log.write(step=step, loss=round(running, 5), samples_per_s=round(rate, 1), lr=f"{lr:.2e}", peak_vram_gb=round(peak, 2))
                if step % args.save_every == 0:
                    save()
                if step % args.preview_every == 0:
                    preview(ema, preview_batch, args.run / f"preview_{step:07d}.png")
                if step >= args.steps or deadline.passed():
                    break
            if deadline.passed():
                break
    except KeyboardInterrupt:
        pass
    save()
    print(f"Saved step {step} to {last_path} (EMA weights for the app: {args.run / 'ema.pt'}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
