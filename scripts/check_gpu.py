"""Check that PyTorch can train on this GPU, and measure it.

Run on the training PC after installing the pinned PyTorch build:
    python scripts/check_gpu.py

It prints the CUDA build, whether the GTX 1080 (compute capability 6.1) is
supported by the installed kernels, free VRAM, fp32 vs fp16 matmul speed,
and a quick layout-model and stylizer step timing at the training batch sizes.
"""

import sys
import time

import torch


def bench_matmul(dtype: torch.dtype, n: int = 4096, reps: int = 20) -> float:
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        a @ b
    torch.cuda.synchronize()
    return 2 * n**3 * reps / (time.perf_counter() - t0) / 1e12


def main() -> int:
    print(f"PyTorch {torch.__version__}, CUDA build {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("CUDA is not available. Install the pinned build from the README (cu126 wheel index).")
        return 1
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    arches = torch.cuda.get_arch_list()
    print(f"GPU: {name}, compute capability {major}.{minor}; kernels built for {' '.join(arches)}")
    # A cubin for sm_60 runs on sm_61; PTX for an older arch can be JIT-compiled.
    ok = any(a in arches for a in (f"sm_{major}{minor}", f"sm_{major}0")) or any(a.startswith("compute_") and int(a[8:]) <= major * 10 + minor for a in arches)
    if not ok:
        print("This PyTorch build has no kernels for this GPU. Reinstall the pinned version from the README.")
        return 1
    free, total = torch.cuda.mem_get_info()
    print(f"VRAM: {free / 2**30:.1f} GB free of {total / 2**30:.1f} GB")
    print(f"fp32 matmul: {bench_matmul(torch.float32):.1f} TFLOPS")
    print(f"fp16 matmul: {bench_matmul(torch.float16):.2f} TFLOPS (Pascal is slow here, so training uses fp32)")

    sys.path.insert(0, "src")
    from logo_wizard.models.layout.model import PRESETS, LayoutConfig, LayoutModel
    from logo_wizard.train.layout import build_vocab

    vocab = build_vocab()
    model = LayoutModel(LayoutConfig(**PRESETS["base"]), vocab).cuda()
    opt = torch.optim.AdamW(model.parameters(), 3e-4)
    src = torch.randint(5, 100, (256, 40), device="cuda")
    tgt = torch.randint(3, vocab.size, (256, vocab.length), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    for i in range(12):
        if i == 2:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        logits = model(src, tgt)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab.size).clamp_min(-1e4), tgt.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    per = (time.perf_counter() - t0) / 10
    print(f"Layout model (base, batch 256): {per * 1000:.0f} ms/step, {256 / per:,.0f} samples/s, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    del model, opt

    try:
        from logo_wizard.models.stylizer.unet import PRESETS as UNET_PRESETS, UNet, UNetConfig
    except ImportError:
        return 0
    unet = UNet(UNetConfig(**UNET_PRESETS["base"])).cuda()
    opt = torch.optim.AdamW(unet.parameters(), 1e-4)
    x = torch.randn(16, 4, 80, 128, device="cuda")
    cond = torch.randn(16, 1, 80, 128, device="cuda")
    t = torch.randint(0, 1000, (16,), device="cuda")
    labels = torch.zeros(16, unet.config.cond_fields, dtype=torch.long, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    for i in range(7):
        if i == 2:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        loss = (unet(x, t, cond, labels) - x).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    per = (time.perf_counter() - t0) / 5
    print(f"Stylizer (base, batch 16, 128x80): {per * 1000:.0f} ms/step, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
