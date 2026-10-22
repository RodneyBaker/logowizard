import numpy as np
import pytest

torch = pytest.importorskip("torch")

from logo_wizard.models.stylizer.diffusion import cosine_alphas, ddim_sample, from_rgba, label_sizes, spec_labels, to_rgba
from logo_wizard.models.stylizer.unet import PRESETS, UNet, UNetConfig
from logo_wizard.render.spec import DesignSpec


def test_rgba_round_trip_premultiplies():
    image = np.zeros((4, 6, 4), np.uint8)
    image[1:3, 2:5] = (200, 40, 10, 255)
    image[0, 0] = (255, 255, 255, 0)
    x = from_rgba(image)
    assert x.shape == (4, 4, 6) and x.min() >= -1 and x.max() <= 1
    back = to_rgba(torch.from_numpy(x)[None])[0]
    assert np.abs(back[1:3, 2:5].astype(int) - image[1:3, 2:5]).max() <= 1
    assert back[0, 0, 3] == 0


def test_labels_fit_their_embeddings():
    spec = DesignSpec(text="x", style="novel", palette="ivory", fill_mode="foil", texture="aged", halftone=0.7, outlines=[1, 2, 3, 4], inline=True, extrude_depth=4)
    labels = spec_labels(spec, ["classic-hero", "ivory"])
    assert labels == [2, 2, 3, 3, 4, 4, 2, 2]
    assert all(0 <= l < n for l, n in zip(labels, label_sizes()))


def test_schedule_is_decreasing():
    a = cosine_alphas()
    assert a.shape == (1000,) and (a[1:] <= a[:-1]).all() and 0 < a[-1] < 0.01


@pytest.mark.parametrize("preset", ["tiny", "small"])
def test_unet_keeps_the_image_shape(preset):
    model = UNet(UNetConfig(**PRESETS[preset])).eval()
    x = torch.randn(2, 4, 80, 128)
    out = model(x, torch.tensor([0, 999]), torch.zeros(2, 1, 80, 128), torch.zeros(2, 8, dtype=torch.long))
    assert out.shape == x.shape


def test_ddim_sampler_outputs_valid_images():
    model = UNet(UNetConfig(**PRESETS["tiny"])).eval()
    mask = torch.zeros(2, 1, 16, 24)
    x = ddim_sample(model, mask, torch.zeros(2, 8, dtype=torch.long), steps=3, generator=torch.Generator().manual_seed(0))
    assert x.shape == (2, 4, 16, 24) and torch.isfinite(x).all() and x.abs().max() <= 1.0 + 1e-5


def test_training_smoke_run_and_stylize(tiny_dataset, tmp_path):
    from logo_wizard.infer.stylize import Stylizer
    from logo_wizard.train import stylizer

    run = tmp_path / "run"
    args = ["--preset", "tiny", "--data", str(tiny_dataset), "--steps", "3", "--batch-size", "2", "--size", "32x20",
            "--device", "cpu", "--workers", "0", "--run", str(run), "--save-every", "2", "--preview-every", "3"]
    assert stylizer.main(args) == 0
    assert (run / "ema.pt").exists() and (run / "preview_0000003.png").exists()
    images = Stylizer(run / "ema.pt", device="cpu").stylize([DesignSpec(text="Kid Rocket")], out_size=(64, 40), steps=2)
    assert images[0].shape == (40, 64, 4)
