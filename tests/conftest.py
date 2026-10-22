import pytest

from logo_wizard.render.shaping import FONT_DIR


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory):
    """A 40-sample dataset at 128x80, rendered once per test run."""
    if not any(FONT_DIR.glob("*/*.ttf")):
        pytest.skip("fonts not downloaded (run logo-wizard-fonts)")
    from logo_wizard.synth import generate

    root = tmp_path_factory.mktemp("dataset")
    generate.main(["--count", "40", "--out", str(root), "--workers", "1", "--no-ocr", "--size", "128x80", "--shard-size", "20"])
    return root
