import numpy as np
import pytest

from logo_wizard.fonts.fetch import parse_metadata, pick_faces
from logo_wizard.render import geometry
from logo_wizard.render.grammar import read_moods, read_style, sample_spec
from logo_wizard.render.shaping import FONT_DIR, load_manifest
from logo_wizard.render.spec import WARPS, DesignSpec, load_palettes

METADATA = """name: "Oswald"
license: "OFL"
category: "SANS_SERIF"
fonts {
  name: "Oswald"
  style: "normal"
  weight: 400
  filename: "Oswald[wght].ttf"
}
axes {
  tag: "wght"
  min_value: 200.0
  max_value: 700.0
}
"""

STATIC = """name: "Abhaya Libre"
license: "OFL"
fonts {
  style: "normal"
  weight: 400
  filename: "AbhayaLibre-Regular.ttf"
}
fonts {
  style: "normal"
  weight: 800
  filename: "AbhayaLibre-ExtraBold.ttf"
}
fonts {
  style: "italic"
  weight: 400
  filename: "AbhayaLibre-Italic.ttf"
}
"""


def test_parse_metadata_and_pick_variable_faces():
    meta = parse_metadata(METADATA)
    assert meta["name"] == "Oswald" and meta["license"] == "OFL"
    assert meta["axes"]["wght"] == (200.0, 700.0)
    assert pick_faces(meta) == [("Oswald[wght].ttf", 400, "wght=400"), ("Oswald[wght].ttf", 700, "wght=700")]


def test_pick_static_faces_skips_italics():
    faces = pick_faces(parse_metadata(STATIC))
    assert [f[0] for f in faces] == ["AbhayaLibre-Regular.ttf", "AbhayaLibre-ExtraBold.ttf"]


def test_manifest_is_open_licensed():
    faces = load_manifest(require_files=False)
    assert len({f.family for f in faces.values()}) >= 200
    assert {f.license for f in faces.values()} <= {"SIL OFL 1.1", "Apache 2.0", "Ubuntu Font License 1.0"}


def test_palettes_have_every_role():
    palettes = load_palettes()
    assert {p.family for p in palettes.values()} == {"comic", "novel"}
    for p in palettes.values():
        assert 1 <= len(p.fill) <= 3 and p.outlines


def test_spec_json_round_trip():
    spec = DesignSpec(text="The Iron Comet", outlines=[3.0, 5.0], warp="arch", warp_amount=0.5)
    assert DesignSpec.from_json(spec.to_json()) == spec


def test_split_lines_puts_connectors_on_small_lines():
    lines = geometry.split_lines("Attack of the Atomic Squid", "stacked")
    assert lines == [("Attack", 1.0), ("of the", geometry.SMALL_SCALE), ("Atomic Squid", 1.0)]
    assert geometry.split_lines("Attack of the Atomic Squid", "straight") == [("Attack of the Atomic Squid", 1.0)]
    assert geometry.split_lines("The Iron Comet", "stacked")[0] == ("The", geometry.SMALL_SCALE)


@pytest.mark.parametrize("warp", [w for w in WARPS if w not in ("none", "stagger")])
def test_warps_keep_points_finite_and_near_the_box(warp):
    xs, ys = np.meshgrid(np.linspace(0, 400, 21), np.linspace(-70, 20, 9))
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
    out = geometry.warp_points(pts, warp, 1.0, (0, -70, 400, 20))
    assert np.isfinite(out).all()
    assert np.abs(out).max() < 1000


def test_moods_and_style_words():
    assert read_moods("Curse of the Star Witch") == ["horror", "scifi"]
    assert read_style("gothic novel") == "novel"
    assert read_style("") == "comic"


have_fonts = pytest.mark.skipif(not any(FONT_DIR.glob("*/*.ttf")), reason="fonts not downloaded (run logo-wizard-fonts)")


@have_fonts
def test_grammar_and_render_draw_the_letters():
    from logo_wizard.render.draw import Renderer

    renderer = Renderer()
    rng = np.random.default_rng(0)
    for style in ("comic", "novel"):
        for _ in range(6):
            spec = sample_spec("Tales of the Iron Comet", renderer.faces, renderer.palettes, rng, style=style, subtitle="Book II")
            assert spec.font in renderer.faces and spec.palette in renderer.palettes
            result = renderer.render(spec, 256, 160)
            assert result.image.shape == (160, 256, 4) and result.mask.shape == (160, 256)
            assert (result.mask > 128).mean() > 0.01
            # Drawn pixels cover the letters (texture may knock a few out).
            covered = result.image[..., 3][result.mask > 200] > 0
            assert covered.mean() > 0.8


@have_fonts
def test_app_generates_and_exports():
    from fastapi.testclient import TestClient

    from logo_wizard.app.server import create_app

    client = TestClient(create_app(use_models=False))
    assert client.get("/").status_code == 200
    res = client.post("/api/generate", json={"text": "Star Patrol", "words": "sci-fi", "count": 2, "seed": 1})
    assert res.status_code == 200
    candidates = res.json()["candidates"]
    assert len(candidates) == 2
    png = client.post("/api/render", json={"spec": candidates[0]["spec"], "width": 300, "height": 200})
    assert png.headers["content-type"] == "image/png" and png.content[:4] == b"\x89PNG"
    assert client.post("/api/generate", json={"text": "  "}).status_code == 400
