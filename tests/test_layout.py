import pytest

torch = pytest.importorskip("torch")

from logo_wizard.models.layout.model import PRESETS, LayoutConfig, LayoutModel
from logo_wizard.models.layout.tokens import BOS, EOS, SpecVocab, encode_text
from logo_wizard.render.spec import DesignSpec

TAGS = {"rye-400": ["mood:western"], "cinzel-400": ["mood:fantasy"], "antique-gold": ["mood:western", "family:novel"]}
VOCAB = SpecVocab(["bangers-400", "cinzel-400", "rye-400"], ["classic-hero", "antique-gold"], TAGS)


def test_spec_tokens_round_trip_through_the_bins():
    spec = DesignSpec(text="Iron Comet", style="comic", subtitle="No. 1", font="rye-400", subtitle_font="cinzel-400",
                      palette="antique-gold", warp="arch", warp_amount=0.62, tracking=0.03, tilt=-3.4, outlines=[3.4, 5.6],
                      extrude_depth=11.0, extrude_angle=60, shadow=4.0, shadow_angle=90, halftone=0.5, inline=True, texture="aged")
    ids = VOCAB.encode(spec)
    assert len(ids) == VOCAB.length
    back = VOCAB.decode(ids, "Iron Comet", "No. 1", seed=spec.seed)
    assert back.font == "rye-400" and back.palette == "antique-gold" and back.subtitle_font == "cinzel-400"
    assert back.outlines == [3.5, 5.5] and back.warp == "arch" and back.warp_amount == 0.65
    assert back.tilt == -4.0 and back.inline and back.texture == "aged"
    assert VOCAB.encode(back) == ids


def test_tags_reach_fonts_palettes_and_model_input():
    matrix = VOCAB.tag_matrix()  # columns: family:novel, mood:fantasy, mood:western
    font, palette, sub = (VOCAB.offsets[VOCAB.index[f]] for f in ("font", "palette", "subtitle_font"))
    assert matrix[font + 2].tolist() == [0, 0, 1] and matrix[palette + 1].tolist() == [1, 0, 1]
    assert matrix[sub + 1 + 1].tolist() == [0, 1, 0]  # subtitle_font starts with "none"
    assert matrix.sum() == 2 + 2 + 2 and SpecVocab.from_dict(VOCAB.to_dict()).tags == TAGS
    # Moods and style read from the words come first, as their own tokens.
    assert encode_text("Outlaws", "western comic")[1:3] != encode_text("Outlaws", "")[1:3]


def test_every_token_sits_in_its_field_block():
    mask = VOCAB.field_mask()
    spec = DesignSpec(text="x", font="bangers-400", palette="classic-hero")
    for step, token in enumerate(VOCAB.encode(spec)):
        assert mask[step, token]
    assert mask.sum(axis=1).tolist() == [len(f.values) for f in VOCAB.fields]


def test_text_encoding():
    ids = encode_text("Iron Comet", "Sci-Fi")
    assert ids[0] == BOS and ids[-1] == EOS
    assert len(encode_text("x" * 500)) == 112


def test_sampling_only_emits_valid_and_allowed_tokens():
    torch.manual_seed(0)
    model = LayoutModel(LayoutConfig(**PRESETS["tiny"]), VOCAB).eval()
    allowed = torch.ones(VOCAB.size, dtype=torch.bool)
    font = VOCAB.index["font"]
    allowed[VOCAB.offsets[font] + VOCAB.fonts.index("rye-400")] = False
    src = torch.tensor([encode_text("The Iron Comet")] * 16)
    out = model.sample(src, temperature=1.5, allowed=allowed)
    mask = torch.from_numpy(VOCAB.field_mask())
    for row in out:
        assert all(mask[i, t] for i, t in enumerate(row.tolist()))
        assert VOCAB.decode(row.tolist(), "The Iron Comet").font != "rye-400"


def test_training_smoke_run_and_predictor(tiny_dataset, tmp_path):
    from logo_wizard.models.layout.predict import LayoutPredictor
    from logo_wizard.train import layout

    run = tmp_path / "run"
    args = ["--preset", "tiny", "--data", str(tiny_dataset), "--epochs", "2", "--batch-size", "8", "--device", "cpu", "--run", str(run)]
    assert layout.main(args) == 0
    assert (run / "best.pt").exists() and (run / "last.pt").exists()
    assert len((run / "log.csv").read_text().splitlines()) == 3
    # A second call resumes from last.pt, which already finished every epoch.
    assert layout.main(args) == 0
    specs = LayoutPredictor(run / "best.pt", installed_fonts={"bangers-400"}).suggest("Star Patrol", "sci-fi", n=4, seed=1)
    assert len(specs) == 4 and all(s.font == "bangers-400" and s.text == "Star Patrol" for s in specs)
