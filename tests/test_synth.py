import json

import numpy as np
import pytest

from logo_wizard.render.shaping import FONT_DIR
from logo_wizard.synth import filters, generate


def _disc(h=64, w=96, fill=(255, 220, 0), edge=(20, 20, 20), edge_width=4):
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - h / 2, xx - w / 2)
    mask = np.where(r < 20, 255, 0).astype(np.uint8)
    image = np.zeros((h, w, 4), np.uint8)
    image[r < 20 + edge_width] = edge + (255,)
    image[r < 20] = fill + (255,)
    return image, mask


def test_contrast_ratio_matches_wcag():
    assert filters.contrast_ratio(np.array([0, 0, 0]), np.array([255, 255, 255])) == pytest.approx(21.0, rel=1e-3)
    assert filters.contrast_ratio(np.array([90, 90, 90]), np.array([90, 90, 90])) == pytest.approx(1.0)


def test_edge_contrast_sees_the_outline():
    image, mask = _disc()
    assert filters.edge_contrast(image, mask) > 10
    muddy, mask = _disc(fill=(120, 120, 120), edge=(130, 130, 130))
    assert filters.edge_contrast(muddy, mask) < 1.2


def test_edge_contrast_without_outline_uses_best_backdrop():
    image, mask = _disc(edge_width=0)
    assert filters.edge_contrast(image, mask) > 10


def test_coverage_and_dhash():
    image, mask = _disc()
    assert filters.coverage(image, mask) == 1.0
    knocked = image.copy()
    knocked[:, :48, 3] = 0
    assert filters.coverage(knocked, mask) == pytest.approx(0.5, abs=0.05)
    assert filters.dhash(image) == filters.dhash(image.copy())
    assert filters.dhash(image) != filters.dhash(knocked)


def test_style_words_are_varied_and_deterministic():
    words = {generate.style_words_for("comic", np.random.default_rng([0, i])) for i in range(200)}
    assert "" in words and len(words) > 10
    assert generate.style_words_for("novel", np.random.default_rng([3, 9])) == generate.style_words_for("novel", np.random.default_rng([3, 9]))


def test_duplicates_keep_the_first_copy(tmp_path):
    stats = [{"shard": 1, "hashes": [["00001000", "ab"]]}, {"shard": 0, "hashes": [["00000001", "ab"], ["00000002", "cd"]]}]
    assert generate.write_duplicates(tmp_path, stats) == 1
    assert (tmp_path / "duplicates.txt").read_text().split() == ["00001000"]


@pytest.mark.skipif(not any(FONT_DIR.glob("*/*.ttf")), reason="fonts not downloaded (run logo-wizard-fonts)")
def test_generator_writes_resumable_shards(tmp_path):
    args = ["--count", "12", "--out", str(tmp_path), "--workers", "1", "--no-ocr", "--size", "128x80"]
    assert generate.main(args) == 0
    shard = tmp_path / "shard_00000"
    records = [json.loads(line) for line in (shard / "specs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert records and (shard / "DONE").exists()
    first = records[0]
    assert (shard / f"{first['id']}.png").exists() and (shard / f"{first['id']}_mask.png").exists()
    assert first["spec"]["text"] == first["text"]
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["kept"] == len(records)
    # Same seed, same samples: a re-run skips the finished shard and changes nothing.
    before = (shard / "specs.jsonl").read_text(encoding="utf-8")
    assert generate.main(args) == 0
    assert (shard / "specs.jsonl").read_text(encoding="utf-8") == before
    assert len(list(generate.iter_records(tmp_path))) == len(records) - summary["duplicates"]
