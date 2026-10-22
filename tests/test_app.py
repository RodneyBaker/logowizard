import json

import pytest

from logo_wizard.render.shaping import FONT_DIR

pytestmark = pytest.mark.skipif(not any(FONT_DIR.glob("*/*.ttf")), reason="fonts not downloaded (run logo-wizard-fonts)")


class FakeLayout:
    """Stands in for a trained model: always proposes the first font and palette."""

    name = "fake layout"

    def __init__(self, renderer):
        self.font = sorted(renderer.faces)[0]
        self.palette = sorted(renderer.palettes)[0]

    def suggest(self, text, words="", subtitle="", n=8, seed=None, **_):
        from logo_wizard.render.spec import DesignSpec

        return [DesignSpec(text=text, subtitle=subtitle, font=self.font, palette=self.palette, seed=i) for i in range(n)]


@pytest.fixture(scope="module")
def renderer():
    from logo_wizard.render.draw import Renderer

    return Renderer()


def make_client(renderer, feedback_dir, monkeypatch=None, layout=None):
    from fastapi.testclient import TestClient

    from logo_wizard.app.server import create_app

    if monkeypatch is None:
        return TestClient(create_app(renderer, feedback_dir=feedback_dir, use_models=False))
    monkeypatch.setattr("logo_wizard.infer.runtime.load_layout", lambda fonts, palettes: layout)
    monkeypatch.setattr("logo_wizard.infer.runtime.load_stylizer", lambda r: None)
    return TestClient(create_app(renderer, feedback_dir=feedback_dir))


@pytest.fixture()
def client(renderer, tmp_path):
    return make_client(renderer, tmp_path)


def test_status_and_options(client):
    s = client.get("/api/status").json()
    assert s["layout"] is None and s["stylizer"] is None and s["fonts"] > 300
    o = client.get("/api/options").json()
    assert {"fonts", "palettes", "warps", "textures"} <= set(o)


def test_generate_mixes_model_and_rules(renderer, tmp_path, monkeypatch):
    c = make_client(renderer, tmp_path, monkeypatch, FakeLayout(renderer))
    data = c.post("/api/generate", json={"text": "Star Patrol", "count": 6, "seed": 3}).json()
    sources = [x["source"] for x in data["candidates"]]
    assert sorted(sources) == ["model"] * 3 + ["rules"] * 3
    rules_only = c.post("/api/generate", json={"text": "Star Patrol", "count": 4, "source": "rules"}).json()
    assert {x["source"] for x in rules_only["candidates"]} == {"rules"}
    assert c.get("/api/status").json()["layout"] == "fake layout"

    pair = c.get("/api/blind", params={"text": "Star Patrol"}).json()
    assert pair["a"] and pair["b"] and pair["text"] == "Star Patrol"
    result = c.post("/api/blind", json={"pair_id": pair["pair_id"], "pick": "a"}).json()
    assert result["model"] + result["rules"] == 1
    assert c.post("/api/blind", json={"pair_id": pair["pair_id"], "pick": "a"}).status_code == 400
    lines = (tmp_path / "feedback.jsonl").read_text().splitlines()
    assert json.loads(lines[-1])["source"].startswith("blind-")


def test_feedback_is_appended(client, tmp_path):
    spec = client.post("/api/generate", json={"text": "Kid Rocket", "count": 1}).json()["candidates"][0]["spec"]
    assert client.post("/api/feedback", json={"spec": spec, "vote": 1, "words": "pulp"}).json()["ok"]
    assert client.post("/api/feedback", json={"spec": spec, "vote": -1}).json()["ok"]
    assert client.post("/api/feedback", json={"spec": spec, "vote": 5}).status_code == 400
    votes = [json.loads(l)["vote"] for l in (tmp_path / "feedback.jsonl").read_text().splitlines()]
    assert votes == [1, -1]


def test_blind_needs_a_model(client):
    assert client.get("/api/blind").status_code == 409


def test_stylize_without_model_is_refused(client):
    spec = client.post("/api/generate", json={"text": "Kid Rocket", "count": 1}).json()["candidates"][0]["spec"]
    assert client.post("/api/render", json={"spec": spec, "width": 200, "height": 125, "stylize": True}).status_code == 409


def test_free_port_skips_a_busy_port():
    import socket

    from logo_wizard.app.server import free_port

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        busy = s.getsockname()[1]
        assert free_port("127.0.0.1", busy) != busy


def test_version_matches_pyproject(client):
    import re

    from logo_wizard import __version__, paths

    pyproject = (paths.DATA_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'^version = "(.+)"', pyproject, re.M).group(1) == __version__
    assert client.get("/api/status").json()["version"] == __version__
