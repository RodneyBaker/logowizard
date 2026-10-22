"""Local web app: type a title, get a grid of candidate logos.

Usage:
    logo-wizard-app            # serves http://127.0.0.1:8765 and opens the browser

Everything runs on this machine. With no trained model the candidates come
from the procedural title grammar.
"""

import argparse
import base64
import io
import threading
import webbrowser
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from ..render.draw import Renderer
from ..render.grammar import read_style, sample_spec
from ..render.spec import DesignSpec

STATIC = Path(__file__).with_name("static")


class GenerateRequest(BaseModel):
    text: str
    words: str = ""
    style: str | None = None
    subtitle: str = ""
    count: int = 8
    seed: int | None = None


class RenderRequest(BaseModel):
    spec: dict
    width: int = 1024
    height: int = 640


def _png(image: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(image, "RGBA").save(buf, "PNG")
    return buf.getvalue()


def create_app(renderer: Renderer | None = None) -> FastAPI:
    app = FastAPI(title="Logo Wizard")
    state = {"renderer": renderer}

    def get_renderer() -> Renderer:
        if state["renderer"] is None:
            state["renderer"] = Renderer()
        return state["renderer"]

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.post("/api/generate")
    def generate(req: GenerateRequest) -> dict:
        text = req.text.strip()
        if not text:
            raise HTTPException(400, "Type a title first.")
        r = get_renderer()
        rng = np.random.default_rng(req.seed)
        style = req.style or read_style(req.words)
        out = []
        for _ in range(max(1, min(req.count, 16))):
            spec = sample_spec(text, r.faces, r.palettes, rng, style=style, style_words=req.words, subtitle=req.subtitle)
            result = r.render(spec, 512, 320)
            out.append({"spec": spec.to_dict(), "png": base64.b64encode(_png(result.image)).decode()})
        return {"style": style, "candidates": out}

    @app.post("/api/render")
    def render(req: RenderRequest) -> Response:
        spec = DesignSpec.from_dict(req.spec)
        width = max(64, min(req.width, 4096))
        height = max(64, min(req.height, 4096))
        result = get_renderer().render(spec, width, height)
        return Response(_png(result.image), media_type="image/png")

    @app.get("/api/options")
    def options() -> dict:
        r = get_renderer()
        return {
            "fonts": [{"id": f.face_id, "family": f.family, "class": f.font_class, "moods": f.moods} for f in r.faces.values()],
            "palettes": [{"name": p.name, "family": p.family, "fill": p.fill, "outlines": p.outlines} for p in r.palettes.values()],
        }

    return app


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(f"http://{args.host}:{args.port}/")).start()
    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0
