"""Local web app: type a title, get a grid of candidate logos, refine, export.

Usage:
    logo-wizard-app            # serves http://127.0.0.1:8765 and opens the browser

Everything runs on this machine. Candidates come from the trained layout
model when local/models or local/runs has one, mixed with the procedural
grammar; without a model they all come from the grammar. Thumbs up/down
and blind-test picks are saved in local/feedback/ for fine-tuning.
"""

import argparse
import base64
import io
import json
import socket
import threading
import time
import uuid
import webbrowser
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from .. import paths
from ..render.draw import Renderer
from ..render.grammar import read_style, sample_spec
from ..render.spec import ARRANGEMENTS, CASES, FILL_MODES, TEXTURES, WARPS, DesignSpec

STATIC = Path(__file__).with_name("static")
FEEDBACK_DIR = paths.FEEDBACK_DIR
PREVIEW = (512, 320)


class GenerateRequest(BaseModel):
    text: str
    words: str = ""
    style: str | None = None
    subtitle: str = ""
    count: int = 12
    source: str = "auto"  # auto, mix, model, rules
    seed: int | None = None


class RenderRequest(BaseModel):
    spec: dict
    width: int = 1024
    height: int = 640
    stylize: bool = False


class FeedbackRequest(BaseModel):
    spec: dict
    words: str = ""
    vote: int  # 1 or -1
    source: str = ""


class PickRequest(BaseModel):
    pair_id: str
    pick: str  # "a" or "b"


def _png(image: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(image, "RGBA").save(buf, "PNG")
    return buf.getvalue()


def _b64(image: np.ndarray) -> str:
    return base64.b64encode(_png(image)).decode()


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def create_app(renderer: Renderer | None = None, feedback_dir: Path = FEEDBACK_DIR, use_models: bool = True) -> FastAPI:
    app = FastAPI(title="Logo Wizard")
    state: dict = {"renderer": renderer, "layout": None, "stylizer": None, "loaded": not use_models, "pairs": {}}
    lock = threading.Lock()

    def get_renderer() -> Renderer:
        if state["renderer"] is None:
            state["renderer"] = Renderer()
        return state["renderer"]

    def models():
        with lock:
            if not state["loaded"]:
                from ..infer.runtime import load_layout, load_stylizer

                r = get_renderer()
                try:
                    state["layout"] = load_layout(set(r.faces), set(r.palettes))
                except Exception as err:  # a broken or mismatched checkpoint must not take the app down
                    print(f"Layout model not loaded: {err}")
                try:
                    state["stylizer"] = load_stylizer(r)
                except Exception as err:
                    print(f"Stylizer not loaded: {err}")
                state["loaded"] = True
        return state["layout"], state["stylizer"]

    def candidates(text, words, style, subtitle, count, source, seed) -> list[tuple[DesignSpec, str]]:
        r = get_renderer()
        layout, _ = models()
        rng = np.random.default_rng(seed)
        if source == "auto":
            source = "mix" if layout else "rules"
        n_model = 0 if layout is None or source == "rules" else count if source == "model" else count // 2
        out: list[tuple[DesignSpec, str]] = []
        if n_model:
            for spec in layout.suggest(text, words, subtitle, n=n_model, seed=int(rng.integers(2**31))):
                if style:
                    spec.style = style
                out.append((spec, "model"))
        style = style or read_style(words)
        while len(out) < count:
            out.append((sample_spec(text, r.faces, r.palettes, rng, style=style, style_words=words, subtitle=subtitle), "rules"))
        order = rng.permutation(len(out))
        return [out[i] for i in order]

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/status")
    def status() -> dict:
        layout, stylizer = models()
        return {
            "layout": getattr(layout, "name", None),
            "stylizer": getattr(stylizer, "name", None),
            "fonts": len(get_renderer().faces),
        }

    @app.post("/api/generate")
    def generate(req: GenerateRequest) -> dict:
        text = req.text.strip()
        if not text:
            raise HTTPException(400, "Type a title first.")
        r = get_renderer()
        out = []
        for spec, source in candidates(text, req.words, req.style, req.subtitle, max(1, min(req.count, 16)), req.source, req.seed):
            out.append({"spec": spec.to_dict(), "source": source, "png": _b64(r.render(spec, *PREVIEW).image)})
        return {"style": req.style or read_style(req.words), "candidates": out}

    @app.post("/api/render")
    def render(req: RenderRequest) -> Response:
        spec = DesignSpec.from_dict(req.spec)
        width = max(64, min(req.width, 4096))
        height = max(64, min(req.height, 4096))
        if req.stylize:
            _, stylizer = models()
            if stylizer is None:
                raise HTTPException(409, "No stylizer is trained yet.")
            image = stylizer([spec], (width, height))[0]
        else:
            image = get_renderer().render(spec, width, height).image
        return Response(_png(image), media_type="image/png")

    @app.post("/api/feedback")
    def feedback(req: FeedbackRequest) -> dict:
        if req.vote not in (1, -1):
            raise HTTPException(400, "vote must be 1 or -1")
        _append(feedback_dir / "feedback.jsonl", {"time": time.time(), "vote": req.vote, "words": req.words, "source": req.source, "spec": req.spec})
        return {"ok": True}

    @app.get("/api/blind")
    def blind_pair(text: str = "", words: str = "") -> dict:
        """Two takes on one title, one from the model and one from the rules, in random order."""
        layout, _ = models()
        if layout is None:
            raise HTTPException(409, "Train or export a layout model first; the blind test compares it with the rules.")
        r = get_renderer()
        rng = np.random.default_rng()
        if not text:
            from ..synth.generate import load_phrases, style_words_for

            phrases = state.setdefault("phrases", load_phrases())
            text, style, _ = phrases[int(rng.integers(len(phrases)))]
            words = style_words_for(style, rng)
        model_spec = layout.suggest(text, words, n=1, seed=int(rng.integers(2**31)))[0]
        rules_spec = sample_spec(text, r.faces, r.palettes, rng, style=read_style(words, model_spec.style), style_words=words)
        model_first = bool(rng.random() < 0.5)
        a, b = (model_spec, rules_spec) if model_first else (rules_spec, model_spec)
        pair_id = uuid.uuid4().hex
        state["pairs"][pair_id] = {"text": text, "words": words, "model": "a" if model_first else "b", "a": a.to_dict(), "b": b.to_dict()}
        return {"pair_id": pair_id, "text": text, "words": words, "a": _b64(r.render(a, *PREVIEW).image), "b": _b64(r.render(b, *PREVIEW).image)}

    @app.post("/api/blind")
    def blind_pick(req: PickRequest) -> dict:
        pair = state["pairs"].pop(req.pair_id, None)
        if pair is None or req.pick not in ("a", "b"):
            raise HTTPException(400, "Unknown pair or pick.")
        winner = "model" if req.pick == pair["model"] else "rules"
        _append(feedback_dir / "blind.jsonl", {"time": time.time(), "winner": winner, **pair})
        # A blind pick is also a preference: the chosen spec counts as a thumbs-up.
        _append(feedback_dir / "feedback.jsonl", {"time": time.time(), "vote": 1, "words": pair["words"], "source": "blind-" + winner, "spec": pair[req.pick]})
        return {"winner": winner, **blind_tally()}

    @app.get("/api/blind/tally")
    def blind_tally() -> dict:
        path = feedback_dir / "blind.jsonl"
        model = rules = 0
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    if json.loads(line)["winner"] == "model":
                        model += 1
                    else:
                        rules += 1
        return {"model": model, "rules": rules}

    @app.get("/api/options")
    def options() -> dict:
        r = get_renderer()
        return {
            "fonts": [{"id": f.face_id, "family": f.family, "weight": f.weight, "class": f.font_class} for f in sorted(r.faces.values(), key=lambda f: (f.family, f.weight))],
            "palettes": [{"name": p.name, "family": p.family, "fill": p.fill, "outlines": p.outlines} for p in r.palettes.values()],
            "warps": list(WARPS), "arrangements": list(ARRANGEMENTS), "cases": list(CASES), "fill_modes": list(FILL_MODES), "textures": list(TEXTURES),
        }

    return app


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    port = free_port(args.host, args.port)
    if port != args.port:
        print(f"Port {args.port} is busy; using {port}.")
    url = f"http://{args.host}:{port}/"
    print(f"Logo Wizard at {url}  (local files: {paths.LOCAL_ROOT})")
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(), host=args.host, port=port)
    return 0


def free_port(host: str, start: int, tries: int = 20) -> int:
    """The first port from start that nothing is listening on (a second copy may already run)."""
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    return start
