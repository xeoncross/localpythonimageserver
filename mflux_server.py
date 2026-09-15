#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["mflux", "fastapi", "uvicorn", "pydantic"]
# ///
"""
Minimal mflux HTTP server.

Loads the model once at startup and keeps it resident, so each request pays
only for denoising, not for weight loading and quantization.

Run:
    uv run mflux_server.py
    # or: uvicorn mflux_server:app --host 127.0.0.1 --port 8000 --workers 1

Call:
    curl -X POST http://127.0.0.1:8000/generate \
         -H 'Content-Type: application/json' \
         -d '{"prompt": "A puffin standing on a cliff", "steps": 6}' \
         --output out.png
"""

import contextlib
import os
import random
import tempfile
import threading

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from mflux.models.z_image import ZImageTurbo

# --- configuration ----------------------------------------------------------

QUANTIZE = 8          # 8 or 4. 4 is faster and smaller, slightly softer output.
WARMUP_STEPS = 2      # cheap first generation to compile Metal kernels
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_STEPS = 6 # 8

# Metal work is serialized anyway, and two concurrent generations will fight
# over unified memory. One request at a time, queued.
_lock = threading.Lock()
_model = None


# --- lifecycle --------------------------------------------------------------

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    print(f"loading Z-Image-Turbo (quantize={QUANTIZE})...")
    _model = ZImageTurbo(quantize=QUANTIZE)

    # First generation is always slow: kernel compilation, lazy graph setup,
    # text encoder warm-up. Pay it at startup instead of on a user's request.
    print("warming up...")
    _model.generate_image(
        prompt="warmup",
        seed=0,
        num_inference_steps=WARMUP_STEPS,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
    )
    print("ready")
    yield
    _model = None


app = FastAPI(title="mflux server", lifespan=lifespan)


# --- API --------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str
    seed: int | None = None
    steps: int = Field(default=DEFAULT_STEPS, ge=1, le=50)
    width: int = Field(default=DEFAULT_WIDTH, ge=256, le=2048)
    height: int = Field(default=DEFAULT_HEIGHT, ge=256, le=2048)


@app.get("/health")
def health():
    return {"status": "ready" if _model is not None else "loading"}


@app.post("/generate")
async def generate(req: GenerateRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="model still loading")

    seed = req.seed if req.seed is not None else random.randint(0, 2**31 - 1)

    with _lock:
        result = _model.generate_image(
            prompt=req.prompt,
            seed=seed,
            num_inference_steps=req.steps,
            width=req.width,
            height=req.height,
        )

    png = _to_png_bytes(result)
    return Response(
        content=png,
        media_type="image/png",
        headers={"X-Seed": str(seed), "X-Steps": str(req.steps)},
    )


def _to_png_bytes(result) -> bytes:
    """mflux returns a wrapper object. Prefer the in-memory PIL image if it is
    exposed; otherwise fall back to the documented .save() method."""
    pil = getattr(result, "image", None)
    if pil is not None:
        import io
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return buf.getvalue()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        path = tmp.name
    try:
        result.save(path)
        with open(path, "rb") as fh:
            return fh.read()
    finally:
        os.unlink(path)


if __name__ == "__main__":
    # workers=1 is deliberate: more workers means more copies of the weights.
    uvicorn.run(app, host="127.0.0.1", port=8000, workers=1)