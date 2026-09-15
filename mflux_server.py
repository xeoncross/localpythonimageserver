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
    
    MFLUX_MODEL=flux2-klein-4b uv run mflux_server.py

Call: 

    curl -X POST http://127.0.0.1:8000/generate \
        -H 'Content-Type: application/json' \
        -d '{"prompt": "A puffin on a cliff", "model": "flux2-klein-4b"}' \
        --output out.png

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
# None is the fastest
QUANTIZE = None     # 49GB of RAM 90 sec
#QUANTIZE = 8        # 8 or 4. 4 is faster and smaller, slightly softer output.

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
    with _lock:
        _ensure_model(DEFAULT_MODEL)
    print("ready")
    yield


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

import gc
import os
import mlx.core as mx

# --- model registry ---------------------------------------------------------

def _z_image(**kw):
    from mflux.models.z_image import ZImageTurbo
    return ZImageTurbo(**kw)

def _flux2_klein(variant):
    def load(**kw):
        # Verify these two paths with the grep command above.
        from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
        from mflux.models.common.config import ModelConfig
        return Flux2Klein(model_config=getattr(ModelConfig, variant)(), **kw)
    return load

MODELS = {
    # 90s on M1 Max
    "z-image-turbo":  {"load": _z_image, "steps": 6, "quantize": None}, 
    # 35GB, 30s on M1 Max
    "flux2-klein-4b": {"load": _flux2_klein("flux2_klein_4b"), "steps": 4, "quantize": None}, 
    # non-commercial license!
    # "flux2-klein-9b": {"load": _flux2_klein("flux2_klein_9b"), "steps": 4, "quantize": 8},
    # A pre-quantized repo from Hugging Face instead of quantizing at load time:
    # 36GB, 36sec
    "flux2-klein-4b-q4": {
        "load": _flux2_klein("flux2_klein_4b"), 
        "steps": 4,
        "quantize": None, 
        "model_path": "RunPod/FLUX.2-klein-4B-mflux-4bit"
    },
}
DEFAULT_MODEL = os.environ.get("MFLUX_MODEL", "flux2-klein-4b")

_lock = threading.Lock()
_model = None
_model_name = None

def _ensure_model(name: str):
    """Load `name`, unloading the previous model first. Caller holds _lock."""
    global _model, _model_name
    if name == _model_name:
        return
    if name not in MODELS:
        raise HTTPException(400, f"unknown model {name!r}; options: {list(MODELS)}")
    spec = MODELS[name]
    _model, _model_name = None, None
    gc.collect()
    mx.clear_cache()  # hand the freed Metal memory back before loading the next model
    kwargs = {"quantize": spec["quantize"]}
    if spec.get("model_path"):
        kwargs["model_path"] = spec["model_path"]
    print(f"loading {name}...")
    _model = spec["load"](**kwargs)
    _model_name = name
    _model.generate_image(prompt="warmup", seed=0, num_inference_steps=1,
                          width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT)



# --- API --------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str
    model: str | None = None      # omit to use whatever is loaded
    seed: int | None = None
    steps: int | None = Field(default=None, ge=1, le=50)
    width: int = Field(default=DEFAULT_WIDTH, ge=256, le=2048)
    height: int = Field(default=DEFAULT_HEIGHT, ge=256, le=2048)

@app.post("/generate")
async def generate(req: GenerateRequest):   # plain def: runs in a threadpool
    seed = req.seed if req.seed is not None else random.randint(0, 2**31 - 1)
    with _lock:
        name = req.model or _model_name or DEFAULT_MODEL
        _ensure_model(name)
        steps = req.steps or MODELS[name]["steps"]
        result = _model.generate_image(prompt=req.prompt, seed=seed,
                                       num_inference_steps=steps,
                                       width=req.width, height=req.height)
    return Response(content=_to_png_bytes(result), media_type="image/png",
                    headers={"X-Seed": str(seed), "X-Steps": str(steps), "X-Model": name})


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