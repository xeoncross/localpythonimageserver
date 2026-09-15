"""
Persistent local text-to-image inference server for Z-Image-Turbo.

Run this ONCE and leave it running. Call it repeatedly from any client
(TypeScript, Python, curl, etc.) instead of spawning a new process per
image. Loading the model takes 10-30+ seconds; a warm server then generates
each image in roughly 1-3 seconds on a decent CUDA GPU, but multiple
minutes on Apple Silicon's MPS backend (this file's default DEVICE) - a
1024x1024, 8-step image measured ~138s on real Apple Silicon hardware.

Setup: see README.md (use its virtual environment, not a bare
`pip install --break-system-packages`).

Run:
    python imageserver.py
    # listens on http://127.0.0.1:8188
"""

import base64
import io
import time
from typing import Optional

import torch
from diffusers import DiffusionPipeline
from fastapi import FastAPI
from pydantic import BaseModel

# Swap for "unsloth/Z-Image-Turbo-FP8" if you're VRAM-limited (~half the memory,
# minor quality trade-off).
MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
# MODEL_ID = "unsloth/Z-Image-Turbo-FP8"

# "cuda" for NVIDIA GPUs, "mps" for Apple Silicon, "cpu" as a slow last resort.
DEVICE = "mps"

app = FastAPI()
pipe: Optional[DiffusionPipeline] = None


@app.on_event("startup")
def load_model():
    global pipe
    print(f"Loading {MODEL_ID} onto {DEVICE} ...")
    pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe = pipe.to(DEVICE)
    print("Model loaded. Ready to generate.")


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    steps: int = 8
    guidance_scale: float = 0.0
    width: int = 1024
    height: int = 1024


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}


@app.post("/generate")
def generate(req: GenerateRequest):
    seed = req.seed if req.seed is not None else int(torch.seed())
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    start = time.time()
    result = pipe(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        num_inference_steps=req.steps,
        guidance_scale=req.guidance_scale,
        width=req.width,
        height=req.height,
        generator=generator,
    )
    elapsed = time.time() - start

    image = result.images[0]
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "image_base64": image_b64,
        "metadata": {
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "seed": seed,
            "steps": req.steps,
            "guidance_scale": req.guidance_scale,
            "width": req.width,
            "height": req.height,
            "model": MODEL_ID,
            "generation_seconds": round(elapsed, 2),
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8188)