"""
Integration test for imageserver.py - loads the REAL Z-Image-Turbo model and
runs REAL inference (no mocking the model). Verifies `GET /health` reports a
loaded model and `POST /generate` returns a genuine PNG of the requested size.

This is slow (model load is 10-30+ seconds, then real inference on top), so
it is marked `slow` and excluded from the default `pytest` run by
`python/pytest.ini`'s `addopts = -m "not slow"`.

Setup:
    pip install -r python/requirements.txt

Run (opt-in - from the `python/` directory, or `python/test_imageserver.py`
from the repo root):
    pytest -m slow -v test_imageserver.py
"""

import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from imageserver import app

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def client():
    # Entering the context manager runs FastAPI's startup event (real model
    # load) once for every test in this module, rather than once per test.
    with TestClient(app) as c:
        yield c


def test_health_reports_the_loaded_model(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"]


def test_generate_returns_a_real_png_of_the_requested_size(client: TestClient) -> None:
    width, height = 256, 256
    response = client.post(
        "/generate",
        json={
            "prompt": "a red bicycle leaning against a brick wall",
            "steps": 4,
            "guidance_scale": 0.0,
            "width": width,
            "height": height,
        },
    )
    assert response.status_code == 200
    body = response.json()

    metadata = body["metadata"]
    assert metadata["prompt"] == "a red bicycle leaning against a brick wall"
    assert metadata["width"] == width
    assert metadata["height"] == height
    assert metadata["model"]
    assert metadata["generation_seconds"] > 0

    raw = base64.b64decode(body["image_base64"])
    image = Image.open(io.BytesIO(raw))
    assert image.format == "PNG"
    assert image.size == (width, height)
