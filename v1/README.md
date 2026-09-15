# Python Image Server

Use a **virtual environment** for this project. 
From your project directory:

```bash
cd python

# Check that Python 3 is available
python3 --version

# Create an isolated environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Upgrade pip inside the virtual environment
python -m pip install --upgrade pip

# Now install the dependencies:
python -m pip install fastapi "uvicorn[standard]" diffusers transformers accelerate

# Install PyTorch separately:
python -m pip install torch torchvision torchaudio

# install project libs
python -m pip install -r requirements.txt
```

PyTorch’s current Apple setup uses the MPS backend for GPU acceleration on Apple-silicon Macs. 

If the command-line tools are not installed, run:

```bash
xcode-select --install
```

Then verify MPS:

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())

if torch.backends.mps.is_available():
    print(torch.ones(1, device="mps"))
PY
```

You should see `MPS available: True` and a tensor on `mps:0`.

Then start the server while the environment is active:

```bash
python imageserver.py
```

## Example request:

Test it from another Terminal window:

```bash
curl http://127.0.0.1:8188/health
```

```bash
curl -X POST "http://127.0.0.1:8188/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cinematic photograph of a red fox standing in a snowy forest at sunrise, soft golden light, highly detailed",
    "negative_prompt": "blurry, low quality, distorted, watermark, text",
    "seed": 12345,
    "steps": 8,
    "guidance_scale": 0.0,
    "width": 1024,
    "height": 1024
  }' \
  -o response.json
jq -r '.image_base64' response.json | base64 --decode > generated.png
```

# Run tests

The test boots its own instance of the app (loading the real model) via FastAPI's `TestClient` -
you do not need `imageserver.py` already running in another terminal first.

```bash
source .venv/bin/activate
pytest -m slow -v test_imageserver.py
```

When finished:

```bash
deactivate
```


# Cache of models

`DiffusionPipeline.from_pretrained(MODEL_ID, ...)` downloads the model the first time and stores it in Hugging Face’s local cache. On later runs, it reuses that cached copy rather than downloading the entire model again. <citation src="2,5"></citation>

On macOS, the default cache is typically:

```text
~/.cache/huggingface/hub
```

The first startup can therefore take a while and consume substantial disk space. Subsequent startups still need to load the model from disk into RAM/GPU memory, but they should not need to download it again.

You can inspect the cache with:

```bash
du -sh ~/.cache/huggingface/hub
```

With newer Hugging Face tooling, you can also run:

```bash
hf cache ls
```

The model cache is separate from your Python virtual environment, so deleting `.venv` will not normally delete the downloaded model.

If you want to store Hugging Face models somewhere else—for example, on an external SSD—set `HF_HOME` before starting the server:

```bash
export HF_HOME="/Volumes/MySSD/huggingface"
python imageserver.py
```

To make that permanent for zsh:

```bash
echo 'export HF_HOME="/Volumes/MySSD/huggingface"' >> ~/.zshrc
source ~/.zshrc
```

You can confirm where the model files are being resolved from by running:

```bash
python - <<'PY'
from huggingface_hub import scan_cache_dir

info = scan_cache_dir()
for repo in info.repos:
    print(repo.repo_id, repo.repo_path)
PY
```

The model will be downloaded again if you:

- Delete the Hugging Face cache.
- Change to a different model ID, such as switching between the FP8 and regular models.
- Use a different Hugging Face cache directory.
- Request a newer model revision whose files are not already cached.
- Run in an environment or container that does not preserve the previous home directory.
