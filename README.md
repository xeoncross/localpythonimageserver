# mflux image server

A local image-generation server in Python that takes 

Requires https://docs.astral.sh/uv/

Start the server (with an optional default model to load):

`MFLUX_MODEL=flux2-klein-4b uv run mflux_server.py`

# Make requests:

Fastest (36sec), least memory (29GB)
```
curl -X POST http://127.0.0.1:8000/generate \
    -H 'Content-Type: application/json' \
    -d '{"prompt": "A puffin on a cliff", "model": "flux2-klein-4b"}' \   
    --output flux2-klein-4b.png
```

Second place:
```
curl -X POST http://127.0.0.1:8000/generate \
    -H 'Content-Type: application/json' \
    -d '{"prompt": "A puffin on a cliff", "model": "flux2-klein-4b-q4"}' \
    --output flux2-klein-4b-q4.png
```

```
curl -X POST http://127.0.0.1:8000/generate \
    -H 'Content-Type: application/json' \
    -d '{"prompt": "A puffin on a cliff", "model": "z-image-turbo"}' \
    --output z-image-turbo.png
```


# Alternatives

Draw Things (closed-source) provides a nice UI editor and has a [http server](https://gist.github.com/xeoncross/a84b2ebcf48f13d4f9cb2f87fba0407b)

Currently I'm using Draw Things as it only uses 30% of the memory (10GB vs 30GB)


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