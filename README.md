# keyframekit

Extract representative keyframes from video files using configurable HuggingFace
vision models and clustering techniques. The default remains CLIP embeddings
with spectral clustering.

## Installation

```bash
pip install -e .
pip install -e ".[plot,dev]"
```

## Python API

```python
import torch
from keyframekit import extract_keyframes, load_vision_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor, model = load_vision_model(model_backend="clip", device=device)

output_dir = extract_keyframes(
    "my_video.mp4",
    model,
    processor,
    chunk_count=10,
    samples_per_chunk=8,
    n_clusters=2,
    cluster_method="spectral",
    output_dir="./keyframes",
    device=device,
)
```

Built-in vision backend aliases:

- `clip` -> `openai/clip-vit-base-patch32`
- `siglip` -> `google/siglip-base-patch16-224`
- `dinov2` -> `facebook/dinov2-base`

You can also pass any compatible HuggingFace model id:

```python
processor, model = load_vision_model(
    model_name="facebook/dinov2-large",
    device=device,
)
```

Built-in clustering methods:

- `spectral`
- `kmeans`
- `agglomerative`

## CLI

```bash
keyframekit embed my_video.mp4 --vision-backend clip --cluster-method spectral
keyframekit embed my_video.mp4 --vision-backend siglip --cluster-method kmeans
keyframekit embed my_video.mp4 --model facebook/dinov2-base --cluster-method agglomerative
```

`keyframekit clip ...` is kept as an alias for `keyframekit embed ...`.

## Project Layout

```text
keyframekit/
  __init__.py
  cli.py
  clustering.py
  extractor.py
  vision.py
extract_single_video.py
test_keyframekit.py
pyproject.toml
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest -q
```
