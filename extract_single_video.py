"""Example: extract keyframes from one video."""

import torch

from keyframekit import extract_keyframes, load_vision_model

VIDEO_PATH = "sample.mp4"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor, model = load_vision_model(
    model_backend="clip",
    device=device,
)

output_dir = extract_keyframes(
    VIDEO_PATH,
    model,
    processor,
    chunk_count=10,
    samples_per_chunk=8,
    n_clusters=2,
    cluster_method="spectral",
    output_dir="./keyframes",
    device=device,
)

print(f"Keyframes saved to: {output_dir}")
