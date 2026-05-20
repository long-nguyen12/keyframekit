"""Vision-model loading and embedding helpers."""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAMES = {
    "clip": "openai/clip-vit-base-patch32",
    "siglip": "google/siglip-base-patch16-224",
    "dinov2": "facebook/dinov2-base",
}


def resolve_model_name(model_name: str | None = None, model_backend: str = "clip") -> str:
    """Resolve a backend alias or explicit HuggingFace model id."""
    if model_name:
        return model_name
    return DEFAULT_MODEL_NAMES.get(model_backend, model_backend)


def load_vision_model(
    model_name: str | None = None,
    device=None,
    *,
    model_backend: str = "clip",
):
    """Load a HuggingFace vision model and image processor.

    Args:
        model_name: HuggingFace model identifier. If omitted, a default is
            selected from ``model_backend``.
        device: Target device. Defaults to CUDA if available, else CPU.
        model_backend: Backend alias used only when ``model_name`` is omitted.
            Built-in aliases are ``clip``, ``siglip``, and ``dinov2``.

    Returns:
        Tuple of ``(processor, model)`` ready for inference.
    """
    import torch
    from transformers import AutoImageProcessor, AutoModel

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved_name = resolve_model_name(model_name, model_backend)
    processor = AutoImageProcessor.from_pretrained(resolved_name)
    model = AutoModel.from_pretrained(resolved_name)
    model.eval()
    model.to(device)
    return processor, model


def embed_frames(
    frames: list,
    processor,
    model,
    device,
    *,
    normalize: bool = True,
) -> object | None:
    """Return image embeddings for a list of RGB frames."""
    import torch
    import torch.nn.functional as F

    if not frames:
        return None

    inputs = processor(images=frames, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        features = _model_image_features(model, inputs)

    if normalize:
        features = F.normalize(features, dim=-1)
    return features


def extract_image_features(
    model,
    processor,
    image,
    device,
    *,
    normalize: bool = True,
) -> object | None:
    """Extract a single image feature vector from a supported vision model."""
    try:
        features = embed_frames([image], processor, model, device, normalize=normalize)
        if features is None:
            return None
        return features.squeeze(0)
    except Exception:
        logger.exception("Feature extraction failed")
        return None


def _model_image_features(model, inputs: dict) -> object:
    import torch

    if hasattr(model, "get_image_features"):
        features = model.get_image_features(**inputs)
        if isinstance(features, torch.Tensor):
            return features
        if hasattr(features, "pooler_output"):
            return features.pooler_output

    outputs = model(**inputs)
    if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
        return outputs.image_embeds
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state[:, 0, :]

    raise RuntimeError("Unsupported vision model output format")
