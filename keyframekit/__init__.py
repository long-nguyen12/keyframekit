"""Extract representative keyframes from video files."""

from .clustering import CLUSTERING_METHODS, cluster_embeddings, select_representative_frame
from .extractor import extract_keyframes
from .vision import (
    DEFAULT_MODEL_NAMES,
    embed_frames,
    extract_image_features,
    load_vision_model,
)

__all__ = [
    "CLUSTERING_METHODS",
    "DEFAULT_MODEL_NAMES",
    "cluster_embeddings",
    "embed_frames",
    "extract_image_features",
    "extract_keyframes",
    "load_vision_model",
    "select_representative_frame",
]

__version__ = "0.1.0"
