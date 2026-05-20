"""Compatibility wrapper for the packaged keyframekit extractor module."""

from keyframekit.extractor import (  # noqa: F401
    _embed_frames,
    _extract_image_features,
    _read_frames_at_indices,
    _reorder_and_rename_images,
    extract_keyframes,
    load_vision_model,
    select_representative_frame,
)
