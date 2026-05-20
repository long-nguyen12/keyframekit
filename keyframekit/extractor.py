"""Core frame extraction logic."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from .clustering import select_representative_frame
from .vision import embed_frames, extract_image_features, load_vision_model

logger = logging.getLogger(__name__)


def _read_frames_at_indices(cap, indices: List[int]) -> list:
    """Seek and decode specific frame indices from an open VideoCapture."""
    import cv2

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            logger.debug("Could not read frame at index %d", idx)
            continue
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return frames


def _embed_frames(
    frames: list,
    processor,
    model,
    device,
) -> object | None:
    """Backward-compatible alias for :func:`keyframekit.vision.embed_frames`."""
    return embed_frames(frames, processor, model, device)


def _extract_image_features(model, processor, image, device):
    """Backward-compatible alias for :func:`keyframekit.vision.extract_image_features`."""
    return extract_image_features(model, processor, image, device)


def extract_keyframes(
    video_path: str | Path,
    model,
    processor,
    *,
    chunk_count: int = 10,
    samples_per_chunk: int = 8,
    spectral_clusters: int | None = 2,
    n_clusters: int | None = None,
    cluster_method: str = "spectral",
    output_dir: str | Path | None = None,
    device=None,
) -> str:
    """Extract one representative keyframe per temporal chunk from a video.

    The video is divided into ``chunk_count`` equal-duration segments. For each
    segment, ``samples_per_chunk`` candidate frames are embedded with the
    provided vision model, clustered, and the medoid of the largest cluster is
    saved as the keyframe.
    """
    import cv2
    import torch

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    target_path = (
        Path(output_dir) / video_path.stem
        if output_dir is not None
        else video_path.parent / video_path.stem
    )
    target_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Video has no frames: {video_path}")

    effective_chunks = min(chunk_count, total_frames)
    chunk_size = total_frames / effective_chunks
    cluster_count = n_clusters if n_clusters is not None else spectral_clusters
    if cluster_count is None:
        cluster_count = 2

    logger.info(
        "Extracting %d keyframes from '%s' using %s clustering on %s",
        effective_chunks,
        video_path.name,
        cluster_method,
        device,
    )

    saved = 0
    try:
        for chunk_idx in range(effective_chunks):
            start = int(chunk_idx * chunk_size)
            end = int(min(total_frames, (chunk_idx + 1) * chunk_size))
            if end <= start:
                continue

            if samples_per_chunk <= 1 or (end - start) <= 1:
                indices = [start]
            else:
                step = max(1, (end - start) // samples_per_chunk)
                indices = list(range(start, end, step))[:samples_per_chunk]

            frames = _read_frames_at_indices(cap, indices)
            embeddings = embed_frames(frames, processor, model, device)
            if embeddings is None:
                logger.warning("No embeddings for chunk %d - skipping", chunk_idx)
                continue

            best_frame = select_representative_frame(
                frames,
                embeddings,
                n_clusters=cluster_count,
                cluster_method=cluster_method,
            )
            if best_frame is None:
                continue

            saved += 1
            out_path = target_path / f"{saved}.jpeg"
            cv2.imwrite(str(out_path), cv2.cvtColor(best_frame, cv2.COLOR_RGB2BGR))
    finally:
        cap.release()

    _reorder_and_rename_images(target_path)

    logger.info("Saved %d keyframes to '%s'", saved, target_path)
    return str(target_path)


def _reorder_and_rename_images(directory: str | Path) -> None:
    """Rename JPEG files in ``directory`` to sequential integers by mtime."""
    directory = Path(directory)
    images = sorted(directory.glob("*.jpeg"), key=lambda path: path.stat().st_mtime)
    if not images:
        logger.debug("No JPEG images found to rename in %s", directory)
        return

    tmp_paths = []
    for idx, image_path in enumerate(images, start=1):
        tmp = image_path.with_name(f"__tmp_{idx:06d}.jpeg")
        image_path.rename(tmp)
        tmp_paths.append(tmp)

    for idx, tmp in enumerate(tmp_paths, start=1):
        tmp.rename(directory / f"{idx}.jpeg")
