"""Unit tests for keyframekit."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# _reorder_and_rename_images
# ---------------------------------------------------------------------------

def test_reorder_and_rename_images(tmp_path):
    from keyframekit.extractor import _reorder_and_rename_images

    # Create dummy JPEG files out-of-order.
    for name in ("3.jpeg", "1.jpeg", "2.jpeg"):
        (tmp_path / name).write_bytes(b"\xff\xd8\xff")  # minimal JPEG header

    _reorder_and_rename_images(tmp_path)

    names = sorted(p.name for p in tmp_path.glob("*.jpeg"))
    assert names == ["1.jpeg", "2.jpeg", "3.jpeg"]


def test_reorder_empty_directory(tmp_path):
    from keyframekit.extractor import _reorder_and_rename_images

    # Should not raise even when no files present.
    _reorder_and_rename_images(tmp_path)


# ---------------------------------------------------------------------------
# select_representative_frame
# ---------------------------------------------------------------------------

def _make_frames(n: int):
    return [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(n)]


def test_select_representative_frame_single():
    from keyframekit.extractor import select_representative_frame

    frames = _make_frames(1)
    emb = torch.randn(1, 512)
    result = select_representative_frame(frames, emb, n_clusters=2)
    assert result is not None
    assert result.shape == (64, 64, 3)


def test_select_representative_frame_multiple():
    from keyframekit.extractor import select_representative_frame

    frames = _make_frames(10)
    emb = torch.randn(10, 512)
    result = select_representative_frame(frames, emb, n_clusters=3)
    assert result is not None


@pytest.mark.parametrize("cluster_method", ["spectral", "kmeans", "agglomerative"])
def test_select_representative_frame_cluster_methods(cluster_method):
    from keyframekit import select_representative_frame

    frames = _make_frames(10)
    emb = torch.randn(10, 512)
    result = select_representative_frame(
        frames,
        emb,
        n_clusters=2,
        cluster_method=cluster_method,
    )
    assert result is not None


def test_select_representative_frame_empty():
    from keyframekit.extractor import select_representative_frame

    result = select_representative_frame([], torch.randn(0, 512), n_clusters=2)
    assert result is None


# ---------------------------------------------------------------------------
# embed_frames (mock model + processor)
# ---------------------------------------------------------------------------

class _FakeProcessor:
    def __call__(self, images, return_tensors):
        n = len(images)
        return {"pixel_values": torch.zeros(n, 3, 224, 224)}


class _FakeOutput:
    def __init__(self, n, d=512):
        self.last_hidden_state = torch.randn(n, 197, d)


class _FakeModel:
    def __call__(self, **inputs):
        n = inputs["pixel_values"].shape[0]
        return _FakeOutput(n)


def test_embed_frames_returns_normalized_tensor():
    from keyframekit.extractor import _embed_frames

    frames = _make_frames(4)
    proc = _FakeProcessor()
    model = _FakeModel()
    device = torch.device("cpu")

    emb = _embed_frames(frames, proc, model, device)
    assert emb is not None
    assert emb.shape == (4, 512)
    # Check L2-normalised (unit norm per row).
    norms = emb.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)


def test_embed_frames_empty():
    from keyframekit.extractor import _embed_frames

    result = _embed_frames([], _FakeProcessor(), _FakeModel(), torch.device("cpu"))
    assert result is None


# ---------------------------------------------------------------------------
# extract_keyframes – error handling
# ---------------------------------------------------------------------------

def test_extract_keyframes_missing_file(tmp_path):
    from keyframekit.extractor import extract_keyframes

    with pytest.raises(FileNotFoundError):
        extract_keyframes(
            tmp_path / "nonexistent.mp4",
            _FakeModel(),
            _FakeProcessor(),
        )


def test_public_model_and_clustering_registries():
    from keyframekit import CLUSTERING_METHODS, DEFAULT_MODEL_NAMES

    assert {"clip", "siglip", "dinov2"}.issubset(DEFAULT_MODEL_NAMES)
    assert {"spectral", "kmeans", "agglomerative"}.issubset(CLUSTERING_METHODS)
