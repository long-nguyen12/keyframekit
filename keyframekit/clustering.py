"""Clustering and representative-frame selection."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

CLUSTERING_METHODS = {"spectral", "kmeans", "agglomerative"}


def select_representative_frame(
    frames: list,
    embeddings: torch.Tensor,
    n_clusters: int = 2,
    *,
    cluster_method: str = "spectral",
    plot_clusters: bool = False,
    plot_path: str | None = None,
    plot_title: str | None = None,
):
    """Pick the medoid of the largest embedding cluster."""
    import torch

    if not frames:
        return None
    if embeddings.shape[0] == 1:
        return frames[0]

    labels = cluster_embeddings(embeddings, n_clusters, method=cluster_method)
    largest_label = max(set(labels), key=lambda label: (labels == label).sum())
    idxs = [idx for idx, label in enumerate(labels) if label == largest_label]
    cluster_emb = embeddings[idxs]
    centroid = cluster_emb.mean(dim=0, keepdim=True)
    distances = torch.cdist(cluster_emb, centroid).squeeze(1)
    best_local = int(torch.argmin(distances).item())
    best_idx = idxs[best_local]

    if plot_clusters:
        _save_cluster_plot(
            embeddings=embeddings,
            labels=labels,
            highlight_idx=best_idx,
            plot_path=plot_path,
            plot_title=plot_title,
        )

    return frames[best_idx]


def cluster_embeddings(
    embeddings,
    n_clusters: int = 2,
    *,
    method: str = "spectral",
):
    """Cluster embedding rows with one of the built-in sklearn methods."""
    if method not in CLUSTERING_METHODS:
        raise ValueError(
            f"Unknown cluster method '{method}'. "
            f"Expected one of: {', '.join(sorted(CLUSTERING_METHODS))}"
        )

    n_samples = int(embeddings.shape[0])
    n_clusters = max(2, min(int(n_clusters), n_samples))
    data = embeddings.cpu().numpy()

    if method == "spectral":
        from sklearn.cluster import SpectralClustering

        n_neighbors = max(1, min(10, n_samples - 1))
        return SpectralClustering(
            n_clusters=n_clusters,
            affinity="nearest_neighbors",
            n_neighbors=n_neighbors,
            assign_labels="kmeans",
            random_state=0,
        ).fit_predict(data)

    if method == "kmeans":
        from sklearn.cluster import KMeans

        return KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit_predict(data)

    from sklearn.cluster import AgglomerativeClustering

    return AgglomerativeClustering(n_clusters=n_clusters).fit_predict(data)


def _save_cluster_plot(embeddings, labels, highlight_idx, plot_path, plot_title):
    """Render and save a 2-D PCA scatter of cluster assignments."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    try:
        reduced = PCA(n_components=2).fit_transform(embeddings.cpu().numpy())
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(
            reduced[:, 0],
            reduced[:, 1],
            c=labels,
            cmap="tab10",
            alpha=0.7,
            s=40,
            edgecolor="k",
            linewidth=0.2,
        )
        medoid_coords = reduced[highlight_idx]
        ax.scatter(
            medoid_coords[0],
            medoid_coords[1],
            c="red",
            s=140,
            marker="*",
            edgecolor="k",
            linewidth=0.8,
            label="selected medoid",
        )
        ax.legend(fontsize=10)
        ax.set_title(plot_title or "Cluster distribution", fontsize=12)
        ax.set_xlabel("PC 1", fontsize=10)
        ax.set_ylabel("PC 2", fontsize=10)
        ax.grid(True, alpha=0.3)

        if plot_path is None:
            plot_path = f"cluster_distribution_{time.time():.0f}.png"
        Path(os.path.dirname(plot_path) or ".").mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.debug("Cluster plot saved to %s", plot_path)
    except Exception:
        logger.warning("Failed to save cluster plot", exc_info=True)
