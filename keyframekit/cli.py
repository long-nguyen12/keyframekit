"""Command-line interface for keyframekit."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

from .clustering import CLUSTERING_METHODS
from .vision import DEFAULT_MODEL_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("keyframekit.cli")

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}


def _collect_videos(paths: list[str]) -> list[Path]:
    """Expand file/directory paths into a flat list of video files."""
    videos: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            for ext in VIDEO_EXTENSIONS:
                videos.extend(path.glob(f"*{ext}"))
        elif path.is_file():
            videos.append(path)
        else:
            logger.warning("Path not found - skipping: %s", path)
    return sorted(set(videos))


def _get_video_length(video_path: str | Path) -> float:
    """Return video duration in seconds, or 0 on failure."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    length = 0.0
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            length = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        cap.release()
    return length


def run_embedding(args: argparse.Namespace) -> None:
    import torch

    from .extractor import extract_keyframes, load_vision_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model or DEFAULT_MODEL_NAMES.get(args.vision_backend, args.vision_backend)
    logger.info("Loading vision model '%s' on %s", model_name, device)
    processor, model = load_vision_model(
        model_name=model_name,
        device=device,
        model_backend=args.vision_backend,
    )

    videos = _collect_videos(args.input)
    if not videos:
        logger.error("No video files found.")
        sys.exit(1)

    rows = []
    for video in videos:
        logger.info("Processing: %s", video.name)
        try:
            t0 = time.time()
            out = extract_keyframes(
                video,
                model,
                processor,
                chunk_count=args.chunks,
                samples_per_chunk=args.samples,
                n_clusters=args.clusters,
                cluster_method=args.cluster_method,
                output_dir=args.output,
                device=device,
            )
            elapsed = time.time() - t0
            rows.append((video.stem, elapsed, _get_video_length(video)))
            logger.info("Done -> %s (%.2f s)", out, elapsed)
        except Exception:
            logger.exception("Failed to process %s", video.name)

    if args.csv and rows:
        with open(args.csv, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["video_id", "time_s", "video_length_s"])
            writer.writerows(rows)
        logger.info("Timing data written to %s", args.csv)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="keyframekit",
        description="Extract representative keyframes from video files.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    embed_p = sub.add_parser(
        "embed",
        aliases=["clip"],
        help="Embedding-model backend with selectable clustering",
    )
    embed_p.add_argument("input", nargs="+", help="Video file(s) or directory/ies")
    embed_p.add_argument(
        "--vision-backend",
        default="clip",
        choices=sorted(DEFAULT_MODEL_NAMES),
        help="Built-in vision backend used when --model is omitted",
    )
    embed_p.add_argument(
        "--model",
        default=None,
        help="Explicit HuggingFace vision model name",
    )
    embed_p.add_argument(
        "--cluster-method",
        default="spectral",
        choices=sorted(CLUSTERING_METHODS),
        help="Embedding clustering technique",
    )
    embed_p.add_argument(
        "--chunks", type=int, default=10, help="Number of temporal chunks / output frames"
    )
    embed_p.add_argument(
        "--samples", type=int, default=8, help="Candidate frames sampled per chunk"
    )
    embed_p.add_argument("--clusters", type=int, default=2, help="Clusters per chunk")
    embed_p.add_argument("--output", default=None, help="Output directory")
    embed_p.add_argument("--csv", default=None, help="Write timing CSV to this path")
    embed_p.set_defaults(func=run_embedding)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
