#!/usr/bin/env python3
"""Normalise audio files from dataset.jsonl in parallel (.pt only, no spectrogram).

Spectrograms are computed later on GPU in batched mode.

Usage
-----
uv run python -m piper_train.tools.cache_audio \
  --dataset /data/piper/dataset-zero-shot-20speakers/dataset.jsonl \
  --cache-dir /data/piper/dataset-zero-shot-20speakers/cache/22050 \
  --sample-rate 22050 \
  --workers 30
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
from hashlib import sha256
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from tqdm import tqdm

from piper_train.norm_audio import _atomic_torch_save, energy_vad_numpy


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Worker helpers (top-level for pickle compatibility)
# ---------------------------------------------------------------------------

# Populated via initializer so each worker has its own copy.
_worker_cache_dir: Path
_worker_sample_rate: int
_worker_vad_threshold: float


def _worker_init(cache_dir: str, sample_rate: int, vad_threshold: float) -> None:
    """Called once per worker process to set module-level state."""
    global _worker_cache_dir, _worker_sample_rate, _worker_vad_threshold  # noqa: PLW0603
    _worker_cache_dir = Path(cache_dir)
    _worker_sample_rate = sample_rate
    _worker_vad_threshold = vad_threshold


def _process_one(audio_path: str) -> tuple[str, str | None, str | None]:
    """Process a single audio file and return (audio_path, norm_path, error).

    Returns
    -------
    (audio_path, norm_path_str, None) on success
    (audio_path, None, error_message) on failure
    """
    import soxr  # noqa: PLC0415  — heavy import, keep out of module top-level

    try:
        abs_path = str(Path(audio_path).absolute())
        cache_id = sha256(abs_path.encode()).hexdigest()
        norm_path = _worker_cache_dir / f"{cache_id}.pt"

        if norm_path.exists():
            return (audio_path, str(norm_path), None)

        # 1. Load audio
        audio_data, src_sr = sf.read(audio_path, dtype="float32", always_2d=False)
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        # 2. Resample to 16 kHz (MQ) for VAD
        if src_sr != 16000:
            audio_16k = soxr.resample(audio_data, src_sr, 16000, quality="MQ")
        else:
            audio_16k = audio_data

        # 3. Energy VAD — get voiced region
        offset_sec, duration_sec = energy_vad_numpy(
            audio_16k, threshold=_worker_vad_threshold
        )

        # 4. Trim at source sample rate
        offset_samples = int(offset_sec * src_sr)
        if duration_sec is not None:
            end_samples = min(
                offset_samples + int(duration_sec * src_sr), len(audio_data)
            )
        else:
            end_samples = len(audio_data)
        trimmed = audio_data[offset_samples:end_samples]

        # 5. Resample to target rate (HQ)
        if src_sr != _worker_sample_rate:
            audio_rs = soxr.resample(trimmed, src_sr, _worker_sample_rate, quality="HQ")
        else:
            audio_rs = trimmed

        # 6. Save as .pt (no spectrogram)
        audio_tensor = torch.from_numpy(audio_rs).unsqueeze(0)
        _atomic_torch_save(audio_tensor, norm_path)

        return (audio_path, str(norm_path), None)

    except Exception as exc:  # noqa: BLE001
        return (audio_path, None, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache normalised audio (.pt) in parallel. "
        "Spectrograms are NOT computed (left for GPU batch)."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to dataset.jsonl",
    )
    parser.add_argument(
        "--cache-dir",
        required=True,
        help="Directory to store .pt files",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Target sample rate (default: 22050)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=mp.cpu_count(),
        help="Number of parallel workers (default: cpu_count)",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.02,
        help="Energy VAD threshold (default: 0.02)",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Read dataset.jsonl and collect audio_paths ----------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger.info("Loading dataset from %s", dataset_path)

    with open(dataset_path, encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]

    items: list[dict[str, Any]] = [json.loads(line) for line in lines]
    audio_paths: list[str] = [item["audio_path"] for item in items]
    logger.info("Found %d utterances", len(audio_paths))

    # --- 2. Parallel normalisation ------------------------------------------
    logger.info(
        "Caching normalised audio with %d workers (sample_rate=%d) ...",
        args.workers,
        args.sample_rate,
    )

    # Build a lookup: audio_path -> norm_path (filled during processing)
    norm_map: dict[str, str] = {}
    fail_count = 0

    with mp.Pool(
        processes=args.workers,
        initializer=_worker_init,
        initargs=(str(cache_dir), args.sample_rate, args.vad_threshold),
    ) as pool:
        for audio_path, norm_path, err in tqdm(
            pool.imap_unordered(_process_one, audio_paths, chunksize=64),
            total=len(audio_paths),
            desc="Caching audio",
        ):
            if err is not None:
                logger.warning("SKIP %s: %s", audio_path, err)
                fail_count += 1
            else:
                assert norm_path is not None
                norm_map[audio_path] = norm_path

    logger.info("Done: %d succeeded, %d failed", len(norm_map), fail_count)

    # --- 3. Update dataset.jsonl with paths ---------------------------------
    logger.info("Updating %s ...", dataset_path)
    updated_count = 0
    with open(dataset_path, "w", encoding="utf-8") as f:
        for item in items:
            ap = item["audio_path"]
            if ap in norm_map:
                item["audio_norm_path"] = norm_map[ap]
                # Spec path: will be populated later by GPU batch processing
                abs_ap = str(Path(ap).absolute())
                cache_id = sha256(abs_ap.encode()).hexdigest()
                item["audio_spec_path"] = str(cache_dir / f"{cache_id}.spec.pt")
                updated_count += 1
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(
        "Updated %d / %d entries in %s", updated_count, len(items), dataset_path
    )


if __name__ == "__main__":
    main()
