#!/usr/bin/env python3
"""
GPU バッチ STFT でスペクトログラムを高速計算するツール。

CPU並列版(~40 file/s)に対して、GPUバッチ処理で10倍以上の高速化を実現。

Usage:
    uv run python -m piper_train.tools.batch_spectrograms \
        --cache-dir /data/piper/dataset-zero-shot-20speakers/cache/22050 \
        --batch-size 128 \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------- constants ----------
N_FFT = 1024
HOP_SIZE = 256
WIN_SIZE = 1024
PAD_SIZE = (N_FFT - HOP_SIZE) // 2  # 384


# ---------- I/O helpers ----------
def _load_pt(path_str: str) -> tuple[str, torch.Tensor | None]:
    """Load a .pt file, return (path, tensor) or (path, None) on failure."""
    try:
        t = torch.load(path_str, weights_only=True, map_location="cpu")
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return (path_str, t.squeeze(0))  # return as 1D (samples,)
    except Exception as exc:
        logger.warning("Skipping corrupt file %s: %s", path_str, exc)
        return (path_str, None)


def _save_spec(args: tuple[str, torch.Tensor]) -> None:
    """Atomically save a spectrogram tensor as .spec.pt in FP16."""
    pt_path_str, spec = args
    spec_path = Path(pt_path_str).with_suffix("").with_suffix(".spec.pt")
    tmp_fd, tmp_path = tempfile.mkstemp(dir=spec_path.parent, suffix=".tmp")
    try:
        os.close(tmp_fd)
        torch.save(spec.half(), tmp_path)
        os.replace(tmp_path, spec_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------- scanning ----------
def _scan_pending(cache_dir: Path) -> list[str]:
    """Return .pt files that lack a corresponding .spec.pt."""
    pending: list[str] = []
    for name in sorted(os.listdir(cache_dir)):
        if not name.endswith(".pt"):
            continue
        if name.endswith(".spec.pt") or name.endswith(".tmp"):
            continue
        spec_name = name.replace(".pt", ".spec.pt")
        if not (cache_dir / spec_name).exists():
            pending.append(str(cache_dir / name))
    return pending


# ---------- GPU batch STFT ----------
def _batch_spectrogram_gpu(
    audios: list[torch.Tensor],
    device: torch.device,
    hann_window: torch.Tensor,
) -> list[torch.Tensor]:
    """Compute spectrograms for a batch of variable-length audios on GPU.

    1. Apply reflect-padding individually
    2. Zero-pad to max length, stack into batch
    3. Run torch.stft() in batch on GPU
    4. Compute magnitude
    5. Trim each result to correct frame count
    """
    # Apply reflect-padding to each audio
    padded = []
    lengths = []
    for audio in audios:
        p = torch.nn.functional.pad(
            audio.unsqueeze(0).unsqueeze(0),  # (1, 1, samples)
            (PAD_SIZE, PAD_SIZE),
            mode="reflect",
        ).squeeze(0).squeeze(0)  # (padded_samples,)
        padded.append(p)
        lengths.append(p.shape[0])

    # Zero-pad all to max length and stack
    max_len = max(lengths)
    batch = torch.zeros(len(padded), max_len)
    for i, p in enumerate(padded):
        batch[i, : lengths[i]] = p

    # GPU STFT
    batch_gpu = batch.to(device, non_blocking=True)
    spec_complex = torch.stft(
        batch_gpu,
        N_FFT,
        hop_length=HOP_SIZE,
        win_length=WIN_SIZE,
        window=hann_window,
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    # Magnitude: (batch, freq_bins, time_frames)
    spec_mag = torch.sqrt(
        torch.view_as_real(spec_complex).pow(2).sum(-1) + 1e-6
    ).cpu()

    # Trim each result to correct frame count
    results = []
    for i in range(len(audios)):
        n_frames = (lengths[i] - N_FFT) // HOP_SIZE + 1
        results.append(spec_mag[i, :, :n_frames])

    return results


# ---------- core logic ----------
def run(
    cache_dir: Path,
    batch_size: int = 128,
    device: str = "cuda:0",
    io_workers: int = 8,
) -> None:
    pending = _scan_pending(cache_dir)
    if not pending:
        logger.info("No pending files found -- nothing to do.")
        return

    logger.info("Found %d files to process (batch=%d, device=%s).", len(pending), batch_size, device)

    dev = torch.device(device)
    hann_window = torch.hann_window(WIN_SIZE, device=dev)

    processed = 0
    skipped = 0
    pbar = tqdm(total=len(pending), desc="Spectrogram", unit="file")

    # Process in batches
    for batch_start in range(0, len(pending), batch_size):
        batch_paths = pending[batch_start : batch_start + batch_size]

        # Parallel I/O: load .pt files
        with ThreadPoolExecutor(max_workers=io_workers) as io_pool:
            loaded = list(io_pool.map(_load_pt, batch_paths))

        # Filter out failures
        valid = [(path, audio) for path, audio in loaded if audio is not None]
        failed_count = len(loaded) - len(valid)
        skipped += failed_count

        if not valid:
            pbar.update(len(batch_paths))
            continue

        paths, audios = zip(*valid)

        # GPU batch STFT
        try:
            specs = _batch_spectrogram_gpu(list(audios), dev, hann_window)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.warning("OOM on batch starting at %s, skipping %d files.", batch_paths[0], len(valid))
            skipped += len(valid)
            pbar.update(len(batch_paths))
            continue

        # Parallel I/O: save .spec.pt files
        save_args = list(zip(paths, specs))
        with ThreadPoolExecutor(max_workers=io_workers) as io_pool:
            list(io_pool.map(_save_spec, save_args))

        processed += len(valid)
        pbar.update(len(batch_paths))

    pbar.close()
    logger.info("Done. processed=%d  skipped=%d", processed, skipped)


# ---------- CLI ----------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU batch spectrogram computation for cached .pt audio files."
    )
    parser.add_argument(
        "--cache-dir", type=Path, required=True,
        help="Directory containing normalised .pt files.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="Files per GPU batch (default: 128).",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Torch device (default: cuda:0).",
    )
    parser.add_argument(
        "--io-workers", type=int, default=8,
        help="I/O thread pool size (default: 8).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    if not args.cache_dir.is_dir():
        logger.error("Cache directory does not exist: %s", args.cache_dir)
        sys.exit(1)

    run(
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        device=args.device,
        io_workers=args.io_workers,
    )


if __name__ == "__main__":
    main()
