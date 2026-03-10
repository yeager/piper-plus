#!/usr/bin/env python3
"""Extract speaker embeddings using CAM++ ONNX model.

Usage:
    # Single WAV file
    uv run python -m piper_train.extract_speaker_embedding \
        --encoder models/campplus.onnx --audio ref.wav --output speaker.npy

    # Directory of WAV files (average embedding)
    uv run python -m piper_train.extract_speaker_embedding \
        --encoder models/campplus.onnx --audio-dir speaker_wavs/ --output speaker.npy

    # Dataset (all speakers at once)
    uv run python -m piper_train.extract_speaker_embedding \
        --encoder models/campplus.onnx --dataset-dir dataset/ --output-dir embeddings/

    # Per-utterance (optimized with DataLoader + batch inference)
    uv run python -m piper_train.extract_speaker_embedding \
        --encoder models/campplus.onnx --dataset-dir dataset/ --per-utterance \
        --batch-size 64 --num-workers 12
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torchaudio


if TYPE_CHECKING:
    import onnxruntime


_LOGGER = logging.getLogger("piper_train.extract_speaker_embedding")

# Resamplerキャッシュ: (source_sr, target_sr) → Resample transform
_RESAMPLER_CACHE: dict[tuple[int, int], torchaudio.transforms.Resample] = {}


def _get_resampler(source_sr: int, target_sr: int) -> torchaudio.transforms.Resample:
    """キャッシュ済みResamplerを取得する。毎回フィルタ再計算を避ける。"""
    key = (source_sr, target_sr)
    if key not in _RESAMPLER_CACHE:
        _RESAMPLER_CACHE[key] = torchaudio.transforms.Resample(source_sr, target_sr)
    return _RESAMPLER_CACHE[key]


def preprocess_audio(wav_path: str | Path, target_sr: int = 16000) -> np.ndarray:
    """WAVファイルを読み込み、Fbank特徴量に変換する。

    Args:
        wav_path: Path to WAV file.
        target_sr: Target sample rate.

    Returns:
        fbank: np.ndarray, shape [T, 80], float32
    """
    waveform, sr = torchaudio.load(str(wav_path))

    # ステレオ → モノラル
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # リサンプリング (キャッシュ済みResampler使用)
    if sr != target_sr:
        waveform = _get_resampler(sr, target_sr)(waveform)

    # 80-dim Fbank (kaldi互換)
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        num_mel_bins=80,
        frame_length=25.0,  # 25ms window
        frame_shift=10.0,  # 10ms hop
        sample_frequency=target_sr,
    )

    # CMVN正規化 (mean-only, matching CAM++ / 3D-Speaker convention)
    fbank = fbank - fbank.mean(dim=0, keepdim=True)

    return fbank.numpy()  # [T, 80]


def _load_audio_from_pt(
    pt_path: Path, source_sr: int = 22050, target_sr: int = 16000
) -> np.ndarray:
    """PTファイルから音声を読み込み、Fbank特徴量に変換する。

    Args:
        pt_path: Path to .pt audio tensor file.
        source_sr: Sample rate of the stored audio tensor.
        target_sr: Target sample rate for Fbank extraction.

    Returns:
        fbank: np.ndarray, shape [T, 80], float32
    """
    audio_tensor = torch.load(pt_path, weights_only=True, map_location="cpu", mmap=True)
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)

    # リサンプリング (キャッシュ済みResampler使用)
    if source_sr != target_sr:
        audio_tensor = _get_resampler(source_sr, target_sr)(audio_tensor)

    # Fbank
    fbank = torchaudio.compliance.kaldi.fbank(
        audio_tensor,
        num_mel_bins=80,
        frame_length=25.0,
        frame_shift=10.0,
        sample_frequency=target_sr,
    )
    # CMVN正規化 (mean-only, matching CAM++ / 3D-Speaker convention)
    fbank = fbank - fbank.mean(dim=0, keepdim=True)

    return fbank.numpy()


def extract_embedding(
    session: onnxruntime.InferenceSession, fbank: np.ndarray
) -> np.ndarray:
    """Fbank特徴量からspeaker embeddingを抽出する。

    Args:
        session: ONNX Runtime session.
        fbank: shape [T, 80], float32.

    Returns:
        embedding: shape [192], float32, L2正規化済み
    """
    # バッチ次元追加: [1, T, 80]
    fbank_input = np.expand_dims(fbank, axis=0).astype(np.float32)

    # ONNX推論
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: fbank_input})
    embedding = outputs[0].squeeze()  # [192]

    # L2正規化
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


def extract_from_files(
    session: onnxruntime.InferenceSession,
    wav_paths: list[Path],
) -> np.ndarray:
    """複数WAVファイルからembeddingを抽出し、平均化する。

    Args:
        session: ONNX Runtime session.
        wav_paths: List of WAV file paths.

    Returns:
        embedding: shape [192], float32, L2正規化済み
    """
    embeddings = []
    for wav_path in wav_paths:
        fbank = preprocess_audio(wav_path)
        emb = extract_embedding(session, fbank)
        embeddings.append(emb)

    # 平均化 + 再正規化
    avg_embedding = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(avg_embedding)
    if norm > 0:
        avg_embedding = avg_embedding / norm

    return avg_embedding


def extract_from_dataset(
    session: onnxruntime.InferenceSession,
    dataset_dir: Path,
    output_dir: Path,
    max_utterances: int = 10,
    min_duration: float = 3.0,
    source_sr: int = 22050,
    workers: int = 4,
) -> None:
    """dataset.jsonl から話者ごとにembeddingを一括抽出する（平均化モード）。

    Args:
        session: ONNX Runtime session.
        dataset_dir: Dataset directory containing dataset.jsonl.
        output_dir: Output directory for speaker embedding .npy files.
        max_utterances: Max utterances per speaker.
        min_duration: Minimum audio duration in seconds (shorter files are skipped).
        source_sr: Sample rate of .pt audio files in the dataset.
        workers: Number of parallel workers (reserved for future use).
    """
    jsonl_path = dataset_dir / "dataset.jsonl"
    if not jsonl_path.exists():
        msg = f"dataset.jsonl not found in {dataset_dir}"
        raise FileNotFoundError(msg)

    # 話者ごとにグループ化
    speaker_utterances: dict[int, list[Path]] = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt = json.loads(line)
            speaker_id = utt.get("speaker_id", 0)
            audio_path = dataset_dir / utt["audio_norm_path"]

            if speaker_id not in speaker_utterances:
                speaker_utterances[speaker_id] = []
            speaker_utterances[speaker_id].append(audio_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    for speaker_id, audio_paths in sorted(speaker_utterances.items()):
        # PTファイルから時間長を推定してフィルタリング
        valid_paths = []
        for pt_path in audio_paths:
            if not pt_path.exists():
                _LOGGER.warning("File not found, skipping: %s", pt_path)
                continue
            try:
                audio_tensor = torch.load(pt_path, weights_only=True, map_location="cpu")
                num_samples = audio_tensor.shape[-1]
                duration = num_samples / source_sr
                if duration >= min_duration:
                    valid_paths.append(pt_path)
            except Exception:
                _LOGGER.warning("Failed to load, skipping: %s", pt_path)
                continue

        # 最大 max_utterances 件を選択
        selected = valid_paths[:max_utterances]

        if not selected:
            _LOGGER.warning(
                "Speaker %d: no valid utterances (>= %.1fs), skipping",
                speaker_id,
                min_duration,
            )
            continue

        _LOGGER.info(
            "Speaker %d: %d utterances (selected %d of %d valid, %d total)",
            speaker_id,
            len(audio_paths),
            len(selected),
            len(valid_paths),
            len(audio_paths),
        )

        # PTファイルからembeddingを抽出
        embeddings = []
        for pt_path in selected:
            fbank = _load_audio_from_pt(pt_path, source_sr=source_sr)
            emb = extract_embedding(session, fbank)
            embeddings.append(emb)

        avg_embedding = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm

        output_path = output_dir / f"speaker_{speaker_id}.npy"
        np.save(str(output_path), avg_embedding)
        _LOGGER.info(
            "Saved: %s (norm=%.4f)", output_path, np.linalg.norm(avg_embedding)
        )


# ---------------------------------------------------------------------------
# Per-utterance extraction with DataLoader + batch ONNX inference
# ---------------------------------------------------------------------------

class _FbankDataset(torch.utils.data.Dataset):
    """DataLoader用Dataset: PTファイルからFbank特徴量を並列抽出する。

    各ワーカープロセスで独立にCPU前処理（torch.load → resample → fbank）を実行し、
    メインプロセスのGPU ONNX推論にバッチで渡す。
    """

    def __init__(
        self,
        items: list[tuple[int, Path, str]],
        source_sr: int,
        target_sr: int = 16000,
    ):
        self.items = items  # (entry_index, pt_path, stem)
        self.source_sr = source_sr
        self.target_sr = target_sr
        # Resamplerをキャッシュ（各ワーカーに1インスタンス、pickle経由でコピー）
        self.resampler = (
            torchaudio.transforms.Resample(source_sr, target_sr)
            if source_sr != target_sr
            else None
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[int, torch.Tensor, str, bool]:
        entry_idx, pt_path, stem = self.items[idx]
        try:
            audio_tensor = torch.load(pt_path, weights_only=True, map_location="cpu", mmap=True)
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            if self.resampler is not None:
                audio_tensor = self.resampler(audio_tensor)
            fbank = torchaudio.compliance.kaldi.fbank(
                audio_tensor,
                num_mel_bins=80,
                frame_length=25.0,
                frame_shift=10.0,
                sample_frequency=self.target_sr,
            )
            fbank = fbank - fbank.mean(dim=0, keepdim=True)
            return entry_idx, fbank, stem, True
        except Exception as e:
            _LOGGER.warning("Worker failed to load %s: %s", pt_path, e)
            return entry_idx, torch.zeros(1, 80), stem, False


def _collate_fbanks(
    batch: list[tuple[int, torch.Tensor, str, bool]],
) -> tuple[list[int], np.ndarray, list[str], list[bool]]:
    """可変長Fbankをゼロパディングしてバッチ化する。"""
    indices, fbanks, stems, valids = zip(*batch, strict=False)
    max_t = max(f.shape[0] for f in fbanks)
    padded = torch.zeros(len(fbanks), max_t, 80)
    for i, f in enumerate(fbanks):
        padded[i, : f.shape[0], :] = f
    return list(indices), padded.numpy().astype(np.float32), list(stems), list(valids)


def _write_updated_jsonl(dataset_dir: Path, entries: list[dict]) -> None:
    """dataset.jsonlをバックアップして更新する。"""
    output_jsonl = dataset_dir / "dataset.jsonl"
    backup_path = dataset_dir / "dataset.jsonl.bak"
    shutil.copy2(output_jsonl, backup_path)
    _LOGGER.info("Backed up original to: %s", backup_path)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for entry in entries:
            json.dump(entry, f, ensure_ascii=True)
            f.write("\n")
    _LOGGER.info(
        "Updated dataset.jsonl with speaker_embedding_path (%d entries)", len(entries)
    )


def extract_per_utterance(
    session: onnxruntime.InferenceSession,
    dataset_dir: Path,
    output_dir: Path,
    source_sr: int = 22050,
    batch_size: int = 64,
    num_workers: int = 12,
) -> None:
    """dataset.jsonl の各発話ごとにembeddingを抽出し、dataset.jsonlを更新する。

    最適化:
    1. DataLoader (num_workers) でCPU前処理を並列化 (GIL回避)
    2. バッチONNX推論でGPU効率を最大化
    3. 既存embedding事前キャッシュでファイルI/O削減
    4. Resamplerキャッシュでフィルタ再計算を回避

    Args:
        session: ONNX Runtime session.
        dataset_dir: Dataset directory containing dataset.jsonl.
        output_dir: Output directory for speaker embedding .npy files.
        source_sr: Sample rate of .pt audio files in the dataset.
        batch_size: Batch size for ONNX inference.
        num_workers: Number of DataLoader workers for CPU preprocessing.
    """
    jsonl_path = dataset_dir / "dataset.jsonl"
    if not jsonl_path.exists():
        msg = f"dataset.jsonl not found in {dataset_dir}"
        raise FileNotFoundError(msg)

    emb_dir = output_dir / "speaker_embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    # 全発話を読み込み
    entries: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    _LOGGER.info("Total utterances: %d", len(entries))

    # 最適化3: 既存.npyファイル名を事前キャッシュ (O(1)ルックアップ)
    existing_stems: set[str] = {p.stem for p in emb_dir.glob("*.npy")}
    _LOGGER.info("Already extracted: %d embeddings (pre-cached)", len(existing_stems))

    # 既存embeddingのパス設定 + 未抽出アイテム収集
    items_to_extract: list[tuple[int, Path, str]] = []
    skipped = 0
    fail = 0

    for i, utt in enumerate(entries):
        audio_norm_path = utt.get("audio_norm_path")
        if not audio_norm_path:
            fail += 1
            continue

        pt_path = Path(audio_norm_path)
        if not pt_path.is_absolute():
            pt_path = dataset_dir / pt_path

        stem = pt_path.stem
        npy_rel = f"speaker_embeddings/{stem}.npy"

        if stem in existing_stems:
            utt["speaker_embedding_path"] = npy_rel
            skipped += 1
            continue

        if not pt_path.exists():
            _LOGGER.warning("File not found, skipping: %s", pt_path)
            fail += 1
            continue

        items_to_extract.append((i, pt_path, stem))

    _LOGGER.info(
        "To extract: %d, skipped (existing): %d, failed: %d",
        len(items_to_extract),
        skipped,
        fail,
    )

    if not items_to_extract:
        _LOGGER.info("All embeddings already extracted")
        _write_updated_jsonl(dataset_dir, entries)
        return

    # 最適化1+4: DataLoader (並列CPU前処理 + Resamplerキャッシュ)
    dataset = _FbankDataset(items_to_extract, source_sr=source_sr)
    loader_kwargs: dict = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": _collate_fbanks,
        "pin_memory": True,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
        loader_kwargs["persistent_workers"] = True
    loader = torch.utils.data.DataLoader(dataset, **loader_kwargs)

    input_name = session.get_inputs()[0].name
    success = skipped
    total_batches = (len(items_to_extract) + batch_size - 1) // batch_size

    _LOGGER.info(
        "Starting batch extraction: %d batches (batch_size=%d, workers=%d)",
        total_batches,
        batch_size,
        num_workers,
    )

    # 最適化2: バッチONNX推論
    for batch_idx, (indices, fbanks_batch, stems, valids) in enumerate(loader):
        # バッチ推論: [B, T_max, 80] → [B, 192]
        embeddings_batch = session.run(None, {input_name: fbanks_batch})[0]

        # L2正規化 (バッチ全体を一括処理)
        norms = np.linalg.norm(embeddings_batch, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        embeddings_batch = embeddings_batch / norms

        for j, (entry_idx, stem, valid) in enumerate(zip(indices, stems, valids, strict=False)):
            if not valid:
                fail += 1
                continue

            npy_path = emb_dir / f"{stem}.npy"
            np.save(str(npy_path), embeddings_batch[j])

            entries[entry_idx]["speaker_embedding_path"] = f"speaker_embeddings/{stem}.npy"
            success += 1

        if (batch_idx + 1) % 50 == 0 or batch_idx + 1 == total_batches:
            _LOGGER.info(
                "Batch %d/%d (success=%d, fail=%d)",
                batch_idx + 1,
                total_batches,
                success,
                fail,
            )

    _LOGGER.info(
        "Extraction complete: %d success, %d failed out of %d total",
        success,
        fail,
        len(entries),
    )

    _write_updated_jsonl(dataset_dir, entries)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        prog="piper_train.extract_speaker_embedding",
        description="Extract speaker embeddings using CAM++ ONNX model",
    )
    parser.add_argument("--encoder", required=True, help="Path to CAM++ ONNX model")
    parser.add_argument("--audio", help="Single WAV file to process")
    parser.add_argument(
        "--audio-dir", help="Directory of WAV files (average embedding)"
    )
    parser.add_argument("--dataset-dir", help="Dataset directory with dataset.jsonl")
    parser.add_argument(
        "--output", help="Output .npy file path (for --audio / --audio-dir)"
    )
    parser.add_argument("--output-dir", help="Output directory (for --dataset-dir)")
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers"
    )
    parser.add_argument(
        "--max-utterances",
        type=int,
        default=10,
        help="Max utterances per speaker in dataset mode",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=3.0,
        help="Min duration in seconds for dataset mode",
    )
    parser.add_argument(
        "--source-sample-rate",
        type=int,
        default=22050,
        help="Source sample rate for .pt files in dataset mode",
    )
    parser.add_argument(
        "--per-utterance",
        action="store_true",
        help="Extract per-utterance embeddings (recommended for zero-shot TTS training). "
        "Updates dataset.jsonl in-place with speaker_embedding_path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for per-utterance ONNX inference (default: 64)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=12,
        help="Number of DataLoader workers for CPU preprocessing (default: 12)",
    )
    args = parser.parse_args()

    # 排他制御
    modes = sum(
        [
            args.audio is not None,
            args.audio_dir is not None,
            args.dataset_dir is not None,
        ]
    )
    if modes == 0:
        parser.error("One of --audio, --audio-dir, or --dataset-dir is required")
    if modes > 1:
        parser.error(
            "Only one of --audio, --audio-dir, or --dataset-dir can be specified"
        )

    if (args.audio or args.audio_dir) and not args.output:
        parser.error("--output is required with --audio or --audio-dir")
    if args.dataset_dir and not args.per_utterance and not args.output_dir:
        parser.error("--output-dir is required with --dataset-dir (unless --per-utterance)")

    # ONNX session (GPU優先、なければCPU)
    import onnxruntime  # noqa: PLC0415

    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_options.enable_mem_reuse = True
    sess_options.enable_mem_pattern = True

    cuda_provider_options = {
        "arena_extend_strategy": "kSameAsRequested",
        "do_copy_in_default_stream": False,
    }

    providers = onnxruntime.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        session = onnxruntime.InferenceSession(
            args.encoder,
            sess_options,
            providers=[
                ("CUDAExecutionProvider", cuda_provider_options),
                "CPUExecutionProvider",
            ],
        )
        _LOGGER.info("Using GPU (CUDAExecutionProvider)")
    else:
        session = onnxruntime.InferenceSession(args.encoder, sess_options)
        _LOGGER.info("Using CPU (CUDAExecutionProvider not available)")
    _LOGGER.info("Loaded speaker encoder: %s", args.encoder)

    if args.audio:
        fbank = preprocess_audio(args.audio)
        embedding = extract_embedding(session, fbank)
        np.save(args.output, embedding)
        _LOGGER.info(
            "Saved: %s (shape=%s, norm=%.4f)",
            args.output,
            embedding.shape,
            np.linalg.norm(embedding),
        )

    elif args.audio_dir:
        audio_dir = Path(args.audio_dir)
        wav_files = sorted(audio_dir.glob("*.wav")) + sorted(audio_dir.glob("*.WAV"))
        if not wav_files:
            _LOGGER.error("No WAV files found in %s", audio_dir)
            return
        _LOGGER.info("Found %d WAV files in %s", len(wav_files), audio_dir)
        embedding = extract_from_files(session, wav_files)
        np.save(args.output, embedding)
        _LOGGER.info(
            "Saved: %s (shape=%s, norm=%.4f)",
            args.output,
            embedding.shape,
            np.linalg.norm(embedding),
        )

    elif args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
        if args.per_utterance:
            extract_per_utterance(
                session,
                dataset_dir=dataset_dir,
                output_dir=dataset_dir,
                source_sr=args.source_sample_rate,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
        else:
            extract_from_dataset(
                session,
                dataset_dir=dataset_dir,
                output_dir=Path(args.output_dir),
                max_utterances=args.max_utterances,
                min_duration=args.min_duration,
                source_sr=args.source_sample_rate,
            )


if __name__ == "__main__":
    main()
