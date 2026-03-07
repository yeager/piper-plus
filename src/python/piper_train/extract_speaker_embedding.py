#!/usr/bin/env python3
"""Extract speaker embeddings using CAM++ ONNX model.

Usage:
    # Single WAV file
    python -m piper_train.extract_speaker_embedding \
        --encoder models/campplus.onnx --audio ref.wav --output speaker.npy

    # Directory of WAV files (average embedding)
    python -m piper_train.extract_speaker_embedding \
        --encoder models/campplus.onnx --audio-dir speaker_wavs/ --output speaker.npy

    # Dataset (all speakers at once)
    python -m piper_train.extract_speaker_embedding \
        --encoder models/campplus.onnx --dataset-dir dataset/ --output-dir embeddings/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torchaudio


if TYPE_CHECKING:
    import onnxruntime


_LOGGER = logging.getLogger("piper_train.extract_speaker_embedding")


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

    # リサンプリング
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    # 80-dim Fbank (kaldi互換)
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        num_mel_bins=80,
        frame_length=25.0,  # 25ms window
        frame_shift=10.0,  # 10ms hop
        sample_frequency=target_sr,
    )

    # CMVN正規化 (utterance-level)
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
    audio_tensor = torch.load(pt_path, weights_only=True)  # [1, samples] or [samples]
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)

    # リサンプリング (e.g. 22050 → 16000)
    if source_sr != target_sr:
        audio_tensor = torchaudio.functional.resample(
            audio_tensor, source_sr, target_sr
        )

    # Fbank
    fbank = torchaudio.compliance.kaldi.fbank(
        audio_tensor,
        num_mel_bins=80,
        frame_length=25.0,
        frame_shift=10.0,
        sample_frequency=target_sr,
    )
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
    """dataset.jsonl から話者ごとにembeddingを一括抽出する。

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
                audio_tensor = torch.load(pt_path, weights_only=True)
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
    if args.dataset_dir and not args.output_dir:
        parser.error("--output-dir is required with --dataset-dir")

    # ONNX session
    import onnxruntime  # noqa: PLC0415

    session = onnxruntime.InferenceSession(args.encoder)
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
        extract_from_dataset(
            session,
            dataset_dir=Path(args.dataset_dir),
            output_dir=Path(args.output_dir),
            max_utterances=args.max_utterances,
            min_duration=args.min_duration,
            source_sr=args.source_sample_rate,
        )


if __name__ == "__main__":
    main()
