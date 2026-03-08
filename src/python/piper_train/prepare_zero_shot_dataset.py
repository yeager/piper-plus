"""Zero-shot TTS学習用データ準備スクリプト.

LibriTTS-R, JVS, moe-speech の3コーパスを統合し、
統一フォーマットの dataset.jsonl + config.json を生成する。

Usage:
    uv run python -m piper_train.prepare_zero_shot_dataset \
        --libritts-dir /data/libritts_r \
        --jvs-dir /data/jvs_ver1 \
        --moe-speech-dir /data/piper/dataset-moe-speech-20speakers-v2 \
        --output-dir /data/piper/dataset-zero-shot \
        --encoder models/campplus.onnx
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import onnxruntime

_LOGGER = logging.getLogger("piper_train.prepare_zero_shot_dataset")

# ------------------------------------------------------------------ #
# Speaker ID 割り当て範囲
# ------------------------------------------------------------------ #
SPEAKER_ID_RANGES: dict[str, tuple[int, int]] = {
    "moe-speech": (0, 19),
    "jvs": (20, 119),
    "libritts": (120, 2575),
}


# ------------------------------------------------------------------ #
# コーパスパーサー
# ------------------------------------------------------------------ #
def _parse_libritts(
    libritts_dir: Path,
) -> list[dict]:
    """LibriTTS-R ディレクトリを走査し、発話リストを返す.

    Returns:
        各要素は {"wav_path", "text", "corpus_speaker_id", "language"}
    """
    utterances: list[dict] = []
    # train-clean-100 などのサブセットを走査
    for subset_dir in sorted(libritts_dir.iterdir()):
        if not subset_dir.is_dir():
            continue
        for speaker_dir in sorted(subset_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            corpus_speaker_id = speaker_dir.name
            for chapter_dir in sorted(speaker_dir.iterdir()):
                if not chapter_dir.is_dir():
                    continue
                for txt_path in sorted(chapter_dir.glob("*.normalized.txt")):
                    stem = txt_path.name.replace(".normalized.txt", "")
                    wav_path = chapter_dir / f"{stem}.wav"
                    if not wav_path.exists():
                        _LOGGER.warning("WAV not found for %s", txt_path)
                        continue
                    text = txt_path.read_text(encoding="utf-8").strip()
                    if not text:
                        continue
                    utterances.append(
                        {
                            "wav_path": wav_path,
                            "text": text,
                            "corpus_speaker_id": corpus_speaker_id,
                            "language": "en",
                        }
                    )
    _LOGGER.info(
        "LibriTTS-R: %d utterances, %d speakers",
        len(utterances),
        len({u["corpus_speaker_id"] for u in utterances}),
    )
    return utterances


def _parse_jvs(jvs_dir: Path) -> list[dict]:
    """JVS ディレクトリを走査し、発話リストを返す.

    parallel100 サブセットのみ使用。
    """
    utterances: list[dict] = []
    for speaker_dir in sorted(jvs_dir.iterdir()):
        if not speaker_dir.is_dir():
            continue
        if not speaker_dir.name.startswith("jvs"):
            continue
        # 話者ID抽出 (jvs001 → 1)
        try:
            corpus_speaker_id = str(int(speaker_dir.name.replace("jvs", "")))
        except ValueError:
            _LOGGER.warning("Skipping non-JVS directory: %s", speaker_dir)
            continue

        # transcripts_utf8.txt を読み込み
        transcript_path = speaker_dir / "parallel100" / "transcripts_utf8.txt"
        if not transcript_path.exists():
            _LOGGER.warning("Transcript not found: %s", transcript_path)
            continue

        # テキストマップ: ファイル名 → テキスト
        text_map: dict[str, str] = {}
        with open(transcript_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "\t" in line:
                    fname, text = line.split("\t", 1)
                elif ":" in line:
                    fname, text = line.split(":", 1)
                else:
                    continue
                text_map[fname.strip()] = text.strip()

        wav_dir = speaker_dir / "parallel100" / "wav24kHz16bit"
        if not wav_dir.exists():
            _LOGGER.warning("WAV dir not found: %s", wav_dir)
            continue

        for wav_path in sorted(wav_dir.glob("*.wav")):
            fname_key = wav_path.stem
            text = text_map.get(fname_key)
            if text is None:
                _LOGGER.debug("No transcript for %s", wav_path.name)
                continue
            utterances.append(
                {
                    "wav_path": wav_path,
                    "text": text,
                    "corpus_speaker_id": corpus_speaker_id,
                    "language": "ja",
                }
            )

    _LOGGER.info(
        "JVS: %d utterances, %d speakers",
        len(utterances),
        len({u["corpus_speaker_id"] for u in utterances}),
    )
    return utterances


def _parse_moe_speech(
    moe_speech_dir: Path,
) -> list[dict]:
    """moe-speech の既存 dataset.jsonl を読み込む.

    パススルー: phoneme_ids, prosody_features, audio パスはそのまま維持。
    """
    jsonl_path = moe_speech_dir / "dataset.jsonl"
    if not jsonl_path.exists():
        msg = f"dataset.jsonl not found in {moe_speech_dir}"
        raise FileNotFoundError(msg)

    utterances: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                utt = json.loads(line)
            except json.JSONDecodeError:
                _LOGGER.warning(
                    "Skipping malformed JSON at line %d in %s",
                    line_num,
                    jsonl_path,
                )
                continue
            utterances.append(utt)

    speakers = {u.get("speaker_id", 0) for u in utterances}
    _LOGGER.info(
        "moe-speech: %d utterances, %d speakers",
        len(utterances),
        len(speakers),
    )
    return utterances


# ------------------------------------------------------------------ #
# Speaker ID 割り当て
# ------------------------------------------------------------------ #
def _assign_speaker_ids(
    corpus_name: str,
    corpus_speaker_ids: list[str],
) -> dict[str, int]:
    """コーパス内の話者をソートし、グローバルIDを割り当てる.

    Returns:
        {corpus_speaker_id: global_speaker_id}
    """
    start, end = SPEAKER_ID_RANGES[corpus_name]
    unique_sorted = sorted(set(corpus_speaker_ids))
    capacity = end - start + 1
    if len(unique_sorted) > capacity:
        _LOGGER.warning(
            "%s: %d speakers exceed range capacity %d, truncating to %d",
            corpus_name,
            len(unique_sorted),
            capacity,
            capacity,
        )
        unique_sorted = unique_sorted[:capacity]

    return {spk: start + idx for idx, spk in enumerate(unique_sorted)}


# ------------------------------------------------------------------ #
# 音素化 + ID変換
# ------------------------------------------------------------------ #
def _phonemize_utterance(
    text: str,
    language: str,
    phoneme_id_map: dict[str, list[int]],
) -> tuple[list[int], list[dict | None]]:
    """テキストを音素化し、phoneme_ids と prosody_features を返す."""
    from piper_train.phonemize.registry import (  # noqa: PLC0415
        get_phonemizer,
    )

    phonemizer = get_phonemizer(language)
    phonemes, prosody_info_list = phonemizer.phonemize_with_prosody(text)

    phoneme_ids: list[int] = []
    prosody_features: list[dict | None] = []
    unknown_count = 0
    unknown_examples: list[str] = []

    for phoneme, prosody_info in zip(phonemes, prosody_info_list, strict=True):
        if phoneme in phoneme_id_map:
            ids = phoneme_id_map[phoneme]
            phoneme_ids.extend(ids)
            for _ in ids:
                if prosody_info is not None:
                    prosody_features.append(
                        {
                            "a1": prosody_info.a1,
                            "a2": prosody_info.a2,
                            "a3": prosody_info.a3,
                        }
                    )
                else:
                    prosody_features.append(None)
        else:
            unknown_count += 1
            if len(unknown_examples) < 5:
                unknown_examples.append(phoneme)

    if unknown_count > 0:
        _LOGGER.warning(
            "Skipped %d unknown phoneme(s) for language '%s' (first examples: %s)",
            unknown_count,
            language,
            unknown_examples,
        )

    # 言語固有のポスト処理 (BOS/EOS/padding)
    phoneme_ids, prosody_features = phonemizer.post_process_ids(
        phoneme_ids, prosody_features, phoneme_id_map
    )

    # 長さ整合性チェック
    if len(phoneme_ids) != len(prosody_features):
        msg = (
            f"phoneme_ids({len(phoneme_ids)}) != "
            f"prosody_features({len(prosody_features)}) "
            f"after post_process_ids for text: {text!r}"
        )
        raise ValueError(msg)

    return phoneme_ids, prosody_features


# ------------------------------------------------------------------ #
# 音声前処理
# ------------------------------------------------------------------ #
def _process_audio(
    wav_path: Path,
    cache_dir: Path,
    sample_rate: int,
) -> tuple[Path, Path] | None:
    """WAVを正規化・スペクトログラム化し、キャッシュパスを返す.

    Returns:
        (norm_path, spec_path) on success, None on failure.
    """
    try:
        from piper_train.norm_audio import (  # noqa: PLC0415
            cache_norm_audio,
            make_silence_detector,
        )
    except Exception:
        _LOGGER.warning(
            "Failed to import norm_audio (librosa/numba may be unavailable), "
            "skipping audio processing for %s",
            wav_path,
            exc_info=True,
        )
        return None

    # SileroVADの検出器はスレッドセーフでないため毎回生成
    # (実際にはシングルプロセスなので問題ない)
    if not hasattr(_process_audio, "_detector"):
        _process_audio._detector = make_silence_detector()  # type: ignore[attr-defined]

    try:
        audio_norm_path, audio_spec_path = cache_norm_audio(
            audio_path=wav_path,
            cache_dir=cache_dir,
            detector=_process_audio._detector,  # type: ignore[attr-defined]
            sample_rate=sample_rate,
        )
    except Exception:
        _LOGGER.warning(
            "Failed to process audio %s, skipping",
            wav_path,
            exc_info=True,
        )
        return None
    return audio_norm_path, audio_spec_path


# ------------------------------------------------------------------ #
# Speaker Embedding 抽出
# ------------------------------------------------------------------ #
def _extract_embeddings_for_corpus(
    session: onnxruntime.InferenceSession,
    speaker_wav_map: dict[int, list[Path]],
    output_dir: Path,
    max_per_speaker: int = 10,
) -> None:
    """話者ごとにWAVからembeddingを抽出し、.npyに保存する."""
    from piper_train.extract_speaker_embedding import (  # noqa: PLC0415
        extract_from_files,
    )

    emb_dir = output_dir / "speaker_embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np  # noqa: PLC0415

    for speaker_id, wav_paths in sorted(speaker_wav_map.items()):
        selected = wav_paths[:max_per_speaker]
        if not selected:
            _LOGGER.warning(
                "Speaker %d: no WAV files, skipping embedding",
                speaker_id,
            )
            continue

        embedding = extract_from_files(session, selected)
        out_path = emb_dir / f"speaker_{speaker_id}.npy"
        np.save(str(out_path), embedding)
        _LOGGER.info(
            "Speaker %d: embedding saved (%d files) -> %s",
            speaker_id,
            len(selected),
            out_path,
        )


def _extract_embeddings_for_moe_speech(
    session: onnxruntime.InferenceSession,
    moe_speech_dir: Path,
    output_dir: Path,
    speaker_id_offset: int,
    max_utterances: int = 10,
    source_sr: int = 22050,
) -> None:
    """moe-speech用: PTファイルからembeddingを抽出する."""
    import shutil
    import tempfile

    from piper_train.extract_speaker_embedding import (
        extract_from_dataset,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        extract_from_dataset(
            session=session,
            dataset_dir=moe_speech_dir,
            output_dir=tmp_path,
            max_utterances=max_utterances,
            source_sr=source_sr,
        )

        emb_dir = output_dir / "speaker_embeddings"
        emb_dir.mkdir(parents=True, exist_ok=True)

        # リネーム: speaker_{original_id}.npy → speaker_{offset+original_id}.npy
        for npy_path in sorted(tmp_path.glob("speaker_*.npy")):
            original_id = int(npy_path.stem.replace("speaker_", ""))
            new_id = speaker_id_offset + original_id
            dest = emb_dir / f"speaker_{new_id}.npy"
            shutil.copy2(npy_path, dest)
            _LOGGER.info(
                "moe-speech speaker %d -> global %d: %s",
                original_id,
                new_id,
                dest,
            )


# ------------------------------------------------------------------ #
# phoneme_id_map 統合
# ------------------------------------------------------------------ #
def _build_merged_phoneme_id_map() -> dict[str, list[int]]:
    """日本語と英語の phoneme_id_map を統合する.

    日本語: JapanesePhonemizer が jp_id_map を通して提供
    英語: piper_phonemize の espeak map (get_phoneme_id_map() が None
          のため、piper_phonemize.get_espeak_map() を使用)

    両者のIDは独立管理されており衝突しないため、そのままマージする。
    """
    from piper_train.phonemize.jp_id_map import (  # noqa: PLC0415
        get_japanese_id_map,
    )

    merged: dict[str, list[int]] = {}

    # 日本語ID
    ja_map = get_japanese_id_map()
    merged.update(ja_map)

    # 英語ID: piper_phonemize の espeak map を試行
    try:
        from piper_phonemize import (  # noqa: PLC0415
            get_espeak_map,
        )

        en_map = get_espeak_map()
        merged.update(en_map)
    except ImportError:
        _LOGGER.warning(
            "piper_phonemize not available; "
            "English phoneme IDs not included in merged map. "
            "English utterances may lack proper ID mapping."
        )

    return merged


# ------------------------------------------------------------------ #
# Validate
# ------------------------------------------------------------------ #
def _validate_dataset(
    output_dir: Path,
    phoneme_id_map: dict[str, list[int]],
) -> int:
    """生成された dataset.jsonl を検証する.

    Returns:
        エラー数
    """
    jsonl_path = output_dir / "dataset.jsonl"
    if not jsonl_path.exists():
        _LOGGER.error("dataset.jsonl not found in %s", output_dir)
        return 1

    errors = 0
    total = 0

    with open(jsonl_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                utt = json.loads(line)
            except json.JSONDecodeError:
                _LOGGER.error("Line %d: malformed JSON, skipping", line_num)
                errors += 1
                continue

            # phoneme_ids が空でないこと
            phoneme_ids = utt.get("phoneme_ids", [])
            if not phoneme_ids:
                _LOGGER.error("Line %d: empty phoneme_ids", line_num)
                errors += 1

            # prosody_features の長さ一致
            prosody = utt.get("prosody_features")
            if prosody is not None:
                if len(phoneme_ids) != len(prosody):
                    _LOGGER.error(
                        "Line %d: phoneme_ids(%d) != prosody_features(%d)",
                        line_num,
                        len(phoneme_ids),
                        len(prosody),
                    )
                    errors += 1

            # audio_norm_path 存在チェック
            norm_path = utt.get("audio_norm_path")
            if norm_path:
                full_path = output_dir / norm_path
                if not full_path.exists():
                    _LOGGER.error(
                        "Line %d: audio_norm_path not found: %s",
                        line_num,
                        full_path,
                    )
                    errors += 1

            # audio_spec_path 存在チェック
            spec_path = utt.get("audio_spec_path")
            if spec_path:
                full_path = output_dir / spec_path
                if not full_path.exists():
                    _LOGGER.error(
                        "Line %d: audio_spec_path not found: %s",
                        line_num,
                        full_path,
                    )
                    errors += 1

            # speaker_embedding_path 存在チェック
            emb_path = utt.get("speaker_embedding_path")
            if emb_path:
                full_path = output_dir / emb_path
                if not full_path.exists():
                    _LOGGER.error(
                        "Line %d: speaker_embedding_path not found: %s",
                        line_num,
                        full_path,
                    )
                    errors += 1

    # phoneme_id_map の日英衝突チェック
    from piper_train.phonemize.jp_id_map import (  # noqa: PLC0415
        get_japanese_id_map,
    )

    ja_map = get_japanese_id_map()
    ja_ids = set()
    for ids in ja_map.values():
        ja_ids.update(ids)

    try:
        from piper_phonemize import (  # noqa: PLC0415
            get_espeak_map,
        )

        en_map = get_espeak_map()
        en_ids = set()
        for ids in en_map.values():
            en_ids.update(ids)

        collision = ja_ids & en_ids
        if collision:
            _LOGGER.error(
                "phoneme_id_map collision: %d IDs overlap between ja and en: %s",
                len(collision),
                sorted(collision)[:20],
            )
            errors += 1
        else:
            _LOGGER.info(
                "phoneme_id_map: no collisions (ja=%d, en=%d IDs)",
                len(ja_ids),
                len(en_ids),
            )
    except ImportError:
        _LOGGER.warning("piper_phonemize not available; skipping ID collision check")

    _LOGGER.info(
        "Validation complete: %d entries, %d errors",
        total,
        errors,
    )
    return errors


# ------------------------------------------------------------------ #
# メイン処理
# ------------------------------------------------------------------ #
def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """エントリーポイント."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="piper_train.prepare_zero_shot_dataset",
        description=(
            "Zero-shot TTS学習用データ準備: LibriTTS-R / JVS / moe-speech を統合"
        ),
    )
    parser.add_argument(
        "--libritts-dir",
        type=Path,
        default=None,
        help="LibriTTS-R ディレクトリ",
    )
    parser.add_argument(
        "--jvs-dir",
        type=Path,
        default=None,
        help="JVS ディレクトリ",
    )
    parser.add_argument(
        "--moe-speech-dir",
        type=Path,
        default=None,
        help="moe-speech データセットディレクトリ",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="出力先ディレクトリ",
    )
    parser.add_argument(
        "--encoder",
        type=Path,
        default=None,
        help="CAM++ ONNXモデルパス (embedding抽出用)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="並列処理数 (将来の並列化予約, default: 4)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Target sample rate (default: 22050)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="生成後の検証を実行",
    )
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="音声前処理をスキップ（テスト用）",
    )
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="embedding抽出をスキップ",
    )
    args = parser.parse_args()

    # 少なくとも1つのコーパスが必要
    if not any([args.libritts_dir, args.jvs_dir, args.moe_speech_dir]):
        parser.error(
            "少なくとも1つのコーパスディレクトリを指定してください "
            "(--libritts-dir, --jvs-dir, --moe-speech-dir)"
        )

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache" / str(args.sample_rate)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- #
    # phoneme_id_map 構築
    # ---------------------------------------------------------------- #
    merged_phoneme_id_map = _build_merged_phoneme_id_map()
    _LOGGER.info(
        "Merged phoneme_id_map: %d symbols",
        len(merged_phoneme_id_map),
    )

    # ---------------------------------------------------------------- #
    # ONNX session (embedding抽出用)
    # ---------------------------------------------------------------- #
    onnx_session: onnxruntime.InferenceSession | None = None
    if args.encoder and not args.skip_embedding:
        import onnxruntime as ort  # noqa: PLC0415

        onnx_session = ort.InferenceSession(str(args.encoder))
        _LOGGER.info("Loaded speaker encoder: %s", args.encoder)

    # ---------------------------------------------------------------- #
    # 各コーパスの処理
    # ---------------------------------------------------------------- #
    all_entries: list[dict] = []
    speaker_id_map: dict[str, int] = {}
    total_speaker_count = 0

    # -- moe-speech パススルー --
    if args.moe_speech_dir:
        _LOGGER.info("=== Processing moe-speech ===")
        moe_utts = _parse_moe_speech(args.moe_speech_dir)
        moe_start = SPEAKER_ID_RANGES["moe-speech"][0]

        # moe-speech の話者IDを収集してマップ
        moe_speakers = sorted({u.get("speaker_id", 0) for u in moe_utts})
        moe_capacity = SPEAKER_ID_RANGES["moe-speech"][1] - moe_start + 1
        if len(moe_speakers) > moe_capacity:
            _LOGGER.warning(
                "moe-speech: %d speakers exceed range capacity %d (IDs %d-%d), "
                "truncating to %d speakers",
                len(moe_speakers),
                moe_capacity,
                moe_start,
                SPEAKER_ID_RANGES["moe-speech"][1],
                moe_capacity,
            )
            moe_speakers = moe_speakers[:moe_capacity]
        moe_speaker_map: dict[int, int] = {}
        for idx, orig_id in enumerate(moe_speakers):
            global_id = moe_start + idx
            moe_speaker_map[orig_id] = global_id
            speaker_id_map[f"moe-speech_{orig_id}"] = global_id

        for utt in moe_utts:
            orig_sid = utt.get("speaker_id", 0)
            global_sid = moe_speaker_map.get(orig_sid)
            if global_sid is None:
                _LOGGER.warning(
                    "moe-speech speaker_id %d not in speaker map, skipping utterance",
                    orig_sid,
                )
                continue
            emb_rel = f"speaker_embeddings/speaker_{global_sid}.npy"
            entry = {
                "phoneme_ids": utt["phoneme_ids"],
                "speaker_id": global_sid,
                "speaker_embedding_path": emb_rel,
                "prosody_features": utt.get("prosody_features"),
                "audio_norm_path": utt.get("audio_norm_path"),
                "audio_spec_path": utt.get("audio_spec_path"),
                "language": "ja",
            }
            all_entries.append(entry)

        total_speaker_count += len(moe_speakers)
        _LOGGER.info(
            "moe-speech: %d entries, %d speakers (ID %d-%d)",
            len(moe_utts),
            len(moe_speakers),
            moe_start,
            moe_start + len(moe_speakers) - 1,
        )

        # Embedding抽出
        if onnx_session is not None:
            _LOGGER.info("Extracting embeddings for moe-speech...")
            _extract_embeddings_for_moe_speech(
                session=onnx_session,
                moe_speech_dir=args.moe_speech_dir,
                output_dir=output_dir,
                speaker_id_offset=moe_start,
                source_sr=args.sample_rate,
            )

    # -- JVS --
    if args.jvs_dir:
        _LOGGER.info("=== Processing JVS ===")
        jvs_utts = _parse_jvs(args.jvs_dir)
        jvs_corpus_speaker_ids = [u["corpus_speaker_id"] for u in jvs_utts]
        jvs_id_map = _assign_speaker_ids("jvs", jvs_corpus_speaker_ids)
        for spk, gid in jvs_id_map.items():
            speaker_id_map[f"jvs_{spk}"] = gid

        # 話者ごとのWAVリスト (embedding用)
        jvs_speaker_wavs: dict[int, list[Path]] = {}

        for utt in jvs_utts:
            global_sid = jvs_id_map[utt["corpus_speaker_id"]]
            emb_rel = f"speaker_embeddings/speaker_{global_sid}.npy"

            # WAVをembedding用に記録
            jvs_speaker_wavs.setdefault(global_sid, []).append(utt["wav_path"])

            # 音声前処理
            audio_norm_path: str | None = None
            audio_spec_path: str | None = None
            if not args.skip_audio:
                result = _process_audio(
                    utt["wav_path"],
                    cache_dir,
                    args.sample_rate,
                )
                if result is None:
                    continue
                norm_p, spec_p = result
                # output_dir からの相対パスに変換
                audio_norm_path = str(norm_p.relative_to(output_dir))
                audio_spec_path = str(spec_p.relative_to(output_dir))

            # 音素化
            phoneme_ids, prosody_features = _phonemize_utterance(
                utt["text"], "ja", merged_phoneme_id_map
            )

            entry = {
                "phoneme_ids": phoneme_ids,
                "speaker_id": global_sid,
                "speaker_embedding_path": emb_rel,
                "prosody_features": prosody_features,
                "audio_norm_path": audio_norm_path,
                "audio_spec_path": audio_spec_path,
                "language": "ja",
            }
            all_entries.append(entry)

        total_speaker_count += len(jvs_id_map)
        _LOGGER.info(
            "JVS: %d entries, %d speakers",
            len(jvs_utts),
            len(jvs_id_map),
        )

        # Embedding抽出
        if onnx_session is not None:
            _LOGGER.info("Extracting embeddings for JVS...")
            _extract_embeddings_for_corpus(
                session=onnx_session,
                speaker_wav_map=jvs_speaker_wavs,
                output_dir=output_dir,
            )

    # -- LibriTTS-R --
    if args.libritts_dir:
        _LOGGER.info("=== Processing LibriTTS-R ===")
        libritts_utts = _parse_libritts(args.libritts_dir)
        libritts_corpus_speaker_ids = [u["corpus_speaker_id"] for u in libritts_utts]
        libritts_id_map = _assign_speaker_ids("libritts", libritts_corpus_speaker_ids)
        for spk, gid in libritts_id_map.items():
            speaker_id_map[f"libritts_{spk}"] = gid

        # 話者ごとのWAVリスト (embedding用)
        libritts_speaker_wavs: dict[int, list[Path]] = {}

        for i, utt in enumerate(libritts_utts):
            global_sid = libritts_id_map[utt["corpus_speaker_id"]]
            emb_rel = f"speaker_embeddings/speaker_{global_sid}.npy"

            # WAVをembedding用に記録
            libritts_speaker_wavs.setdefault(global_sid, []).append(utt["wav_path"])

            # 音声前処理
            audio_norm_path = None
            audio_spec_path = None
            if not args.skip_audio:
                result = _process_audio(
                    utt["wav_path"],
                    cache_dir,
                    args.sample_rate,
                )
                if result is None:
                    continue
                norm_p, spec_p = result
                audio_norm_path = str(norm_p.relative_to(output_dir))
                audio_spec_path = str(spec_p.relative_to(output_dir))

            # 音素化
            phoneme_ids, prosody_features = _phonemize_utterance(
                utt["text"], "en", merged_phoneme_id_map
            )

            entry = {
                "phoneme_ids": phoneme_ids,
                "speaker_id": global_sid,
                "speaker_embedding_path": emb_rel,
                "prosody_features": prosody_features,
                "audio_norm_path": audio_norm_path,
                "audio_spec_path": audio_spec_path,
                "language": "en",
            }
            all_entries.append(entry)

            if (i + 1) % 1000 == 0:
                _LOGGER.info(
                    "LibriTTS-R: processed %d / %d",
                    i + 1,
                    len(libritts_utts),
                )

        total_speaker_count += len(libritts_id_map)
        _LOGGER.info(
            "LibriTTS-R: %d entries, %d speakers",
            len(libritts_utts),
            len(libritts_id_map),
        )

        # Embedding抽出
        if onnx_session is not None:
            _LOGGER.info("Extracting embeddings for LibriTTS-R...")
            _extract_embeddings_for_corpus(
                session=onnx_session,
                speaker_wav_map=libritts_speaker_wavs,
                output_dir=output_dir,
            )

    # ---------------------------------------------------------------- #
    # dataset.jsonl 出力
    # ---------------------------------------------------------------- #
    jsonl_path = output_dir / "dataset.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for entry in all_entries:
            json.dump(entry, f, ensure_ascii=True)
            f.write("\n")
    _LOGGER.info("Wrote %d entries to %s", len(all_entries), jsonl_path)

    # ---------------------------------------------------------------- #
    # config.json 生成
    # ---------------------------------------------------------------- #
    config = {
        "audio": {"sample_rate": args.sample_rate},
        "inference": {
            "noise_scale": 0.667,
            "length_scale": 1,
            "noise_w": 0.8,
        },
        "num_speakers": total_speaker_count,
        "speaker_id_map": speaker_id_map,
        "phoneme_id_map": merged_phoneme_id_map,
        "num_symbols": len(merged_phoneme_id_map),
        "use_zero_shot": True,
        "spk_embed_dim": 192,
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=True, indent=2)
    _LOGGER.info("Wrote config to %s", config_path)

    # ---------------------------------------------------------------- #
    # サマリー
    # ---------------------------------------------------------------- #
    _LOGGER.info("=" * 60)
    _LOGGER.info("Dataset preparation complete")
    _LOGGER.info("  Total entries:  %d", len(all_entries))
    _LOGGER.info("  Total speakers: %d", total_speaker_count)
    _LOGGER.info("  Output dir:     %s", output_dir)
    _LOGGER.info("=" * 60)

    # ---------------------------------------------------------------- #
    # 検証
    # ---------------------------------------------------------------- #
    if args.validate:
        _LOGGER.info("Running validation...")
        error_count = _validate_dataset(output_dir, merged_phoneme_id_map)
        if error_count > 0:
            _LOGGER.error(
                "Validation failed with %d error(s)",
                error_count,
            )
            sys.exit(1)
        _LOGGER.info("Validation passed")


if __name__ == "__main__":
    main()
