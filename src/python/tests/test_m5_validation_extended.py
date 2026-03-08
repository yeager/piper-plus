"""
M5 Extended: _validate_dataset / _extract_embeddings_for_moe_speech /
_assign_speaker_ids の追加テスト.

既存の test_m5_prepare_zero_shot_dataset.py と並行して別エージェントが
編集している可能性があるため、独立したファイルとして作成。
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from piper_train.prepare_zero_shot_dataset import (
    SPEAKER_ID_RANGES,
    _assign_speaker_ids,
    _extract_embeddings_for_moe_speech,
    _validate_dataset,
)


# Ensure piper_train.extract_speaker_embedding is importable even when
# torch is not installed.  The module is imported lazily inside
# _extract_embeddings_for_moe_speech, but unittest.mock.patch needs to
# traverse the dotted path at patch time.  If the real module cannot be
# imported (torch missing), we insert a lightweight stub so that patch()
# can resolve the attribute.
if "piper_train.extract_speaker_embedding" not in sys.modules:
    try:
        import piper_train.extract_speaker_embedding  # noqa: F401
    except ImportError:
        _stub = types.ModuleType("piper_train.extract_speaker_embedding")
        _stub.extract_from_dataset = None  # type: ignore[attr-defined]
        sys.modules["piper_train.extract_speaker_embedding"] = _stub


# ================================================================== #
# 1. _validate_dataset() 追加テストケース
# ================================================================== #
class TestValidateDatasetExtended:
    """_validate_dataset の追加検証."""

    @pytest.mark.unit
    def test_prosody_features_length_mismatch(self, tmp_path: Path) -> None:
        """prosody_features の長さが phoneme_ids と不一致の場合エラーになる."""
        jsonl_path = tmp_path / "dataset.jsonl"
        entry = {
            "phoneme_ids": [1, 2, 3],
            "speaker_id": 0,
            "prosody_features": [None, None],  # 長さ2 != phoneme_ids長さ3
            "language": "ja",
        }
        with open(jsonl_path, "w") as f:
            json.dump(entry, f)
            f.write("\n")

        errors = _validate_dataset(tmp_path, {})
        assert errors >= 1, "prosody_features length mismatch should produce an error"

    @pytest.mark.unit
    def test_speaker_embedding_path_not_found(self, tmp_path: Path) -> None:
        """speaker_embedding_path が存在しない場合にエラーが返る."""
        jsonl_path = tmp_path / "dataset.jsonl"
        entry = {
            "phoneme_ids": [1, 2],
            "speaker_id": 0,
            "speaker_embedding_path": "speaker_embeddings/speaker_999.npy",
            "prosody_features": [None, None],
            "language": "ja",
        }
        with open(jsonl_path, "w") as f:
            json.dump(entry, f)
            f.write("\n")

        errors = _validate_dataset(tmp_path, {})
        assert errors >= 1, "missing speaker_embedding_path should produce an error"

    @pytest.mark.unit
    def test_dataset_jsonl_not_found(self, tmp_path: Path) -> None:
        """dataset.jsonl が存在しない場合に return 1."""
        # tmp_path には dataset.jsonl を作らない
        errors = _validate_dataset(tmp_path, {})
        assert errors == 1, "missing dataset.jsonl should return 1"

    @pytest.mark.unit
    def test_multiple_errors_detected(self, tmp_path: Path) -> None:
        """複数エラーが同時に検出される場合のエラーカウント."""
        jsonl_path = tmp_path / "dataset.jsonl"
        with open(jsonl_path, "w") as f:
            # エントリ1: 空phoneme_ids (1 error) + 存在しないaudio_norm_path (1 error)
            entry1 = {
                "phoneme_ids": [],
                "speaker_id": 0,
                "audio_norm_path": "nonexistent_audio.pt",
                "language": "ja",
            }
            json.dump(entry1, f)
            f.write("\n")

            # エントリ2: prosody長不一致 (1 error) + 存在しないspec_path (1 error)
            entry2 = {
                "phoneme_ids": [1, 2, 3],
                "speaker_id": 1,
                "prosody_features": [None],  # 長さ1 != 3
                "audio_spec_path": "nonexistent_spec.pt",
                "language": "ja",
            }
            json.dump(entry2, f)
            f.write("\n")

        errors = _validate_dataset(tmp_path, {})
        # entry1: empty phoneme_ids + missing audio_norm_path = 2
        # entry2: prosody mismatch + missing audio_spec_path = 2
        # total >= 4
        assert errors >= 4, (
            f"Expected at least 4 errors from multiple issues, got {errors}"
        )

    @pytest.mark.unit
    def test_validate_phoneme_id_map_basic(self, tmp_path: Path) -> None:
        """_validate_dataset はphoneme_id_map衝突チェックも行う (基本validation)."""
        # 正常なデータセットで衝突チェック部分が例外を出さないことを確認
        jsonl_path = tmp_path / "dataset.jsonl"
        entry = {
            "phoneme_ids": [1, 2],
            "speaker_id": 0,
            "prosody_features": [None, None],
            "language": "ja",
        }
        with open(jsonl_path, "w") as f:
            json.dump(entry, f)
            f.write("\n")

        # phoneme_id_map は _validate_dataset 内部で直接使わず、
        # jp_id_map + piper_phonemize から再構築する。
        # piper_phonemize が無い環境でも例外にならないことを確認。
        errors = _validate_dataset(tmp_path, {"a": [1], "b": [2]})
        # piper_phonemize がなければ衝突チェックはスキップされ、
        # 基本バリデーションは通るので errors == 0
        assert errors == 0


# ================================================================== #
# 2. _extract_embeddings_for_moe_speech() ID オフセットロジック
# ================================================================== #
class TestExtractEmbeddingsForMoeSpeech:
    """_extract_embeddings_for_moe_speech の ID オフセットテスト."""

    @pytest.mark.unit
    def test_speaker_id_offset_renames_files(self, tmp_path: Path) -> None:
        """speaker_id_offset により出力ファイルが speaker_{offset+id}.npy になる."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        moe_speech_dir = tmp_path / "moe_speech"
        moe_speech_dir.mkdir()

        offset = 5

        def mock_extract_from_dataset(
            session,
            dataset_dir,
            output_dir: Path,
            max_utterances=10,
            source_sr=22050,
            **kwargs,
        ):
            """モック: tmp_path に speaker_0.npy, speaker_1.npy を生成."""
            output_dir.mkdir(parents=True, exist_ok=True)
            np.save(str(output_dir / "speaker_0.npy"), np.zeros(192))
            np.save(str(output_dir / "speaker_1.npy"), np.ones(192))

        with patch(
            "piper_train.extract_speaker_embedding.extract_from_dataset",
            side_effect=mock_extract_from_dataset,
        ):
            _extract_embeddings_for_moe_speech(
                session=None,  # モックなので不使用
                moe_speech_dir=moe_speech_dir,
                output_dir=output_dir,
                speaker_id_offset=offset,
                max_utterances=10,
                source_sr=22050,
            )

        emb_dir = output_dir / "speaker_embeddings"
        assert (emb_dir / f"speaker_{offset}.npy").exists(), (
            f"speaker_{offset}.npy should exist (offset + 0)"
        )
        assert (emb_dir / f"speaker_{offset + 1}.npy").exists(), (
            f"speaker_{offset + 1}.npy should exist (offset + 1)"
        )

        # 元のID名のファイルは存在しないこと
        assert not (emb_dir / "speaker_0.npy").exists()
        assert not (emb_dir / "speaker_1.npy").exists()

    @pytest.mark.unit
    def test_speaker_id_offset_preserves_embedding_content(
        self, tmp_path: Path
    ) -> None:
        """オフセット適用後もembeddingの内容が保持される."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        moe_speech_dir = tmp_path / "moe_speech"
        moe_speech_dir.mkdir()

        offset = 10
        expected_emb = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        def mock_extract_from_dataset(session, dataset_dir, output_dir: Path, **kwargs):
            output_dir.mkdir(parents=True, exist_ok=True)
            np.save(str(output_dir / "speaker_0.npy"), expected_emb)

        with patch(
            "piper_train.extract_speaker_embedding.extract_from_dataset",
            side_effect=mock_extract_from_dataset,
        ):
            _extract_embeddings_for_moe_speech(
                session=None,
                moe_speech_dir=moe_speech_dir,
                output_dir=output_dir,
                speaker_id_offset=offset,
            )

        result = np.load(
            str(output_dir / "speaker_embeddings" / f"speaker_{offset}.npy")
        )
        np.testing.assert_array_equal(result, expected_emb)


# ================================================================== #
# 3. _assign_speaker_ids() 容量超過テスト
# ================================================================== #
class TestAssignSpeakerIdsCapacity:
    """_assign_speaker_ids の容量超過時の truncation テスト."""

    @pytest.mark.unit
    def test_moe_speech_truncates_excess_speakers(self) -> None:
        """moe-speech 容量20に25話者を渡すと20話者に切り詰められる."""
        start, end = SPEAKER_ID_RANGES["moe-speech"]
        capacity = end - start + 1  # 20
        assert capacity == 20, f"Expected moe-speech capacity 20, got {capacity}"

        # 25話者を渡す
        speakers = [str(i) for i in range(25)]
        mapping = _assign_speaker_ids("moe-speech", speakers)

        assert len(mapping) == capacity, (
            f"Expected {capacity} speakers after truncation, got {len(mapping)}"
        )
        # 割り当てられたIDが全て範囲内であること
        for global_id in mapping.values():
            assert start <= global_id <= end, (
                f"Global ID {global_id} out of range [{start}, {end}]"
            )

    @pytest.mark.unit
    def test_jvs_truncates_excess_speakers(self) -> None:
        """JVS 容量100に110話者を渡すと100話者に切り詰められる."""
        start, end = SPEAKER_ID_RANGES["jvs"]
        capacity = end - start + 1  # 100

        speakers = [f"jvs{i:03d}" for i in range(1, 111)]  # 110話者
        mapping = _assign_speaker_ids("jvs", speakers)

        assert len(mapping) == capacity, (
            f"Expected {capacity} speakers after truncation, got {len(mapping)}"
        )
        for global_id in mapping.values():
            assert start <= global_id <= end

    @pytest.mark.unit
    def test_within_capacity_no_truncation(self) -> None:
        """容量以内であれば切り詰めが起こらない."""
        speakers = [str(i) for i in range(10)]  # 10 < 20 (moe-speech capacity)
        mapping = _assign_speaker_ids("moe-speech", speakers)

        assert len(mapping) == 10
        start, end = SPEAKER_ID_RANGES["moe-speech"]
        for global_id in mapping.values():
            assert start <= global_id <= end

    @pytest.mark.unit
    def test_exact_capacity(self) -> None:
        """容量と同じ話者数であれば切り詰めなし."""
        start, end = SPEAKER_ID_RANGES["moe-speech"]
        capacity = end - start + 1
        speakers = [str(i) for i in range(capacity)]
        mapping = _assign_speaker_ids("moe-speech", speakers)

        assert len(mapping) == capacity
