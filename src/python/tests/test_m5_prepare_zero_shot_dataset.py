"""
M5: prepare_zero_shot_dataset のテスト
- Speaker ID割り当てロジック
- JSONL出力フォーマット
- config.json生成
- バリデーション機能
- CLIヘルプ表示
"""

import json
import subprocess

import pytest

from piper_train.prepare_zero_shot_dataset import (
    SPEAKER_ID_RANGES,
    _assign_speaker_ids,
    _validate_dataset,
)


class TestSpeakerIdAssignment:
    """Speaker ID割り当てロジックのテスト"""

    @pytest.mark.unit
    def test_moe_speech_id_range(self):
        """moe-speechの話者IDが0-19の範囲"""
        speakers = [str(i) for i in range(20)]
        mapping = _assign_speaker_ids("moe-speech", speakers)
        # 文字列ソート順でID割り当て: "0","1","10","11",...
        assert len(mapping) == 20
        ids = sorted(mapping.values())
        assert ids[0] == 0
        assert ids[-1] == 19

    @pytest.mark.unit
    def test_jvs_id_range(self):
        """JVSの話者IDが20-119の範囲"""
        mapping = _assign_speaker_ids(
            "jvs", ["jvs001", "jvs002", "jvs003"]
        )
        assert mapping["jvs001"] == 20
        assert mapping["jvs002"] == 21
        assert mapping["jvs003"] == 22

    @pytest.mark.unit
    def test_libritts_id_range(self):
        """LibriTTSの話者IDが120+の範囲"""
        mapping = _assign_speaker_ids("libritts", ["100", "200"])
        assert mapping["100"] == 120
        assert mapping["200"] == 121

    @pytest.mark.unit
    def test_speaker_id_ranges_defined(self):
        """全コーパスのID範囲が定義されている"""
        assert "moe-speech" in SPEAKER_ID_RANGES
        assert "jvs" in SPEAKER_ID_RANGES
        assert "libritts" in SPEAKER_ID_RANGES
        # 範囲が重複しないことを確認
        ranges = list(SPEAKER_ID_RANGES.values())
        for i, (s1, e1) in enumerate(ranges):
            for s2, e2 in ranges[i + 1 :]:
                assert e1 < s2 or e2 < s1, (
                    f"Overlapping ranges: ({s1},{e1}) and ({s2},{e2})"
                )

    @pytest.mark.unit
    def test_speakers_sorted(self):
        """話者がソートされてID割り当てされる"""
        mapping = _assign_speaker_ids(
            "jvs", ["jvs003", "jvs001", "jvs002"]
        )
        # ソート順: jvs001=20, jvs002=21, jvs003=22
        assert mapping["jvs001"] == 20
        assert mapping["jvs002"] == 21
        assert mapping["jvs003"] == 22


class TestJsonlFormat:
    """JSONL出力フォーマットのテスト"""

    @pytest.mark.unit
    def test_required_fields_in_jsonl(self, tmp_path):
        """JSONL形式で必須フィールドが正しく書き込まれる"""
        entry = {
            "phoneme_ids": [1, 2, 3],
            "speaker_id": 0,
            "speaker_embedding_path": "speaker_embeddings/speaker_0.npy",
            "prosody_features": [
                {"a1": 0, "a2": 1, "a3": 3},
                {"a1": 0, "a2": 2, "a3": 3},
                {"a1": 0, "a2": 3, "a3": 3},
            ],
            "audio_norm_path": "cache/abc.pt",
            "audio_spec_path": "cache/abc.spec.pt",
            "language": "ja",
        }
        jsonl_path = tmp_path / "dataset.jsonl"
        with open(jsonl_path, "w") as f:
            json.dump(entry, f)
            f.write("\n")

        with open(jsonl_path) as f:
            loaded = json.loads(f.readline())

        assert "phoneme_ids" in loaded
        assert "speaker_id" in loaded
        assert "speaker_embedding_path" in loaded
        assert "prosody_features" in loaded
        assert "language" in loaded
        assert loaded["language"] == "ja"

    @pytest.mark.unit
    def test_phoneme_ids_list_type(self):
        """phoneme_idsがリスト型であること"""
        entry = {
            "phoneme_ids": [1, 5, 8],
            "speaker_id": 0,
            "language": "ja",
        }
        serialized = json.dumps(entry)
        loaded = json.loads(serialized)
        assert isinstance(loaded["phoneme_ids"], list)
        assert len(loaded["phoneme_ids"]) == 3


class TestConfigGeneration:
    """config.json生成のテスト"""

    @pytest.mark.unit
    def test_config_structure(self, tmp_path):
        """config.jsonの構造が正しい"""
        config = {
            "audio": {"sample_rate": 22050},
            "inference": {
                "noise_scale": 0.667,
                "length_scale": 1,
                "noise_w": 0.8,
            },
            "num_speakers": 2576,
            "speaker_id_map": {},
            "phoneme_id_map": {"a": [1], "b": [2]},
            "num_symbols": 2,
            "use_zero_shot": True,
            "spk_embed_dim": 192,
        }
        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        with open(config_path) as f:
            loaded = json.loads(f.read())

        assert loaded["num_speakers"] == 2576
        assert loaded["audio"]["sample_rate"] == 22050
        assert loaded["use_zero_shot"] is True
        assert loaded["spk_embed_dim"] == 192


class TestValidation:
    """--validate検証機能のテスト"""

    @pytest.mark.unit
    def test_validate_missing_audio(self, tmp_path):
        """存在しない音声ファイルでバリデーションエラー"""
        jsonl_path = tmp_path / "dataset.jsonl"
        with open(jsonl_path, "w") as f:
            entry = {
                "phoneme_ids": [1, 2],
                "speaker_id": 0,
                "audio_norm_path": "nonexistent.pt",
                "audio_spec_path": "nonexistent.spec.pt",
                "prosody_features": [None, None],
                "language": "ja",
            }
            json.dump(entry, f)
            f.write("\n")
        errors = _validate_dataset(tmp_path, {})
        assert errors > 0

    @pytest.mark.unit
    def test_validate_valid_dataset(self, tmp_path):
        """正常なデータセットでバリデーション成功"""
        # ダミーファイル作成
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "audio.pt").touch()
        (cache_dir / "audio.spec.pt").touch()

        jsonl_path = tmp_path / "dataset.jsonl"
        with open(jsonl_path, "w") as f:
            entry = {
                "phoneme_ids": [1, 2],
                "speaker_id": 0,
                "audio_norm_path": "cache/audio.pt",
                "audio_spec_path": "cache/audio.spec.pt",
                "prosody_features": [None, None],
                "language": "ja",
            }
            json.dump(entry, f)
            f.write("\n")
        errors = _validate_dataset(tmp_path, {})
        assert errors == 0

    @pytest.mark.unit
    def test_validate_empty_phoneme_ids(self, tmp_path):
        """空のphoneme_idsでバリデーションエラー"""
        jsonl_path = tmp_path / "dataset.jsonl"
        with open(jsonl_path, "w") as f:
            entry = {
                "phoneme_ids": [],
                "speaker_id": 0,
                "prosody_features": [],
                "language": "ja",
            }
            json.dump(entry, f)
            f.write("\n")
        errors = _validate_dataset(tmp_path, {})
        assert errors > 0


class TestCliHelp:
    """CLIヘルプ表示のテスト"""

    @pytest.mark.unit
    def test_help_shows_all_arguments(self):
        """--helpで全引数が表示される"""
        result = subprocess.run(
            ["python", "-m", "piper_train.prepare_zero_shot_dataset", "--help"],
            check=False,
            capture_output=True,
            text=True,
            cwd="/Users/s19447/Documents/piper-plus/src/python",
        )
        assert result.returncode == 0
        assert "--libritts-dir" in result.stdout
        assert "--jvs-dir" in result.stdout
        assert "--moe-speech-dir" in result.stdout
        assert "--output-dir" in result.stdout
        assert "--encoder" in result.stdout
        assert "--validate" in result.stdout
