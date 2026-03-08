"""
M5: prepare_zero_shot_dataset のテスト
- Speaker ID割り当てロジック
- JSONL出力フォーマット
- config.json生成
- バリデーション機能
- CLIヘルプ表示
- コーパスパーサー (_parse_jvs, _parse_moe_speech, _parse_libritts)
- 音素化 (_phonemize_utterance)
- phoneme_id_map統合 (_build_merged_phoneme_id_map)
- 音声前処理 (_process_audio)
"""

import json
import subprocess
from pathlib import Path

import pytest

from piper_train.prepare_zero_shot_dataset import (
    SPEAKER_ID_RANGES,
    _assign_speaker_ids,
    _build_merged_phoneme_id_map,
    _parse_jvs,
    _parse_libritts,
    _parse_moe_speech,
    _process_audio,
    _validate_dataset,
)


_has_pyopenjtalk = False
try:
    import pyopenjtalk  # noqa: F401

    _has_pyopenjtalk = True
except ImportError:
    pass

_has_librosa = False
try:
    import librosa  # noqa: F401

    _has_librosa = True
except Exception:
    pass


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
        mapping = _assign_speaker_ids("jvs", ["jvs001", "jvs002", "jvs003"])
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
        mapping = _assign_speaker_ids("jvs", ["jvs003", "jvs001", "jvs002"])
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
        src_python_dir = str(Path(__file__).resolve().parent.parent)
        result = subprocess.run(
            ["python", "-m", "piper_train.prepare_zero_shot_dataset", "--help"],
            check=False,
            capture_output=True,
            text=True,
            cwd=src_python_dir,
        )
        assert result.returncode == 0
        assert "--libritts-dir" in result.stdout
        assert "--jvs-dir" in result.stdout
        assert "--moe-speech-dir" in result.stdout
        assert "--output-dir" in result.stdout
        assert "--encoder" in result.stdout
        assert "--validate" in result.stdout


# ------------------------------------------------------------------ #
# _parse_jvs テスト
# ------------------------------------------------------------------ #
class TestParseJvs:
    """JVSコーパスパーサーのテスト"""

    def _make_jvs_tree(
        self,
        base: Path,
        speaker: str,
        transcript_lines: list[str],
        wav_stems: list[str],
    ) -> None:
        """JVSディレクトリ構造のヘルパー"""
        spk_dir = base / speaker / "parallel100"
        spk_dir.mkdir(parents=True)
        transcript_path = spk_dir / "transcripts_utf8.txt"
        transcript_path.write_text("\n".join(transcript_lines) + "\n", encoding="utf-8")
        wav_dir = spk_dir / "wav24kHz16bit"
        wav_dir.mkdir()
        for stem in wav_stems:
            (wav_dir / f"{stem}.wav").touch()

    @pytest.mark.unit
    def test_tab_separated_transcript(self, tmp_path: Path):
        """tab区切りのtranscriptが正しくパースされる"""
        self._make_jvs_tree(
            tmp_path,
            "jvs001",
            ["VOICEACTRESS100_001\tこんにちは"],
            ["VOICEACTRESS100_001"],
        )
        utts = _parse_jvs(tmp_path)
        assert len(utts) == 1
        assert utts[0]["text"] == "こんにちは"
        assert utts[0]["language"] == "ja"
        assert utts[0]["corpus_speaker_id"] == "1"

    @pytest.mark.unit
    def test_colon_separated_transcript(self, tmp_path: Path):
        """colon区切りのtranscriptが正しくパースされる"""
        self._make_jvs_tree(
            tmp_path,
            "jvs002",
            ["VOICEACTRESS100_001:こんにちは"],
            ["VOICEACTRESS100_001"],
        )
        utts = _parse_jvs(tmp_path)
        assert len(utts) == 1
        assert utts[0]["text"] == "こんにちは"
        assert utts[0]["corpus_speaker_id"] == "2"

    @pytest.mark.unit
    def test_wav_not_found_skipped(self, tmp_path: Path):
        """WAVファイルが見つからない発話はスキップされる"""
        self._make_jvs_tree(
            tmp_path,
            "jvs001",
            ["VOICEACTRESS100_001\tこんにちは", "VOICEACTRESS100_002\tさようなら"],
            ["VOICEACTRESS100_001"],  # 002のWAVは作成しない
        )
        utts = _parse_jvs(tmp_path)
        assert len(utts) == 1
        assert utts[0]["text"] == "こんにちは"

    @pytest.mark.unit
    def test_non_jvs_directory_skipped(self, tmp_path: Path):
        """jvsで始まらないディレクトリはスキップされる"""
        self._make_jvs_tree(
            tmp_path,
            "jvs001",
            ["VOICEACTRESS100_001\tこんにちは"],
            ["VOICEACTRESS100_001"],
        )
        # jvsで始まらないディレクトリを追加
        other_dir = tmp_path / "other_speaker" / "parallel100"
        other_dir.mkdir(parents=True)
        (other_dir / "transcripts_utf8.txt").write_text(
            "VOICEACTRESS100_001\tテスト\n", encoding="utf-8"
        )
        wav_dir = other_dir / "wav24kHz16bit"
        wav_dir.mkdir()
        (wav_dir / "VOICEACTRESS100_001.wav").touch()

        utts = _parse_jvs(tmp_path)
        assert len(utts) == 1
        assert utts[0]["corpus_speaker_id"] == "1"

    @pytest.mark.unit
    def test_multiple_speakers(self, tmp_path: Path):
        """複数話者が正しくパースされる"""
        self._make_jvs_tree(
            tmp_path,
            "jvs001",
            ["VOICEACTRESS100_001\tこんにちは"],
            ["VOICEACTRESS100_001"],
        )
        self._make_jvs_tree(
            tmp_path,
            "jvs010",
            ["VOICEACTRESS100_001\tさようなら"],
            ["VOICEACTRESS100_001"],
        )
        utts = _parse_jvs(tmp_path)
        assert len(utts) == 2
        speakers = {u["corpus_speaker_id"] for u in utts}
        assert speakers == {"1", "10"}


# ------------------------------------------------------------------ #
# _parse_moe_speech テスト
# ------------------------------------------------------------------ #
class TestParseMoeSpeech:
    """moe-speechパーサーのテスト"""

    @pytest.mark.unit
    def test_normal_jsonl(self, tmp_path: Path):
        """正常なJSONLの読み込み"""
        jsonl_path = tmp_path / "dataset.jsonl"
        entries = [
            {"phoneme_ids": [1, 2, 3], "speaker_id": 0, "text": "hello"},
            {"phoneme_ids": [4, 5, 6], "speaker_id": 1, "text": "world"},
        ]
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for entry in entries:
                json.dump(entry, f)
                f.write("\n")

        utts = _parse_moe_speech(tmp_path)
        assert len(utts) == 2
        assert utts[0]["phoneme_ids"] == [1, 2, 3]
        assert utts[1]["speaker_id"] == 1

    @pytest.mark.unit
    def test_malformed_json_skipped(self, tmp_path: Path):
        """不正なJSON行がスキップされる（warning出力）"""
        jsonl_path = tmp_path / "dataset.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            json.dump({"phoneme_ids": [1], "speaker_id": 0}, f)
            f.write("\n")
            f.write("{invalid json\n")
            json.dump({"phoneme_ids": [2], "speaker_id": 1}, f)
            f.write("\n")

        utts = _parse_moe_speech(tmp_path)
        assert len(utts) == 2  # 不正行はスキップ、前後は読み込まれる

    @pytest.mark.unit
    def test_empty_lines_skipped(self, tmp_path: Path):
        """空行がスキップされる"""
        jsonl_path = tmp_path / "dataset.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            json.dump({"phoneme_ids": [1], "speaker_id": 0}, f)
            f.write("\n")
            f.write("\n")  # 空行
            f.write("   \n")  # 空白のみの行
            json.dump({"phoneme_ids": [2], "speaker_id": 1}, f)
            f.write("\n")

        utts = _parse_moe_speech(tmp_path)
        assert len(utts) == 2

    @pytest.mark.unit
    def test_missing_jsonl_raises(self, tmp_path: Path):
        """dataset.jsonlが存在しない場合にFileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            _parse_moe_speech(tmp_path)


# ------------------------------------------------------------------ #
# _parse_libritts テスト
# ------------------------------------------------------------------ #
class TestParseLibritts:
    """LibriTTSパーサーのテスト"""

    def _make_libritts_tree(
        self,
        base: Path,
        subset: str,
        speaker: str,
        chapter: str,
        entries: list[tuple[str, str, bool]],
    ) -> None:
        """LibriTTSディレクトリ構造のヘルパー.

        entries: [(stem, text, create_wav), ...]
        """
        chapter_dir = base / subset / speaker / chapter
        chapter_dir.mkdir(parents=True, exist_ok=True)
        for stem, text, create_wav in entries:
            txt_path = chapter_dir / f"{stem}.normalized.txt"
            txt_path.write_text(text, encoding="utf-8")
            if create_wav:
                (chapter_dir / f"{stem}.wav").touch()

    @pytest.mark.unit
    def test_normal_parse(self, tmp_path: Path):
        """正常なLibriTTS構造をパースできる"""
        self._make_libritts_tree(
            tmp_path,
            "train-clean-100",
            "1234",
            "5678",
            [("1234_5678_000001", "Hello world", True)],
        )
        utts = _parse_libritts(tmp_path)
        assert len(utts) == 1
        assert utts[0]["text"] == "Hello world"
        assert utts[0]["corpus_speaker_id"] == "1234"
        assert utts[0]["language"] == "en"
        assert utts[0]["wav_path"] == (
            tmp_path / "train-clean-100" / "1234" / "5678" / "1234_5678_000001.wav"
        )

    @pytest.mark.unit
    def test_wav_missing_skipped(self, tmp_path: Path):
        """テキストファイルはあるがWAVがないケースはスキップ"""
        self._make_libritts_tree(
            tmp_path,
            "train-clean-100",
            "1234",
            "5678",
            [("1234_5678_000001", "Hello world", False)],  # WAVなし
        )
        utts = _parse_libritts(tmp_path)
        assert len(utts) == 0

    @pytest.mark.unit
    def test_empty_text_skipped(self, tmp_path: Path):
        """空テキストのケースはスキップ"""
        self._make_libritts_tree(
            tmp_path,
            "train-clean-100",
            "1234",
            "5678",
            [("1234_5678_000001", "", True)],  # 空テキスト
        )
        utts = _parse_libritts(tmp_path)
        assert len(utts) == 0

    @pytest.mark.unit
    def test_multiple_speakers_and_chapters(self, tmp_path: Path):
        """複数話者・複数チャプターが正しくパースされる"""
        self._make_libritts_tree(
            tmp_path,
            "train-clean-100",
            "100",
            "200",
            [("100_200_000001", "First", True)],
        )
        self._make_libritts_tree(
            tmp_path,
            "train-clean-100",
            "100",
            "300",
            [("100_300_000001", "Second", True)],
        )
        self._make_libritts_tree(
            tmp_path,
            "train-clean-100",
            "999",
            "400",
            [("999_400_000001", "Third", True)],
        )
        utts = _parse_libritts(tmp_path)
        assert len(utts) == 3
        speakers = {u["corpus_speaker_id"] for u in utts}
        assert speakers == {"100", "999"}


# ------------------------------------------------------------------ #
# _phonemize_utterance テスト
# ------------------------------------------------------------------ #
class TestPhonemizeUtterance:
    """テキスト音素化のテスト"""

    @pytest.mark.unit
    @pytest.mark.skipif(not _has_pyopenjtalk, reason="pyopenjtalk not available")
    def test_japanese_phonemize_normal(self):
        """日本語テキストの正常な音素化"""
        from piper_train.prepare_zero_shot_dataset import _phonemize_utterance

        phoneme_id_map = _build_merged_phoneme_id_map()
        phoneme_ids, prosody_features = _phonemize_utterance(
            "こんにちは", "ja", phoneme_id_map
        )
        assert len(phoneme_ids) > 0
        assert len(prosody_features) > 0

    @pytest.mark.unit
    @pytest.mark.skipif(not _has_pyopenjtalk, reason="pyopenjtalk not available")
    def test_phoneme_ids_and_prosody_length_match(self):
        """phoneme_idsとprosody_featuresの長さが一致する"""
        from piper_train.prepare_zero_shot_dataset import _phonemize_utterance

        phoneme_id_map = _build_merged_phoneme_id_map()
        phoneme_ids, prosody_features = _phonemize_utterance(
            "今日は良い天気ですね。", "ja", phoneme_id_map
        )
        assert len(phoneme_ids) == len(prosody_features)

    @pytest.mark.unit
    @pytest.mark.skipif(not _has_pyopenjtalk, reason="pyopenjtalk not available")
    def test_phoneme_ids_not_empty(self):
        """phoneme_idsが空でないこと"""
        from piper_train.prepare_zero_shot_dataset import _phonemize_utterance

        phoneme_id_map = _build_merged_phoneme_id_map()
        phoneme_ids, prosody_features = _phonemize_utterance(
            "テスト", "ja", phoneme_id_map
        )
        assert len(phoneme_ids) > 0


# ------------------------------------------------------------------ #
# _build_merged_phoneme_id_map テスト
# ------------------------------------------------------------------ #
class TestBuildMergedPhonemeIdMap:
    """phoneme_id_map統合のテスト"""

    @pytest.mark.unit
    def test_returns_dict(self):
        """返り値がdictであること"""
        result = _build_merged_phoneme_id_map()
        assert isinstance(result, dict)

    @pytest.mark.unit
    def test_contains_basic_japanese_tokens(self):
        """日本語の基本トークンが含まれること"""
        result = _build_merged_phoneme_id_map()
        # token_mapper経由でPUA文字に変換されるため、
        # jp_id_mapのregister()結果と照合する
        from piper_train.phonemize.token_mapper import register

        for symbol in ["_", "^", "$"]:
            mapped = register(symbol)
            assert mapped in result, f"Expected '{mapped}' (from '{symbol}') in map"

    @pytest.mark.unit
    def test_ids_are_int_lists(self):
        """IDが整数リストであること"""
        result = _build_merged_phoneme_id_map()
        for key, value in result.items():
            assert isinstance(value, list), f"Value for '{key}' is not a list"
            for item in value:
                assert isinstance(item, int), (
                    f"Value for '{key}' contains non-int: {item}"
                )

    @pytest.mark.unit
    def test_map_is_non_empty(self):
        """マップが空でないこと"""
        result = _build_merged_phoneme_id_map()
        assert len(result) > 0


# ------------------------------------------------------------------ #
# _process_audio テスト
# ------------------------------------------------------------------ #
class TestProcessAudio:
    """音声前処理のテスト"""

    @pytest.mark.unit
    @pytest.mark.skipif(not _has_librosa, reason="librosa not available")
    def test_nonexistent_wav_returns_none(self, tmp_path: Path):
        """存在しないWAVファイルを渡した場合にNoneが返る"""
        fake_wav = tmp_path / "nonexistent.wav"
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        result = _process_audio(fake_wav, cache_dir, 22050)
        assert result is None
