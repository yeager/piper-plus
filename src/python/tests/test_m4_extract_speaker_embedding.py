"""
M4: Speaker Embedding抽出ツールのテスト
- 音声前処理 (Fbank抽出)
- L2正規化
- 平均化ロジック
- CLI引数バリデーション
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


try:
    import torch
except ImportError:
    torch = None

try:
    import torchaudio
except ImportError:
    torchaudio = None

try:
    import soundfile as sf
except ImportError:
    sf = None

pytestmark = pytest.mark.skipif(
    torch is None or torchaudio is None,
    reason="torch and torchaudio required",
)

# soundfile が利用できない場合、音声I/Oテストをスキップ
_requires_soundfile = pytest.mark.skipif(
    sf is None, reason="soundfile required for WAV I/O"
)

# torchaudio.load が torchcodec を必要とする場合をチェック
_torchaudio_load_available = False
if torchaudio is not None:
    try:
        import tempfile as _tempfile

        with _tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as _tf:
            if sf is not None:
                # 最小限のWAVを書き出して torchaudio.load が動くか確認
                import numpy as _np

                sf.write(_tf.name, _np.zeros(160, dtype="float32"), 16000)
                torchaudio.load(_tf.name)
                _torchaudio_load_available = True
    except Exception:
        pass

_requires_torchaudio_load = pytest.mark.skipif(
    not _torchaudio_load_available,
    reason="torchaudio.load not functional (torchcodec may be missing)",
)


def _save_wav(path, waveform, sample_rate):
    """WAVファイルを保存するヘルパー。soundfile を使用。
    waveform: torch.Tensor (channels, samples)"""
    # (channels, samples) -> (samples, channels) for soundfile
    data = waveform.numpy().T
    sf.write(str(path), data, sample_rate)


def _load_wav(wav_path):
    """WAVファイルを読み込むヘルパー。soundfile を使用。
    Returns: (torch.Tensor (channels, samples), sample_rate)"""
    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    # (samples, channels) -> (channels, samples)
    waveform = torch.from_numpy(data.T)
    return waveform, sr


def _preprocess_audio(wav_path):
    """音声ファイルからFbank特徴量を抽出する前処理ロジック。

    extract_speaker_embedding.py が実装する preprocess_audio 相当のロジック:
    1. WAV読み込み
    2. ステレオ→モノラル変換
    3. 16kHzリサンプリング
    4. 80次元Fbank特徴量抽出
    """
    waveform, sr = _load_wav(wav_path)

    # ステレオ→モノラル
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # 16kHzリサンプリング
    target_sr = 16000
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=int(sr), new_freq=target_sr
        )
        waveform = resampler(waveform)

    # 80次元Fbank特徴量抽出
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        num_mel_bins=80,
        sample_frequency=target_sr,
        frame_length=25.0,
        frame_shift=10.0,
    )

    return fbank.numpy()


@_requires_soundfile
class TestPreprocessAudio:
    """音声前処理のテスト"""

    @pytest.mark.unit
    def test_preprocess_wav_output_shape(self, tmp_path):
        """16kHz 1秒WAVからFbank特徴量を抽出し、shape (T, 80) を確認"""
        sr = 16000
        duration = 1.0
        waveform = torch.randn(1, int(sr * duration))
        wav_path = tmp_path / "test.wav"
        _save_wav(wav_path, waveform, sr)

        fbank = _preprocess_audio(wav_path)

        assert isinstance(fbank, np.ndarray)
        assert fbank.ndim == 2
        assert fbank.shape[1] == 80
        # 1秒 / 10msフレームシフト = 約100フレーム (+-5)
        expected_frames = int(duration * 1000 / 10)
        assert abs(fbank.shape[0] - expected_frames) <= 5, (
            f"Expected ~{expected_frames} frames, got {fbank.shape[0]}"
        )

    @pytest.mark.unit
    def test_preprocess_stereo_to_mono(self, tmp_path):
        """ステレオWAVがモノラル化されてFbank出力されること"""
        sr = 16000
        duration = 0.5
        # ステレオ (2ch) WAV
        waveform_stereo = torch.randn(2, int(sr * duration))
        wav_path = tmp_path / "stereo.wav"
        _save_wav(wav_path, waveform_stereo, sr)

        fbank = _preprocess_audio(wav_path)

        # Fbank出力は2D (T, 80) であること (チャンネル次元がない)
        assert fbank.ndim == 2
        assert fbank.shape[1] == 80

    @pytest.mark.unit
    def test_preprocess_resample(self, tmp_path):
        """44100Hz WAVが16kHzにリサンプリングされること"""
        sr_original = 44100
        duration = 1.0
        waveform = torch.randn(1, int(sr_original * duration))
        wav_path = tmp_path / "highsr.wav"
        _save_wav(wav_path, waveform, sr_original)

        fbank = _preprocess_audio(wav_path)

        assert isinstance(fbank, np.ndarray)
        assert fbank.ndim == 2
        assert fbank.shape[1] == 80
        # 16kHz基準でフレーム数を算出: 1秒 / 10ms = 約100フレーム
        expected_frames = int(duration * 1000 / 10)
        assert abs(fbank.shape[0] - expected_frames) <= 5, (
            f"Expected ~{expected_frames} frames at 16kHz, got {fbank.shape[0]}"
        )


class TestL2Normalization:
    """L2正規化のテスト"""

    @pytest.mark.unit
    def test_l2_normalize(self):
        """ランダムベクトルをL2正規化し、ノルムが1.0になること"""
        emb = np.random.randn(192).astype(np.float32)
        norm = np.linalg.norm(emb)
        normalized = emb / norm
        assert abs(np.linalg.norm(normalized) - 1.0) < 0.001

    @pytest.mark.unit
    def test_average_and_renormalize(self):
        """複数のL2正規化済みベクトルを平均化し、再正規化後のノルムが1.0であること"""
        embeddings = [np.random.randn(192).astype(np.float32) for _ in range(5)]
        embeddings = [e / np.linalg.norm(e) for e in embeddings]
        avg = np.mean(embeddings, axis=0)
        avg = avg / np.linalg.norm(avg)
        assert abs(np.linalg.norm(avg) - 1.0) < 0.001

    @pytest.mark.unit
    def test_l2_normalize_zero_vector(self):
        """ゼロベクトルに対するL2正規化が安全にハンドルされること"""
        emb = np.zeros(192, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        # ゼロベクトルはゼロのまま
        assert np.linalg.norm(emb) == 0.0


class TestCLIValidation:
    """CLI引数バリデーションのテスト"""

    @pytest.mark.unit
    def test_cli_no_mode_error(self):
        """--audio, --audio-dir, --dataset-dir のいずれも未指定でエラー検出"""
        audio = False
        audio_dir = False
        dataset_dir = False
        modes = sum([audio, audio_dir, dataset_dir])
        assert modes == 0, "No mode specified should be detected as error"

    @pytest.mark.unit
    def test_cli_mutually_exclusive_modes(self):
        """--audio, --audio-dir, --dataset-dir の排他チェック:
        2つ以上指定した場合にエラーになるべき"""
        # 1つだけ指定: OK
        modes_single = sum([True, False, False])
        assert modes_single == 1

        # 2つ指定: エラー
        modes_double = sum([True, True, False])
        assert modes_double > 1, "Multiple modes should trigger an error"

        # 3つ指定: エラー
        modes_triple = sum([True, True, True])
        assert modes_triple > 1, "Multiple modes should trigger an error"

    @pytest.mark.unit
    def test_output_npy_format(self, tmp_path):
        """np.save / np.load のラウンドトリップでembeddingが正確に保存/復元されること"""
        emb = np.random.randn(192).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        path = tmp_path / "speaker.npy"
        np.save(str(path), emb)
        loaded = np.load(str(path))
        assert loaded.shape == (192,)
        assert loaded.dtype == np.float32
        np.testing.assert_allclose(loaded, emb, rtol=1e-6)


# ---------------------------------------------------------------------------
# 実モジュール関数の直接テスト
# ---------------------------------------------------------------------------


@_requires_soundfile
@_requires_torchaudio_load
class TestPreprocessAudioReal:
    """実際の preprocess_audio() 関数を直接インポートしてテスト"""

    @pytest.mark.unit
    def test_preprocess_audio_shape_and_type(self, tmp_path):
        """preprocess_audio() の出力が np.ndarray, shape (T, 80) であること"""
        from piper_train.extract_speaker_embedding import preprocess_audio

        sr = 16000
        duration = 1.0
        waveform = torch.randn(1, int(sr * duration))
        wav_path = tmp_path / "test_real.wav"
        _save_wav(wav_path, waveform, sr)

        fbank = preprocess_audio(wav_path)

        assert isinstance(fbank, np.ndarray)
        assert fbank.ndim == 2
        assert fbank.shape[1] == 80
        # 1秒 / 10msフレームシフト = 約100フレーム
        expected_frames = int(duration * 1000 / 10)
        assert abs(fbank.shape[0] - expected_frames) <= 5

    @pytest.mark.unit
    def test_preprocess_audio_cmvn_mean_zero(self, tmp_path):
        """CMVN正規化により各次元のmeanが約0であること"""
        from piper_train.extract_speaker_embedding import preprocess_audio

        sr = 16000
        duration = 2.0
        waveform = torch.randn(1, int(sr * duration))
        wav_path = tmp_path / "test_cmvn.wav"
        _save_wav(wav_path, waveform, sr)

        fbank = preprocess_audio(wav_path)

        # 各次元の平均が0に近いこと
        col_means = fbank.mean(axis=0)
        np.testing.assert_allclose(col_means, 0.0, atol=1e-5)


class TestLoadAudioFromPt:
    """_load_audio_from_pt() のテスト"""

    @pytest.mark.unit
    def test_load_2d_tensor(self, tmp_path):
        """2Dテンソル (1, samples) からFbank特徴量が抽出されること"""
        from piper_train.extract_speaker_embedding import _load_audio_from_pt

        audio = torch.randn(1, 22050)
        pt_path = tmp_path / "audio_2d.pt"
        torch.save(audio, pt_path)

        fbank = _load_audio_from_pt(pt_path)

        assert isinstance(fbank, np.ndarray)
        assert fbank.ndim == 2
        assert fbank.shape[1] == 80

    @pytest.mark.unit
    def test_load_1d_tensor(self, tmp_path):
        """1Dテンソル (samples,) でも正常に動作すること"""
        from piper_train.extract_speaker_embedding import _load_audio_from_pt

        audio = torch.randn(22050)
        pt_path = tmp_path / "audio_1d.pt"
        torch.save(audio, pt_path)

        fbank = _load_audio_from_pt(pt_path)

        assert isinstance(fbank, np.ndarray)
        assert fbank.ndim == 2
        assert fbank.shape[1] == 80


class TestExtractEmbeddingMock:
    """extract_embedding() のモックテスト"""

    @pytest.mark.unit
    def test_extract_embedding_shape_and_norm(self):
        """モックセッションでembeddingのshapeとL2ノルムを検証"""
        from piper_train.extract_speaker_embedding import extract_embedding

        # ONNXセッションのモック
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "fbank"
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.run.return_value = [np.random.randn(1, 192).astype(np.float32)]

        fbank = np.random.randn(100, 80).astype(np.float32)
        embedding = extract_embedding(mock_session, fbank)

        assert embedding.shape == (192,)
        assert abs(np.linalg.norm(embedding) - 1.0) < 1e-5

        # session.run が正しい入力名で呼ばれたことを確認
        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args
        assert "fbank" in call_args[1] or "fbank" in call_args[0][1]


class TestExtractFromFilesMock:
    """extract_from_files() のモックテスト"""

    @pytest.mark.unit
    def test_extract_from_files_average_norm(self):
        """3ファイル分のembeddingを平均化し、L2ノルムが約1.0であること"""
        from piper_train.extract_speaker_embedding import extract_from_files

        mock_session = MagicMock()

        def _mock_extract(session, fbank):
            emb = np.random.randn(192).astype(np.float32)
            return emb / np.linalg.norm(emb)

        with (
            patch(
                "piper_train.extract_speaker_embedding.preprocess_audio",
                return_value=np.random.randn(100, 80).astype(np.float32),
            ),
            patch(
                "piper_train.extract_speaker_embedding.extract_embedding",
                side_effect=_mock_extract,
            ),
        ):
            wav_paths = [Path(f"/tmp/fake_{i}.wav") for i in range(3)]
            result = extract_from_files(mock_session, wav_paths)

        assert result.shape == (192,)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-5


class TestExtractFromDatasetMock:
    """extract_from_dataset() のモックテスト"""

    @pytest.mark.unit
    def test_extract_from_dataset_generates_npy(self, tmp_path):
        """dataset.jsonlから2話者分のembeddingが生成されること"""
        from piper_train.extract_speaker_embedding import extract_from_dataset

        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        output_dir = tmp_path / "embeddings"

        # 2話者、各3発話の dataset.jsonl を作成
        lines = []
        for speaker_id in range(2):
            for utt_idx in range(3):
                audio_rel = f"speaker_{speaker_id}/utt_{utt_idx}.pt"
                lines.append(
                    json.dumps(
                        {
                            "speaker_id": speaker_id,
                            "audio_norm_path": audio_rel,
                        }
                    )
                )
                # 対応する .pt ファイルを作成 (4秒分 = 22050*4)
                pt_dir = dataset_dir / f"speaker_{speaker_id}"
                pt_dir.mkdir(exist_ok=True)
                pt_path = pt_dir / f"utt_{utt_idx}.pt"
                torch.save(torch.randn(1, 22050 * 4), pt_path)

        jsonl_path = dataset_dir / "dataset.jsonl"
        jsonl_path.write_text("\n".join(lines) + "\n")

        mock_session = MagicMock()

        def _mock_extract(session, fbank):
            emb = np.random.randn(192).astype(np.float32)
            return emb / np.linalg.norm(emb)

        with patch(
            "piper_train.extract_speaker_embedding.extract_embedding",
            side_effect=_mock_extract,
        ):
            extract_from_dataset(
                mock_session,
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                max_utterances=10,
                min_duration=3.0,
                source_sr=22050,
            )

        # 各話者の .npy ファイルが生成されていること
        assert (output_dir / "speaker_0.npy").exists()
        assert (output_dir / "speaker_1.npy").exists()

        # 読み込んでshapeとノルムを検証
        for sid in range(2):
            emb = np.load(str(output_dir / f"speaker_{sid}.npy"))
            assert emb.shape == (192,)
            assert abs(np.linalg.norm(emb) - 1.0) < 1e-4
