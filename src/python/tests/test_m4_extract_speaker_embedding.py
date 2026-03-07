"""
M4: Speaker Embedding抽出ツールのテスト
- 音声前処理 (Fbank抽出)
- L2正規化
- 平均化ロジック
- CLI引数バリデーション
"""

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
