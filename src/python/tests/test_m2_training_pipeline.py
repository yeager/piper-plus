"""
M2: 学習パイプラインのテスト
- SCL/DINO損失関数
- Dataset speaker_embedding対応
- 既存学習ループregression
"""

import pytest


try:
    import torch
except ImportError:
    torch = None

pytestmark = pytest.mark.skipif(torch is None, reason="torch not installed")

if torch is not None:
    from piper_train.vits.losses import dino_loss, speaker_consistency_loss


class TestSpeakerConsistencyLoss:
    """SCL損失関数のテスト"""

    @pytest.mark.unit
    def test_identical_embeddings_zero_loss(self):
        """同一embeddingでSCL=0"""
        emb = torch.randn(4, 192)
        loss = speaker_consistency_loss(emb, emb)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.unit
    def test_opposite_embeddings_high_loss(self):
        """逆方向embeddingでSCL≈2"""
        emb = torch.randn(4, 192)
        loss = speaker_consistency_loss(emb, -emb)
        assert loss.item() == pytest.approx(2.0, abs=1e-5)

    @pytest.mark.unit
    def test_loss_range(self):
        """ランダムembeddingでSCL ∈ [0, 2]"""
        gen_emb = torch.randn(8, 192)
        ref_emb = torch.randn(8, 192)
        loss = speaker_consistency_loss(gen_emb, ref_emb)
        assert 0.0 <= loss.item() <= 2.0

    @pytest.mark.unit
    def test_gradient_flow(self):
        """SCLのgradientが正しく流れる"""
        gen_emb = torch.randn(4, 192, requires_grad=True)
        ref_emb = torch.randn(4, 192)
        loss = speaker_consistency_loss(gen_emb, ref_emb)
        loss.backward()
        assert gen_emb.grad is not None
        assert gen_emb.grad.abs().sum() > 0


class TestDINOLoss:
    """DINO損失関数のテスト"""

    @pytest.mark.unit
    def test_positive_loss(self):
        """DINO損失が正の値を返す"""
        student = torch.randn(4, 192)
        teacher = torch.randn(4, 192)
        center = torch.zeros(192)
        loss = dino_loss(student, teacher, center)
        assert loss.item() > 0

    @pytest.mark.unit
    def test_identical_inputs_low_loss(self):
        """同一入力でDINO損失が低い"""
        emb = torch.randn(4, 192)
        center = torch.zeros(192)
        loss_same = dino_loss(emb, emb, center)
        loss_diff = dino_loss(emb, torch.randn(4, 192), center)
        # 同一入力の方が損失が低いことを確認
        assert loss_same.item() < loss_diff.item()

    @pytest.mark.unit
    def test_gradient_flow(self):
        """DINO損失のgradientがstudentに流れる"""
        student = torch.randn(4, 192, requires_grad=True)
        teacher = torch.randn(4, 192)
        center = torch.zeros(192)
        loss = dino_loss(student, teacher, center)
        loss.backward()
        assert student.grad is not None

    @pytest.mark.unit
    def test_temperature_effect(self):
        """温度パラメータが損失値に影響する"""
        student = torch.randn(4, 192)
        teacher = torch.randn(4, 192)
        center = torch.zeros(192)
        loss_high_temp = dino_loss(student, teacher, center, tau_s=1.0, tau_t=1.0)
        loss_low_temp = dino_loss(student, teacher, center, tau_s=0.01, tau_t=0.01)
        # 異なる温度で異なる損失値になることを確認
        assert loss_high_temp.item() != pytest.approx(loss_low_temp.item(), abs=0.01)


class TestDatasetSpeakerEmbedding:
    """Dataset speaker_embedding対応のテスト"""

    @pytest.mark.unit
    def test_batch_has_speaker_embeddings_field(self):
        """Batch dataclassにspeaker_embeddingsフィールドがある"""
        from piper_train.vits.dataset import Batch

        batch = Batch(
            phoneme_ids=torch.zeros(1, 10, dtype=torch.long),
            phoneme_lengths=torch.LongTensor([10]),
            spectrograms=torch.zeros(1, 80, 50),
            spectrogram_lengths=torch.LongTensor([50]),
            audios=torch.zeros(1, 1, 8192),
            audio_lengths=torch.LongTensor([8192]),
            speaker_embeddings=torch.randn(1, 192),
        )
        assert batch.speaker_embeddings is not None
        assert batch.speaker_embeddings.shape == (1, 192)

    @pytest.mark.unit
    def test_batch_speaker_embeddings_default_none(self):
        """Batch.speaker_embeddingsのデフォルトはNone"""
        from piper_train.vits.dataset import Batch

        batch = Batch(
            phoneme_ids=torch.zeros(1, 10, dtype=torch.long),
            phoneme_lengths=torch.LongTensor([10]),
            spectrograms=torch.zeros(1, 80, 50),
            spectrogram_lengths=torch.LongTensor([50]),
            audios=torch.zeros(1, 1, 8192),
            audio_lengths=torch.LongTensor([8192]),
        )
        assert batch.speaker_embeddings is None

    @pytest.mark.unit
    def test_utterance_has_speaker_embedding_path(self):
        """Utterance dataclassにspeaker_embedding_pathフィールドがある"""
        from pathlib import Path

        from piper_train.vits.dataset import Utterance

        utt = Utterance(
            phoneme_ids=[1, 2, 3],
            audio_norm_path=Path("/tmp/audio.pt"),
            audio_spec_path=Path("/tmp/spec.pt"),
            speaker_embedding_path=Path("/tmp/speaker.npy"),
        )
        assert utt.speaker_embedding_path == Path("/tmp/speaker.npy")


class TestUtteranceCollateSpeakerEmbedding:
    """UtteranceCollate.__call__() の speaker_embedding スタッキングテスト"""

    @pytest.mark.unit
    def test_collate_stacks_speaker_embeddings(self):
        """speaker_embedding付きUtteranceTensors 3つでcollateするとshape=(3,192)"""
        from piper_train.vits.dataset import UtteranceCollate, UtteranceTensors

        utts = []
        for i in range(3):
            utts.append(
                UtteranceTensors(
                    phoneme_ids=torch.LongTensor([1, 2, 3]),
                    spectrogram=torch.randn(80, 10 + i),
                    audio_norm=torch.randn(1, 8192 + i * 100),
                    speaker_embedding=torch.randn(192),
                )
            )

        collate = UtteranceCollate(is_multispeaker=False, segment_size=8192)
        batch = collate(utts)

        assert batch.speaker_embeddings is not None
        assert batch.speaker_embeddings.shape == (3, 192)

    @pytest.mark.unit
    def test_collate_mixed_speaker_embeddings_zero_fill(self):
        """speaker_embeddingがNone混在の場合、ゼロ埋めされること"""
        from piper_train.vits.dataset import UtteranceCollate, UtteranceTensors

        utt_with_emb = UtteranceTensors(
            phoneme_ids=torch.LongTensor([1, 2, 3]),
            spectrogram=torch.randn(80, 10),
            audio_norm=torch.randn(1, 8192),
            speaker_embedding=torch.randn(192),
        )
        utt_without_emb = UtteranceTensors(
            phoneme_ids=torch.LongTensor([4, 5, 6]),
            spectrogram=torch.randn(80, 12),
            audio_norm=torch.randn(1, 8300),
            speaker_embedding=None,
        )
        utt_with_emb2 = UtteranceTensors(
            phoneme_ids=torch.LongTensor([7, 8]),
            spectrogram=torch.randn(80, 8),
            audio_norm=torch.randn(1, 8100),
            speaker_embedding=torch.randn(192),
        )

        collate = UtteranceCollate(is_multispeaker=False, segment_size=8192)
        batch = collate([utt_with_emb, utt_without_emb, utt_with_emb2])

        assert batch.speaker_embeddings is not None
        assert batch.speaker_embeddings.shape == (3, 192)
        # sorted by decreasing spec_length: utt_without_emb(12), utt_with_emb(10), utt_with_emb2(8)
        # index 0 = utt_without_emb (None -> zero), should be all zeros
        assert torch.all(batch.speaker_embeddings[0] == 0.0)
        # index 1 and 2 should have non-zero embeddings
        assert batch.speaker_embeddings[1].abs().sum() > 0
        assert batch.speaker_embeddings[2].abs().sum() > 0


class TestNpLoadSecurity:
    """np.load(allow_pickle=False) セキュリティテスト"""

    @pytest.mark.unit
    def test_np_load_allow_pickle_false_normal_array(self, tmp_path):
        """正常な配列を保存し、allow_pickle=Falseで正常に読める"""
        import numpy as np

        arr = np.random.randn(192).astype(np.float32)
        npy_path = tmp_path / "normal.npy"
        np.save(str(npy_path), arr)

        loaded = np.load(str(npy_path), allow_pickle=False)
        assert loaded.shape == (192,)
        assert loaded.dtype == np.float32
        np.testing.assert_array_equal(loaded, arr)

    @pytest.mark.unit
    def test_np_load_allow_pickle_false_rejects_pickle(self, tmp_path):
        """pickleオブジェクトを含む.npyはallow_pickle=Falseで読めない"""
        import pickle

        import numpy as np

        # Create a pickle-based .npy file manually
        pickle_path = tmp_path / "pickle_obj.npy"
        obj = {"malicious": "data"}
        with open(str(pickle_path), "wb") as f:
            pickle.dump(obj, f)

        with pytest.raises((ValueError, OSError)):
            np.load(str(pickle_path), allow_pickle=False)


class TestTrainingLoopRegression:
    """既存学習ループのregressionテスト"""

    @pytest.mark.unit
    def test_existing_losses_unchanged(self):
        """既存損失関数が変更されていない"""
        from piper_train.vits.losses import (
            discriminator_loss,
            feature_loss,
            generator_loss,
        )

        # feature_loss
        fmap_r = [[torch.randn(2, 64, 100)]]
        fmap_g = [[torch.randn(2, 64, 100)]]
        fl = feature_loss(fmap_r, fmap_g)
        assert fl.item() >= 0

        # discriminator_loss
        dr = [torch.randn(2, 1)]
        dg = [torch.randn(2, 1)]
        dl, _, _ = discriminator_loss(dr, dg)
        assert dl.item() >= 0

        # generator_loss
        gl, _ = generator_loss(dg)
        assert gl.item() >= 0
