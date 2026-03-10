"""
Zero-Shot TTS dual-mode tests for SynthesizerTrn
Tests for use_zero_shot=True (spk_proj) and use_zero_shot=False (emb_g) modes
"""

import pytest

try:
    import torch
except ImportError:
    torch = None

pytestmark = pytest.mark.skipif(torch is None, reason="torch not installed")

if torch is not None:
    from piper_train.vits.models import SynthesizerTrn

# 最小限のモデルパラメータ
MODEL_PARAMS = dict(
    n_vocab=50,
    spec_channels=513,
    segment_size=8192,
    inter_channels=192,
    hidden_channels=192,
    filter_channels=768,
    n_heads=2,
    n_layers=6,
    kernel_size=3,
    p_dropout=0.1,
    resblock="1",
    resblock_kernel_sizes=[3, 7, 11],
    resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    upsample_rates=[8, 8, 2, 2],
    upsample_initial_channel=512,
    upsample_kernel_sizes=[16, 16, 4, 4],
    prosody_dim=16,
)


class TestZeroShotInit:
    """Test dual-mode initialization"""

    @pytest.mark.unit
    def test_zero_shot_creates_spk_proj(self):
        """use_zero_shot=True で spk_proj が存在し、emb_g が存在しない"""
        model = SynthesizerTrn(
            **MODEL_PARAMS,
            n_speakers=1,
            gin_channels=768,
            use_zero_shot=True,
            spk_embed_dim=192,
        )
        assert hasattr(model, 'spk_proj')
        assert not hasattr(model, 'emb_g')
        assert model.spk_proj.weight.shape == (768, 192)

    @pytest.mark.unit
    def test_multispeaker_creates_emb_g(self):
        """use_zero_shot=False, n_speakers>1 で emb_g が存在し、spk_proj が存在しない"""
        model = SynthesizerTrn(
            **MODEL_PARAMS,
            n_speakers=20,
            gin_channels=768,
            use_zero_shot=False,
        )
        assert hasattr(model, 'emb_g')
        assert not hasattr(model, 'spk_proj')
        assert model.emb_g.weight.shape == (20, 768)

    @pytest.mark.unit
    def test_single_speaker_no_embedding(self):
        """use_zero_shot=False, n_speakers=1 でどちらも存在しない"""
        model = SynthesizerTrn(
            **MODEL_PARAMS,
            n_speakers=1,
            gin_channels=0,
            use_zero_shot=False,
        )
        assert not hasattr(model, 'emb_g')
        assert not hasattr(model, 'spk_proj')

    @pytest.mark.unit
    def test_spk_proj_parameter_count(self):
        """spk_proj のパラメータ数 = 192 * 768 + 768 = 148,224"""
        model = SynthesizerTrn(
            **MODEL_PARAMS,
            n_speakers=1,
            gin_channels=768,
            use_zero_shot=True,
            spk_embed_dim=192,
        )
        param_count = sum(p.numel() for p in model.spk_proj.parameters())
        assert param_count == 192 * 768 + 768  # weight + bias

    @pytest.mark.unit
    def test_zero_shot_with_multi_speakers(self):
        """use_zero_shot=True + n_speakers>1 で spk_proj と emb_g の両方が作られる (Dual-Mode)"""
        model = SynthesizerTrn(
            **MODEL_PARAMS,
            n_speakers=20,
            gin_channels=768,
            use_zero_shot=True,
            spk_embed_dim=192,
        )
        assert hasattr(model, 'spk_proj')
        assert hasattr(model, 'emb_g')

    @pytest.mark.unit
    def test_use_zero_shot_attribute(self):
        """use_zero_shot属性が正しく保存される"""
        model_zs = SynthesizerTrn(**MODEL_PARAMS, n_speakers=1, gin_channels=768, use_zero_shot=True)
        model_normal = SynthesizerTrn(**MODEL_PARAMS, n_speakers=20, gin_channels=768, use_zero_shot=False)
        assert model_zs.use_zero_shot is True
        assert model_normal.use_zero_shot is False


class TestZeroShotForward:
    """Test forward pass with speaker_embedding"""

    @pytest.mark.unit
    @pytest.mark.training
    def test_forward_with_speaker_embedding(self):
        """forward()にspeaker_embeddingを渡して正常動作"""
        torch.manual_seed(42)
        model = SynthesizerTrn(
            **MODEL_PARAMS,
            n_speakers=1,
            gin_channels=768,
            use_zero_shot=True,
            spk_embed_dim=192,
        )
        model.train()

        batch_size = 2
        text_len = 10
        spec_len = 50

        x = torch.randint(0, 50, (batch_size, text_len))
        x_lengths = torch.LongTensor([text_len, text_len])
        y = torch.randn(batch_size, 513, spec_len)
        y_lengths = torch.LongTensor([spec_len, spec_len])
        speaker_embedding = torch.randn(batch_size, 192)

        output = model.forward(
            x, x_lengths, y, y_lengths,
            speaker_embedding=speaker_embedding,
        )
        # output: (o, l_length, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q))
        assert output[0] is not None  # audio output
        assert output[0].shape[0] == batch_size

    @pytest.mark.unit
    @pytest.mark.training
    def test_forward_multispeaker_regression(self):
        """既存のmultispeaker forward (sid) が引き続き動作"""
        torch.manual_seed(42)
        model = SynthesizerTrn(
            **MODEL_PARAMS,
            n_speakers=20,
            gin_channels=768,
            use_zero_shot=False,
        )
        model.train()

        batch_size = 2
        text_len = 10
        spec_len = 50

        x = torch.randint(0, 50, (batch_size, text_len))
        x_lengths = torch.LongTensor([text_len, text_len])
        y = torch.randn(batch_size, 513, spec_len)
        y_lengths = torch.LongTensor([spec_len, spec_len])
        sid = torch.LongTensor([0, 1])

        output = model.forward(x, x_lengths, y, y_lengths, sid=sid)
        assert output[0] is not None
        assert output[0].shape[0] == batch_size

    @pytest.mark.unit
    @pytest.mark.training
    def test_forward_zero_shot_requires_speaker_embedding(self):
        """use_zero_shot=True で speaker_embedding=None だと ValueError"""
        torch.manual_seed(42)
        model = SynthesizerTrn(
            **MODEL_PARAMS,
            n_speakers=1,
            gin_channels=768,
            use_zero_shot=True,
            spk_embed_dim=192,
        )
        model.train()

        batch_size = 2
        text_len = 10
        spec_len = 50

        x = torch.randint(0, 50, (batch_size, text_len))
        x_lengths = torch.LongTensor([text_len, text_len])
        y = torch.randn(batch_size, 513, spec_len)
        y_lengths = torch.LongTensor([spec_len, spec_len])

        with pytest.raises(ValueError, match="speaker_embedding is required"):
            model.forward(x, x_lengths, y, y_lengths)

    @pytest.mark.unit
    @pytest.mark.training
    def test_forward_output_is_finite(self):
        """forward() の出力が有限値であること"""
        torch.manual_seed(42)
        model = SynthesizerTrn(
            **MODEL_PARAMS,
            n_speakers=1,
            gin_channels=768,
            use_zero_shot=True,
            spk_embed_dim=192,
        )
        model.train()

        batch_size = 2
        text_len = 10
        spec_len = 50

        x = torch.randint(0, 50, (batch_size, text_len))
        x_lengths = torch.LongTensor([text_len, text_len])
        y = torch.randn(batch_size, 513, spec_len)
        y_lengths = torch.LongTensor([spec_len, spec_len])
        speaker_embedding = torch.randn(batch_size, 192)

        output = model.forward(
            x, x_lengths, y, y_lengths,
            speaker_embedding=speaker_embedding,
        )
        audio = output[0]
        assert torch.isfinite(audio).all(), "Output contains NaN or Inf"


class TestZeroShotInfer:
    """Test inference with speaker_embedding"""

    @pytest.mark.unit
    @pytest.mark.inference
    def test_infer_with_speaker_embedding(self):
        """infer()にspeaker_embeddingを渡して音声テンソルを返す"""
        torch.manual_seed(42)
        model = SynthesizerTrn(
            **MODEL_PARAMS,
            n_speakers=1,
            gin_channels=768,
            use_zero_shot=True,
            spk_embed_dim=192,
        )
        model.eval()

        text_len = 10
        x = torch.randint(0, 50, (1, text_len))
        x_lengths = torch.LongTensor([text_len])
        speaker_embedding = torch.randn(1, 192)

        with torch.no_grad():
            output = model.infer(
                x, x_lengths,
                speaker_embedding=speaker_embedding,
            )
        # output: (o, attn, y_mask, (z, z_p, m_p, logs_p))
        audio = output[0]
        assert audio is not None
        assert audio.dim() == 3  # [batch, channels, time]
        assert audio.shape[0] == 1

    @pytest.mark.unit
    @pytest.mark.inference
    def test_infer_multispeaker_regression(self):
        """既存のmultispeaker infer (sid) が引き続き動作"""
        torch.manual_seed(42)
        model = SynthesizerTrn(
            **MODEL_PARAMS,
            n_speakers=20,
            gin_channels=768,
            use_zero_shot=False,
        )
        model.eval()

        x = torch.randint(0, 50, (1, 10))
        x_lengths = torch.LongTensor([10])
        sid = torch.LongTensor([0])

        with torch.no_grad():
            output = model.infer(x, x_lengths, sid=sid)
        assert output[0] is not None
        assert output[0].shape[0] == 1

    @pytest.mark.unit
    @pytest.mark.inference
    def test_infer_single_speaker_regression(self):
        """シングルスピーカーモデルが引き続き動作"""
        torch.manual_seed(42)
        model = SynthesizerTrn(
            **MODEL_PARAMS,
            n_speakers=1,
            gin_channels=0,
            use_zero_shot=False,
        )
        model.eval()

        x = torch.randint(0, 50, (1, 10))
        x_lengths = torch.LongTensor([10])

        with torch.no_grad():
            output = model.infer(x, x_lengths)
        assert output[0] is not None
        assert output[0].shape[0] == 1

    @pytest.mark.unit
    @pytest.mark.inference
    def test_infer_zero_shot_requires_speaker_embedding(self):
        """infer() で use_zero_shot=True かつ speaker_embedding=None だと ValueError"""
        torch.manual_seed(42)
        model = SynthesizerTrn(
            **MODEL_PARAMS,
            n_speakers=1,
            gin_channels=768,
            use_zero_shot=True,
            spk_embed_dim=192,
        )
        model.eval()

        x = torch.randint(0, 50, (1, 10))
        x_lengths = torch.LongTensor([10])

        with torch.no_grad():
            with pytest.raises(ValueError, match="speaker_embedding is required"):
                model.infer(x, x_lengths)

    @pytest.mark.unit
    @pytest.mark.inference
    def test_infer_output_range(self):
        """infer() の出力が [-1, 1] 範囲であること (tanh)"""
        torch.manual_seed(42)
        model = SynthesizerTrn(
            **MODEL_PARAMS,
            n_speakers=1,
            gin_channels=768,
            use_zero_shot=True,
            spk_embed_dim=192,
        )
        model.eval()

        x = torch.randint(0, 50, (1, 10))
        x_lengths = torch.LongTensor([10])
        speaker_embedding = torch.randn(1, 192)

        with torch.no_grad():
            output = model.infer(x, x_lengths, speaker_embedding=speaker_embedding)
        audio = output[0]
        assert torch.isfinite(audio).all(), "Output contains NaN or Inf"
        assert audio.abs().max() <= 1.0, "Audio output should be in [-1, 1] (tanh)"
