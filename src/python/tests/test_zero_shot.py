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
    from piper_train.vits.models import (
        DurationPredictor,
        Generator,
        ResidualCouplingBlock,
        StochasticDurationPredictor,
        SynthesizerTrn,
        TextEncoder,
    )

# 最小限のモデルパラメータ
MODEL_PARAMS = {
    "n_vocab": 50,
    "spec_channels": 513,
    "segment_size": 8192,
    "inter_channels": 192,
    "hidden_channels": 192,
    "filter_channels": 768,
    "n_heads": 2,
    "n_layers": 6,
    "kernel_size": 3,
    "p_dropout": 0.1,
    "resblock": "1",
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "upsample_rates": [8, 8, 2, 2],
    "upsample_initial_channel": 512,
    "upsample_kernel_sizes": [16, 16, 4, 4],
    "prosody_dim": 16,
}


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
        # spk_proj is now nn.Sequential; first layer maps spk_embed_dim -> gin_channels
        assert model.spk_proj[0].weight.shape == (768, 192)

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
        """Verify spk_proj parameter count is in a reasonable range.

        The spk_proj MLP has: Linear(192,768) + LayerNorm(768) + GELU + Linear(768,768).
        Total architecture parameter count may shift as other components change,
        so we verify the spk_proj sub-module in isolation.
        """
        model = SynthesizerTrn(
            **MODEL_PARAMS,
            n_speakers=1,
            gin_channels=768,
            use_zero_shot=True,
            spk_embed_dim=192,
        )
        param_count = sum(p.numel() for p in model.spk_proj.parameters())
        # Linear(192,768): 192*768 + 768 = 148,224
        # LayerNorm(768): 768 + 768 = 1,536
        # Linear(768,768): 768*768 + 768 = 590,592
        # Total: 740,352
        expected = (192 * 768 + 768) + (768 + 768) + (768 * 768 + 768)
        assert param_count == expected, (
            f"spk_proj param count {param_count} != expected {expected}"
        )

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

        with pytest.raises(ValueError, match="speaker_embedding.*must be provided|speaker_embedding is required"):
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
            with pytest.raises(ValueError, match="speaker_embedding.*must be provided|speaker_embedding is required"):
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


class TestTextEncoderSpeakerConditioning:
    """Test TextEncoder accepts g parameter and produces speaker-dependent output"""

    @pytest.mark.unit
    def test_text_encoder_has_cond_layer(self):
        """TextEncoder with gin_channels > 0 has a cond layer"""
        enc = TextEncoder(
            n_vocab=50,
            out_channels=192,
            hidden_channels=192,
            filter_channels=768,
            n_heads=2,
            n_layers=6,
            kernel_size=3,
            p_dropout=0.1,
            gin_channels=512,
        )
        assert hasattr(enc, 'cond'), "TextEncoder should have cond layer when gin_channels > 0"
        # cond maps gin_channels -> hidden_channels
        assert enc.cond.weight.shape == (192, 512, 1), (
            f"Expected cond weight shape (192, 512, 1), got {enc.cond.weight.shape}"
        )

    @pytest.mark.unit
    def test_text_encoder_no_cond_layer_without_gin(self):
        """TextEncoder with gin_channels=0 has no cond layer"""
        enc = TextEncoder(
            n_vocab=50,
            out_channels=192,
            hidden_channels=192,
            filter_channels=768,
            n_heads=2,
            n_layers=6,
            kernel_size=3,
            p_dropout=0.1,
            gin_channels=0,
        )
        assert not hasattr(enc, 'cond'), "TextEncoder should not have cond layer when gin_channels=0"

    @pytest.mark.unit
    def test_text_encoder_accepts_g_parameter(self):
        """TextEncoder forward accepts g and runs without error"""
        torch.manual_seed(42)
        enc = TextEncoder(
            n_vocab=50,
            out_channels=192,
            hidden_channels=192,
            filter_channels=768,
            n_heads=2,
            n_layers=6,
            kernel_size=3,
            p_dropout=0.1,
            gin_channels=512,
        )
        enc.eval()

        batch_size = 2
        text_len = 10
        x = torch.randint(0, 50, (batch_size, text_len))
        x_lengths = torch.LongTensor([text_len, text_len])
        g = torch.randn(batch_size, 512, 1)  # [batch, gin_channels, 1]

        with torch.no_grad():
            out_x, m, logs, x_mask = enc(x, x_lengths, g=g)

        assert out_x.shape == (batch_size, 192, text_len)
        assert m.shape == (batch_size, 192, text_len)
        assert logs.shape == (batch_size, 192, text_len)

    @pytest.mark.unit
    def test_text_encoder_different_g_produces_different_output(self):
        """Different speaker embeddings g produce different TextEncoder outputs"""
        torch.manual_seed(42)
        enc = TextEncoder(
            n_vocab=50,
            out_channels=192,
            hidden_channels=192,
            filter_channels=768,
            n_heads=2,
            n_layers=6,
            kernel_size=3,
            p_dropout=0.1,
            gin_channels=512,
        )
        # cond is zero-initialized for stability; set non-zero weights to test conditioning
        with torch.no_grad():
            enc.cond.weight.normal_(0, 0.01)
        enc.eval()

        x = torch.randint(0, 50, (1, 10))
        x_lengths = torch.LongTensor([10])
        g1 = torch.randn(1, 512, 1)
        g2 = torch.randn(1, 512, 1)

        with torch.no_grad():
            out1, m1, logs1, _ = enc(x, x_lengths, g=g1)
            out2, m2, logs2, _ = enc(x, x_lengths, g=g2)

        # Different g should produce different outputs
        assert not torch.allclose(m1, m2, atol=1e-5), (
            "TextEncoder should produce different m with different speaker conditioning"
        )

    @pytest.mark.unit
    def test_text_encoder_g_none_still_works(self):
        """TextEncoder with gin_channels > 0 still works when g=None (no conditioning)"""
        torch.manual_seed(42)
        enc = TextEncoder(
            n_vocab=50,
            out_channels=192,
            hidden_channels=192,
            filter_channels=768,
            n_heads=2,
            n_layers=6,
            kernel_size=3,
            p_dropout=0.1,
            gin_channels=512,
        )
        enc.eval()

        x = torch.randint(0, 50, (1, 10))
        x_lengths = torch.LongTensor([10])

        with torch.no_grad():
            out_x, m, logs, x_mask = enc(x, x_lengths, g=None)

        assert out_x.shape[0] == 1
        assert torch.isfinite(m).all()


class TestGeneratorFiLMConditioning:
    """Test Generator FiLM (Feature-wise Linear Modulation) conditioning"""

    @pytest.mark.unit
    def test_generator_cond_doubled_output(self):
        """Generator cond layer has 2x upsample_initial_channel output for FiLM (scale + shift)"""
        gen = Generator(
            initial_channel=192,
            resblock="1",
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_rates=[8, 8, 2, 2],
            upsample_initial_channel=512,
            upsample_kernel_sizes=[16, 16, 4, 4],
            gin_channels=768,
        )
        assert hasattr(gen, 'cond'), "Generator should have cond layer when gin_channels > 0"
        # FiLM: output is 2 * upsample_initial_channel (scale + shift)
        expected_out_channels = 512 * 2  # upsample_initial_channel * 2
        assert gen.cond.weight.shape[0] == expected_out_channels, (
            f"Generator cond output should be {expected_out_channels} for FiLM, "
            f"got {gen.cond.weight.shape[0]}"
        )

    @pytest.mark.unit
    def test_generator_cond_input_channels(self):
        """Generator cond layer input matches gin_channels"""
        gin_channels = 512
        gen = Generator(
            initial_channel=192,
            resblock="1",
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_rates=[8, 8, 2, 2],
            upsample_initial_channel=512,
            upsample_kernel_sizes=[16, 16, 4, 4],
            gin_channels=gin_channels,
        )
        assert gen.cond.weight.shape[1] == gin_channels, (
            f"Generator cond input should be {gin_channels}, got {gen.cond.weight.shape[1]}"
        )

    @pytest.mark.unit
    def test_generator_film_forward(self):
        """Generator forward with g produces valid output using FiLM conditioning"""
        torch.manual_seed(42)
        gen = Generator(
            initial_channel=192,
            resblock="1",
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_rates=[8, 8, 2, 2],
            upsample_initial_channel=512,
            upsample_kernel_sizes=[16, 16, 4, 4],
            gin_channels=768,
        )
        gen.eval()

        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, 192, seq_len)
        g = torch.randn(batch_size, 768, 1)

        with torch.no_grad():
            out = gen(x, g=g)

        assert out.shape[0] == batch_size
        assert out.shape[1] == 1  # mono audio
        assert torch.isfinite(out).all(), "Generator FiLM output should be finite"

    @pytest.mark.unit
    def test_generator_film_different_g(self):
        """Different g vectors produce different Generator outputs (FiLM is effective)"""
        torch.manual_seed(42)
        gen = Generator(
            initial_channel=192,
            resblock="1",
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_rates=[8, 8, 2, 2],
            upsample_initial_channel=512,
            upsample_kernel_sizes=[16, 16, 4, 4],
            gin_channels=768,
        )
        # FiLM cond is zero-initialized for stability; set non-zero weights to test
        with torch.no_grad():
            gen.cond.weight.normal_(0, 0.01)
        gen.eval()

        x = torch.randn(1, 192, 10)
        g1 = torch.randn(1, 768, 1)
        g2 = torch.randn(1, 768, 1)

        with torch.no_grad():
            out1 = gen(x, g=g1)
            out2 = gen(x, g=g2)

        assert not torch.allclose(out1, out2, atol=1e-5), (
            "Generator should produce different outputs with different FiLM conditioning"
        )

    @pytest.mark.unit
    def test_generator_no_cond_without_gin(self):
        """Generator with gin_channels=0 has no cond layer"""
        gen = Generator(
            initial_channel=192,
            resblock="1",
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_rates=[8, 8, 2, 2],
            upsample_initial_channel=512,
            upsample_kernel_sizes=[16, 16, 4, 4],
            gin_channels=0,
        )
        assert not hasattr(gen, 'cond'), "Generator should not have cond layer when gin_channels=0"


class TestFlowVarianceLearning:
    """Test that Flow uses mean_only=False for variance learning"""

    @pytest.mark.unit
    def test_flow_mean_only_false(self):
        """ResidualCouplingBlock layers have mean_only=False"""
        flow = ResidualCouplingBlock(
            channels=192,
            hidden_channels=192,
            kernel_size=5,
            dilation_rate=2,
            n_layers=4,
            n_flows=4,
            gin_channels=768,
        )
        for i, module in enumerate(flow.flows):
            if hasattr(module, 'mean_only'):
                assert module.mean_only is False, (
                    f"Flow layer {i} should have mean_only=False, got True"
                )

    @pytest.mark.unit
    def test_flow_post_output_channels_doubled(self):
        """With mean_only=False, ResidualCouplingLayer.post outputs 2x half_channels (mean + log_var)"""
        flow = ResidualCouplingBlock(
            channels=192,
            hidden_channels=192,
            kernel_size=5,
            dilation_rate=2,
            n_layers=4,
            n_flows=4,
            gin_channels=768,
        )
        half_channels = 192 // 2  # 96
        for module in flow.flows:
            if hasattr(module, 'post'):
                # mean_only=False: post outputs half_channels * 2 = channels
                expected = half_channels * 2
                actual = module.post.weight.shape[0]
                assert actual == expected, (
                    f"Flow post output should be {expected} (mean_only=False), got {actual}"
                )

    @pytest.mark.unit
    def test_synthesizer_flow_mean_only_false(self):
        """SynthesizerTrn flow uses mean_only=False"""
        model = SynthesizerTrn(
            **MODEL_PARAMS,
            n_speakers=1,
            gin_channels=768,
            use_zero_shot=True,
            spk_embed_dim=192,
        )
        for module in model.flow.flows:
            if hasattr(module, 'mean_only'):
                assert module.mean_only is False, (
                    "SynthesizerTrn flow should use mean_only=False"
                )

    @pytest.mark.unit
    def test_flow_dilation_rate_2(self):
        """SynthesizerTrn flow uses dilation_rate=2"""
        model = SynthesizerTrn(
            **MODEL_PARAMS,
            n_speakers=1,
            gin_channels=768,
            use_zero_shot=True,
            spk_embed_dim=192,
        )
        assert model.flow.dilation_rate == 2, (
            f"Flow dilation_rate should be 2, got {model.flow.dilation_rate}"
        )

    @pytest.mark.unit
    def test_flow_forward_reverse_roundtrip(self):
        """Flow forward then reverse should approximately recover the input"""
        torch.manual_seed(42)
        flow = ResidualCouplingBlock(
            channels=192,
            hidden_channels=192,
            kernel_size=5,
            dilation_rate=2,
            n_layers=4,
            n_flows=4,
            gin_channels=768,
        )
        flow.eval()

        batch_size = 1
        seq_len = 10
        x = torch.randn(batch_size, 192, seq_len)
        x_mask = torch.ones(batch_size, 1, seq_len)
        g = torch.randn(batch_size, 768, 1)

        with torch.no_grad():
            z = flow(x, x_mask, g=g, reverse=False)
            x_hat = flow(z, x_mask, g=g, reverse=True)

        assert torch.allclose(x, x_hat, atol=1e-4), (
            f"Flow forward-reverse roundtrip error too large: "
            f"max diff = {(x - x_hat).abs().max().item()}"
        )


class TestDurationPredictorScale:
    """Test DurationPredictor and StochasticDurationPredictor have cond_scale"""

    @pytest.mark.unit
    def test_duration_predictor_has_cond_scale(self):
        """DurationPredictor has cond_scale attribute when gin_channels > 0"""
        dp = DurationPredictor(
            in_channels=192,
            filter_channels=256,
            kernel_size=3,
            p_dropout=0.5,
            gin_channels=768,
        )
        assert hasattr(dp, 'cond_scale'), (
            "DurationPredictor should have cond_scale when gin_channels > 0"
        )
        # cond_scale maps gin_channels -> in_channels
        assert dp.cond_scale.weight.shape == (192, 768, 1), (
            f"cond_scale weight shape should be (192, 768, 1), got {dp.cond_scale.weight.shape}"
        )

    @pytest.mark.unit
    def test_duration_predictor_no_cond_scale_without_gin(self):
        """DurationPredictor without gin_channels has no cond_scale"""
        dp = DurationPredictor(
            in_channels=192,
            filter_channels=256,
            kernel_size=3,
            p_dropout=0.5,
            gin_channels=0,
        )
        assert not hasattr(dp, 'cond_scale'), (
            "DurationPredictor should not have cond_scale when gin_channels=0"
        )

    @pytest.mark.unit
    def test_stochastic_duration_predictor_has_cond_scale(self):
        """StochasticDurationPredictor has cond_scale attribute when gin_channels > 0"""
        sdp = StochasticDurationPredictor(
            in_channels=192,
            filter_channels=192,
            kernel_size=3,
            p_dropout=0.5,
            n_flows=4,
            gin_channels=768,
        )
        assert hasattr(sdp, 'cond_scale'), (
            "StochasticDurationPredictor should have cond_scale when gin_channels > 0"
        )

    @pytest.mark.unit
    def test_stochastic_duration_predictor_no_cond_scale_without_gin(self):
        """StochasticDurationPredictor without gin_channels has no cond_scale"""
        sdp = StochasticDurationPredictor(
            in_channels=192,
            filter_channels=192,
            kernel_size=3,
            p_dropout=0.5,
            n_flows=4,
            gin_channels=0,
        )
        assert not hasattr(sdp, 'cond_scale'), (
            "StochasticDurationPredictor should not have cond_scale when gin_channels=0"
        )

    @pytest.mark.unit
    def test_duration_predictor_scale_forward(self):
        """DurationPredictor forward with g uses cond_scale (multiplicative conditioning)"""
        torch.manual_seed(42)
        dp = DurationPredictor(
            in_channels=192,
            filter_channels=256,
            kernel_size=3,
            p_dropout=0.0,
            gin_channels=768,
        )
        dp.eval()

        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, 192, seq_len)
        x_mask = torch.ones(batch_size, 1, seq_len)
        g = torch.randn(batch_size, 768, 1)

        with torch.no_grad():
            out = dp(x, x_mask, g=g)

        assert out.shape == (batch_size, 1, seq_len)
        assert torch.isfinite(out).all(), "DurationPredictor output should be finite"

    @pytest.mark.unit
    def test_duration_predictor_different_g_different_output(self):
        """DurationPredictor produces different outputs with different speaker conditioning"""
        torch.manual_seed(42)
        dp = DurationPredictor(
            in_channels=192,
            filter_channels=256,
            kernel_size=3,
            p_dropout=0.0,
            gin_channels=768,
        )
        dp.eval()

        x = torch.randn(1, 192, 10)
        x_mask = torch.ones(1, 1, 10)
        g1 = torch.randn(1, 768, 1)
        g2 = torch.randn(1, 768, 1)

        with torch.no_grad():
            out1 = dp(x, x_mask, g=g1)
            out2 = dp(x, x_mask, g=g2)

        assert not torch.allclose(out1, out2, atol=1e-5), (
            "DurationPredictor should produce different outputs with different g "
            "(cond_scale provides multiplicative conditioning)"
        )

    @pytest.mark.unit
    def test_synthesizer_dp_has_cond_scale(self):
        """SynthesizerTrn's duration predictor has cond_scale when gin_channels > 0"""
        model = SynthesizerTrn(
            **MODEL_PARAMS,
            n_speakers=1,
            gin_channels=768,
            use_zero_shot=True,
            spk_embed_dim=192,
        )
        assert hasattr(model.dp, 'cond_scale'), (
            "SynthesizerTrn's duration predictor should have cond_scale"
        )
