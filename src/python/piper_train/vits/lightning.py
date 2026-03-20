import logging
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torchaudio
from torch import autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from .commons import slice_segments
from .dataset import Batch, PiperDataset, SpeakerBalancedBatchSampler, UtteranceCollate
from .losses import (
    dino_loss,
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
    speaker_consistency_loss,
)
from .mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from .models import MultiPeriodDiscriminator, SynthesizerTrn, WavLMDiscriminator


_LOGGER = logging.getLogger("vits.lightning")


class CamPPSpeakerEncoder:
    """Lightweight wrapper around CAM++ ONNX model for speaker embedding extraction.

    Runs on CPU via ONNX Runtime (no GPU memory overhead). Not an nn.Module
    because ONNX is non-differentiable and should not participate in
    state_dict / checkpoint saving.

    Pipeline: waveform (22050 Hz) -> resample (16000 Hz) -> 80-dim Fbank -> CMVN -> CAM++ -> L2 norm
    """

    def __init__(self, onnx_path: str, source_sr: int = 22050, target_sr: int = 16000):
        import onnxruntime  # noqa: PLC0415

        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        )
        sess_options.inter_op_num_threads = 2
        sess_options.intra_op_num_threads = 2

        # Always run on CPU to avoid competing with training for GPU memory
        self._session = onnxruntime.InferenceSession(
            onnx_path,
            sess_options,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self.source_sr = source_sr
        self.target_sr = target_sr
        self._resampler = (
            torchaudio.transforms.Resample(source_sr, target_sr)
            if source_sr != target_sr
            else None
        )
        _LOGGER.info(
            "CamPPSpeakerEncoder loaded: %s (CPU, %d->%d Hz)",
            onnx_path,
            source_sr,
            target_sr,
        )

    @torch.no_grad()
    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract speaker embeddings from a batch of waveforms.

        Parameters
        ----------
        audio : torch.Tensor
            Waveform tensor of shape ``[B, T]`` at ``source_sr`` Hz.

        Returns
        -------
        torch.Tensor
            Speaker embeddings ``[B, 192]``, L2-normalised, on the same device
            as the input (transferred back after CPU-side ONNX inference).
        """
        device = audio.device
        audio_cpu = audio.detach().float().cpu()

        embeddings = []
        for i in range(audio_cpu.size(0)):
            wav = audio_cpu[i]  # [T]
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)  # [1, T]

            # Resample to 16 kHz
            if self._resampler is not None:
                wav = self._resampler(wav)

            # 80-dim Fbank (Kaldi-compatible)
            fbank = torchaudio.compliance.kaldi.fbank(
                wav,
                num_mel_bins=80,
                frame_length=25.0,
                frame_shift=10.0,
                sample_frequency=self.target_sr,
            )  # [T_frames, 80]

            # CMVN normalisation (mean-only, matching CAM++ / 3D-Speaker)
            fbank = fbank - fbank.mean(dim=0, keepdim=True)

            # ONNX inference: [1, T_frames, 80] -> [1, 192]
            fbank_np = fbank.unsqueeze(0).numpy().astype(np.float32)
            emb = self._session.run(None, {self._input_name: fbank_np})[0].squeeze()

            # L2 normalisation
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm

            embeddings.append(torch.from_numpy(emb))

        return torch.stack(embeddings).to(device)  # [B, 192]


# Memory cleanup frequency (iterations)
MEMORY_CLEANUP_FREQUENCY = 500


class VitsModel(pl.LightningModule):
    def __init__(
        self,
        num_symbols: int,
        num_speakers: int,
        # audio
        resblock="2",
        resblock_kernel_sizes=(3, 5, 7),
        resblock_dilation_sizes=(
            (1, 2),
            (2, 6),
            (3, 12),
        ),
        upsample_rates=(8, 8, 4),
        upsample_initial_channel=256,
        upsample_kernel_sizes=(16, 16, 8),
        # mel
        filter_length: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        mel_channels: int = 80,
        sample_rate: int = 22050,
        sample_bytes: int = 2,
        channels: int = 1,
        mel_fmin: float = 0.0,
        mel_fmax: float | None = None,
        # model
        inter_channels: int = 192,
        hidden_channels: int = 192,
        filter_channels: int = 768,
        n_heads: int = 2,
        n_layers: int = 6,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
        n_layers_q: int = 3,
        use_spectral_norm: bool = False,
        gin_channels: int = 0,
        use_sdp: bool = True,
        segment_size: int = 8192,
        prosody_dim: int = 16,
        # training
        dataset: list[str | Path] | None = None,
        learning_rate: float = 2e-4,
        betas: tuple[float, float] = (0.8, 0.99),
        eps: float = 1e-9,
        batch_size: int = 1,
        lr_decay: float = 0.999875,
        init_lr_ratio: float = 1.0,
        warmup_epochs: int = 0,
        c_mel: int = 45,
        c_kl: float = 1.0,
        grad_clip: float | None = None,
        num_workers: int = 1,
        seed: int = 1234,
        num_test_examples: int = 5,
        validation_split: float = 0.1,
        max_phoneme_ids: int | None = None,
        # Zero-shot TTS (enabled by default for multi-speaker models)
        use_zero_shot: bool = True,
        spk_embed_dim: int = 192,
        c_spk: float = 1.0,
        c_dino: float = 0.5,
        speaker_encoder_path: str | None = None,
        freeze_speaker_encoder_steps: int = 100000,
        # Speaker embedding dropout for dual-mode training
        spk_emb_dropout: float = 0.5,
        # KL annealing: linearly increase KL weight from 0.1 to c_kl over this many epochs
        kl_annealing_epochs: int = 10,
        # WavLM Discriminator (enabled by default for improved audio quality)
        use_wavlm_discriminator: bool = True,
        wavlm_model_name: str = "microsoft/wavlm-base-plus",
        c_wavlm: float = 0.5,
        # Training loop optimization
        d_update_interval: int = 1,
        max_spec_length: int = 700,
        **kwargs,
    ):
        super().__init__()
        self.automatic_optimization = (
            False  # Multiple optimizers require manual optimization
        )

        # Fix gin_channels BEFORE save_hyperparameters() so the correct value is saved
        if (use_zero_shot or num_speakers > 1) and (gin_channels <= 0):
            gin_channels = 512

        self.save_hyperparameters()

        # Discriminator update interval (D:G = 1:d_update_interval)
        self.d_update_interval = self.hparams.get("d_update_interval", 1)

        # DINO center buffer and EMA teacher for zero-shot training
        if use_zero_shot:
            self.register_buffer("dino_center", torch.zeros(gin_channels))
            # EMA copy of spk_proj used as DINO teacher.
            # Initialized after model_g is created (see below).

        # Set up models
        self.model_g = SynthesizerTrn(
            n_vocab=self.hparams.num_symbols,
            spec_channels=self.hparams.filter_length // 2 + 1,
            segment_size=self.hparams.segment_size // self.hparams.hop_length,
            inter_channels=self.hparams.inter_channels,
            hidden_channels=self.hparams.hidden_channels,
            filter_channels=self.hparams.filter_channels,
            n_heads=self.hparams.n_heads,
            n_layers=self.hparams.n_layers,
            kernel_size=self.hparams.kernel_size,
            p_dropout=self.hparams.p_dropout,
            resblock=self.hparams.resblock,
            resblock_kernel_sizes=self.hparams.resblock_kernel_sizes,
            resblock_dilation_sizes=self.hparams.resblock_dilation_sizes,
            upsample_rates=self.hparams.upsample_rates,
            upsample_initial_channel=self.hparams.upsample_initial_channel,
            upsample_kernel_sizes=self.hparams.upsample_kernel_sizes,
            n_speakers=self.hparams.num_speakers,
            gin_channels=self.hparams.gin_channels,
            use_sdp=self.hparams.use_sdp,
            prosody_dim=self.hparams.prosody_dim,
            use_zero_shot=self.hparams.use_zero_shot,
            spk_embed_dim=self.hparams.spk_embed_dim,
        )
        # Initialize DINO EMA teacher from spk_proj (must be after model_g creation)
        if use_zero_shot and hasattr(self.model_g, "spk_proj"):
            import copy  # noqa: PLC0415

            self.spk_proj_teacher = copy.deepcopy(self.model_g.spk_proj)
            self.spk_proj_teacher.requires_grad_(False)

        self.model_d = MultiPeriodDiscriminator(
            use_spectral_norm=self.hparams.use_spectral_norm
        )

        # WavLM Discriminator (optional)
        self.model_d_wavlm = None
        if self.hparams.use_wavlm_discriminator:
            _LOGGER.info(
                f"Initializing WavLM Discriminator with model: {self.hparams.wavlm_model_name}"
            )
            self.model_d_wavlm = WavLMDiscriminator(
                model_name=self.hparams.wavlm_model_name,
                source_sample_rate=self.hparams.sample_rate,
            )

        # CAM++ Speaker Encoder for SCL (optional, CPU-only ONNX, not an nn.Module)
        self.speaker_encoder: CamPPSpeakerEncoder | None = None
        if use_zero_shot and speaker_encoder_path is not None:
            encoder_path = Path(speaker_encoder_path)
            if encoder_path.exists():
                self.speaker_encoder = CamPPSpeakerEncoder(
                    str(encoder_path),
                    source_sr=sample_rate,
                )
            else:
                _LOGGER.warning(
                    "speaker_encoder_path not found, SCL disabled: %s", encoder_path
                )

        # Dataset splits
        self._train_dataset: Dataset | None = None
        self._val_dataset: Dataset | None = None
        self._test_dataset: Dataset | None = None
        self._load_datasets(
            validation_split,
            num_test_examples,
            max_phoneme_ids,
            max_spec_length,
        )

        # State kept between training optimizers
        self._y = None
        self._y_hat = None

        # Track whether torch.compile has been applied
        self._compiled = False

    def on_train_start(self):
        """Apply torch.compile after checkpoint restoration.

        Called after setup() and checkpoint loading, so state_dict keys
        won't have _orig_mod. prefix conflicts during checkpoint restore.
        """
        if self._compiled:
            return

        # --no-compile flag disables torch.compile entirely
        if self.hparams.get("no_compile", False):
            _LOGGER.info("torch.compile disabled by --no-compile flag (eager mode)")
            self._compiled = True
            return

        if not hasattr(torch, "compile"):
            _LOGGER.info(
                "torch.compile not available (PyTorch < 2.0), using eager mode"
            )
            self._compiled = True
            return

        # Compile Generator decoder (HiFi-GAN) — the most compute-intensive part
        try:
            self.model_g.dec = torch.compile(self.model_g.dec, mode="default")
            _LOGGER.info("torch.compile applied to Generator decoder (mode=default)")
        except Exception as e:
            _LOGGER.warning(
                "torch.compile failed for Generator decoder, using eager mode: %s", e
            )

        # Compile MultiPeriodDiscriminator
        try:
            self.model_d = torch.compile(self.model_d, mode="default")
            _LOGGER.info(
                "torch.compile applied to MultiPeriodDiscriminator (mode=default)"
            )
        except Exception as e:
            _LOGGER.warning(
                "torch.compile failed for MultiPeriodDiscriminator, "
                "using eager mode: %s",
                e,
            )

        self._compiled = True

    def _load_datasets(
        self,
        validation_split: float,
        num_test_examples: int,
        max_phoneme_ids: int | None = None,
        max_spec_length: int | None = None,
    ):
        if self.hparams.dataset is None:
            _LOGGER.debug("No dataset to load")
            return

        full_dataset = PiperDataset(
            self.hparams.dataset,
            max_phoneme_ids=max_phoneme_ids,
            max_spec_length=max_spec_length,
            filter_length=self.hparams.filter_length,
        )

        valid_set_size = int(len(full_dataset) * validation_split)
        train_set_size = len(full_dataset) - valid_set_size - num_test_examples

        self._train_dataset, self._test_dataset, self._val_dataset = random_split(
            full_dataset, [train_set_size, num_test_examples, valid_set_size]
        )

    def forward(
        self,
        text,
        text_lengths,
        scales,
        sid=None,
        prosody_features=None,
        speaker_embedding=None,
    ):
        noise_scale = scales[0]
        length_scale = scales[1]
        noise_scale_w = scales[2]
        audio, *_ = self.model_g.infer(
            text,
            text_lengths,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_scale_w=noise_scale_w,
            sid=sid,
            prosody_features=prosody_features,
            speaker_embedding=speaker_embedding,
        )

        return audio

    def on_train_epoch_start(self):
        """エポック開始時にSpeakerBalancedBatchSamplerのepochを更新"""
        if (
            hasattr(self, "_train_batch_sampler")
            and self._train_batch_sampler is not None
        ):
            self._train_batch_sampler.set_epoch(self.current_epoch)
            _LOGGER.debug(
                "Set SpeakerBalancedBatchSampler epoch to %d", self.current_epoch
            )

    def on_train_epoch_end(self):
        """Epoch終了時にLR schedulerをステップする。

        automatic_optimization=False のため、ExponentialLR は自動でステップ
        されない。手動でepoch単位のdecayを適用する。
        """
        schedulers = self.lr_schedulers()
        if schedulers is not None:
            if isinstance(schedulers, list | tuple):
                for sch in schedulers:
                    sch.step()
            else:
                schedulers.step()

            # Log current learning rates
            opt_g, opt_d = self.optimizers()
            _LOGGER.info(
                "Epoch %d LR stepped: G=%.2e, D=%.2e",
                self.current_epoch,
                opt_g.param_groups[0]["lr"],
                opt_d.param_groups[0]["lr"],
            )

    def train_dataloader(self):
        # Check if pin_memory should be disabled (for memory-constrained multi-GPU setups)
        pin_memory = not getattr(self.hparams, "no_pin_memory", False)

        collate_fn = UtteranceCollate(
            is_multispeaker=self.hparams.num_speakers > 1 or self.hparams.use_zero_shot,
            segment_size=self.hparams.segment_size,
        )

        # マルチスピーカーでsamples_per_speakerが設定されている場合は
        # SpeakerBalancedBatchSamplerを使用
        samples_per_speaker = getattr(self.hparams, "samples_per_speaker", 0)
        if self.hparams.num_speakers > 1 and samples_per_speaker > 0:
            self._train_batch_sampler = SpeakerBalancedBatchSampler(
                self._train_dataset,
                batch_size=self.hparams.batch_size,
                samples_per_speaker=samples_per_speaker,
                drop_last=True,
            )
            _LOGGER.info(
                "Using SpeakerBalancedBatchSampler: batch_size=%d, samples_per_speaker=%d, "
                "speakers_per_batch=%d",
                self.hparams.batch_size,
                samples_per_speaker,
                self.hparams.batch_size // samples_per_speaker,
            )
            return DataLoader(
                self._train_dataset,
                collate_fn=collate_fn,
                batch_sampler=self._train_batch_sampler,
                num_workers=self.hparams.num_workers,
                pin_memory=pin_memory,
                persistent_workers=(True if self.hparams.num_workers > 0 else False),
                prefetch_factor=(2 if self.hparams.num_workers > 0 else None),
            )
        else:
            # 従来の動作（ランダムサンプリング）
            self._train_batch_sampler = None
            return DataLoader(
                self._train_dataset,
                collate_fn=collate_fn,
                num_workers=self.hparams.num_workers,
                batch_size=self.hparams.batch_size,
                pin_memory=pin_memory,
                persistent_workers=(
                    True if self.hparams.num_workers > 0 else False
                ),  # Multi-GPU optimization
                prefetch_factor=(2 if self.hparams.num_workers > 0 else None),
            )

    def val_dataloader(self):
        # Check if pin_memory should be disabled (for memory-constrained multi-GPU setups)
        pin_memory = not getattr(self.hparams, "no_pin_memory", False)
        # Validation uses fewer workers than training to reduce memory pressure.
        # With DDP (4 GPUs), train workers are already persistent, so val workers
        # add significant memory overhead. Cap at 2 workers per process.
        val_workers = min(self.hparams.num_workers, 2)
        return DataLoader(
            self._val_dataset,
            collate_fn=UtteranceCollate(
                is_multispeaker=self.hparams.num_speakers > 1
                or self.hparams.use_zero_shot,
                segment_size=self.hparams.segment_size,
            ),
            num_workers=val_workers,
            batch_size=self.hparams.batch_size,
            pin_memory=pin_memory,
            persistent_workers=False,  # Don't keep val workers alive between epochs
            prefetch_factor=(2 if val_workers > 0 else None),
        )

    def test_dataloader(self):
        return DataLoader(
            self._test_dataset,
            collate_fn=UtteranceCollate(
                is_multispeaker=self.hparams.num_speakers > 1
                or self.hparams.use_zero_shot,
                segment_size=self.hparams.segment_size,
            ),
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
        )

    def training_step(self, batch: Batch, batch_idx: int):
        # Manual optimization for multiple optimizers
        opt_g, opt_d = self.optimizers()
        grad_clip = self.hparams.grad_clip

        # Train generator (every step)
        opt_g.zero_grad()
        loss_g = self.training_step_g(batch)
        self.manual_backward(loss_g)
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model_g.parameters(), grad_clip)
        opt_g.step()

        # Train discriminator (every d_update_interval steps)
        if self.global_step % self.d_update_interval == 0:
            opt_d.zero_grad()
            loss_d = self.training_step_d(batch)
            self.manual_backward(loss_d)
            if grad_clip is not None:
                d_params = list(self.model_d.parameters())
                if self.model_d_wavlm is not None:
                    d_params = d_params + list(self.model_d_wavlm.parameters())
                torch.nn.utils.clip_grad_norm_(d_params, grad_clip)
            opt_d.step()
        else:
            _LOGGER.debug(
                "Skipping D update at global_step=%d (interval=%d)",
                self.global_step,
                self.d_update_interval,
            )

        # Periodic memory cleanup (infrequent — GPU utilization is stable)
        if batch_idx % MEMORY_CLEANUP_FREQUENCY == 0:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                if batch_idx == 0:
                    _LOGGER.info(
                        "D:G ratio = 1:%d | Memory cache clearing every %d iterations",
                        self.d_update_interval,
                        MEMORY_CLEANUP_FREQUENCY,
                    )
                else:
                    _LOGGER.debug("Memory cache cleared at iteration %d", batch_idx)

    def _log_with_batch_info(
        self, key: str, value, batch: Batch = None, batch_size: int = None
    ):
        """Helper method to log with proper batch_size and sync_dist settings."""
        if batch_size is None:
            if batch is not None:
                batch_size = batch.phoneme_ids.size(0)
            else:
                batch_size = self._y.size(0) if hasattr(self, "_y") else None

        sync_dist = self.trainer.world_size > 1
        self.log(key, value, batch_size=batch_size, sync_dist=sync_dist)

    def training_step_g(self, batch: Batch):
        x, x_lengths, y, _, spec, spec_lengths, speaker_ids, prosody_features = (
            batch.phoneme_ids,
            batch.phoneme_lengths,
            batch.audios,
            batch.audio_lengths,
            batch.spectrograms,
            batch.spectrogram_lengths,
            batch.speaker_ids if batch.speaker_ids is not None else None,
            batch.prosody_features if batch.prosody_features is not None else None,
        )
        speaker_embeddings = (
            batch.speaker_embeddings if batch.speaker_embeddings is not None else None
        )

        # Dual-mode training: deterministically drop speaker embeddings to train emb_g.
        # Uses global_step-based hash instead of torch.rand() to ensure all DDP ranks
        # make the same dropout decision (prevents NCCL allreduce deadlock).
        if (
            self.training
            and speaker_embeddings is not None
            and speaker_ids is not None
            and self.hparams.spk_emb_dropout > 0
        ):
            # Deterministic pseudo-random: hash(step) gives same value on all ranks
            drop = ((self.global_step * 2654435761) & 0xFFFFFFFF) / 0xFFFFFFFF
            if drop < self.hparams.spk_emb_dropout:
                speaker_embeddings = None

        # Speaker embedding perturbation for zero-shot generalization
        if self.training and speaker_embeddings is not None:
            speaker_embeddings = (
                speaker_embeddings + torch.randn_like(speaker_embeddings) * 0.02
            )

        (
            y_hat,
            l_length,
            _attn,
            ids_slice,
            _x_mask,
            z_mask,
            (_z, z_p, m_p, logs_p, _m_q, logs_q),
        ) = self.model_g(
            x,
            x_lengths,
            spec,
            spec_lengths,
            speaker_ids,
            prosody_features=prosody_features,
            speaker_embedding=speaker_embeddings,
        )
        self._y_hat = y_hat.contiguous()

        mel = spec_to_mel_torch(
            spec,
            self.hparams.filter_length,
            self.hparams.mel_channels,
            self.hparams.sample_rate,
            self.hparams.mel_fmin,
            self.hparams.mel_fmax,
        )
        y_mel = slice_segments(
            mel,
            ids_slice,
            self.hparams.segment_size // self.hparams.hop_length,
        )
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1),
            self.hparams.filter_length,
            self.hparams.mel_channels,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.mel_fmin,
            self.hparams.mel_fmax,
        )
        y = slice_segments(
            y,
            ids_slice * self.hparams.hop_length,
            self.hparams.segment_size,
        )  # slice

        # Ensure contiguous memory layout to prevent fragmentation
        y = y.contiguous()
        y_hat = y_hat.contiguous()

        # Save for training_step_d
        self._y = y

        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.model_d(y, y_hat)

        with autocast(self.device.type, enabled=False):
            # KL annealing: linearly increase from 0.1 to c_kl over kl_annealing_epochs
            if (
                self.hparams.kl_annealing_epochs > 0
                and self.current_epoch < self.hparams.kl_annealing_epochs
            ):
                kl_weight = self.hparams.c_kl * (
                    0.1 + 0.9 * self.current_epoch / self.hparams.kl_annealing_epochs
                )
            else:
                kl_weight = self.hparams.c_kl

            # Generator loss
            loss_dur = torch.sum(l_length.float())
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * self.hparams.c_mel
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * kl_weight

            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, _losses_gen = generator_loss(y_d_hat_g)

            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl

            # WavLM Discriminator loss (optional)
            if self.model_d_wavlm is not None:
                y_d_hat_r_wlm, y_d_hat_g_wlm, fmap_r_wlm, fmap_g_wlm = (
                    self.model_d_wavlm(y, y_hat)
                )

                loss_fm_wavlm = feature_loss(fmap_r_wlm, fmap_g_wlm)
                loss_gen_wavlm, _ = generator_loss(y_d_hat_g_wlm)
                loss_wavlm = (loss_gen_wavlm + loss_fm_wavlm) * self.hparams.c_wavlm
                loss_gen_all = loss_gen_all + loss_wavlm

                # Log WavLM losses
                self._log_with_batch_info("loss_gen_wavlm", loss_gen_wavlm, batch)
                self._log_with_batch_info("loss_fm_wavlm", loss_fm_wavlm, batch)

            # --- Speaker Consistency Loss (SCL) ---
            # Uses CAM++ ONNX (CPU, non-differentiable) to extract speaker
            # embeddings from generated audio and compare them against the
            # reference embeddings via cosine similarity.
            #
            # Because the ONNX encoder is non-differentiable, gen_embedding
            # is detached from the computation graph. The loss value still
            # contributes to the total generator loss and provides a useful
            # training signal: when SCL is high the generator is penalised,
            # encouraging it to preserve speaker identity through other
            # differentiable paths (mel reconstruction, KL, etc.).
            if (
                self.hparams.c_spk > 0
                and speaker_embeddings is not None
                and self.speaker_encoder is not None
            ):
                with torch.no_grad():
                    gen_embedding = self.speaker_encoder(y_hat.squeeze(1))
                loss_spk = (
                    speaker_consistency_loss(gen_embedding, speaker_embeddings)
                    * self.hparams.c_spk
                )
                loss_gen_all = loss_gen_all + loss_spk
                self._log_with_batch_info("loss_spk", loss_spk, batch)

            # --- DINO Self-Distillation Loss ---
            # Student uses current spk_proj; teacher uses an EMA copy
            # (spk_proj_teacher) that evolves slowly, providing stable targets.
            if (
                self.hparams.c_dino > 0
                and speaker_embeddings is not None
                and hasattr(self, "spk_proj_teacher")
            ):
                student_emb = self.model_g.spk_proj(speaker_embeddings)
                with torch.no_grad():
                    teacher_emb = self.spk_proj_teacher(speaker_embeddings)
                loss_dino = (
                    dino_loss(student_emb, teacher_emb, self.dino_center)
                    * self.hparams.c_dino
                )
                loss_gen_all = loss_gen_all + loss_dino
                self._log_with_batch_info("loss_dino", loss_dino, batch)

                # Update DINO teacher EMA and center only during training
                # (NaN root cause #5: validation_step calls training_step_g,
                #  which was updating center during validation)
                if self.training:
                    with torch.no_grad():
                        for p_ema, p in zip(
                            self.spk_proj_teacher.parameters(),
                            self.model_g.spk_proj.parameters(),
                            strict=True,
                        ):
                            p_ema.mul_(0.996).add_(p.data, alpha=0.004)

                    # Update DINO center with EMA
                    with torch.no_grad():
                        batch_center = teacher_emb.mean(dim=0)
                        if torch.distributed.is_initialized():
                            torch.distributed.all_reduce(
                                batch_center, op=torch.distributed.ReduceOp.AVG
                            )
                        self.dino_center.mul_(0.996).add_(batch_center, alpha=0.004)
                        self.dino_center.clamp_(min=-10, max=10)

            self._log_with_batch_info("loss_gen_all", loss_gen_all, batch)
            self._log_with_batch_info("kl_weight", kl_weight, batch)

            return loss_gen_all

    def training_step_d(self, batch: Batch):
        # From training_step_g
        y = self._y
        y_hat = self._y_hat
        # Ensure detached tensors are contiguous
        y_hat_detached = y_hat.detach().contiguous()

        # Full discriminator forward with detached fake audio.
        # NOTE: We must use fresh (non-detached) real-side outputs here so that
        # gradients flow back through D for the real-side LSGAN loss term
        # mean((1 - D(real))^2). Using detached/cached real-side outputs would
        # zero out these gradients and prevent D from learning.
        y_d_hat_r, y_d_hat_g, _, _ = self.model_d(y, y_hat_detached)

        with autocast(self.device.type, enabled=False):
            # Discriminator
            loss_disc, _losses_disc_r, _losses_disc_g = discriminator_loss(
                y_d_hat_r, y_d_hat_g
            )
            loss_disc_all = loss_disc

            # WavLM Discriminator loss (optional)
            if self.model_d_wavlm is not None:
                y_d_hat_r_wlm, y_d_hat_g_wlm, _, _ = self.model_d_wavlm(
                    y, y_hat_detached
                )
                loss_disc_wavlm, _, _ = discriminator_loss(y_d_hat_r_wlm, y_d_hat_g_wlm)
                loss_disc_all = loss_disc_all + loss_disc_wavlm * self.hparams.c_wavlm

                # Log WavLM discriminator loss
                self._log_with_batch_info("loss_disc_wavlm", loss_disc_wavlm, batch)

            self._log_with_batch_info("loss_disc_all", loss_disc_all, batch)

            return loss_disc_all

    def validation_step(self, batch: Batch, batch_idx: int):
        val_loss = self.training_step_g(batch) + self.training_step_d(batch)
        self._log_with_batch_info("val_loss", val_loss, batch)

        # Audio generation is done in on_validation_epoch_end (outside DDP forward)
        # to avoid NCCL sync issues between ranks.

        return val_loss

    def on_validation_epoch_end(self):
        """Generate audio examples on rank 0 only (outside DDP forward)."""
        if self.global_rank != 0:
            return
        for utt_idx, test_utt in enumerate(self._test_dataset):
            text = test_utt.phoneme_ids.unsqueeze(0).to(self.device)
            text_lengths = torch.LongTensor([len(test_utt.phoneme_ids)]).to(self.device)
            sid = (
                test_utt.speaker_id.to(self.device)
                if test_utt.speaker_id is not None
                else None
            )
            # Use real speaker embedding from test utterance instead of zeros
            spk_emb = None
            if self.hparams.use_zero_shot:
                if test_utt.speaker_embedding is not None:
                    spk_emb = test_utt.speaker_embedding.unsqueeze(0).to(self.device)
                else:
                    spk_emb = torch.zeros(
                        1, self.hparams.spk_embed_dim, device=self.device
                    )
            with torch.no_grad():
                test_audio, *_ = self.model_g.infer(
                    text,
                    text_lengths,
                    sid=sid,
                    noise_scale=0.4,
                    length_scale=1.0,
                    noise_scale_w=0.5,
                    speaker_embedding=spk_emb,
                )
            # Log raw amplitude to detect posterior collapse early
            max_amp = abs(test_audio.max())
            self.log("val_max_amplitude", max_amp, prog_bar=False)
            if max_amp < 0.1:
                _LOGGER.warning(
                    "Low audio amplitude (%.4f) — possible posterior collapse", max_amp
                )
            tag = test_utt.text or str(utt_idx)
            self.logger.experiment.add_audio(
                tag, test_audio, sample_rate=self.hparams.sample_rate
            )

    def configure_optimizers(self):
        # Collect discriminator parameters (including WavLM if enabled)
        d_params = list(self.model_d.parameters())
        if self.model_d_wavlm is not None:
            d_params = d_params + list(self.model_d_wavlm.parameters())

        optimizers = [
            torch.optim.AdamW(
                self.model_g.parameters(),
                lr=self.hparams.learning_rate,
                betas=self.hparams.betas,
                eps=self.hparams.eps,
                fused=torch.cuda.is_available(),
            ),
            torch.optim.AdamW(
                d_params,
                lr=self.hparams.learning_rate,
                betas=self.hparams.betas,
                eps=self.hparams.eps,
                fused=torch.cuda.is_available(),
            ),
        ]
        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                optimizers[0], gamma=self.hparams.lr_decay
            ),
            torch.optim.lr_scheduler.ExponentialLR(
                optimizers[1], gamma=self.hparams.lr_decay
            ),
        ]

        return optimizers, schedulers

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("VitsModel")
        parser.add_argument("--batch-size", type=int, required=True)
        parser.add_argument("--validation-split", type=float, default=0.1)
        parser.add_argument("--num-test-examples", type=int, default=5)
        parser.add_argument(
            "--max-phoneme-ids",
            type=int,
            help="Exclude utterances with phoneme id lists longer than this",
        )
        parser.add_argument("--hidden-channels", type=int, default=192)
        parser.add_argument("--inter-channels", type=int, default=192)
        parser.add_argument("--filter-channels", type=int, default=768)
        parser.add_argument("--n-layers", type=int, default=6)
        parser.add_argument("--n-heads", type=int, default=2)
        parser.add_argument(
            "--gin-channels",
            type=int,
            default=0,
            help="Speaker embedding size for multi-speaker models (default: 0 for single, 768 for multi)",
        )
        parser.add_argument(
            "--prosody-dim",
            type=int,
            default=16,
            help="Dimension for prosody feature projection (A1/A2/A3). Default: 16 (enabled)",
        )
        parser.add_argument(
            "--num-workers",
            type=int,
            default=4,
            help="Number of workers for DataLoader (default: 4). "
            "With multi-GPU DDP, total workers = num_workers × num_GPUs. "
            "For 4-GPU setups, use 2-4 to avoid CPU RAM OOM.",
        )
        return parent_parser
