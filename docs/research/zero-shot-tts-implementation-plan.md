# Zero-Shot TTS 実装計画書

**作成日**: 2026-03-07
**最終更新**: 2026-03-08
**ブランチ**: `feat/zero-shot-tts`
**実装状態**: M0-M4 完了 (M5 未着手)
**前提ドキュメント**: [zero-shot-tts-research.md](./zero-shot-tts-research.md)

---

## 目次

1. [要件定義](#1-要件定義)
2. [最終アーキテクチャ](#2-最終アーキテクチャ)
3. [Speaker Encoder選定](#3-speaker-encoder選定)
4. [ONNXモデル設計](#4-onnxモデル設計)
5. [コード変更仕様](#5-コード変更仕様)
6. [学習戦略](#6-学習戦略)
7. [学習データ](#7-学習データ)
8. [期待精度](#8-期待精度)
9. [評価パイプライン](#9-評価パイプライン)
10. [既存機能との互換性](#10-既存機能との互換性)
11. [要件充足チェックリスト](#11-要件充足チェックリスト)
12. [実装スケジュール](#12-実装スケジュール)
13. [参考文献](#13-参考文献)

---

## 1. 要件定義

### 1.1 必須要件

| # | 要件 | 優先度 |
|---|------|--------|
| R1 | **推論速度ゼロインパクト**: zero-shot導入後もONNX推論速度が現在と同一であること | 最高 |
| R2 | **軽量性維持**: ONNXモデルサイズの増加を最小限に抑えること | 最高 |
| R3 | **既存機能の完全互換**: speaker-id, prosody, WavLM discriminator等の既存機能が全て動作すること | 最高 |
| R4 | **GPL-free**: 追加する全依存がApache-2.0/MITライセンスであること | 高 |
| R5 | **CPU推論対応**: CUDA_VISIBLE_DEVICES="" での推論が可能であること | 高 |
| R6 | **多言語対応**: 日本語・英語・その他言語の話者zero-shot再現が実用的な品質であること。既存のPhonemizerレジストリによる多言語対応と整合すること | 高 |

### 1.2 品質目標

| 指標 | 目標値 | 測定方法 |
|------|--------|---------|
| SECS (話者類似度) | > 0.65 (cross-encoder) | WeSpeaker ResNet293 |
| WER (英語) | < 8% | Whisper large-v3 |
| CER (日本語) | < 8% | Whisper large-v3 |
| CER/WER (その他言語) | < 10% | Whisper large-v3 |
| UTMOS (自然性) | > 3.5 | UTMOSv2 |

**多言語対応方針**: Speaker Encoderは言語非依存（音声の音響特徴から話者性を抽出）であるため、zero-shot話者再現自体は言語を問わず機能する。Phonemizerレジストリに言語を追加すれば、その言語でもzero-shot合成が可能になる。

### 1.3 推論速度の制約（定量）

現在のPiper推論パイプライン（100文字、~5秒音声、CPU）:

| ステップ | 処理時間 | 割合 |
|---------|---------|------|
| Phonemize | 10-50ms | ~10% |
| TextEncoder | 30-50ms | ~15% |
| Duration Predictor | 10-20ms | ~5% |
| Flow | 20-40ms | ~10% |
| Generator (Decoder) | 80-150ms | ~50% |
| WAV書き出し | 5-10ms | ~3% |
| **合計** | **150-310ms** | RTF 0.03-0.06 |

**zero-shot導入後の許容追加コスト: <1ms**（ファイルI/Oのみ）

---

## 2. 最終アーキテクチャ

### 2.1 全体構成図

```
==============================================================================
 オフライン処理（1回のみ、推論パイプライン外）
==============================================================================

  参照音声 (WAV, 5-10秒)
    |
    v
  [リサンプリング 22050Hz -> 16kHz]
    |
    v
  [Fbank特徴抽出 (80dim, 25ms窓, 10msホップ)]
    |
    v
  [CAM++ Speaker Encoder (ONNX, 28MB)]  ← 別ONNXモデル、推論時不使用
    |
    v
  speaker_embedding (float32, 192dim, L2正規化済み)
    |
    v
  [保存: speaker_embeddings/speaker_name.npy]

==============================================================================
 オンライン推論（毎回実行、現在と同一速度）
==============================================================================

  テキスト入力
    |
    v
  [Phonemize (言語別)]  →  phoneme_ids + prosody_features
    |
    |  speaker_name.npy を読み込み (<1ms)
    |    |
    v    v
  [Piper TTS ONNX モデル]
    |
    |  内部処理:
    |  speaker_embedding (192dim)
    |    → nn.Linear(192, gin_channels) → g [batch, gin_channels, 1]  ← 射影層
    |    → Duration Predictor (g で条件付け)
    |    → Flow (g で条件付け)
    |    → Generator/Decoder (g で条件付け)
    |
    v
  音声出力 (WAV, 22050Hz)
```

### 2.2 設計の核心

**Speaker Encoderを推論パイプラインから完全に分離する。**

- **学習時**: CAM++ Speaker Encoderをモデル内に内蔵し、参照音声からembeddingを動的計算
- **ONNXエクスポート時**: CAM++を除外し、事前計算済みembeddingを直接入力する構造にエクスポート
- **推論時**: `.npy`ファイルから192次元embeddingを読み込み、ONNXモデルに入力

この設計により:
- TTSモデル本体のONNX計算グラフは、Embedding Gather → Linear射影への変更以外は**完全に同一**
- 推論速度への影響は**実質ゼロ**（<1ms）

### 2.3 nn.Embedding vs 外部embedding の速度同一性

```
現在:   sid (int64) → [Gather: emb_g.weight] → g [1, gin_channels, 1]   ← 数μs
変更後: spk_emb (float32, 192) → [Linear: 192→gin_channels] → g [1, gin_channels, 1]  ← 数μs

差分: Linear(192, gin_channels) の演算コスト = 192 * gin_channels の行列乗算 ≈ 数μs
全体推論時間 (150-310ms) に対して 0.001% 未満 → 実質ゼロ
```

Coqui TTS (YourTTS) が `nn.Embedding` (speaker_id方式) と `d_vector` (外部embedding方式) の両方をサポートし、推論速度に有意差がないことを確認済み。

---

## 3. Speaker Encoder選定

### 3.1 CAM++ (推奨)

| 項目 | 値 |
|------|-----|
| モデル | CAM++ (Context-Aware Masking) |
| 出力次元 | 192 |
| パラメータ数 | 7.2M |
| ONNXサイズ | 28MB (INT8量子化: ~8MB) |
| EER (VoxCeleb1-O) | 0.73% |
| CPU推論 (5秒音声) | ~30-40ms |
| ライセンス | Apache-2.0 |
| 入力仕様 | 80-dim Fbank @ 16kHz |
| 入手先 | WeSpeaker / ModelScope / sherpa-onnx |

**選定理由**:
1. EER 0.73% — ECAPA-TDNN (1.0%) より高精度
2. 7.2Mパラメータ — ECAPA-TDNN (6.2M) と同等の軽量さ
3. Apache-2.0 — GPL-free方針と完全適合
4. CosyVoiceでの実績 — Alibabaが標準Speaker Encoderとして採用
5. ONNX対応済み — WeSpeaker/sherpa-onnxで事前学習済みモデル提供

**事前学習済みモデル**:
- `iic/speech_campplus_sv_zh-cn_16k-common` (ModelScope, 200K話者で学習)
- WeSpeaker: `wespeaker/wespeaker-models` (HuggingFace)

### 3.2 次元射影の設計

CAM++出力（192次元）とPiperの`gin_channels`（768）が不一致のため、線形射影層を追加:

```python
# models.py SynthesizerTrn.__init__
self.spk_proj = nn.Linear(spk_embed_dim, gin_channels)  # 192 → 768

# 使用時
g = self.spk_proj(speaker_embedding).unsqueeze(-1)  # [B, 768, 1]
```

**射影層のパラメータ数**: 192 * 768 + 768 = 148,224 (~0.15M) → 無視できるサイズ

### 3.3 Embedding正規化

業界標準のL2正規化 + 平均化:

```python
# 単一参照音声
emb = cam_plus(audio)
emb = emb / np.linalg.norm(emb)  # L2正規化

# 複数参照音声の平均
embs = [cam_plus(audio_i) / np.linalg.norm(cam_plus(audio_i)) for audio_i in audios]
emb_avg = np.mean(embs, axis=0)
emb_avg = emb_avg / np.linalg.norm(emb_avg)  # 再正規化
```

---

## 4. ONNXモデル設計

### 4.1 入出力仕様

**現在のONNX入力**:

| 入力名 | 型 | 形状 | 説明 |
|--------|-----|------|------|
| `input` | int64 | `[B, T_phoneme]` | Phoneme IDs |
| `input_lengths` | int64 | `[B]` | 音素列長 |
| `scales` | float32 | `[3]` | noise_scale, length_scale, noise_scale_w |
| `sid` | int64 | `[B]` | Speaker ID (マルチスピーカー時) |
| `prosody_features` | int64 | `[B, T_phoneme, 3]` | A1/A2/A3 (prosody有効時) |

**zero-shot対応後のONNX入力**:

| 入力名 | 型 | 形状 | 説明 |
|--------|-----|------|------|
| `input` | int64 | `[B, T_phoneme]` | Phoneme IDs |
| `input_lengths` | int64 | `[B]` | 音素列長 |
| `scales` | float32 | `[3]` | noise_scale, length_scale, noise_scale_w |
| **`speaker_embedding`** | **float32** | **`[B, 192]`** | **事前計算済みSpeaker Embedding** |
| `prosody_features` | int64 | `[B, T_phoneme, 3]` | A1/A2/A3 (prosody有効時) |

**変更点**: `sid` (int64, scalar) → `speaker_embedding` (float32, 192dim)

### 4.2 ONNXグラフ内の処理

```
speaker_embedding [B, 192]
    |
    v
[Linear: spk_proj (192 → gin_channels)]    ← ONNXグラフに含まれる（数μs）
    |
    v
[Unsqueeze(-1)]
    |
    v
g [B, gin_channels, 1]
    |
    +---> Duration Predictor (cond)
    +---> Flow (coupling layers)
    +---> Generator/Decoder (cond)
```

### 4.3 モデルサイズの影響

| コンポーネント | パラメータ数 | ONNXサイズ増加 |
|--------------|------------|---------------|
| spk_proj (192→768) | ~0.15M | ~0.6MB |
| emb_g (20話者×768) 削除 | -0.015M | -0.06MB |
| **純増** | **~0.13M** | **~0.5MB** |

現在のONNXモデル (~74MB) に対して **+0.7%** の増加。無視できるレベル。

---

## 5. コード変更仕様

### 5.1 変更ファイル一覧

| ファイル | 変更種別 | 変更規模 | 内容 |
|---------|---------|---------|------|
| `vits/models.py` | 修正 | ~40行 | SynthesizerTrn: spk_proj追加、dual-mode対応 |
| `vits/lightning.py` | 修正 | ~80行 | CAM++ロード、SCL/DINO損失計算 |
| `vits/losses.py` | 修正 | ~30行 | speaker_consistency_loss, dino_loss追加 |
| `vits/config.py` | 修正 | ~5行 | spk_embed_dim, use_zero_shot フィールド追加 |
| `vits/dataset.py` | 修正 | ~20行 | speaker_embedding読み込み対応 |
| `export_onnx.py` | 修正 | ~30行 | speaker_embedding入力対応 |
| `infer_onnx.py` | 修正 | ~40行 | --speaker-embedding オプション追加 |
| `__main__.py` | 修正 | ~15行 | --zero-shot CLI引数追加 |
| **`extract_speaker_embedding.py`** | **新規** | **~280行** | **オフラインembedding抽出ツール** |
| `src/python/tests/test_zero_shot.py` | 新規 | ~360行 | M1: dual-modeテスト (15テスト) |
| `src/python/tests/test_m2_training_pipeline.py` | 新規 | ~180行 | M2: SCL/DINO/Datasetテスト (12テスト) |
| `src/python/tests/test_m3_inference_pipeline.py` | 新規 | ~410行 | M3: ONNX export/inferテスト (5テスト) |
| `src/python/tests/test_m4_extract_speaker_embedding.py` | 新規 | ~170行 | M4: 抽出ツールテスト (9テスト) |

### 5.2 models.py — SynthesizerTrn

#### `__init__` (現在: 行731-828)

```python
# 変更: __init__ のシグネチャにパラメータ追加
def __init__(self, ..., prosody_dim=0,
             use_zero_shot=False, spk_embed_dim=192):  # NEW
    ...
    self.use_zero_shot = use_zero_shot

    # 変更: speaker embedding の初期化
    if use_zero_shot:
        # zero-shot mode: 外部embedding → gin_channels への射影
        self.spk_proj = nn.Linear(spk_embed_dim, gin_channels)
    elif n_speakers > 1:
        # 従来mode: discrete speaker ID → embedding lookup
        self.emb_g = nn.Embedding(n_speakers, gin_channels)
```

#### `forward` (現在: 行868-932)

```python
def forward(self, x, x_lengths, y, y_lengths, sid=None,
            prosody_features=None,
            speaker_embedding=None):  # NEW
    ...
    # 変更: g ベクトルの生成
    if self.use_zero_shot:
        if speaker_embedding is None:
            raise ValueError("speaker_embedding is required when use_zero_shot=True")
        g = self.spk_proj(speaker_embedding).unsqueeze(-1)  # [B, gin_channels, 1]
    elif self.n_speakers > 1:
        assert sid is not None, "Missing speaker id"
        g = self.emb_g(sid).unsqueeze(-1)
    else:
        g = None
    # 以降は変更なし — g の形状 [B, gin_channels, 1] が共通
```

#### `infer` (現在: 行934-983)

`forward` と同様の変更を適用。

### 5.3 lightning.py — 学習ループ

> **実装状態 (2026-03-08)**: PyTorch版CAM++の統合は保留中。現在のlightning.pyには
> `c_spk`, `c_dino`, `speaker_encoder_path`, `freeze_speaker_encoder_steps` パラメータ、
> `dino_center` バッファ、および `training_step_g` でのspeaker_embedding受け渡しが実装済み。
> SCL/DINO損失関数は `losses.py` に定義済みだが、training_step_g への統合は
> PyTorch Speaker Encoder (または `onnx2torch` 変換) 完了後に実施予定。

#### CAM++ Speaker Encoder の読み込み

```python
class VitsModel(pl.LightningModule):
    def __init__(self, ..., use_zero_shot=False, spk_embed_dim=192,
                 c_spk=9.0, c_dino=0.1,
                 freeze_speaker_encoder_steps=100000):
        ...
        if use_zero_shot:
            from .speaker_encoder import CamPlusSpeakerEncoder
            self.speaker_encoder = CamPlusSpeakerEncoder(
                pretrained="iic/speech_campplus_sv_zh-cn_16k-common",
                output_dim=spk_embed_dim,
            )
            # Phase 1: Encoder凍結
            for param in self.speaker_encoder.parameters():
                param.requires_grad = False
```

#### Speaker Consistency Loss の計算

```python
def training_step_g(self, batch):
    ...
    if self.hparams.use_zero_shot:
        # 参照音声（同一話者の別発話 or 同一発話）からembedding抽出
        ref_embedding = self.speaker_encoder(ref_audio)  # [B, 192]

        # TTS合成音声からembedding抽出
        gen_embedding = self.speaker_encoder(y_hat.squeeze(1))  # [B, 192]

        # Speaker Consistency Loss
        loss_scl = speaker_consistency_loss(gen_embedding, ref_embedding)
        loss_gen_all += self.hparams.c_spk * loss_scl
```

#### DINO Loss の計算

```python
        # DINO Loss (self-supervised speaker discriminability)
        if self.hparams.c_dino > 0:
            loss_dino = dino_loss(
                student_emb=gen_embedding,
                teacher_emb=ref_embedding.detach(),
                center=self.dino_center,
                tau_s=0.1, tau_t=self.current_tau_t,
            )
            loss_gen_all += self.hparams.c_dino * loss_dino
            # Update center (EMA)
            with torch.no_grad():
                self.dino_center = 0.996 * self.dino_center + 0.004 * ref_embedding.mean(0)
```

#### Phase切り替え（Encoder凍結解除）

```python
def on_train_batch_start(self, batch, batch_idx):
    if self.hparams.use_zero_shot:
        if self.global_step == self.hparams.freeze_speaker_encoder_steps:
            _LOGGER.info("Unfreezing speaker encoder at step %d", self.global_step)
            for param in self.speaker_encoder.parameters():
                param.requires_grad = True
            # Encoder用の低学習率を設定
            # (configure_optimizersでparam groupsを分けておく)
```

### 5.4 losses.py — 損失関数追加

```python
def speaker_consistency_loss(gen_embedding, ref_embedding):
    """Speaker Consistency Loss (SCL).
    1 - cosine_similarity(gen, ref)
    """
    return 1.0 - F.cosine_similarity(gen_embedding, ref_embedding, dim=-1).mean()


def dino_loss(student_emb, teacher_emb, center, tau_s=0.1, tau_t=0.04):
    """DINO self-supervised loss for speaker encoder."""
    student_out = F.log_softmax(student_emb / tau_s, dim=-1)
    teacher_out = F.softmax((teacher_emb - center) / tau_t, dim=-1)
    return -(teacher_out * student_out).sum(dim=-1).mean()
```

### 5.5 export_onnx.py — ONNXエクスポート

```python
# infer_forward 関数内
def infer_forward(text, text_lengths, scales, sid=None,
                  prosody_features=None, speaker_embedding=None):
    ...
    if use_zero_shot and speaker_embedding is not None:
        g = model_g.spk_proj(speaker_embedding).unsqueeze(-1)
    elif model_g.n_speakers > 1 and sid is not None:
        g = model_g.emb_g(sid).unsqueeze(-1)
    else:
        g = None
    ...

# ONNX入力名
if use_zero_shot:
    input_names = ["input", "input_lengths", "scales", "speaker_embedding"]
    dynamic_axes["speaker_embedding"] = {0: "batch_size"}
    # dummy input
    dummy_spk = torch.randn(1, zero_shot_embed_dim, dtype=torch.float32)
elif num_speakers > 1:
    input_names = ["input", "input_lengths", "scales", "sid"]
    dynamic_axes["sid"] = {0: "batch_size"}
```

### 5.6 infer_onnx.py — 推論スクリプト

```python
# 新CLI引数
parser.add_argument("--speaker-embedding",
    help="事前計算済みSpeaker Embedding (.npy)")

# 入力構築
if "speaker_embedding" in input_names and args.speaker_embedding:
    spk_emb = np.load(args.speaker_embedding).astype(np.float32)
    inputs["speaker_embedding"] = spk_emb.reshape(1, -1)
elif "sid" in input_names and args.speaker_id is not None:
    inputs["sid"] = np.array([args.speaker_id], dtype=np.int64)
```

### 5.7 extract_speaker_embedding.py — 新規ツール

```python
"""オフラインSpeaker Embedding抽出ツール.

使用例:
  # 単一音声ファイル
  uv run python -m piper_train.extract_speaker_embedding \
    --encoder campplus.onnx \
    --audio reference.wav \
    --output speaker.npy

  # ディレクトリ内の全WAVファイルを平均化
  uv run python -m piper_train.extract_speaker_embedding \
    --encoder campplus.onnx \
    --audio-dir /path/to/speaker_wavs/ \
    --output speaker.npy

  # データセットの全話者を一括抽出
  uv run python -m piper_train.extract_speaker_embedding \
    --encoder campplus.onnx \
    --dataset-dir /data/piper/dataset-moe-speech-20speakers-v2 \
    --output-dir /path/to/embeddings/
"""
```

処理フロー:
1. WAVファイルを16kHzにリサンプリング
2. 80次元Fbank特徴を抽出（25ms窓、10msホップ）
3. CAM++ ONNXモデルで192次元embeddingを計算
4. L2正規化
5. 複数ファイルの場合は平均化して再正規化
6. `.npy`ファイルとして保存

---

## 6. 学習戦略

### 6.1 2フェーズ学習

現在のPiperの学習ではWavLM Discriminatorは使用していない（実験中に音割れ問題が確認されており、v2モデルはWavLMなしで学習済み）。そのため、zero-shot学習もWavLMなしの2フェーズ構成とする。

| Phase | Iterations | CAM++ | 学習率 (VITS) | 学習率 (CAM++) | batch/GPU |
|-------|-----------|-------|-------------|-------------|-----------|
| **Phase 1** | ~100K | 凍結 (最終層除く) | 2e-4 | 0 (最終層: 2e-5) | 14-16 |
| **Phase 2** | ~200K | 全解凍 | 2e-4 → 5e-5 | 2e-5 → 5e-7 (lr_ratio=0.1) | 10-12 |

**オプション: WavLM Discriminator追加 (Phase 3)**

WavLM Discriminatorの音割れ問題が解決した場合、Phase 2完了後にWavLM有効化のPhase 3を追加できる:

| Phase | Iterations | 学習率 | WavLM Disc. | batch/GPU |
|-------|-----------|--------|-------------|-----------|
| Phase 3 (オプション) | ~50K | 5e-5 | 有効 (c_wavlm=0.2) | 8-10 |

### 6.2 損失関数の構成

| # | 損失関数 | 重み | Phase 1 | Phase 2 |
|---|---------|------|---------|---------|
| 1 | VITS損失 (recon + KL + adv) | 既存 | 有効 | 有効 |
| 2 | Speaker Consistency Loss | c_spk=9.0 | 有効 | 有効 |
| 3 | DINO Loss | c_dino=0.1 | 有効 | 有効 |

WavLM Perceptual Loss (c_wavlm) はオプションのPhase 3でのみ使用。現時点では学習計画に含めない。

**Speaker Consistency Loss (SCL)**:
```
L_SCL = 1 - cosine_similarity(CAM++(合成音声), CAM++(参照音声))
```
YourTTSで提案。c_spk=9.0はYourTTS論文の推奨値。

**DINO Loss**:
- DINO-VITS (Interspeech 2024) で提案
- Student (学習中のCAM++) と Teacher (EMA) の自己蒸留
- ノイズの多いデータでも話者分離性を維持
- 温度: tau_s=0.1, tau_t=0.04→0.07 (30epochでウォームアップ)

### 6.3 GPUメモリ見積もり (L4 16GB)

| Phase | コンポーネント | メモリ/GPU |
|-------|-------------|-----------|
| Phase 1 | VITS (~30M) + CAM++ 凍結 (7.2M) + SCL/DINO | ~10-11GB |
| Phase 2 | VITS + CAM++ 解凍 + SCL/DINO | ~12-14GB |
| Phase 3 (オプション) | VITS + CAM++ + WavLM Disc. (~95M) + SCL/DINO | ~15-16GB |

WavLMなしの構成では、Phase 1-2ともにメモリに十分な余裕がある。

### 6.4 推定学習時間 (L4 x4)

| Phase | Iterations | 推定時間 |
|-------|-----------|---------|
| Phase 1 | 100K | ~60-80h |
| Phase 2 | 200K | ~130-170h |
| **合計** | **300K** | **~190-250h** |
| Phase 3 (オプション) | 50K | ~40-50h |

### 6.5 学習コマンド

```bash
# Phase 1: CAM++凍結
NCCL_DEBUG=WARN NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
uv run python -m piper_train \
  --dataset-dir /data/piper/dataset-zero-shot-merged \
  --prosody-dim 16 \
  --zero-shot --spk-embed-dim 192 \
  --c-spk 9.0 --c-dino 0.1 \
  --freeze-speaker-encoder-steps 100000 \
  --accelerator gpu --devices 4 --precision 16-mixed \
  --max_epochs 200 --batch-size 14 --samples-per-speaker 2 \
  --checkpoint-epochs 1 --quality medium \
  --base_lr 2e-4 --disable_auto_lr_scaling \
  --ema-decay 0.9995 --num-workers 0 --no-pin-memory \
  --default_root_dir /data/piper/output-zero-shot-phase1

# Phase 2: CAM++解凍、リジューム
NCCL_DEBUG=WARN NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
uv run python -m piper_train \
  --dataset-dir /data/piper/dataset-zero-shot-merged \
  --prosody-dim 16 \
  --zero-shot --spk-embed-dim 192 \
  --c-spk 9.0 --c-dino 0.1 \
  --accelerator gpu --devices 4 --precision 16-mixed \
  --max_epochs 400 --batch-size 10 --samples-per-speaker 2 \
  --checkpoint-epochs 1 --quality medium \
  --base_lr 2e-4 --disable_auto_lr_scaling \
  --ema-decay 0.9995 --num-workers 0 --no-pin-memory \
  --default_root_dir /data/piper/output-zero-shot-phase2 \
  --resume_from_checkpoint /data/piper/output-zero-shot-phase1/lightning_logs/.../last.ckpt
```

---

## 7. 学習データ

### 7.1 データ構成

#### コア言語（日本語・英語）

| コーパス | 言語 | 話者数 | 時間 | サンプリングレート | ライセンス |
|----------|------|--------|------|------------------|-----------|
| LibriTTS-R | 英語 | 2,456 | 585h | 24kHz → 22050Hz | CC-BY-4.0 |
| JVS | 日本語 | 100 | 30h | 24kHz → 22050Hz | CC-BY-SA-4.0 |
| moe-speech-20speakers-v2 | 日本語 | 20 | ~100h | 22050Hz | 独自 |
| **コア合計** | **日英** | **~2,576** | **~715h** | **22050Hz** | - |

#### 多言語拡張候補

Speaker Encoder (CAM++) は言語非依存のため、Phonemizerが対応する言語であればデータを追加するだけでzero-shot対応が可能。以下は拡張候補コーパス:

| コーパス | 言語 | 話者数 | 時間 | ライセンス | 備考 |
|----------|------|--------|------|-----------|------|
| Common Voice 16.0 | 多言語 (100+) | 多数 | 数千h | CC-0 | 品質ばらつき大、選別が必要 |
| CSS10 | 10言語 | 各1話者 | 各10-20h | CC-0 | 単一話者のため話者多様性なし |
| Multilingual LibriSpeech (MLS) | 8言語 | 数千 | 50K+h | CC-BY-4.0 | 大規模、品質良好 |
| AISHELL-3 | 中国語 | 218 | 85h | Apache-2.0 | 高品質、マルチスピーカー |
| M-AILABS | 9言語 | 複数 | ~1000h | public domain相当 | 品質良好 |

**拡張優先度**: Phonemizerレジストリに登録済みの言語（現在: `ja`, `en`）から順に対応。新言語追加時は `Phonemizer` ABCを実装して `registry.py` に登録するだけでzero-shot合成が可能。

#### 多言語対応のアーキテクチャ上の利点

1. **Speaker Encoderは言語非依存**: CAM++は音響特徴から話者性を抽出するため、日本語話者の参照音声で抽出したembeddingを使って英語を合成することも可能（cross-lingual voice cloning）
2. **Phonemizerレジストリとの自然な統合**: zero-shot機能は `speaker_embedding` → `g` ベクトルの経路のみ変更。テキスト処理側（Phonemizer）は完全に独立しているため、新言語の追加はPhonemizerの実装のみで完結
3. **混合学習による相乗効果**: YourTTSの実験で、英語+他言語の混合学習がzero-shot性能を向上させることが確認されている

### 7.2 データ準備手順

1. **サンプリングレート統一**: 全コーパスを22050Hzにリサンプリング
2. **phoneme_id_map統合**: 各言語のPhonemizerが個別管理するため、言語タグ付きで統合（衝突なし）
3. **Speaker ID割り当て**: moe-speech: 0-19, JVS: 20-119, LibriTTS-R: 120-2575, (拡張言語: 2576+)
4. **Speaker Embedding事前計算**: CAM++で全話者のembeddingを抽出し`speaker_embeddings/`に保存
5. **JSONL統合**: 全コーパスのdataset.jsonlを統合、`speaker_embedding` + `language`フィールドを追加

### 7.3 統合JSONLフォーマット

```json
{
  "phoneme_ids": [1, 8, 5, 39, ...],
  "speaker_id": 0,
  "speaker_embedding_path": "speaker_embeddings/speaker_0.npy",
  "prosody_features": [{"a1": -2, "a2": 1, "a3": 5}, ...],
  "language": "ja"
}
```

`language` フィールドは推論時のPhonemizerレジストリ選択に使用。学習時はphoneme_idsが事前計算済みのため不要だが、メタデータとして保持する。

```json
// 英語の例
{
  "phoneme_ids": [1, 45, 12, ...],
  "speaker_id": 150,
  "speaker_embedding_path": "speaker_embeddings/libritts_speaker_150.npy",
  "prosody_features": [{"a1": 0, "a2": 1, "a3": 3}, ...],
  "language": "en"
}
```

### 7.4 Data Augmentation (学習時)

| 手法 | 効果 | 適用対象 |
|------|------|---------|
| MUSAN noise injection | ノイズ耐性向上 | 参照音声（Encoder入力） |
| Room Impulse Response (RIR) | 残響耐性 | 参照音声 |
| Speed perturbation (0.9x/1.1x) | データ量3倍 | 全音声 |

---

## 8. 期待精度

### 8.1 VITSベースzero-shotの到達可能精度

| 構成 | SECS (same-encoder) | SIM-O (cross-encoder) | MOS |
|------|--------------------|-----------------------|-----|
| Piper現行 (speaker_id) | ~0.90+ (学習済み話者のみ) | N/A (zero-shot非対応) | ~4.0 |
| + CAM++ 凍結 (Phase 1) | ~0.83-0.87 | ~0.58-0.65 | ~3.7-3.9 |
| **+ CAM++ 解凍 + DINO + SCL (Phase 2)** | **~0.87-0.92** | **~0.65-0.73** | **~3.8-4.1** |
| + WavLM (オプション Phase 3) | ~0.88-0.92 | ~0.65-0.75 | ~4.0-4.2 |
| VITSの理論上限 | ~0.92 | ~0.75 | ~4.2 |

**参考: 他モデルの実測値**:

| モデル | SIM-O | アーキテクチャ |
|--------|-------|---------------|
| YourTTS (VITS) | ~0.65-0.72 | VITS + d-vector |
| CosyVoice 3 | 0.774 | LLM + Flow Matching |
| MaskGCT | 0.687-0.777 | Masked Codec Transformer |

### 8.2 根拠

- **YourTTS** (VITS + d-vector, 1151話者): SECS ~0.864 (same-encoder), LibriSpeech WER ~7-8%
- **DINO-VITS** (VITS + CAM++ + DINO): YourTTSからSIM +0.03-0.05改善（特にノイズ条件）
- **SCL追加**: YourTTS論文で+0.01-0.03のSIM改善（条件による）
- **CAM++がd-vectorより高精度**: EER 0.73% vs ~7-10%、embedding品質の向上がSIMに直結

### 8.3 VITSの構造的制約

VITSアーキテクチャには以下の制約があり、SIM-O ~0.75がおおよその上限:

1. Speaker Embeddingとprosodyが結合 → 未知話者への汎化が弱い
2. GANベースの学習の不安定性 → 大規模スケーリングが困難
3. Flow/Diffusionに比べて表現力の天井が低い

将来的にSIM-O 0.75以上を目指す場合はアーキテクチャの刷新が必要（CosyVoice方式等）。

---

## 9. 評価パイプライン

### 9.1 循環バイアスの回避

学習に使用するSpeaker Encoder (CAM++) と評価に使用するEncoderを分離する。同一Encoderで評価すると、Encoderに最適化された音声が高スコアを得てしまい、実際の知覚類似度を反映しない。

| 用途 | Encoder | 理由 |
|------|---------|------|
| 学習 (SCL/DINO) | CAM++ (7.2M) | 軽量、学習効率 |
| **評価 (SECS)** | **WeSpeaker ResNet293** (大規模) | **異アーキテクチャで公平** |
| 参考値 | CAM++ (学習用と同一) | バイアスあり、参考のみ |

**WeSpeaker ResNet293**: EER 0.447%、VoxCeleb2 (5994話者) で学習。HuggingFace: `Wespeaker/wespeaker-voxceleb-resnet293-LM`

### 9.2 評価指標一覧

| 指標 | ツール | 測定内容 | 目標値 |
|------|--------|---------|--------|
| SECS | WeSpeaker ResNet293 | 話者類似度 (cross-encoder) | > 0.65 |
| SECS-CAM | CAM++ | 話者類似度 (参考、バイアスあり) | > 0.85 |
| WER | Whisper large-v3 | 英語明瞭度 | < 8% |
| CER | Whisper large-v3 | 日本語明瞭度 | < 8% |
| UTMOS | UTMOSv2 / SpeechMOS | 自然性 (自動MOS推定) | > 3.5 |

### 9.3 評価手順

```bash
# 1. テスト話者のembeddingを抽出（未見話者、各5-10秒）
uv run python -m piper_train.extract_speaker_embedding \
  --encoder campplus.onnx \
  --audio test_speaker_ref.wav \
  --output test_speaker.npy

# 2. zero-shot合成
CUDA_VISIBLE_DEVICES="" uv run python -m piper_train.infer_onnx \
  --model model-zero-shot.onnx \
  --config config.json \
  --output-dir eval_output/ \
  --text "評価用テキスト" \
  --speaker-embedding test_speaker.npy

# 3. SECS計算（WeSpeaker ResNet293でcross-encoder評価）
# 4. WER/CER計算（Whisper large-v3）
# 5. UTMOS計算
```

---

## 10. 既存機能との互換性

### 10.1 互換性マトリクス

| 既存機能 | 互換性 | 影響 | 備考 |
|---------|--------|------|------|
| `--speaker-id` による話者指定 | **非互換** (zero-shotモデル) | speaker_embedding入力に変更 | 既存emb_gモデルは別途維持可能 |
| 日本語 Phonemizer | 影響なし | 独立モジュール | |
| 英語 Phonemizer (g2p-en) | 影響なし | 独立モジュール | |
| Prosody Features (A1/A2/A3) | **完全互換** | Duration Predictorへの入力は変更なし | |
| ONNX推論 | **互換** | 入力名がsid→speaker_embeddingに変更 | |
| CPU推論 | **完全互換** | 推論速度ゼロインパクト | |
| WavLM Discriminator | **完全互換** | 現在は未使用。将来有効化する場合も学習時のみで独立 | |
| EMA重み | **完全互換** | export_onnxの処理は変更なし | |
| `--noise-scale` | **完全互換** | stochastic sampling変更なし | |
| 疑問詞マーカー拡張 (Issue #204) | **影響なし** | Phonemizer内部処理、独立 | |
| 文脈依存N variants (Issue #207) | **影響なし** | Phonemizer内部処理、独立 | |
| SpeakerBalancedBatchSampler | **互換** | zero-shot学習でも話者バランスに有効 | |
| FP16 Mixed Precision | **完全互換** | 学習・推論ともに変更なし | |

### 10.2 既存speaker_idモデルからの移行

既存の20話者モデル (`moe-speech-20speakers-v2.onnx`) は引き続き `--speaker-id` で利用可能。zero-shotモデルは**別モデル**として並行して提供する。

既存話者のembeddingは、学習済みチェックポイントの `emb_g.weight` から抽出可能:

```python
# 既存モデルからembedding抽出
ckpt = torch.load("last.ckpt")
emb_weights = ckpt["state_dict"]["model_g.emb_g.weight"]  # [20, 768]
for i in range(20):
    np.save(f"speaker_{i}.npy", emb_weights[i].numpy())
```

### 10.3 推論コマンド比較

```bash
# 既存: speaker-idモード（変更なし）
uv run python -m piper_train.infer_onnx \
  --model moe-speech-20speakers-v2.onnx \
  --config config.json \
  --text "こんにちは" --speaker-id 0

# 新規: zero-shotモード
uv run python -m piper_train.infer_onnx \
  --model model-zero-shot.onnx \
  --config config.json \
  --text "こんにちは" --speaker-embedding speaker.npy
```

---

## 11. 要件充足チェックリスト

| # | 要件 | 充足 | 根拠 |
|---|------|------|------|
| R1 | 推論速度ゼロインパクト | **充足** | Speaker Encoderは推論パイプライン外。ONNXグラフ内の変更はGather→Linear(192→768)のみで数μs。推論時間への影響は0.001%未満 |
| R2 | 軽量性維持 | **充足** | ONNXモデルサイズ増加は~0.5MB (+0.7%)。Speaker Encoder (28MB) は別ファイルで推論時不要 |
| R3 | 既存機能の完全互換 | **充足** | Prosody, WavLM Disc., CPU推論, EMA等は全て影響なし。既存speaker-idモデルは別途維持 |
| R4 | GPL-free | **充足** | CAM++ (Apache-2.0), WeSpeaker (Apache-2.0), SpeechBrain (Apache-2.0) |
| R5 | CPU推論対応 | **充足** | ONNX Runtimeで推論。Speaker Encoder不要。推論速度は現在と同一 |
| R6 | 多言語対応 | **充足** | Speaker Encoder (CAM++) は言語非依存。Phonemizerレジストリに言語を追加するだけでzero-shot対応可能。日本語(JVS+moe-speech: 120話者)、英語(LibriTTS-R: 2456話者)をコアとし、追加言語はMLS等のコーパスで拡張可能。Cross-lingual voice cloning（日本語参照→英語合成等）も原理的に対応 |

---

## 12. 実装スケジュール

### 12.1 フェーズ分割

| Phase | 内容 | 工数 (実装) | 工数 (学習) |
|-------|------|-----------|-----------|
| **Impl-1** | models.py / config.py 変更 + 単体テスト | 1日 | - |
| **Impl-2** | lightning.py 変更 (CAM++統合, SCL/DINO損失) | 2日 | - |
| **Impl-3** | export_onnx.py / infer_onnx.py 変更 | 1日 | - |
| **Impl-4** | extract_speaker_embedding.py 新規作成 | 1日 | - |
| **Impl-5** | データ準備 (LibriTTS-R + JVS統合) | 2日 | - |
| **Train-1** | Phase 1学習 (CAM++凍結) | - | ~60-80h |
| **Train-2** | Phase 2学習 (CAM++解凍 + DINO + SCL) | - | ~130-170h |
| **Eval** | 評価パイプライン実行 + 結果分析 | 1日 | - |
| **合計** | | **~8日** | **~190-250h** |

#### 実績 (2026-03-08)

| Phase | 内容 | 実績 |
|-------|------|------|
| **Impl-1** (M1) | models.py / config.py 変更 + 単体テスト | ✅ 完了 (15テスト) |
| **Impl-2** (M2) | lightning.py / losses.py / dataset.py / __main__.py | ✅ 完了 (12テスト、SCL/DINO統合は保留) |
| **Impl-3** (M3) | export_onnx.py / infer_onnx.py 変更 | ✅ 完了 (5テスト) |
| **Impl-4** (M4) | extract_speaker_embedding.py 新規作成 | ✅ 完了 (9テスト) |
| **Impl-5** (M5) | データ準備スクリプト | 未着手 |

### 12.2 前提条件

- 現在の20話者WavLM学習 (200epoch) が完了していること
- L4 GPU x4 が利用可能であること
- LibriTTS-R / JVS コーパスがダウンロード済みであること
- CAM++ ONNXモデルがダウンロード済みであること

### 12.3 成果物

| 成果物 | 形式 | 説明 |
|--------|------|------|
| zero-shotモデル | `.onnx` | speaker_embedding入力対応のTTSモデル |
| CAM++ Encoder | `.onnx` (28MB) | オフラインembedding抽出用（別配布） |
| 抽出ツール | Python script | `extract_speaker_embedding.py` |
| 既存20話者embedding | `.npy` x20 | 既存話者のembeddingファイル |
| 評価結果レポート | Markdown | SECS, WER/CER, UTMOS の計測結果 |

---

## 13. 参考文献

### 論文

- YourTTS (ICML 2022): https://arxiv.org/abs/2112.02418
- DINO-VITS (Interspeech 2024): https://arxiv.org/abs/2311.09770
- CAM++ (Interspeech 2023): https://arxiv.org/abs/2303.00332
- VITS: https://arxiv.org/abs/2106.06103
- ECAPA-TDNN: https://arxiv.org/abs/2005.07143
- MaskGCT (ICLR 2025): https://arxiv.org/abs/2409.00750
- CosyVoice: https://arxiv.org/abs/2407.05407
- LibriTTS-R: https://arxiv.org/abs/2305.18802
- JVS Corpus: https://arxiv.org/abs/1908.06248

### OSS実装

- WeSpeaker: https://github.com/wenet-e2e/wespeaker (Apache-2.0)
- 3D-Speaker / CAM++: https://github.com/modelscope/3D-Speaker (Apache-2.0)
- sherpa-onnx: https://github.com/k2-fsa/sherpa-onnx (Apache-2.0)
- Coqui TTS / YourTTS: https://github.com/coqui-ai/TTS
- CosyVoice: https://github.com/FunAudioLLM/CosyVoice (Apache-2.0)
- SpeechBrain: https://github.com/speechbrain/speechbrain (Apache-2.0)

### 評価ツール

- WeSpeaker ResNet293: https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet293-LM
- UTMOSv2: https://github.com/sarulab-speech/UTMOSv2
- SpeechMOS: https://github.com/tarepan/SpeechMOS
- Whisper large-v3: https://huggingface.co/openai/whisper-large-v3
