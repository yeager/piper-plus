# Zero-Shot 話者再現 TTS 技術調査レポート

**調査日**: 2026-03-07 (最終更新: 2026-03-10)
**目的**: Piper TTS (VITSアーキテクチャ) にzero-shot話者再現機能を導入するための技術調査

---

## 目次

1. [エグゼクティブサマリー](#1-エグゼクティブサマリー)
2. [Zero-shot TTSアーキテクチャ概観](#2-zero-shot-ttsアーキテクチャ概観)
3. [Speaker Encoder手法の比較](#3-speaker-encoder手法の比較)
4. [VITSベースのzero-shot拡張手法](#4-vitsベースのzero-shot拡張手法)
5. [OSS実装の実用性・日本語対応状況](#5-oss実装の実用性日本語対応状況)
6. [Piperへの推奨アプローチ](#6-piperへの推奨アプローチ)
7. [高精度アプローチの定量評価](#7-高精度アプローチの定量評価)
8. [既存機能との互換性設計](#8-既存機能との互換性設計)
9. [学習戦略・データ要件](#9-学習戦略データ要件)
10. [具体的な実装計画](#10-具体的な実装計画)
11. [推論速度ゼロインパクト設計（追加調査）](#11-推論速度ゼロインパクト設計追加調査)
12. [エグゼクティブサマリー更新](#12-エグゼクティブサマリー更新)

---

## 1. エグゼクティブサマリー

### 結論（セクション11-12で更新済み）

> **注**: 本セクションは初期調査時の結論です。推論速度ゼロインパクトの制約を加えた**最終推奨はセクション12**を参照してください。Speaker Encoderの推奨もECAPA-TDNN → **CAM++** に変更されています。

Piperにzero-shot話者再現を導入するための最終推奨アプローチは **事前計算Embedding方式 + CAM++ Speaker Encoder (Apache-2.0) + Dual-Mode Speaker Conditioning (emb_g + spk_proj)** である。Speaker Encoderを推論パイプラインから完全に分離することで、推論速度ゼロインパクトを実現する。

- **推論時**: 事前計算済みembedding (.npy) をファイルから読み込むだけ（追加コスト <1ms）
- Speaker Encoderには **CAM++** (Apache-2.0, EER 0.73%, 7.2Mパラメータ, 27MB ONNX) を採用
- **gin_channels=512**（768はガビガビ音が発生したため不採用）
- **アーキテクチャ**: `emb_g` (nn.Embedding) + `spk_proj` (nn.Linear(192, 512)) のDual-Mode
- **学習環境**: RTX 6000 Ada 48GB x1、batch_size=160、samples_per_speaker=8、bf16-mixed、200 epochs
- **学習データ**: 20話者データセット (`dataset-moe-speech-20speakers`) + per-utterance embedding
- 具体的な実装計画は [zero-shot-tts-implementation-plan.md](./zero-shot-tts-implementation-plan.md) を参照

### VITSの品質上限と将来展望

定量評価による調査の結果、VITSアーキテクチャのzero-shot話者類似度の上限は **SIM-O ~0.75** である。最新のLLM+Flow Matching系モデル (Qwen3-TTS: SIM 0.789, CosyVoice 3: SIM 77.4%, MaskGCT: SIM-O 0.687) との差は約0.05-0.10ポイント存在する。将来的にさらなる品質向上を目指す場合は、アーキテクチャの刷新（セクション7で詳述）を検討する。

### 推奨ロードマップ

| フェーズ | 内容 | 工数 | 状態 |
|---------|------|------|------|
| Phase 1 | CAM++ Speaker Encoder + Dual-Mode Speaker Conditioning | 実装~6日 + 学習200 epochs | **実装・学習完了** (2026-03-10) |
| Phase 2 | TextEncoder Speaker Conditioning追加 (VITS2方式) | 1-2日 + 再学習 | 未着手 |
| Phase 3 | MB-iSTFT-VITS デコーダ高速化 (オプション) | 3-5日 + 再学習 | 未着手 |
| 将来 | アーキテクチャ刷新 (Flow Matching等) | 大規模 |

---

## 2. Zero-shot TTSアーキテクチャ概観

### 2.1 アーキテクチャ分類

Zero-shot TTSは大きく4カテゴリに分類される。

#### A. Neural Codec Language Model系
- **代表**: VALL-E/VALL-E 2, CLaM-TTS, Bark
- 音声を離散トークンに変換し、LMで自己回帰的に生成
- 大規模データ（60K〜100K時間）が必要
- 品質は高いが推論が比較的遅い

#### B. GPT + Vocoder/VITS ハイブリッド系
- **代表**: XTTS, GPT-SoVITS, Fish Speech, CosyVoice
- GPT/LLMでsemantic token予測 → vocoder/VITSで波形生成
- Few-shot対応が容易、日本語対応が進んでいる
- **Piperとの親和性が最も高い**（VITSデコーダを共有可能）

#### C. Diffusion/Flow Matching系
- **代表**: VoiceBox, NaturalSpeech 2/3, F5-TTS
- 連続空間でのdiffusion/flow matchingによる高品質生成
- 非自己回帰で高速
- 実装複雑度が高い

#### D. Style Transfer + GAN系
- **代表**: StyleTTS 2, Mega-TTS 2, OpenVoice
- スタイル/音色を分離して転写
- データ効率が良い（250倍少ないデータでVALL-E並み）
- Reference encoderベースで実装が比較的シンプル

### 2.2 主要モデル比較表

| モデル | 開発元 | アーキテクチャ | 話者再現の仕組み |
|--------|--------|---------------|-----------------|
| VALL-E / VALL-E 2 | Microsoft | Neural Codec LM (AR + NAR) | In-context learning (3秒プロンプト) |
| XTTS v2 | Coqui | GPT + VITSデコーダ | Speaker latents (self-attention) |
| StyleTTS 2 | Columbia Univ. | Style Diffusion + Adversarial | Reference encoder (3秒) |
| VoiceBox | Meta | Non-AR Flow Matching | 音声コンテキスト infilling |
| NaturalSpeech 2/3 | Microsoft | Latent Diffusion + Neural Codec | Speech prompting |
| Mega-TTS 2 | Zhejiang Univ. | Acoustic AE + Prosody LLM | Multi-reference timbre encoder |
| CLaM-TTS | KRAFTON | Mel-VAE + Probabilistic RVQ | Audio prompt conditioning |
| Bark | Suno | GPT-style Transformer (~1.5B) | Speaker prompt |
| GPT-SoVITS | RVC-Boss | GPT + SoVITS (改良VITS) | Reference audio (5秒) |
| Fish Speech v1.5 | Fish Audio | Dual-AR + GFSQ | Reference audio prompt |
| OpenVoice v2 | MyShell/MIT | Tone Color Converter | Tone Color特徴抽出 |
| F5-TTS | SWivid | Non-AR Flow Matching + DiT | フィラートークンパディング |
| CosyVoice 1/2/3 | Alibaba | LLM + Conditional Flow Matching | Semantic tokens + 参照音声 |
| Qwen3-TTS | Alibaba Qwen | Dual-track LM + Dual Tokenizer | Semantic tokens + 参照音声 |
| MegaTTS 3 | ByteDance | Latent Diffusion Transformer | WavVAE latent + prosody LM |
| Kokoro | hexgrad | StyleTTS2 + ISTFTNet | Style encoder (3秒) |
| OpenAudio S1 | Fish Audio | Dual-AR + FireflyCodec | Reference audio prompt |
| Chatterbox | Resemble AI | Flow Transformer | Speaker conditioning (5秒) |

### 2.3 性能・品質比較

| モデル | リファレンス音声長 | 報告MOS/品質 | RTF | ライセンス | OSS |
|--------|-------------------|-------------|-----|-----------|-----|
| VALL-E 2 | 3秒 | Human parity | 遅い | 非公開 | 非公式のみ |
| XTTS v2 | 6-30秒 | 16言語SOTA | ~0.25 | CPML | Yes |
| StyleTTS 2 | 3秒 | 人間超え (CMOS +1.07) | 高速 | MIT | Yes |
| VoiceBox | 数秒 | WER 1.9% | VALL-Eの20倍 | 非公開 | No |
| NaturalSpeech 2/3 | 3-10秒 | Ground truth同等 | 不明 | 非公開 | 部分的 |
| GPT-SoVITS | 5秒 | 高品質 | ~0.5 (CPU) | MIT | Yes |
| Fish Speech v1.5 | 数秒 | 高品質 | 高速 | Apache/CC-BY-NC-SA | Yes |
| F5-TTS | 数秒 | WER 2.42 | 0.03-0.15 | MIT/CC-BY-NC | Yes |
| CosyVoice 3 | 数秒 | MOS 5.53 | 低レイテンシ | Apache-2.0 | Yes |
| OpenVoice v2 | 数秒 | 良好 | 高速 | MIT | Yes |
| Bark | プリセット | 良好 | ~リアルタイム | MIT | Yes |
| Qwen3-TTS | 数秒 | WER 1.24, SIM 0.789 | 97msレイテンシ | Apache-2.0 | Yes |
| MegaTTS 3 | 数秒 | MOS 4.28-4.32 | 高速 (45Mパラメータ) | Apache-2.0 | Yes |
| Kokoro | 数秒 | 高品質 | 高速 | Apache-2.0 | Yes |
| OpenAudio S1 | 数秒 | TTS-Arena2 1位 | 高速 | CC-BY-NC-SA | Yes |
| Chatterbox | 5秒 | ElevenLabs超え | 200msレイテンシ | MIT | Yes |

### 2.4 日本語対応状況

| モデル | 日本語対応 | 詳細 |
|--------|-----------|------|
| VALL-E X | 対応 | 英中日の3言語 |
| XTTS v2 | 対応 | 16言語の1つ |
| StyleTTS 2 | 英語のみ | 学習にはphonemizerが必要 |
| VoiceBox | 非対応 | 英仏西葡独波 |
| GPT-SoVITS | **ネイティブ対応** | v4で48kHz出力、日本語改善、コミュニティ活発 |
| OpenAudio S1 (旧Fish Speech) | **ネイティブ対応** | 日本語対応、4Bパラメータ |
| F5-TTS | 対応 | Emiliaデータセットで多言語学習 |
| CosyVoice | **ネイティブ対応** | 9言語、cross-lingual対応 |
| OpenVoice v2 | 対応 | 6言語 |
| Bark | 対応 | 品質は英語以外劣る |

---

## 3. Speaker Encoder手法の比較

### 3.1 手法一覧

| 手法 | 出力次元 | EER (VoxCeleb) | 計算コスト | 最小入力長 | ライセンス |
|------|---------|---------------|-----------|-----------|-----------|
| d-vector (GE2E) | 256 | ~7-10% | 低 | ~1-3秒 | 実装による |
| x-vector (TDNN) | 512 | ~3-5% | 低 | ~0.5秒 | Apache 2.0 |
| **ECAPA-TDNN** | **192** | **0.86-1.71%** | **中** | **~0.5秒** | **Apache 2.0** |
| WavLM (SSL) | 768/1024 | 0.99% (fine-tuned) | 高 | ~1秒 | MIT |
| WavLM + ECAPA | 192-256 | **0.39%** | 高 | ~1秒 | MIT + Apache |
| Resemblyzer | 256 | 7.46-26.47% | 低 | ~1-3秒 | Apache 2.0 |
| SpeechBrain | 192/512 | 0.80% (ECAPA) | 中 | ~0.5秒 | Apache 2.0 |
| Coqui/TTS SE | 256 | ~7-10% | 低 | ~1-3秒 | MPL 2.0 |
| GST | 256 | N/A | 中 | 発話全体 | 実装による |
| CLAP | 512 | N/A | 高 | ~1秒 | Apache 2.0 |

### 3.2 各手法の詳細

#### ECAPA-TDNN（初期調査時の推奨）
- x-vectorの改良版。Squeeze-and-Excitation, Res2Net, Attentive Statistical Poolingを導入
- 192次元で軽量ながら高精度 (EER 0.86%)
- **非英語話者（日本語含む）でトップクラスの識別性能** (EER 1.71%)
- SpeechBrain (Apache 2.0), NVIDIA NeMo (Apache 2.0) で事前学習済みモデル利用可能

> **最終決定 (2026-03-10)**: ECAPA-TDNNではなく**CAM++** (EER 0.73%, Apache-2.0) を採用。精度・効率・ONNX対応済みである点で総合的に優位と判断。詳細はセクション11.5を参照。

#### WavLM / HuBERT (自己教師学習)
- 音響的に豊かな表現。fine-tuning後は最高精度 (EER 0.99%)
- 計算コストが高い（Transformerベース）
- **非英語で単体性能低下 (EER 10.88%)** — 日本語ではECAPAが優位

#### WavLM + ECAPA-TDNN（最高精度）
- WavLMをフロントエンド、ECAPA-TDNNをバックエンドとして組み合わせ
- **EER 0.39%** (VoxCeleb1-O) — 現在の最高水準
- 計算コストが最も高い

#### d-vector (GE2E)
- シンプル、軽量、推論高速（GPU上で約1000x RT）
- YourTTSで実績あり
- 精度はECAPA-TDNNに劣る

#### GST / CLAP
- GSTは話者識別ではなくスタイル転送向け。話者クローニングには不向き
- CLAPは粗い音声記述ベースで個別話者特徴を捉えられない

### 3.3 推奨ランキング

> **最終決定 (2026-03-10)**: 下記は初期調査時のランキング。最終的に**CAM++** (EER 0.73%, 7.2Mパラメータ, 27MB ONNX, Apache-2.0) を採用した。CosyVoiceでの実績、ONNX対応済み、精度/サイズの最良バランスが決め手。詳細はセクション11.5を参照。

| 順位 | 手法 | 理由 |
|------|------|------|
| 1 | **SpeechBrain ECAPA-TDNN** | Apache 2.0、192次元で軽量、日本語でトップ精度、HuggingFace統合 |
| 2 | ECAPA-TDNN (NVIDIA NeMo) | Apache 2.0、TitaNetモデル利用可能 |
| 3 | d-vector (GE2E) 256次元 | シンプルさと速度重視。YourTTSで実績 |
| 参考 | WavLM + ECAPA | 精度最優先の場合 (EER 0.39%) だが計算コスト大 |

### 3.4 重要な知見

- **非英語（日本語含む）ではECAPA-TDNNが最も安定**: WavLM単体は英語中心の学習データのため非英語でEER 10.88%に低下
- **CLAP/GSTは話者識別には不適**: スタイル/感情転送には有用だが話者クローニングには向かない
- **SpeechBrainのApache 2.0ライセンスはPiperのGPL-free方針と完全合致**

---

## 4. VITSベースのzero-shot拡張手法

### 4.1 Piperの現在の話者条件付けアーキテクチャ

Piperの `SynthesizerTrn` (models.py) では、マルチスピーカー対応を話者ID embedding lookupで実現:

```python
# 現在の実装
if n_speakers > 1:
    self.emb_g = nn.Embedding(n_speakers, gin_channels)  # gin_channels=512
```

**`g` (speaker embedding) が注入される4箇所:**

| 箇所 | モジュール | 注入方法 |
|------|-----------|---------|
| Duration Predictor | `dp` | `self.cond(g)` で加算 |
| Posterior Encoder | `enc_q` | WN層の `cond_layer(g)` 経由 |
| Flow | `flow` | ResidualCouplingLayer内のWN層 |
| Generator/Decoder | `dec` | `self.cond(g)` で `conv_pre` 出力に加算 |

**重要**: TextEncoderには話者条件付けがない（オリジナルVITSと同じ）。

### 4.2 拡張手法の比較

| 手法 | 変更規模 | 実装難易度 | 品質期待値 | 既存モデル移行 | 推論速度影響 |
|------|---------|-----------|-----------|-------------|------------|
| **A: YourTTS方式** | 小 | 低 | 中 | 容易 | 微小 |
| B: VITS2方式 | 小 | 低 | 中低 | 非常に容易 | 微小 |
| C: XTTS方式 | 大 | 非常に高 | 最高 | 不可能 | 大幅低下 |
| D: MB-iSTFT-VITS | 中 (Decoderのみ) | 中 | 同等 | 再学習必要 | 4.1倍高速化 |
| E: DINO-VITS | 中 | 中 | 高 | 部分的 | 小 |
| F: GPT-SoVITS | 大 | 高 | 高 | 困難 | かなり低下 |

### 4.3 YourTTS方式の詳細（初期調査時の推奨）

VITSをベースに、話者ID embeddingを外部Speaker Encoderのd-vector/ECAPA embeddingに置換する方法。

> **最終実装 (2026-03-10)**: 下記はYourTTS方式の調査段階のコード例。最終実装では、Speaker Encoderをモデル内に組み込むのではなく、**事前計算したembeddingを`nn.Linear`で射影するDual-Mode方式**を採用した。`emb_g` (speaker ID) と `spk_proj` (speaker embedding射影) が共存し、`--export-mode {sid, zero-shot}` でONNXモデルを分離する設計。詳細はセクション11-12および実装ファイル (`models.py`) を参照。

**コード変更の核心:**
```python
# 変更前 (models.py)
if n_speakers > 1:
    self.emb_g = nn.Embedding(n_speakers, gin_channels)
g = self.emb_g(sid).unsqueeze(-1)  # [b, gin_channels, 1]

# 調査時の想定
self.speaker_encoder = SpeakerEncoder(output_dim=gin_channels)
g = self.speaker_encoder(ref_audio).unsqueeze(-1)  # [b, gin_channels, 1]

# 最終実装（Dual-Mode）
if n_speakers > 1:
    self.emb_g = nn.Embedding(n_speakers, gin_channels)  # speaker ID用（維持）
if use_zero_shot:
    self.spk_proj = nn.Linear(spk_embed_dim, gin_channels)  # embedding射影用（192→512）
```

残りの全コードは `g` の形状が同じためそのまま動作する。

**変更が必要なファイル:**

| ファイル | 変更内容 | 変更量 |
|----------|----------|--------|
| `vits/models.py` (SynthesizerTrn) | `emb_g` をSpeaker Encoderに置換 | ~30行 |
| `vits/models.py` (新規) | `SpeakerEncoder` クラス追加 | ~50-100行 |
| `vits/lightning.py` | 学習ループでembedding計算追加 | ~30行 |
| `vits/dataset.py` | 参照音声のロード処理追加 | ~20行 |
| `export_onnx.py` | Speaker Encoder出力をONNX入力に変更 | ~20行 |
| `infer_onnx.py` | 参照音声からembedding抽出追加 | ~30行 |

**Speaker Encoder選択肢:**
- ECAPA-TDNN (SpeechBrain): 高品質、Apache-2.0 — 初期推奨
- Resemblyzer: 軽量、MIT
- WavLM (既にPiperに統合済み) を流用: 追加依存なし
- **CAM++ (3D-Speaker/CosyVoice): Apache-2.0 — 最終採用**

**学習戦略:**
1. Speaker Encoderを凍結して VITS 部分のみ学習（50-100 epoch）
2. Speaker Encoderを解凍して end-to-end 微調整（50 epoch）

**既存モデルからの移行:**
- 既存話者のembeddingを事前計算して保存
- 新モデルで同じembeddingを使用すれば既存話者も再現可能

---

## 5. OSS実装の実用性・日本語対応状況

### 5.1 OSS実装比較表

| 実装 | Stars | ライセンス | 日本語 | リファレンス | RTF | GPU VRAM | CPU推論 | ONNX | メンテナンス |
|------|-------|-----------|--------|------------|-----|----------|---------|------|------------|
| GPT-SoVITS | ~40K+ | **MIT** | ネイティブ | 5秒 | ~0.5 (CPU) | 6-8GB | 可 | 部分的 | 活発 (v3) |
| Fish Speech | ~25K | Apache/CC-BY-NC-SA | ネイティブ | 5-10秒 | 高速 | 4-8GB | 困難 | 未対応 | 活発 |
| F5-TTS | ~15K+ | MIT/CC-BY-NC | 多言語 | 5-15秒 | 0.03-0.15 | 4-8GB | 可 (OpenVINO) | **対応** | 活発 |
| CosyVoice | ~15K+ | **Apache-2.0** | ネイティブ | 3-10秒 | 低レイテンシ | 4-8GB | 可 (遅い) | 部分的 | 非常に活発 |
| MaskGCT | ~8K | CC-BY-NC-4.0 | 対応 | 5秒 | 中程度 | 8-16GB | 困難 | 未対応 | 研究寄り |
| XTTS v2 | ~35K+ | CPML | 対応 | 6-10秒 | 中程度 | 4-6GB | 可 (遅い) | 未対応 | 停滞 |
| OpenVoice v2 | ~30K+ | **MIT** | 対応 | 5-10秒 | 高速 | 4GB | 可 | 未確認 | 中程度 |
| Bark | ~36K+ | **MIT** | 対応 (品質劣) | プリセット | ~RT | 8-12GB | 可 (10x遅) | 未対応 | 停滞 |
| StyleTTS 2 | ~5K+ | **MIT** | 学習可能 | 5-10秒 | 高速 | 3-4GB | 可 | **対応** (Kokoro) | 中程度 |

### 5.2 日本語zero-shot TTSとして実用的なTOP3

#### 第1位: GPT-SoVITS (MIT)
- MIT（完全に商用利用可能、GPL問題なし）
- v3で日本語のtimbre再現性が大幅改善
- 5秒のリファレンスでzero-shot、1分でfew-shot高品質合成
- M4 CPUでRTF 0.5（実用レベル）
- 40K+スター、非常に活発なコミュニティ

#### 第2位: CosyVoice (Apache-2.0)
- Apache-2.0（商用利用可能、GPL問題なし）
- Alibaba FunAudioLLMによる大規模多言語学習
- v3で150msの低レイテンシ、MOS 5.53
- v3は1.5Bパラメータと大きく、edge推論には不向き

#### 第3位: F5-TTS (MIT)
- コード MIT（自前学習すればモデルも制約なし）
- 非自己回帰でRTF 0.03-0.15と非常に高速
- ONNX対応、OpenVINO経由でCPU推論可能
- 事前学習モデルはCC-BY-NC（商用利用不可、自前データで再学習が必要）

### 5.3 ライセンス面の結論

- **GPT-SoVITS (MIT)** と **CosyVoice (Apache-2.0)** はGPL排除方針と完全適合
- Fish SpeechのモデルCC-BY-NC-SA、F5-TTSの事前学習モデルCC-BY-NCは商用利用不可
- コード自体がMIT/Apacheのものは自前データで学習すればライセンス問題なし

---

## 6. Piperへの推奨アプローチ

> **注**: 本セクションは初期調査時の推奨です。推論速度ゼロインパクトの制約を加えた**最終推奨はセクション11-12**を参照してください。ONNX設計がref_mel入力方式 → 事前計算speaker_embedding入力方式に変更されています。

### 6.1 段階的導入ロードマップ（精度最優先版）

#### Phase 1: Speaker Encoder + DINO/SCL Loss（最優先）

> **最終実装 (2026-03-10)**: Speaker EncoderはECAPA-TDNNではなく**CAM++ ONNX** (192次元, 27MB, Apache-2.0) を採用。モデル内にSpeaker Encoderを組み込むのではなく、事前計算embedding + `nn.Linear(192, 512)` 射影のDual-Mode方式で実装。gin_channels=512（768はガビガビ音が発生したため不採用）。WavLM Discriminatorは`--no-wavlm`で無効化して学習。学習環境はRTX 6000 Ada 48GB x1、batch_size=160、bf16-mixed。

**概要**: Speaker Encoder (ECAPA-TDNN) を導入し、DINO LossとSpeaker Consistency Lossで精度を最大化

**技術選定:**
- Speaker Encoder: **ECAPA-TDNN** (事前学習済み SpeechBrain, Apache-2.0) → **最終: CAM++ ONNX (Apache-2.0)**
- 損失関数: 既存VITS損失 + Speaker Consistency Loss + DINO Loss + WavLM Perceptual Loss
- Dual-mode: `--speaker-id` (既存) と `--ref-audio` (zero-shot) の共存 → **最終: `--speaker-id` と `--speaker-embedding` の共存**
- 注入箇所: 既存の4箇所 (DP, PosteriorEnc, Flow, Decoder) そのまま

**学習戦略 (3フェーズ):**
1. Speaker Encoder事前学習済みモデル使用（学習時間ゼロ）
2. TTS本体学習 — Speaker Encoder凍結（95K iterations, ~70h）
3. Joint Training — Speaker Encoder解凍（175K iterations, ~140h）

**学習データ:**
- LibriTTS-R (585h, 2,456話者) + JVS (30h, 100話者) + 現行20話者
- 合計: ~615h, ~2,576話者

**推定工数**: 5-7日（実装）+ 180-240h（学習）

#### Phase 2: TextEncoder Speaker Conditioning追加

**概要**: VITS2方式でTextEncoderにもspeaker conditioningをCross-Attentionで追加

**変更内容:**
- `enc_p` (TextEncoder) にCross-Attention + speaker conditioning
- attentions.pyの `Encoder` クラスに `ConditionedEncoder` として拡張

**期待効果**: 話者類似度のさらなる向上 (SIM-O +0.02-0.05)

**推定工数**: 1-2日 + 再学習

#### Phase 3: デコーダ高速化 (オプション)

**概要**: MB-iSTFT-VITSデコーダによる推論4倍高速化

**推定工数**: 3-5日 + 再学習

#### 将来: アーキテクチャ刷新

VITSの品質上限 (SIM-O ~0.65) を超えるために、Flow Matching/Diffusion系への移行を検討。
詳細はセクション7.4を参照。

### 6.2 参考にすべきプロジェクト

| プロジェクト | 参考になる点 |
|-------------|-------------|
| YourTTS | VITSへのSpeaker Encoder統合のリファレンス実装 |
| DINO-VITS | DINO Lossによるspeaker encoderの頑健性向上 |
| SpeechBrain | ECAPA-TDNNの事前学習済みモデル |
| CosyVoice | Apache-2.0、日本語対応、話者embeddingキャッシュ |
| GPT-SoVITS | 日本語zero-shotの品質ベンチマーク |

---

## 7. 高精度アプローチの定量評価

> 追加調査 (2026-03-07): 精度最優先の観点から、主要モデルのベンチマーク数値を収集。

### 7.1 LibriSpeech test-clean ベンチマーク

| モデル | WER (%) | SIM-O | アーキテクチャ | 学習データ |
|--------|---------|-------|---------------|-----------|
| Ground Truth | ~1.94 | 基準 | - | - |
| NaturalSpeech 3 | **1.81** | 0.67 | Factorized Diffusion | 非公開 |
| MegaTTS 3 | ~1.88 | ~0.698 | Latent Diffusion Transformer | 非公開 (大規模) |
| MaskGCT | 2.634 | **0.687** | Masked Generative Codec Transformer | 100K時間 |
| F5-TTS | 2.42 | ~0.65 | Flow Matching (DiT) | 50K時間 |
| VALL-E 2 | 3.8-4.2 | 0.803-0.807 | AR Codec LM | 60K時間 |
| StyleTTS 2 | N/A | MOS-S 高 | Style Diffusion + GAN | 245時間 |
| **YourTTS (VITS)** | **~7-8** | **~0.55-0.60** | **VITS + Speaker Encoder** | **数百時間** |

### 7.2 Seed-TTS Eval ベンチマーク

**test-en (英語):** *(2026-03-07 更新)*

| モデル | WER (%) | SIM (%) | 備考 |
|--------|---------|---------|------|
| **Qwen3-TTS (1.7B)** | **1.24** | **78.9** | 2026-01, 現在のSOTA |
| CosyVoice 3 (1.5B RL) | 1.45 | 77.4 | 2025-12 |
| VoxCPM | 1.85 | N/A | 2025-09 |
| F5-TTS | 2.00 | 67.0 | |
| Seed-TTS | 2.25 | 76.2 | |
| MaskGCT | 2.62 | 71.7 | |
| CosyVoice 2 | 3.09 | 65.9 | |

**test-zh (中国語):** *(2026-03-07 更新)*

| モデル | CER (%) | SIM (%) | 備考 |
|--------|---------|---------|------|
| CosyVoice 3 (1.5B RL) | **0.71** | N/A | 2025-12 |
| **Qwen3-TTS (1.7B)** | 0.77 | 高 | 2026-01 |
| CosyVoice 3 (0.5B) | 0.81 | 77.4 | |
| GLM-TTS (RL) | 0.89 | 76.4 | 2025-12 |
| VoxCPM | 0.93 | N/A | 2025-09 |
| MaskGCT | 2.27 | 77.4 | |

### 7.3 主観評価 (MOS / CMOS / SMOS)

| モデル | MOS/CMOS | SMOS (話者類似) | 備考 |
|--------|----------|----------------|------|
| MegaTTS 3 | CMOS 4.42, MOS 4.28-4.32 | SOTA | EN/ZH両方で最高クラス |
| NaturalSpeech 3 | CMOS 0.00 (人間同等) | SMOS 4.01 | 全baselines超え |
| MaskGCT | CMOS +0.12-0.37 | SMOS 最高 | ICLR 2025で検証 |
| F5-TTS | CMOS 0.31 (EN) | SMOS 3.89 | 非ARで高品質 |
| StyleTTS 2 | MOS 4.1+ | MOS-S 高 | 人間レベル到達 |
| VALL-E 2 | 人間同等 | 高 | Microsoft内部評価 |

### 7.4 VITSの品質上限と将来のアーキテクチャ選択肢

#### VITSアーキテクチャ内での到達可能精度

| 改善策 | 期待SIM-O | 期待WER | 実現性 |
|--------|----------|---------|--------|
| 現状 (Piper multi-speaker) | ~0.40-0.50 | N/A | 現状 |
| + Speaker Encoder置換 (YourTTS方式) | ~0.55-0.62 | ~7-8 | 高 |
| + DINO Loss + SCL Loss追加 | ~0.58-0.65 | ~6-7 | 高 |
| + TextEncoder Conditioning | ~0.60-0.65 | ~5-6 | 高 |
| **VITSの理論上限** | **~0.65** | **~5** | gin_channels=512が実用上限。768はガビガビ音発生 |

#### VITSの根本的制約
- Speaker Embeddingとprosodyが結合 → 未知話者への汎化が弱い
- GAN学習の不安定性 → スケーリングが困難
- Flow/Diffusionに比べて表現力の天井が低い
- **gin_channels=768ではガビガビ音が発生** → gin_channels=512が実用上の上限（2026-03-10確認）

#### アーキテクチャ変更による精度向上幅

| アプローチ | 期待SIM-O | 変更規模 | 備考 |
|-----------|----------|---------|------|
| VITSベースのまま（Phase 1-2） | ~0.65 | 小-中 | **今回推奨** |
| VITSデコーダ + Flow Matching | ~0.68-0.72 | 中-大 | 将来候補 |
| 完全新アーキテクチャ (F5-TTS/MaskGCT) | ~0.70-0.75 | 大 | 将来候補 |
| CosyVoice方式 (LLM + Flow Matching) | ~0.75-0.80 | 非常に大 | 最高品質 |

#### 技術的優位点の分析

**話者再現方式の比較:**

| 方式 | 話者類似度 (SIM-O) | 代表モデル |
|------|-------------------|-----------|
| Speaker Encoder (ECAPA等) | ~0.55-0.65 | YourTTS, VITS |
| Reference Encoder | ~0.60-0.68 | StyleTTS 2 |
| In-context Learning | **~0.70-0.80** | Seed-TTS, VALL-E 2, CosyVoice |

**生成方式の比較:**

| 方式 | 品質 | 推論速度 | 安定性 |
|------|------|---------|--------|
| Flow Matching | 高 | 速い (RTF 0.15) | 高 |
| Diffusion | 最高 | 中程度 | 高 |
| Autoregressive | 高 | 遅い | 低め |
| **VITS (GAN)** | **中** | **最速** | **中** |

**連続潜在表現 vs 離散トークン:**
- 連続潜在表現: +0.80 UTMOS, +0.40 SIG 改善（量子化損失を回避）
- 51.5Mパラメータでも VALL-E超えが可能

#### 将来のアーキテクチャ刷新候補

精度最優先で将来的にVITSを超える場合の候補:

| 候補 | 総合品質 | 話者類似度 | OSS | ライセンス | 日本語 |
|------|---------|----------|-----|-----------|--------|
| **Qwen3-TTS** | S+ | S+ | 完全OSS | **Apache-2.0** | 対応 (10言語) |
| **CosyVoice 3** | S | S | 完全OSS | **Apache-2.0** | 対応 (9言語) |
| **VoxCPM** | S | A+ | 完全OSS | **Apache-2.0** | 対応見込み |
| **MegaTTS 3** | A+ | A+ | 完全OSS | **Apache-2.0** | 未対応 |
| **MaskGCT** | S | A+ | 完全OSS | CC-BY-NC | 学習データ次第 |
| **F5-TTS** | A+ | A | 完全OSS | MIT/CC-BY-NC | 学習データ次第 |
| GPT-SoVITS v4 | A | A+ | 完全OSS | **MIT** | **最良** |
| **Chatterbox** | A+ | A+ | 完全OSS | **MIT** | 対応 (23言語) |
| Kokoro | A | A | 完全OSS | **Apache-2.0** | 対応 (5言語) |

---

## 8. 既存機能との互換性設計

> 追加調査 (2026-03-07): 既存の全機能を維持しながらzero-shotを追加するための設計パターン。

### 8.1 Dual-mode Speaker Conditioning

既存の `emb_g` (speaker ID) と新規の speaker embedding射影 (zero-shot) を共存させる設計:

> **最終実装 (2026-03-10)**: 下記は調査段階の設計案。最終実装ではモデル内にSpeaker Encoderを組み込むのではなく、事前計算済みembeddingを`nn.Linear`で射影する方式を採用。`use_speaker_encoder` ではなく `use_zero_shot` フラグ（マルチスピーカーモデルでデフォルト有効）で制御。

```python
# 調査段階の設計案
class SynthesizerTrn(nn.Module):
    def __init__(self, ..., use_speaker_encoder=False, speaker_embed_dim=192):
        ...
        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)
        if use_speaker_encoder:
            self.speaker_encoder = SpeakerEncoder(mel_channels=80, gin_channels=gin_channels)

# 最終実装（models.py）
class SynthesizerTrn(nn.Module):
    def __init__(self, ..., use_zero_shot=False, spk_embed_dim=192):
        ...
        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)  # speaker ID用
        if use_zero_shot:
            self.spk_proj = nn.Linear(spk_embed_dim, gin_channels)  # 192→512射影

    def forward(self, ..., sid=None, speaker_embedding=None):
        if speaker_embedding is not None and hasattr(self, "spk_proj"):
            g = self.spk_proj(speaker_embedding).unsqueeze(-1)  # [b, gin_channels, 1]
        elif sid is not None and hasattr(self, "emb_g"):
            g = self.emb_g(sid).unsqueeze(-1)                   # [b, gin_channels, 1]
        else:
            g = None
```

**重要**: `g` の形状 `[batch, gin_channels, 1]` が共通なので、下流の4コンポーネント（Duration Predictor, Posterior Encoder, Flow, Decoder）は一切変更不要。`gin_channels=512` が最終値（768はガビガビ音発生のため不採用）。

### 8.2 ONNX推論でのdual-mode対応

> **最終実装 (2026-03-10)**: 1モデル統合ではなく、`--export-mode {auto, sid, zero-shot}` で**別々のONNXモデル**として出力する方式を採用。sid用モデルは`sid`入力、zero-shot用モデルは`speaker_embedding`入力を持つ。ref_mel入力方式は採用せず、事前計算済みの192次元embeddingを直接入力する。

調査段階では1モデル統合 + Optional入力方式を想定:

```python
# 調査段階の設計案
input_names = ["input", "input_lengths", "scales"]
if num_speakers > 1:
    input_names.append("sid")
if use_speaker_encoder:
    input_names.append("ref_mel")  # optional入力

# 最終実装（export_onnx.py）
# --export-mode sid: input_names = ["input", "input_lengths", "scales", "sid", ...]
# --export-mode zero-shot: input_names = ["input", "input_lengths", "scales", "speaker_embedding", ...]
```

```python
# 最終実装（infer_onnx.py）
if args.speaker_embedding:
    spk_emb = np.load(args.speaker_embedding)  # 事前計算済み192次元embedding
    inputs["speaker_embedding"] = spk_emb.reshape(1, -1)
elif args.speaker_id is not None:
    inputs["sid"] = np.array([args.speaker_id], dtype=np.int64)
```

### 8.3 Speaker Encoder分離設計

> **最終実装 (2026-03-10)**: この分離設計が最終的に採用された。CAM++ ONNX (`campplus.onnx`) をオフラインツール (`extract_speaker_embedding.py`) で使用し、事前計算済み `.npy` ファイルを推論時に読み込む方式。

Speaker Encoderを別ONNXモデルとして分離し、embeddingキャッシュに対応:

```
推論パイプライン（最終実装）:
  ref_audio → [CAM++ ONNX (extract_speaker_embedding.py)] → speaker.npy (192次元)
                                                               ↓
  text → [Piper TTS ONNX (spk_proj: 192→512)] → audio
```

**embeddingキャッシュの利点:**
- 同一話者の2回目以降の推論でSpeaker Encoder不要
- CPU推論時のレイテンシ削減（0ms追加）
- 事前登録話者を `.npy` ファイルで呼び出し可能

### 8.4 チェックポイント互換性

> **最終実装 (2026-03-10)**: `strict=False` でロード。`spk_proj` はランダム初期化される。

```python
# 既存チェックポイントからの移行
model = VitsModel.load_from_checkpoint(ckpt_path, strict=False)
# strict=False により spk_proj 部分がない場合でもロード可能
# spk_proj はランダム初期化 → 学習で最適化
```

**段階的移行パス:**
1. `strict=False` で既存チェックポイントをロード → `emb_g` はそのまま使える
2. Per-utterance embeddingを事前計算 (`extract_speaker_embedding.py --per-utterance`)
3. 学習開始 → `spk_proj` が最適化される
4. `--export-mode sid` / `--export-mode zero-shot` でONNXモデルを出力

### 8.5 維持される既存機能一覧

> **最終確認 (2026-03-10)**: 全機能の互換性を実装で確認済み。

| 機能 | 互換性 | 備考 |
|------|--------|------|
| `--speaker-id` による話者指定 | 完全互換 | `emb_g` をそのまま保持（`--export-mode sid`で出力） |
| `--speaker-embedding` による零ショット | **新規** | `spk_proj` で192→512射影（`--export-mode zero-shot`で出力） |
| 日本語/英語 Phonemizer | 影響なし | 独立モジュール |
| Prosody Features (A1/A2/A3) | 影響なし | Duration Predictorへの入力は変更なし |
| ONNX推論 | 互換 | `--export-mode` で入力形式を分離 |
| CPU推論 | 互換 | Speaker Encoderは推論パイプライン外（追加コスト0） |
| WavLM Discriminator | 互換 | 学習時のみ、`--no-wavlm` で無効化可能 |
| EMA重み | 互換 | export_onnxの処理は変更なし |

---

## 9. 学習戦略・データ要件

> 追加調査 (2026-03-07): 精度を最大化するための学習データ・戦略・評価手法。

### 9.1 データ要件

#### 話者数と品質の関係

| 規模 | 話者数 | 期待品質 | 根拠 |
|------|--------|----------|------|
| 最低限 | 100-200 | 限定的なzero-shot | YourTTS: VCTK 109話者で基本動作 |
| **推奨** | **500-2,000** | **実用的なzero-shot** | StyleTTS2: 1,151話者で高品質達成 |
| 理想 | 5,000+ | 高品質zero-shot | CosyVoice: 6,000話者以上 |

**コストパフォーマンスの閾値**: VALL-E 2の実験で「10K時間で50K時間とほぼ同等の性能」が確認。

#### 推奨データ構成

| コーパス | 話者数 | 時間 | ライセンス | 用途 |
|----------|--------|------|-----------|------|
| **LibriTTS-R** | 2,456 | 585h | CC-BY-4.0 | メイン学習データ（英語） |
| **JVS** | 100 | 30h | 研究用無料 | 日本語話者の多様性 |
| **現行データセット** | 20 | ~100h | 独自 | 高品質日本語 |
| **VoxCeleb2** | 6,112 | 2,442h | 研究用 | Speaker Encoder事前学習 |
| **合計** | **~2,576** | **~615h** | - | TTS本体の学習 |

#### 多言語データの活用
YourTTSの実験で、英語+他言語の混合学習がzero-shot性能を向上させることが確認されている。日本語コーパスは話者数が限られるため、英語データによる話者多様性の補完が重要。

#### Data Augmentation

| 手法 | 効果 | 推奨度 |
|------|------|--------|
| **MUSAN noise injection** | ノイズ耐性向上 | 必須 |
| **Room Impulse Response (RIR)** | 残響耐性 | 高 |
| **Speed perturbation** (0.9x/1.0x/1.1x) | 3倍のデータ量 | 高 |
| SpecAugment | 周波数/時間マスキング | 中 |

### 9.2 3フェーズ学習スケジュール

> **最終実装 (2026-03-10)**: 下記は調査段階の計画。実際の学習ではSpeaker Encoderをモデルに組み込まず、CAM++ ONNXで事前計算したper-utterance embeddingを使用。学習環境はRTX 6000 Ada 48GB x1、batch_size=160、samples_per_speaker=8、bf16-mixed、`--no-wavlm`、200 epochs。Speaker Encoderの凍結/解凍は不要（モデル外部で事前計算するため）。

#### Phase 1: Speaker Encoder準備（学習時間ゼロ）
- **方法**: SpeechBrain事前学習済みECAPA-TDNNを採用 → **最終: CAM++ ONNX (27MB, Apache-2.0)**
- **モデル**: `spkrec-ecapa-voxceleb` (Apache-2.0) → **最終: `campplus.onnx` (CosyVoice-300Mリポジトリから取得)**
- 学習不要、ダウンロードのみ
- **追加**: per-utterance embedding抽出 (`extract_speaker_embedding.py --per-utterance`) でdataset.jsonlを更新

#### Phase 2: TTS本体学習 — Speaker Encoder凍結
- **データ**: LibriTTS-R + JVS + 現行20話者 (~615h)
- **設定**: Speaker Encoderの重みを凍結、TTS本体のみ学習
- **iterations**: 95,000 (DINO-VITSの設定に準拠)
- **batch_size**: 10-12 (L4 16GBに収まるよう調整)
- **推定学習時間**: **約60-80時間** (L4 x4)

#### Phase 3: Joint Training — Speaker Encoder解凍
- **データ**: 同上
- **設定**: Speaker Encoder全体を解凍、低学習率で微調整
- **iterations**: 175,000
- **batch_size**: 8-10 (encoder解凍でメモリ増加)
- **推定学習時間**: **約120-160時間** (L4 x4)

**注意**: Phase 2ではWavLM Discriminatorを無効化し、Phase 3で有効化する戦略が有効（GPUメモリ制約）。

> **実際の学習設定 (2026-03-10)**: Speaker Encoder外部化によりPhase 2/3の区別は不要に。実際のコマンド:
> ```bash
> uv run python -m piper_train \
>   --dataset-dir /home/shadeform/data/piper/dataset-moe-speech-20speakers \
>   --prosody-dim 16 --accelerator gpu --devices 1 --precision bf16-mixed \
>   --max_epochs 200 --batch-size 160 --samples-per-speaker 8 \
>   --checkpoint-epochs 2 --quality medium --base_lr 2e-4 \
>   --ema-decay 0.9995 --num-workers 8 --no-wavlm \
>   --default_root_dir /home/shadeform/data/piper/output-moe-speech-20speakers
> ```

### 9.3 損失関数の組み合わせ

| # | 損失関数 | 重み | 効果 | 状態 |
|---|---------|------|------|------|
| 1 | VITS損失 (reconstruct + KL + adversarial) | 既存 | 基本音質 | 実装済み |
| 2 | **Speaker Consistency Loss** | c_spk=9.0 | 話者再現精度 | **新規** |
| 3 | WavLM Perceptual Loss | c_wavlm=0.5 | 知覚品質 | 実装済み |
| 4 | **DINO Loss** | c_dino=0.1 | ノイズ耐性、話者分離性 | **新規** |

**Speaker Consistency Loss (SCL):**
```
L_scl = 1 - cosine_similarity(
    speaker_encoder(synthesized_audio),
    speaker_encoder(reference_audio)
)
```
YourTTS/XTTS/Coqui TTSで広く使用。実装コストが低く効果が高い。

**DINO Loss:**
- DINO-VITSで提案 (Interspeech 2024)
- 2つのランダムセグメントで自己蒸留
- ノイズの多いデータでも効果的

### 9.4 GPUメモリ見積もり (L4 16GB)

> **最終環境 (2026-03-10)**: RTX 6000 Ada 48GBを使用。Speaker Encoderはモデル外部のため学習時メモリ不要。WavLM Discriminatorは`--no-wavlm`で無効化。batch_size=160が可能。

| コンポーネント | メモリ | 備考 |
|--------------|-------|------|
| VITS本体 | ~8GB | 現行と同等 |
| Speaker Encoder (ECAPA-TDNN) | ~1-2GB | 学習時/推論時 → **最終: 外部化のため0** |
| WavLM Discriminator | ~1-2GB | 学習時のみ → **最終: --no-wavlmで無効化** |
| データ/勾配 | ~3-4GB | batch_size依存 |
| **合計** | **~13-16GB** | ギリギリ収まる → **最終: RTX 6000 Ada 48GBで余裕** |

### 9.5 評価指標と自動評価パイプライン

| 指標 | 測定内容 | ツール | 目標値 |
|------|---------|--------|--------|
| **SECS** | 話者類似度 (cosine similarity) | ECAPA-TDNN / WavLM-SV | > 0.60 |
| **SV-EER** | 話者検証等エラー率 | ASVモデル | ~50% (理想) |
| **UTMOS** | 自動MOS推定 | UTMOS | > 3.5 |
| **WER** | 明瞭度 | Whisper large-v3 | < 5% |

**評価パイプライン:**
1. テスト話者のリファレンス音声を用意（各5-10秒、未見話者）
2. リファレンスからspeaker embeddingを抽出
3. 同じ話者のテキストでTTS合成
4. 合成音声のspeaker embeddingを抽出
5. cosine similarityを計算 (SECS)
6. UTMOS, WERも自動計算

---

## 10. 具体的な実装計画

> 追加調査 (2026-03-07): Piperコードベースを分析した上での具体的な変更計画。

> **最終実装 (2026-03-10)**: 下記の実装計画は調査段階のもの。最終実装ではSpeaker Encoderクラス (`speaker_encoder.py`) は新規作成せず、`spk_proj = nn.Linear(spk_embed_dim, gin_channels)` としてSynthesizerTrn内に直接実装。Speaker Encoding処理は `extract_speaker_embedding.py` としてオフラインツールに分離。ref_mel入力方式は採用せず、事前計算済みembedding (192次元 .npy) を直接入力する設計。

### 10.1 新規ファイル

#### `src/python/piper_train/vits/speaker_encoder.py` (~200行) — **未採用: 代わりにextract_speaker_embedding.pyを実装**

```python
class SpeakerEncoder(nn.Module):
    """ECAPA-TDNN based speaker encoder.
    Output: [batch, gin_channels]
    """
    def __init__(self, mel_channels=80, gin_channels=512,
                 channels=512, kernel_sizes=[5,3,3,3,1],
                 dilations=[1,2,3,4,1]):
        super().__init__()
        self.conv_stem = Conv1d(mel_channels, channels, 5, padding=2)
        self.se_res2blocks = nn.ModuleList([
            SERes2Block(channels, kernel_size=k, dilation=d)
            for k, d in zip(kernel_sizes[1:], dilations[1:])
        ])
        self.asp = AttentiveStatisticsPooling(channels)
        self.proj = nn.Linear(channels * 2, gin_channels)

    def forward(self, mel, mel_lengths=None):
        x = F.relu(self.conv_stem(mel))
        for block in self.se_res2blocks:
            x = block(x)
        x = self.asp(x, mel_lengths)
        return self.proj(x)

class SERes2Block(nn.Module):
    """Squeeze-Excitation Res2Net Block"""
    ...

class AttentiveStatisticsPooling(nn.Module):
    """Attentive statistics pooling for variable-length input"""
    ...
```

### 10.2 既存ファイルの変更

#### `models.py` — SynthesizerTrn (~30行追加)

> **最終実装 (2026-03-10)**: 下記は調査段階の設計案。最終実装では `use_zero_shot=False, spk_embed_dim=192` パラメータで、`spk_proj = nn.Linear(spk_embed_dim, gin_channels)` として実装。Speaker Encoderクラスは組み込まず、事前計算embeddingを `speaker_embedding` 引数で受け取る。

```python
# 調査段階の設計案
def __init__(self, ..., use_speaker_encoder=False):
    if n_speakers > 1:
        self.emb_g = nn.Embedding(n_speakers, gin_channels)
    if use_speaker_encoder:
        self.speaker_encoder = SpeakerEncoder(mel_channels=80, gin_channels=gin_channels)

# 最終実装（models.py）
def __init__(self, ..., use_zero_shot=False, spk_embed_dim=192):
    if n_speakers > 1:
        self.emb_g = nn.Embedding(n_speakers, gin_channels)  # speaker ID用
    if use_zero_shot:
        self.spk_proj = nn.Linear(spk_embed_dim, gin_channels)  # 192→512

def forward(self, x, x_lengths, y, y_lengths, sid=None,
            prosody_features=None, speaker_embedding=None):
    if speaker_embedding is not None and hasattr(self, "spk_proj"):
        g = self.spk_proj(speaker_embedding).unsqueeze(-1)
    elif sid is not None and hasattr(self, "emb_g"):
        g = self.emb_g(sid).unsqueeze(-1)
    else:
        g = None
    # 以降の処理は全く同じ（gの形状が[b, gin_channels, 1]で共通）
```

#### `lightning.py` — 学習ループ (~40行追加)

> **最終実装 (2026-03-10)**: ref_mel方式ではなく、dataset.jsonlから事前計算済みspeaker_embeddingをロードして`model_g.forward(..., speaker_embedding=speaker_embeddings)`に渡す方式。`use_zero_shot=True`がデフォルト（マルチスピーカーモデル）。DINO center bufferを初期化。c_spk=9.0, c_dino=0.1がデフォルト値。

```python
# 調査段階の設計案（ref_mel方式）
def training_step_g(self, batch):
    ref_mel = spec_to_mel_torch(spec, ...)
    (y_hat, l_length, ...) = self.model_g(
        x, x_lengths, spec, spec_lengths, speaker_ids,
        ref_mel=ref_mel, ref_mel_lengths=spec_lengths)

# 最終実装（事前計算embedding方式）
def training_step_g(self, batch):
    speaker_embeddings = batch.speaker_embeddings  # from dataset.jsonl
    (y_hat, l_length, ...) = self.model_g(
        x, x_lengths, spec, spec_lengths, speaker_ids,
        prosody_features=prosody_features,
        speaker_embedding=speaker_embeddings)
```

#### `export_onnx.py` (~30行追加)

> **最終実装 (2026-03-10)**: ref_mel方式ではなく `--export-mode {auto, sid, zero-shot}` で出力を分離。zero-shotモードでは `speaker_embedding` (float32[batch, 192]) を入力として受け取る。

```python
# 調査段階の設計案
if model_g.use_speaker_encoder:
    input_names.append("ref_mel")

# 最終実装
if use_zero_shot:
    input_names.append("speaker_embedding")
    dynamic_axes["speaker_embedding"] = {0: "batch_size"}
```

#### `infer_onnx.py` (~50行追加)

> **最終実装 (2026-03-10)**: `--speaker-embedding` で .npy ファイルを指定。`--ref-audio` は未実装。

```python
# 最終実装のCLI引数
parser.add_argument("--speaker-embedding", help="Pre-computed embedding .npy file")

# 推論時
if args.speaker_embedding:
    spk_emb = np.load(args.speaker_embedding).astype(np.float32)
    inputs["speaker_embedding"] = spk_emb.reshape(1, -1)
```

#### `__main__.py` (~15行追加)

> **最終実装 (2026-03-10)**: `--use-speaker-encoder` フラグは不要（`use_zero_shot=True`がデフォルト）。`c_spk=9.0`、`c_dino=0.1` が `lightning.py` のデフォルト値。

```python
# 調査段階の設計案
parser.add_argument("--use-speaker-encoder", action="store_true")
parser.add_argument("--c-spk", type=float, default=1.0)

# 最終実装: フラグ不要（マルチスピーカーなら自動有効化）
# lightning.py: use_zero_shot=True, c_spk=9.0, c_dino=0.1 がデフォルト
```

### 10.3 変更対象ファイル一覧

> **最終実装 (2026-03-10)**: 実際に変更/新規作成されたファイルと、調査時の計画との対比。

| ファイル（調査時計画） | 変更種別 | 最終実装の状態 |
|---------|---------|---------|
| `vits/speaker_encoder.py` | **新規** | **未作成** — `spk_proj` として `models.py` 内に1行で実装 |
| `vits/models.py` | 修正 | **実装済み** — `spk_proj`, `use_zero_shot`, Dual-Mode forward |
| `vits/lightning.py` | 修正 | **実装済み** — `use_zero_shot=True` デフォルト, DINO center buffer |
| `vits/dataset.py` | 修正 | **実装済み** — `speaker_embedding_path` / per-utterance embedding読み込み |
| `export_onnx.py` | 修正 | **実装済み** — `--export-mode {auto, sid, zero-shot}` |
| `infer_onnx.py` | 修正 | **実装済み** — `--speaker-embedding` オプション |
| `__main__.py` | 修正 | **実装済み** — zero-shotはフラグ不要（自動有効化） |
| `extract_speaker_embedding.py` | **計画になし** | **新規作成** — CAM++ ONNXでオフラインembedding抽出 (~680行) |
| `prepare_zero_shot_dataset.py` | **計画になし** | **新規作成** — 複数コーパス統合ツール |
| `vits/losses.py` | 修正 | 変更なし |
| `test/test_speaker_encoder.py` | **新規** | 未作成 |
| `test/test_zero_shot_integration.py` | **新規** | 未作成 |

### 10.4 パラメータ数の増加

> **最終実装 (2026-03-10)**: Speaker Encoderはモデル外部化されたため、ONNXモデルにはEncoder含まれない。追加されるのは `spk_proj` (nn.Linear(192, 512)) のみ。

| コンポーネント | パラメータ数 | ONNXサイズ増加 |
|--------------|------------|---------------|
| ~~ECAPA-TDNN Speaker Encoder~~ | ~~6-8M~~ | ~~+25-30MB~~ → **モデル外部化のため0** |
| spk_proj (nn.Linear(192, 512)) | ~98K | <0.5MB |
| 推論速度への影響 | Linear(192→512) 1回 | ~数μs（実質ゼロ） |
| 本番ONNXモデルサイズ | - | 74MB (`moe-speech-20speakers-v2.onnx`) |

### 10.5 学習コマンド例

> **最終実装 (2026-03-10)**: `--use-speaker-encoder` フラグは不要（マルチスピーカーなら自動有効化）。`--zero-shot` フラグも削除済み。

```bash
# 調査段階のコマンド（参考）
NCCL_DEBUG=WARN NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
uv run python -m piper_train \
  --dataset-dir /data/piper/dataset-multispeaker-merged \
  --prosody-dim 16 --use-speaker-encoder --c-spk 1.0 \
  --accelerator gpu --devices 4 --precision 16-mixed \
  --max_epochs 200 --batch-size 10 --samples-per-speaker 2 \
  --checkpoint-epochs 1 --quality medium \
  --base_lr 2e-4 --disable_auto_lr_scaling \
  --ema-decay 0.9995 --num-workers 0 --no-pin-memory \
  --default_root_dir /data/piper/output-zero-shot-phase2

# 最終実装のコマンド（RTX 6000 Ada 48GB x1）
uv run python -m piper_train \
  --dataset-dir /home/shadeform/data/piper/dataset-moe-speech-20speakers \
  --prosody-dim 16 \
  --accelerator gpu --devices 1 --precision bf16-mixed \
  --max_epochs 200 --batch-size 160 --samples-per-speaker 8 \
  --checkpoint-epochs 2 --quality medium \
  --base_lr 2e-4 --ema-decay 0.9995 --num-workers 8 \
  --no-wavlm \
  --default_root_dir /home/shadeform/data/piper/output-moe-speech-20speakers
```

### 10.6 推論コマンド例

> **最終実装 (2026-03-10)**: `--ref-audio` は未実装。事前計算済み `.npy` ファイルを `--speaker-embedding` で指定する方式。

```bash
# Zero-shot: 事前計算済みembeddingで推論
CUDA_VISIBLE_DEVICES="" uv run python -m piper_train.infer_onnx \
  --model /path/to/zero_shot_model.onnx \
  --config /path/to/config.json \
  --output-dir /path/to/output \
  --text "こんにちは、今日は良い天気ですね。" \
  --speaker-embedding /path/to/speaker.npy

# 従来のspeaker-idモード（完全互換）
CUDA_VISIBLE_DEVICES="" uv run python -m piper_train.infer_onnx \
  --model /path/to/sid_model.onnx \
  --config /path/to/config.json \
  --output-dir /path/to/output \
  --text "こんにちは、今日は良い天気ですね。" \
  --speaker-id 0
```

### 10.7 テスト計画

| テスト | 内容 | 合格基準 |
|--------|------|---------|
| 既存機能regression | 全既存テスト通過、speaker-idモードの推論品質維持 | テスト全パス |
| Speaker Encoder単体 | 出力形状、同一話者の類似度 | cosine similarity > 0.8 |
| Zero-shot integration | speaker_embedding→生成音声の話者類似度 | SECS > 0.60 |
| ONNX export/import | `--export-mode sid` / `--export-mode zero-shot` 両方 | エラーなし |
| Checkpoint migration | strict=Falseロード成功 | 既存モデル読み込み可 |
| CPU推論 | CUDA_VISIBLE_DEVICES="" で動作 | 推論成功 |

---

## 11. 推論速度ゼロインパクト設計（追加調査）

> 追加調査 (2026-03-07): Piperの「軽量・高速推論」要件を守りつつzero-shotを実現するための設計。推論速度を一切落とさないことが必須要件。

### 11.1 現在の推論パイプラインの速度特性

#### エンドツーエンド推論時間（100文字、~5秒音声）

| ステップ | 処理時間 (CPU) | 割合 |
|---------|--------------|------|
| Phonemize | 10-50ms | ~10% |
| TextEncoder | 30-50ms | ~15% |
| Duration Predictor | 10-20ms | ~5% |
| Flow (逆向き) | 20-40ms | ~10% |
| **Generator (Decoder)** | **80-150ms** | **~50%** |
| WAV書き出し | 5-10ms | ~3% |
| **合計** | **150-310ms** | RTF ≈ 0.03-0.06 |

**ボトルネック**: Generator（256倍アップサンプリング + ResBlock×9）が支配的。

#### Speaker ID処理のコスト

現在の `emb_g(sid)` (nn.Embedding lookup) はONNX上で **Gather** オペレーションに変換される。このコストは**マイクロ秒オーダー**であり、全体の推論時間（ミリ秒〜秒オーダー）に対して完全に無視できる。

```
sid (int64) → [Gather: embedding_table] → g (float32, [1, gin_channels, 1])
                  ↑ 数μs（無視できる）
```

### 11.2 推論速度ゼロインパクトの設計原則

**核心的洞察**: Speaker Encoderの計算コストが問題になるのは「推論時に毎回実行する場合」のみ。Speaker Embeddingを事前計算してファイルに保存し、推論時にはファイルから読み込むだけなら、TTSモデル本体の推論は現在と**完全に同一速度**になる。

```
┌─────────────────────────────────┐   ┌─────────────────────────────────┐
│  オフライン（1回のみ）            │   │  オンライン推論（毎回）           │
│                                 │   │                                 │
│  参照音声                        │   │  テキスト                        │
│    ↓                            │   │    ↓                            │
│  [Speaker Encoder ONNX]         │   │  [Phonemize]                    │
│    ↓                            │   │    ↓                            │
│  embedding.npy (保存)     ─────────→  [Piper TTS ONNX] ← embedding   │
│                                 │   │    ↓                            │
│  ※推論パイプラインに含まれない     │   │  音声出力                        │
│  ※数十ms、1回だけ               │   │  ※現在と完全に同一速度            │
└─────────────────────────────────┘   └─────────────────────────────────┘
```

### 11.3 3つの設計パターン比較

| パターン | 推論速度への影響 | 実装複雑度 | 新規話者追加 | 既存互換性 |
|---------|----------------|-----------|-------------|-----------|
| **A: Encoder内蔵（毎回実行）** | **-20〜50% 低下** | 中 | リアルタイム | 維持 |
| **B: 別モデル + キャッシュ** | 初回のみ50-110ms | 中 | 初回のみ遅い | 維持 |
| **C: 事前計算embedding** | **ゼロ** | **低** | オフライン登録 | **維持** |

#### パターンA: Speaker Encoderをモデル内蔵（非推奨）

Speaker Encoderを毎回実行する方式。**推論速度が20-50%低下するため、Piperの要件に反する。**

| Speaker Encoder | 追加レイテンシ (CPU) | 推論速度低下率 |
|----------------|---------------------|--------------|
| ECAPA-TDNN | ~70ms | ~30% |
| CAM++ | ~30-40ms | ~15% |
| TitaNet | ~110ms | ~45% |
| WavLM fine-tuned | 数百ms | >100% |

#### パターンB: 別モデル + メモリキャッシュ

Speaker Encoderを別ONNXモデルとし、初回のみ実行してメモリにキャッシュ。2回目以降はキャッシュから読み込み。

- **初回**: Speaker Encoder実行 (50-110ms) → キャッシュ
- **2回目以降**: キャッシュ利用 (0ms追加)
- **採用例**: XTTS v2, CosyVoice

#### パターンC: 事前計算embedding（推奨）

Speaker Embeddingを完全にオフラインで事前計算し、`.npy`/`.json`ファイルとして保存。推論時はファイル読み込みのみ（<1ms）。

- **採用例**: YourTTS (`compute_embeddings.py`), OpenVoice (`se.pth`), CosyVoice (`spk2embedding.pt`), SpeechT5
- **TTSモデルのONNX推論は現在と完全に同一速度**

### 11.4 推奨設計: Embedding置換方式 + 事前計算パイプライン

#### ONNX入力の変更

```python
# 現在のONNXグラフ
sid (int64) → [Gather: emb_g.weight] → g (float32, [1, gin_channels, 1])

# 変更後のONNXグラフ（最終実装）
speaker_embedding (float32, [1, 192]) → [Linear: spk_proj(192→512)] → [Unsqueeze] → g (float32, [1, 512, 1])
```

変わるのは入力付近のGather → Linear+Unsqueezeへの変更のみ。Linear(192→512)の演算コストは数μsであり、計算量の99.9%以上を占めるEncoder/Flow/Decoder部分は**完全に同一**。

> **注**: gin_channelsは`lightning.py`でマルチスピーカーまたはzero-shotモデルの場合 **512** に自動設定される（`gin_channels <= 0` の場合）。768ではガビガビ音が発生したため512を採用。CAM++の出力(192次元)は`nn.Linear(192, 512)`でONNXグラフ内で射影される。

#### 速度同一性の根拠

1. **Gatherオペレーション（現在）のコスト**: テーブルから1行取得 = O(gin_channels)のメモリコピー = 数μs
2. **Linear+Unsqueezeオペレーション（変更後）のコスト**: 192*512の行列乗算+reshape = 数μs
3. **両者の差**: 実質ゼロ（VITS全体の推論時間の0.001%未満）
4. **実証**: Coqui TTS (YourTTS) が `nn.Embedding` (speaker_id方式) と `d_vector` (外部embedding方式) の両方をサポートし、推論速度に有意差がないことを確認済み

#### export_onnx.py の変更

> **最終実装 (2026-03-10)**: `--export-mode {auto, sid, zero-shot}` フラグで出力モデルを分離。`auto`はモデルの`use_zero_shot`属性から自動判定。

```python
# 調査段階の設計案
if speaker_embedding is not None:
    g = speaker_embedding.unsqueeze(-1)
elif model_g.n_speakers > 1 and sid is not None:
    g = model_g.emb_g(sid).unsqueeze(-1)

# 最終実装（export_onnx.py の infer_forward 関数内）
if use_zero_shot:
    g = model_g.spk_proj(speaker_embedding).unsqueeze(-1)  # [b, 512, 1]
elif model_g.n_speakers > 1 and sid is not None:
    g = model_g.emb_g(sid).unsqueeze(-1)  # [b, 512, 1]
```

#### infer_onnx.py の変更

> **最終実装 (2026-03-10)**: `--speaker-embedding` オプションで .npy ファイルを指定。`--ref-audio` は未実装（オフラインで `extract_speaker_embedding.py` を使用する設計）。

```python
# 最終実装のCLI引数
parser.add_argument("--speaker-embedding", help="事前計算済みembedding (.npy)")

# 推論時
if args.speaker_embedding:
    # 事前計算済みembeddingをロード（<1ms）
    spk_emb = np.load(args.speaker_embedding).astype(np.float32)
    inputs["speaker_embedding"] = spk_emb.reshape(1, -1)
elif args.speaker_id is not None:
    inputs["sid"] = np.array([args.speaker_id], dtype=np.int64)
```

### 11.5 Speaker Encoder選定（オフライン処理用）

推論時には実行しないため、速度よりも**精度とライセンス**を重視して選定。

#### 定量比較

| モデル | パラメータ数 | ONNX Size | EER (VoxCeleb1-O) | CPU推論 (5秒音声) | ライセンス |
|--------|------------|-----------|-------------------|------------------|-----------|
| **CAM++** | 7.2M | **28MB** | **0.73%** | ~30-40ms | **Apache-2.0** |
| ECAPA-TDNN (C512) | 6.2M | ~25MB | ~1.0% | ~40-50ms | Apache-2.0 |
| ECAPA-TDNN (C1024) | 14.7M | ~83MB | ~0.80% | ~70ms | Apache-2.0 |
| LE-CAM++ | 6.6M | ~26MB | 0.69% | <30ms | Apache-2.0 |
| ERes2NetV2 | 17.8M | ~71MB | **0.61%** | ~60ms | Apache-2.0 |
| ECAPA2 | ~16M | ~64MB | **0.58%** | ~70ms | Apache-2.0 |
| d-vector (GE2E) | 2.4M | ~17MB | ~7-10% | ~10-20ms | MIT |
| TitaNet-S | 6.4M | ~25MB | ~0.82% | ~30-40ms | CC-BY-4.0 |
| WavLM Base+ (SV) | 94.7M | ~360MB | ~1.84% | 数百ms | MIT |

#### 推奨: CAM++ (Apache-2.0)

**推奨理由**:
1. **精度/サイズの最良バランス**: EER 0.73%でECAPA-TDNNより高精度、かつパラメータ数はほぼ同等
2. **ONNX対応済み**: WeSpeaker / 3D-Speaker / sherpa-onnx で事前学習済みONNXモデルが利用可能
3. **INT8量子化で~8MB**: 配布サイズを大幅削減可能
4. **Apache-2.0**: PiperのGPL-free方針と完全適合
5. **CosyVoiceでの実績**: Alibaba CosyVoiceが標準Speaker Encoderとして採用

**入手先**:
- WeSpeaker: `wespeaker/wespeaker-models` (HuggingFace)
- 3D-Speaker: `iic/speech_campplus_sv_zh-cn_16k-common` (ModelScope)
- sherpa-onnx: ONNX + INT8量子化済みモデル提供
- **最終採用**: CosyVoice-300Mリポジトリの `campplus.onnx` (27MB, Apache-2.0)
  ```bash
  wget -q "https://huggingface.co/model-scope/CosyVoice-300M/resolve/main/campplus.onnx" \
    -O /home/shadeform/data/piper/models/campplus.onnx
  ```

#### 次点: ECAPA-TDNN (SpeechBrain, Apache-2.0)

前回調査で推奨したモデル。CAM++と比較して精度はやや劣るが、SpeechBrainエコシステムとの統合が容易。

### 11.6 事前計算ツール設計

> **実装済み (2026-03-10)**: `extract_speaker_embedding.py` として実装。per-utteranceモードではDataLoader + バッチONNX推論で高速化。

```bash
# 新規話者のembedding抽出（オフライン、1回のみ）
uv run python -m piper_train.extract_speaker_embedding \
  --encoder /home/shadeform/data/piper/models/campplus.onnx \
  --audio /path/to/reference_voice.wav \
  --output /path/to/speaker_embedding.npy

# Per-utterance embedding抽出（学習用・推奨）
uv run python -m piper_train.extract_speaker_embedding \
  --encoder /home/shadeform/data/piper/models/campplus.onnx \
  --dataset-dir /home/shadeform/data/piper/dataset-moe-speech-20speakers \
  --per-utterance --batch-size 64 --num-workers 12

# Per-speaker embedding抽出（推論用reference）
uv run python -m piper_train.extract_speaker_embedding \
  --encoder /home/shadeform/data/piper/models/campplus.onnx \
  --dataset-dir /home/shadeform/data/piper/dataset-moe-speech-20speakers \
  --output-dir /path/to/embeddings
```

**出力ファイル構成（per-utterance）**:
```
dataset-dir/
├── speaker_embeddings/
│   ├── {audio_stem_1}.npy    # float32[192] (896バイト)
│   ├── {audio_stem_2}.npy
│   └── ...
├── dataset.jsonl             # speaker_embedding_path が自動追加される
└── dataset.jsonl.bak         # バックアップ
```

**出力ファイル構成（per-speaker）**:
```
embeddings/
├── speaker_0.npy    # float32[192]
├── speaker_1.npy
├── ...
└── speaker_19.npy
```

### 11.7 既存話者の移行パス

> **最終実装 (2026-03-10)**: Dual-Modeモデルでは `emb_g` と `spk_proj` が共存するため、既存のspeaker IDモードはそのまま使える。zero-shot推論用には `--export-mode zero-shot` でONNXを別途出力する。

既存のspeaker_idモデルからの移行は以下の手順で実現:

1. 学習済みモデルの学習データから各話者のCAM++ embeddingを抽出（`extract_speaker_embedding.py`）
2. `.npy` ファイルとして保存
3. `--export-mode zero-shot` で新ONNXモデルをエクスポート
4. `--speaker-embedding speaker_X.npy` で推論

```bash
# 既存データセットからCAM++ embeddingを抽出
uv run python -m piper_train.extract_speaker_embedding \
  --encoder /home/shadeform/data/piper/models/campplus.onnx \
  --dataset-dir /path/to/dataset --output-dir /path/to/embeddings

# zero-shot用ONNXモデルをエクスポート
CUDA_VISIBLE_DEVICES="" uv run python -m piper_train.export_onnx \
  --export-mode zero-shot /path/to/checkpoint.ckpt /path/to/output_zs.onnx
```

### 11.8 OSS実装の事前計算方式の採用実績

| プロジェクト | 方式 | Embedding保存形式 | 詳細 |
|-------------|------|-----------------|------|
| **YourTTS** (Coqui TTS) | `compute_embeddings.py` で事前計算 | JSON dict (d-vector 256dim) | `d_vector_file.json`に全話者を格納。推論時はEncoder不要 |
| **CosyVoice** | `extract_embedding.py` + `add_zero_shot_spk()` | `spk2embedding.pt` | 事前計算後、speaker IDで参照可能。`save_spkinfo()`で永続化 |
| **XTTS v2** | `get_conditioning_latents()` でキャッシュ | メモリ/ファイル | `gpt_cond_latent` + `speaker_embedding` を事前計算して再利用 |
| **OpenVoice v2** | `se_extractor.get_se()` | `.pth`ファイル | `checkpoints_v2/ses/`に事前保存。自動キャッシュ機能あり |
| **SpeechT5** | x-vector事前計算 | `.npy`ファイル | `np.save("speaker.npy", embedding)`で保存、推論時にロード |

### 11.9 推論速度への影響まとめ

| 処理 | 現在 | zero-shot導入後 | 差分 |
|------|------|----------------|------|
| Phonemize | 10-50ms | 10-50ms | **±0** |
| Embedding取得 | ~0μs (Gather) | **<1ms (ファイルロード)** | **+<1ms** |
| TextEncoder | 30-50ms | 30-50ms | **±0** |
| Duration Predictor | 10-20ms | 10-20ms | **±0** |
| Flow | 20-40ms | 20-40ms | **±0** |
| Generator | 80-150ms | 80-150ms | **±0** |
| **合計** | **150-310ms** | **150-311ms** | **<1ms増（実質ゼロ）** |

### 11.10 更新された推奨ロードマップ

| フェーズ | 内容 | 推論速度影響 | 工数 | 状態 |
|---------|------|------------|------|------|
| **Phase 1a** | 事前計算embedding方式のONNX対応 | **ゼロ** | 2-3日 | **完了** |
| **Phase 1b** | CAM++ Speaker Encoder統合（オフラインツール） | なし（推論時不使用） | 1-2日 | **完了** |
| **Phase 1c** | TTS本体の学習（Speaker Embedding条件付け） | **ゼロ** | 200 epochs | **完了** |
| Phase 2 | TextEncoder Speaker Conditioning | **ゼロ** | 1-2日 + 再学習 | 未着手 |
| Phase 3 | MB-iSTFT-VITSデコーダ高速化 | **推論4倍高速化** | 3-5日 + 再学習 | 未着手 |

---

## 12. エグゼクティブサマリー更新

> 追加調査 (2026-03-07): 推論速度ゼロインパクトの制約を加えた最終推奨。

### 最終推奨アプローチ

**事前計算Embedding方式 + CAM++ Speaker Encoder (Apache-2.0)**

Piperの「軽量・高速推論」要件を完全に維持しつつ、zero-shot話者再現を実現する。

#### 設計の核心

1. **Speaker Encoderは推論パイプラインに含めない**: オフラインツールとして分離
2. **事前計算したembeddingをファイルから読み込み**: 推論時の追加コスト <1ms
3. **ONNXモデルの変更は入力の型変更のみ**: `sid (int64)` → `speaker_embedding (float32[192])`（`spk_proj`で512次元に射影）
4. **計算グラフの99.9%以上は完全に同一**: 推論速度はゼロインパクト
5. **gin_channels=512**: 768はガビガビ音が発生、256は表現力不足

#### 前回推奨からの変更点

| 項目 | 前回推奨 | 最終実装 (2026-03-10) |
|------|---------|----------------|
| Speaker Encoder | ECAPA-TDNN (SpeechBrain) | **CAM++ ONNX** (27MB, 192次元, Apache-2.0) |
| 推論時のEncoder実行 | あり（初回のみ） | **なし（完全にオフライン分離: `extract_speaker_embedding.py`）** |
| gin_channels | 768 | **512**（768はガビガビ音発生のため不採用） |
| アーキテクチャ | SpeakerEncoder class内蔵 | **Dual-Mode: `emb_g` + `spk_proj` (nn.Linear(192, 512))** |
| ONNX設計 | ref_mel入力 + モデル内Encoder | **`--export-mode {sid, zero-shot}` で分離出力** |
| 学習フラグ | `--use-speaker-encoder` | **不要（`use_zero_shot=True` デフォルト）** |
| 学習環境 | L4 16GB x4 | **RTX 6000 Ada 48GB x1, batch_size=160, bf16-mixed** |
| Per-utterance embedding | 未計画 | **実装済み（DataLoader + バッチONNX推論で高速化）** |
| 推論速度への影響 | 初回~50ms追加 | **実質ゼロ（<1ms）** |

---

## 参考文献

- YourTTS: https://arxiv.org/abs/2112.02418
- VALL-E / VALL-E 2: https://arxiv.org/abs/2301.02111
- StyleTTS 2: https://arxiv.org/abs/2306.07691
- F5-TTS: https://arxiv.org/abs/2410.06885
- ECAPA-TDNN: https://arxiv.org/abs/2005.07143
- VITS: https://arxiv.org/abs/2106.06103
- VITS2: https://arxiv.org/abs/2307.16430
- DINO-VITS: https://arxiv.org/html/2311.09770v3
- NaturalSpeech 3: https://arxiv.org/abs/2403.03100
- MaskGCT: https://arxiv.org/abs/2409.00750
- MegaTTS 3: https://github.com/bytedance/MegaTTS3
- Seed-TTS: https://arxiv.org/abs/2406.02430
- CosyVoice: https://github.com/FunAudioLLM/CosyVoice
- GPT-SoVITS: https://github.com/RVC-Boss/GPT-SoVITS
- SpeechBrain: https://github.com/speechbrain/speechbrain
- MB-iSTFT-VITS: https://github.com/MasayaKawamura/MB-iSTFT-VITS
- JVS Corpus: https://arxiv.org/abs/1908.06248
- LibriTTS-R: http://www.openslr.org/141/
- ZS-TTS-Evaluation: https://github.com/Edresson/ZS-TTS-Evaluation
- Wespeaker ECAPA-TDNN: https://huggingface.co/Wespeaker/wespeaker-ecapa-tdnn512-LM
- CAM++: https://arxiv.org/abs/2303.00332
- ECAPA2: https://arxiv.org/abs/2401.08342
- ERes2NetV2: https://arxiv.org/abs/2406.02167
- WeSpeaker: https://github.com/wenet-e2e/wespeaker
- 3D-Speaker: https://github.com/modelscope/3D-Speaker
- sherpa-onnx: https://github.com/k2-fsa/sherpa-onnx
- Resemblyzer: https://github.com/resemble-ai/Resemblyzer
- TitaNet: https://arxiv.org/abs/2110.04410
- VI-Speaker: https://github.com/PlayVoice/VI-Speaker
- OpenVoice: https://github.com/myshell-ai/OpenVoice
- SV2TTS: https://github.com/CorentinJ/Real-Time-Voice-Cloning
- Coqui TTS (VITS): https://github.com/coqui-ai/TTS
- Qwen3-TTS: https://github.com/QwenLM/Qwen3-TTS
- VoxCPM: https://github.com/OpenBMB/VoxCPM
- GLM-TTS: https://github.com/zai-org/GLM-TTS
- MegaTTS 3: https://github.com/bytedance/MegaTTS3
- Kokoro TTS: https://github.com/hexgrad/kokoro
- OpenAudio S1 (旧Fish Speech): https://github.com/fishaudio/fish-speech
- Chatterbox: https://github.com/resemble-ai/chatterbox
- IndexTTS 2.5: https://github.com/index-tts/index-tts
- LE-CAM++: https://ieeexplore.ieee.org/document/10800177/
