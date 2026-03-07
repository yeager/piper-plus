# Zero-Shot 話者再現 TTS 技術調査レポート

**調査日**: 2026-03-07
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

---

## 1. エグゼクティブサマリー

### 結論

Piperにzero-shot話者再現を導入するための推奨アプローチは **ECAPA-TDNN Speaker Encoder + DINO Loss + Speaker Consistency Loss** による高精度統合方式である。既存の話者IDモードとの完全な後方互換性を維持しつつ、VITSアーキテクチャ内で到達可能な最高品質 (SIM-O ~0.65) を目指す。

- 既存VITSの `gin_channels` メカニズムをそのまま流用（dual-mode: speaker ID / speaker embedding）
- Speaker Encoderには **ECAPA-TDNN** (Apache-2.0) を採用、DINO LossとSpeaker Consistency Lossで精度最大化
- 学習データ: LibriTTS-R (585h, 2,456話者) + JVS (30h, 100話者) + 現行20話者 = **~615h, ~2,576話者**
- 3フェーズ学習: Speaker Encoder事前学習済み → TTS学習 (encoder凍結) → Joint Training (encoder解凍)
- 推定学習時間: 約180-240時間 (L4 GPU x4)

### VITSの品質上限と将来展望

定量評価による調査の結果、VITSアーキテクチャのzero-shot話者類似度の上限は **SIM-O ~0.65** である。最新のFlow Matching/Diffusion系モデル (CosyVoice 3: SIM 77.4%, MaskGCT: SIM-O 0.687) との差は約0.15-0.20ポイント存在する。将来的にさらなる品質向上を目指す場合は、アーキテクチャの刷新（セクション7で詳述）を検討する。

### 推奨ロードマップ

| フェーズ | 内容 | 工数 |
|---------|------|------|
| Phase 1 | ECAPA-TDNN Speaker Encoder + DINO/SCL Loss 導入 | 5-7日 + 学習180-240h |
| Phase 2 | TextEncoder Speaker Conditioning追加 (VITS2方式) | 1-2日 + 再学習 |
| Phase 3 | MB-iSTFT-VITS デコーダ高速化 (オプション) | 3-5日 + 再学習 |
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

### 2.4 日本語対応状況

| モデル | 日本語対応 | 詳細 |
|--------|-----------|------|
| VALL-E X | 対応 | 英中日の3言語 |
| XTTS v2 | 対応 | 16言語の1つ |
| StyleTTS 2 | 英語のみ | 学習にはphonemizerが必要 |
| VoiceBox | 非対応 | 英仏西葡独波 |
| GPT-SoVITS | **ネイティブ対応** | v3で日本語改善、コミュニティ活発 |
| Fish Speech | **ネイティブ対応** | 日本語10万時間超で学習 |
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

#### ECAPA-TDNN（推奨）
- x-vectorの改良版。Squeeze-and-Excitation, Res2Net, Attentive Statistical Poolingを導入
- 192次元で軽量ながら高精度 (EER 0.86%)
- **非英語話者（日本語含む）でトップクラスの識別性能** (EER 1.71%)
- SpeechBrain (Apache 2.0), NVIDIA NeMo (Apache 2.0) で事前学習済みモデル利用可能

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

### 4.3 YourTTS方式の詳細（推奨）

VITSをベースに、話者ID embeddingを外部Speaker Encoderのd-vector/ECAPA embeddingに置換する方法。

**コード変更の核心:**
```python
# 変更前 (models.py)
if n_speakers > 1:
    self.emb_g = nn.Embedding(n_speakers, gin_channels)
g = self.emb_g(sid).unsqueeze(-1)  # [b, gin_channels, 1]

# 変更後
self.speaker_encoder = SpeakerEncoder(output_dim=gin_channels)
g = self.speaker_encoder(ref_audio).unsqueeze(-1)  # [b, gin_channels, 1]
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
- ECAPA-TDNN (SpeechBrain): 高品質、Apache-2.0 — **推奨**
- Resemblyzer: 軽量、MIT
- WavLM (既にPiperに統合済み) を流用: 追加依存なし

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

### 6.1 段階的導入ロードマップ（精度最優先版）

#### Phase 1: ECAPA-TDNN Speaker Encoder + DINO/SCL Loss（最優先）

**概要**: Speaker Encoder (ECAPA-TDNN) を導入し、DINO LossとSpeaker Consistency Lossで精度を最大化

**技術選定:**
- Speaker Encoder: **ECAPA-TDNN** (事前学習済み SpeechBrain, Apache-2.0)
- 損失関数: 既存VITS損失 + Speaker Consistency Loss + DINO Loss + WavLM Perceptual Loss
- Dual-mode: `--speaker-id` (既存) と `--ref-audio` (zero-shot) の共存
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

**test-en (英語):**

| モデル | WER (%) | SIM (%) |
|--------|---------|---------|
| CosyVoice 3 (1.5B RL) | **1.45** | **77.4** |
| Seed-TTS | 2.25 | 76.2 |
| F5-TTS | 2.00 | 67.0 |
| MaskGCT | 2.62 | 71.7 |
| CosyVoice 2 | 3.09 | 65.9 |
| CosyVoice 1 | 4.29 | 60.9 |

**test-zh (中国語):**

| モデル | CER (%) | SIM (%) |
|--------|---------|---------|
| CosyVoice 3 (1.5B RL) | **0.71** | N/A |
| CosyVoice 3 (0.5B) | 0.81 | **77.4** |
| MaskGCT | 2.27 | 77.4 |

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
| **VITSの理論上限** | **~0.65** | **~5** | - |

#### VITSの根本的制約
- Speaker Embeddingとprosodyが結合 → 未知話者への汎化が弱い
- GAN学習の不安定性 → スケーリングが困難
- Flow/Diffusionに比べて表現力の天井が低い

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

| 候補 | 総合品質 | 話者類似度 | OSS | 日本語 |
|------|---------|----------|-----|--------|
| **CosyVoice 2/3** | S | S | 完全OSS (Apache-2.0) | 対応 |
| **MaskGCT** | S | A+ | 完全OSS (CC-BY-NC) | 学習データ次第 |
| **F5-TTS** | A+ | A | 完全OSS (MIT) | 学習データ次第 |
| GPT-SoVITS v3 | A | A+ | 完全OSS (MIT) | **最良** |

---

## 8. 既存機能との互換性設計

> 追加調査 (2026-03-07): 既存の全機能を維持しながらzero-shotを追加するための設計パターン。

### 8.1 Dual-mode Speaker Conditioning

既存の `emb_g` (speaker ID) と新規の Speaker Encoder (zero-shot) を共存させる設計:

```python
class SynthesizerTrn(nn.Module):
    def __init__(self, ..., use_speaker_encoder=False, speaker_embed_dim=192):
        ...
        # 既存: speaker-id mode (常に保持)
        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)

        # 新規: zero-shot mode
        if use_speaker_encoder:
            self.speaker_encoder = SpeakerEncoder(mel_channels=80, gin_channels=gin_channels)

    def _get_speaker_embedding(self, sid=None, ref_mel=None, ref_mel_lengths=None):
        """Dual-mode: speaker_id または参照音声から g を生成"""
        if ref_mel is not None and hasattr(self, 'speaker_encoder'):
            g = self.speaker_encoder(ref_mel, ref_mel_lengths).unsqueeze(-1)
        elif sid is not None and self.n_speakers > 1:
            g = self.emb_g(sid).unsqueeze(-1)
        else:
            g = None
        return g  # [batch, gin_channels, 1] — 下流モジュールへの共通インターフェース
```

**重要**: `g` の形状 `[batch, gin_channels, 1]` が共通なので、下流の4コンポーネント（Duration Predictor, Posterior Encoder, Flow, Decoder）は一切変更不要。

### 8.2 ONNX推論でのdual-mode対応

1モデル統合 + Optional入力方式を推奨:

```python
# export_onnx.py: 入力にref_melを追加
input_names = ["input", "input_lengths", "scales"]
if num_speakers > 1:
    input_names.append("sid")
if use_speaker_encoder:
    input_names.append("ref_mel")  # optional入力
if has_prosody:
    input_names.append("prosody_features")
```

```python
# infer_onnx.py: 推論時の切り替え
if args.ref_audio:
    # zero-shot mode: mel-spectrogramを計算してTTSモデルに渡す
    ref_mel = compute_mel_from_wav(args.ref_audio)
    inputs["ref_mel"] = ref_mel
elif args.speaker_id is not None:
    # speaker-id mode: 従来通り
    inputs["sid"] = np.array([args.speaker_id], dtype=np.int64)
```

### 8.3 Speaker Encoder分離設計

Speaker Encoderを別ONNXモデルとして分離し、embeddingキャッシュに対応:

```
推論パイプライン:
  ref_audio → [Speaker Encoder ONNX] → spk_emb → キャッシュ保存
                                          ↓
  text → [Piper TTS ONNX (+ spk_emb)] → audio
```

**embeddingキャッシュの利点:**
- 同一話者の2回目以降の推論でSpeaker Encoder不要
- CPU推論時のレイテンシ削減（初回~50ms → 2回目以降0ms）
- 事前登録話者を `--speaker-id` 感覚で呼び出し可能

### 8.4 チェックポイント互換性

```python
# 既存チェックポイントからの移行
model = VitsModel.load_from_checkpoint(ckpt_path, strict=False)
# strict=False により speaker_encoder 部分がない場合でもロード可能
# speaker_encoder はランダム初期化 → Phase 2以降で学習
```

**段階的移行パス:**
1. `strict=False` で既存チェックポイントをロード → `emb_g` はそのまま使える
2. `speaker_encoder` 部分のみを学習（`emb_g` は凍結）
3. 完全なdual-mode対応モデルが完成

### 8.5 維持される既存機能一覧

| 機能 | 互換性 | 備考 |
|------|--------|------|
| `--speaker-id` による話者指定 | 完全互換 | `emb_g` をそのまま保持 |
| 日本語/英語 Phonemizer | 影響なし | 独立モジュール |
| Prosody Features (A1/A2/A3) | 影響なし | Duration Predictorへの入力は変更なし |
| ONNX推論 | 互換 | optional入力追加のみ |
| CPU推論 | 互換 | Speaker Encoder ~50ms追加 (初回のみ) |
| WavLM Discriminator | 互換 | 学習時のみ、独立 |
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

#### Phase 1: Speaker Encoder準備（学習時間ゼロ）
- **方法**: SpeechBrain事前学習済みECAPA-TDNNを採用
- **モデル**: `spkrec-ecapa-voxceleb` (Apache-2.0)
- 学習不要、ダウンロードのみ

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

### 9.3 損失関数の組み合わせ

| # | 損失関数 | 重み | 効果 | 状態 |
|---|---------|------|------|------|
| 1 | VITS損失 (reconstruct + KL + adversarial) | 既存 | 基本音質 | 実装済み |
| 2 | **Speaker Consistency Loss** | c_spk=1.0 | 話者再現精度 | **新規** |
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

| コンポーネント | メモリ | 備考 |
|--------------|-------|------|
| VITS本体 | ~8GB | 現行と同等 |
| Speaker Encoder (ECAPA-TDNN) | ~1-2GB | 学習時/推論時 |
| WavLM Discriminator | ~1-2GB | 学習時のみ |
| データ/勾配 | ~3-4GB | batch_size依存 |
| **合計** | **~13-16GB** | ギリギリ収まる |

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

### 10.1 新規ファイル

#### `src/python/piper_train/vits/speaker_encoder.py` (~200行)

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

```python
# __init__: speaker encoder対応追加
def __init__(self, ..., use_speaker_encoder=False):
    ...
    if n_speakers > 1:
        self.emb_g = nn.Embedding(n_speakers, gin_channels)  # 既存維持
    self.use_speaker_encoder = use_speaker_encoder
    if use_speaker_encoder:
        from .speaker_encoder import SpeakerEncoder
        self.speaker_encoder = SpeakerEncoder(
            mel_channels=80, gin_channels=gin_channels)

# forward/infer: dual-mode対応
def forward(self, x, x_lengths, y, y_lengths, sid=None,
            prosody_features=None,
            ref_mel=None, ref_mel_lengths=None):  # NEW
    ...
    if self.use_speaker_encoder and ref_mel is not None:
        g = self.speaker_encoder(ref_mel, ref_mel_lengths).unsqueeze(-1)
    elif self.n_speakers > 1 and sid is not None:
        g = self.emb_g(sid).unsqueeze(-1)
    else:
        g = None
    # 以降の処理は全く同じ（gの形状が[b, gin_channels, 1]で共通）
```

#### `lightning.py` — 学習ループ (~40行追加)

```python
# training_step_g: ref_mel生成 + Speaker Consistency Loss
def training_step_g(self, batch):
    ...
    # Self-reconstruction: 同一発話のmelを参照として使用
    ref_mel = spec_to_mel_torch(spec, ...)

    (y_hat, l_length, ...) = self.model_g(
        x, x_lengths, spec, spec_lengths, speaker_ids,
        prosody_features=prosody_features,
        ref_mel=ref_mel, ref_mel_lengths=spec_lengths)

    # Speaker Consistency Loss
    if self.model_g.use_speaker_encoder:
        gen_mel = mel_spectrogram_torch(y_hat.squeeze(1), ...)
        ref_emb = self.model_g.speaker_encoder(ref_mel, spec_lengths)
        gen_emb = self.model_g.speaker_encoder(gen_mel)
        loss_spk = F.cosine_embedding_loss(
            ref_emb, gen_emb,
            torch.ones(ref_emb.size(0), device=ref_emb.device))
        loss_gen_all += loss_spk * self.hparams.c_spk
```

#### `export_onnx.py` (~30行追加)

```python
# ref_mel入力の追加
if model_g.use_speaker_encoder:
    input_names.append("ref_mel")
    dynamic_axes["ref_mel"] = {0: "batch_size", 2: "ref_time"}
```

#### `infer_onnx.py` (~50行追加)

```python
# 新しいCLI引数
parser.add_argument("--ref-audio", help="Reference WAV for zero-shot cloning")
parser.add_argument("--speaker-encoder", help="Speaker encoder ONNX (optional)")
parser.add_argument("--speaker-embeddings", help="Cached embeddings JSON (optional)")

# ref_audio処理
if args.ref_audio:
    ref_wav, sr = librosa.load(args.ref_audio, sr=22050)
    ref_mel = compute_mel_from_wav(ref_wav)
    inputs["ref_mel"] = ref_mel
```

#### `__main__.py` (~15行追加)

```python
parser.add_argument("--use-speaker-encoder", action="store_true")
parser.add_argument("--c-spk", type=float, default=1.0)
parser.add_argument("--c-dino", type=float, default=0.1)
parser.add_argument("--speaker-encoder-pretrained", type=str, default=None)
```

### 10.3 変更対象ファイル一覧

| ファイル | 変更種別 | 変更規模 |
|---------|---------|---------|
| `vits/speaker_encoder.py` | **新規** | ~200行 |
| `vits/models.py` | 修正 | ~30行追加 |
| `vits/lightning.py` | 修正 | ~40行追加 |
| `vits/dataset.py` | 修正 | ~20行追加 |
| `export_onnx.py` | 修正 | ~30行追加 |
| `infer_onnx.py` | 修正 | ~50行追加 |
| `__main__.py` | 修正 | ~15行追加 |
| `vits/losses.py` | 修正 | ~10行追加 |
| `test/test_speaker_encoder.py` | **新規** | ~100行 |
| `test/test_zero_shot_integration.py` | **新規** | ~150行 |

### 10.4 パラメータ数の増加

| コンポーネント | パラメータ数 | ONNXサイズ増加 |
|--------------|------------|---------------|
| ECAPA-TDNN Speaker Encoder | ~6-8M | +25-30MB |
| 推論速度への影響 | Speaker Encoder forward 1回 | ~5ms (CPU), ~1ms (GPU) |

### 10.5 学習コマンド例

```bash
# Phase 2: Speaker Encoder凍結、TTS本体学習
NCCL_DEBUG=WARN NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
uv run python -m piper_train \
  --dataset-dir /data/piper/dataset-multispeaker-merged \
  --prosody-dim 16 \
  --use-speaker-encoder \
  --c-spk 1.0 \
  --accelerator gpu --devices 4 --precision 16-mixed \
  --max_epochs 200 --batch-size 10 --samples-per-speaker 2 \
  --checkpoint-epochs 1 --quality medium \
  --base_lr 2e-4 --disable_auto_lr_scaling \
  --ema-decay 0.9995 --num-workers 0 --no-pin-memory \
  --default_root_dir /data/piper/output-zero-shot-phase2
```

### 10.6 推論コマンド例

```bash
# Zero-shot: 参照音声から話者クローン
CUDA_VISIBLE_DEVICES="" uv run python -m piper_train.infer_onnx \
  --model /path/to/model-zeroshot.onnx \
  --config /path/to/config.json \
  --output-dir /path/to/output \
  --text "こんにちは、今日は良い天気ですね。" \
  --ref-audio /path/to/reference_voice.wav

# 従来のspeaker-idモード（完全互換）
CUDA_VISIBLE_DEVICES="" uv run python -m piper_train.infer_onnx \
  --model /path/to/model-zeroshot.onnx \
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
| Zero-shot integration | 参照音声→生成音声の話者類似度 | SECS > 0.60 |
| ONNX export/import | ref_mel入力あり/なし両方 | エラーなし |
| Checkpoint migration | strict=Falseロード成功 | 既存モデル読み込み可 |
| CPU推論 | CUDA_VISIBLE_DEVICES="" で動作 | 推論成功 |

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
