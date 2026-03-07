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

---

## 1. エグゼクティブサマリー

### 結論

Piperに最も適したzero-shot導入アプローチは **YourTTS方式（Speaker Encoder置換）** である。

- 既存VITSの `gin_channels` メカニズムをそのまま流用可能
- 話者ID embedding (`emb_g`) を外部Speaker Encoder出力に置換するだけで動作
- Speaker Encoderには **SpeechBrain ECAPA-TDNN** (Apache-2.0, 192次元) を推奨
- 推定実装工数: 2-3日 + 学習時間

### 推奨ロードマップ

| フェーズ | 内容 | 工数 |
|---------|------|------|
| Phase 1 | YourTTS方式 Speaker Encoder導入 | 2-3日 + 学習 |
| Phase 2 | TextEncoder Speaker Conditioning追加 (VITS2方式) | 1日 |
| Phase 3 | MB-iSTFT-VITS デコーダ高速化 (オプション) | 3-5日 + 再学習 |

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

### 6.1 段階的導入ロードマップ

#### Phase 1: YourTTS方式 Speaker Encoder導入（最優先）

**概要**: 話者ID embeddingを外部Speaker Encoder出力に置換

**技術選定:**
- Speaker Encoder: **SpeechBrain ECAPA-TDNN** (Apache-2.0, 192次元)
- 統合方式: `emb_g` → `speaker_encoder(ref_audio)` に置換
- 注入箇所: 既存の4箇所 (DP, PosteriorEnc, Flow, Decoder) そのまま

**学習戦略:**
1. Speaker Encoderを凍結してVITS部分のみ学習 (50-100 epoch)
2. Speaker Encoderを解凍してend-to-end微調整 (50 epoch)

**既存モデルからの移行:**
- 既存話者のembeddingを事前計算して保存可能
- 新旧モデルの共存が可能

**推定工数**: 2-3日（実装）+ 学習時間

#### Phase 2: TextEncoder Speaker Conditioning追加

**概要**: VITS2方式でTextEncoderにも話者条件付けを追加

**変更内容:**
- `enc_p` (TextEncoder) に `gin_channels` 対応を追加
- attentions.pyの `Encoder` クラスにconditioning パスを追加

**期待効果**: 話者類似度のさらなる向上 (MOS +0.2程度)

**推定工数**: 1日

#### Phase 3: デコーダ高速化 (オプション)

**概要**: MB-iSTFT-VITSデコーダによる推論4倍高速化

**推定工数**: 3-5日 + 再学習

### 6.2 非推奨のアプローチ

| アプローチ | 非推奨理由 |
|-----------|-----------|
| XTTS方式 (GPT + VITSデコーダ) | 完全な再設計が必要、autoregressive で推論遅い |
| GPT-SoVITS直接統合 | モデルサイズ大、edge推論に不向き |
| F5-TTS移行 | VITSとは根本的に異なるアーキテクチャ、完全移行に近い |

### 6.3 代替アプローチ: 外部zero-shot TTSの活用

Piperのアーキテクチャを変更せず、外部のzero-shot TTSシステムを使って話者embeddingを抽出し、Piperの既存マルチスピーカーモデルに注入する2段階パイプラインも検討に値する:

1. 外部Speaker Encoder（ECAPA-TDNN等）でリファレンス音声からembeddingを抽出
2. 抽出したembeddingをPiperのspeaker conditioningに注入
3. Piperの軽量・高速な推論パイプラインはそのまま維持

### 6.4 調査で参考にすべきプロジェクト

| プロジェクト | 参考になる点 |
|-------------|-------------|
| YourTTS | VITSへのSpeaker Encoder統合のリファレンス実装 |
| SpeechBrain | ECAPA-TDNNの事前学習済みモデル |
| CosyVoice | Apache-2.0、日本語対応、LLM+Flow Matching |
| Kokoro (StyleTTS 2派生) | 82MパラメータでONNXブラウザ推論実現 |
| GPT-SoVITS | 日本語zero-shotの品質ベンチマーク |

---

## 参考文献

- YourTTS Paper: https://arxiv.org/abs/2112.02418
- VALL-E: https://arxiv.org/abs/2301.02111
- StyleTTS 2: https://arxiv.org/abs/2306.07691
- F5-TTS: https://arxiv.org/abs/2410.06885
- ECAPA-TDNN: https://arxiv.org/abs/2005.07143
- VITS: https://arxiv.org/abs/2106.06103
- VITS2: https://arxiv.org/abs/2307.16430
- CosyVoice: https://github.com/FunAudioLLM/CosyVoice
- GPT-SoVITS: https://github.com/RVC-Boss/GPT-SoVITS
- SpeechBrain: https://github.com/speechbrain/speechbrain
- MB-iSTFT-VITS: https://github.com/MasayaKawamura/MB-iSTFT-VITS
