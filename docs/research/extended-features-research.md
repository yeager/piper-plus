# piper-plus 拡張機能調査・参考文献

> roadmap.md から分離された調査資料 (調査日: 2026-03-12〜13)

---

## 調査ソース一覧

### 論文

| 論文 | 会議/年 | 関連Phase |
|------|---------|----------|
| Yoneyama et al. "Comparative Analysis of Neural Vocoders" | Interspeech 2025 | Phase 1 (ONNX実測) |
| MB-iSTFT-VITS (Kawamura et al.) | 2023 | Phase 1 |
| FLY-TTS | Interspeech 2024 | Phase 4 |
| VITS2 (Kong et al.) | 2023 | Phase 4 |
| VNet | 2024 | Phase 1 (Discriminator上限) |
| BigVGAN / BigVGAN v2 | - | Phase 1 (Snake activation) |
| YourTTS (Casanova et al.) | ICML 2022 | Phase 2, 3 |
| StyleTTS2 (Li et al.) | 2023 | Phase 4 |
| XTTS (Coqui) | Interspeech 2024 | Phase 2 |
| MMS-TTS (Meta) | 2023 | Phase 2 |
| Q-VITS | 2024 | Phase 4 |
| F5-TTS | 2025 | Phase 4 |
| Matcha-TTS | ICASSP 2024 | Phase 4 |
| NaturalSpeech 1-3 (Microsoft) | 2022-2024 | 参考 |
| W3C SSML 1.1 Specification | - | Phase 1 |

### OSSリポジトリ

| プロジェクト | URL | ライセンス |
|------------|-----|-----------|
| mobile-vocoder | github.com/ayutaz/mobile-vocoder | MIT |
| MB-iSTFT-VITS | github.com/MasayaKawamura/MB-iSTFT-VITS | MIT |
| mush42/istft-onnx | github.com/mush42/istft-onnx | - |
| VITS2 (p0p4k) | github.com/p0p4k/vits2_pytorch | MIT |
| StyleTTS2 | github.com/yl4579/StyleTTS2 | MIT |
| GPT-SoVITS | github.com/RVC-Boss/GPT-SoVITS | MIT |
| Bert-VITS2 | github.com/fishaudio/Bert-VITS2 | AGPL-3.0 |
| Coqui TTS | github.com/coqui-ai/TTS | MPL-2.0 |
| Fish Speech | github.com/fishaudio/fish-speech | 独自 |
| Kokoro | github.com/hexgrad/kokoro | Apache-2.0 |
| F5-TTS | github.com/SWivid/F5-TTS | CC-BY-NC-4.0 |
| CosyVoice | github.com/FunAudioLLM/CosyVoice | Apache-2.0 |
| OpenVoice | github.com/myshell-ai/OpenVoice | MIT |
| Wyoming Protocol | github.com/OHF-Voice/wyoming | MIT |
| Wyoming Piper | github.com/rhasspy/wyoming-piper | MIT |
| sherpa-onnx | github.com/k2-fsa/sherpa-onnx | Apache-2.0 |
| NeMo Text Processing | github.com/NVIDIA/NeMo-text-processing | Apache-2.0 |
| jpreprocess | github.com/jpreprocess/jpreprocess | MIT |
| kanjize | github.com/nagataaaas/Kanjize | MIT |
| num2words | pypi.org/project/num2words | LGPL |
| uPiper | github.com/ayutaz/uPiper | - |
| SpeechBrain ECAPA-TDNN | huggingface.co/speechbrain/spkrec-ecapa-voxceleb | Apache-2.0 |
| AudioSeal | github.com/facebookresearch/audioseal | MIT |
| Perth (Resemble AI) | github.com/resemble-ai/Perth | MIT |
| Chatterbox | github.com/resemble-ai/chatterbox | MIT |
| DeepFilterNet | github.com/Rikorose/DeepFilterNet | Apache-2.0 |
| noisereduce | github.com/timsainb/noisereduce | MIT |
| pyloudnorm | github.com/csteinmetz1/pyloudnorm | MIT |
| Pedalboard (Spotify) | github.com/spotify/pedalboard | GPL-3.0 |
| pydub | github.com/jiaaro/pydub | MIT |
| FlashSR | github.com/ysharma3501/FlashSR | MIT |
| HiFi-GAN BWE | github.com/brentspell/hifi-gan-bwe | MIT |
| VITS-fast-fine-tuning | github.com/Plachtaa/VITS-fast-fine-tuning | MIT |
| Seed-VC | github.com/Plachtaa/seed-vc | MIT |
| RVC | github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI | MIT |
| VISinger | - | - |

### GPT-SoVITS 調査ソース

| ソース | URL |
|--------|-----|
| GPT-SoVITS メインリポジトリ | github.com/RVC-Boss/GPT-SoVITS |
| GPT-SoVITS v3/v4 wiki | github.com/RVC-Boss/GPT-SoVITS/wiki |
| GPT-SoVITS v4 (HuggingFace) | huggingface.co/kevinwang676/GPT-SoVITS-v4 |
| GPT-SoVITS 技術解説 (ailia) | medium.com/axinc-ai/gpt-sovits |
| GPT-SoVITS 推論フロー解析 | medium.com/@alex1923221/gpt-sovits-audio-inference-process-analysis |
| GPT-SoVITS DeepWiki | deepwiki.com/RVC-Boss/GPT-SoVITS |
| gpt-sovits-python (PyPI) | pypi.org/project/gpt-sovits-python |

### Speaker Encoder事前学習モデル

| モデル | プラットフォーム | 学習データ |
|--------|----------------|-----------|
| spkrec-ecapa-voxceleb | SpeechBrain (HuggingFace) | VoxCeleb 1+2 |
| wespeaker-ecapa-tdnn512-LM | Wespeaker (HuggingFace) | VoxCeleb2 (5994話者) |
| dvector (yistLin) | GitHub | LibriSpeech |

### 最新注目モデル (2025-2026)

| モデル | 特筆点 | piper-plusへの示唆 |
|--------|--------|-------------------|
| Kokoro-82M | 82Mパラメータで高品質 | 超軽量モデルの実現可能性 |
| CosyVoice 2.0 | ストリーミング150ms、多言語ゼロショット | ストリーミング+ゼロショットの統合 |
| F5-TTS | DiT + Flow Matching、OSS | 次世代アーキテクチャの候補 |
| KittenTTS | 25MB以下でSOTA | エッジデバイス向けの極限最適化 |
| GPT-SoVITS v4 | 48k出力、5言語対応 | ゼロショット手法の参考 |
| IndexTTS-2 | 感情/話者分離制御 | 独立した制御軸の設計 |
| Chatterbox (Resemble AI) | 感情exaggeration制御、Perth透かし | 即効性の高い感情パラメータ |
| VoxMorph | SLERP話者モーフィング (ICASSP 2026) | 話者ブレンドの理論的裏付け |
| AudioSeal (Meta) | 音声透かし (MIT, 1.1 GFLOPs/s) | AI生成コンテンツ識別 |
| Style-BERT-VITS2 | BERT+wespeaker 4次元マージ | 感情/声質分離制御 |
| VISinger 2 | VITSベース歌声合成 | 歌声合成の参考 |
| EmoSphere-TTS | VAD球面座標感情制御 | 連続値感情制御 |

---

## 12. 拡張機能調査 — VITS系OSS横断分析

> 調査日: 2026-03-13 | 15エージェントによる並列調査
>
> GPT-SoVITS, Style-BERT-VITS2, Chatterbox, CosyVoice, Fish Speech, Kokoro, OpenVoice 等の
> VITS系OSSライブラリを横断的に調査し、piper-plusに移植可能な拡張機能を網羅的に整理した結果。

### 12.1 話者モーフィング・ベクトル演算

#### 概要

話者埋め込み空間での補間・演算により、学習データに存在しない中間的な声質を生成する技術。
既存の `piper-sample-generator` に `--slerp-weights` オプションが存在するが、本体への統合は未実施。

#### 補間手法の比較

| 手法 | 数式 | 特性 | 推奨用途 |
|------|------|------|---------|
| **LERP** | `(1-α)·v₁ + α·v₂` | 直線補間。ノルム減少 | 近距離の話者ペア |
| **SLERP** | `sin((1-α)θ)/sin(θ)·v₁ + sin(αθ)/sin(θ)·v₂` | 球面補間。ノルム保存 | **推奨** (VoxMorph ICASSP 2026で有効性実証) |
| **N-way SLERP** | 再帰的SLERP | 3話者以上のブレンド | マルチ話者モーフィング |

#### ONNX実装方法

現在のpiper-plus ONNXモデルは `sid` (整数) を入力として `nn.Embedding` で話者ベクトルを取得している。
モーフィングを実現するには、ONNXグラフの `Gather` ノード (Embedding lookup) を `Identity` に置換し、
事前計算済みの話者埋め込みベクトルを直接入力として渡す方式に変更する。

```python
# ONNX推論時のモーフィング例
import numpy as np

def slerp(v0, v1, t):
    """球面線形補間"""
    v0_norm = v0 / np.linalg.norm(v0)
    v1_norm = v1 / np.linalg.norm(v1)
    omega = np.arccos(np.clip(np.dot(v0_norm, v1_norm), -1, 1))
    if omega < 1e-10:
        return (1 - t) * v0 + t * v1
    return (np.sin((1 - t) * omega) / np.sin(omega)) * v0 + \
           (np.sin(t * omega) / np.sin(omega)) * v1

# 話者0と話者5を50%ずつブレンド
emb_blended = slerp(speaker_embeddings[0], speaker_embeddings[5], 0.5)
```

#### 属性ベクトル演算

話者埋め込み空間で意味的な方向ベクトルを抽出し、声質属性を操作する技術。

```
# 性別方向ベクトルの抽出
gender_direction = mean(male_embeddings) - mean(female_embeddings)

# 声質変換: 女性話者に「よりハスキーな」方向を加算
modified_emb = female_emb + 0.3 * husky_direction
```

#### piper-plusへの実装提案

| 優先度 | 機能 | 工数 | 必要な変更 |
|--------|------|------|-----------|
| **1** | `--blend-speakers "0:0.7,5:0.3"` CLI引数 | 2-3日 | `infer_onnx.py` + ONNX入力変更 |
| **2** | SLERP/LERP切り替え | 1日 | 補間関数の追加 |
| **3** | WebUI話者ブレンドスライダー | 1-2日 | Gradioインターフェース |
| 4 | 属性ベクトル抽出ツール | 3-5日 | 話者分析スクリプト |
| 5 | リアルタイムモーフィング (WebSocket) | 1週間 | ストリーミングAPI統合 |

---

### 12.2 感情・スタイル制御

#### アプローチ比較

| 手法 | 代表OSS | 制御方法 | 学習コスト | 推論コスト | piper-plus互換性 |
|------|---------|---------|-----------|-----------|----------------|
| **GST (Global Style Tokens)** | Tacotron 2 GST | 参照音声 → スタイルベクトル | 中 | 低 | ✅ 高 |
| **VAD (Valence-Arousal-Dominance)** | EmoSphere-TTS | 3次元連続値 | 中 | 低 | ✅ 高 |
| **Style-BERT-VITS2方式** | Style-BERT-VITS2 | wespeaker 256d + BERT 1024d | 高 | 中 | ○ 中 |
| **Exaggeration Parameter** | Chatterbox | 単一スカラー (0.25-2.0) | 低 | 最低 | ✅ 最高 |
| **T-VecTTS方式** | T-VecTTS | 凍結VITSに軽量ブランチ接続 | 低 | 低 | ✅ 高 |

#### Chatterbox方式: 最小実装で最大効果

Chatterbox (Resemble AI) の `exaggeration` パラメータは、推論時にnoise_scaleを非線形にスケーリングすることで感情の強度を制御する。piper-plusの既存 `noise_scale` パラメータを拡張するだけで実装可能。

```python
# 推論時: exaggeration=1.5 で感情を誇張
noise_scale_adjusted = noise_scale * exaggeration  # 0.25-2.0
```

| exaggeration | 効果 |
|-------------|------|
| 0.25 | 非常に抑制的。ニュースアンカー調 |
| 1.0 | 通常 |
| 1.5 | やや誇張。感情的な読み上げ |
| 2.0 | 最大誇張。ドラマティック |

#### GST (Global Style Tokens): 推奨アプローチ

10-20個の学習可能なスタイルトークンを導入し、参照音声またはスタイルIDでアテンション重みを選択する。

```python
class GlobalStyleTokens(nn.Module):
    def __init__(self, num_tokens=10, token_dim=256, num_heads=4):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(num_tokens, token_dim))
        self.attention = nn.MultiheadAttention(token_dim, num_heads)

    def forward(self, reference_embedding):
        # reference_embedding: 参照音声のメル特徴量の平均
        tokens = self.tokens.unsqueeze(0).expand(batch_size, -1, -1)
        style_embedding, weights = self.attention(
            reference_embedding.unsqueeze(1), tokens, tokens
        )
        return style_embedding.squeeze(1)  # (batch, token_dim)
```

**既存VITSへの統合**: `gin_channels` に `style_embedding` を連結して各モジュールに注入。

#### Style-BERT-VITS2のBERT統合 (参考)

Style-BERT-VITS2は `nn.Conv1d(1024, hidden_channels, 1)` でBERT出力をText Encoderに射影している。
JP-Extra版は日本語BERTのみ使用。piper-plusでの独自実装 (Apache-2.0) は Phase 4 で計画済み。

#### piper-plusへの実装提案

| 優先度 | 機能 | 工数 | Phase |
|--------|------|------|-------|
| **1** | `--exaggeration` パラメータ | 数時間 | Phase 1 |
| **2** | GST (10トークン) | 1-2週間 | Phase 4 |
| 3 | VAD 3次元連続制御 | 2-3週間 | Phase 4 |
| 4 | BERT プロソディ (独自実装) | 2-4週間 | Phase 4 |

---

### 12.3 非言語音声・表現力拡張

#### PUAトークンアプローチ (piper-plus互換)

piper-plusは既にIssue #204 (疑問詞マーカー) と Issue #207 (Nバリアント) でUnicode Private Use Area (PUA) トークンを活用しており、非言語音声にも同じパターンを適用できる。

| トークン | Unicode (提案) | 用途 | 学習データの必要量 |
|---------|---------------|------|------------------|
| `[laugh]` | 0xE020 | 笑い声 | 500-1000発話 |
| `[breath]` | 0xE021 | 息継ぎ | 自動検出で付与可能 |
| `[sigh]` | 0xE022 | ため息 | 200-500発話 |
| `[pause_long]` | 0xE023 | 長い間 | 既存無音トークンの拡張 |
| `[whisper]` | 0xE024 | ささやき | 500-1000発話 |
| `[cry]` | 0xE025 | 泣き声 | 300-500発話 |

#### 実装アーキテクチャ

```
入力テキスト: "それは[laugh]本当に面白いですね"
  → Phonemizer: "s o r e w a [0xE020] h o N_n t o o n i ..."
    → phoneme_id_map: [laugh] → ID 350 (例)
      → VITS: 通常の音素系列として処理
```

**利点**: モデルアーキテクチャの変更が不要。`phoneme_id_map` へのトークン追加と学習データのみで実現。

#### 息継ぎ自動検出

VAD (Voice Activity Detection) を使用して息継ぎ位置を自動検出し、`[breath]` トークンを自動付与。

```python
# silero-vad による息継ぎ検出
import torch
model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad')
# 短い無音 (200-500ms) を [breath]、長い無音 (500ms+) を [pause_long] に分類
```

#### 歌声合成 (長期研究)

VISinger / VISinger2 がVITSベースの歌声合成を実現しており、piper-plusアーキテクチャとの互換性が高い。ただし、楽譜情報 (MIDI) の入力インターフェースや、F0制御の追加が必要で工数が大きい。

| 要素 | 必要な変更 | 工数 |
|------|-----------|------|
| 楽譜入力 (MusicXML/MIDI) | 新規パーサー | 2-3週間 |
| F0条件付けモジュール | Duration Predictor拡張 | 1-2週間 |
| ビブラート制御 | ポストフィルター | 1週間 |

#### piper-plusへの実装提案

| 優先度 | 機能 | 工数 | Phase |
|--------|------|------|-------|
| **1** | `[breath]` 自動検出 + トークン化 | 3-5日 | Phase 1 |
| **2** | `[laugh]`, `[sigh]` 等の非言語トークン | 1-2週間 (データ収集含む) | Phase 2 |
| 3 | `[whisper]` モード | 2-3週間 (データ収集含む) | Phase 3 |
| 4 | 歌声合成 (VISinger方式) | 1-2ヶ月 | Phase 4+ |

---

### 12.4 音声後処理・エンハンスメント

#### 音声超解像 (Bandwidth Extension)

22050Hz → 44100Hz / 48000Hz のアップサンプリングを後処理として実行。
学習時に高サンプルレートで学習するよりも低コストで品質向上を実現。

| 手法 | RTF (A100) | 品質 | ONNX互換 | ライセンス |
|------|-----------|------|---------|-----------|
| **Vocos BWE** | 0.0001 | 高 | △ (iSTFT膨張問題) | MIT |
| **HiFi-GAN BWE** | 0.001 | 高 | ✅ | MIT |
| AudioSR | 0.1 | 最高 | △ | CC-BY-NC |
| ClearerVoice-Studio | - | 高 | - | Apache-2.0 |

**推奨**: HiFi-GAN BWE (ONNX互換、MITライセンス)

#### ノイズ除去

| 手法 | 用途 | ライセンス |
|------|------|-----------|
| **DeepFilterNet** | リアルタイムノイズ除去 (ONNX対応) | Apache-2.0 |
| RNNoise | 超軽量ノイズ除去 (48KB) | BSD |
| noisereduce | Spectral noise gating (Python) | MIT |

#### ラウドネス正規化

WavLMモデルの音割れ対策として最も即効性が高い。

```python
import pyloudnorm as pyln

meter = pyln.Meter(22050)
loudness = meter.integrated_loudness(audio)
audio_normalized = pyln.normalize.loudness(audio, loudness, -23.0)  # EBU R128
```

#### Soft-knee コンプレッサー

音割れ防止のための動的レンジ圧縮。ハードクリッピング (`np.clip`) よりも自然。

```python
def soft_clip(audio, threshold=0.9, knee=0.1):
    """Soft-knee コンプレッサー"""
    abs_audio = np.abs(audio)
    mask = abs_audio > threshold
    compressed = threshold + knee * np.tanh((abs_audio[mask] - threshold) / knee)
    audio[mask] = np.sign(audio[mask]) * compressed
    return audio
```

#### piper-plusへの実装提案

| 優先度 | 機能 | 工数 | Phase |
|--------|------|------|-------|
| **1** | ラウドネス正規化 (`--loudness-normalize`) | 数時間 | Phase 1 |
| **2** | Soft-knee コンプレッサー | 数時間 | Phase 1 |
| **3** | HiFi-GAN BWE (22k→44.1k) | 1-2週間 | Phase 2 |
| 4 | DeepFilterNet 統合 (オプション) | 1週間 | Phase 2 |

---

### 12.5 ストリーミング改善・LLM統合

#### Hann窓クロスフェード

現在のストリーミング実装が線形クロスフェードを使用している場合、Hann窓に変更することでチャンク接合部のアーティファクトを低減。

```python
def hann_crossfade(chunk_a, chunk_b, overlap_samples=1024):
    """Hann窓によるスムーズなチャンク接合"""
    fade_out = np.hanning(2 * overlap_samples)[:overlap_samples]
    fade_in = np.hanning(2 * overlap_samples)[overlap_samples:]
    chunk_a[-overlap_samples:] *= fade_out
    chunk_b[:overlap_samples] *= fade_in
    overlap = chunk_a[-overlap_samples:] + chunk_b[:overlap_samples]
    return np.concatenate([chunk_a[:-overlap_samples], overlap, chunk_b[overlap_samples:]])
```

#### ファーストチャンク最適化

CosyVoice 2.0の手法を参考に、最初のチャンクのレイテンシを最小化。

| 手法 | 初回レイテンシ | 実装難易度 |
|------|-------------|-----------|
| 文分割 + パイプライン | 200-500ms | 低 |
| チャンク分割推論 | 100-200ms | 中 |
| **文分割 + 非同期バッファ** | **150-300ms** | **低 (推奨)** |

#### LLM統合アーキテクチャ

```
[LLM] → テキストストリーム → [バッファ] → 文分割 → [piper-plus] → 音声チャンク → [クライアント]
                                                        ↑
                                               sentence_end検出で即時合成開始
```

#### API設計

| API方式 | ユースケース | 実装コスト |
|---------|------------|-----------|
| **WebSocket** | LLMとのリアルタイム対話 | 1-2週間 |
| **Server-Sent Events (SSE)** | 単方向ストリーミング | 数日 |
| gRPC | 高パフォーマンスサーバー間通信 | 2-3週間 |
| **OpenAI互換 `/v1/audio/speech`** | 既存エコシステム統合 | 数日 |

#### piper-plusへの実装提案

| 優先度 | 機能 | 工数 | Phase |
|--------|------|------|-------|
| **1** | Hann窓クロスフェード | 数時間 | Phase 1 |
| **2** | OpenAI互換API (`stream: true`) | 数日 | Phase 1 |
| **3** | WebSocket API | 1-2週間 | Phase 1 |
| 4 | LLM統合デモ (Ollama + piper-plus) | 3-5日 | Phase 2 |

---

### 12.6 ボイスクローニング・ファインチューニング

#### LoRA (Low-Rank Adaptation)

大規模モデルの重みを凍結し、低ランク行列のみを学習することで少量データ (数分) でファインチューニング。

```python
class LoRALinear(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()
        self.original = original_layer
        self.original.weight.requires_grad_(False)
        d_in, d_out = original_layer.weight.shape
        self.lora_A = nn.Parameter(torch.randn(d_in, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, d_out))
        self.scaling = alpha / rank

    def forward(self, x):
        return self.original(x) + (x @ self.lora_A @ self.lora_B) * self.scaling
```

| 構成 | 学習パラメータ | 全パラメータ比 | 必要データ | 学習時間 |
|------|-------------|-------------|----------|---------|
| **LoRA rank=8** | ~0.5M | **1.8%** | **3-5分** | **10-30分** |
| LoRA rank=16 | ~1.0M | 3.6% | 5-10分 | 30-60分 |
| フルファインチューニング | ~28M | 100% | 30分+ | 数時間 |

#### Speaker Consistency Loss (SCL)

ゼロショットTTS (Phase 3) における話者類似度を向上させる損失関数。
生成音声の話者埋め込みと参照音声の話者埋め込みのコサイン類似度を最大化。

```python
loss_scl = 1 - F.cosine_similarity(
    speaker_encoder(audio_generated),
    speaker_encoder(audio_reference)
).mean()
```

#### GPT-SoVITS v2Pro/v2ProPlus アプローチ

GPT-SoVITS v2Pro は Speaker Verification (SV) 埋め込みによるガイダンスで、v3/v4レベルの品質を低コスト (v2相当の学習コスト) で達成。piper-plusのPhase 3実装の参考になる。

#### piper-plusへの実装提案

| 優先度 | 機能 | 工数 | Phase |
|--------|------|------|-------|
| **1** | LoRAファインチューニングスクリプト | 1-2週間 | Phase 2 |
| **2** | SCL損失関数 (Phase 3と統合) | 1週間 | Phase 3 |
| 3 | WebUI LoRA学習インターフェース | 1-2週間 | Phase 3 |
| 4 | LoRA ONNXマージツール | 3-5日 | Phase 3 |

---

### 12.7 マルチスピーカー高度機能

#### Voice Design (テキストプロンプトからの話者生成)

自然言語で話者を記述し、対応する話者埋め込みを生成する技術。

```
プロンプト: "30代女性、落ち着いた声、やや低め"
  → [CLIP/BERT] → テキスト埋め込み → [話者空間への射影] → 話者ベクトル → [VITS] → 音声
```

**前提条件**: 話者埋め込み空間に意味的な構造が存在すること (Phase 3のSpeaker Encoder導入後)

#### 対話モード (Dialogue TTS)

複数話者の対話を1つの入力で処理し、自然な掛け合いを生成。

```
# 入力形式
<speaker id="0">こんにちは、今日の調子はどうですか？</speaker>
<speaker id="3">とても良いですよ、ありがとうございます。</speaker>
```

**実装**: SSMLパーサー (Phase 1) の拡張として対応可能。

#### 話者の継続学習 (Continual Learning)

新しい話者を追加する際に既存話者の品質を劣化させない手法。

| 手法 | 内容 | 既存話者への影響 |
|------|------|----------------|
| Elastic Weight Consolidation (EWC) | Fisher情報行列による重要パラメータ保護 | 低 |
| **Speaker Embedding拡張** | 新話者の埋め込みのみ追加 (モデル本体は凍結) | **なし** |
| LoRA追加 | 話者ごとにLoRAアダプターを追加 | なし |

**推奨**: Speaker Embedding拡張 (Phase 3の基盤) + LoRA (12.6節) の組み合わせ。

#### piper-plusへの実装提案

| 優先度 | 機能 | 工数 | Phase |
|--------|------|------|-------|
| **1** | 対話モード (SSMLベース) | 3-5日 | Phase 1 |
| **2** | Speaker Embedding拡張学習 | 1週間 | Phase 3 |
| 3 | Voice Design (テキスト→話者) | 2-4週間 | Phase 4 |
| 4 | 継続学習 (EWC) | 2-3週間 | Phase 4 |

---

### 12.8 多言語・コードスイッチング

#### コードスイッチング (言語混在)

1つの文中に複数言語が混在する発話を自然に合成する技術。

```
入力: "この project は really interesting ですね"
  → 言語検出: [ja] "この" [en] "project" [ja] "は" [en] "really interesting" [ja] "ですね"
    → 統一IPA: 各セグメントを言語別Phonemizerで音素化 → 結合
```

#### DiaMoE-TTS アプローチ

Mixture of Experts (MoE) を使用し、言語ごとに異なるエキスパートネットワークを活性化。

```python
class LanguageMoE(nn.Module):
    def __init__(self, num_languages, hidden_dim, num_experts=4):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(hidden_dim + num_languages, num_experts)

    def forward(self, x, language_embedding):
        gate_input = torch.cat([x, language_embedding], dim=-1)
        weights = F.softmax(self.gate(gate_input), dim=-1)
        output = sum(w.unsqueeze(-1) * expert(x)
                     for w, expert in zip(weights.unbind(-1), self.experts))
        return output
```

#### アクセントベクトル

言語埋め込みとは別に、アクセント/方言を制御するベクトルを導入。

```
言語埋め込み: [日本語] → 音素体系の選択
アクセントベクトル: [関西弁] → イントネーションパターンの選択
```

#### 自動言語検出

文字種 (Unicode範囲) とn-gramモデルの組み合わせで、入力テキストの言語をセグメント単位で自動検出。

```python
def detect_language_segments(text):
    """Unicode範囲ベースの簡易言語検出"""
    segments = []
    for char in text:
        if '\u3040' <= char <= '\u30FF' or '\u4E00' <= char <= '\u9FFF':
            lang = 'ja'
        elif '\u0041' <= char <= '\u007A':
            lang = 'en'
        # ... 他言語
        segments.append((char, lang))
    return merge_adjacent(segments)
```

#### piper-plusへの実装提案

| 優先度 | 機能 | 工数 | Phase |
|--------|------|------|-------|
| **1** | 自動言語検出 (Unicode範囲) | 2-3日 | Phase 2 |
| **2** | コードスイッチング (統一IPA結合) | 1-2週間 | Phase 2 |
| 3 | アクセントベクトル | 2-3週間 | Phase 3 |
| 4 | MoE言語エキスパート | 1-2ヶ月 | Phase 4 |

---

### 12.9 安全性・透かし

#### 音声透かし (Audio Watermarking)

生成音声にAI由来であることを示す不可聴の透かしを埋め込む技術。EU AI Act (2026年8月施行) により、AI生成コンテンツの明示義務が発生。

| 手法 | 計算コスト | 検出精度 | ライセンス | 特記 |
|------|-----------|---------|-----------|------|
| **AudioSeal (Meta)** | 1.1 GFLOPs/s | 高 (局所化可能) | **MIT** | **推奨** |
| SilentCipher | 中 | 高 | Apache-2.0 | AudioSealの後継的位置づけ |
| Perth (Chatterbox) | 低 | 中 | MIT | Chatterbox内蔵 |
| WavMark | 中 | 高 | MIT | 32bitペイロード |

#### AudioSeal統合

```python
from audioseal import AudioSeal

# 透かし埋め込み (推論パイプラインに追加)
watermark_model = AudioSeal.load_generator("audioseal_wm_16bits")
watermarked_audio = watermark_model(audio_tensor, message=secret_bits)

# 検出
detector = AudioSeal.load_detector("audioseal_detector_16bits")
result, message = detector.detect_watermark(watermarked_audio)
```

**推論速度への影響**: RTF +0.001未満 (無視できるレベル)

#### C2PA (Content Provenance)

Coalition for Content Provenance and Authenticity 標準。音声ファイルのメタデータにAI生成情報を埋め込む。

```python
# WAVファイルのメタデータにC2PA情報を付与
metadata = {
    "c2pa:generator": "piper-plus v1.7.0",
    "c2pa:model": "moe-speech-20speakers-v2",
    "c2pa:created": "2026-03-13T00:00:00Z"
}
```

#### piper-plusへの実装提案

| 優先度 | 機能 | 工数 | Phase |
|--------|------|------|-------|
| **1** | `--watermark` オプション (AudioSeal) | 3-5日 | Phase 1 |
| **2** | C2PAメタデータ付与 | 1-2日 | Phase 1 |
| 3 | 透かし検出API | 2-3日 | Phase 2 |
| 4 | 話者同意管理システム | 1-2週間 | Phase 3 |

---

### 12.10 UX・クリエイティブ応用

#### バッチ処理

複数テキストの一括合成。オーディオブック、字幕読み上げに必須。

```bash
# テキストファイルから一括合成
piper-plus batch --input chapters.txt --output-dir ./audiobook/ \
  --speaker-id 0 --format wav --split-by paragraph
```

#### SRT/VTT字幕対応

字幕ファイルからタイムコード付き音声を生成。

```python
# SRTパーサー
def parse_srt(srt_path):
    """SRT字幕ファイルを解析し、タイムコード付きセグメントを返す"""
    segments = []
    for subtitle in srt.parse(srt_path):
        segments.append({
            "text": subtitle.content,
            "start": subtitle.start.total_seconds(),
            "end": subtitle.end.total_seconds(),
            "speaker_id": extract_speaker_tag(subtitle.content)
        })
    return segments
```

#### オーディオブック機能

| 機能 | 内容 | 工数 |
|------|------|------|
| チャプター分割 | 入力テキストをチャプターごとに分割合成 | 1-2日 |
| 話者自動割り当て | 地の文=ナレーター、台詞=キャラクター | 3-5日 |
| 間の制御 | 段落間の自然な間 (500ms-2s) | 1日 |
| MP3/M4B出力 | オーディオブック標準フォーマット | 1日 |

#### モデルマージ (Style-BERT-VITS2参考)

Style-BERT-VITS2は4次元での部分マージをサポート:
- **声質 (voice)**: Generator の重み
- **ピッチ (pitch)**: Duration Predictor + Flow の重み
- **感情 (emotion)**: スタイルエンコーダの重み
- **テンポ (tempo)**: Duration Predictor の重み

```python
def merge_models(model_a, model_b, ratios):
    """部分マージ: 各次元で異なるブレンド比率"""
    merged = {}
    for name, param in model_a.items():
        if 'dec.' in name:  # Generator (声質)
            ratio = ratios['voice']
        elif 'dp.' in name:  # Duration Predictor (テンポ/ピッチ)
            ratio = ratios['tempo']
        elif 'flow.' in name:  # Flow (ピッチ)
            ratio = ratios['pitch']
        else:
            ratio = 0.5
        merged[name] = (1 - ratio) * param + ratio * model_b[name]
    return merged
```

#### Diff Merge

2つのファインチューニング済みモデル間の差分を抽出し、ベースモデルに適用。

```
diff = finetuned_model - base_model
new_model = another_base + α * diff  # α: 適用強度
```

#### piper-plusへの実装提案

| 優先度 | 機能 | 工数 | Phase |
|--------|------|------|-------|
| **1** | バッチ処理 CLI | 2-3日 | Phase 1 |
| **2** | SRT/VTT字幕入力 | 2-3日 | Phase 1 |
| **3** | モデルマージツール | 3-5日 | Phase 2 |
| 4 | オーディオブック機能 | 1-2週間 | Phase 2 |
| 5 | Diff Mergeツール | 2-3日 | Phase 2 |

---

### 12.11 GPT-SoVITS 詳細分析

> GitHub: [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) (55,700+ stars, MIT License)
>
> GPTスタイルのセマンティックモデリングとVITSベースの音響モデリングを組み合わせた、Few-shot音声クローニング/TTSシステム。

#### バージョン履歴とベンチマーク

| バージョン | リリース日 | 主な変更 | WER↓ | SIM↑ | GPT Params | piper-plus移植可能性 |
|-----------|----------|---------|------|------|-----------|-------------------|
| v1 | 2024-01 | 初版 (中/日/英) | 0.025 | 0.526 | 90M | 低 |
| v2 | 2024-08 | 韓国語/広東語追加、2x高速化 | 0.017 | 0.549 | 90M | 低 |
| v3 | 2025-02 | CFM-DiT導入、7000時間学習データ | 0.014 | 0.702 | 330M | 低 |
| v4 | 2025-04 | 48kHz出力、カスタムボコーダ | 0.013 | 0.735 | 330M | 部分的 |
| **v2Pro** | 2025-06 | **v2コストでv3/v4品質** | 0.016 | 0.709 | 133M | **高** |
| v2ProPlus | 2025-06 | S2デコーダConv層拡張 | 0.016 | 0.737 | 152M | **高** |

*(SeedTTS中国語テストセット。Ground Truth: WER=0.013, SIM=0.750)*

#### 2段階アーキテクチャ

```
テキスト入力
  → [G2P + BERT特徴抽出] ← 言語別処理 (中国語/日本語/英語/韓国語/広東語)
    → [S1: GPT Semantic Model] ← 参照音声のSSL特徴量 (スタイル条件)
      → セマンティックトークン列
        → [S2: SoVITS / CFM-DiT] ← 参照音声のスタイル埋め込み
          → [ボコーダ (HiFi-GAN / BigVGAN / カスタム)]
            → 音声波形出力 (24kHz or 48kHz)
```

| 段階 | v1/v2 | v3/v4 | v2Pro/v2ProPlus |
|------|-------|-------|----------------|
| S1 (GPT) | 90M Transformer | 330M + 7000hr学習 | 133-152M + SV埋め込み |
| S2 | 標準VITS | **shortcut-CFM-DiT** | VITS + SV埋め込みガイダンス |
| ボコーダ | HiFi-GAN (32kHz) | BigVGANv2 (24kHz) | HiFi-GAN (32kHz) |

#### v3/v4 の技術革新

**CFM-DiT (v3):** 参照音声の拡散補完 (diffusion outpainting) による生成。音色類似度が大幅向上 (SIM: 0.549→0.702)。サンプリングステップ4-8で高速推論、32で最高品質。

**48kHz出力 (v4):** カスタムボコーダによるネイティブ48kHz。v3のBigVGANv2は非整数倍アップサンプリングによるメタリックアーティファクトが問題だった (piper-plusでの教訓: ボコーダのhop sizeとSSLのhop sizeの整合性に注意)。

**v2Pro/v2ProPlus:** S2モデルにSV (Speaker Verification) 話者埋め込みガイダンスを追加し、話者埋め込み次元を1024チャネルに拡張。v2の推論速度/ハードウェアコストでv3/v4レベルの品質を達成。

#### ゼロショット/Few-shot手法

| 要素 | GPT-SoVITS | piper-plus (現状/計画) |
|------|-----------|----------------------|
| 話者条件付け | 参照音声SSL特徴量 + テキスト | speaker_id → Phase 3: d-vector |
| ゼロショット | 5秒参照音声で即時対応 | Phase 3で対応予定 |
| Few-shot | 1分データでファインチューニング | Phase 2: LoRA対応予定 |
| スタイル表現 | CNHuBERT SSL (時系列ごとの詳細スタイル) | d-vector (固定長ベクトル) |
| ONNX互換性 | 困難 (大きなSSLモデル) | 良好 (小さなEncoder) |
| エッジ実行 | 困難 | 可能 |

#### テキストフロントエンドの比較

| 機能 | GPT-SoVITS | piper-plus |
|------|-----------|-----------|
| 日本語G2P | pyopenjtalk | OpenJTalk (A1/A2/A3韻律情報 + PUA文字) |
| 英語G2P | g2p_en | g2p-en (同じエンジン, Apache-2.0) |
| 中国語G2P | G2PW + BERT | pypinyin (Phase 2) |
| 言語レジストリ | 手動分岐 | **Phonemizer ABC + レジストリ (拡張性高)** |
| 混合言語 | 自動検出 + 分割 | Phase 2で対応予定 |
| 韻律情報 | BERTのみ (中国語) | **A1/A2/A3 + 疑問詞マーカー + Nバリアント** |

**piper-plusの優位点:** OpenJTalkのA1/A2/A3韻律情報とPUA文字による詳細な音素制御は、GPT-SoVITSには存在しない独自の強み。

#### 推論パラメータ制御

| 制御軸 | GPT-SoVITS | piper-plus |
|--------|-----------|-----------|
| 話速 | `speed_factor` (連続値) | `length_scale` (scales[1]) |
| ピッチ変動 | `temperature` (間接的) | `noise_scale` (scales[0]) |
| Duration変動 | GPT自己回帰で自然変動 | `noise_w` (scales[2]) |
| 繰り返し抑制 | `repetition_penalty=1.35` | 不要 (非自己回帰) |
| バッチ推論 | `batch_size` + `split_bucket` | 未実装 → **Phase 1で追加** |
| ストリーミング | 4段階 (0/1/2/3) | C++側で一部対応 |

**GPT-SoVITSのストリーミングモード:**

| モード | 品質 | 応答速度 | 説明 |
|--------|------|---------|------|
| 0 | 最高 | 遅い | 全文一括生成 |
| 1 | 最高 | やや遅い | セマンティックトークン単位 |
| 2 | 中 | やや速い | 粗いチャンク単位 |
| 3 | やや低 | 速い | 最小チャンク単位 |

#### WebUI機能 (piper-plusへの示唆)

GPT-SoVITSのWebUIはデータセット準備〜学習〜推論の統合環境:

| タブ | 機能 | piper-plus対応 |
|------|------|---------------|
| 音声分離 | UVR5によるボーカル/伴奏分離 | 外部ツール依存 |
| 音声分割 | 自動セグメント分割 | 外部ツール依存 |
| ASR | Whisper/Damoによる書き起こし | 外部ツール依存 |
| テキスト校正 | ASR結果の手動修正UI | なし |
| 学習 | GPT/SoVITSファインチューニング | CLI対応済み |
| 推論 | パラメータ調整+音声合成 | Gradio WebUI対応済み |

#### データセット作成ツール

| 機能 | GPT-SoVITS | piper-plus |
|------|-----------|-----------|
| 音声分離 | UVR5内蔵 | 外部ツール |
| 音声分割 | 内蔵スライサー | 外部ツール |
| ASR | Whisper/Damo内蔵 | 外部ツール |
| 韻律付与 | なし | **`add_prosody_features` ツール** |
| データ形式 | `path\|speaker\|lang\|text` (CSV風) | JSONL (phoneme_ids, prosody_features) |
| 品質管理 | MOSベースフィルタリング (7000hr) | Phase 1で導入予定 |

#### 移植可能性分析

**移植が有望 (高優先度):**

| 機能 | 移植方式 | 効果 | 工数 |
|------|---------|------|------|
| **SV埋め込みガイダンス** (v2Pro) | Phase 3 Speaker Encoder統合 | ゼロショット品質向上 | Phase 3と同時 |
| **バッチ推論** (split_bucket) | Python API拡張 | 長文処理効率化 | 1-2週間 |
| **ストリーミング品質レベル** | API拡張 | UX向上 | 1週間 |
| **MOS品質フィルタリング** | データセットツール | 学習データ品質向上 | 数日 |

**参考にすべき (中優先度):**

| 機能 | piper-plusへの示唆 |
|------|------------------|
| 48kHz出力 | 短期は後処理BWE、長期はネイティブ48kHz学習 |
| CFM技術 | Phase 4「Flow Matching統合」の参考 (VITSのNF→OT-CFM置換) |
| BERT統合 | Phase 4「BERTプロソディ」の参考 (日本語BERT活用) |
| 自動言語検出 | Phase 2マルチ言語の参考 |

**直接移植は困難:**

| 機能 | 理由 |
|------|------|
| 2段階アーキテクチャ (GPT+SoVITS) | 330M追加、エッジ対応と矛盾 |
| CNHuBERTスタイル抽出 | 数百MBモデル、ONNX困難 |
| GPT自己回帰トークン生成 | 推論速度不安定、VITSのDP方式が優位 |

---

### 12.12 Style-BERT-VITS2 詳細分析

> GitHub: [fishaudio/Bert-VITS2](https://github.com/fishaudio/Bert-VITS2) (8,600+ stars, **AGPL-3.0**)
>
> ⚠️ コードの直接移植は不可。手法のみを参考にし、Apache-2.0互換の独自実装が必要。

#### 技術要素と移植可能性

| 技術要素 | 内容 | 移植可能性 | ライセンス注意 |
|---------|------|-----------|-------------|
| **BERT プロソディ** | `nn.Conv1d(1024, hidden, 1)` でBERT射影 | ○ (独自実装必要) | AGPL回避必須 |
| **wespeaker スタイルベクトル** | 256d話者/スタイル埋め込み | ✅ 高 | wespeaker自体はApache-2.0 |
| **4次元モデルマージ** | voice/pitch/emotion/tempo分離マージ | ✅ 高 | ツール自体は問題なし |
| **Diff Merge** | ファインチューニング差分の移植 | ✅ 高 | ツール自体は問題なし |
| JP-Extra | 日本語特化 (日本語BERTのみ使用) | ○ (参考にできる) | AGPL回避必須 |

---

### 12.13 移植優先度マトリクス (GPT-SoVITS + Style-BERT-VITS2)

```
                    移植容易性 →
                低           中           高
          ┌──────────┬──────────┬──────────┐
    高    │          │BERT      │モデルマージ│
    ↑     │          │プロソディ │SV埋め込み │
    効    │          │          │Diff Merge│
    果    ├──────────┼──────────┼──────────┤
    ↓     │GPT 2段階 │MoE言語   │exaggeration│
    低    │歌声合成  │エキスパート│LoRA      │
          └──────────┴──────────┴──────────┘
```

---

### 12.14 軽量モデル技術

#### Kokoro-82M (StyleTTS2 + iSTFTNet)

82Mパラメータで商用品質を達成。知識蒸留 + アーキテクチャ最適化の成果。

| 技術 | 内容 | piper-plus適用性 |
|------|------|----------------|
| StyleTTS2ベース | SLM Discriminator + スタイル拡散 | 中 (アーキテクチャ差異) |
| iSTFTNet デコーダ | iSTFT+Conv1d ハイブリッド | △ (ONNX膨張問題、4.1節参照) |
| 82Mパラメータ | 大規模学習 → 小規模蒸留 | ✅ (蒸留手法は汎用) |

#### KittenTTS (15M-80M + QAT)

| 技術 | 内容 | piper-plus適用性 |
|------|------|----------------|
| **QAT (量子化対応学習)** | 学習時にINT8量子化をシミュレート | ✅ 高 |
| グループ化パラメータ共有 | HiFi-GANの上位ブロックの重みを共有 | ✅ 高 |
| Nix-TTS蒸留 | モジュール別蒸留 (89.34%パラメータ削減) | ✅ 高 |

#### QAT実装例

```python
import torch.quantization as quant

# QAT対応学習
model_fp32 = VitsModel(...)
model_fp32.qconfig = quant.get_default_qat_qconfig('qnnpack')
model_prepared = quant.prepare_qat(model_fp32)

# 通常通り学習
for epoch in range(num_epochs):
    train(model_prepared, ...)

# INT8モデルに変換
model_int8 = quant.convert(model_prepared)
```

#### FLY-TTS ConvNeXt V2 (再掲)

Phase 4で既に計画済み (7.6節) だが、軽量化の文脈でも重要。
ConvNeXt V2のグループ化パラメータ共有により、デコーダパラメータを50-70%削減可能。

#### piper-plusへの実装提案

| 優先度 | 機能 | 工数 | Phase |
|--------|------|------|-------|
| **1** | QAT対応学習 | 1-2週間 | Phase 1 |
| **2** | グループ化パラメータ共有 | 1週間 | Phase 1 |
| **3** | モジュール別知識蒸留 | 2-3週間 | Phase 4 |
| 4 | `quality: nano` ティア (4MB目標) | 1ヶ月 | Phase 4 |

---

### 12.15 拡張機能の統合ロードマップ

#### Phase 1 即時追加候補 (数時間〜数日)

| # | 機能 | 工数 | カテゴリ |
|---|------|------|---------|
| E1 | `--exaggeration` パラメータ | 数時間 | 感情制御 |
| E2 | ラウドネス正規化 + Soft-knee | 数時間 | 後処理 |
| E3 | Hann窓クロスフェード | 数時間 | ストリーミング |
| E4 | バッチ処理 CLI | 2-3日 | UX |
| E5 | `--watermark` (AudioSeal) | 3-5日 | 安全性 |
| E6 | SRT/VTT字幕入力 | 2-3日 | UX |
| E7 | `--blend-speakers` SLERP | 2-3日 | モーフィング |

#### Phase 2 短期追加候補 (数週間)

| # | 機能 | 工数 | カテゴリ |
|---|------|------|---------|
| E8 | 非言語トークン (`[breath]`, `[laugh]`) | 1-2週間 | 表現力 |
| E9 | HiFi-GAN BWE (22k→44.1k) | 1-2週間 | 後処理 |
| E10 | LoRAファインチューニング | 1-2週間 | ボイスクローニング |
| E11 | モデルマージツール | 3-5日 | UX |
| E12 | コードスイッチング (統一IPA) | 1-2週間 | 多言語 |
| E13 | WebSocket API | 1-2週間 | ストリーミング |

#### Phase 3-4 中長期追加候補 (数ヶ月)

| # | 機能 | 工数 | カテゴリ |
|---|------|------|---------|
| E14 | GST (Global Style Tokens) | 1-2週間 | 感情制御 |
| E15 | QAT対応学習 | 1-2週間 | 軽量化 |
| E16 | Voice Design (テキスト→話者) | 2-4週間 | マルチスピーカー |
| E17 | BERT プロソディ (独自実装) | 2-4週間 | 品質向上 |
| E18 | MoE言語エキスパート | 1-2ヶ月 | 多言語 |
| E19 | 歌声合成 (VISinger方式) | 1-2ヶ月 | 表現力 |

#### 最もインパクトが大きい拡張機能の組み合わせ

**即効 (数日で差別化):**
```
exaggeration + ラウドネス正規化 + SLERP話者ブレンド + バッチ処理
→ 感情制御 + 音割れ解決 + 新しい声の創出 + 実用性向上
```

**短期 (数週間でエコシステム):**
```
AudioSeal透かし + WebSocket API + LoRA + 非言語トークン
→ 安全性 + LLM統合 + 少量データ適応 + 自然な表現力
```

**中長期 (競合優位):**
```
GST感情制御 + BERTプロソディ + コードスイッチング + QAT軽量化
→ VITSベースTTS初の包括的感情/多言語/エッジ対応
```

---

## 付録: オリジナルpiperの全対応言語一覧

アラビア語、カタルーニャ語、チェコ語、ウェールズ語、デンマーク語、ドイツ語、ギリシャ語、英語 (GB/US)、スペイン語 (AR/ES/MX)、ペルシャ語、フィンランド語、フランス語、ヒンディー語、ハンガリー語、アイスランド語、イタリア語、ジョージア語、カザフ語、ルクセンブルク語、ラトビア語、マラヤーラム語、ネパール語、オランダ語 (BE/NL)、ノルウェー語、ポーランド語、ポルトガル語 (BR/PT)、ルーマニア語、ロシア語、スロバキア語、スロベニア語、セルビア語、スウェーデン語、スワヒリ語、トルコ語、ウクライナ語、ベトナム語、中国語

piper-plusはこれらのupstreamモデルと互換性を維持しており、上記言語の既存モデルをそのまま使用可能です。
