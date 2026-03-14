# VITS2 アーキテクチャ移行調査 & TTS最新動向

> **調査日**: 2026-03-14
> **目的**: piper-plusのベースアーキテクチャVITSをVITS2に置き換えることで精度向上が可能かを調査。加えてVITS2以降のTTSアーキテクチャ動向を整理し、次世代への移行候補を検討。
> **VITS2論文**: "VITS2: Improving Quality and Efficiency of Single-Stage Text-to-Speech with Adversarial Learning and Architecture Design" (Interspeech 2023, [arXiv:2307.16430](https://arxiv.org/abs/2307.16430))

---

## 1. VITS2の概要

VITS2はVITSの後継アーキテクチャで、以下の5つの改良により音質・学習効率・推論速度を同時に向上させる。

### 1.1 報告されたMOS改善

| 指標 | VITS | VITS2 | 差分 |
|------|------|-------|------|
| 自然性MOS (LJSpeech) | 4.38 (±0.06) | **4.47** (±0.06) | +0.09 |
| CMOS vs VITS | — | **+0.201** (±0.105) | — |
| 話者類似度MOS (VCTK) | 3.79 (±0.09) | **3.99** (±0.08) | **+0.20** |
| Ground Truth MOS | 4.43 (±0.06) | — | — |
| 文字誤り率 CER (音素入力) | 4.26 | **3.92** | -0.34 |
| 文字誤り率 CER (正規化テキスト) | 5.07 | **4.01** | -1.06 |

VITS2はGround Truth (4.43) を上回るMOS 4.47を達成。

### 1.2 速度改善

| 指標 | VITS | VITS2 | 改善率 |
|------|------|-------|--------|
| 合成速度 | 1,779 kHz | 2,144 kHz | **+22.7%** |
| リアルタイムファクタ | ×80.68 | ×97.25 | +20.5% |
| 学習速度 | 1.227 s/step | 0.951 s/step | **+22.5%** |

---

## 2. VITS2の5つの改良点

### (A) 敵対的Duration Predictor

VITSのFlow-based Stochastic Duration PredictorをGAN的訓練のDuration Predictorに置換。

- **Generator**: テキスト隠れ表現 `h_text` + ガウスノイズ `z_d` → duration `d_hat` を予測
- **Discriminator**: `h_text` + duration（MAS真値 or 予測値）→ 時間ステップ単位の条件付き判別
- **損失**: Least-squares adversarial loss + MSE regression loss
- **MOS影響**: 除去時 **-0.14 MOS**

### (B) Normalizing Flow + Transformerブロック

畳み込みベースのNormalizing FlowにTransformerブロックを残差接続付きで追加。

- 畳み込みでは受容野の制約により捉えられない長距離依存性を解決
- Transformerの自己注意機構により様々な位置から情報を収集
- **MOS影響**: 除去時 **-0.06 MOS**

### (C) Noise-Scaled MAS

MASのQ値にガウスノイズを追加（学習初期のみ）。

- ノイズスケール: **0.01** → ステップごとに **2×10⁻⁶** ずつ減衰
- テキスト正規化入力時に特に効果的（音素変換への依存を軽減）
- **MOS影響**: 除去時 **-0.15 MOS**（最も影響大）

### (D) Speaker-Conditioned Text Encoder

TextEncoderの第3 Transformerブロックに話者ベクトルを条件付け。

- 各話者固有の発音やイントネーションをモデリング
- **話者類似度MOS**: 3.79 → **3.99** (+0.20)
- マルチスピーカーモデル専用

### (E) Mel Spectrogram Posterior Encoder

Linear Spectrogram (513ch) → Mel Spectrogram (80ch) に変更。

- 再構成損失の入力と同じMel Spectrogramを後部エンコーダにも使用
- FFTサイズ: 1024、窓サイズ: 1024、ホップサイズ: 256
- チャンネル数削減によるメモリ効率化

---

## 3. 現在のpiper-plus VITSアーキテクチャとの対応

### 3.1 モジュール対応表

| 現在のモジュール | ファイル | VITS2での変更 |
|-----------------|---------|--------------|
| `TextEncoder` | `models.py` L170-211 | 話者条件付け追加 (D) |
| `PosteriorEncoder` | `models.py` L259-298 | Mel Spec入力に変更 (E) |
| `StochasticDurationPredictor` | `models.py` L122-167 | 敵対的DPに置換 (A) |
| `ResidualCouplingBlock` (Flow) | `models.py` L214-256 | Transformerブロック追加 (B) |
| `Generator` (HiFi-GAN) | `models.py` L301-379 | 変更なし |
| `MultiPeriodDiscriminator` | `models.py` | 変更なし |
| `WavLMDiscriminator` (piper-plus独自) | `models.py` L526-716 | 維持（VITS2論文には含まれない） |
| `prosody_proj` (piper-plus独自) | `models.py` L830-866 | 維持 |

### 3.2 現在のアーキテクチャ全体像

```
SynthesizerTrn
├─ enc_p (TextEncoder)              ← VITS2: 話者条件付け追加 (D)
│  ├─ emb (nn.Embedding)
│  ├─ encoder (attentions.Encoder)
│  └─ proj (nn.Conv1d)
├─ enc_q (PosteriorEncoder)         ← VITS2: Mel Spec入力 (E)
│  ├─ pre (Conv1d)
│  ├─ enc (WN)
│  └─ proj (Conv1d)
├─ dp (StochasticDurationPredictor) ← VITS2: 敵対的DP (A)
├─ flow (ResidualCouplingBlock)     ← VITS2: Transformer追加 (B)
│  └─ flows (ResidualCouplingLayer × 4 + Flip)
├─ prosody_proj (Linear 3→16)      ← piper-plus独自、維持
├─ dec (Generator/HiFi-GAN)        ← 変更なし
│  ├─ conv_pre → ups × 4 → resblocks × 12 → conv_post
├─ emb_g (Speaker Embedding)
└─ [学習時のみ]
   ├─ model_d (MultiPeriodDiscriminator)  ← 変更なし
   │  ├─ DiscriminatorS
   │  └─ DiscriminatorP (periods: 2,3,5,7,11)
   └─ model_d_wavlm (WavLMDiscriminator) ← piper-plus独自、維持
```

---

## 4. piper-plusへの移行分析

### 4.1 変更が必要なファイルと規模

| ファイル | 変更内容 | 追加行数 |
|----------|----------|----------|
| `vits/models.py` (994行) | DurationDiscriminator新規、TransformerCouplingLayer追加、TextEncoder話者条件付け | +300〜500行 |
| `vits/lightning.py` (500+行) | Duration Discriminator学習ループ、Noise-Scaled MAS制御 | +200〜300行 |
| `vits/modules.py` (526行) | Transformerブロック関連ユーティリティ | +100〜200行 |
| `export_onnx.py` | 推論グラフの更新 | +50〜100行 |
| `vits/config.py` (331行) | 新フラグ追加 | +20〜30行 |
| **合計** | **5〜8ファイル変更** | **約700〜1,200行追加** |

### 4.2 費用対効果

**メリット:**

1. **マルチスピーカー品質の大幅向上** — 話者類似度MOS +0.20は20話者モデルに有意
2. **学習・推論速度22%向上** — 現在の学習時間（200epoch≈90時間）を約70時間に短縮
3. **Duration Predictor安定化** — 敵対的訓練がDP崩壊問題の別解法になりうる
4. **段階的導入可能** — 各改良がフラグで独立制御可能

**リスク・注意点:**

1. **チェックポイント非互換** — 既存v2/WavLMモデルは再学習が必要
2. **spec_channels変更 (E)** — 513→80は前処理パイプライン全体に影響
3. **WavLMとの相乗効果は未検証** — VITS2論文にはWavLM Discriminatorなし
4. **公式実装なし** — 非公式実装(p0p4k)は「趣味プロジェクト」と明記
5. **単話者のMOS改善は+0.09と控えめ** — WavLM (+0.15〜+0.25) と比べると小幅

### 4.3 推奨導入順序

段階的に導入し、各段階で効果を検証するアプローチを推奨:

| 優先度 | 改良 | 理由 | 変更量 | 期待効果 |
|--------|------|------|--------|----------|
| 1 | **(C) Noise-Scaled MAS** | 最小変更で最大効果 | ~20行 | MOS -0.15防止 |
| 2 | **(D) Speaker-Conditioned TextEncoder** | マルチスピーカーに大きな効果 | ~50行 | 話者類似度+0.20 |
| 3 | **(A) 敵対的Duration Predictor** | DP崩壊問題への効果検証 | ~400行 | MOS +0.14 |
| 4 | **(B) Transformer Flow** | 長文プロソディ改善 | ~200行 | MOS +0.06 |
| 5 | **(E) Mel Posterior Encoder** | 影響範囲が大きいため最後に | ~100行 | メモリ効率化 |

### 4.4 piper-plus独自拡張との互換性

| piper-plus独自機能 | VITS2互換性 |
|-------------------|-------------|
| Prosody Features (A1/A2/A3) | 互換 — Duration Predictor入力に変更なし |
| WavLM Discriminator | 互換 — 独立したDiscriminatorとして維持 |
| SpeakerBalancedBatchSampler | 互換 — バッチ構成は変更不要 |
| EMA (Generator) | 互換 — HiFi-GANに変更なし |
| FP16 Mixed Precision | 互換 — VITS2論文でもMixed Precision使用 |

---

## 5. オープンソース実装の状況

### 5.1 主要リポジトリ

| リポジトリ | Stars | 完成度 | ライセンス | 備考 |
|-----------|-------|--------|-----------|------|
| [p0p4k/vits2_pytorch](https://github.com/p0p4k/vits2_pytorch) | 547 | 中〜高 | MIT | 全機能フラグ制御可。ONNX対応。**参考実装として最適** |
| [daniilrobnikov/vits2](https://github.com/daniilrobnikov/vits2) | 634 | 中 (WIP) | MIT | Duration Discriminator未実装 |
| [fishaudio/Bert-VITS2](https://github.com/fishaudio/Bert-VITS2) | 8,700 | 高 | — | VITS2 + BERT。メンテナンス終了、Fish Speechに移行 |
| [litagin02/Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) | 1,200 | 高 | AGPL-3.0 | 日本語特化。スタイル制御追加 |
| [FENRlR/MB-iSTFT-VITS2](https://github.com/FENRlR/MB-iSTFT-VITS2) | 133 | 実験的 | — | MB-iSTFTデコーダとのハイブリッド |

### 5.2 p0p4k/vits2_pytorch の設定フラグ

```json
{
  "use_duration_discriminator": true,
  "use_noise_scaled_mas": true,
  "use_transformer_flows": true,
  "transformer_flow_type": "pre_conv",
  "use_spk_conditioned_encoder": true,
  "use_mel_posterior_encoder": true
}
```

各機能を個別にON/OFFでき、段階的導入に適している。

### 5.3 VITS2を採用した他のプロジェクト事例

| プロジェクト | 採用状況 | 結果 |
|-------------|---------|------|
| Bert-VITS2 | VITS2バックボーン + multilingual BERT | 8.7k stars、日中英対応 |
| Style-Bert-VITS2 | Bert-VITS2 + スタイル制御 | 日本語TTSで最も実用的な実装の一つ |
| GPT-SoVITS | Transformer Flowを部分採用 | GPT + SoVITSハイブリッド |
| LIMMITS 2024 Challenge | マルチスピーカー音声クローニング | 話者類似度MOS 4.17（第1位） |
| Coqui TTS | Issue #2828で提案 → wontfix | XTTSv2に注力 |

---

## 6. VITS2学習設定（論文より）

| 項目 | 値 |
|------|-----|
| ハードウェア | NVIDIA V100 × 4 |
| オプティマイザ | AdamW (β1=0.8, β2=0.99, λ=0.01) |
| 初期学習率 | 2×10⁻⁴ |
| 学習率減衰 | 0.999^(1/8) / epoch |
| バッチサイズ | 256 |
| Generator学習 | 800kステップ |
| Duration Predictor別途学習 | 30kステップ |
| 精度 | Mixed Precision |

---

## 7. 結論

### VITS2移行は実行可能か？

**結論: 段階的導入であれば実行可能であり、特にマルチスピーカーモデルで有意な効果が期待できる。**

- **最小コスト・最大効果**: Noise-Scaled MAS (C) + Speaker-Conditioned TextEncoder (D) の2つで、約70行の変更で話者類似度+0.20とアライメント安定化が得られる
- **フル導入**: 約700〜1,200行の追加で全5改良を導入可能。ただしチェックポイント再学習が必要
- **既存拡張との互換**: piper-plus独自機能（Prosody, WavLM, EMA, SpeakerBalancedBatchSampler）はすべてVITS2と互換

### 次のステップ案

1. Noise-Scaled MAS (C) を実装し、既存モデルで効果を検証
2. Speaker-Conditioned TextEncoder (D) を実装し、20話者モデルで検証
3. 効果確認後、敵対的Duration Predictor (A) の実装に進む

---

## 8. VITS2以降のTTSアーキテクチャ動向 (2023-2026)

> **調査日**: 2026-03-14

### 8.1 VITS3は存在するか？

**正式な「VITS3」は存在しない。** VITS原著者チーム (Jaehyeon Kim等) による後継論文はVITS2以降発表されていない。ただし、VITSの設計思想を継承・発展させた多数の派生モデルが存在する。

### 8.2 TTSアーキテクチャの世代変遷

```
2021-2023: VITS系 (VAE + Normalizing Flow + GAN)
    ↓
2023-2024: Diffusion / Flow Matching系 (Matcha-TTS, StyleTTS2, F5-TTS)
    ↓
2024-2026: LLM + Codec系 ← 現在の主流
           (CosyVoice, Fish Speech, IndexTTS, Spark-TTS)
```

### 8.3 VITS系直系派生モデル

| モデル | 発表年 | アーキテクチャ | 日本語 | OSS | 備考 |
|--------|--------|--------------|--------|-----|------|
| **Bert-VITS2** | 2023 | VITS2 + 多言語BERT | 対応 | MIT | Fish Speechに移行 |
| **Style-Bert-VITS2** | 2024/02 | Bert-VITS2 + Style embedding | 対応 (JP-Extra) | AGPL-3.0 | 日本語特化、表現力向上 |
| **GPT-SoVITS** | 2024/02 | GPT + SoVITS (VITS改良版) | 対応 | OSS | v3以降はCFM-DiTに移行 |
| **Kokoro-82M** | 2025/01 | StyleTTS2ベース + ISTFTNet | 対応 | Apache-2.0 | 82Mパラメータ、$1000で学習可能 |

### 8.4 Flow Matching系モデル

| モデル | 発表年 | アーキテクチャ | 日本語 | OSS | 特徴 |
|--------|--------|--------------|--------|-----|------|
| **Matcha-TTS** | 2024 (ICASSP) | OT-CFM | 不明 | MIT | 少ステップで高品質 |
| **VoiceFlow** | 2024 (ICASSP) | Rectified Flow Matching | 不明 | OSS | ODE軌道の直線化 |
| **F5-TTS** | 2024/10 | Flow Matching + DiT | 対応 | OSS | アライメント不要、RTF 0.15 |
| **E2-TTS** | 2024 | Flow Matching Transformer | 不明 | 参照実装 | 簡素設計 |

### 8.5 LLM + Codec系モデル（現在の主流）

| モデル | 発表年 | アーキテクチャ | 日本語 | OSS | 特徴 |
|--------|--------|--------------|--------|-----|------|
| **CosyVoice 1.0** | 2024 | LLM + Flow Matching | 対応 | Apache-2.0 | Alibaba、多言語 |
| **CosyVoice 2.0** | 2024/12 | FSQ + streamlined LLM | 対応 | Apache-2.0 | 150ms低遅延streaming |
| **CosyVoice 3.0** | 2025/12 | LLM + RL学習 | 対応 | Apache-2.0 | 9言語+18方言、CER 0.81% |
| **Fish Speech 1.5** | 2024/11 | Dual-AR + GFSQ | 対応 | Apache-2.0 | 30万時間学習 |
| **IndexTTS 2.5** | 2025 | GPT Transformer + BigVGAN2 + RL | 対応 | OSS | 多言語感情音声 |
| **Spark-TTS** | 2025/03 | BiCodec + Qwen2.5 LLM | 不明 | OSS | CoT生成、Flow Matching不要 |
| **Chatterbox** | 2025 | Modified Llama (500M) | 対応 | MIT | Zero-shot cloning、23言語 |
| **Sesame CSM-1B** | 2025 | Llama + Mimi Audio Codec | 不明 | Apache-2.0 | 会話文脈保持 |
| **VibeVoice-1.5B** | 2025 | Qwen2.5 + σ-VAE + Diffusion | 不明 | MIT | 90分・4話者長文対応 |

### 8.6 その他の重要モデル

| モデル | 発表年 | アーキテクチャ | MOS (報告値) | 日本語 | OSS |
|--------|--------|--------------|-------------|--------|-----|
| **StyleTTS 2** | 2023/06 | Style Diffusion + WavLM adversarial | 人間超え CMOS+0.28 | 未対応 | MIT |
| **NaturalSpeech 2** | 2023 | Latent Diffusion + Neural Audio Codec | 人間レベル | 不明 | 非公開 |
| **NaturalSpeech 3** | 2024/03 | FACodec + Factorized Diffusion | 人間レベル | 不明 | 非公開 |

### 8.7 VITSベースのアプローチの現在地

**VITSは「最先端」ではないが、特定用途では依然として強み:**

| 観点 | VITSの強み | VITSの弱み |
|------|-----------|-----------|
| モデルサイズ | 軽量（数十MB）、エッジ展開可能 | — |
| 推論速度 | 非常に高速（リアルタイム×80以上） | — |
| 学習コスト | データ量・計算量が少ない | — |
| 安定性 | 学習が安定 | — |
| 音質 | — | 最新モデルに劣る |
| Zero-shot | — | 話者ごとの学習が必要 |
| テキスト理解 | — | LLMベースに劣る |
| スケーラビリティ | — | 大量データ・多言語で劣る |

### 8.8 piper-plusの次世代移行候補

軽量性を維持しつつ品質向上を目指す場合の候補:

| 候補 | 方式 | モデルサイズ | メリット | デメリット |
|------|------|------------|---------|-----------|
| **VITS2段階的導入** | VAE + Flow + GAN | ~74MB (現行同等) | 既存資産を活用、最小コスト | 改善幅は限定的 (+0.09 MOS) |
| **Kokoro-82M方式** | StyleTTS2ベース | ~82M | 軽量で高品質、$1000学習 | アーキテクチャ全面書換え |
| **F5-TTS方式** | Flow Matching + DiT | 数百MB | 音質大幅改善、日本語対応 | モデルサイズ増大 |
| **CosyVoice方式** | LLM + Flow Matching | 0.5B+ | 最高品質、多言語 | 重量級、エッジ展開困難 |

---

## 参考文献

### VITS2関連
- [VITS2 論文 (arXiv:2307.16430)](https://arxiv.org/abs/2307.16430)
- [VITS2 デモページ](https://vits-2.github.io/demo/)
- [p0p4k/vits2_pytorch](https://github.com/p0p4k/vits2_pytorch) — MIT, 最も機能的な非公式実装
- [daniilrobnikov/vits2](https://github.com/daniilrobnikov/vits2) — MIT
- [fishaudio/Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [litagin02/Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2)
- [ISCA Archive - Interspeech 2023](https://www.isca-archive.org/interspeech_2023/kong23_interspeech.html)

### VITS2以降のTTSアーキテクチャ
- [StyleTTS 2 (NeurIPS 2023)](https://arxiv.org/abs/2306.07691)
- [Matcha-TTS (ICASSP 2024)](https://arxiv.org/abs/2309.03199)
- [F5-TTS](https://arxiv.org/abs/2410.06885)
- [NaturalSpeech 3 (ICML 2024)](https://arxiv.org/abs/2403.03100)
- [CosyVoice 2](https://arxiv.org/abs/2412.10117)
- [CosyVoice 3.0](https://funaudiollm.github.io/cosyvoice3/)
- [Fish Speech 1.5](https://arxiv.org/abs/2411.01156)
- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
- [Spark-TTS](https://arxiv.org/abs/2503.01710)
- [IndexTTS 2.5](https://arxiv.org/html/2601.03888v1)
- [Chatterbox (Resemble AI)](https://github.com/resemble-ai/chatterbox)
- [Sesame CSM-1B](https://github.com/SesameAILabs/csm)
- [VibeVoice (Microsoft)](https://github.com/microsoft/VibeVoice)
- [Benchmarking VITS vs Style-BERT-VITS2 for Japanese](https://arxiv.org/html/2505.17320v1)
- [The State of TTS (Interspeech 2025)](https://www.isca-archive.org/interspeech_2025/srinivasavaradhan25_interspeech.pdf)
- [State of Voice AI 2024 (Cartesia)](https://cartesia.ai/blog/state-of-voice-ai-2024)
