# VITS+ 品質最大化戦略: 制約内での最高品質を目指して

## 概要

15エージェント並列調査 (2026-03-27) に基づく、piper-plus の品質最大化戦略。
制約 (34MB ONNX FP16 / CPU リアルタイム / 推論速度維持) を守りつつ、MOS ~3.8-4.0+ を目標とする。

**前提ドキュメント:** `docs/design/vits3-architecture-proposal.md` (アーキテクチャ変更の詳細)

---

## 品質改善の5層モデル

アーキテクチャ変更だけでなく、**5つの独立した品質改善レバー**が存在する。各層は独立に効果を発揮し、累積的に品質を向上させる。

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 5: 推論時最適化                          [コスト: 0]  │
│   noise_scale 言語別最適化、denoiser有効化、ONNX最適化     │
├─────────────────────────────────────────────────────────────┤
│ Layer 4: Phonemizer / G2P 改善             [パラメータ: 0]  │
│   テキスト正規化、異読語解消、リエゾン、多音字              │
├─────────────────────────────────────────────────────────────┤
│ Layer 3: データ改善                          [前処理のみ]   │
│   品質フィルタリング、合成データ追加、データセット拡大      │
├─────────────────────────────────────────────────────────────┤
│ Layer 2: 学習レシピ + Post-training      [推論モデル不変]   │
│   LR/EMA/損失関数改善、マルチタスク、UTMOS報酬、蒸留        │
├─────────────────────────────────────────────────────────────┤
│ Layer 1: アーキテクチャ変更                  [再学習必要]   │
│   Snake、Flow改善、Adversarial DP、iSTFT、パラメータ再配分  │
└─────────────────────────────────────────────────────────────┘
```

---

## Layer 5: 推論時最適化 (コスト0、即効)

### 5.1 noise_scale 言語別最適化

現在のデフォルト: `noise_scale=0.667`, `noise_scale_w=0.8`。言語/用途に応じた最適値:

| 言語/条件 | noise_scale | noise_scale_w | 理由 |
|-----------|-------------|---------------|------|
| JA/ZH (モーラ/音節拍) | 0.5-0.6 | 0.5-0.6 | 厳密なタイミング |
| EN/ES/FR/PT (ストレス拍) | 0.6-0.667 | 0.7-0.8 | duration 変動が自然 |
| 短い発話 (<10 phonemes) | 0.4-0.5 | 0.6 | エラー余地が少ない |
| 疑問文 | 0.667 | 0.9 | 韻律変動が大きい |
| 単話者ファインチューン | 0.5 | 0.5-0.6 | DP が多話者分布で学習済み |

### 5.2 Denoiser 有効化

`infer_onnx.py` に **既に実装済みだがコメントアウト** (line 432)。無音入力で bias spectrum を取得し、全出力から減算。

```python
# 現在コメントアウト → 有効化すべき
audio = denoise(audio, bias_spec, denoiser_strength=0.005)
```

### 5.3 ONNX Runtime 最適化

`benchmark_onnx.py` には最適化設定があるが `infer_onnx.py` には未適用:

```python
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.enable_mem_reuse = True
sess_options.enable_cpu_mem_arena = True
```

**推定効果:** Layer 5 全体で +0.05-0.15 MOS (noise_scale 最適化が最大の寄与)

---

## Layer 4: Phonemizer / G2P 改善 (パラメータ追加0)

### 4.1 全言語共通: テキスト正規化

**完全に未実装。** 数字、日付、略語、通貨等が未処理のまま phonemizer に渡される。

```
phonemize/normalize/
    base.py          # 抽象正規化クラス
    ja_normalize.py  # 日本語 (年月日、電話番号、通貨)
    en_normalize.py  # 英語 (序数、略語、$100→"one hundred dollars")
    zh_normalize.py  # 中国語 (万/億単位)
    es_normalize.py  # スペイン語 (性数一致: un/una)
    fr_normalize.py  # フランス語 (soixante-dix等)
    pt_normalize.py  # ポルトガル語 (性数一致)
```

### 4.2 言語別最優先改善

| 言語 | 改善 | 効果 | 工数 |
|------|------|------|------|
| **EN** | 異読語テーブル (top 100, POS-based) | 高 | 低 |
| **ZH** | `pypinyin-dict` phrase辞書 (1行追加) | 高 | 極低 |
| **FR** | 必須リエゾン (les/des/mes + 母音語) | 高 | 中 |
| **PT** | 前強勢母音弱化 (e→i, o→u) | 中 | 低 |
| **JA** | カタカナ未知語フォールバック | 中 | 低 |
| **ES** | 外来語例外辞書 | 低 | 極低 |

### 4.3 韻律特徴の全言語有効化

**重大な発見: EN/ES/FR/PT の prosody が完全にゼロ化されている。**

`models.py` line 897-910 の `_prepare_prosody_input()` で `prosody_language_ids = {0}` (JA のみ)。
EN は `nn.Linear(3, 16)` に常にゼロベクトルを入力。**既存インフラが完全に未活用。**

修正: 各言語の A1/A2/A3 に意味のある値を設定し、全言語を `prosody_language_ids` に追加:

| 言語 | A1 (提案) | A2 (現在/提案) | A3 (現在/提案) |
|------|----------|---------------|---------------|
| JA | アクセント核距離 (現状維持) | モーラ位置 (現状維持) | アクセント句長 (現状維持) |
| EN | **句境界距離** (新) | ストレス 0/1/2 (維持) | **音節数/prosodic phrase** (改善) |
| ZH | 声調 1-5 (維持) | **フレーズ内位置** (改善) | **フレーズ長** (改善) |
| ES | **語内音節位置** (新) | ストレス 0/2 (維持) | 語音素数 (維持) |
| FR | **語内音節位置** (新) | **フレーズ位置** (新) | 語音素数 (維持) |
| PT | **語内音節位置** (新) | ストレス 0/2 (維持) | 語音素数 (維持) |

**推定効果:** Layer 4 全体で +0.1-0.3 MOS (prosody 有効化 + テキスト正規化が最大)

---

## Layer 3: データ改善

### 3.1 品質フィルタリング (850h → 350-430K 発話)

カスケードパイプライン:

```
Stage 1: Duration/長さフィルタ        (除去 ~2-5%)
  └→ < 0.5s or > 20s、CPS外れ値 (>2.5σ/言語)
Stage 2: SNR フィルタ (WADA-SNR)      (除去 ~3-8%)
  └→ SNR >= 15dB
Stage 3: MOS フィルタ (UTMOS)          (除去 ~5-15%)
  └→ UTMOS >= 3.5 (言語別閾値)
Stage 4: 話者一貫性 (ECAPA-TDNN)      (除去 ~3-5%)
  └→ 話者重心との cosine sim >= 0.7
Stage 5: 書き起こし検証 (Whisper)     (除去 ~2-5%)
  └→ CER <= 30%
```

**推定効果:** +0.1-0.3 MOS。「Less is more」— 品質フィルタ後の350-430K発話が全508Kより高品質。

### 3.2 データセット拡大 (850h → 3,300h+, CC-BY-4.0)

| データセット | 言語 | 追加量 | ライセンス | インパクト |
|---|---|---|---|---|
| LibriTTS-R **全量** | EN | +460h | CC-BY-4.0 | 現在の4.7x |
| **WenetSpeech4TTS Premium** | ZH | +945h | CC-BY-4.0 | 現在の**12x** |
| VCTK | EN | +44h | CC-BY-4.0 | アクセント多様性 |
| MLS-Sidon ES | ES | +500h | CC-BY-4.0 | CML-TTS補完 |
| MLS-Sidon FR | FR | +500h | CC-BY-4.0 | CML-TTS補完 |
| MLS-Sidon PT | PT | +100h | CC-BY-4.0 | PT 2x 拡大 |
| Emilia-YODAS JA (filtered) | JA | +400h | CC-BY-4.0 | JA 5x 拡大 |

### 3.3 合成データ蒸留

CosyVoice 2/3 (Apache-2.0) で 2,000-3,000 時間の合成音声を生成:
- コスト: $100-500
- 不足言語 (PT/FR) に集中投下
- 50:50 real:synthetic で混合
- 2段階学習: Phase 1 mixed → Phase 2 real only

**推定効果:** Layer 3 全体で +0.2-0.4 MOS (フィルタリング + データ拡大 + 合成データの累積)

---

## Layer 2: 学習レシピ + Post-training (推論モデル不変)

### 2.1 学習レシピ修正 (前回提案 + 新発見)

| 改善 | 効果 | 工数 | 備考 |
|------|------|------|------|
| Cosine LR (eta_min=1e-5) | +0.02-0.1 | 極低 | 現在のExponentialLRは実質無効 |
| **Gradient norm clipping (1.0)** | 安定性大幅向上 | ~4行 | **未実装、全VITS後継で標準** |
| **EMA を model_g 全体に拡張** | +0.05-0.1 | **1行変更** | 現在 dec のみ |
| Multi-resolution mel loss | +0.1-0.2 | ~80行 | 512/1024/2048 FFT |
| KL annealing + free bits | 安定性向上 | ~15行 | |
| **n_heads 2→4** | 無料改善 | 設定変更 | パラメータ数同一 |

### 2.2 マルチタスク補助損失 (推論時0コスト)

| タスク | 接続点 | 学習params | 推論params | 効果 |
|--------|-------|-----------|-----------|------|
| CTC Phoneme Recognition | **z_p** (flow出力) | 37K | 0 | 音響空間の音素保持 |
| Language ID | z (posterior) | 25K | 0 | 言語混同防止 |
| Speaker Classification | z (posterior) | 160K | 0 | 話者分離改善 |
| WavLM Feature Distillation (回帰) | y_hat | 0 | 0 | 既存WavLM再利用 |
| Mel Predictor on Latent | z | 15K | 0 | 勾配ショートカット |
| F0 Auxiliary Loss | x (encoder出力) | 50K | 0 | ピッチ精度向上 |

### 2.3 韻律改善 (極少パラメータ)

| 改善 | 追加params | 効果 | 備考 |
|------|-----------|------|------|
| **Prosody を Flow global conditioning に追加** | ~600 | 高 | 韻律がpitch/energyにも影響 |
| **Hierarchical prosody convolution** (k=1/5/15) | ~362 | 中-高 | nn.Linear(3,16) 置換 |
| Sentence Prosody Predictor (attention pooling) | ~125K | 高 | 文レベル韻律区別 |
| OpenJTalk 拡張特徴 (E1/F/G fields) | ~112 | 中 | JA: 50+フィールド中3つしか使用していない |
| Accent type embedding | ~1.5K | 中 | JA: 平板/頭高/中高/尾高の区別 |

### 2.4 Post-training (UTMOS 報酬)

**CosyVoice 2/3 の「differentiable reward model」と同じ技法。** WavLM Discriminator と同じパターンで実装可能:

```python
# training_step_g に追加 (WavLM-D と同じパターン)
if self.mos_predictor is not None:
    mos_score = self.mos_predictor(y_hat.squeeze(1))
    loss_mos = -mos_score.mean() * self.hparams.c_mos
    loss_gen_all += loss_mos
```

- 学習済みモデルに対して 10-30 epoch の追加学習
- 元の損失を 0.3-0.5x に減衰、UTMOS 報酬を主目的化
- **推定効果: +0.1-0.3 MOS**

### 2.5 知識蒸留

| 手法 | 教師 | 効果 | 工数 |
|------|------|------|------|
| Duration/Prosody 蒸留 | CosyVoice 2 | +0.05-0.15 | 低-中 |
| Spectrogram L2 蒸留 | CosyVoice 2 | +0.05-0.15 | 低 |
| Pre-trained Phoneme MLM | 自己 (508K発話) | +0.1-0.2 | 中 |

**推定効果:** Layer 2 全体で +0.3-0.6 MOS (学習レシピ + マルチタスク + post-training + 蒸留)

---

## Layer 1: アーキテクチャ変更

前回の VITS+ 提案 (Phase 0-2) に加え、新たなパラメータ再配分の知見:

### 1.1 パラメータ再配分 (推論パラメータ内訳)

| コンポーネント | 現在 | % | 最適化後 |
|-------------|------|---|---------|
| TextEncoder | 6.4M | 32% | 6.0M (8層, FFN=512) |
| Flow | **10.2M** | **50%** | **7.7M** (WN 4→3層) |
| Generator | 1.8M | 9% | 3.1M (ResBlock1) |
| Duration Predictor | 1.8M | 9% | 0.5M (Adversarial DP) |
| **合計** | ~20.2M | | ~17.3M |

**核心: Flow が推論の50%を占有しているのに対し、Generator はわずか9%。** パラメータを Flow→Generator に再配分:
- Flow WN 層 4→3: -2.0M
- filter_channels 768→512: -1.4M
- ResBlock1 採用: +1.3M
- encoder 6→8 層: +1.2M

### 1.2 アーキテクチャ変更サマリー (前回提案 + 新発見)

| 変更 | 効果 | Phase |
|------|------|-------|
| **n_heads 2→4** (無料) | attention多様化 | Phase 0 |
| Snake activation | +0.15 PESQ | Phase 1 |
| Anti-aliased upsampling | +0.10 PESQ | Phase 1 |
| MR-STFT Discriminator | +0.1-0.2 PESQ | Phase 1 |
| **ResBlock2→1** | 音声忠実度向上 | Phase 1 |
| **filter_channels 768→512** | -1.4M節約 | Phase 1 |
| Flow mean_only=False | +5-10% 表現力 | Phase 2a |
| Flow dilation 1→2 | 受容野拡大 | Phase 2a |
| **Flow WN 4→3 層** | -2.0M節約 | Phase 2a |
| **Encoder 6→8 層** | 韻律改善 | Phase 2a |
| Adversarial DP | 自然なリズム | Phase 2b |
| iSTFT 最終段 | +20% 推論高速 | Phase 2c |

**推定効果:** Layer 1 全体で +0.3-0.5 MOS

---

## 品質到達予測 (全Layer累積)

### 楽観/中央/保守シナリオ

各Layerの効果は独立ではなく相互作用するため、単純加算ではなく 0.5-0.7x の割引を適用:

| 現在 → 目標 | 保守的 | 中央推定 | 楽観的 |
|---|---|---|---|
| Layer 5 (推論最適化) | +0.05 | +0.10 | +0.15 |
| Layer 4 (Phonemizer) | +0.10 | +0.15 | +0.25 |
| Layer 3 (データ改善) | +0.10 | +0.20 | +0.35 |
| Layer 2 (学習レシピ) | +0.15 | +0.25 | +0.40 |
| Layer 1 (アーキテクチャ) | +0.15 | +0.25 | +0.40 |
| **累積 (割引後)** | **+0.35** | **+0.55** | **+0.85** |
| **到達MOS** | **~3.85** | **~4.05** | **~4.35** |

```
MOS推定 (全Layer実装後)
4.7+ ── Fish S2 Pro         (4.4B, 10M時間)
4.5+ ── CosyVoice 3         (0.5-1.5B, 1M時間)
4.3  ── VITS+ 楽観的上限  ← 全Layer最良ケース ★
4.05 ── VITS+ 中央推定    ← 現実的な目標 ★★
3.85 ── VITS+ 保守的推定
3.5  ── piper-plus 現在
```

**中央推定 MOS ~4.05 は、30M params / 34MB / CPU リアルタイムモデルとしては前例のない水準。**

---

## 推奨実装ロードマップ

### Sprint 0 (1日): 即効改善 [Layer 5]
- noise_scale 言語別最適化 (パラメータスイープ)
- denoiser 有効化 (コメントアウト解除 + 強度調整)
- ONNX Runtime 最適化設定を infer_onnx.py に適用

### Sprint 1 (1週間): Phonemizer + データ前処理 [Layer 4 + 3]
- テキスト正規化 (全6言語)
- EN 異読語テーブル、ZH pypinyin-dict
- FR リエゾン、PT 前強勢母音弱化
- 全言語 prosody 有効化 (A1/A2/A3 意味づけ)
- UTMOS/SNR データフィルタリングパイプライン構築

### Sprint 2 (1週間): データ拡大 [Layer 3]
- LibriTTS-R 全量、WenetSpeech4TTS Premium
- MLS-Sidon ES/FR/PT、Emilia-YODAS JA (filtered)
- CosyVoice 2 合成データ生成 (PT/FR 重点)
- 全データの LJSpeech 形式変換 + フィルタリング

### Sprint 3 (3-5日): 学習レシピ刷新 [Layer 2 基本]
- Cosine LR + Gradient norm clipping
- EMA model_g 全体化
- n_heads 2→4
- Multi-resolution mel loss
- KL annealing + free bits
- → **拡大データで学習開始**

### Sprint 4 (1-2週間): マルチタスク + アーキテクチャ [Layer 2 + 1]
- CTC (z_p), LID, Speaker Classification 補助損失
- Prosody → Flow global conditioning
- Hierarchical prosody convolution
- Snake activation + AMP
- ResBlock1, filter_channels=512
- Flow WN 3層, encoder 8層
- MR-STFT Discriminator
- → **フルアーキテクチャで再学習**

### Sprint 5 (1週間): Post-training [Layer 2 応用]
- UTMOS 微分可能報酬での追加学習 (10-30 epoch)
- Duration/Prosody 蒸留 (CosyVoice 2 ターゲット)
- Best-of-N rejection sampling fine-tuning
- → **最終モデル完成**

### Sprint 6 (3-5日): 検証 + リリース
- 6言語 A/B テスト (36発話)
- ONNX パリティテスト
- C#/Rust/WASM 推論パイプラインテスト
- FP16 + Snake 精度テスト
- ファインチューニングワークフロー検証

---

## 参考文献 (追加分)

| カテゴリ | 参考 |
|---------|------|
| データ品質 | UTMOS (Saeki+, 2022), DNSMOS (Microsoft), TTS-CURATE (NeurIPS 2024) |
| 合成データ | KokoroTTS data policy, SimpleSpeech (Yang+, 2024) |
| Post-training | CosyVoice 2 differentiable reward, Fish S2 GRPO, SpeechAlign DPO |
| マルチタスク | CosyVoice 3 multi-task tokenizer, DistilHuBERT (Chang+, 2022) |
| Phonemizer | Misaki (KokoroTTS G2P), pypinyin-dict, DeepPhonemizer |
| データセット | WenetSpeech4TTS, MLS-Sidon, Emilia-YODAS, VCTK, HiFi-TTS |
| アーキテクチャ | AutoTTS NAS (Miao+, 2024), LightSpeech NAS (Luo+, 2021) |
| 韻律 | PortaSpeech (Ren+, 2022), ProsoSpeech (Ren+, 2022) |
| 日本語 | Style-Bert-VITS2, VOICEVOX, OJAD accent dictionary |
| 推論最適化 | VITS noise_scale tuning, HiFi-GAN bias removal |
