# VITS+ アーキテクチャ提案: piper-plus 品質向上ロードマップ

## 概要

piper-plus の音声品質を現行 VITS1 ベースから大幅に向上させる「VITS+」アーキテクチャの設計提案。モデルサイズ (~34MB ONNX FP16) と推論速度を維持しつつ、34MB/CPUクラスでの最高品質を目指す。

**調査日**: 2026-03-27
**調査手法**:
- 第1ラウンド: 10エージェント並列調査 (VITS2論文, KokoroTTS, CosyVoice, 最新TTS手法, Discriminator, Flow, Duration, Encoder/Decoder, 学習テクニック + コードベース分析)
- レビュー: 5エージェント並列レビュー (技術正確性, 実現可能性/リスク, 実装品質, 競合分析, 学習コスト)
- 第2ラウンド: 15エージェント並列深掘り調査 (知識蒸留, 合成データ, テキストエンコーダ, データ品質, Post-training, 韻律, アーキテクチャ配分, 日本語特化, 英語特化, 推論最適化, 学習テクニック, データセット, マルチタスク, Phonemizer, 小型モデルSOTA)

> **関連ドキュメント:** 第2ラウンド調査の統合レポートは [`vits-plus-quality-maximization.md`](vits-plus-quality-maximization.md) を参照。本ドキュメントのアーキテクチャ変更 (Layer 1) に加え、推論最適化 (Layer 5)、Phonemizer改善 (Layer 4)、データ改善 (Layer 3)、学習レシピ・Post-training (Layer 2) の4つの追加品質改善レバーを記述。

> **名称について:** 当初「VITS3」と仮称していたが、レビューにより「VITS+」に変更。
> - VITS2 は Kong et al. (2023) の学術論文。「VITS3」は後継論文を暗示し混乱を招く
> - 本提案は VITS2 の一部 + BigVGAN + 学習テクニック + データ戦略の組み合わせであり、世代的飛躍ではない
> - 「VITS+」は piper-plus のブランディングと整合

---

## 目次

1. [競合分析](#1-競合分析)
2. [現在の実装状態](#2-現在の実装状態)
3. [既存Issue/PRとの関係](#3-既存issueprとの関係)
4. [VITS+ アーキテクチャ変更サマリー](#4-vits3-アーキテクチャ変更サマリー)
5. [Phase 0: 学習レシピ修正](#5-phase-0-学習レシピ修正)
6. [Phase 1: Generator改善](#6-phase-1-generator改善)
7. [Phase 2: Flow + Duration改善](#7-phase-2-flow--duration改善)
8. [Phase 3: 長期的検討](#8-phase-3-長期的検討)
9. [品質到達予測](#9-品質到達予測)
10. [推奨実装順序](#10-推奨実装順序)
11. [参考文献](#11-参考文献)

---

## 1. 競合分析

### KokoroTTS

| 項目 | 詳細 |
|------|------|
| アーキテクチャ | StyleTTS 2 ベース (VITS系ではない) |
| パラメータ数 | 82M |
| テキストエンコーダ | PLBert (12層, 768dim, 12ヘッド) |
| 韻律モデリング | Style Diffusion (潜在変数として韻律を生成) |
| デコーダ | ISTFTNet (upsample 10x→6x + iSTFT) |
| Discriminator | WavLM (SLM as Discriminator) |
| 対応言語 | 9言語, 39+ボイス |
| 推論 | GPU推奨 |

**品質の鍵:**
1. **Style Diffusion** -- 韻律を潜在確率変数として拡散モデルで生成。VITSの「平坦な韻律」問題を根本解決
2. **PLBert** -- 12層BERTベースのテキスト理解。文脈依存の強勢・リズムを実現
3. **ISTFTNet** -- iSTFTによる位相再構成でTransposed Convのアーティファクト除去
4. **WavLM Discriminator** -- 人間の聴覚に近い知覚空間での品質判定
5. **Snake Activation** -- 音声波形の周期性に最適化された活性化関数

### CosyVoice

| 項目 | 詳細 |
|------|------|
| アーキテクチャ | LLM + Conditional Flow Matching (2段階) |
| パラメータ数 | ~2.5B |
| Stage 1 | Qwen LLM: テキスト → 離散音声トークン (autoregressive) |
| Stage 2 | CFM: 音声トークン → mel → HiFi-GAN → 波形 |
| トークン | Supervised Semantic Tokens (単一コードブック) |
| 学習データ | ~170,000時間 (多言語) |
| 推論 | GPU必須 (RTF ~0.5-2x) |

**品質の鍵:**
1. **Supervised Semantic Tokens** -- ASR監視で学習された意味的音声トークン。ハルシネーション大幅削減
2. **LLM autoregressive** -- 長距離の韻律パターンを自然にモデリング
3. **Conditional Flow Matching** -- 拡散モデルより少ないステップ (10-30) で高品質生成
4. **データスケール** -- 170K時間のデータは品質の根本要因
5. **CosyVoice 2 (2024年12月)** -- FSQ (Finite Scalar Quantization) でコードブック崩壊を防止、ストリーミング対応

### CosyVoice 3 (2025年12月)

| 項目 | 詳細 |
|------|------|
| アーキテクチャ | LLM + マルチタスク監視学習トークナイザ + Flow Matching |
| パラメータ数 | **0.5B / 1.5B** (2モデル) |
| 学習データ | **100万時間**, 9言語 + 18中国語方言 |
| 新技術 | マルチタスク監視学習音声トークナイザ (ASR, 感情認識, 話者分析等)、微分可能報酬モデル (post-training) |
| 機能 | 発音修正 (Pronunciation Inpainting)、速度/感情/音量制御 (Instruct) |
| 遅延 | 150ms (ストリーミング) |
| 推論 | GPU必須 |

**参考:** [CosyVoice 3.0 公式](https://funaudiollm.github.io/cosyvoice3/) / [論文 (arXiv)](https://arxiv.org/html/2505.17589v2) / [HuggingFace](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)

### Fish Audio S2 Pro (2026年3月)

| 項目 | 詳細 |
|------|------|
| アーキテクチャ | Dual-AR: Slow AR (4B, 時間軸) + Fast AR (400M, 深さ軸) + RVQ codec (10コードブック) |
| パラメータ数 | **~4.4B** |
| 学習データ | **1,000万時間**, 80+言語 |
| Post-training | Group Relative Policy Optimization (GRPO) + multi-dimensional reward |
| 品質 | gpt-4o-mini-tts に対して **win rate 81.88%** (オープン・クローズド含む最高スコア) |
| 制御 | `[laugh]`, `[whispers]`, `[super happy]` 等の自然言語タグでインライン制御 |
| 遅延 | ~100ms (H200 GPU) |
| ライセンス | 研究・非商用のみ無料 |

**参考:** [Fish Audio S2 公式](https://fish.audio/s2/) / [Technical Report (arXiv)](https://arxiv.org/html/2603.08823v2) / [HuggingFace](https://huggingface.co/fishaudio/s2-pro)

### 比較テーブル (2026年3月時点)

| 項目 | piper-plus | KokoroTTS | CosyVoice 3 | Fish S2 Pro |
|------|-----------|-----------|-------------|-------------|
| パラダイム | VAE+GAN+Flow | Style Diffusion | LLM+CFM | Dual-AR+GRPO |
| パラメータ | ~30M | 82M | 0.5-1.5B | ~4.4B |
| 学習データ | ~850時間 | ~数百時間 | **100万時間** | **1,000万時間** |
| モデルサイズ | **34MB** | ~170MB | ~2-6GB | ~16GB+ |
| RTF (CPU) | **リアルタイム** | 遅い | 不可能 | 不可能 |
| ライセンス | **Apache-2.0** | Apache-2.0 | Apache-2.0 | 非商用のみ |
| Zero-shot | 非対応 | スタイルベクトル | 3秒プロンプト | 10-30秒参照 |
| 感情制御 | なし | 限定的 | Instruct | 自然言語タグ |
| 品質 (MOS推定) | ~3.5 | ~4.0-4.5 | ~4.5+ | ~4.7+ |

**結論:** 2026年3月時点のSOTA TTS (CosyVoice 3, Fish S2 Pro) は、パラメータ数 (50-150倍)、学習データ (1,200-12,000倍)、アーキテクチャ (LLM/Dual-AR + RLHF post-training) のすべてで根本的に異なるリーグにある。piper-plus が品質で直接競合することは不可能。

**piper-plus の競争優位は「品質」ではなく「展開効率」:**
- **34MB / CPU / 完全オフライン / Apache-2.0** というニッチで唯一無二
- 組込み、IoT、プライバシー重視、モバイルオフライン、ブラウザWASM で最適解
- VITS+ で MOS ~3.8-4.0 を達成すれば、**同サイズクラス (50MB以下/CPU) では世界最高品質**

---

## 2. 現在の実装状態

### アーキテクチャ詳細 (第2ラウンド調査で精密計算)

| コンポーネント | 実装 | パラメータ | 推論時 | % (推論) |
|-------------|------|----------|-------|---------|
| TextEncoder | Transformer 6L, 2H, 192dim, FFN 768 | ~6.4M | Yes | 32% |
| StochasticDurationPredictor | 4+4 ConvFlow + DDSConv | ~1.8M | Yes | 9% |
| PosteriorEncoder | WN 16L, k=5, d=1, 192dim | ~10.4M | **No** (学習時のみ) | - |
| Flow | 4 WN coupling (4L each), mean_only=True | **~10.2M** | Yes | **50%** |
| Generator (HiFi-GAN) | ResBlock2, upsample (8,8,4), 256ch | ~1.8M | Yes | 9% |
| Embeddings/Prosody | emb_lang, prosody_proj | ~3K | Yes | <1% |
| **学習時合計** | | **~30.6M** | | |
| **推論時合計** | | **~20.2M** | | |

> **重要な発見 (第2ラウンド):**
> - **PosteriorEncoder (~10.4M, 33%) は ONNX エクスポートで除去される。** 推論モデルは実質 ~20.2M params (~38.6MB FP16 raw)
> - **Flow が推論パラメータの 50% を占有** (~10.2M) — 過剰割当の可能性が高い
> - **Generator はわずか 9%** (~1.8M) — パラメータ再配分の余地大
> - 初期提案の「~2M」「~0.5M」等は過小評価だった (gin_channels=512 の conditioning layer を未計上)
| ResidualCouplingBlock (Flow) | 4 WN coupling, mean_only=True, Flip | ~8M |
| Generator (HiFi-GAN) | ResBlock2, upsample (8,8,4), 256ch | ~7M |
| Discriminator (MPD) | DiscS + DiscP [2,3,5,7,11] | 学習時のみ |
| WavLM Discriminator | WavLM-base-plus, layers [6,9,12] | 学習時のみ |

### 既知の制限

**アーキテクチャ:**
1. **Flow: mean_only=True** -- スケールパラメータなし。各カップリング層はシフトのみで表現力が制限
2. **Flow: Flip のみ** -- 学習可能な置換なし (Glow の 1x1 conv が欠如)
3. **Flow: dilation_rate=1** -- 受容野が ~17 フレームと狭い
4. **Flow: 推論の50%を占有** -- WN 4層 x 4 coupling で ~10.2M。Generator (9%) と不均衡
5. **Generator: LeakyReLU** -- 音声の周期性に対する帰納バイアスなし
6. **Generator: ResBlock2** -- ResBlock1 より低品質だが軽量。パラメータに余裕あり
7. **TextEncoder: n_heads=2** -- 4 heads にしてもパラメータ数同一 (無料改善の未活用)
8. **Duration: SDP が複雑** -- 4+4 ConvFlow の normalizing flow は推論コストが高く、品質面で adversarial DP に劣る

**学習レシピ:**
9. **LRスケジュール: 実質無効** -- ExponentialLR gamma=0.999875, per-epoch stepping で75epochの総減衰率0.93%
10. **Gradient norm clipping: 未実装** -- VITS2/Matcha-TTS/StyleTTS2 すべてで標準的に使用
11. **EMA: dec のみに限定** -- `model_g` 全体に拡張すべき (1行変更)
12. **損失関数: 単一解像度mel** -- 1024 FFT のみ。マルチスケール情報の欠如

**韻律・Phonemizer (第2ラウンド調査で発見):**
13. **EN/ES/FR/PT の prosody が完全にゼロ化** -- `prosody_language_ids = {0}` (JA のみ)。nn.Linear(3,16) に常にゼロベクトル
14. **Prosody が Duration Predictor にしか届かない** -- Flow/Generator は韻律情報を受け取らない
15. **OpenJTalk の 50+ フィールド中 3つしか使用していない** -- E1(アクセント型), F1/F2(呼気段落) 等が未活用
16. **テキスト正規化が全言語で未実装** -- 数字、日付、略語がそのまま phonemizer に渡される
17. **F0 Predictor がコードに存在するが未接続** -- `ONNXFriendlyF0Predictor` in `onnx_attention.py`

**推論:**
18. **denoiser が実装済みだがコメントアウト** -- `infer_onnx.py` line 432
19. **ONNX Runtime 最適化が未適用** -- `benchmark_onnx.py` にはあるが `infer_onnx.py` には未設定

**データ:**
20. **ZH データが不足** -- AISHELL-3 の 85h に対し、WenetSpeech4TTS (CC-BY-4.0) で +945h が即座に利用可能

---

## 3. 既存Issue/PRとの関係

| Issue/PR | 内容 | 状態 | VITS+での扱い |
|----------|------|------|-------------|
| PR #212 | WavLM Discriminator | **実装済み** | 維持 (KokoroTTSと同等) |
| PR #196 | A1/A2/A3 Prosody | **実装済み** | 維持 |
| PR #239 | FP16 ONNX | **実装済み** | 維持 |
| Issue #268 | MB-iSTFT-VITS2 Decoder | 調査済み・未実装 | **Phase 2で採用** |
| Issue #200 | MR-STFT Discriminator | 見送り | **再検討 → Phase 1で採用** |
| Issue #201 | データ拡張 | NOT_PLANNED | Fine-tuning時のみ検討 |
| Issue #202 | Duration正則化 | NOT_PLANNED | Adversarial DPで代替 |
| Issue #206 | 離散Prosody Embedding | NOT_PLANNED | 優先度低 |

### Issue #200 再検討の理由

当初の見送り理由は「WavLM Discriminator と重複し diminishing returns」だったが、調査の結果:

- MR-STFT-D は**スペクトル/位相コヒーレンス**を判定 (周波数領域)
- WavLM-D は**知覚品質**を判定 (意味的特徴空間)
- MPD は**波形周期性**を判定 (時間領域)

三者は**相補的**であり、重複ではない。BigVGAN v2, DAC, EnCodec 等の現代システムはすべて MPD + MR-STFT-D の組み合わせを採用。WavLM-D + MR-STFT-D の組み合わせは学習コスト増 (+10-20%) だが、WavLM-D は `--wavlm-every-n-steps` で間引き可能なため、MR-STFT-D は毎ステップ、WavLM-D は N ステップ毎とすれば共存可能。

---

## 4. VITS+ アーキテクチャ変更サマリー

```
VITS1 (現在)                              VITS+ (提案)
──────────────────────────────────────────────────────────────────
TextEncoder                               TextEncoder (変更なし)
  └ Transformer 6L, 2H, 192dim             └ 同左

StochasticDurationPredictor               Adversarial Duration Predictor [Phase 2]
  └ 4 ConvFlow (normalizing flow)           └ シンプルCNN DP + Duration Discriminator
                                            └ CTC 補助損失 (多言語alignment改善)

PosteriorEncoder                          PosteriorEncoder (変更なし)
  └ WN 16L, k=5, d=1                       └ 同左

ResidualCouplingBlock                     Improved Flow [Phase 2]
  └ 4 WN coupling, mean_only=True          └ 4 coupling, mean_only=False (scale有効化)
  └ Flip                                    └ 1x1 Invertible Conv (学習可能な置換)
                                            └ dilation_rate 1→2 (受容野 17→61)

Generator (HiFi-GAN)                      Generator (HiFi-GAN + Snake + iSTFT) [Phase 1-2]
  └ LeakyReLU                               └ Snake activation (周期的帰納バイアス)
  └ ConvTranspose ×3 (8,8,4)               └ Anti-aliased upsampling (Kaiser FIR)
  └ ResBlock2                               └ iSTFT 最終段 (Issue #268, Phase 2)

MultiPeriodDiscriminator                  MPD + MR-STFT-D [Phase 1, 学習時のみ]
  └ MPD + MSD                               └ MPD + MSD + Multi-Resolution STFT-D

WavLMDiscriminator                        WavLMDiscriminator (変更なし)
  └ layers [6,9,12]                         └ 同左

Losses                                    Improved Losses [Phase 0, 学習時のみ]
  └ mel L1 (c_mel=45)                       └ Multi-resolution mel loss (512/1024/2048)
  └ c_kl=1.0 (固定)                         └ KL annealing + free bits
  └ ExponentialLR (実質無効)                  └ Cosine annealing + warmup
  └ feature_loss (均一)                      └ R1 gradient penalty
                                             └ Layer-weighted feature matching
```

---

## 5. Phase 0: 学習レシピ修正

**リスク: 最小。チェックポイント互換。推論モデル変更なし。**

Phase 0 は既存チェックポイントから継続学習可能であり、推論グラフに一切変更がない「無料の改善」群。

### 5.1 LRスケジュール修正 (Cosine Annealing + Warmup)

**問題:** 現在の `ExponentialLR(gamma=0.999875)` は per-epoch stepping で、75 epoch での総減衰率はわずか 0.93%。実質的にスケジュールが機能していない。

**解決:**

```python
# lightning.py: configure_optimizers()
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

warmup = LinearLR(opt_g, start_factor=0.01, total_iters=5)  # 5 epoch warmup
cosine = CosineAnnealingLR(opt_g, T_max=max_epochs - 5, eta_min=1e-6)
scheduler_g = SequentialLR(opt_g, [warmup, cosine], milestones=[5])
# 同様に opt_d 用も作成
```

**CLIオプション:** `--lr-schedule cosine` (デフォルト), `--lr-warmup-epochs 5`, `--lr-min 1e-6`

**推定効果:** PESQ +0.02-0.1

### 5.2 Multi-Resolution Mel Spectrogram Loss

**概要:** 現在の単一解像度 (FFT=1024) mel L1 に加え、FFT=512 と FFT=2048 でもmel損失を計算。短窓は子音の過渡応答、長窓は周波数分解能を改善。

**実装:**

```python
# 新ファイル: vits/mr_stft_loss.py
class MultiResolutionMelLoss(nn.Module):
    def __init__(self, resolutions=[(512, 128, 512), (1024, 256, 1024), (2048, 512, 2048)]):
        ...
    def forward(self, y, y_hat):
        loss = 0
        for n_fft, hop, win in self.resolutions:
            S_real = mel_spectrogram_torch(y, n_fft, 80, self.sample_rate, hop, win, ...)
            S_gen = mel_spectrogram_torch(y_hat, n_fft, 80, self.sample_rate, hop, win, ...)
            loss += F.l1_loss(S_real, S_gen)
        return loss / len(self.resolutions)
```

**CLIオプション:** `--multi-resolution-mel-loss` (デフォルト有効), `--c-mr-mel 45`

**推定効果:** PESQ +0.1-0.2

### 5.3 R1 Gradient Penalty

**概要:** Discriminatorの実データに対する勾配ノルムにペナルティを課す正則化。Discriminatorの過学習を防ぎ、GAN学習を安定化。StyleTTS 2, BigVGAN v2 で採用。

**実装:**

```python
# lightning.py: training_step_d 内
# 16ステップ毎に適用 (コスト削減)
if self.global_step % 16 == 0:
    y_real = y.requires_grad_(True)
    y_d_hat_r, _, _, _ = self.model_d(y_real, y_hat.detach())
    r1_grads = torch.autograd.grad(
        outputs=[s.sum() for s in y_d_hat_r],
        inputs=y_real,
        create_graph=True,
    )
    r1_penalty = sum(g.pow(2).sum() for g in r1_grads) / y.shape[0]
    loss_disc_all += (self.hparams.r1_gamma / 2) * r1_penalty
```

**CLIオプション:** `--r1-gamma 5.0` (デフォルト), `--r1-every-n-steps 16`

**推定効果:** PESQ +0.05-0.15、学習安定性向上

### 5.4 KL Annealing + Free Bits

**概要:** KL重みを学習初期は0から徐々に1.0まで上昇させ、VAE潜在空間の学習を安定化。Free bits (最低KL閾値) で個別次元のKL崩壊を防止。

**実装:**

```python
# lightning.py: training_step_g 内
if self.hparams.kl_warmup_epochs > 0:
    kl_weight = min(1.0, self.current_epoch / self.hparams.kl_warmup_epochs)
else:
    kl_weight = 1.0
loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask,
                  free_bits=self.hparams.free_bits) * self.hparams.c_kl * kl_weight

# losses.py: kl_loss 内
def kl_loss(z_p, logs_q, m_p, logs_p, z_mask, free_bits=0.0):
    kl = logs_p - logs_q - 0.5 + 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    if free_bits > 0:
        kl = torch.clamp(kl, min=free_bits)
    return torch.sum(kl * z_mask) / torch.sum(z_mask)
```

**CLIオプション:** `--kl-warmup-epochs 10`, `--free-bits 0.1`

**推定効果:** 学習安定性向上、特にファインチューニング時に効果大

### 5.5 Feature Matching 層重み付け

**概要:** Discriminatorの深い層ほど高レベルの特徴を捉える。均一重みから線形増加重みに変更。

**実装:**

```python
# losses.py
def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        n_layers = len(dr)
        for i, (rl, gl) in enumerate(zip(dr, dg)):
            rl = rl.float().detach()
            gl = gl.float()
            weight = (i + 1) / n_layers  # 線形増加
            loss += weight * torch.mean(torch.abs(rl - gl))
    return loss * 2
```

**推定効果:** PESQ +0.02-0.05

### Phase 0 合計

| 改善 | 推定効果 | コード変更量 | チェックポイント互換 |
|------|---------|------------|-------------------|
| LRスケジュール修正 | +0.02-0.1 PESQ | ~15行 | 互換 |
| Multi-resolution mel loss | +0.1-0.2 PESQ | ~80行 (新ファイル) | 互換 |
| R1 gradient penalty | +0.05-0.15 PESQ | ~20行 | 互換 |
| KL annealing + free bits | 安定性向上 | ~15行 | 互換 |
| Feature matching 重み付け | +0.02-0.05 PESQ | ~10行 | 互換 |
| **合計** | **PESQ +0.2-0.4** | **~140行** | **全て互換** |

---

## 6. Phase 1: Generator改善

**リスク: 低-中。推論モデル変更あり。再学習必要。ONNXサイズ同等。**

### 6.1 Snake Activation

**概要:** LeakyReLU を Snake activation `x + (1/α) * sin²(αx)` に置換。音声波形の周期構造に対する帰納バイアスを提供。KokoroTTS (StyleTTS 2), BigVGAN, BigVGAN v2 の全てが採用する、現代音声合成の事実上の標準。

**なぜ効くのか:** LeakyReLU は非周期的な活性化関数であり、ニューラルネットワークが音声の周期成分を学習するのを自然には支援しない。Snake は `sin²` を含むことで周期的な信号の生成を容易にし、高調波の再構成品質を大幅に改善する。

**実装:**

```python
# modules.py: 新クラス
class SnakeActivation(nn.Module):
    def __init__(self, channels, alpha_init=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1) * alpha_init)

    def forward(self, x):
        return x + (1.0 / self.alpha) * torch.sin(self.alpha * x) ** 2
```

**適用箇所:**
- `Generator.forward()` の `F.leaky_relu(x, LRELU_SLOPE)` → `self.snake(x)`
- `ResBlock1` / `ResBlock2` 内の LeakyReLU → Snake

**CLIオプション:** `--snake-activation` (デフォルト無効、互換性のため)

**パラメータ影響:** +~1K params (α は per-channel scalar)。ONNXサイズ影響なし。

**ONNX互換性:** `sin` はONNX標準op。問題なし。

**推定効果:** PESQ +0.15、「メタリック」アーティファクト大幅低減

### 6.2 Anti-Aliased Upsampling (AMP)

**概要:** Transposed Convolution 後に Kaiser 窓 FIR ローパスフィルタを適用し、エイリアシングを除去。BigVGAN の Anti-aliased Multi-Periodicity Composition (AMP) ブロックから。

**なぜ効くのか:** Transposed Convolution は本質的にエイリアシングを生成する (ナイキスト周波数を超える折り返し成分)。これが音声の「ブーンという音」や金属的なアーティファクトの主要因。ローパスフィルタでこれを除去する。

**実装:**

```python
# modules.py: Kaiser FIR lowpass filter
class LowPassFilter1d(nn.Module):
    def __init__(self, cutoff_freq, kernel_size=12, beta=14.0):
        super().__init__()
        # Kaiser窓sinc FIRフィルタ (固定係数、学習パラメータ0)
        filter_coeffs = self._design_kaiser_lowpass(cutoff_freq, kernel_size, beta)
        self.register_buffer('filter', filter_coeffs)

    def forward(self, x):
        return F.conv1d(x, self.filter, groups=x.shape[1], padding='same')
```

**適用箇所:** 各 `ConvTranspose1d` の後に挿入

**パラメータ影響:** 0 (FIR係数は固定バッファ)

**推定効果:** PESQ +0.10 (Snake との組み合わせで累積 +0.25)

### 6.3 Multi-Resolution STFT Discriminator (MR-STFT-D)

**概要:** 複数のFFTサイズ (256/512/1024/2048) でSTFTを計算し、その実部+虚部 (2チャネル) を2D CNNで判別。スペクトル/位相コヒーレンスを直接的に評価。

**なぜ追加するのか (Issue #200 再検討):**
- MPD: 波形周期性 (時間領域)
- WavLM-D: 知覚品質 (意味的特徴空間)
- **MR-STFT-D: スペクトル/位相コヒーレンス (周波数領域)** ← 現在カバーなし

三者は相補的。MR-STFT-D のV100追加VRAMは +0.5-1GB で、WavLM-D (~1-2GB) より軽量。

**実装:**

```python
# models.py: 新クラス
class MultiResolutionSTFTDiscriminator(nn.Module):
    def __init__(self, resolutions=[(2048, 512, 2048), (1024, 256, 1024),
                                     (512, 128, 512), (256, 64, 256)]):
        super().__init__()
        self.discriminators = nn.ModuleList([
            STFTSubDiscriminator(n_fft, hop, win) for n_fft, hop, win in resolutions
        ])

    def forward(self, y, y_hat):
        # 既存の discriminator_loss / feature_loss インターフェースに準拠
        ...

class STFTSubDiscriminator(nn.Module):
    def __init__(self, n_fft, hop_size, win_size):
        super().__init__()
        # Input: complex STFT (2ch: real + imag) -> 2D Conv layers
        self.convs = nn.ModuleList([
            nn.Conv2d(2, 32, (3, 9), padding=(1, 4)),
            nn.Conv2d(32, 128, (3, 9), stride=(1, 2), padding=(1, 4)),
            nn.Conv2d(128, 512, (3, 9), stride=(1, 2), padding=(1, 4)),
            nn.Conv2d(512, 1024, (3, 3), padding=(1, 1)),
            nn.Conv2d(1024, 1, (3, 3), padding=(1, 1)),
        ])
```

**CLIオプション:** `--mr-stft-discriminator` (デフォルト有効), `--c-mr-stft 1.0`

**推論影響:** なし (学習時のみ)

**推定効果:** PESQ +0.1-0.2

### Phase 1 合計

| 改善 | 推定効果 | パラメータ影響 | 推論速度影響 |
|------|---------|-------------|------------|
| Snake activation | +0.15 PESQ | +~1K (~0%) | -5% (無視可能) |
| Anti-aliased upsampling | +0.10 PESQ | 0 (固定FIR) | -5% (無視可能) |
| MR-STFT Discriminator | +0.1-0.2 PESQ | 推論影響なし | なし |
| **合計** | **PESQ +0.3-0.5** | **~0%** | **~-10%** |

---

## 7. Phase 2: Flow + Duration改善

**リスク: 中。アーキテクチャ変更あり。再学習必要。ONNXサイズ微減。**

### 7.1 Adversarial Duration Predictor (VITS2)

**概要:** 現在のStochasticDurationPredictor (4 ConvFlow) を、シンプルなCNN + GAN判別器に置換。VITS2 の核心的改善。

**なぜ効くのか:** SDP は最尤推定 (ELBO) で学習するが、これは知覚的に良いリズムを保証しない。Adversarial training はリアルなduration分布を直接学習するため、韻律の自然さが向上。

**現在のSDP問題点:**
- 4 ConvFlow + DDSConv は推論コストが高い
- normalizing flow の逆変換が必要
- 韻律が平滑化される傾向

**実装:**

```python
# models.py: 新クラス
class DurationDiscriminator(nn.Module):
    """VITS2スタイルの Duration 判別器"""
    def __init__(self, in_channels, filter_channels=256, kernel_size=3,
                 n_layers=3, gin_channels=0):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv1d(in_channels + 1, filter_channels, kernel_size, padding=1))
        for _ in range(n_layers - 1):
            self.convs.append(nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=1))
        self.proj = nn.Conv1d(filter_channels, 1, 1)
        if gin_channels > 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

    def forward(self, x, dur, x_mask, g=None):
        # x: text encoder hidden [B, C, T]
        # dur: log-duration [B, 1, T]
        h = torch.cat([x, dur], dim=1) * x_mask
        for conv in self.convs:
            h = F.leaky_relu(conv(h * x_mask), 0.1)
            if g is not None:
                h = h + self.cond(g)
        return self.proj(h * x_mask)
```

**学習:**
- Duration Discriminator は MAS の real duration vs DP の predicted duration を判別
- Generator側: SDP を通常のDPに置換 + adversarial loss
- 既存の2-optimizer構造 (G + D) に Duration D を D optimizer に追加

**CLIオプション:** `--adversarial-dp` (デフォルト有効), `--c-dur-adv 1.0`

**パラメータ影響:** SDP (-0.5M) を除去し、Duration Discriminator (+~50K, 学習時のみ) を追加。推論モデルは**縮小**。

**推定効果:** MOS +0.05-0.1、韻律の自然さ向上

### 7.2 Flow Scale有効化 (mean_only=False)

**概要:** ResidualCouplingLayer の `mean_only=True` を `False` に変更。各カップリング層でシフトに加えスケール変換を有効化し、flow の表現力を倍増。

**実装:**

```python
# models.py: ResidualCouplingBlock 内
# 変更前:
self.flows.append(
    modules.ResidualCouplingLayer(..., mean_only=True)
)
# 変更後:
self.flows.append(
    modules.ResidualCouplingLayer(..., mean_only=False)
)
```

**安定化:** log-scale のクランプを追加

```python
# modules.py: ResidualCouplingLayer.forward() 内
m = stats[:, :self.half_channels, :]
logs = stats[:, self.half_channels:, :]
logs = torch.clamp(logs, min=-5.0, max=5.0)  # 安定化
```

**パラメータ影響:** post-conv の出力チャネルが `half_ch → half_ch * 2` に変化。+~0.01%。

**推定効果:** flow表現力 +5-10%

### 7.3 Flow Dilation Rate 1→2

**概要:** Flow内WNの dilation_rate を 1→2 に変更。受容野が ~17 → ~61 フレームに拡大。パラメータ変更なし。

**実装:**

```python
# models.py: VitsModel.__init__() 内
# 変更前:
self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, ...)
# 変更後:
self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 2, 4, ...)
```

**パラメータ影響:** 0

**推定効果:** 長距離スペクトル相関の改善

### 7.4 1x1 Invertible Convolution (Flip置換)

**概要:** Glow由来の学習可能なチャネル置換。固定 Flip (チャネル反転) をQR分解初期化の 1x1 invertible conv に置換。

**実装:**

```python
# modules.py: 新クラス
class InvertibleConv1x1(nn.Module):
    def __init__(self, channels):
        super().__init__()
        W = torch.linalg.qr(torch.randn(channels, channels))[0]
        self.weight = nn.Parameter(W)

    def forward(self, x, x_mask, reverse=False):
        if not reverse:
            x = F.conv1d(x, self.weight.unsqueeze(-1))
            logdet = torch.slogdet(self.weight)[1] * x_mask.sum(dim=[1, 2])
            return x, logdet
        else:
            weight_inv = torch.inverse(self.weight)
            x = F.conv1d(x, weight_inv.unsqueeze(-1))
            return x
```

**パラメータ影響:** +192×192 = ~37K params per flow step (微小)

**推定効果:** flow表現力向上

### 7.5 CTC 補助損失

**概要:** TextEncoder にCTCヘッドを追加し、多言語alignment品質を均一化。特にZHの短いduration (1.21s vs 2-3s) 問題の改善が期待。

**実装:**

```python
# models.py: TextEncoder に追加
self.ctc_proj = nn.Linear(hidden_channels, n_vocab)

# lightning.py: training_step_g に追加
ctc_logits = self.model_g.enc_p.ctc_proj(x.transpose(1, 2))  # [B, T, V]
ctc_log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)  # [T, B, V]
loss_ctc = F.ctc_loss(ctc_log_probs, phoneme_targets, input_lengths, target_lengths)
loss_gen_all += self.hparams.c_ctc * loss_ctc
```

**CLIオプション:** `--c-ctc 0.1`

**パラメータ影響:** +173 × 192 = ~33K params (推論時は除去可能)

**推定効果:** 多言語alignment改善、特にZH/FR/PT

### 7.6 iSTFT 最終段 (Issue #268)

**概要:** HiFi-GAN の最終 ConvTranspose1d (4x upsampling) を iSTFT + PQMF に置換。Issue #268 で詳細に調査済み。

**効果:**
- End-to-end推論速度 ~1.2x 高速化
- Decoder パラメータ -37K
- ONNX互換 (DFT行列ベースのiSTFT)
- PQMF固定係数 (4バンド, 62タップ, Kaiser窓)

**実装:** Issue #268 の技術詳細に従う。

### Phase 2 合計

| 改善 | 推定効果 | パラメータ影響 | 推論速度影響 |
|------|---------|-------------|------------|
| Adversarial DP | MOS +0.05-0.1 | -0.45M (SDP除去) | 高速化 |
| Flow scale有効化 | +5-10% 表現力 | +~0.01% | 同等 |
| Flow dilation 1→2 | 長距離改善 | 0% | 同等 |
| 1x1 Invertible Conv | 表現力向上 | +~150K | 同等 |
| CTC 補助損失 | alignment改善 | +33K (除去可) | なし |
| iSTFT 最終段 | - | -37K | **+20% 高速** |
| **合計** | **MOS +0.2-0.3** | **微減** | **+10-20% 高速** |

---

## 8. Phase 3: 長期的検討

以下は品質をさらに押し上げる可能性があるが、実装リスクが高い変更。Phase 0-2 完了後に検討。

### 8.1 Transformer Coupling Layer (VITS2)

Flow の WN conditioner を Transformer に置換。グローバル受容野を獲得。

- 既存の `attentions.Encoder` を再利用可能
- 全4層置換ではなく、後半2層のみ Transformer にする hybrid 構成が実用的
- 計算量: O(T²) attention だが、mel frame 列 (200-800) なら Flash Attention で高速
- **課題:** 推論速度への影響が未知。事前ベンチマーク必要

### 8.2 Vocos Decoder (全段iSTFT)

ConvNeXt backbone + iSTFT による完全な周波数領域デコーダ。

- HiFi-GAN 比 3-5x 推論高速化
- Transposed Convolution 完全除去
- **課題:** VITS の segment slicing との境界処理、品質検証必要

### 8.3 Conditional Flow Matching (Matcha-TTS style)

ResidualCouplingBlock を CFM に置換。ODE solver で prior → posterior を変換。

- 2-4 Euler step で十分 (normalizing flow の 4 coupling と同等コスト)
- 学習が単純 (L2 regression on velocity field)
- **課題:** multi-step 推論が ONNX 互換性に影響する可能性

### 8.4 RoPE (Rotary Position Embedding)

TextEncoder の相対位置埋め込み (window_size=4) を RoPE に置換。

- 長いphoneme列 (max_phoneme_ids=400) への汎化改善
- パラメータ微減
- **課題:** 既存チェックポイント非互換

### Phase 3 比較

| 改善 | 期待効果 | リスク | 推論速度影響 | 工数 |
|------|---------|-------|------------|------|
| Transformer Coupling (hybrid) | 中 | 中 | -10~20% | 1-2週間 |
| Vocos Decoder | 低-中 (品質) + 高 (速度) | 高 | +200-500% | 2-3週間 |
| Conditional Flow Matching | 高 | 高 | 要検討 | 3-4週間 |
| RoPE | 低-中 | 低 | 同等 | 3-5日 |

---

## 9. 品質到達予測

```
品質 (MOS推定)
5.0 ├──────────────────────────────── 人間の音声
    │
4.5 ├── CosyVoice 2 ────────────── LLM+CFM (2.5B params, GPU必須)
    │── KokoroTTS ───────────────── StyleTTS2 (82M, GPU推奨)
    │
4.0 ├── VITS+ 楽観的上限 ────────← Phase 0-2 最良ケース
    │
3.8 ├── VITS+ 中央推定 ──────────← 現実的な目標 ★
    │     ↑ +0.3-0.5 MOS
    │
3.5 ├── piper-plus 現在 ──────── VITS1 + WavLM (34MB, CPU可)
    │
3.0 ├── 標準 VITS (espeak-ng)
```

### Phase別累積効果推定 (レビュー修正後)

個別効果は論文の単一変更ベンチマークに基づくが、複数変更の組み合わせでは diminishing returns が生じるため、**累積効果に 0.6-0.7x の割引率**を適用。

| Phase | PESQ改善 (累積) | MOS改善 (累積) | ONNXサイズ | 推論速度 |
|-------|----------------|---------------|----------|---------|
| 現在 | 基準 | ~3.5 | 34MB | 基準 |
| Phase 0 完了 (R1なし) | +0.1-0.25 | ~3.6-3.7 | 34MB | 同等 |
| Phase 1 完了 | +0.3-0.6 | ~3.7-3.9 | ~34MB | -10% |
| Phase 2a-c 完了 | +0.4-0.8 | ~3.8-4.0 | ~33MB | +10-20% |
| Phase 3 (選択的) | +0.5-1.0 | ~3.9-4.1 | ~33MB | 要検討 |

**注意:** これらはレビュー修正後の推定値であり、楽観的上限ではなく中央推定を示す。実際の効果は学習データ、ハイパーパラメータ、言語によって変動する。A/B テストによる検証が必須。

### KokoroTTS/CosyVoice との根本的差異

VITS+ で MOS ~3.8-4.0 に到達しても、KokoroTTS/CosyVoice との品質差は完全には埋まらない。その理由:

1. **韻律モデリングの壁**: VITS の Duration Predictor (CNN/flow) vs StyleTTS2 の Style Diffusion / CosyVoice の LLM autoregressive。長距離韻律パターンのモデリングに根本的差がある
2. **テキスト理解の壁**: 6層 Transformer (192dim) vs PLBert (12層 768dim) / Qwen LLM。文脈理解能力に大きな差
3. **データスケールの壁**: ~500K発話 vs CosyVoice ~170K時間

ただし、**34MB ONNX / CPU リアルタイム** という展開条件で MOS ~3.8-4.0 を達成できれば、**同サイズクラス (50MB以下/CPU/オフライン) では最高水準の品質**。競争優位は「絶対品質」ではなく「**MOS-per-megabyte**」で評価すべき。

---

## 10. 推奨実装順序 (レビュー修正後)

> **注:** セクション13にレビュー結果を踏まえた最終版の実装順序あり。以下は初期提案。

---

## 11. 参考文献

### 論文

| 論文 | 年 | 関連技術 |
|------|-----|---------|
| VITS2 (Kong et al.) | 2023 | Adversarial DP, Transformer flow |
| StyleTTS 2 (Li et al.) | 2023 | Style diffusion, WavLM-D, R1 |
| CosyVoice (Alibaba) | 2024 | LLM + CFM, supervised tokens |
| BigVGAN (Lee et al.) | 2023 | Snake, AMP, MRD |
| BigVGAN v2 (Lee et al.) | 2024 | R1, multi-scale loss |
| Matcha-TTS (Mehta et al.) | 2024 | OT-CFM, Gaussian upsampling |
| Vocos (Siuzdak) | 2023 | ConvNeXt + iSTFT decoder |
| ISTFTNet (Kaneko et al.) | 2022 | Hybrid iSTFT decoder |
| MB-iSTFT-VITS (Kawamura et al.) | 2023 | Multi-band iSTFT for VITS |
| EnCodec (Defossez et al.) | 2022 | MR-STFT discriminator |
| DAC (Kumar et al.) | 2023 | MB-MS-STFT discriminator |
| Glow (Kingma & Dhariwal) | 2018 | 1x1 invertible conv, ActNorm |

### オープンソース実装

| リポジトリ | 内容 |
|-----------|------|
| `p0p4k/vits2_pytorch` | VITS2 PyTorch 実装 |
| `daniilrobnikov/vits2` | VITS2 クリーン実装 |
| `fishaudio/Bert-VITS2` | BERT + VITS2 (中国語/日本語) |
| `hexgrad/kokoro` | KokoroTTS (StyleTTS2, Apache-2.0) |
| `FunAudioLLM/CosyVoice` | CosyVoice (Apache-2.0) |
| `NVIDIA/BigVGAN` | BigVGAN v2 (MIT) |
| `gemelo-ai/vocos` | Vocos decoder (MIT) |
| `shivammehta25/Matcha-TTS` | Matcha-TTS (MIT) |
| `SWivid/F5-TTS` | F5-TTS (flow matching + DiT) |

---

## 12. エージェントチームレビュー結果

5エージェント並列レビュー (2026-03-27) の統合結果。

### 12.1 技術正確性レビュー

**検証済み (正確):**
- 全アーキテクチャ記述 (層数, 次元数等) はコードベースと一致
- LRスケジュール分析: ExponentialLR gamma=0.999875, per-epoch stepping で75epochの総減衰率0.93% → **正確**
- VITS2/KokoroTTS/CosyVoice の技術記述は正確
- パラメータ数推定は妥当

**修正が必要な問題:**

| 重要度 | 問題 | 修正 |
|--------|------|------|
| **高** | `InvertibleConv1x1` が `torch.inverse` / `torch.slogdet` を使用。ONNX opset 15 非互換 | → ONNX エクスポート前に `W_inv` を事前計算し buffer として保存。推論パスでは `F.conv1d(x, self.weight_inv.unsqueeze(-1))` を使用。または **Phase 3 に延期** (品質効果 <0.01 MOS) |
| **高** | CTC損失スニペットが `SynthesizerTrn.forward()` の戻り値にない `x` (text encoder hidden) を参照 | → `SynthesizerTrn.forward()` を修正して text encoder output を追加返却するか、別途 CTC forward を実行 |
| **高** | `LowPassFilter1d` の FIR バッファ形状がチャネル数と不一致の可能性 | → `__init__` でチャネル数を受け取り、`(channels, 1, kernel_size)` 形状で register_buffer |
| **中** | クラス名が `VitsModel` と記載されているが、実際は `SynthesizerTrn` | → 記述修正 |
| **中** | Feature matching 重み変更が WavLM-D の feature maps にも影響。WavLM layers [6,9,12] と MPD conv layers では意味が異なる | → `layer_weighted` を MPD のみに適用し、WavLM-D は均一重みを維持 |
| **中** | R1 スニペットが `model_d(y_real, y_hat.detach())` で real+fake 両方を forward。real のみで十分 | → MPD sub-discriminators を個別にループして real のみ forward |
| **低** | SDP は `self.flows` (4 ConvFlow) + `self.post_flows` (4 ConvFlow) = 計8 ConvFlow。「4 ConvFlow」は不完全 | → 記述修正 |

### 12.2 実現可能性・リスクレビュー

**致命的リスク:**

| リスク | 詳細 | 緩和策 |
|--------|------|--------|
| **Phase 0 のチェックポイント互換性は不完全** | LR scheduler を ExponentialLR → CosineAnnealingLR に変更すると、チェックポイントの scheduler state が不一致で `ckpt_path` resume が失敗する | → `--resume-from-multispeaker-checkpoint` (optimizer state リセット) での再開は可能。通常 resume は LR scheduler state を無視するように fallback を追加 |
| **Cosine annealing `eta_min=1e-6` が攻撃的すぎる** | 75 epoch で LR が 2e-4 → 1e-6 (200x 減衰) は過剰 | → **`eta_min=1e-5` に修正** (20x 減衰)。または `CosineAnnealingWarmRestarts` を検討 |
| **Phase 2 は既存ベースモデルからのファインチューニング不可能** | `mean_only=False` で flow の post conv 出力チャネルが変化。1x1 inv conv で Flip が置換。いずれもチェックポイント非互換 | → Phase 2 ベースモデルを一から再学習し、そこからファインチューニング |
| **Snake alpha + EMA の相互作用** | EMA decay=0.9995 は conv weight 用に調整済み。Snake の `alpha` (周波数パラメータ) は収束ダイナミクスが異なる | → **Snake alpha を EMA から除外**するか、alpha 用の別 decay を設定 |
| **FP16 ONNX + Snake の精度** | `sin(alpha * x)` で `alpha * x` が大きい場合、FP16 の `sin` 精度が低下 | → FP16 エクスポート後に必ず品質テスト。必要なら Snake 計算を FP32 に維持 |

**Phase 2 分割の推奨:**

```
Phase 2 (現在の一括提案)     →     Phase 2a + 2b + 2c (推奨)
├── Adversarial DP                  Phase 2a: Flow 改善
├── Flow scale                        ├── mean_only=False
├── Flow dilation                     ├── dilation_rate 1→2
├── 1x1 Invertible Conv              └── (1x1 conv は Phase 3 に延期)
├── CTC 補助損失
└── iSTFT 最終段              Phase 2b: Duration 改善
                                      ├── Adversarial DP
                                      └── CTC 補助損失

                               Phase 2c: Decoder 改善
                                      └── iSTFT 最終段 (Issue #268)
```

**1x1 Invertible Conv の再評価:** レビューにより、品質効果は <0.01 MOS と推定。ONNX互換性問題とファインチューニング非互換を考慮し、**Phase 3 (長期検討) に降格**を推奨。

### 12.3 実装品質レビュー

**CLI設計の修正:**

| 提案時 | 問題 | 修正後 |
|--------|------|--------|
| `--multi-resolution-mel-loss` (default ON) | デフォルトONなら `--no-X` パターンが適切 | `--no-mr-mel-loss` で無効化 |
| `--mr-stft-discriminator` (default ON) | 同上 | `--no-mr-stft-d` で無効化 |
| `--adversarial-dp` (default ON) | アーキテクチャ変更はデフォルトOFFが安全 | `--adversarial-dp` でオプトイン |
| `--r1-gamma 5.0` | `--c-X` 命名規則と不整合 | 文献での慣例名なので `--r1-gamma` を維持 |

**未記載の必須実装項目:**

1. **`save_hyperparameters()` への新フラグ追加**: `snake_activation`, `adversarial_dp`, `flow_mean_only` 等を `VitsModel.__init__` パラメータに追加し、チェックポイントに保存
2. **CTC phoneme target の抽出**: 現在の `batch.phoneme_ids` は blank-interspersed。CTC には raw phoneme sequence が必要
3. **`max_epochs` の `configure_optimizers()` での参照**: Trainer 未アタッチ時の問題。→ `self.hparams.max_epochs` として保存
4. **MR-mel loss と既存 `c_mel` の関係**: MR版が既存単一解像度版を**置換** (補完ではない) と明確化
5. **DurationDiscriminator の学習ループ統合**: 既存 D optimizer に追加 (3つ目の optimizer は不要)
6. **`remove_weight_norm()` と LowPassFilter1d の共存**: FIR フィルタは weight_norm 対象外とする

**テスト戦略:**

各 Phase で必須のテスト:

| Phase | テストファイル | 内容 |
|-------|-------------|------|
| 0 | `tests/test_losses.py` | free_bits, layer-weighted feature matching |
| 0 | `tests/test_lr_schedule.py` | Cosine warmup 動作確認 |
| 1 | `tests/test_snake_activation.py` | 形状, ONNX export, FP16精度 |
| 1 | `tests/test_mr_stft_discriminator.py` | 出力形式が MPD と互換 |
| 2a | `tests/test_flow_improvements.py` | mean_only=False 可逆性 |
| 2b | `tests/test_adversarial_dp.py` | freeze-dp 互換性 |
| **全Phase** | `tests/test_export_onnx.py` 拡張 | **ONNXエクスポートパリティテスト必須** |

### 12.4 競合分析レビュー

**追加すべき競合脅威:**

| 脅威 | 重要度 | 詳細 |
|------|--------|------|
| **sherpa-onnx + Kokoro ONNX** | **最大** | sherpa-onnx が Kokoro ONNX (~170MB) を統合済み。piper-plus と同じ展開ターゲット (C++/C#/WASM offline) でより高品質 |
| F5-TTS | 中 | Flow matching + DiT, ~330M params, zero-shot 対応。Matcha-TTS より新しいリファレンス |
| MaskGCT | 中 | 完全 non-autoregressive、explicit duration prediction 不要。TTS の方向性を示す |
| Edge TTS (Microsoft) | 中 | クラウドAPI経由で MOS 4.0+ が無料。オフライン不要なユースケースでの競合 |
| WebGPU 成熟 | 低-中 | ブラウザGPU計算の成熟により、170MB Kokoro on WebGPU が 34MB piper-plus on CPU を凌駕する可能性 |

**piper-plus の競争優位の再定義:**

「KokoroTTS/CosyVoice に品質で追いつく」ではなく、**「50MB以下/CPU/オフラインクラスでの最高品質」**にリフレーム。

```
展開シナリオ別の最適解:
┌────────────────────────┬──────────┬────────┬───────────┐
│ シナリオ               │piper-plus│ Kokoro │ CosyVoice │
├────────────────────────┼──────────┼────────┼───────────┤
│ 組込み/IoT (<256MB RAM)│ ★最適    │ ×      │ ×         │
│ オフラインモバイル     │ ★最適    │ △可能  │ ×         │
│ ブラウザWASM (GPU無し) │ ★最適    │ △遅い  │ ×         │
│ プライバシー重視       │ ★最適    │ △可能  │ △クラウド │
│ エッジサーバー (大量)  │ ○良好    │ ○良好  │ △GPU必要  │
│ デモ/ショーケース      │ △十分    │ ○良好  │ ★最高    │
└────────────────────────┴──────────┴────────┴───────────┘
```

### 12.5 学習コスト分析レビュー

**V100 16GB メモリ分析:**

| 構成 | 推定ピークメモリ | batch_size=20 で収まるか |
|------|----------------|----------------------|
| 現在 (--no-wavlm) | ~10-12.5 GB | **OK** (余裕 3.5-6 GB) |
| + Phase 0 (R1なし) | ~10.5-13 GB | **OK** |
| + Phase 0 (R1あり) | ~12-17 GB | **危険** (R1 spike で +2-4GB) |
| + Phase 0+1 全部 | ~13-18.5 GB | **NG** (batch_size=12-14 に要削減) |

**重要な結論:**
- **R1 gradient penalty は V100 16GB では `create_graph=True` のメモリスパイク (+2-4GB) によりOOMリスク大**
- **推奨: Phase 0 は R1 抜きで開始**。R1 は A100 利用可能時に追加
- MR-STFT-D 単独なら +0.5-1GB で収まる可能性あり (batch_size=18程度)

**修正された学習時間推定:**

| 構成 | 推定時間 | vs 現在92h | batch_size |
|------|---------|-----------|-----------|
| Phase 0 (R1なし) | ~95h | 1.03x | 20 |
| Phase 0+1 (Snake+AMP+MR-STFT-D) | ~115-125h | 1.25-1.35x | 18 |
| Phase 0+1+2 全部 | ~150-170h | 1.6-1.8x | 14-16 |

**A100 80GB の推奨:**

Phase 0+1+2 の開発・A/Bテストには **A100 80GB 1台** が最もコスト効率が高い:
- FP16 mixed precision が動作 (V100では不可) → 2-3x speedup
- 全変更を batch_size=32+ で同時適用可能
- DDP overhead なし、NCCL問題なし
- 推定コスト: $400-520 (4x V100の$600-680より安い)

**Phase 0 実装工数の修正: 1-2日 → 3-5日**

内訳:
- コード実装: 1日
- V100 メモリプロファイリング: 0.5-1日
- DDP 4GPU テスト: 0.5日
- チェックポイント継続互換テスト: 0.5日
- WandB ロギング統合 + 短時間サニティチェック: 0.5-1日

### 12.6 MOS推定の修正

レビューにより、累積効果に **0.6-0.7x の割引率** を適用。個別効果は論文のクリーンな単一変更ベンチマークから引用されており、複数変更の組み合わせでは diminishing returns が生じる。

| Phase | 提案時推定 | 修正後推定 (中央値) | 備考 |
|-------|-----------|-------------------|------|
| 現在 | MOS ~3.5 | MOS ~3.5 | - |
| Phase 0 完了 | ~3.7-3.8 | **~3.6-3.7** | R1なしで効果減 |
| Phase 1 完了 | ~3.9-4.0 | **~3.7-3.9** | Snake が最大寄与 |
| Phase 2 完了 | ~4.0-4.2 | **~3.8-4.0** | 4.0は楽観的上限 |

**修正後の品質到達予測:**

```
品質 (MOS推定)
5.0 ├──────────────────────────────── 人間の音声
    │
4.5 ├── CosyVoice 2 ────────────── LLM+CFM (2.5B params, GPU必須)
    │── KokoroTTS ───────────────── StyleTTS2 (82M, GPU推奨)
    │
4.0 ├── VITS+ 楽観的上限 ────────← Phase 0-2 最良ケース
    │
3.8 ├── VITS+ 中央推定 ──────────← 現実的な目標 ★
    │     ↑ +0.3-0.5 MOS
    │
3.5 ├── piper-plus 現在 ──────── VITS1 + WavLM (34MB, CPU可)
    │
3.0 ├── 標準 VITS (espeak-ng)
```

**MOS ~3.8-4.0 は 34MB CPUモデルとしては最高水準。** 同サイズクラスで明確な優位性を持つ。

---

## 13. 修正された推奨実装順序

レビュー結果を踏まえた最終推奨:

```
Phase 0 (3-5日)                     既存ckptから部分互換で継続学習可能
├── LR スケジュール修正 (Cosine, eta_min=1e-5)
├── Multi-resolution mel loss (既存 c_mel を置換)
├── KL annealing (warmup 10 epoch) + free bits (0.1)
├── Feature matching 層重み付け (MPDのみ、WavLM-Dは均一維持)
└── ★ R1 gradient penalty は V100 では保留、A100 利用時に追加
        │
        ▼  効果検証 (6言語 x 6文 = 36発話 A/B テスト)
        │
Phase 1 (1-2週間)                   再学習必要
├── Snake activation
├── Anti-aliased upsampling (AMP, Kaiser FIR)
├── MR-STFT Discriminator (batch_size=18に調整の可能性)
├── ★ Snake alpha を EMA から除外
└── ★ FP16 ONNX + Snake の精度テスト必須
        │
        ▼  効果検証 + ONNX パリティテスト
        │
Phase 2a (3-5日)                    再学習必要
├── Flow scale 有効化 (mean_only=False + clamping)
└── Flow dilation_rate 1→2
        │
Phase 2b (1週間)                    2aと同時 or 直後
├── Adversarial Duration Predictor (デフォルトOFF、オプトイン)
└── CTC 補助損失 (SynthesizerTrn.forward() 修正必要)
        │
Phase 2c (Issue #268)               2a/2b安定後
└── iSTFT 最終段 (DFT matrix方式)
        │
        ▼  効果検証 + 全推論パイプライン (C#/Rust/WASM) テスト
        │
Phase 3 (長期検討)
├── 1x1 Invertible Conv (LU分解方式、ONNX互換確認後)
├── Transformer Coupling Layer (hybrid, 後半2層のみ)
├── Vocos Decoder (全段iSTFT)
├── Conditional Flow Matching (Matcha-TTS / F5-TTS style)
└── RoPE (Text Encoder)
```

### ハードウェア推奨

| 用途 | 推奨 | 理由 |
|------|------|------|
| Phase 0 開発・テスト | 既存 4x V100 16GB | R1なしなら batch_size=20 で収まる |
| Phase 1+2 開発 | **A100 80GB 1台** | FP16可、全変更同時適用、コスト効率最良 |
| 本番学習 (最速) | 4x A100 40GB | 最大スループット |
| A/B テスト | 3回以上のフル学習を予算化 | Phase毎の検証に最低3 run 必要 |

### `--vits-plus-preset` フラグの提案

ハイパーパラメータ爆発を防ぐため、検証済みデフォルトをまとめたプリセット:

```bash
python -m piper_train --vits-plus-preset \
  --dataset-dir /data/piper/dataset-multilingual-6lang-filtered \
  --default_root_dir /data/piper/output-vits-plus
```

`--vits-plus-preset` 展開時の等価フラグ:
```
--lr-schedule cosine --lr-warmup-epochs 5 --lr-min 1e-5
--no-mr-mel-loss  # (有効化、無効化オプション)
--kl-warmup-epochs 10 --free-bits 0.1
--snake-activation
--no-mr-stft-d  # (有効化、無効化オプション)
--n-heads 4
--resblock 1
--filter-channels 512
--grad-clip-norm 1.0
```

---

## 14. 5層品質改善モデル (第2ラウンド調査結果)

> **詳細:** [`vits-plus-quality-maximization.md`](vits-plus-quality-maximization.md) 参照

第2ラウンドの15エージェント調査により、アーキテクチャ変更 (Layer 1) に加えて4つの独立した品質改善レバーが特定された。各層は累積的に効果を発揮する。

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 5: 推論時最適化                          [コスト: 0]  │
│   noise_scale言語別最適化, denoiser有効化, ONNX最適化       │
├─────────────────────────────────────────────────────────────┤
│ Layer 4: Phonemizer / G2P 改善             [パラメータ: 0]  │
│   テキスト正規化, 異読語解消, リエゾン, prosody全言語有効化 │
├─────────────────────────────────────────────────────────────┤
│ Layer 3: データ改善                          [前処理のみ]   │
│   品質フィルタ, 合成データ, WenetSpeech4TTS, MLS-Sidon     │
├─────────────────────────────────────────────────────────────┤
│ Layer 2: 学習レシピ + Post-training      [推論モデル不変]   │
│   grad clip, EMA全体化, マルチタスク, UTMOS報酬, 蒸留      │
├─────────────────────────────────────────────────────────────┤
│ Layer 1: アーキテクチャ変更                  [再学習必要]   │
│   Snake, AMP, Flow再配分, ResBlock1, Adversarial DP, iSTFT │
└─────────────────────────────────────────────────────────────┘
```

### 第2ラウンドで発見された最重要改善 (TOP 10)

| # | 発見 | 推定効果 | コスト |
|---|------|---------|-------|
| 1 | **EN/ES/FR/PT の prosody が完全にゼロ化** | +0.1-0.2 MOS | 0 params |
| 2 | **データフィルタリングで 15-30% 除去 → 品質向上** (UTMOS/SNR) | +0.1-0.3 MOS | 前処理のみ |
| 3 | **WenetSpeech4TTS で ZH データ 12x 拡大** (CC-BY-4.0) | +0.1-0.2 MOS | データ追加 |
| 4 | **UTMOS 微分可能報酬 (CosyVoice方式)** — WavLM-D と同じパターン | +0.1-0.3 MOS | 学習時のみ |
| 5 | **EMA が dec のみ → model_g 全体に拡張** | +0.05-0.1 MOS | **1行変更** |
| 6 | **Gradient norm clipping 未実装** | 安定性大幅向上 | **4行** |
| 7 | **n_heads 2→4 が完全無料** (パラメータ数同一) | attention改善 | 設定変更 |
| 8 | **Flow が推論の 50% 占有 → WN 4→3層で -2M 節約** | サイズ削減 | 設定変更 |
| 9 | **テキスト正規化が全言語で未実装** | 可読性向上 | ルールベース |
| 10 | **F0 Predictor がコードに存在するが未接続** | ピッチ精度向上 | 接続のみ |

### 修正された品質到達予測 (全5層累積)

| シナリオ | MOS改善 | 到達MOS |
|---------|---------|---------|
| 保守的 | +0.35 | ~3.85 |
| **中央推定** | **+0.55** | **~4.05** |
| 楽観的 | +0.85 | ~4.35 |

**中央推定 MOS ~4.05 は、30M params / 34MB / CPU リアルタイムモデルとして前例のない水準。**
同パラメータ規模の VITS 原論文 (LJSpeech) の報告 MOS 4.36 に対し、6言語マルチリンガルモデルでの ~4.05 は非常に強力。

### 統合ロードマップ (Sprint形式)

```
Sprint 0 (1日) ─── Layer 5: 推論最適化
  noise_scale 最適化, denoiser 有効化, ONNX 最適化

Sprint 1 (1週間) ── Layer 4+3: Phonemizer + データ前処理
  テキスト正規化, 異読語, リエゾン, prosody 全言語有効化
  UTMOS/SNR フィルタリング, データセット拡大 (3,300h+)

Sprint 2 (1週間) ── Layer 3: 合成データ
  CosyVoice 2 による合成音声生成 (PT/FR重点, 2,000-3,000h)

Sprint 3 (3-5日) ── Layer 2: 学習レシピ基本
  Cosine LR, grad clip, EMA全体化, n_heads=4
  MR-mel loss, KL annealing, ResBlock1, filter=512
  → 拡大データで学習開始

Sprint 4 (1-2週間) ── Layer 2+1: マルチタスク + アーキテクチャ
  CTC/LID/Speaker補助損失, Prosody→Flow
  Snake, AMP, Flow WN 3層, encoder 8層
  MR-STFT-D, Adversarial DP
  → フルアーキテクチャで再学習

Sprint 5 (1週間) ── Layer 2: Post-training
  UTMOS 報酬学習, Duration蒸留, Best-of-N fine-tuning

Sprint 6 (3-5日) ── 検証 + リリース
  6言語 A/B テスト, ONNX パリティ, C#/Rust/WASM テスト
```
