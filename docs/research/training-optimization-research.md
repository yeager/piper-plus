# 学習パイプライン高速化 調査レポート

**日付**: 2026-03-10
**ブランチ**: `feat/zero-shot-tts`
**環境**: RTX 6000 Ada 48GB × 1, PyTorch 2.x, bf16-mixed
**現状ベースライン**: ~8分/epoch (batch_size=360, max_spec_length=700, D:G=1:2)

---

## 調査ソース

1. **10エージェント並列調査** — PyTorch/TTS最適化の網羅的調査
2. **feat/multilingual-phonemizer ブランチ** — 多言語モデル向け実装済み最適化
3. **feat/bilingual-phonemizer ブランチ** — 二言語モデル向け実装済み最適化

---

## 実装済みの最適化 (現ブランチ)

| 最適化 | 効果 | 実装箇所 |
|--------|------|---------|
| bf16-mixed precision | 学習速度2-3x, VRAM ~50%削減 | __main__.py |
| torch.compile (mode=default) | Generator decoder + MPD コンパイル | lightning.py `on_train_start()` |
| monotonic_align GPU実装 | 6x高速化 (117ms→19ms/batch) | monotonic_align/ |
| slice_segments vectorization | 180x高速化 (torch.gather) | commons.py |
| D:G=1:2 (d_update_interval=2) | D計算50%削減 | lightning.py |
| D real-side caching | G→D間の重複forward削除 | lightning.py |
| max_spec_length=700 | VRAM 41GB→19GB, パディング62%→34% | dataset.py |
| SpeakerBalancedBatchSampler | Duration Predictor安定化 | dataset.py |
| TF32 matmul precision | matmul高速化 | __main__.py |
| EMA decay=0.9995 | 推論品質向上 | lightning.py |
| Fused AdamW | optimizer step 10-20%高速化 | lightning.py `configure_optimizers()` |
| parametrizations.weight_norm | torch.compile graph break解消, 15-30%高速化 | models.py, modules.py |
| DataLoader prefetch_factor=2 | CPU→GPU転送オーバーラップ, 5-15%高速化 | lightning.py `train/val_dataloader()` |
| Validation頻度削減 (5 epochごと) | 学習ループ 3-5%高速化 | __main__.py (`--val-every-n-epochs`, `--limit-val-batches`) |

---

## 未実装の最適化候補

### 優先度2: GPU Clock最大化 (推定効果: 10-24%)

**概要**: RTX 6000 Adaのデフォルトクロック (2505 MHz) を最大 (3105 MHz) に固定。

**実装**:
```bash
sudo nvidia-smi -lgc 2505,3105
# 学習終了後に戻す場合:
# sudo nvidia-smi -rgc
```

**リスク**: 低。消費電力・発熱が増加するが、サーバー環境では問題なし。
**出典**: 10エージェント調査。NVIDIA公式ドキュメント準拠。

---

### 優先度6: inductor config + dynamic=False (推定効果: 5-15%)

**概要**: torch.compileのバックエンド (inductor) に追加チューニングを適用。

**実装**:
```python
# __main__.py or lightning.py
import torch._inductor.config as inductor_config
inductor_config.conv_1x1_as_mm = True
inductor_config.coordinate_descent_tuning = True

# torch.compile呼び出し
torch.compile(model, mode="default", dynamic=False)
```

**注意点**:
- `dynamic=False` はバッチサイズが固定の場合のみ有効（drop_last=True推奨）
- CUDA Graphsの恩恵を受けやすくなる

**リスク**: 低〜中。dynamic=Falseは可変長バッチで問題になる場合あり。
**出典**: 10エージェント調査。

---

### 優先度7: CosineAnnealingLR (効果: 収束品質改善)

**概要**: 現行の ExponentialLR (gamma=0.999875) は200 epochで学習率が97.5%にしか減衰せず、実質的に定数LR。CosineAnnealingLRでwarmup→cosine decayに変更。

**実装**:
```python
# lightning.py — configure_optimizers()
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler_g = CosineAnnealingLR(optim_g, T_max=max_epochs, eta_min=1e-6)
scheduler_d = CosineAnnealingLR(optim_d, T_max=max_epochs, eta_min=1e-6)
```

**リスク**: 低。VITSの多くの実装で使用実績あり。
**出典**: 10エージェント調査。

---

### 優先度8: Generator ResBlock改善 (推定効果: 微小)

**概要**: HiFi-GAN decoderのResBlock出力累積でCPU tensorを避ける。

**実装**:
```python
# models.py — Generator.forward()
# Before
xs = torch.zeros(x.shape, dtype=x.dtype, device=x.device)  # or torch.zeros(1)
for resblock in self.resblocks:
    xs += resblock(x)
# After
xs = None
for resblock in self.resblocks:
    if xs is None:
        xs = resblock(x)
    else:
        xs = xs + resblock(x)  # non-in-place for autograd
```

**リスク**: 低。
**出典**: feat/multilingual-phonemizer, feat/bilingual-phonemizer で実装済み。

---

### 優先度9: Length-aware bucketing (推定効果: 30-40% パディング削減)

**概要**: 類似長の発話をバケットにまとめ、パディング無駄を最小化。現在のGPU利用率は約54%（パディングで46%がムダ）。

**実装**: SpeakerBalancedBatchSamplerにバケット機能を追加。話者バランスを維持しつつ、各バケット内で類似長のサンプルをグループ化。

**リスク**: 高。SpeakerBalancedBatchSamplerとの統合が複雑。
**出典**: 10エージェント調査。

---

### 優先度10: NPYデータ形式変換 (推定効果: 1.8x I/O高速化)

**概要**: スペクトログラムキャッシュを torch.load (.pt) → numpy.load (.npy) に変更。

**実装**: dataset.pyのキャッシュ保存/読み込みを numpy形式に変更。

**リスク**: 中。既存キャッシュの再生成が必要。
**出典**: 10エージェント調査。

---

### ~~優先度11: MPD periods削減~~ (非推奨)

**概要**: MultiPeriodDiscriminator の periods を [2,3,5,7,11] → [2,5,11] に削減。

**リスク**: **高（非推奨）**。以下の理由により採用しない:
- Period 3/7は母音品質・ピッチの自然さに寄与する倍音構造を捕捉しており、削除は音質劣化に直結
- D:G=1:2 で既にD計算を50%削減済みであり、さらにperiodsを減らすとDの学習能力不足でGへのフィードバック品質が劣化
- VITS2論文でのperiod削減はMRD追加等の代替Discriminatorとの併用が前提であり、単純削除とは条件が異なる
- 音質劣化は200 epoch学習後にしか判明せず、ロールバックコストが極めて高い

**出典**: 10エージェント調査。VITS2論文での報告あり（ただし条件が異なる）。

---

### 優先度12: float16 スペクトログラムキャッシュ (推定効果: ディスクI/O 50%削減)

**概要**: キャッシュをfloat16で保存し、読み込み時にfloat32に変換。

**実装**:
```python
# dataset.py
if spectrogram.dtype == torch.float16:
    spectrogram = spectrogram.float()
```

**リスク**: 低。数値精度の損失は無視可能。
**出典**: feat/multilingual-phonemizer で実装済み。

---

## 推定効果まとめ

### 実装容易 × 効果大（推奨セット: 優先度1-6）

| 優先度 | 最適化 | 推定効果 | 状態 |
|--------|--------|---------|------|
| 1 | Fused AdamW | 10-20% | ✅ 実装済み |
| 2 | GPU Clock最大化 | 10-24% | ⏳ 学習時に実行 |
| 3 | weight_norm移行 | 15-30% | ✅ 実装済み |
| 4 | DataLoader prefetch | 5-15% | ✅ 実装済み |
| 5 | Validation頻度削減 | 3-5% | ✅ 実装済み |
| 6 | inductor + dynamic=False | 5-15% | 未実装 |

**全て適用した場合の推定**:
- 現在: ~8分/epoch
- 適用後: **~3-4分/epoch** (2-2.5x高速化)
- 学習全体: **1.1日 → 0.5-0.6日** (200 epochs)

### 追加で検討可能（高効果だが実装コスト高）

| 優先度 | 最適化 | 推定効果 | 実装時間 |
|--------|--------|---------|---------|
| 7 | CosineAnnealingLR | 収束改善 | 10分 |
| 8 | ResBlock改善 | 微小 | 5分 |
| 9 | Length-aware bucketing | 30-40%削減 | 2-3時間 |
| 10 | NPYデータ変換 | 1.8x I/O | 1時間 |
| 11 | ~~MPD periods削減~~ | ~~33% D削減~~ | 非推奨 |
| 12 | float16キャッシュ | I/O 50%削減 | 15分 |

---

## 多言語ブランチから移植すべき変更 (高速化以外)

| 変更 | 説明 | 移植推奨 |
|------|------|---------|
| Cache validation (`--validate-cache`) | 破損.ptファイルの事前検出 | ○ 安全性向上 |
| EarlyStopping削除 | 手動管理に移行 | △ 好み次第 |
| AsyncCheckpointIO削除 | 同期チェックポイント | △ 安定性向上 |
| Memory cleanup 500→ (現1000) | より頻繁なGCで安定化 | ○ |
| torch.cuda.synchronize() in cleanup | GPU-CPU同期の改善 | ○ |
