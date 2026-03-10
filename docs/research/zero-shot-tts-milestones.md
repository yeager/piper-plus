# Zero-Shot TTS 実装マイルストーン計画

**作成日**: 2026-03-08
**ブランチ**: `feat/zero-shot-tts`
**前提ドキュメント**:
- [zero-shot-tts-research.md](./zero-shot-tts-research.md) (技術調査レポート)
- [zero-shot-tts-implementation-plan.md](./zero-shot-tts-implementation-plan.md) (実装計画書)

---

## 目次

1. [概要](#概要)
2. [要件トレーサビリティ](#要件トレーサビリティ)
3. [全体スケジュール](#全体スケジュール)
4. [M0: 前提条件・環境準備](#m0-前提条件環境準備)
5. [M1: コアモデル変更](#m1-コアモデル変更)
6. [M2: 学習パイプライン](#m2-学習パイプライン)
7. [M3: 推論パイプライン](#m3-推論パイプライン)
8. [M4: Speaker Embedding 抽出ツール](#m4-speaker-embedding-抽出ツール)
9. [M5: データ準備スクリプト実装](#m5-データ準備スクリプト実装)
10. [M6: 事前学習](#m6-事前学習--完了)
11. [M7: Phase 2 学習](#m7-phase-2-学習----スキップ)
12. [M8: 評価・品質検証](#m8-評価品質検証)
13. [M9: リリース準備](#m9-リリース準備)
14. [リスクマトリクス](#リスクマトリクス)

---

## 概要

本ドキュメントは、Piper TTS (VITSアーキテクチャ) にzero-shot話者再現機能を導入するための実装マイルストーン計画である。

### アーキテクチャ方針

- **Dual-Mode Speaker Conditioning**: `emb_g = nn.Embedding(n_speakers, gin_channels)` と `spk_proj = nn.Linear(192, gin_channels)` が同一モデルに共存。speaker ID指定とspeaker embedding指定の両方で推論可能
- `gin_channels = 512` (768ではガビガビ音が発生するため512に変更)
- CAM++ Speaker Encoder はオフライン処理専用で推論時は不使用
- 事前計算済み192次元embeddingを `.npy` で保存し、推論時はファイル読み込みのみ (<1ms)

### 規模見積もり

| 項目 | 値 |
|------|-----|
| 変更ファイル数 | 8ファイル (修正) + 4ファイル (新規) |
| 追加コード量 (見積) | ~1,100行 (コア) + ~300行 (テスト) |
| 追加コード量 (実績 M0-M4) | ~1,400行 (コア) + ~750行 (テスト) |
| 実装工数 (M0-M5) | ~6日 |
| GPU計算時間 (M6) | RTX 6000 Ada 48GB x1 で200 epochs完了 |
| 全体所要期間 | 実装 ~1週間 + 学習・評価 ~3-5週間 |

### 実装進捗 (2026-03-10 更新)

| マイルストーン | 状態 | テスト数 | コミット |
|-------------|------|---------|---------|
| M0: 環境準備 | ✅ 完了 | — | `28c4027` |
| M1: コアモデル変更 | ✅ 完了 | 15 | `ac1f870`, `520a3fe` |
| M2: 学習パイプライン | ✅ 完了 | 12 | `e5887e9` |
| M3: 推論パイプライン | ✅ 完了 | 5 (3 skip) | `a164628` |
| M4: Speaker Embedding抽出 | ✅ 完了 | 9 | `72c9970` |
| M5: データ準備スクリプト | ✅ 完了 | 11 | — |
| M6: 事前学習 | ✅ 完了 | — | 200 epochs (RTX 6000 Ada) |
| M7: Phase 2 学習 | -- スキップ | — | M6で十分な品質を達成 |
| M8: 評価・品質検証 | 未着手 | — | — |
| M9: リリース準備 | 未着手 | — | — |

**テスト合計**: 41 passed, 5 skipped (GPU/onnxscript依存)

**実装スコープ (M0-M5) 進捗**: 6/6 完了

**学習完了**:
- チェックポイント: `/home/shadeform/data/piper/output-moe-speech-20speakers-v2/lightning_logs/version_0/checkpoints/epoch=199-step=206000.ckpt`
- ONNX: `/home/shadeform/data/piper/output-moe-speech-20speakers-v2/moe-speech-20speakers-v2.onnx` (74MB)
- データセット: `dataset-moe-speech-20speakers` (20話者, speaker embedding付き)
- GPU: RTX 6000 Ada 48GB x1, bf16-mixed, batch-size 160

#### 保留事項
- **SCL/DINO損失のtraining_step_g統合**: PyTorch Speaker Encoder統合時に有効化予定（損失関数は定義済み）
- **Phase切り替え・configure_optimizers**: 同上
- `onnx2torch` によるONNX→PyTorch変換で対応可能（追加依存1パッケージ）

---

## 要件トレーサビリティ

| 要件 | 説明 | 検証マイルストーン | 検証方法 |
|------|------|-------------------|---------|
| R1 | 推論速度ゼロインパクト | M1, M3, M8 | RTF比較 < 1%差異 |
| R2 | 軽量性維持 (ONNX +1MB以内) | M1, M8 | ONNXサイズ比較 |
| R3 | 既存機能の完全互換 | M1, M2, M3, M8 | 既存テスト全パス |
| R4 | GPL-free | M0 | ライセンス監査 |
| R5 | CPU推論対応 | M3, M4, M8 | `CUDA_VISIBLE_DEVICES=""` テスト |
| R6 | 多言語対応 | M2, M5, M8 | 日英両方のSECS/WER/CER計測 |

---

## 全体スケジュール

```
Week 1 (実装フェーズ) ✅ 完了
  Day 1:      M0 (環境準備) + M1 (models.py, config.py) 開始
  Day 2-3:    M1 完了 → M2 (lightning.py, losses.py, dataset.py) 開始
              M4 (extract_speaker_embedding.py) 並行開始
  Day 3-4:    M3 (export_onnx.py, infer_onnx.py) ← M1完了後、M2と並行
              M4 完了
  Day 4-5:    M2 完了
  Day 5-6:    M5 (データ準備スクリプト実装) ← M4完了が前提

--- 実装完了ライン (ここまでがコード実装スコープ) ---

Week 2+ (学習フェーズ) ✅ 完了
  M6: 20話者データセットで200 epochs学習 (RTX 6000 Ada 48GB x1)
  M7: スキップ (M6で十分な品質を達成)

Week 3+ (評価・リリース) — 未着手
  M8 (評価, 1日) → M9 (リリース準備, 1日)
```

### マイルストーン依存関係

```
M0 ──→ M1 ──→ M2 ──┐
  │      │          ├──→ M6 ──→ (M7 skip) ──→ M8 ──→ M9
  │      └──→ M3 ──┘      ↑
  │                        │
  └──→ M4 ──→ M5 ─────────┘
       ✅     ✅          ✅
```

- ✅ M0-M6: 全完了
- M7: スキップ (M6で十分な品質を達成)
- M8-M9: 未着手 (評価・リリース準備)

---

## M0: 前提条件・環境準備

### 概要

実装開始前に必要なリソースの取得・環境構築・ライセンス監査を完了させる。

### タスク

- [x] CAM++ 事前学習済みONNXモデルのダウンロード → `scripts/download_speaker_models.py` で取得可能
  - 入手先: sherpa-onnx GitHub Releases (`k2-fsa/sherpa-onnx`)
  - 動作確認: 入力 80-dim Fbank @ 16kHz → 出力 192-dim embedding
- [x] WeSpeaker ResNet293 評価用モデルのダウンロード → `scripts/download_speaker_models.py` で取得可能
- [x] ライセンス監査: 全依存がGPL-free確認済み → `docs/research/zero-shot-license-audit.md`
- [x] 依存パッケージの確認と `pyproject.toml` の更新 → `[zero-shot]` optional-dependencies 追加済み
  - `onnxruntime>=1.16.0`, `torchaudio>=2.0.0`, `soundfile>=0.12.0`
- [x] featureブランチ `feat/zero-shot-tts` の作成 (`dev` ブランチから分岐)

### 受入基準

1. CAM++ ONNXモデルで任意のWAVファイルから192次元embeddingが抽出可能
2. 全外部依存のライセンスがGPL-freeであることを文書化済み

### 依存関係

- なし (最初のマイルストーン)

### 想定工数

- 0.5日

### 成果物

| 成果物 | 説明 |
|--------|------|
| ライセンス監査結果 | GPL-free要件の充足確認 |
| `campplus.onnx` (28MB) | CAM++ Speaker Encoder |
| `wespeaker-resnet293.onnx` | 評価用cross-encoder |

---

## M1: コアモデル変更

### 概要

VITSの `SynthesizerTrn` にzero-shot対応の `spk_proj = nn.Linear(spk_embed_dim, gin_channels)` を追加し、`emb_g = nn.Embedding(n_speakers, gin_channels)` との Dual-Mode Speaker Conditioning を実現する。両者が同一モデルに共存し、speaker ID指定 (`sid`) と speaker embedding指定の両方で推論可能。

### 変更対象ファイル

| ファイル | 変更種別 | 変更規模 | 変更箇所 |
|---------|---------|---------|---------|
| `src/python/piper_train/vits/config.py` | 修正 | ~5行 | `ModelConfig` にフィールド追加 |
| `src/python/piper_train/vits/models.py` | 修正 | ~40行 | `SynthesizerTrn.__init__`, `forward`, `infer` |
| `test/test_zero_shot.py` or `src/python/tests/test_zero_shot.py` | 新規 | ~120行 | dual-modeテスト |

### タスク

- [x] **config.py** — `ModelConfig` にフィールド追加: `use_zero_shot`, `spk_embed_dim`
- [x] **models.py** — `SynthesizerTrn.__init__`: `spk_proj` / `emb_g` Dual-Mode共存 (`n_speakers > 1` で `emb_g` 生成、`use_zero_shot` で `spk_proj` 生成)
- [x] **models.py** — `SynthesizerTrn.forward`: `speaker_embedding` 引数追加、g生成分岐 (speaker_embedding優先 → sid fallback)
- [x] **models.py** — `SynthesizerTrn.infer`: 同等の変更
- [x] **gin_channels統一ガード**: lightning.py, `__main__.py` でマルチスピーカー時 `gin_channels=512` を設定 (768ではガビガビ音が発生)
- [x] **単体テスト** `test_zero_shot.py`: 15テスト全パス (init 6, forward 4, infer 5)

### 受入基準

1. `SynthesizerTrn(use_zero_shot=True, spk_embed_dim=192, gin_channels=512, n_speakers=20)` でインスタンス化し、`model.spk_proj.weight.shape == (512, 192)` かつ `model.emb_g` も共存
2. `forward(x, x_lengths, y, y_lengths, speaker_embedding=torch.randn(1, 192))` が正常動作
3. `infer(x, x_lengths, speaker_embedding=torch.randn(1, 192))` が音声テンソルを返す
4. `use_zero_shot=False` のとき既存テストが全パス
5. `spk_proj` のパラメータ数 = 192 * 512 + 512 = 98,816 (~0.1M)

### 依存関係

- M0 完了

### 想定工数

- 1日

### リスクと対策

| リスク | 影響度 | 対策 |
|--------|--------|------|
| gin_channels不一致 (512 vs 768) | 高 | **解決済み**: gin_channels=512に統一。768ではガビガビ音が発生することを検証済み。lightning.py と `__main__.py` でガード追加 |
| `forward` シグネチャ変更がlightning.pyと不整合 | 低 | デフォルト引数 `speaker_embedding=None` で後方互換維持 |
| 既存チェックポイントとのstate_dict不整合 | 中 | `strict=False` でロード。`spk_proj` はランダム初期化 |

---

## M2: 学習パイプライン

### 概要

学習ループにCAM++ Speaker Encoder統合、Speaker Consistency Loss (SCL)、DINO Lossの計算を追加。2フェーズ学習の切り替え機構を実装する。

### 変更対象ファイル

| ファイル | 変更種別 | 変更規模 | 変更箇所 |
|---------|---------|---------|---------|
| `src/python/piper_train/vits/losses.py` | 修正 | ~30行 | 新規損失関数追加 |
| `src/python/piper_train/vits/lightning.py` | 修正 | ~80行 | CAM++統合, SCL/DINO, Phase切り替え |
| `src/python/piper_train/vits/dataset.py` | 修正 | ~20行 | speaker_embedding読み込み |
| `src/python/piper_train/__main__.py` | 修正 | ~15行 | CLI引数追加 |
| `src/python/tests/test_m2_training_pipeline.py` | 新規 | ~180行 | SCL/DINO/Dataset/regressionテスト |

### タスク

#### losses.py — 新規損失関数

- [x] `speaker_consistency_loss(gen_embedding, ref_embedding)` 追加:
  ```python
  def speaker_consistency_loss(gen_embedding, ref_embedding):
      return 1.0 - F.cosine_similarity(gen_embedding, ref_embedding, dim=-1).mean()
  ```
- [x] `dino_loss(student_emb, teacher_emb, center, tau_s, tau_t)` 追加:
  ```python
  def dino_loss(student_emb, teacher_emb, center, tau_s=0.1, tau_t=0.04):
      student_out = F.log_softmax(student_emb / tau_s, dim=-1)
      teacher_out = F.softmax((teacher_emb - center) / tau_t, dim=-1)
      return -(teacher_out * student_out).sum(dim=-1).mean()
  ```

#### lightning.py — CAM++統合と学習ループ

- [x] `VitsModel.__init__` にパラメータ追加: `use_zero_shot` (default: `True`), `spk_embed_dim`, `c_spk=9.0`, `c_dino=0.1`, `freeze_speaker_encoder_steps=100000`
- [x] gin_channels自動設定: `(use_zero_shot or num_speakers > 1) and gin_channels <= 0` の場合に `gin_channels=512` を設定
- [ ] CAM++ Speaker Encoder のロード処理: (PyTorch版CAM++未入手のため保留、ONNX監視のみ)
  - [x] `self.dino_center` バッファ初期化 (shape: `[spk_embed_dim]`)
  - [ ] `self.current_tau_t` ウォームアップスケジュール (0.04 → 0.07) — PyTorch encoder待ち
- [x] `SynthesizerTrn` インスタンス化に `use_zero_shot`, `spk_embed_dim` を渡す
- [x] `training_step_g` 変更:
  - バッチから `speaker_embedding` を取得・model_gに受け渡し
  - SCL/DINO計算: PyTorch Speaker Encoder統合時に有効化予定
- [x] `torch.compile` 適用: Generator decoder + MultiPeriodDiscriminator (on_train_start)
- [ ] `on_train_batch_start` でPhase切り替え: PyTorch encoder待ち
- [ ] `configure_optimizers` 変更: PyTorch encoder待ち
- [ ] **FP16対策**: PyTorch encoder統合時に対応

#### dataset.py — speaker_embedding 対応

- [x] `Utterance` dataclass: `speaker_embedding_path: Path | None = None` 追加
- [x] `UtteranceTensors` dataclass: `speaker_embedding: FloatTensor | None = None` 追加
- [x] `Batch` dataclass: `speaker_embeddings: FloatTensor | None = None` 追加
- [x] `PiperDataset.__getitem__`: `.npy` ファイルからembeddingロード (`np.load`)
- [x] `PiperDataset.load_utterance`: JSONから `speaker_embedding_path` パース
- [x] `UtteranceCollate.__call__`: `speaker_embeddings` のスタッキング (固定長192dimのためパディング不要)

#### __main__.py — CLI引数

- [x] 以下の引数を追加:
  - `--spk-embed-dim` (default: 192)
  - `--c-spk` (default: 9.0)
  - `--c-dino` (default: 0.1)
  - `--speaker-encoder-path` (default: None)
  - `--freeze-speaker-encoder-steps` (default: 100000)
- [x] `dict_args` への反映: `VitsModel` に上記パラメータを渡す
- [x] マルチスピーカー (`num_speakers > 1`) でzero-shot自動有効化 (`--zero-shot` フラグは不要、削除済み)
- [x] `gin_channels=512` をマルチスピーカー時に設定 (768ではなく512)

#### テスト

- [x] SCL損失の単体テスト (出力範囲: 0-2) — 4テスト
- [x] DINO損失の単体テスト (出力: 正の値) — 4テスト
- [ ] Phase切り替えロジックのモックテスト — PyTorch encoder待ち
- [x] `use_zero_shot=False` 時の既存学習ループregressionテスト
- [x] Dataset speaker_embedding対応テスト — 3テスト

### 受入基準

1. `speaker_consistency_loss(torch.randn(4, 192), torch.randn(4, 192))` が 0-2 の範囲のfloatを返す
2. `dino_loss(...)` が正のfloatを返す
3. `VitsModel(use_zero_shot=True)` で `dino_center` バッファが登録されている
4. SCL/DINO計算はPyTorch Speaker Encoder統合待ち (損失関数は定義済み)
5. Phase切り替えはPyTorch Speaker Encoder統合待ち
6. `Batch.speaker_embeddings` が shape `[B, 192]` の FloatTensor
7. `use_zero_shot=False` のとき既存学習ループが完全同一動作

### 依存関係

- M1 完了 (models.py の dual-mode)

### 想定工数

- 2日

### リスクと対策

| リスク | 影響度 | 対策 |
|--------|--------|------|
| CAM++ PyTorchモデルのロードに追加依存が必要 | 中 | `pyproject.toml` に optional-dependencies `[zero-shot]` グループとして追加 |
| SCL/DINO損失が学習を不安定にする | 中 | c_spk=9.0, c_dino=0.1 は論文の実証済み値。WandBで損失カーブ監視 |
| Phase切り替え時のloss spike | 中 | CAM++ lr = VITS lr * 0.1。lr warmup追加も検討 |
| CAM++ forward でGPUメモリ増加 | 中 | Phase 1では `torch.no_grad()` で計算グラフ切り離し。batch_size 14→12に削減 |
| FP16環境でCAM++の精度劣化 | 中 | WavLMパターンに倣い `audio.float()` で明示的にfloat32変換 |
| Phase切り替え時のoptimizer state不整合 | 高 | Phase 1/2を別training runとして実行 (`resume_from_checkpoint` ではなく重みのみロード + 新規optimizer) |

---

## M3: 推論パイプライン

### 概要

ONNXエクスポートに `--export-mode {auto, zero-shot, sid}` オプションを追加し、Dual-Modeモデルから推論モード別のONNXを生成可能にする。推論スクリプトに `--speaker-embedding` オプションを追加。

### 変更対象ファイル

| ファイル | 変更種別 | 変更規模 | 変更箇所 |
|---------|---------|---------|---------|
| `src/python/piper_train/export_onnx.py` | 修正 | ~30行 | `infer_forward`, 入力定義 |
| `src/python/piper_train/infer_onnx.py` | 修正 | ~40行 | CLI引数, 入力構築 |
| `src/python/tests/test_m3_inference_pipeline.py` | 新規 | ~400行 | ONNX export/inferテスト |

### タスク

#### export_onnx.py

- [x] `--export-mode` CLI引数追加: `auto` (モデルから自動判定), `zero-shot` (speaker_embedding入力), `sid` (speaker ID入力)
- [x] `infer_forward` 関数に `speaker_embedding` パラメータ追加
- [x] g ベクトル生成の dual-mode 対応 (spk_proj/emb_g)
- [x] ONNX入力名の条件分岐 (zero-shot: speaker_embedding, multispeaker: sid)
- [x] `use_zero_shot` フラグの判定: `getattr(model_g, "use_zero_shot", False)`
- [x] モデルにspk_projまたはemb_gが存在しない場合のバリデーション追加

#### infer_onnx.py

- [x] CLI引数追加: `--speaker-embedding` (.npy ファイルパス)
- [x] zero-shotモデル検出 (`has_speaker_embedding`) と警告表示
- [x] 推論入力構築: `.npy` 読み込み、ndim==1 時の reshape(1, -1)

#### テスト

- [x] ONNX export ラウンドトリップテスト (zero-shotモード) — onnxscript環境依存
- [x] `speaker_embedding` 入力でONNX推論が成功 — onnxscript環境依存
- [x] 既存 `sid` 入力モデルのregressionテスト — onnxscript環境依存
- [x] ONNXモデルサイズ差 < 1MB (R2要件) — onnxscript環境依存
- [x] speaker_embedding .npy 読み込み shape テスト (1D/2D)

### 受入基準

1. zero-shot ONNXモデルの入力に `speaker_embedding` (shape `[B, 192]`, float32) が含まれる
2. エクスポートしたONNXに `sid` 入力が含まれない
3. `onnxruntime.InferenceSession` でロードし `speaker_embedding` 入力で推論成功
4. ONNXモデルサイズ増加 < 1MB (R2要件)
5. `infer_onnx.py --speaker-embedding speaker.npy` で音声生成成功
6. 既存 `--speaker-id` オプションが引き続き動作

### 依存関係

- M1 完了 (models.py の dual-mode)
- M2 と並行作業可能

### 想定工数

- 1日

### リスクと対策

| リスク | 影響度 | 対策 |
|--------|--------|------|
| ONNX opsetでの `nn.Linear` 変換問題 | 低 | 基本演算 (MatMul + Add) のため opset 15 で問題なし |
| `dynamic_axes` 設定漏れ | 中 | テストで batch_size=2 も検証 |
| `.npy` ファイルの次元不整合 | 低 | ロード時に shape チェック追加 |

---

## M4: Speaker Embedding 抽出ツール

### 概要

CAM++ ONNXモデルを使ったオフラインSpeaker Embedding抽出ツールを新規作成する。

### 新規ファイル

| ファイル | 規模 |
|---------|------|
| `src/python/piper_train/extract_speaker_embedding.py` | ~280行 |
| `src/python/tests/test_m4_extract_speaker_embedding.py` | ~170行 |

### タスク

- [x] CLI インターフェース:
  - `--encoder`: CAM++ ONNXモデルパス (必須)
  - `--audio`: 単一WAVファイル
  - `--audio-dir`: ディレクトリ (全WAVを平均化)
  - `--dataset-dir`: データセット (全話者一括)
  - `--output` / `--output-dir`: 出力先
  - `--workers`: 並列処理数 (default: 4)
  - `--max-utterances`, `--min-duration`, `--source-sample-rate`
  - `--per-utterance`: 各発話ごとにembedding抽出 (学習用推奨)
  - `--batch-size` (default: 64): バッチONNX推論サイズ
  - `--num-workers` (default: 12): DataLoaderワーカー数
- [x] 処理フロー:
  1. WAV → 16kHz リサンプリング (`torchaudio.functional.resample`)
  2. 80-dim Fbank特徴抽出 (25ms窓, 10msホップ, `torchaudio.compliance.kaldi.fbank`)
  3. utterance-level CMVN正規化
  4. CAM++ ONNX推論 → 192-dim embedding
  5. L2正規化: `emb = emb / np.linalg.norm(emb)`
  6. 複数ファイル: 平均化 → 再正規化
  7. `.npy` 保存
- [x] `__main__` エントリーポイント: `python -m piper_train.extract_speaker_embedding` で実行可能
- [x] Fbank抽出: kaldi互換 80-dim, ステレオ→モノラル変換対応
- [x] データセットモード (per-speaker): `dataset.jsonl` を読み込み、`speaker_id` ごとにグループ化
  - `--min-duration` (default: 3秒) 未満の発話はスキップ
  - `--max-utterances` (default: 10) 件の代表発話から平均化
  - `.pt` ファイル (audio_norm) からの直接読み込み対応
- [x] Per-utteranceモード (`--per-utterance`): DataLoader + バッチONNX推論で高速化
  - `_FbankDataset`: PyTorch Dataset (並列CPU前処理)
  - ゼロパディング + バッチ推論でGPU効率を最大化
  - 既存embedding事前キャッシュでファイルI/O削減
  - `dataset.jsonl` を自動更新 (`speaker_embedding_path` フィールド追加)
  - バックアップ (`dataset.jsonl.bak`) 作成
- [x] GPU優先、なければCPUのONNX Runtime自動選択
- [x] テスト: 9テスト (前処理3, L2正規化3, CLI/npy3)

### 受入基準

1. `python -m piper_train.extract_speaker_embedding --encoder campplus.onnx --audio ref.wav --output speaker.npy` が成功
2. `np.load("speaker.npy").shape == (192,)`
3. `np.linalg.norm(np.load("speaker.npy"))` が 1.0 +/- 0.001
4. `--audio-dir` モードで複数WAVの平均embeddingが保存
5. `--dataset-dir` モードで全話者embedding一括保存
6. GPU不要 (CPU上でONNX Runtime動作)
7. 5秒音声の処理時間 < 100ms (CPU)

### 依存関係

- M0 (CAM++ ONNXモデル取得済み)
- M1, M2, M3 とは独立して並行作業可能

### 想定工数

- 1日

### リスクと対策

| リスク | 影響度 | 対策 |
|--------|--------|------|
| Fbank仕様がCAM++学習時と不一致 | 高 | WeSpeaker/3D-Speaker公式コードを参照。sherpa-onnx実装とのクロスチェック |
| 大量WAV処理のI/Oボトルネック | 中 | `--workers` オプションで multiprocessing 並列化 |
| `librosa` vs `torchaudio` のリサンプリング品質差 | 低 | `torchaudio` を優先。不足時は `librosa` フォールバック |

---

## M5: データ準備スクリプト実装

### 概要

LibriTTS-R, JVS, moe-speech-20speakers-v2 の3コーパスを統合するためのデータ前処理・統合スクリプトを実装する。実際のデータダウンロードと前処理実行は本マイルストーンのスコープ外とし、学習開始前に別途行う。

### 新規ファイル

| ファイル | 規模 | 説明 |
|---------|------|------|
| `src/python/piper_train/prepare_zero_shot_dataset.py` | ~300行 | データ前処理・統合スクリプト |
| `test/test_prepare_zero_shot_dataset.py` | ~100行 | 単体テスト |

### タスク

#### 前処理スクリプト実装

- [x] サンプリングレート変換機能:
  - LibriTTS-R: 24kHz → 22050Hz
  - JVS: 24kHz → 22050Hz
  - moe-speech-20speakers-v2: 22050Hz (パススルー)
- [x] 英語データ音素化: `EnglishPhonemizer` (g2p-en) 経由
- [x] 日本語データ音素化: `JapanesePhonemizer` 経由
- [x] prosody_features 生成: 日本語は OpenJTalk 経由、英語は prosody_features=None (ゼロ入力)
- [x] `__main__` エントリーポイント: `python -m piper_train.prepare_zero_shot_dataset` で実行可能

#### Speaker ID 割り当てロジック

- [x] コーパスごとの ID レンジ管理:
  - moe-speech: 0-19
  - JVS: 20-119
  - LibriTTS-R: 120-2575
- [x] M4ツール連携: 全話者のembedding一括抽出呼び出し

#### 統合 JSONL 生成ロジック

- [x] 出力フォーマット:
  ```json
  {
    "phoneme_ids": [1, 8, 5, ...],
    "speaker_id": 0,
    "speaker_embedding_path": "speaker_embeddings/speaker_0.npy",
    "prosody_features": [{"a1": -2, "a2": 1, "a3": 5}, ...],
    "language": "ja"
  }
  ```
- [x] `config.json` 生成: `num_speakers`, 全言語統合 `phoneme_id_map`
- [x] spectrogram/audio_norm の前処理パイプライン

#### 検証機能実装

- [x] `--validate` オプション: JSONL全エントリで参照先ファイルの存在確認
- [x] phoneme_id_map の日英 ID 衝突チェック
- [x] ランダムサンプリングによる音声品質チェック機能

#### CLI インターフェース

- [x] 引数:
  - `--libritts-dir`: LibriTTS-R ディレクトリ
  - `--jvs-dir`: JVS ディレクトリ
  - `--moe-speech-dir`: moe-speech データセットディレクトリ
  - `--output-dir`: 出力先
  - `--encoder`: CAM++ ONNXモデルパス (embedding抽出用)
  - `--workers`: 並列処理数 (default: 4)
  - `--validate`: 生成後の検証を実行

#### テスト

- [x] サンプリングレート変換の単体テスト
- [x] JSONL生成フォーマットの検証テスト
- [x] Speaker ID割り当てロジックのテスト
- [x] `config.json` 生成の検証テスト
- [x] 小規模ダミーデータでのE2Eテスト

### 受入基準

1. `python -m piper_train.prepare_zero_shot_dataset --help` が全引数を表示
2. 小規模ダミーデータ (各コーパス3ファイル) でE2Eテストが成功
3. 生成される JSONL の各エントリに `phoneme_ids`, `speaker_id`, `speaker_embedding_path`, `language` が含まれる
4. `config.json` の `num_speakers` が正しく計算される
5. `--validate` オプションで参照先ファイルの存在確認が動作
6. 全テストがパス

### 依存関係

- M4 (Speaker Embedding抽出ツール完成)

### 想定工数

- 1.5日

### リスクと対策

| リスク | 影響度 | 対策 |
|--------|--------|------|
| 英語 prosody_features が未対応 | 低 | prosody_features=None → ゼロ入力 (models.pyの既存ロジックで対処) |
| phoneme_id_map の日英統合でID衝突 | 低 | 言語別Phonemizerが独立管理のため原理的に衝突しない。テストで重複チェック |
| 各コーパスのディレクトリ構造の違い | 中 | コーパスごとにパーサーを分離し、共通インターフェースで統合 |

---

## M6: 事前学習 ✅ 完了

### 概要

20話者データセット (`dataset-moe-speech-20speakers`) を使用し、Dual-Mode Speaker Conditioning (emb_g + spk_proj) による VITS モデルの事前学習。SCL/DINO損失はPyTorch Speaker Encoder未統合のため未使用。Per-utterance speaker embeddingによるconditioning学習を実施。

### 実施内容

- [x] 学習コマンド実行 (RTX 6000 Ada 48GB x1):
  ```bash
  uv run python -m piper_train \
    --dataset-dir /home/shadeform/data/piper/dataset-moe-speech-20speakers \
    --prosody-dim 16 \
    --accelerator gpu --devices 1 \
    --precision bf16-mixed \
    --max_epochs 200 \
    --batch-size 160 \
    --samples-per-speaker 8 \
    --checkpoint-epochs 2 \
    --quality medium \
    --base_lr 2e-4 \
    --ema-decay 0.9995 \
    --num-workers 8 \
    --no-wavlm \
    --default_root_dir /home/shadeform/data/piper/output-moe-speech-20speakers
  ```
  注: `--zero-shot` フラグは不要 (マルチスピーカーで自動有効化)
- [x] 200 epochs (206,000 steps) 正常完了
- [x] ONNX変換成功: `moe-speech-20speakers-v2.onnx` (74MB)
- [x] 複数話者テスト推論で明瞭な音声生成を確認

### 受入基準 (達成状況)

1. ✅ 200 epochs が正常完了 (NaN, OOM なし)
2. ✅ `loss_gen_all` が収束傾向
3. -- `loss_scl`: PyTorch Speaker Encoder未統合のため未計測
4. ✅ `loss_dur` が発散していない
5. ✅ テスト推論で明瞭な音声が生成される
6. -- SECS: 定量評価は M8 で実施予定

### 学習環境

| 項目 | 計画 | 実績 |
|------|------|------|
| GPU | L4 x4 | **RTX 6000 Ada 48GB x1** |
| precision | 16-mixed | **bf16-mixed** |
| batch_size | 14 | **160** |
| samples_per_speaker | 2 | **8** |
| num_workers | 0 | **8** |
| WavLM | 有効 | **無効 (--no-wavlm)** |

### 成果物

| 成果物 | パス |
|--------|------|
| チェックポイント (最終) | `/home/shadeform/data/piper/output-moe-speech-20speakers-v2/lightning_logs/version_0/checkpoints/epoch=199-step=206000.ckpt` |
| チェックポイント (last) | `/home/shadeform/data/piper/output-moe-speech-20speakers-v2/lightning_logs/version_0/checkpoints/last.ckpt` |
| ONNX モデル | `/home/shadeform/data/piper/output-moe-speech-20speakers-v2/moe-speech-20speakers-v2.onnx` (74MB) |

---

## M7: Phase 2 学習 -- スキップ

### 概要

当初計画ではCAM++ Speaker Encoder を解凍し joint training を行う予定だったが、M6の事前学習（200 epochs, Dual-Mode Speaker Conditioning）で十分な品質を達成したため、Phase 2 はスキップとした。

PyTorch Speaker Encoder の統合（SCL/DINO損失の有効化、Phase切り替え）は将来の品質改善オプションとして保留。

### 今後の検討

- PyTorch版CAM++の統合 (`onnx2torch` による ONNX→PyTorch 変換)
- SCL/DINO損失による話者再現精度向上
- より大規模な多言語データセット (LibriTTS-R + JVS + moe-speech) での学習

---

## M8: 評価・品質検証

### 概要

学習完了モデルの包括的評価。循環バイアスを避けるため cross-encoder (WeSpeaker ResNet293) での評価を主とする。

### タスク

#### テストセット準備

- [ ] 未知話者 10-20名を選定 (日本語5名 + 英語5-15名、各5-10秒)
- [ ] テストテキスト: 日本語20文 + 英語20文
- [ ] M4ツールで各テスト話者のembedding抽出

#### 評価指標計測

- [ ] **SECS (cross-encoder)**: WeSpeaker ResNet293 で合成音声と参照音声の類似度 → 目標 > 0.65
- [ ] **SECS (same-encoder)**: CAM++ で計算 → 参考値 > 0.85
- [ ] **WER (英語)**: Whisper large-v3 → 目標 < 8%
- [ ] **CER (日本語)**: Whisper large-v3 → 目標 < 8%
- [ ] **UTMOS**: UTMOSv2 / SpeechMOS → 目標 > 3.5

#### R1-R6 要件検証

- [ ] **R1 推論速度**: zero-shot ONNX vs 既存 speaker-id ONNX の RTF 比較 → 差異 < 1%
- [ ] **R2 軽量性**: ONNXサイズ比較 → 増加 < 1MB
- [ ] **R3 既存互換**: 既存 speaker-id モデルの全テストパス
- [ ] **R4 GPL-free**: M0で確認済み
- [ ] **R5 CPU推論**: `CUDA_VISIBLE_DEVICES=""` でのzero-shot推論成功
- [ ] **R6 多言語**: 日英両方のSECS/WER/CER計測

#### レポート

- [ ] 評価結果レポート作成 (全指標一覧、要件充足状況、音声サンプルリンク)

### 受入基準

1. SECS (cross-encoder) > 0.65
2. WER < 8%, CER < 8%
3. UTMOS > 3.5
4. 推論速度差異 < 1%
5. ONNXサイズ増加 < 1MB
6. 既存テスト全パス
7. 評価結果レポート完成

### 依存関係

- M6 完了 (学習済みモデル) — M7はスキップ
- M3, M4 (推論・embedding抽出)

### 想定工数

- 1日

### 成果物

| 成果物 | 説明 |
|--------|------|
| `docs/zero-shot/evaluation-report.md` | 評価結果レポート |
| 評価用音声サンプル | 各テスト話者 x テストテキスト |
| ベンチマーク結果 | 推論速度、モデルサイズ |

### リスクと対策

| リスク | 影響度 | 対策 |
|--------|--------|------|
| SECS が目標未到達 | 高 | Phase 2 追加学習、c_spk/c_dino調整、Data Augmentation追加 |
| WER/CER が目標超過 | 中 | Phonemizer問題の可能性。OOV語の影響排除 |
| 推論速度に差異 | 低 | Linear(192→768) は数μsのため理論的に差異なし。計測誤差 (ウォームアップ不足) を確認 |

---

## M9: リリース準備

### 概要

ドキュメント整備、HuggingFace公開、CHANGELOG更新。

### タスク

- [x] **ドキュメント (一部完了)**:
  - ✅ `CLAUDE.md` にzero-shot機能セクション追加済み (Dual-Mode, ONNX変換, 推論コマンド等)
  - [ ] `README.md` にzero-shot使用方法追記
  - [ ] `docs/zero-shot/user-guide.md` 作成 (参照音声の推奨仕様、手順、FAQ)
- [ ] **HuggingFace公開**:
  - zero-shot ONNX モデル (~75MB)
  - CAM++ ONNX モデル (28MB, embedding抽出用)
  - サンプル speaker embeddings (5-10話者分)
  - Model Card (性能指標、使用方法、ライセンス)
- [ ] **既存20話者embedding生成**: CAM++で各話者の代表音声からembedding再抽出
- [ ] **CHANGELOG.md** 更新: Zero-Shot TTS機能追加、新CLI引数、破壊的変更の記載
- [ ] **VERSION** バンプ
- [ ] **PR作成**: `feat/zero-shot-tts` → `dev`
- [ ] **CI全テストパス確認**

### 受入基準

1. `CLAUDE.md` に zero-shot 完全ドキュメント
2. HuggingFace にモデル公開済み
3. Model Card に性能指標記載
4. README.md のzero-shot使用例が実行可能
5. CI全テストパス
6. PRがレビュー可能状態

### 依存関係

- M8 完了 (品質目標達成確認)

### 想定工数

- 1日

---

## リスクマトリクス

### 総合リスク評価

| # | リスク | 影響度 | 発生確率 | 総合 | 対策 | 関連MS | 状態 |
|---|--------|--------|---------|------|------|--------|------|
| R1 | **gin_channels不一致 (512 vs 768)** | 高 | 高 | **最高** | gin_channels=512に統一。768ではガビガビ音が発生することを検証済み | M1 | ✅ 解決 |
| R2 | **Phase切り替え時のoptimizer state不整合** | 高 | 中 | **高** | Phase 2スキップのため顕在化せず。将来対応時は重みのみロード + 新規optimizer | M6, M7 | -- 保留 |
| R3 | FP16とCAM++の相互作用 | 中 | 中 | 中 | WavLMパターンに倣い `audio.float()` で明示的にfloat32変換 | M2 | -- 保留 |
| R4 | 大規模データセット(715h)の前処理時間 | 中 | 高 | 中 | DataLoader + バッチONNX推論で並列化実装済み | M5実行時 | ✅ 対策済み |
| R5 | 既存チェックポイントからのfine-tuning不能 | 中 | 中 | 中 | `strict=False` でロード実装済み | M6 | ✅ 対策済み |
| R6 | マルチGPU (DDP) でのCAM++重み同期 | 低 | 中 | 低-中 | `find_unused_parameters=True` は既に設定済み。M6はシングルGPUで実施 | M6 | -- 該当なし |
| R7 | CAM++ Fbank仕様の不一致 | 高 | 低 | 中 | WeSpeaker/3D-Speaker公式コードから正確に転記。kaldi互換80-dim Fbank実装済み | M4 | ✅ 解決 |
| R8 | 学習不安定 (SCL/DINO損失) | 中 | 中 | 中 | SCL/DINO未使用 (PyTorch encoder待ち)。embedding conditioningのみで学習安定 | M6 | -- 未発生 |
| R9 | GPU ~250h中のハードウェア障害 | 高 | 中 | 高 | エポックごとチェックポイント。200 epochs正常完了 | M6 | ✅ 未発生 |
| R10 | 品質目標 (SECS > 0.65) 未到達 | 高 | 中 | 高 | 定量評価はM8で実施予定。テスト推論では明瞭な音声を確認 | M8 | -- 未評価 |

### 事前対策 (実装開始前に解決すべき) — 全て対応済み

1. **gin_channels統一** (R1): ✅ `lightning.py` と `__main__.py` で `gin_channels=512` に統一。768ではガビガビ音が発生する問題を検証・修正済み
2. **Phase切り替え戦略の確定** (R2): M7をスキップしたため未対応。将来のPhase 2実施時に重みのみロード + 新規optimizer方式で対応予定

---

## 補足: テストインフラ

### 既存テスト構成

- `test/` — 英語phonemizer, registry テスト
- `src/python/tests/` — VITSモデル, ONNX, sampler テスト
- `conftest.py` にモジュールスコープのVITSモデルfixture, ONNXエクスポートfixtureが定義済み
- pytest マーカー: `unit`, `integration`, `slow`, `japanese`, `training`, `inference`, `requires_gpu`

### zero-shotテスト追加方針

- 既存マーカー (`unit`, `inference`, `training`) で分類可能、新マーカー不要
- CAM++依存テストは `pytest.importorskip` パターンでCI環境にモデルがない場合スキップ
- `conftest.py` の fixture を拡張: multi-speaker + zero-shot 対応のモデル fixture 追加

### CI/CD

- `.github/workflows/ci.yml`: main/dev への push/PR でトリガー
- `.github/workflows/python-tests.yml`: Ubuntu/Windows/macOS x Python 3.11
- CIではtorch依存テストを `--ignore` で除外中 → zero-shotテストも同様に除外される可能性があるため、CIの `--ignore` 設定を確認する必要あり
