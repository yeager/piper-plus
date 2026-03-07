# Zero-Shot TTS 実装マイルストーン計画

**作成日**: 2026-03-08
**ブランチ**: `docs/zero-shot-tts-research`
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
9. [M5: データ準備・統合](#m5-データ準備統合)
10. [M6: Phase 1 学習](#m6-phase-1-学習)
11. [M7: Phase 2 学習](#m7-phase-2-学習)
12. [M8: 評価・品質検証](#m8-評価品質検証)
13. [M9: リリース準備](#m9-リリース準備)
14. [リスクマトリクス](#リスクマトリクス)

---

## 概要

本ドキュメントは、Piper TTS (VITSアーキテクチャ) にzero-shot話者再現機能を導入するための実装マイルストーン計画である。

### アーキテクチャ方針

- VITSモデルの `emb_g = nn.Embedding(n_speakers, gin_channels)` を `spk_proj = nn.Linear(192, 768)` に置換
- CAM++ Speaker Encoder はオフライン処理専用で推論時は不使用
- 事前計算済み192次元embeddingを `.npy` で保存し、推論時はファイル読み込みのみ (<1ms)

### 規模見積もり

| 項目 | 値 |
|------|-----|
| 変更ファイル数 | 8ファイル (修正) + 3ファイル (新規) |
| 追加コード量 | ~800行 (コア) + ~200行 (テスト) |
| 実装工数 | ~8日 |
| GPU計算時間 | ~190-250時間 (L4 x4) |
| 全体所要期間 | 約4-6週間 |

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
Week 1 (実装フェーズ)
  Day 1:      M0 (環境準備) + M1 (models.py, config.py) 開始
  Day 2-3:    M1 完了 → M2 (lightning.py, losses.py, dataset.py) 開始
              M4 (extract_speaker_embedding.py) 並行開始
  Day 3-4:    M3 (export_onnx.py, infer_onnx.py) ← M1完了後、M2と並行
              M4 完了
  Day 4-5:    M2 完了

Week 2 (データ準備フェーズ)
  Day 5-7:    M5 (データ準備・統合) ← M4完了が前提

Week 2-4 (学習フェーズ 1)
  Day 7+:     M6 (Phase 1 学習) ~60-80h GPU時間

Week 4-6 (学習フェーズ 2)
              M7 (Phase 2 学習) ~130-170h GPU時間

Week 6 (評価・リリース)
              M8 (評価, 1日) → M9 (リリース準備, 1日)
```

### マイルストーン依存関係

```
M0 ──→ M1 ──→ M2 ──→ M6 ──→ M7 ──→ M8 ──→ M9
  │      │            ↑
  │      └──→ M3 ─────┘
  │
  └──→ M4 ──→ M5 ────┘
```

- M1, M3, M4 は M0 完了後に並行作業可能
- M6 開始には M1, M2, M3, M5 の全完了が必要
- M4 は他の実装タスクと独立

---

## M0: 前提条件・環境準備

### 概要

実装開始前に必要なリソースの取得・環境構築・ライセンス監査を完了させる。

### タスク

- [ ] 20話者WavLM学習 (200epoch) の完了確認と最終モデル品質評価
- [ ] CAM++ 事前学習済みONNXモデルのダウンロード
  - 入手先: sherpa-onnx (ONNX直接利用可能) or ModelScope (`iic/speech_campplus_sv_zh-cn_16k-common`)
  - 動作確認: 入力 80-dim Fbank @ 16kHz → 出力 192-dim embedding
- [ ] WeSpeaker ResNet293 評価用モデルのダウンロード (`Wespeaker/wespeaker-voxceleb-resnet293-LM`)
- [ ] 学習データコーパスの取得
  - [ ] LibriTTS-R (585h, 2,456話者, CC-BY-4.0)
  - [ ] JVS (30h, 100話者, CC-BY-SA-4.0)
  - [ ] moe-speech-20speakers-v2 (100h, 20話者) 利用可能確認
- [ ] ライセンス監査: CAM++ (Apache-2.0), WeSpeaker (Apache-2.0), onnxruntime (MIT), torchaudio (BSD-2) → GPL-free確認
- [ ] L4 GPU x4 の利用可能性確認
- [ ] 依存パッケージの確認と `pyproject.toml` の更新計画
  - 新規 optional-dependencies グループ `[zero-shot]` の設計
  - 必要パッケージ: `onnxruntime` (既存暗黙使用), `torchaudio` (既存WavLMで使用), `soundfile`
- [ ] featureブランチ `feat/zero-shot-tts` の作成 (`dev` ブランチから分岐)

### 受入基準

1. CAM++ ONNXモデルで任意のWAVファイルから192次元embeddingが抽出可能
2. 全外部依存のライセンスがGPL-freeであることを文書化済み
3. 全学習データが利用可能な状態
4. L4 GPU x4 にSSH接続し学習開始可能

### 依存関係

- なし (最初のマイルストーン)

### 想定工数

- 0.5-1日

### 成果物

| 成果物 | 説明 |
|--------|------|
| ライセンス監査結果 | GPL-free要件の充足確認 |
| `campplus.onnx` (28MB) | CAM++ Speaker Encoder |
| `wespeaker-resnet293.onnx` | 評価用cross-encoder |
| ダウンロード済みデータセット | LibriTTS-R, JVS |

---

## M1: コアモデル変更

### 概要

VITSの `SynthesizerTrn` にzero-shot対応の `spk_proj` (Linear射影層) を追加し、`emb_g` (Embedding lookup) との dual-mode を実現する。

### 変更対象ファイル

| ファイル | 変更種別 | 変更規模 | 変更箇所 |
|---------|---------|---------|---------|
| `src/python/piper_train/vits/config.py` | 修正 | ~5行 | `ModelConfig` にフィールド追加 |
| `src/python/piper_train/vits/models.py` | 修正 | ~40行 | `SynthesizerTrn.__init__`, `forward`, `infer` |
| `test/test_zero_shot.py` or `src/python/tests/test_zero_shot.py` | 新規 | ~120行 | dual-modeテスト |

### タスク

- [ ] **config.py** — `ModelConfig` にフィールド追加:
  - `use_zero_shot: bool = False`
  - `spk_embed_dim: int = 192`
- [ ] **models.py** — `SynthesizerTrn.__init__` (行731-828) 変更:
  ```python
  self.use_zero_shot = use_zero_shot
  if use_zero_shot:
      self.spk_proj = nn.Linear(spk_embed_dim, gin_channels)  # 192 → 768
  elif n_speakers > 1:
      self.emb_g = nn.Embedding(n_speakers, gin_channels)
  ```
- [ ] **models.py** — `SynthesizerTrn.forward` (行868-932) 変更:
  ```python
  def forward(self, x, x_lengths, y, y_lengths, sid=None,
              prosody_features=None, speaker_embedding=None):
      if self.use_zero_shot and speaker_embedding is not None:
          g = self.spk_proj(speaker_embedding).unsqueeze(-1)  # [B, 768, 1]
      elif self.n_speakers > 1 and sid is not None:
          g = self.emb_g(sid).unsqueeze(-1)
      else:
          g = None
      # 以降変更なし — g の形状 [B, gin_channels, 1] が共通
  ```
- [ ] **models.py** — `SynthesizerTrn.infer` (行934-983) に同等の変更を適用
- [ ] **gin_channels統一ガード**: `use_zero_shot=True` かつ `gin_channels=0` の場合に768を強制設定
  - 背景: `config.py` L108-109と`lightning.py` L94-95で512にフォールバックするパスがあるが、`__main__.py` L322-323ではマルチスピーカー時に768を設定。zero-shot時はgin_channels=768を保証する必要がある
- [ ] **単体テスト** `test_zero_shot.py`:
  - `SynthesizerTrn(use_zero_shot=True)` で `spk_proj` が存在
  - `SynthesizerTrn(use_zero_shot=False, n_speakers=20)` で `emb_g` が存在 (従来モード)
  - `forward` に `speaker_embedding` (shape `[1, 192]`) を渡し出力形状が正常
  - `infer` に `speaker_embedding` を渡し音声出力の形状が正常
  - `spk_proj.weight.shape == (768, 192)` の検証
  - 既存 `n_speakers > 1` + `sid` モードのregressionテスト

### 受入基準

1. `SynthesizerTrn(use_zero_shot=True, spk_embed_dim=192, gin_channels=768)` でインスタンス化し、`model.spk_proj.weight.shape == (768, 192)`
2. `forward(x, x_lengths, y, y_lengths, speaker_embedding=torch.randn(1, 192))` が正常動作
3. `infer(x, x_lengths, speaker_embedding=torch.randn(1, 192))` が音声テンソルを返す
4. `use_zero_shot=False` のとき既存テストが全パス
5. `spk_proj` のパラメータ数 = 192 * 768 + 768 = 148,224 (~0.15M)

### 依存関係

- M0 完了

### 想定工数

- 1日

### リスクと対策

| リスク | 影響度 | 対策 |
|--------|--------|------|
| gin_channels不一致 (512 vs 768) | 高 | `use_zero_shot=True` 時に `gin_channels` を768に強制するガードを追加。テストで明示的に検証 |
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

### タスク

#### losses.py — 新規損失関数

- [ ] `speaker_consistency_loss(gen_embedding, ref_embedding)` 追加:
  ```python
  def speaker_consistency_loss(gen_embedding, ref_embedding):
      return 1.0 - F.cosine_similarity(gen_embedding, ref_embedding, dim=-1).mean()
  ```
- [ ] `dino_loss(student_emb, teacher_emb, center, tau_s, tau_t)` 追加:
  ```python
  def dino_loss(student_emb, teacher_emb, center, tau_s=0.1, tau_t=0.04):
      student_out = F.log_softmax(student_emb / tau_s, dim=-1)
      teacher_out = F.softmax((teacher_emb - center) / tau_t, dim=-1)
      return -(teacher_out * student_out).sum(dim=-1).mean()
  ```

#### lightning.py — CAM++統合と学習ループ

- [ ] `VitsModel.__init__` にパラメータ追加: `use_zero_shot`, `spk_embed_dim`, `c_spk=9.0`, `c_dino=0.1`, `freeze_speaker_encoder_steps=100000`
- [ ] CAM++ Speaker Encoder のロード処理:
  - `use_zero_shot=True` の場合にPyTorch版CAM++をロード
  - Phase 1: 全パラメータを `requires_grad=False` に設定
  - `self.dino_center` バッファ初期化 (shape: `[spk_embed_dim]`)
  - `self.current_tau_t` ウォームアップスケジュール (0.04 → 0.07)
- [ ] `SynthesizerTrn` インスタンス化に `use_zero_shot`, `spk_embed_dim` を渡す (行100-121)
- [ ] `training_step_g` 変更:
  - バッチから `speaker_embedding` を取得
  - SCL計算: `self.speaker_encoder(y_hat)` → `speaker_consistency_loss` → `loss_gen_all += c_spk * loss_scl`
  - DINO計算: `dino_loss(gen_emb, ref_emb.detach(), self.dino_center)` → `loss_gen_all += c_dino * loss_dino`
  - DINO center EMA更新: `self.dino_center = 0.996 * self.dino_center + 0.004 * ref_emb.mean(0)`
  - ログ: `loss_scl`, `loss_dino` をWandBに記録
- [ ] `on_train_batch_start` でPhase切り替え:
  - `global_step == freeze_speaker_encoder_steps` でCAM++解凍
- [ ] `configure_optimizers` 変更:
  - CAM++ パラメータを別param_groupとして追加 (lr_ratio=0.1)
- [ ] **FP16対策**: CAM++への入力を明示的に `float32` に変換 (`audio.float()`)
  - WavLM Discriminatorと同様のパターン (models.py L647参照)

#### dataset.py — speaker_embedding 対応

- [ ] `Utterance` dataclass (行18-24): `speaker_embedding_path: Path | None = None` 追加
- [ ] `UtteranceTensors` dataclass (行28-34): `speaker_embedding: FloatTensor | None = None` 追加
- [ ] `Batch` dataclass (行42-50): `speaker_embeddings: FloatTensor | None = None` 追加
- [ ] `PiperDataset.__getitem__` (行82-132): `.npy` ファイルからembeddingロード
  - I/O最適化: 192dim x float32 = 768 bytes と軽量だが、初期化時に全embeddingをメモリにプリロード (辞書キャッシュ) することを検討
- [ ] `PiperDataset.load_utterance` (行192-201): JSONから `speaker_embedding_path` パース
- [ ] `UtteranceCollate.__call__` (行209-305): `speaker_embeddings` のスタッキング (固定長192dimのためパディング不要)

#### __main__.py — CLI引数

- [ ] 以下の引数を追加:
  - `--zero-shot` (`action="store_true"`)
  - `--spk-embed-dim` (default: 192)
  - `--c-spk` (default: 9.0)
  - `--c-dino` (default: 0.1)
  - `--freeze-speaker-encoder-steps` (default: 100000)
- [ ] `dict_args` への反映: `VitsModel` に上記パラメータを渡す
- [ ] `--zero-shot` 時に `gin_channels=768` を強制

#### テスト

- [ ] SCL損失の単体テスト (出力範囲: 0-2)
- [ ] DINO損失の単体テスト (出力: 正の値)
- [ ] Phase切り替えロジックのモックテスト
- [ ] `use_zero_shot=False` 時の既存学習ループregressionテスト

### 受入基準

1. `speaker_consistency_loss(torch.randn(4, 192), torch.randn(4, 192))` が 0-2 の範囲のfloatを返す
2. `dino_loss(...)` が正のfloatを返す
3. `VitsModel(use_zero_shot=True)` で `hasattr(model, 'speaker_encoder')` が True
4. `training_step_g` が `loss_scl`, `loss_dino` をログに記録
5. `global_step == freeze_speaker_encoder_steps` で `speaker_encoder.parameters()` の `requires_grad` が True に変化
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

ONNXエクスポートで `sid` 入力を `speaker_embedding` 入力に置換し、推論スクリプトに `--speaker-embedding` オプションを追加する。

### 変更対象ファイル

| ファイル | 変更種別 | 変更規模 | 変更箇所 |
|---------|---------|---------|---------|
| `src/python/piper_train/export_onnx.py` | 修正 | ~30行 | `infer_forward`, 入力定義 |
| `src/python/piper_train/infer_onnx.py` | 修正 | ~40行 | CLI引数, 入力構築 |

### タスク

#### export_onnx.py

- [ ] `infer_forward` 関数 (行177-231) に `speaker_embedding` パラメータ追加
- [ ] g ベクトル生成の dual-mode 対応
- [ ] ONNX入力名の条件分岐 (行274-278):
  ```python
  if use_zero_shot:
      input_names = ["input", "input_lengths", "scales", "speaker_embedding"]
      dynamic_axes["speaker_embedding"] = {0: "batch_size"}
      dummy_spk = torch.randn(1, spk_embed_dim, dtype=torch.float32)
  elif num_speakers > 1:
      input_names = ["input", "input_lengths", "scales", "sid"]
  ```
- [ ] `use_zero_shot` フラグの判定: `model_g.use_zero_shot` 属性参照

#### infer_onnx.py

- [ ] CLI引数追加: `--speaker-embedding` (.npy ファイルパス)
- [ ] `--speaker-embedding` と `--speaker-id` の排他制御
- [ ] 推論入力構築:
  ```python
  if "speaker_embedding" in input_names and args.speaker_embedding:
      spk_emb = np.load(args.speaker_embedding).astype(np.float32)
      assert spk_emb.shape[-1] == 192, f"Expected 192-dim, got {spk_emb.shape}"
      inputs["speaker_embedding"] = spk_emb.reshape(1, -1)
  elif "sid" in input_names and args.speaker_id is not None:
      inputs["sid"] = np.array([args.speaker_id], dtype=np.int64)
  ```

#### テスト

- [ ] ONNX export/import ラウンドトリップテスト (zero-shotモード)
- [ ] `speaker_embedding` 入力でONNX推論が成功
- [ ] 既存 `sid` 入力モデルのregressionテスト
- [ ] batch_size=2 でのdynamic axes検証

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
| `src/python/piper_train/extract_speaker_embedding.py` | ~150行 |
| `test/test_extract_speaker_embedding.py` | ~80行 |

### タスク

- [ ] CLI インターフェース:
  - `--encoder`: CAM++ ONNXモデルパス (必須)
  - `--audio`: 単一WAVファイル
  - `--audio-dir`: ディレクトリ (全WAVを平均化)
  - `--dataset-dir`: データセット (全話者一括)
  - `--output` / `--output-dir`: 出力先
  - `--workers`: 並列処理数 (default: 4)
- [ ] 処理フロー:
  1. WAV → 16kHz リサンプリング (`torchaudio.transforms.Resample`)
  2. 80-dim Fbank特徴抽出 (25ms窓, 10msホップ, `torchaudio.compliance.kaldi.fbank`)
  3. CAM++ ONNX推論 → 192-dim embedding
  4. L2正規化: `emb = emb / np.linalg.norm(emb)`
  5. 複数ファイル: 平均化 → 再正規化
  6. `.npy` 保存
- [ ] `__main__` エントリーポイント: `python -m piper_train.extract_speaker_embedding` で実行可能
- [ ] Fbank抽出の仕様を WeSpeaker/3D-Speaker 公式コードから正確に転記
  - sherpa-onnx の実装も参考
- [ ] データセットモード: `dataset.jsonl` を読み込み、`speaker_id` ごとにグループ化
  - 3秒未満の発話はスキップ
  - 各話者最大10件の代表発話から平均化

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

## M5: データ準備・統合

### 概要

LibriTTS-R, JVS, moe-speech-20speakers-v2 の3コーパスを統合し、zero-shot学習用データセットを作成する。

### タスク

#### 前処理

- [ ] サンプリングレート統一:
  - LibriTTS-R: 24kHz → 22050Hz
  - JVS: 24kHz → 22050Hz
  - moe-speech-20speakers-v2: 22050Hz (そのまま)
- [ ] 英語データ音素化: LibriTTS-R テキストを `EnglishPhonemizer` (g2p-en) で変換
- [ ] 日本語データ音素化: JVS テキストを `JapanesePhonemizer` で変換
- [ ] prosody_features 生成: 日本語は OpenJTalk 経由、英語は prosody_features=None (ゼロ入力)

#### Speaker ID・Embedding

- [ ] Speaker ID 割り当て:
  - moe-speech: 0-19
  - JVS: 20-119
  - LibriTTS-R: 120-2575
- [ ] M4ツールで全2,576話者のembeddingを抽出 → `speaker_embeddings/speaker_N.npy`

#### 統合

- [ ] 統合 JSONL 作成:
  ```json
  {
    "phoneme_ids": [1, 8, 5, ...],
    "speaker_id": 0,
    "speaker_embedding_path": "speaker_embeddings/speaker_0.npy",
    "prosody_features": [{"a1": -2, "a2": 1, "a3": 5}, ...],
    "language": "ja"
  }
  ```
- [ ] `config.json` 作成: `num_speakers: 2576`, 全言語統合 `phoneme_id_map`
- [ ] spectrogram/audio_norm の前処理実行

#### 検証

- [ ] JSONL 全エントリで参照先ファイル (audio, spec, speaker_embedding) の存在確認
- [ ] phoneme_id_map の日英 ID 衝突チェック (Phonemizerレジストリが独立管理のため原理的に衝突しないが、念のため検証)
- [ ] ランダムサンプリングによる音声品質確認

### 受入基準

1. `/data/piper/dataset-zero-shot-merged/dataset.jsonl` が存在し 60,000+ エントリ
2. 各エントリに `phoneme_ids`, `speaker_id`, `speaker_embedding_path`, `language` が含まれる
3. `speaker_embeddings/` に 2,576 個の `.npy` ファイル (全て shape `(192,)`)
4. `config.json` の `num_speakers` が 2576
5. ランダム10エントリの audio/spec が正常に読み込み可能

### 依存関係

- M0 (データセット取得済み)
- M4 (Speaker Embedding抽出ツール完成)

### 想定工数

- 2日

### リスクと対策

| リスク | 影響度 | 対策 |
|--------|--------|------|
| LibriTTS-R (585h) の前処理に長時間 | 中 | multiprocessing で並列化。GPU 1台を前処理に割り当て |
| 英語 prosody_features が未対応 | 低 | prosody_features=None → ゼロ入力 (models.pyの既存ロジックで対処) |
| JVS 短発話のembedding品質低下 | 低 | 3秒未満の発話をembedding抽出対象から除外。複数発話平均化で品質担保 |
| phoneme_id_map の日英統合でID衝突 | 低 | 言語別Phonemizerが独立管理のため原理的に衝突しない。統合後に重複チェック |

---

## M6: Phase 1 学習

### 概要

CAM++ Speaker Encoder を凍結した状態で VITS 本体を学習する (100K iterations)。SCL + DINO 損失を有効にし、話者embedding空間の初期学習を行う。

### タスク

- [ ] 学習コマンド実行:
  ```bash
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
  ```
- [ ] WandB 監視:
  - `loss_gen_all`, `loss_scl`, `loss_dino`, `loss_dur`, `loss_kl` の推移
  - 10K steps ごとの中間チェック
- [ ] 50K steps マイルストーン: SECS (CAM++, same-encoder) > 0.80 を確認
- [ ] 100K steps (Phase 1完了): ONNX変換 + 複数話者テスト推論

### 受入基準

1. 100K iterations が正常完了 (NaN, OOM なし)
2. `loss_gen_all` が収束傾向
3. `loss_scl` が 0.5 以下に低下
4. `loss_dur` が発散していない
5. テスト推論で明瞭な音声が生成される
6. SECS (CAM++, same-encoder) > 0.80 (学習済み話者)

### 依存関係

- M1, M2, M3, M5 の全完了

### 想定工数

- 実装作業: 0.5日 (コマンド準備、監視設定)
- GPU計算時間: ~60-80時間 (L4 x4)

### 成果物

| 成果物 | 説明 |
|--------|------|
| Phase 1 チェックポイント | `/data/piper/output-zero-shot-phase1/.../last.ckpt` |
| 中間 ONNX | `zero-shot-phase1-100K.onnx` |
| WandB ログ | 損失カーブ、中間評価結果 |

### リスクと対策

| リスク | 影響度 | 対策 |
|--------|--------|------|
| GPU 80h 中のハードウェア障害 | 高 | エポックごとのチェックポイント + `resume_from_checkpoint` |
| Duration Predictor 崩壊 (ピー音) | 高 | `SpeakerBalancedBatchSampler` で対策済み。`loss_dur` 監視 |
| `loss_scl` が下がらない | 中 | c_spk=9.0 → 4.5 に調整。DINO center 初期化確認 |
| GPUメモリ不足 | 中 | batch_size 14→12→10 に段階的削減 |

---

## M7: Phase 2 学習

### 概要

CAM++ Speaker Encoder を解凍し、VITS 本体と joint training (200K iterations)。話者再現精度のさらなる向上。

### タスク

- [ ] Phase 1チェックポイントからPhase 2学習開始:
  ```bash
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
    --resume_from_checkpoint /data/piper/output-zero-shot-phase1/.../last.ckpt
  ```
  - batch_size 14→10 (CAM++解凍によるメモリ増加対応)
- [ ] WandB監視: CAM++解凍後の損失変化、gradient norm安定性
- [ ] 150K total: SECS が Phase 1 より改善を確認
- [ ] 200K total: 未知話者でのzero-shot合成、cross-encoder SECS計測
- [ ] 300K total (Phase 2完了): 最終チェックポイント + ONNX変換

### 受入基準

1. 200K iterations が正常完了
2. `loss_scl` が Phase 1 終了時よりさらに低下
3. SECS (CAM++, same-encoder) > 0.87 (学習済み話者)
4. SECS (WeSpeaker ResNet293, cross-encoder) > 0.60 (未知話者)
5. zero-shot合成の話者性が知覚的に認識可能
6. ONNX変換成功

### 依存関係

- M6 完了 (Phase 1 チェックポイント)

### 想定工数

- 実装作業: 0.5日
- GPU計算時間: ~130-170時間 (L4 x4)

### 成果物

| 成果物 | 説明 |
|--------|------|
| Phase 2 チェックポイント | `/data/piper/output-zero-shot-phase2/.../last.ckpt` |
| 最終 ONNX | `zero-shot-phase2-300K.onnx` |
| WandB ログ | 損失カーブ、中間/最終評価 |

### リスクと対策

| リスク | 影響度 | 対策 |
|--------|--------|------|
| 170h GPU計算中のハードウェア障害 | 高 | エポックごとチェックポイント。50K steps ごとに中間評価し早期終了検討 |
| CAM++解凍後のloss spike/発散 | 高 | CAM++ lr = VITS lr * 0.1。発散時はさらに0.01倍に |
| 品質目標 (SECS > 0.65) に未到達 | 中 | 追加50-100K steps継続。c_spk/c_dino調整、Data Augmentation強化 |
| GPUメモリ不足 (batch=10) | 中 | batch_size 8 に削減。gradient accumulation=2 も検討 |

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

- M7 完了 (学習済みモデル)
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

- [ ] **ドキュメント**:
  - `CLAUDE.md` にzero-shot機能セクション追加
  - `README.md` にzero-shot使用方法追記
  - `docs/zero-shot/user-guide.md` 作成 (参照音声の推奨仕様、手順、FAQ)
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

| # | リスク | 影響度 | 発生確率 | 総合 | 対策 | 関連MS |
|---|--------|--------|---------|------|------|--------|
| R1 | **gin_channels不一致 (512 vs 768)** | 高 | 高 | **最高** | `use_zero_shot=True` 時に768強制。3箇所のフォールバック値をテストで検証 | M1 |
| R2 | **Phase切り替え時のoptimizer state不整合** | 高 | 中 | **高** | Phase 1/2を別training runとして実行。重みのみロード + 新規optimizer | M6, M7 |
| R3 | FP16とCAM++の相互作用 | 中 | 中 | 中 | WavLMパターンに倣い `audio.float()` で明示的にfloat32変換 | M2 |
| R4 | 大規模データセット(715h)の前処理時間 | 中 | 高 | 中 | multiprocessing並列化。スモールセットで事前検証 | M5 |
| R5 | 既存チェックポイントからのfine-tuning不能 | 中 | 中 | 中 | zero-shotモデルはスクラッチ学習。Generator/Encoder/Flowは `strict=False` で引き継ぎ可能 | M6 |
| R6 | マルチGPU (DDP) でのCAM++重み同期 | 低 | 中 | 低-中 | `find_unused_parameters=True` は既に設定済み (`__main__.py` L56) | M6, M7 |
| R7 | CAM++ Fbank仕様の不一致 | 高 | 低 | 中 | WeSpeaker/3D-Speaker公式コードから正確に転記。sherpa-onnx実装とクロスチェック | M4 |
| R8 | 学習不安定 (SCL/DINO損失) | 中 | 中 | 中 | 論文実証済みハイパラ使用。WandB監視。必要に応じてc値調整 | M6, M7 |
| R9 | GPU ~250h中のハードウェア障害 | 高 | 中 | 高 | エポックごとチェックポイント。50K stepsごと中間評価 | M6, M7 |
| R10 | 品質目標 (SECS > 0.65) 未到達 | 高 | 中 | 高 | 追加学習、ハイパラ調整、Data Augmentation。VITSの理論上限 SIM-O ~0.75 を認識した上で目標設定 | M8 |

### 事前対策 (実装開始前に解決すべき)

1. **gin_channels統一** (R1): `config.py`, `lightning.py`, `__main__.py` の3箇所でフォールバック値が異なる問題をzero-shot実装前にクリーンアップすることを強く推奨
2. **Phase切り替え戦略の確定** (R2): Phase 1→2で `resume_from_checkpoint` を使うか、重みのみロード + 新規optimizerとするかを事前テストで確認

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
