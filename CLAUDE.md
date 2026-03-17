# Piper TTS - プロジェクト概要

Piper TTSは高品質なニューラルテキスト音声合成システムです。VITSアーキテクチャを採用し、日本語を含む多言語に対応しています。

---

## 🟡 現在の状態: Zero-Shot TTS 事前学習準備中

**ブランチ**: `feat/zero-shot-tts`

### 概要 (2026-03-17 更新)

| 項目 | 値 |
|------|-----|
| データセット | `dataset-moe-speech-20speakers`（speaker embedding付き） |
| 話者数 | 20 |
| Speaker Encoder | CAM++ (192次元, ONNX, Apache-2.0) |
| アーキテクチャ | Dual-Mode Speaker Conditioning (emb_g + spk_proj MLP) |
| gin_channels | 512 |
| SCL | Speaker Consistency Loss（`--speaker-encoder-path` 指定時有効、c_spk=1.0） |
| DINO | 自己蒸留（EMA teacher、momentum=0.996、c_dino=0.5） |
| KLアニーリング | 10エポック（0.1→1.0 線形増加） |
| Flow | mean_only=False, dilation_rate=2 |
| Decoder | FiLM条件付け (scale+shift) |
| 推論デフォルト | noise_scale=0.4, noise_scale_w=0.5 |

### CAM++ ONNXモデルのダウンロード

```bash
# HuggingFace CosyVoice-300M リポジトリから取得 (27MB, Apache-2.0)
mkdir -p /home/shadeform/data/piper/models
wget -q "https://huggingface.co/model-scope/CosyVoice-300M/resolve/main/campplus.onnx" \
  -O /home/shadeform/data/piper/models/campplus.onnx
```

**モデル仕様:**
| 項目 | 値 |
|------|-----|
| 入力 | Fbank `[batch, T, 80]` (16kHz, 80-dim) |
| 出力 | Speaker Embedding `[batch, 192]` (L2正規化済み) |
| サイズ | 27MB |
| ライセンス | Apache-2.0 |
| 出典 | 3D-Speaker / ModelScope CAM++ |

### Speaker Embedding前処理

```bash
# Step 1: Per-utterance embedding抽出（学習用・推奨）
# 各発話の音声から個別にembeddingを生成し、dataset.jsonlを自動更新
# DataLoaderで並列CPU前処理、推論は1発話ずつ実行（ゼロパディング回避）
uv run python -m piper_train.extract_speaker_embedding \
  --encoder /home/shadeform/data/piper/models/campplus.onnx \
  --dataset-dir /home/shadeform/data/piper/dataset-moe-speech-20speakers \
  --per-utterance --batch-size 64 --num-workers 12

# Step 2: Per-speaker embedding抽出（推論用reference）
# 話者ごとに複数発話の平均embeddingを生成
uv run python -m piper_train.extract_speaker_embedding \
  --encoder /home/shadeform/data/piper/models/campplus.onnx \
  --dataset-dir /home/shadeform/data/piper/dataset-moe-speech-20speakers \
  --output-dir /path/to/embeddings
```

**Per-utterance vs Per-speaker:**
- 学習にはper-utterance推奨（推論時の条件と一致、zero-shot精度向上）
- per-utteranceは `dataset.jsonl` に `speaker_embedding_path` を自動追加
- 出力: 各発話ごとに `speaker_embeddings/{hash}.npy` (192次元, 896バイト)
- 個別ONNX推論で正確なembeddingを保証（ゼロパディングによるembedding破損を回避）
- DataLoaderの `batch_size` はCPU前処理（fbank抽出）の並列度を制御

### 完了済みモデル

```
/home/shadeform/data/piper/output-moe-speech-20speakers-v2/
├── lightning_logs/version_0/checkpoints/
│   ├── epoch=199-step=206000.ckpt
│   └── last.ckpt
└── moe-speech-20speakers-v2.onnx  ← 本番用モデル (74MB)
```

### 推論テスト

```bash
CUDA_VISIBLE_DEVICES="" uv run python -m piper_train.infer_onnx \
  --model /home/shadeform/data/piper/output-moe-speech-20speakers-v2/moe-speech-20speakers-v2.onnx \
  --config /home/shadeform/data/piper/dataset-moe-speech-20speakers-v2/config.json \
  --output-dir /path/to/output \
  --text "こんにちは、今日は良い天気ですね。" \
  --speaker-id 0
```

<details>
<summary>WavLM学習情報（アーカイブ）</summary>

WavLM Discriminator学習は150/200 epochで中断。音割れ（クリッピング）が発生していた。
- WandB: https://wandb.ai/yousan/piper-tts/runs/0eftq9nt
- チェックポイント: `/home/shadeform/data/piper/output-moe-speech-20speakers-wavlm/`

</details>

---

## 実装済み機能

### Zero-Shot Speaker Conditioning改善 Phase 2 ✅ NEW (2026-03-17)

Zero-Shot TTS の話者再現精度・学習安定性・推論品質を大幅に向上させる10項目の改善。Phase 1（spk_proj MLP、SCL、DINO、L2正規化除去）をベースに、モデルアーキテクチャと学習手法の両面から強化。

**1. Flow改善: mean_only=False + dilation_rate=2:**
- `ResidualCouplingBlock` で `mean_only=False`（分散学習有効化）。affine coupling の scale も学習されるようになり、posterior→prior 変換の表現力が向上
- `dilation_rate=2`: 指数的受容野の拡大（1,2,4,8,...）で長距離依存性をキャプチャ
- 旧VITS実装の `mean_only=True` から変更

**2. TextEncoder話者条件付け:**
- `TextEncoder` に `gin_channels` 入力を追加、`self.cond = nn.Conv1d(gin_channels, hidden_channels, 1)`
- encoder出力に話者条件 `g` を加算: `x = x + self.cond(g)`
- prior分布 (m_p, logs_p) が話者依存に。zero-shot時のprior推定精度が向上

**3. Decoder FiLM条件付け:**
- 旧: additive conditioning (`x = x + self.cond(g)`)
- 新: FiLM (Feature-wise Linear Modulation) `x = x * (1.0 + scale) + shift`
- `self.cond = nn.Conv1d(gin_channels, upsample_initial_channel * 2, 1)` で scale/shift を同時生成
- 話者特性がデコーダの各特徴量に対してスケール・シフト両方で作用

**4. Duration Predictor乗算スケーリング:**
- `StochasticDurationPredictor` / `DurationPredictor` に `cond_scale` を追加
- `scale = torch.sigmoid(self.cond_scale(g)) + 0.5`、`x = x * scale + self.cond(g)`
- 話者別の発話速度をモデリング（加算のみでは表現できない話者固有のテンポ差に対応）

**5. KLアニーリング:**
- `--kl-annealing-epochs 10`（デフォルト有効）
- KL重みを 0.1 → 1.0 へ線形増加（`kl_weight = c_kl * (0.1 + 0.9 * epoch / kl_annealing_epochs)`）
- 学習初期にdecoderがメル再構成に集中し、posterior collapse を回避

**6. Speaker Embedding摂動:**
- 学習時にspeaker embeddingへ `sigma=0.02` のGaussianノイズを追加
- `speaker_embeddings = speaker_embeddings + torch.randn_like(speaker_embeddings) * 0.02`
- 推論時の未知話者embeddingに対する汎化性能を向上

**7. Validation実embedding:**
- 旧: zero-shot validation時にゼロembedding (`torch.zeros(1, 192)`) を使用 → 意味のないaudio生成
- 新: テスト発話の `speaker_embedding` が存在すれば実embeddingを使用、なければゼロにフォールバック
- validation audioの品質が実際のzero-shot推論を反映

**8. EMA spk_proj対応:**
- `EMACallback` がデコーダEMAに加え `spk_proj` もEMA追跡
- `ema_spk_proj`: validation時にshadow weightsを適用、学習時に元の重みを復元
- checkpoint保存/復元にも `ema_spk_proj_state` を含む

**9. 推論デフォルト最適化:**
- `noise_scale=0.4`（旧: 0.667）、`noise_scale_w=0.5`（旧: 0.8）
- zero-shot推論での話者類似度に最適化されたデフォルト値
- `infer_onnx.py` のargparseデフォルトを更新

**10. ロス重み調整:**
- `c_spk=1.0`（デフォルト）: 非微分SCLの適切な重み
- `c_dino=0.5`（デフォルト、旧: 0.1）: DINO自己蒸留の正則化効果を強化
- `spk_emb_dropout=0.5`（デフォルト）: dual-mode学習でemb_gとspk_projの両方を活用

**実装ファイル:**
- `src/python/piper_train/vits/models.py` — Flow mean_only=False、TextEncoder gin_channels条件付け、Decoder FiLM、DP cond_scale
- `src/python/piper_train/vits/lightning.py` — KLアニーリング、speaker embedding摂動、validation実embedding
- `src/python/piper_train/vits/ema.py` — EMA spk_proj対応
- `src/python/piper_train/infer_onnx.py` — 推論デフォルト最適化
- `src/python/piper_train/__main__.py` — --kl-annealing-epochs、c_spk/c_dino/spk_emb_dropoutデフォルト

### Zero-Shot Speaker Conditioning 改善 Phase 1 ✅ (2026-03-17)

Zero-Shot TTS の話者再現精度と学習安定性を向上させる4つの改善。

**1. spk_proj を 2-layer MLP に変更:**
- 旧: `nn.Linear(192, 512)` → 新: `Linear(192,512) -> LayerNorm -> GELU -> Linear(512,512)`
- 非線形射影によりspeaker embedding空間の表現力が向上

**2. SCL (Speaker Consistency Loss) の有効化:**
- `--speaker-encoder-path` で CAM++ ONNX モデルを指定すると有効
- 生成音声からspeaker embeddingを抽出し、参照embeddingとのコサイン類似度で損失計算
- CAM++ は CPU ONNX で推論（CamPPSpeakerEncoder、非nn.Module）
- `--c-spk` で重み調整（デフォルト1.0）

**3. L2正規化の除去 (`_get_speaker_condition`):**
- MLP (LayerNorm + GELU) がスケーリングを学習するため、L2正規化が不要に
- embedding を `spk_proj` に通した結果をそのまま使用

**4. Per-utterance embedding のゼロパディング修正:**
- 旧: バッチ内で長さを揃えるためにゼロパディング → CAM++がゼロフレームを実データとして処理し embedding が破損
- 新: DataLoader で並列 CPU 前処理（fbank 抽出）しつつ、ONNX推論は1発話ずつ実行
- `_collate_fbanks` がリスト形式で返し、各発話を個別推論

**DINO 自己蒸留:**
- `spk_proj_teacher` (EMA teacher、momentum=0.996) で話者埋め込み空間を正則化
- `dino_center` バッファで教師出力のセンタリング
- `--c-dino` で重み調整（デフォルト0.5）

**実装ファイル:**
- `src/python/piper_train/vits/models.py` — spk_proj MLP定義、L2正規化除去
- `src/python/piper_train/vits/lightning.py` — CamPPSpeakerEncoder、SCL・DINO統合
- `src/python/piper_train/vits/losses.py` — speaker_consistency_loss、dino_loss
- `src/python/piper_train/extract_speaker_embedding.py` — 個別ONNX推論、_collate_fbanks

### C++/Python CLI UX改善 ✅ NEW (2026-03-16)

C++/Python CLIにモデル管理・テキスト直接入力機能を追加。Windows UTF-8対応。

**新CLIオプション:**
| オプション | C++ | Python | 説明 |
|-----------|-----|--------|------|
| `--text TEXT` | ✅ | - | テキスト直接入力（パイプ不要） |
| `--list-models [LANG]` | ✅ | ✅ | モデル一覧表示 |
| `--download-model NAME` | ✅ | ✅ | モデルダウンロード |
| `--model-dir DIR` | ✅ | - | ダウンロード先指定 |
| `--version` | ✅ | ✅ | バージョン表示 |

**環境変数:** `PIPER_DEFAULT_MODEL`, `PIPER_DEFAULT_CONFIG`, `PIPER_MODEL_DIR`

**実装ファイル:**
- `src/cpp/main.cpp` — C++ CLI新オプション
- `src/cpp/model_manager.cpp/hpp` — モデルカタログ管理
- `src/python_run/piper/download.py` — Python モデルダウンロード
- `scripts/speak.bat`, `scripts/speak.ps1` — Windows ヘルパー

### 3段階前処理パイプライン ✅ NEW (2026-03-11)

従来の一体型前処理（~12 utt/s）を3段階に分離して大幅に高速化。合計約10-15分（従来約80分）で5-8倍高速化。

**ステージ1: 音素化（--skip-audio）**
```bash
uv run python -m piper_train.preprocess \
  --input-dir /data/moe-speech-20speakers-ljspeech \
  --output-dir /data/piper/dataset-zero-shot-20speakers \
  --language ja --sample-rate 22050 --dataset-format ljspeech \
  --max-workers 30 --skip-audio
```
- 60,233件を1分51秒で完了（538 utt/s）
- audio pathはNone（後で更新）

**ステージ2: 音声正規化（cache_audio）**
```bash
uv run python -m piper_train.tools.cache_audio \
  --dataset /data/piper/dataset-zero-shot-20speakers/dataset.jsonl \
  --cache-dir /data/piper/dataset-zero-shot-20speakers/cache/22050 \
  --sample-rate 22050 \
  --workers 30
```
- Energy VAD + soxr リサンプリング
- 60,233件を2分20秒で完了（429 it/s）
- .ptファイル保存、dataset.jsonl自動更新

**ステージ3: スペクトログラム計算（batch_spectrograms）**
```bash
uv run python -m piper_train.tools.batch_spectrograms \
  --cache-dir /data/piper/dataset-zero-shot-20speakers/cache/22050 \
  --workers 30
```
- CPU並列でスペクトログラム計算
- .spec.ptファイル保存（FP16）

**Energy VADサポート:**
- `--energy-vad` フラグ（preprocess.pyにデフォルト有効）
- `--no-energy-vad` でSilero VADにフォールバック
- `energy_vad_numpy()`: numpy vectorized RMS VAD
- `cache_norm_audio_fast()`: Energy VAD + soxr（Sileroの50倍高速）

**実装ファイル:**
- `src/python/piper_train/tools/cache_audio.py` — 音声正規化（Energy VAD + soxr）
- `src/python/piper_train/tools/batch_spectrograms.py` — CPU並列スペクトログラム計算

### Zero-Shot TTS (Dual-Mode Speaker Conditioning) ✅ (2026-03-17 更新)

マルチスピーカーモデルでデフォルト有効（フラグ不要）。`emb_g` (nn.Embedding) と `spk_proj` (2-layer MLP) が同一モデルに共存し、speaker ID指定とspeaker embedding指定の両方で推論可能。

**特徴:**
- ONNX変換時に `--export-mode {auto, zero-shot, sid}` で推論モードを分離
- gin_channels=512（768ではガビガビ音発生）
- Speaker Encoder: CAM++ ONNX (27MB, 192次元, Apache-2.0)
- Per-utterance embedding抽出で学習-推論条件を一致
- spk_proj は 2-layer MLP (`Linear(192,512) -> LayerNorm -> GELU -> Linear(512,512)`)
- `_get_speaker_condition` で L2正規化は不要（MLP がスケーリングを学習）
- SCL (Speaker Consistency Loss): `--speaker-encoder-path` 指定時に有効。CAM++ ONNX (CPU) で生成音声からspeaker embeddingを抽出し、コサイン類似度で話者一貫性を評価
- DINO 自己蒸留: EMA teacher (`spk_proj_teacher`, momentum=0.996) で話者埋め込み空間を正則化
- CamPPSpeakerEncoder: CPU ONNX ラッパー（非nn.Module、state_dict/checkpointに含まれない）
- spk_emb_dropout: 学習時にspeaker embeddingを確率的にドロップし emb_g も学習（デフォルト 0.5）

**実装ファイル:**
- `src/python/piper_train/vits/models.py` — Dual-mode SynthesizerTrn, spk_proj MLP
- `src/python/piper_train/vits/lightning.py` — CamPPSpeakerEncoder, SCL, DINO, VitsModel
- `src/python/piper_train/vits/losses.py` — speaker_consistency_loss, dino_loss
- `src/python/piper_train/export_onnx.py` — `--export-mode` フラグ
- `src/python/piper_train/extract_speaker_embedding.py` — CAM++ embedding抽出（個別推論）
- `src/python/piper_train/prepare_zero_shot_dataset.py` — 複数コーパス統合

### Phonemizer ABC + 言語レジストリ ✅ (2026-02-01)

`Phonemizer` 抽象基底クラスと言語レジストリにより、if/elif分岐を解消。新言語追加が容易に。

**新言語追加手順:**
1. `Phonemizer` を継承したクラスを作成 (`phonemize`, `phonemize_with_prosody`, `get_phoneme_id_map` を実装)
2. 必要に応じて `post_process_ids` をオーバーライド (BOS/EOS等)
3. `registry.py` の `_auto_register()` に登録

**実装ファイル:**
- `src/python/piper_train/phonemize/base.py` — `Phonemizer` ABC, 共通 `ProsodyInfo`
- `src/python/piper_train/phonemize/registry.py` — 言語レジストリ
- `src/python/piper_train/phonemize/japanese.py` — `JapanesePhonemizer`
- `src/python/piper_train/phonemize/english.py` — `EnglishPhonemizer`
- `src/python/tests/test_phonemizer_registry.py` — レジストリ・ABCテスト

### GPL-free 英語G2P (g2p-en) ✅ (2026-01-31)

g2p-en (Apache-2.0) を使用したespeak-ng互換の英語音素化。espeak-ng/piper-phonemize (GPL) なしで英語推論が可能。

**実装ファイル:**
- `src/python/piper_train/phonemize/english.py` — G2P変換
- `src/python/piper_train/infer_onnx.py` — BOS/EOS・パディング挿入
- `src/python/tests/test_english_phonemizer.py` — 42テスト

### WavLM Discriminator ✅ (2026-01-08)

Microsoft WavLMベースの知覚品質判別器。`--no-wavlm` で無効化可能。

**実装ファイル:**
- `src/python/piper_train/vits/models.py` - `WavLMDiscriminator`クラス
- `src/python/piper_train/vits/lightning.py` - 学習ループ統合

### テキスト直接入力推論 ✅ (2026-01-08)

`infer_onnx.py`に`--text`オプション追加。JSONLなしでテキストから直接音声生成。

### Issue #204: 疑問詞マーカーの拡張 ✅

日本語の疑問文の種類を区別するマーカー (`?!`, `?.`, `?~`)。

### Issue #207: 文脈依存「ん」(N) バリアント ✅

「ん」の発音が後続音によって変わることを反映 (`N_m`, `N_n`, `N_ng`, `N_uvular`)。

### prosody_features (A1/A2/A3) モデル統合 ✅

OpenJTalkから抽出されるA1/A2/A3値をDuration Predictorの入力として活用。

**A1/A2/A3の意味:**

| フィールド | 意味 | 値の例 |
|-----------|------|--------|
| A1 | アクセント核からの相対位置 | -4, -3, ..., 0, 1, ... |
| A2 | アクセント句内のモーラ位置 | 1, 2, 3, ... |
| A3 | アクセント句内の総モーラ数 | 1-10+ |

**デフォルト有効:** `--prosody-dim 16`

### SpeakerBalancedBatchSampler ✅

マルチスピーカーモデルのDuration Predictor崩壊問題を解決するカスタムバッチサンプラー。

### FP16 Mixed Precision ✅

デフォルトで有効。学習速度2-3倍向上、GPUメモリ約50%削減。

---

## 学習設定

### 推奨設定 (20話者、T4 15GB × 4、Zero-Shot事前学習用) ✅ 最新

```bash
NCCL_DEBUG=WARN NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
uv run python -m piper_train \
  --dataset-dir /data/piper/dataset-zero-shot-20speakers \
  --prosody-dim 16 \
  --accelerator gpu --devices 4 --precision 16-mixed \
  --max_epochs 200 --batch-size 20 --samples-per-speaker 4 \
  --checkpoint-epochs 2 --quality medium \
  --base_lr 2e-4 --disable_auto_lr_scaling \
  --ema-decay 0.9995 --num-workers 4 --no-pin-memory \
  --no-wavlm --no-compile \
  --max-spec-length 500 \
  --kl-annealing-epochs 10 \
  --speaker-encoder-path /data/piper/models/campplus.onnx \
  --c-spk 1.0 --c-dino 0.5 \
  --default_root_dir /data/piper/output-zero-shot-20speakers
```

**T4 × 4 メモリ対策:**
- `--num-workers 4`: DDP × 4GPU = 16ワーカー (val用は自動で2に制限)
- `--no-pin-memory`: ロックRAM使用を回避
- `--no-compile`: T4ではtorch.compileのオーバーヘッドが大きい
- `--batch-size 20`: T4の15GB VRAMに安全に収まるサイズ
- `--max-spec-length 500`: 長すぎる発話を除外してOOM防止

**SCL / DINO / KLアニーリング設定:**
- `--speaker-encoder-path`: CAM++ ONNXモデルのパスを指定するとSCL有効化
- `--c-spk 1.0`: SCL重み（デフォルト1.0、非微分SCL用）
- `--c-dino 0.5`: DINO自己蒸留重み（デフォルト0.5、Phase 1の0.1から引き上げ）
- `--spk-emb-dropout 0.5`: speaker embeddingドロップ率（デフォルト、dual-mode学習用）
- `--kl-annealing-epochs 10`: KL重みを0.1→1.0へ10エポックかけて線形増加（0で無効化）

注: `--zero-shot` フラグは不要（マルチスピーカーなら自動有効化）

### GPU別の推奨設定

| パラメータ | T4 (15GB × 4) | V100 (16GB × 4) | RTX 6000 Ada (48GB × 1) |
|---|---|---|---|
| `--devices` | **4** | 4 | 1 |
| `--precision` | **`16-mixed`** | `16-mixed` | `bf16-mixed` |
| `--batch-size` | **20** | 12 | 160 |
| `--samples-per-speaker` | **4** | 2 | 8 |
| `--num-workers` | **4** | 0 | 8 |
| `--no-pin-memory` | **必須** | 必須 | 不要 |
| `--no-compile` | **推奨** | 推奨 | 不要 |

**DDP multi-GPU メモリ注意事項:**
- total workers = `--num-workers` × `--devices` (persistent_workers=True)
- 4GPU時に `--num-workers 12` = 48ワーカー → OOM危険
- 4GPU時は `--num-workers 2-4 --no-pin-memory` を推奨

### 話者数別の推奨設定

| 話者数 | batch_size | samples_per_speaker | 実効バッチ | 備考 |
|-------|------------|---------------------|-----------|------|
| 5話者 | 20 | 4 | 20 | ✅ 検証済み |
| **20話者 (T4×4)** | **20** | **4** | **80** | Zero-Shot推奨 |
| **20話者 (Ada×1)** | **160** | **8** | **160** | RTX 6000 Ada推奨 |
| 20話者 (旧) | 32 | 4 | 32 | WavLMあり |

---

## 重要なファイルパス

### ソースコード

| 用途 | パス |
|------|------|
| 学習スクリプト | `src/python/piper_train/__main__.py` |
| VITS実装 | `src/python/piper_train/vits/` |
| 損失関数 (SCL, DINO) | `src/python/piper_train/vits/losses.py` |
| EMAコールバック (dec + spk_proj) | `src/python/piper_train/vits/ema.py` |
| Phonemizer ABC | `src/python/piper_train/phonemize/base.py` |
| 言語レジストリ | `src/python/piper_train/phonemize/registry.py` |
| 英語音素化 | `src/python/piper_train/phonemize/english.py` |
| 日本語音素化 | `src/python/piper_train/phonemize/japanese.py` |
| IDマップ | `src/python/piper_train/phonemize/jp_id_map.py` |
| トークンマッパー | `src/python/piper_train/phonemize/token_mapper.py` |
| ONNXエクスポート | `src/python/piper_train/export_onnx.py` |
| 推論スクリプト | `src/python/piper_train/infer_onnx.py` |
| Speaker Embedding抽出 | `src/python/piper_train/extract_speaker_embedding.py` |
| Zero-shotデータ準備 | `src/python/piper_train/prepare_zero_shot_dataset.py` |

### データセット・モデル

| 用途 | パス |
|------|------|
| **Zero-Shot 20話者** ✅最新 | `/data/piper/dataset-zero-shot-20speakers/` |
| 20話者 (embedding付き) | `/home/shadeform/data/piper/dataset-moe-speech-20speakers/` |
| 20話者 v2 | `/home/shadeform/data/piper/dataset-moe-speech-20speakers-v2/` |
| CAM++ ONNXモデル | `/data/piper/models/campplus.onnx` (新) / `/home/shadeform/data/piper/models/campplus.onnx` (旧) |
| 20話者 v2 ONNX | `/home/shadeform/data/piper/output-moe-speech-20speakers-v2/moe-speech-20speakers-v2.onnx` |
| つくよみちゃん | HuggingFace: `ayousanz/piper-plus-tsukuyomi-chan` |

### 便利ツール

| ツール | 実行コマンド | 用途 |
|--------|-------------|------|
| `piper_train.tools.add_prosody_features` | `uv run python -m piper_train.tools.add_prosody_features` | 既存データセットにprosody_features追加＋phoneme_ids再生成 |
| `piper_train.tools.cache_audio` | `uv run python -m piper_train.tools.cache_audio` | 音声正規化（Energy VAD + soxr リサンプリング） |
| `piper_train.tools.batch_spectrograms` | `uv run python -m piper_train.tools.batch_spectrograms` | CPU並列スペクトログラム計算 |

---

## 基本コマンド

### ONNX変換

```bash
# Speaker ID モード（従来互換）
CUDA_VISIBLE_DEVICES="" uv run python -m piper_train.export_onnx \
  --export-mode sid \
  /path/to/checkpoint.ckpt /path/to/output_sid.onnx

# Zero-shot モード（speaker embedding入力）
CUDA_VISIBLE_DEVICES="" uv run python -m piper_train.export_onnx \
  --export-mode zero-shot \
  /path/to/checkpoint.ckpt /path/to/output_zs.onnx
```

| オプション | デフォルト | 説明 |
|-----------|----------|------|
| `--export-mode` | `auto` | `auto`: 自動判定, `sid`: speaker ID入力, `zero-shot`: speaker embedding入力 |
| `--stochastic` | off | noise_scaleによるサンプリングを有効化 |
| `--use-ema` | on | チェックポイントのEMA重みをデコーダに適用 |
| `--no-ema` | - | EMA重み適用を無効化 |

### 推論テスト

```bash
# Speaker ID指定（従来方式）
CUDA_VISIBLE_DEVICES="" uv run python -m piper_train.infer_onnx \
  --model /path/to/sid_model.onnx \
  --config /path/to/config.json \
  --output-dir /path/to/output \
  --text "こんにちは、今日は良い天気ですね。" \
  --speaker-id 0

# Zero-shot推論（参照音声から自動embedding抽出）
CUDA_VISIBLE_DEVICES="" uv run python -m piper_train.infer_onnx \
  --model /path/to/zero_shot_model.onnx \
  --config /path/to/config.json \
  --output-dir /path/to/output \
  --text "こんにちは、今日は良い天気ですね。" \
  --speaker-audio /path/to/reference.wav \
  --speaker-encoder /path/to/campplus.onnx

# JSONL入力
cat test.jsonl | CUDA_VISIBLE_DEVICES="" uv run python -m piper_train.infer_onnx \
  --model /path/to/model.onnx \
  --output-dir /path/to/output
```

**Zero-shot推論の注意:**
- `--speaker-audio`: 参照音声WAVファイル。CAM++で自動的にspeaker embeddingを抽出
- `--speaker-encoder`: CAM++ ONNXモデルのパス。未指定時はモデルファイル近傍で `campplus.onnx` を自動検索
- speaker_embedding.npy: 192次元float32配列。`extract_speaker_embedding.py` で事前生成も可能

---

## トラブルシューティング

### 推論音声が「ピー」音になる

**原因**: Duration Predictorの学習失敗

**対処法**:
1. `--samples-per-speaker` を使用
2. `--disable_auto_lr_scaling` を使用
3. 学習率を下げる（`--base_lr 1e-4`）

### gin_channels=768でガビガビ音が発生

**原因**: gin_channelsが大きすぎてspeaker conditioningが過剰に影響

**対処法**: gin_channels=512を使用（デフォルト）

### GPUメモリ不足 (GPU OOM)

**対処法**:
1. `batch_size` と `samples_per_speaker` を下げる
2. 異なるバッチサイズからのリジュームを避ける

### システムRAM不足でインスタンス再起動 (CPU OOM)

**原因**: DDP multi-GPUでDataLoaderワーカーが増殖
- total workers = `num_workers` × `devices` × 2 (train + val)
- `persistent_workers=True` でワーカーが常駐
- `pin_memory=True` でロックされたRAMが増大

**対処法**:
1. `--num-workers 2-4 --no-pin-memory` を使用
2. val_dataloaderは自動でworkers数を制限 (max 2)

### Zero-shot推論でエラーが出る

**対処法**:
1. `--speaker-audio` で指定する参照WAVファイルの存在を確認
2. `--speaker-encoder` で指定する CAM++ ONNXモデルの存在を確認（未指定時はモデル近傍で自動検索）
3. `--export-mode zero-shot` で変換したONNXモデルを使用しているか確認
4. 事前生成した `.npy` を使う場合は `numpy.load(path).shape` が `(192,)` であることを確認

### SCLが有効にならない

**原因**: `--speaker-encoder-path` が未指定または指定パスにCAM++ ONNXが存在しない

**対処法**:
1. `--speaker-encoder-path /data/piper/models/campplus.onnx` を学習コマンドに追加
2. ファイルの存在を確認: `ls -la /data/piper/models/campplus.onnx`
3. ログで `CamPPSpeakerEncoder loaded` が出力されていることを確認

### c_spkが大きすぎてSCLが学習を支配する

**原因**: `--c-spk` の値が高すぎる場合がある（非微分SCLなので過大な重みは不安定化を招く）

**対処法**: デフォルト `--c-spk 1.0` から開始し、loss_spk の推移を見ながら調整

### Speaker embeddingの品質が低い（話者再現精度が悪い）

**原因**: Per-utterance抽出でゼロパディングが使用されている（旧バージョン）

**対処法**:
1. 最新版の `extract_speaker_embedding.py` を使用（個別ONNX推論、パディングなし）
2. 既存の `speaker_embeddings/` ディレクトリを削除して再抽出

### KLアニーリングが効かない

**原因**: `--kl-annealing-epochs 0` が設定されている、またはepoch数が既にアニーリング期間を超過

**対処法**:
1. `--kl-annealing-epochs 10`（デフォルト）を確認。0に設定するとアニーリング無効
2. ログで `kl_weight` の値を確認。アニーリング中は 0.1 から 1.0 へ線形増加
3. 途中再開の場合、`current_epoch` がアニーリング期間内か確認

### ONNX変換エラー

- `CUDA_VISIBLE_DEVICES=""`でCPUモードを使用

---

## HuggingFaceリソース

| リソース | URL |
|----------|-----|
| つくよみちゃんモデル | `ayousanz/piper-plus-tsukuyomi-chan` |
| 20話者データセット | `ayousanz/moe-speech-20speakers-ljspeech` |
| ベースモデル | `ayousanz/piper-plus-base` |

---

## 関連PR/Issue

| PR/Issue | 内容 | 状態 |
|----------|------|------|
| PR #244 | C++/Python CLI UX改善 (--text, --list-models, --download-model) | Open |
| `feat/zero-shot-tts` | Zero-Shot TTS (Dual-Mode Speaker Conditioning) | 開発中 |
| PR #230 | Docker テスト強化・ブランチ統一 | Merged |
| PR #229 | C++/Python音素化パイプライン同期 (M1-M4) | Merged |
| PR #212 | WavLM Discriminator追加 | Open |
| PR #210 | Issue #204, #207 実装 | Open |
| Issue #204 | 疑問詞マーカーの拡張 | 実装完了 |
| Issue #207 | 文脈依存N phoneme variants | 実装完了 |
| Issue #198 | WavLM Discriminator | 実装完了 |
| PR #196 | A1/A2/A3 prosody機能 | Merged |
| PR #195 | FP16 Mixed Precision | Merged |
