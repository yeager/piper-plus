# Piper TTS - プロジェクト概要

Piper TTSは高品質なニューラルテキスト音声合成システムです。VITSアーキテクチャを採用し、日本語を含む多言語に対応しています。

---

## 🟡 現在の状態: Zero-Shot TTS 事前学習準備中

**ブランチ**: `feat/zero-shot-tts`

### 概要 (2026-03-10 更新)

| 項目 | 値 |
|------|-----|
| データセット | `dataset-moe-speech-20speakers`（speaker embedding付き） |
| 話者数 | 20 |
| Speaker Encoder | CAM++ (192次元, ONNX, Apache-2.0) |
| アーキテクチャ | Dual-Mode Speaker Conditioning (emb_g + spk_proj) |
| gin_channels | 512 |

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
uv run python -m piper_train.extract_speaker_embedding \
  --encoder /home/shadeform/data/piper/models/campplus.onnx \
  --dataset-dir /home/shadeform/data/piper/dataset-moe-speech-20speakers \
  --per-utterance

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

### Zero-Shot TTS (Dual-Mode Speaker Conditioning) ✅ NEW (2026-03-10)

マルチスピーカーモデルでデフォルト有効（フラグ不要）。`emb_g` (nn.Embedding) と `spk_proj` (nn.Linear) が同一モデルに共存し、speaker ID指定とspeaker embedding指定の両方で推論可能。

**特徴:**
- ONNX変換時に `--export-mode {auto, zero-shot, sid}` で推論モードを分離
- gin_channels=512（768ではガビガビ音発生）
- Speaker Encoder: CAM++ ONNX (27MB, 192次元, Apache-2.0)
- Per-utterance embedding抽出で学習-推論条件を一致

**実装ファイル:**
- `src/python/piper_train/vits/models.py` — Dual-mode SynthesizerTrn
- `src/python/piper_train/vits/lightning.py` — use_zero_shot デフォルト有効
- `src/python/piper_train/export_onnx.py` — `--export-mode` フラグ
- `src/python/piper_train/extract_speaker_embedding.py` — CAM++ embedding抽出
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

### 推奨設定 (20話者、RTX 6000 Ada 48GB × 1、事前学習用)

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

注: `--zero-shot` フラグは不要（マルチスピーカーなら自動有効化）

### V100 → RTX 6000 Ada 移行時の変更点

| パラメータ | V100 (16GB × 4) | RTX 6000 Ada (48GB × 1) | 理由 |
|---|---|---|---|
| `--devices` | 4 | **1** | シングルGPU |
| `--precision` | `16-mixed` | **`bf16-mixed`** | Ada LovelaceはBF16ネイティブ対応 |
| `--batch-size` | 12 | **160** | 48GB VRAMで大幅増加可能 |
| `--samples-per-speaker` | 2 | **8** | メモリ余裕でDuration Predictor安定化 |
| `--num-workers` | 0 | **8** | シングルGPU＋12コアCPUで効率化 |

### 話者数別の推奨設定

| 話者数 | batch_size | samples_per_speaker | 実効バッチ | 備考 |
|-------|------------|---------------------|-----------|------|
| 5話者 | 20 | 4 | 20 | ✅ 検証済み |
| **20話者** | **160** | **8** | **160** | RTX 6000 Ada推奨 |
| 20話者 (旧) | 32 | 4 | 32 | WavLMあり |

<details>
<summary>V100/L4マルチGPU向け設定（参考）</summary>

```bash
NCCL_DEBUG=WARN NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
uv run python -m piper_train \
  --dataset-dir /home/shadeform/data/piper/dataset-moe-speech-20speakers-v2 \
  --prosody-dim 16 \
  --accelerator gpu --devices 4 --precision 16-mixed \
  --max_epochs 200 --batch-size 12 --samples-per-speaker 2 \
  --checkpoint-epochs 1 --quality medium \
  --base_lr 2e-4 --disable_auto_lr_scaling \
  --ema-decay 0.9995 --num-workers 0 --no-pin-memory \
  --default_root_dir /home/shadeform/data/piper/output-moe-speech-20speakers-wavlm
```

</details>

---

## 重要なファイルパス

### ソースコード

| 用途 | パス |
|------|------|
| 学習スクリプト | `src/python/piper_train/__main__.py` |
| VITS実装 | `src/python/piper_train/vits/` |
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
| CAM++ ONNXモデル | `/home/shadeform/data/piper/models/campplus.onnx` |
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

# Zero-shot推論（speaker embedding入力）
CUDA_VISIBLE_DEVICES="" uv run python -m piper_train.infer_onnx \
  --model /path/to/zero_shot_model.onnx \
  --config /path/to/config.json \
  --output-dir /path/to/output \
  --text "こんにちは、今日は良い天気ですね。" \
  --speaker-embedding /path/to/speaker.npy

# JSONL入力
cat test.jsonl | CUDA_VISIBLE_DEVICES="" uv run python -m piper_train.infer_onnx \
  --model /path/to/model.onnx \
  --output-dir /path/to/output
```

**speaker_embedding.npy**: 192次元float32配列。`extract_speaker_embedding.py` で生成。

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

### GPUメモリ不足 (OOM)

**対処法**:
1. `batch_size` と `samples_per_speaker` を下げる
2. 異なるバッチサイズからのリジュームを避ける

### Zero-shot推論でエラーが出る

**対処法**:
1. `--speaker-embedding` で指定する `.npy` ファイルの存在を確認
2. `numpy.load(path).shape` が `(192,)` であることを確認
3. `--export-mode zero-shot` で変換したONNXモデルを使用しているか確認

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
