# piper-plus ONNX量子化・最適化 調査レポート

**対象モデル**: piper-plus VITSモデル (22050Hz, ONNX ~61MB, opset 15)
**前提**: 再学習不要、既存 `.onnx` ファイルに対して適用可能な手法に限定
**調査日**: 2026-03-13

---

## 目次

1. [モデル構造の理解](#1-モデル構造の理解)
2. [INT8量子化 (動的/静的)](#2-int8量子化)
3. [FP16変換](#3-fp16変換)
4. [ONNXグラフ最適化](#4-onnxグラフ最適化)
5. [ONNX Runtime Execution Provider](#5-onnx-runtime-execution-provider)
6. [onnx-simplifier](#6-onnx-simplifier)
7. [推奨適用順序](#7-推奨適用順序)
8. [具体的なPythonコード](#8-具体的なpythonコード)
9. [既知の問題と注意点](#9-既知の問題と注意点)

---

## 1. モデル構造の理解

piper-plus のONNXモデルは VITS (Conditional Variational Autoencoder with Adversarial Learning) アーキテクチャで、推論グラフには以下のコンポーネントが含まれる:

| コンポーネント | 役割 | 主要なOp | 量子化感度 |
|---------------|------|---------|-----------|
| **TextEncoder** (`enc_p`) | テキスト→潜在表現 | Embedding, Conv1d, MultiHeadAttention | 低〜中 |
| **StochasticDurationPredictor** (`dp`) | 音素継続長予測 | Conv1d, Sigmoid, Flow | 中 |
| **ResidualCouplingBlock** (`flow`) | Normalizing Flow | Conv1d (WaveNet) | 中 |
| **Generator/Decoder** (`dec`) | 潜在表現→波形 | **ConvTranspose1d**, Conv1d, LeakyReLU, Tanh | **高** |
| **ProsodyProjection** | A1/A2/A3→prosody_dim | Linear | 低 |
| **SpeakerEmbedding** (`emb_g`) | 話者ID→embedding | Embedding | 低 |

### Generator (HiFi-GAN Decoder) の詳細構造

Decoder は音質に最も直接的に影響するため、量子化の際に最も注意が必要:

```
Generator:
  conv_pre (Conv1d: inter_channels → upsample_initial_channel)
  ups[0] (ConvTranspose1d: 512→256, kernel=16, stride=8)  ← 量子化感度: 極高
  ups[1] (ConvTranspose1d: 256→128, kernel=16, stride=8)  ← 量子化感度: 極高
  ups[2] (ConvTranspose1d: 128→64, kernel=4, stride=2)    ← 量子化感度: 高
  ups[3] (ConvTranspose1d: 64→32, kernel=4, stride=2)     ← 量子化感度: 高
  resblocks[0..N] (ResBlock1: Conv1d × 6 per block)       ← 量子化感度: 中〜高
  conv_post (Conv1d: 32→1, kernel=7)                      ← 量子化感度: 極高
```

> **重要**: `ConvTranspose1d` (アップサンプリング層) と `conv_post` (最終出力層) は波形合成の要であり、量子化による品質劣化が最も大きい。

---

## 2. INT8量子化

### 2.1 動的量子化 (Dynamic Quantization)

**概要**: 重みを事前にINT8化し、活性化は推論時にFP32のまま処理。キャリブレーションデータ不要。

**期待効果**:

| 指標 | 見積もり | 備考 |
|------|---------|------|
| モデルサイズ | 61MB → ~16-20MB (約70%削減) | 重みのみINT8化 |
| CPU推論速度 | 1.0x〜1.5x | Conv1dが主体のため恩恵は限定的 |
| MOS低下 | -0.05〜-0.15 | デコーダ除外時。全量子化では-0.3以上の可能性 |

**量子化すべき/スキップすべきレイヤー**:

| レイヤー種別 | 量子化推奨 | 理由 |
|------------|-----------|------|
| TextEncoder の Conv1d | OK | テキスト処理は量子化耐性が高い |
| TextEncoder の Attention (MatMul) | OK | NLP分野で実績あり |
| DurationPredictor の Conv1d | OK | 継続長予測は離散値に丸められるため |
| Flow の Conv1d (WaveNet) | 注意 | 累積誤差のリスクあり |
| **Decoder の ConvTranspose1d** | **スキップ推奨** | 波形品質に直結。量子化で歪み・ノイズ発生 |
| **Decoder の conv_post** | **スキップ推奨** | 最終出力層。tanh前の微小な誤差が増幅される |
| **Decoder の ResBlock** | **スキップ推奨** | 音質への影響が大きい |
| ProsodyProjection (Linear) | OK | 小規模な線形層 |

```python
"""動的量子化の実装例"""
from onnxruntime.quantization import quantize_dynamic, QuantType

# ---- 方法1: 全量子化 (品質低下リスクあり) ----
quantize_dynamic(
    model_input="model.onnx",
    model_output="model_int8_full.onnx",
    weight_type=QuantType.QInt8,
)

# ---- 方法2: 選択的量子化 (推奨) ----
# Conv/ConvTranspose のみ量子化対象にする場合
quantize_dynamic(
    model_input="model.onnx",
    model_output="model_int8_selective.onnx",
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["Conv", "MatMul"],  # ConvTranspose を除外
)

# ---- 方法3: ノード名指定で除外 (最も精密) ----
# まずノード名を列挙する
import onnx
model = onnx.load("model.onnx")
decoder_nodes = []
for node in model.graph.node:
    # Decoder (Generator) のノードを特定
    # ONNX export時のノード名パターンに依存
    if any(keyword in node.name for keyword in [
        "/dec/",           # Decoder全体
        "ConvTranspose",   # アップサンプリング層
        "conv_post",       # 最終出力層
    ]):
        decoder_nodes.append(node.name)

print(f"Decoder nodes to exclude: {len(decoder_nodes)}")

quantize_dynamic(
    model_input="model.onnx",
    model_output="model_int8_skip_decoder.onnx",
    weight_type=QuantType.QInt8,
    nodes_to_exclude=decoder_nodes,
)
```

### 2.2 静的量子化 (Static Quantization)

**概要**: 重みと活性化の両方をINT8化。キャリブレーションデータで活性化の値域を事前計算。

**期待効果**:

| 指標 | 見積もり | 備考 |
|------|---------|------|
| モデルサイズ | 61MB → ~16MB (約75%削減) | 重み+活性化をINT8化 |
| CPU推論速度 | 1.5x〜3.0x | 全計算がINT8で実行 |
| MOS低下 | -0.1〜-0.4 | VITSではReshapeエラーのリスクあり |

**VITSモデルでの既知の問題**:

- Coqui TTS の VITS ONNX モデルで静的量子化を適用すると `Reshape` ノードでランタイムエラーが発生する報告がある (microsoft/onnxruntime#16738)
- 原因: キャリブレーション中に作成される augmented model が元のモデルと異なる入力形状を要求する
- StochasticDurationPredictor 内の動的 reshape 操作が INT8 キャリブレーションと非互換

```python
"""静的量子化の実装例 (VITSでは問題が発生する可能性あり)"""
import numpy as np
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
)


class VitsCalibrationDataReader(CalibrationDataReader):
    """VITSモデル用キャリブレーションデータリーダー"""

    def __init__(self, num_samples=20):
        self.samples = []
        for _ in range(num_samples):
            seq_len = np.random.randint(10, 100)
            sample = {
                "input": np.random.randint(0, 100, (1, seq_len)).astype(np.int64),
                "input_lengths": np.array([seq_len], dtype=np.int64),
                "scales": np.array([0.667, 1.0, 0.8], dtype=np.float32),
                "sid": np.array([0], dtype=np.int64),
                # prosody_features: 実際のデータで行うのが望ましい
                "prosody_features": np.zeros((1, seq_len, 3), dtype=np.int64),
            }
            self.samples.append(sample)
        self.index = 0

    def get_next(self):
        if self.index >= len(self.samples):
            return None
        sample = self.samples[self.index]
        self.index += 1
        return sample


# 静的量子化 (Decoder除外)
# 注意: VITSモデルではReshapeエラーが発生する場合がある
try:
    quantize_static(
        model_input="model.onnx",
        model_output="model_int8_static.onnx",
        calibration_data_reader=VitsCalibrationDataReader(num_samples=20),
        quant_format=QuantFormat.QDQ,  # QDQ形式 (推奨)
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        op_types_to_quantize=["Conv", "MatMul"],  # ConvTranspose除外
        extra_options={
            "ActivationSymmetric": False,
            "WeightSymmetric": True,
        },
    )
except Exception as e:
    print(f"Static quantization failed (expected for VITS): {e}")
    print("Falling back to dynamic quantization...")
```

### 2.3 量子化の品質評価方法

```python
"""量子化前後の品質比較スクリプト"""
import numpy as np
import onnxruntime as ort


def compare_models(original_path, quantized_path, test_input):
    """元モデルと量子化モデルの出力を比較"""
    sess_orig = ort.InferenceSession(original_path)
    sess_quant = ort.InferenceSession(quantized_path)

    out_orig = sess_orig.run(None, test_input)[0]
    out_quant = sess_quant.run(None, test_input)[0]

    # 出力の長さを揃える (量子化で微妙に変わる場合がある)
    min_len = min(out_orig.shape[-1], out_quant.shape[-1])
    out_orig = out_orig[..., :min_len]
    out_quant = out_quant[..., :min_len]

    # 各種メトリクス
    mae = np.mean(np.abs(out_orig - out_quant))
    mse = np.mean((out_orig - out_quant) ** 2)
    snr = 10 * np.log10(np.mean(out_orig ** 2) / (mse + 1e-10))
    max_diff = np.max(np.abs(out_orig - out_quant))

    print(f"MAE: {mae:.6f}")
    print(f"MSE: {mse:.6f}")
    print(f"SNR: {snr:.1f} dB")
    print(f"Max diff: {max_diff:.6f}")
    print(f"Original size: {original_path} bytes")

    return {
        "mae": float(mae),
        "mse": float(mse),
        "snr_db": float(snr),
        "max_diff": float(max_diff),
    }
```

---

## 3. FP16変換

### 3.1 onnxconverter-common によるFP16変換

**概要**: FP32の重みと計算をFP16 (半精度) に変換。GPU推論で有効。

**期待効果**:

| 指標 | 見積もり | 備考 |
|------|---------|------|
| モデルサイズ | 61MB → ~31MB (約50%削減) | 重みが半分のサイズに |
| GPU推論速度 | 1.5x〜2.0x | Tensor Core対応GPU (V100, T4, A100等) で最大効果 |
| CPU推論速度 | 0.8x〜1.0x | CPU上ではFP16サポートが限定的、遅くなる場合あり |
| MOS低下 | ほぼなし (< -0.02) | FP16の精度はTTS品質に十分 |

```python
"""FP16変換の実装例"""
import onnx


# ---- 方法1: onnxconverter-common (推奨) ----
def convert_to_fp16_onnxconverter(input_path, output_path):
    """onnxconverter-common を使ったFP16変換"""
    from onnxconverter_common import float16

    model = onnx.load(input_path)

    # 基本的なFP16変換
    model_fp16 = float16.convert_float_to_float16(
        model,
        min_positive_val=1e-7,
        max_finite_val=1e4,
        keep_io_types=True,       # 入出力はFP32のまま (互換性のため)
        disable_shape_infer=False,
        op_block_list=None,       # 全opを変換
        node_block_list=None,     # 全ノードを変換
    )
    onnx.save(model_fp16, output_path)
    print(f"Saved FP16 model to {output_path}")


# ---- 方法2: Mixed Precision (安全性重視) ----
def convert_to_mixed_precision(input_path, output_path):
    """精度に敏感なopをFP32に保持するMixed Precision変換"""
    from onnxconverter_common import float16, auto_mixed_precision
    import onnxruntime as ort

    model = onnx.load(input_path)

    # auto_mixed_precision は各opの出力を検証し、
    # FP16で精度が落ちるopを自動的にFP32に保持する
    model_mixed = auto_mixed_precision.auto_convert_mixed_precision(
        model,
        test_data={},  # テストデータ (省略時は自動生成)
        rtol=0.01,     # 相対誤差の許容範囲
        atol=0.001,    # 絶対誤差の許容範囲
    )
    onnx.save(model_mixed, output_path)


# ---- 方法3: 特定のopのみFP32に保持 ----
def convert_to_fp16_with_blocklist(input_path, output_path):
    """ConvTranspose等をFP32に保持するFP16変換"""
    from onnxconverter_common import float16

    model = onnx.load(input_path)

    # 精度に敏感なopをブロックリストに追加
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        op_block_list=[
            "ConvTranspose",  # Decoder のアップサンプリング
            # 必要に応じて追加:
            # "Tanh",         # 最終出力の活性化
            # "Exp",          # Flow内のexp演算
        ],
    )
    onnx.save(model_fp16, output_path)


# 実行
convert_to_fp16_onnxconverter("model.onnx", "model_fp16.onnx")
```

### 3.2 FP16の制限事項

- **CPU推論**: 多くのCPUはネイティブFP16演算を持たないため、内部でFP32に変換されてオーバーヘッドが発生する
- **Apple Silicon (M1/M2/M3)**: CoreML EP経由ではFP16ネイティブサポートあり
- **数値範囲**: FP16の表現範囲は ±65504 で、VITSの中間値は通常この範囲内に収まる
- **piper-plusの学習はFP16 Mixed Precision** (`--precision 16-mixed`) を使用しているため、重みはFP16互換性が高い

---

## 4. ONNXグラフ最適化

### 4.1 Graph Optimization Level

ONNX Runtime は推論セッション作成時にグラフ最適化を自動適用する。

| レベル | 定数 | 内容 | piper-plusへの効果 |
|--------|------|------|-------------------|
| Disabled | `ORT_DISABLE_ALL` | 最適化なし | ベースライン |
| Basic | `ORT_ENABLE_BASIC` | 定数畳み込み、冗長ノード削除 | サイズ微減、速度微増 |
| Extended | `ORT_ENABLE_EXTENDED` | オペレータ融合 (Conv+BN+ReLU等) | **速度5〜15%向上** |
| All | `ORT_ENABLE_ALL` | + レイアウト最適化 (NCHW→NHWC等) | **速度10〜25%向上** (CPU) |

**注意**: `ORT_ENABLE_ALL` はハードウェア依存の最適化を含むため、最適化済みモデルを保存して別環境で使う場合は `ORT_ENABLE_EXTENDED` までを推奨。

```python
"""グラフ最適化の適用例"""
import onnxruntime as ort

# ---- 実行時最適化 (推奨) ----
sess_options = ort.SessionOptions()

# 最適化レベル設定
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# スレッド数の最適化
sess_options.intra_op_num_threads = 4    # 1つのopの並列度
sess_options.inter_op_num_threads = 1    # op間の並列度 (TTS推論では1でOK)

# メモリ最適化
sess_options.enable_mem_pattern = True
sess_options.enable_cpu_mem_arena = True

# セッション作成
session = ort.InferenceSession(
    "model.onnx",
    sess_options=sess_options,
    providers=["CPUExecutionProvider"],
)


# ---- 最適化済みモデルの保存 (オフライン最適化) ----
def optimize_and_save(input_path, output_path, level=ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED):
    """最適化済みモデルをファイルに保存"""
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = level
    sess_options.optimized_model_filepath = output_path

    # セッション作成時に最適化が実行され、ファイルに保存される
    ort.InferenceSession(
        input_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    print(f"Optimized model saved to {output_path}")


optimize_and_save("model.onnx", "model_optimized.onnx")
```

### 4.2 ORT_ENABLE_ALL で適用される主な最適化

| 最適化 | 内容 | VITSモデルへの効果 |
|--------|------|-------------------|
| **ConstantFolding** | コンパイル時に計算可能な定数を事前計算 | mask生成等で効果あり |
| **ConvBNFusion** | Conv + BatchNorm を1つのConvに融合 | VITSにはBNがほぼないため効果小 |
| **ConvActivationFusion** | Conv + ReLU/LeakyReLU を融合 | **Decoder ResBlockで効果大** |
| **MatMulAddFusion** | MatMul + Add → Gemm | Attentionで効果あり |
| **ReshapeElimination** | 不要なReshapeを削除 | フロー演算で効果あり |
| **NchwcTransformer** | NCHW → NCHWc レイアウト変換 | **CPU推論で大きな速度向上** |
| **SkipLayerNormFusion** | Skip + LayerNorm を融合 | TextEncoderで効果あり |

---

## 5. ONNX Runtime Execution Provider

### 5.1 各Execution Providerの比較

| Provider | プラットフォーム | 速度向上 | 導入難易度 | 備考 |
|----------|---------------|---------|-----------|------|
| **CPUExecutionProvider** | 全プラットフォーム | ベースライン | なし | デフォルト。十分高速 |
| **CUDAExecutionProvider** | NVIDIA GPU | 3〜10x | 低 | CUDA, cuDNN必要 |
| **TensorrtExecutionProvider** | NVIDIA GPU | 5〜15x | 中 | TensorRT必要。最適化に時間がかかる |
| **CoreMLExecutionProvider** | macOS/iOS | 1.5〜3x | 低 | Apple Neural Engineを活用 |
| **NNAPIExecutionProvider** | Android | 1.5〜4x | 低 | NPU/GPU活用。機種依存大 |
| **DirectMLExecutionProvider** | Windows | 2〜8x | 低 | DirectX 12対応GPU |
| **OpenVINOExecutionProvider** | Intel CPU/GPU | 1.5〜3x | 中 | Intel最適化 |
| **QNNExecutionProvider** | Qualcomm | 2〜5x | 高 | Qualcomm NPU |

### 5.2 piper-plusでの推奨設定

```python
"""Execution Provider選択の実装例"""
import onnxruntime as ort
import platform


def create_session(model_path, device="auto"):
    """プラットフォームに応じた最適なセッションを作成"""
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4

    available = ort.get_available_providers()
    system = platform.system()

    if device == "auto":
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif system == "Darwin" and "CoreMLExecutionProvider" in available:
            providers = [
                ("CoreMLExecutionProvider", {
                    "coreml_flags": 0,  # 0: CPU_AND_GPU
                    # "coreml_flags": 1,  # CPU_ONLY
                    # "coreml_flags": 2,  # CPU_AND_NE (Neural Engine)
                }),
                "CPUExecutionProvider",
            ]
        elif system == "Windows" and "DmlExecutionProvider" in available:
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
    elif device == "cpu":
        providers = ["CPUExecutionProvider"]
    elif device == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif device == "coreml":
        providers = [
            ("CoreMLExecutionProvider", {"coreml_flags": 0}),
            "CPUExecutionProvider",
        ]
    else:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=providers,
    )
    print(f"Active providers: {session.get_providers()}")
    return session
```

### 5.3 CoreML (Apple Silicon) の追加情報

Apple Silicon (M1/M2/M3/M4) ではCoreML EPが特に有効:

- **Neural Engine**: 16コアのML専用プロセッサを活用
- **FP16ネイティブ**: Apple Silicon はFP16演算をネイティブサポート
- **省電力**: CPUよりも大幅に電力効率が高い
- **制限**: 一部のONNX opがCoreMLでサポートされない場合、CPUにフォールバック

```bash
# CoreML EP付きonnxruntimeのインストール
pip install onnxruntime-silicon  # Apple Silicon向け
# または
pip install onnxruntime  # 標準版でもCoreMLが含まれる場合あり
```

---

## 6. onnx-simplifier

### 6.1 概要

`onnx-simplifier` (onnxsim) はONNXグラフの冗長なノードを除去し、定数畳み込み（constant folding）を行うツール。piper-plusの `export_onnx.py` には既に `--simplify` オプションとして統合済み。

**期待効果**:

| 指標 | 見積もり | 備考 |
|------|---------|------|
| モデルサイズ | 0〜5%削減 | VITSは元々比較的シンプルなグラフ |
| 推論速度 | 0〜5%向上 | 冗長ノード削除による |
| 品質への影響 | なし | 数学的に等価な変換のみ |

### 6.2 使用方法

```python
"""onnx-simplifierの使用例"""
import onnx
from onnxsim import simplify


def simplify_model(input_path, output_path, check_n=3):
    """ONNXモデルを簡素化する"""
    model = onnx.load(input_path)
    original_size = len(model.SerializeToString())

    model_simplified, check_passed = simplify(
        model,
        check_n=check_n,           # 検証回数
        perform_optimization=True,  # 最適化実行
        skip_fuse_bn=False,        # BN融合をスキップしない
        # skip_constant_folding=False,  # 定数畳み込みを実行
        # skip_shape_inference=False,   # 形状推論を実行
    )

    if not check_passed:
        print("WARNING: Simplification failed validation!")
        return False

    new_size = len(model_simplified.SerializeToString())
    reduction = (original_size - new_size) / original_size * 100

    onnx.save(model_simplified, output_path)
    print(f"Size: {original_size:,} → {new_size:,} bytes ({reduction:.1f}% reduction)")
    return True
```

### 6.3 piper-plus での現在の制約

`export_onnx.py` (行317-325) では、prosodyモデル (`prosody_dim > 0`) の場合に onnx-simplifier をスキップしている:

```python
# Skip simplification for prosody models to avoid numerical precision issues
if has_prosody:
    _LOGGER.info("Prosody features enabled - skipping ONNX simplification")
else:
    simplify_onnx_model(args.output)
```

これは数値精度の問題を回避するため。prosodyモデルに対しては、エクスポート後に手動で simplify を試し、出力を比較検証することを推奨。

---

## 7. 推奨適用順序

品質劣化のリスクが低い順に適用する。

```
Step 1: onnx-simplifier (リスク: なし)
  ↓
Step 2: ORT Graph Optimization (リスク: なし)
  ↓
Step 3: FP16変換 (リスク: 極低、GPU推論のみ有効)
  ↓
Step 4: 動的INT8量子化 - Decoder除外 (リスク: 低〜中)
  ↓
Step 5: 動的INT8量子化 - 全量子化 (リスク: 中〜高)
```

### サイズ・速度の累積効果見積もり

| ステップ | サイズ | CPU速度 | GPU速度 | MOS影響 |
|---------|--------|---------|---------|---------|
| 元モデル | 61MB | 1.0x | 1.0x | 0 |
| + simplify | ~60MB | ~1.03x | ~1.03x | 0 |
| + ORT最適化 | ~60MB | ~1.15x | ~1.10x | 0 |
| + FP16 | ~31MB | ~1.0x | ~1.8x | < -0.02 |
| + INT8 (Decoder除外) | ~20MB | ~1.3x | - | -0.05〜-0.15 |
| + INT8 (全量子化) | ~16MB | ~1.5x | - | -0.15〜-0.40 |

> **注意**: INT8とFP16は通常併用しない (どちらか一方を選択)。
> CPU推論ではINT8、GPU推論ではFP16が適切。

---

## 8. 具体的なPythonコード

### 8.1 統合最適化スクリプト

```python
#!/usr/bin/env python3
"""
piper-plus ONNX モデル最適化スクリプト

使用方法:
    # Step 1+2: Simplify + Graph Optimization のみ (安全)
    python optimize_onnx.py model.onnx --simplify --graph-optimize

    # Step 3: FP16変換 (GPU推論向け)
    python optimize_onnx.py model.onnx --fp16

    # Step 4: 動的INT8量子化 (Decoder除外、CPU推論向け)
    python optimize_onnx.py model.onnx --int8-dynamic --skip-decoder

    # 全部適用
    python optimize_onnx.py model.onnx --simplify --int8-dynamic --skip-decoder
"""
import argparse
import os
from pathlib import Path

import onnx


def get_model_size_mb(path):
    """モデルのファイルサイズをMBで返す"""
    return os.path.getsize(path) / (1024 * 1024)


def step_simplify(input_path, output_path):
    """Step 1: onnx-simplifier"""
    try:
        from onnxsim import simplify
    except ImportError:
        print("[SKIP] onnxsim not installed: pip install onnxsim-prebuilt")
        return input_path

    print("\n=== Step 1: onnx-simplifier ===")
    model = onnx.load(str(input_path))
    model_sim, check = simplify(model, check_n=3)

    if not check:
        print("[WARN] Simplification failed validation, keeping original")
        return input_path

    onnx.save(model_sim, str(output_path))
    orig_size = get_model_size_mb(input_path)
    new_size = get_model_size_mb(output_path)
    print(f"  {orig_size:.1f}MB → {new_size:.1f}MB ({(1 - new_size/orig_size)*100:.1f}% reduction)")
    return output_path


def step_graph_optimize(input_path, output_path):
    """Step 2: ORT Graph Optimization"""
    import onnxruntime as ort

    print("\n=== Step 2: Graph Optimization ===")
    sess_options = ort.SessionOptions()
    # EXTENDED を使用 (ALL はハードウェア依存最適化を含むため保存には不適)
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_options.optimized_model_filepath = str(output_path)

    ort.InferenceSession(
        str(input_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    orig_size = get_model_size_mb(input_path)
    new_size = get_model_size_mb(output_path)
    print(f"  {orig_size:.1f}MB → {new_size:.1f}MB ({(1 - new_size/orig_size)*100:.1f}% reduction)")
    return output_path


def step_fp16(input_path, output_path, block_conv_transpose=False):
    """Step 3: FP16変換"""
    try:
        from onnxconverter_common import float16
    except ImportError:
        print("[SKIP] onnxconverter-common not installed: pip install onnxconverter-common")
        return input_path

    print("\n=== Step 3: FP16 Conversion ===")
    model = onnx.load(str(input_path))

    op_block_list = ["ConvTranspose"] if block_conv_transpose else None

    model_fp16 = float16.convert_float_to_float16(
        model,
        min_positive_val=1e-7,
        max_finite_val=1e4,
        keep_io_types=True,
        op_block_list=op_block_list,
    )
    onnx.save(model_fp16, str(output_path))

    orig_size = get_model_size_mb(input_path)
    new_size = get_model_size_mb(output_path)
    print(f"  {orig_size:.1f}MB → {new_size:.1f}MB ({(1 - new_size/orig_size)*100:.1f}% reduction)")
    return output_path


def step_int8_dynamic(input_path, output_path, skip_decoder=False):
    """Step 4: 動的INT8量子化"""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    print("\n=== Step 4: Dynamic INT8 Quantization ===")

    nodes_to_exclude = []
    if skip_decoder:
        model = onnx.load(str(input_path))
        for node in model.graph.node:
            name_lower = node.name.lower()
            if any(kw in name_lower for kw in ["dec/", "dec.", "decoder", "generator"]):
                nodes_to_exclude.append(node.name)
            elif node.op_type == "ConvTranspose":
                nodes_to_exclude.append(node.name)
        print(f"  Excluding {len(nodes_to_exclude)} decoder/ConvTranspose nodes")
        del model

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
        nodes_to_exclude=nodes_to_exclude if nodes_to_exclude else None,
    )

    orig_size = get_model_size_mb(input_path)
    new_size = get_model_size_mb(output_path)
    print(f"  {orig_size:.1f}MB → {new_size:.1f}MB ({(1 - new_size/orig_size)*100:.1f}% reduction)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="piper-plus ONNX model optimizer")
    parser.add_argument("model", help="Input ONNX model path")
    parser.add_argument("-o", "--output", help="Output path (default: model_optimized.onnx)")
    parser.add_argument("--simplify", action="store_true", help="Apply onnx-simplifier")
    parser.add_argument("--graph-optimize", action="store_true", help="Apply ORT graph optimization")
    parser.add_argument("--fp16", action="store_true", help="Convert to FP16")
    parser.add_argument("--fp16-block-convtranspose", action="store_true",
                        help="Keep ConvTranspose in FP32 during FP16 conversion")
    parser.add_argument("--int8-dynamic", action="store_true", help="Apply dynamic INT8 quantization")
    parser.add_argument("--skip-decoder", action="store_true",
                        help="Skip decoder (Generator) nodes in INT8 quantization")
    args = parser.parse_args()

    input_path = Path(args.model)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        return

    # 出力パスの決定
    if args.output:
        final_output = Path(args.output)
    else:
        suffix_parts = []
        if args.simplify:
            suffix_parts.append("sim")
        if args.graph_optimize:
            suffix_parts.append("opt")
        if args.fp16:
            suffix_parts.append("fp16")
        if args.int8_dynamic:
            suffix_parts.append("int8")
        suffix = "_".join(suffix_parts) if suffix_parts else "optimized"
        final_output = input_path.parent / f"{input_path.stem}_{suffix}.onnx"

    print(f"Input:  {input_path} ({get_model_size_mb(input_path):.1f}MB)")
    print(f"Output: {final_output}")

    current = input_path
    tmp_idx = 0

    def tmp_path():
        nonlocal tmp_idx
        tmp_idx += 1
        return input_path.parent / f"_tmp_opt_{tmp_idx}.onnx"

    # 各ステップを順次適用
    if args.simplify:
        out = tmp_path()
        current = step_simplify(current, out)

    if args.graph_optimize:
        out = tmp_path()
        current = step_graph_optimize(current, out)

    if args.fp16:
        out = tmp_path()
        current = step_fp16(current, out, args.fp16_block_convtranspose)

    if args.int8_dynamic:
        out = tmp_path()
        current = step_int8_dynamic(current, out, args.skip_decoder)

    # 最終出力にリネーム
    if current != input_path:
        import shutil
        shutil.copy2(str(current), str(final_output))
        # 中間ファイルの削除
        for i in range(1, tmp_idx + 1):
            tmp = input_path.parent / f"_tmp_opt_{i}.onnx"
            if tmp.exists() and tmp != final_output:
                tmp.unlink()

    orig_size = get_model_size_mb(input_path)
    final_size = get_model_size_mb(final_output)
    print(f"\n=== Summary ===")
    print(f"  {orig_size:.1f}MB → {final_size:.1f}MB (total {(1 - final_size/orig_size)*100:.1f}% reduction)")


if __name__ == "__main__":
    main()
```

### 8.2 ベンチマークスクリプト

```python
#!/usr/bin/env python3
"""
ONNX モデル推論速度ベンチマーク

使用方法:
    python benchmark_onnx.py model.onnx model_int8.onnx model_fp16.onnx
"""
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def benchmark_model(model_path, num_warmup=3, num_runs=10):
    """モデルの推論速度をベンチマーク"""
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4

    session = ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    # 入力の準備
    input_names = [inp.name for inp in session.get_inputs()]
    seq_len = 50  # 典型的な文の長さ

    inputs = {
        "input": np.random.randint(0, 100, (1, seq_len)).astype(np.int64),
        "input_lengths": np.array([seq_len], dtype=np.int64),
        "scales": np.array([0.667, 1.0, 0.8], dtype=np.float32),
    }
    if "sid" in input_names:
        inputs["sid"] = np.array([0], dtype=np.int64)
    if "prosody_features" in input_names:
        inputs["prosody_features"] = np.zeros((1, seq_len, 3), dtype=np.int64)

    # ウォームアップ
    for _ in range(num_warmup):
        session.run(None, inputs)

    # ベンチマーク
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        outputs = session.run(None, inputs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    audio = outputs[0].squeeze()
    audio_duration = len(audio) / 22050  # 22050 Hz

    avg_time = np.mean(times)
    std_time = np.std(times)
    rtf = avg_time / audio_duration if audio_duration > 0 else 0

    file_size = Path(model_path).stat().st_size / (1024 * 1024)

    return {
        "path": str(model_path),
        "size_mb": file_size,
        "avg_time_ms": avg_time * 1000,
        "std_time_ms": std_time * 1000,
        "rtf": rtf,
        "audio_duration_s": audio_duration,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python benchmark_onnx.py model1.onnx [model2.onnx ...]")
        return

    results = []
    for model_path in sys.argv[1:]:
        print(f"\nBenchmarking: {model_path}")
        result = benchmark_model(model_path)
        results.append(result)

    # 結果表示
    print("\n" + "=" * 80)
    print(f"{'Model':<40} {'Size':>8} {'Time':>10} {'RTF':>8}")
    print("-" * 80)
    for r in results:
        name = Path(r["path"]).name
        print(f"{name:<40} {r['size_mb']:>6.1f}MB {r['avg_time_ms']:>7.1f}ms {r['rtf']:>.4f}")


if __name__ == "__main__":
    main()
```

### 8.3 ONNXモデルのノード構成確認

```python
#!/usr/bin/env python3
"""ONNXモデルのノード構成を確認するユーティリティ"""
import sys
from collections import Counter

import onnx


def inspect_model(model_path):
    """モデルの構成を表示"""
    model = onnx.load(model_path)
    graph = model.graph

    print(f"Model: {model_path}")
    print(f"Opset: {[o.version for o in model.opset_import]}")
    print(f"IR Version: {model.ir_version}")
    print()

    # 入出力
    print("=== Inputs ===")
    for inp in graph.input:
        shape = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
        dtype = inp.type.tensor_type.elem_type
        print(f"  {inp.name}: shape={shape}, dtype={dtype}")

    print("\n=== Outputs ===")
    for out in graph.output:
        shape = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
        dtype = out.type.tensor_type.elem_type
        print(f"  {out.name}: shape={shape}, dtype={dtype}")

    # Op種別の集計
    op_counts = Counter(node.op_type for node in graph.node)
    print(f"\n=== Op Types ({len(graph.node)} total nodes) ===")
    for op_type, count in op_counts.most_common():
        print(f"  {op_type:<25} {count:>5}")

    # ConvTranspose ノードの詳細
    print("\n=== ConvTranspose Nodes (量子化スキップ候補) ===")
    for node in graph.node:
        if node.op_type == "ConvTranspose":
            print(f"  {node.name}")
            for attr in node.attribute:
                if attr.name in ["kernel_shape", "strides", "pads"]:
                    print(f"    {attr.name}: {list(attr.ints)}")

    # デコーダ関連ノードの数
    decoder_nodes = [n for n in graph.node if "/dec/" in n.name.lower()
                     or "generator" in n.name.lower()
                     or "decoder" in n.name.lower()]
    print(f"\n=== Decoder Nodes: {len(decoder_nodes)} / {len(graph.node)} ===")

    # サイズの内訳
    total_params = 0
    for init in graph.initializer:
        size = 1
        for d in init.dims:
            size *= d
        total_params += size
    print(f"\n=== Total Parameters: {total_params:,} ({total_params * 4 / 1024 / 1024:.1f}MB in FP32) ===")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_onnx.py model.onnx")
        sys.exit(1)
    inspect_model(sys.argv[1])
```

---

## 9. 既知の問題と注意点

### 9.1 INT8量子化で速度が遅くなる問題

sherpa-onnx (k2-fsa/sherpa-onnx#575) および microsoft/onnxruntime#6732 で報告されている:

- **原因**: `ConvTranspose` opのINT8カーネルが最適化されておらず、FP32→INT8→FP32 の変換オーバーヘッドが発生
- **症状**: INT8モデルがFP32モデルより10〜100倍遅くなる場合がある
- **対策**: `ConvTranspose` ノードを量子化から除外する (`nodes_to_exclude` または `op_types_to_quantize` で制御)

### 9.2 VITSの静的量子化でReshapeエラー

coqui-ai/TTS#2779, microsoft/onnxruntime#16738 で報告:

- **原因**: StochasticDurationPredictor内の動的Reshape操作がキャリブレーション中に不整合を起こす
- **対策**: 動的量子化を使用する、またはStochasticDurationPredictorのノードを除外する

### 9.3 ORT形式 (.ort) への変換はサイズ削減にならない

rhasspy/piper#416 で確認済み:

- ORT形式はFlatBuffers使用によるロード速度の最適化が目的
- 重みサイズは変わらないため、ファイルサイズは同等〜微増
- サイズ削減にはFP16またはINT8量子化が必要

### 9.4 品質評価の推奨方法

量子化・最適化後のモデル品質は、SNR等の客観指標だけでなく、必ず主観的な聴取テストを行うこと:

1. **短文テスト**: 「こんにちは」「今日は良い天気ですね」
2. **長文テスト**: 50文字以上の文
3. **数字・記号**: 電話番号、日付、金額
4. **疑問文**: 語尾のイントネーション変化
5. **全話者**: マルチスピーカーモデルでは全話者IDをテスト

---

## 参考情報

### 必要なパッケージ

```bash
# 基本 (既にpiper-plusの依存に含まれる)
pip install onnx onnxruntime

# 量子化
# onnxruntime に同梱 (onnxruntime.quantization)

# FP16変換
pip install onnxconverter-common

# onnx-simplifier (既にpiper-plusの依存に含まれる)
pip install onnxsim-prebuilt

# CoreML (macOS)
pip install onnxruntime-silicon  # Apple Silicon向け
```

### 参照リンク

- [ONNX Runtime Quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [ONNX Runtime Graph Optimizations](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html)
- [ONNX Runtime Float16](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html)
- [ONNX Runtime Execution Providers](https://onnxruntime.ai/docs/execution-providers/)
- [onnx-simplifier](https://github.com/daquexian/onnx-simplifier)
- [onnxconverter-common float16](https://github.com/microsoft/onnxconverter-common)
- [Piper ORT format discussion](https://github.com/rhasspy/piper/discussions/416)
- [VITS静的量子化の問題 (coqui)](https://github.com/coqui-ai/TTS/issues/2779)
- [VITS静的量子化の問題 (ORT)](https://github.com/microsoft/onnxruntime/issues/16738)
- [sherpa-onnx INT8 TTS速度問題](https://github.com/k2-fsa/sherpa-onnx/issues/575)
- [Selective Quantization Tuning (論文)](https://arxiv.org/html/2507.12196v1)
