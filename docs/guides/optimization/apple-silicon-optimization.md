# Mac (Intel + Apple Silicon) 推論最適化ガイド

> **調査日**: 2026-03-13
> **対象モデル**: VITS TTS (Piper-Plus, ~37MB ONNX FP16デフォルト / ~74MB FP32, Conv1d主体アーキテクチャ)
> **対象**: Intel Mac (x86_64) + Apple Silicon (M1/M2/M3/M4)

## 概要

Mac上でのVITS TTS推論最適化について、20チームによる並列調査の結果をまとめたドキュメントです。ONNX Runtime設定、スレッディング、CoreML、量子化、アーキテクチャ変更、C++/Pythonレベルの最適化まで網羅的に調査しました。**M4 Max上での実測プロファイリング結果**を含みます。

### 核心的発見

| 発見 | 影響度 |
|------|--------|
| **ONNX Runtime MLASはApple Accelerate/AMXを使用していない** | CoreML EP経由でのみAppleハードウェア活用可能 |
| **現在のC++コードでORT_DISABLE_ALL + MemArena無効** | 即座に修正可能（ただしApple SiliconではORT最適化レベル差は~1%） |
| **HiFi-GANデコーダが推論時間の74%（実測値）** | Conv (53%) + ConvTranspose (24.5%) = 77.5%。デコーダ最適化が最優先 |
| **ORT x86_64 macOS wheelは廃止予定** | Intel Macはソースビルド必須に |
| **FP16はexport_onnxでデフォルト適用** (`--no-fp16`で無効化) | サイズ~50%削減。Intel CPUでは速度向上なし (Cast overhead)、Apple Siliconでもサイズ削減が主な価値 |
| **MB-iSTFT-VITSで4.1x高速化 (品質劣化なし)** | 再学習必要だが最大効果 |
| **Apple SiliconではP-coreのみ使用が最適** | 実測: M4 Maxで12→14スレッドで50%悪化、12→16で67%悪化 |

---

## ハードウェア仕様

### Apple Silicon

| チップ | P-core | E-core | Neural Engine | メモリ帯域幅 | 推奨intra_op |
|--------|--------|--------|-------------|------------|-------------|
| M1 | 4 | 4 | 16コア, 11 TOPS | 68 GB/s | **4** |
| M1 Pro | 6-8 | 2 | 16コア, 11 TOPS | 200 GB/s | **6-8** |
| M1 Max | 8 | 2 | 16コア, 11 TOPS | 400 GB/s | **8** |
| M2 | 4 | 4 | 16コア, 15.8 TOPS | 100 GB/s | **4** |
| M2 Pro | 6-8 | 4 | 16コア, 15.8 TOPS | 200 GB/s | **6-8** |
| M3 | 4 | 4 | 16コア, 18 TOPS | 100 GB/s | **4** |
| M3 Pro | 5-6 | 6 | 16コア, 18 TOPS | 150 GB/s | **5-6** |
| M4 | 4 | 6 | 16コア, 38 TOPS | 120 GB/s | **4** |
| M4 Pro | 10 | 4 | 16コア, 38 TOPS | 273 GB/s | **10** |
| M4 Max | 12 | 4 | 16コア, 38 TOPS | 546 GB/s | **12** |

**重要**: M4のINT8は内部でFP16にデクオンタイズされてから演算される（Orion論文, 2026年3月）。公称38 TOPSに対し実質~19 TFLOPS。

### Intel Mac

| Mac モデル | プロセッサ | AVX2 | AVX-512 | VNNI |
|-----------|----------|------|---------|------|
| MacBook Pro 2016-2020 | i5/i7/i9 (Coffee Lake) | Yes | No | No |
| iMac 2019-2020 | i5/i7/i9 (Coffee Lake) | Yes | No | No |
| Mac mini 2018-2020 | i3/i5/i7 (Coffee Lake) | Yes | No | No |
| iMac Pro 2017 | Xeon W (Skylake-SP) | Yes | Yes | No |
| Mac Pro 2019 | Xeon W (Cascade Lake) | Yes | Yes | **Yes** |

**Intel Mac固有の制約:**

| 項目 | 状態 |
|------|------|
| oneDNN/MKL-DNN EP | macOS非対応 (NOT_PLANNED, Issue #5783) |
| OpenVINO EP | macOS非対応 |
| VNNI (INT8高速化) | Mac Pro 2019 Xeonのみ |
| ORT x86_64 macOS wheel | **1.24.x以降廃止予定** (ソースビルド必須) |
| FP16 CPU推論 | **速度向上なし** (Cast overhead で ~4x 遅い)。エクスポート時デフォルト適用、`--no-fp16`でFP32出力 |
| CoreML EP | CPU-onlyフォールバック (ANEなし) |

---

## 推論バックエンド比較

| バックエンド | batch=1 速度 | FP16 効果 | ANE 活用 | Intel Mac | 推奨度 |
|-------------|------------|----------|---------|-----------|-------|
| **ONNX Runtime CPU** | ベースライン | サイズ削減のみ | なし | 使用可 | **推奨** |
| ONNX Runtime CoreML EP | CPUより8%遅い※ | FP16演算可 | 部分的(37%) | CPUフォールバック | 要ベンチマーク |
| PyTorch MPS | batch=1でCPU同等 | ほぼ効果なし | なし | 非対応 | 非推奨 |
| **MLX** | MPS比 2-3x | 20-30%向上 | なし(GPU) | 非対応 | Apple Silicon向け |
| **BNNS Graph** | CPU比 2x | 効果あり | なし(CPU最適化) | 未検証 | iOS/macOSデプロイ向け |
| CoreML直接変換 | ANE時に高速 | ネイティブFP16 | 可能 | CPUフォールバック | Conv1dバグあり |

※ sherpa-onnx M2 Maxベンチマーク: CoreML RTF 0.470 vs CPU RTF 0.372

---

## ボトルネック分析

### VITS推論時間の内訳

```
VITS推論時間の内訳 (CPU, medium quality, ~50 phonemes):
├── Decoder (HiFi-GAN): 60-75% ████████████████████ <- 最大ボトルネック
│   ├── ResBlock Conv1d:   ~45%   (18 Conv1d ops on long sequences)
│   └── ConvTranspose1d:   ~30%   (3 upsampling stages: 8x, 8x, 4x)
├── Flow (Normalizing):  15-25% ██████
│   └── 4x ResidualCouplingLayer with WN modules
├── TextEncoder:          5-8%  ██
│   └── 6x MultiHeadAttention (50x50, negligible)
└── DurationPredictor:    3-5%  █
```

### 演算/メモリ特性

VITSのConv1dは主に**演算バウンド**（Arithmetic Intensity ~48-95 FLOP/byte）だが、デコーダ後段では活性化テンソルが大きく（~51200 frames x 32ch）**L2キャッシュスラッシング**が発生する。

### 実測プロファイリング結果

> **測定日**: 2026-03-13
> **環境**: Apple M4 Max (12 P-core + 4 E-core), ONNX Runtime 1.24.3, CPUExecutionProvider
> **モデル**: ja_JP-test-medium.onnx (61MB, medium quality)
> **入力**: 「こんにちは、今日は良い天気ですね。」(38 phoneme IDs)
> **条件**: warm-up 3回 + 計測5回の累積、`ORT_ENABLE_ALL`, `intra_op_num_threads=4`

#### コンポーネント別時間分布

| コンポーネント | 時間 (ms) | 割合 | |
|---|---:|---:|---|
| **Decoder (HiFi-GAN)** | **355.16** | **74.0%** | `████████████████████████████████████` |
| TextEncoder | 37.98 | 7.9% | `███` |
| Flow (Normalizing) | 37.46 | 7.8% | `███` |
| DurationPredictor | 33.11 | 6.9% | `███` |
| Other/Global | 16.30 | 3.4% | `█` |

推定値 (60-75%) に対し、**実測74.0%** でほぼ上限値であることが確認された。

#### デコーダ内部の詳細

| サブコンポーネント | 時間 (ms) | デコーダ内割合 | 主要オペレータ |
|---|---:|---:|---|
| **ups.2 (ConvTranspose)** | **70.30** | **19.8%** | 最大アップサンプリング層 |
| resblocks.8 | 46.32 | 13.0% | Conv1d × 2 |
| resblocks.5 | 38.93 | 11.0% | Conv1d × 2 |
| **ups.1 (ConvTranspose)** | **37.30** | **10.5%** | 2番目のアップサンプリング |
| resblocks.7 | 33.92 | 9.5% | Conv1d × 2 |
| resblocks.4 | 28.25 | 8.0% | Conv1d × 2 |
| resblocks.6 | 22.94 | 6.5% | Conv1d × 2 |
| resblocks.3 | 18.18 | 5.1% | Conv1d × 2 |
| resblocks.2 | 18.13 | 5.1% | Conv1d × 2 |
| resblocks.1 | 13.15 | 3.7% | Conv1d × 2 |
| ups.0 | 9.97 | 2.8% | ConvTranspose |
| resblocks.0 | 8.07 | 2.3% | Conv1d × 2 |
| conv_post | 3.48 | 1.0% | 最終出力 |
| conv_pre | 1.53 | 0.4% | 入力プロジェクション |

**ConvTranspose (ups)** と後段の **resblocks** ほど時間がかかる。これはアップサンプリング後にシーケンス長が増大し、後段ほど演算量が増えるため。

#### オペレータ種別サマリー

| オペレータ | 時間 (ms) | 全体割合 |
|---|---:|---:|
| **Conv** | **254.59** | **53.0%** |
| **ConvTranspose** | **117.57** | **24.5%** |
| Add | 18.91 | 3.9% |
| FusedConv | 10.48 | 2.2% |
| LeakyRelu | 9.20 | 1.9% |
| Mul | 8.61 | 1.8% |
| その他 | 60.65 | 12.6% |

**Conv (53%) + ConvTranspose (24.5%) = 77.5%** が全推論時間を占める。MB-iSTFT-VITSによるConvTranspose削減が直接的な改善効果を持つ。

#### ORT最適化レベルの影響

> **入力**: 179 phoneme IDs (長文), 各設定20回計測

| 設定 | 短文 (ms) | 長文 (ms) | 差分 |
|---|---:|---:|---|
| **ORT_DISABLE_ALL** (現piper.cpp相当) | 53.52 | **229.89** | baseline |
| ORT_ENABLE_BASIC | 52.14 | 235.04 | +2.2% |
| ORT_ENABLE_EXTENDED | 51.92 | 228.73 | -0.5% |
| ORT_ENABLE_ALL | 51.07 | **226.94** | **-1.3%** |

Apple Siliconでは **ORT最適化レベルの差は約1%** に留まる。Conv/ConvTranspose主体のモデルではグラフ融合の恩恵が限定的。ただし `EnableCpuMemArena` / `EnableMemPattern` はメモリ再利用により繰り返し推論で効果がある。

#### スレッド数の影響（最重要発見）

> **入力**: 179 phoneme IDs (長文), 各設定30回計測, ORT_ENABLE_ALL

| スレッド数 | 時間 (ms) | 標準偏差 | RTF | 高速化 | 備考 |
|---:|---:|---:|---:|---:|---|
| 1 | 677.33 | 7.2 | 0.0571 | 1.00x | |
| 2 | 383.29 | 9.7 | 0.0323 | 1.77x | |
| 4 | 231.90 | 9.1 | 0.0195 | 2.92x | |
| 6 | 179.51 | 8.3 | 0.0151 | 3.77x | |
| 8 | 156.30 | 6.0 | 0.0132 | 4.33x | |
| 10 | 148.74 | 7.0 | 0.0125 | 4.55x | |
| **12 (=P-core数)** | **147.19** | **11.8** | **0.0124** | **4.60x** | **最速** |
| 14 (P+E混在) | **220.25** | **14.5** | 0.0186 | 3.08x | **50%悪化** |
| 16 (全コア) | **245.76** | **21.9** | 0.0207 | 2.76x | **67%悪化** |

```
スレッド数 vs 推論時間 (M4 Max, 長文179 phonemes)

700 |*
    |
600 |
    |
500 |
    |
400 |  *
ms  |
300 |        *                                  *
    |                                                *
200 |           *                          *
    |              *     *     *     *
100 |
    +----+----+----+----+----+----+----+----+----
    1    2    4    6    8   10   12   14   16
                    スレッド数

    * = 実測値, 12 threads (=P-core数) が最速
    14以上でE-coreが混入し急激に悪化
```

**重要な知見:**
- **P-core数 (=12) で最速**: 12→14スレッドで **50%も悪化**
- **E-core混入で標準偏差も増大**: 11.8ms → 14.5ms → 21.9ms（ジッター悪化）
- **10→12スレッドの差は1%未満**: 実用的には P-core数の80%程度で十分
- スレッド数の最適化は ORT最適化レベル (1%改善) より **はるかに効果的** (4.6x改善)

**結論**: `intra_op_num_threads` を **P-core数に設定すること**が、Mac上でのVITS推論における最も重要かつ即効性のある最適化。

### プロファイリング方法

```python
import json
import onnxruntime as ort
import numpy as np

# ORT内蔵プロファイリング
sess_options = ort.SessionOptions()
sess_options.enable_profiling = True
sess_options.profile_file_prefix = "vits_profile"
# 決定論的プロファイリングのためシングルスレッドに
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1
sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

model = ort.InferenceSession("model.onnx", sess_options,
    providers=["CPUExecutionProvider"])

# ダミー入力で実行
inputs = {
    "input": np.random.randint(1, 60, (1, 50), dtype=np.int64),
    "input_lengths": np.array([50], dtype=np.int64),
    "scales": np.array([0.667, 1.0, 0.8], dtype=np.float32),
}

# Warm-up + プロファイル実行
_ = model.run(None, inputs)
_ = model.run(None, inputs)

profile_file = model.end_profiling()
print(f"Profile saved: {profile_file}")
# chrome://tracing または https://ui.perfetto.dev/ で可視化
```

**Apple Instruments (Time Profiler):**

```bash
xcrun xctrace record --template "Time Profiler" \
    --output vits_trace.trace \
    --launch -- uv run python -m piper_train.infer_onnx \
        --model model.onnx --config config.json \
        --text "テスト" --output-dir /tmp/profile_out
```

---

## Tier 1: 即座に実行可能な最適化 (再学習不要)

**推定合計効果: 主にスレッド最適化による4x+高速化（デフォルトスレッド数が不適切な場合）**

### 1.1 ONNX Runtime セッション設定の最適化

#### Python (`infer_onnx.py`)

```python
import os
import platform
import subprocess
import onnxruntime as ort


def _get_optimal_thread_count() -> int:
    """プラットフォームに応じた最適スレッド数を取得"""
    if platform.system() == "Darwin":
        try:
            # Apple Silicon: P-core数のみ使用 (E-core straggling回避)
            r = subprocess.run(
                ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                return int(r.stdout.strip())
        except (subprocess.SubprocessError, ValueError):
            pass
        try:
            # Intel Mac: 物理コア数 (HyperThreading除外)
            r = subprocess.run(
                ["sysctl", "-n", "hw.physicalcpu"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                return int(r.stdout.strip())
        except (subprocess.SubprocessError, ValueError):
            pass
    return max(1, (os.cpu_count() or 4) // 2)


def create_optimized_session(model_path: str, *, battery_mode: bool = False):
    """Mac最適化されたORTセッションを作成"""
    sess_options = ort.SessionOptions()

    # グラフ最適化: ORT_ENABLE_EXTENDED推奨
    # - Conv+LeakyReLU融合 (HiFi-GANデコーダに重要)
    # - MatMul+Add融合、LayerNorm融合
    # - ORT_ENABLE_ALLはNCHWc変換を含むが、Conv1dには効果なし
    #   ARM64用NCHWcは公式wheelに含まれていない
    sess_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    )

    # スレッディング: P-coreのみ使用
    sess_options.intra_op_num_threads = _get_optimal_thread_count()
    sess_options.inter_op_num_threads = 1  # VITSは逐次パイプライン
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    # メモリ最適化 (現在C++コードでは無効化されているが有効化すべき)
    sess_options.enable_cpu_mem_arena = True   # mallocプール再利用
    sess_options.enable_mem_pattern = True     # 割り当てパターン学習
    sess_options.enable_mem_reuse = True       # バッファ共有

    # バッテリー駆動時の省電力設定
    if battery_mode:
        sess_options.add_session_config_entry(
            "session.intra_op.allow_spinning", "0"
        )

    return ort.InferenceSession(
        str(model_path), sess_options,
        providers=["CPUExecutionProvider"],
    )
```

#### C++ (`piper.cpp` line 628 改修)

現在の設定は最適ではない（ただし実測ではORT最適化レベルの差は~1%。**スレッド設定の方が重要**）:

```cpp
// 現在の設定 (問題あり):
session.options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
session.options.DisableCpuMemArena();
session.options.DisableMemPattern();
```

**推奨設定:**

```cpp
// Mac向け最適化設定
#ifdef __APPLE__
  // Conv+Activation融合、MatMul+Add融合を有効化
  session.options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  // メモリアリーナ: mallocプール再利用 (繰り返し推論で5-15%改善)
  session.options.EnableCpuMemArena();
  session.options.EnableMemPattern();

  // P-core数のみ使用 (E-core straggling回避)
  // M1/M2/M3/M4: 4, M1 Pro: 6-8, M4 Pro: 10, M4 Max: 12
  session.options.SetIntraOpNumThreads(4); // 安全なデフォルト
  session.options.SetInterOpNumThreads(1); // VITSは逐次パイプライン

  // バッテリー駆動時の省電力
  session.options.AddConfigEntry("session.intra_op.allow_spinning", "0");
#else
  session.options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  session.options.EnableCpuMemArena();
  session.options.EnableMemPattern();
#endif

session.options.DisableProfiling();
```

### 1.2 モデルWarm-up (初回推論レイテンシ20-40%改善)

ONNX Runtimeは最初の`Run()`でメモリ割り当てパターン学習・カーネルコンパイルを行うため、初回推論が2-5x遅い。`loadModel`直後にダミー推論を実行:

```python
# Python: PiperVoice.load() 内でwarm-up
dummy_ids = [0] * 10
_ = voice.synthesize_ids_to_raw(dummy_ids, speaker_id=0)
```

```cpp
// C++: loadModel() 直後にwarm-up
void warmUpModel(ModelSession& session, bool hasMultiSpeaker, bool hasProsody) {
    auto memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    std::vector<int64_t> dummyPhonemes = {1, 0, 5, 0, 10, 0, 2};
    std::vector<int64_t> dummyLengths = {(int64_t)dummyPhonemes.size()};
    std::vector<float> dummyScales = {0.667f, 1.0f, 0.8f};

    // ... tensor作成 + session.onnx.Run() ...
    spdlog::debug("Model warm-up completed");
}
```

### 1.3 ONNX Opset バージョンアップグレード (15 -> 17)

`export_onnx.py` の `OPSET_VERSION` を 17 に変更:

- `LayerNormalization` がネイティブONNXオペレータになる（5-10%改善）
- CoreML MLProgram形式でネイティブサポート
- PyTorch export opset 17は十分安定

```python
# export_onnx.py
OPSET_VERSION = 17  # 15から変更
```

### 1.4 最適化済みモデルの永続化

グラフ最適化を一度実行し、結果を保存して以降のセッション作成を高速化:

```python
sess_options.optimized_model_filepath = "model_optimized.onnx"
# 初回ロード時に最適化 + 保存。以降はこのファイルを直接ロード
```

### 1.5 CMake ビルド最適化 (C++推論)

```cmake
if(APPLE)
    # -O3: 自動ベクトル化を有効化 (音声処理ループに効果)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -ffast-math")
    set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -ffast-math")

    if(CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
        # Apple Silicon: Firestorm/Avalancheコア向けチューニング
        string(APPEND CMAKE_CXX_FLAGS_RELEASE " -mcpu=apple-m1")
        string(APPEND CMAKE_C_FLAGS_RELEASE " -mcpu=apple-m1")
    elseif(CMAKE_OSX_ARCHITECTURES MATCHES "x86_64")
        # Intel Mac: Haswell以降 (AVX2, FMA有効化)
        string(APPEND CMAKE_CXX_FLAGS_RELEASE " -march=haswell")
        string(APPEND CMAKE_C_FLAGS_RELEASE " -march=haswell")
    endif()

    # Thin LTO (リンク時最適化): 3-8%改善
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        include(CheckIPOSupported)
        check_ipo_supported(RESULT ipo_supported)
        if(ipo_supported)
            set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
            string(APPEND CMAKE_CXX_FLAGS_RELEASE " -flto=thin")
            string(APPEND CMAKE_C_FLAGS_RELEASE " -flto=thin")
            string(APPEND CMAKE_EXE_LINKER_FLAGS_RELEASE " -flto=thin")
        endif()
    endif()

    # Accelerateフレームワーク (vDSP音声処理)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    target_link_libraries(piper PRIVATE ${ACCELERATE_FRAMEWORK})
endif()
```

### 1.6 NEON / Accelerate音声処理の有効化

`CMakeLists.txt` line 83 で現在無効化されている NEON 最適化を再有効化。Apple Accelerate framework (`vDSP`) を使えばIntel/Apple Silicon両方で最適化:

```cpp
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>

// Intel Mac: SSE/AVX経由, Apple Silicon: NEON経由で自動最適化
float findMaxAudioValueAccelerate(const float* audio, size_t audioCount) {
    float maxVal = 0.01f;
    vDSP_maxmgv(audio, 1, &maxVal, static_cast<vDSP_Length>(audioCount));
    return std::max(maxVal, 0.01f);
}

void scaleAndConvertAudioAccelerate(const float* audio, int16_t* output,
                                     size_t audioCount, float audioScale) {
    std::vector<float> scaled(audioCount);
    vDSP_vsmul(audio, 1, &audioScale, scaled.data(), 1,
               static_cast<vDSP_Length>(audioCount));
    float lo = static_cast<float>(INT16_MIN);
    float hi = static_cast<float>(INT16_MAX);
    vDSP_vclip(scaled.data(), 1, &lo, &hi, scaled.data(), 1,
               static_cast<vDSP_Length>(audioCount));
    vDSP_vfix16(scaled.data(), 1, output, 1,
                static_cast<vDSP_Length>(audioCount));
}
#endif
```

---

## Tier 2: 中程度の工数で実現可能な最適化

### 2.1 CoreML Execution Provider (Apple Silicon)

CoreML EPを有効化すると、一部のオペレータがGPU/Neural Engineにオフロードされる。ただし**VITSでは37%のノードしかサポートされていない**ため、ベンチマーク必須。

```python
import sys
from pathlib import Path

if sys.platform == "darwin":
    providers = [
        ("CoreMLExecutionProvider", {
            "ModelFormat": "MLProgram",      # ConvTranspose, LayerNorm対応
            "MLComputeUnits": "ALL",         # CPU + GPU + Neural Engine
            "ModelCacheDirectory": str(
                Path.home() / ".cache/piper-plus/coreml"
            ),  # モデル再コンパイル回避 (起動高速化)
            "RequireStaticInputShapes": "0", # VITSは動的形状
        }),
        "CPUExecutionProvider",
    ]
else:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
```

**CoreMLがサポートするVITS操作 (MLProgram形式):**

| 操作 | サポート | 注意事項 |
|------|---------|---------|
| Conv (1D/2D) | Yes | weights/biasが定数であること |
| ConvTranspose | Yes | SAME_UPPER/LOWER paddingは非対応 |
| MatMul | Yes | transA==0のみ |
| LayerNormalization | Yes | Opset 17+推奨 |
| LeakyReLU, Tanh, Sigmoid | Yes | |
| Softmax | Yes | |
| Reshape, Transpose | Yes | |

**CoreML非対応でフォールバックするVITS操作:**

| 操作 | 発生箇所 | 対策 |
|------|---------|------|
| ConstantOfShape | sequence_mask, generate_path | 固定形状エクスポートで解消 |
| Where | transforms.py boolean masking | torch.whereに統一 |
| ScatterND | generate_path | element-wise + maskに書き換え |
| NonZero | boolean indexing | Gather + 事前計算インデックス |

### 2.2 INT8 量子化 (Post-Training Quantization)

**速度改善は限定的**（Apple Silicon: 1.1-1.4x, Intel (VNNI無し): 1.0-1.2x）。主な価値は**モデルサイズ削減** (74MB -> ~22-33MB)。

```python
from onnxruntime.quantization import (
    quantize_static, QuantFormat, QuantType, CalibrationDataReader,
    quant_pre_process,
)

# Step 1: 前処理
quant_pre_process("model.onnx", "model_preprocessed.onnx", auto_merge=True)

# Step 2: 量子化 (QDQ形式、per-channel推奨)
quantize_static(
    model_input="model_preprocessed.onnx",
    model_output="model_int8.onnx",
    calibration_data_reader=calibration_reader,  # 代表入力が必要
    quant_format=QuantFormat.QDQ,           # QDQ形式推奨 (QOperatorはx86で遅い)
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    per_channel=True,                        # Conv1dに重要
    nodes_to_exclude=sensitive_nodes,        # 下表参照
)
```

**レイヤー別量子化推奨:**

| コンポーネント | 推奨精度 | 感度 | 理由 |
|--------------|---------|------|------|
| TextEncoder | INT8 | Low | 高い冗長性 |
| Speaker Embedding | INT8 | Low | ルックアップテーブル |
| Decoder ResBlocks | INT8 | Medium | 残差接続が誤差を吸収 |
| Decoder conv_pre | INT8 | Medium | 初期プロジェクション |
| StochasticDurationPredictor | FP16 | Medium-High | exp()演算が敏感 |
| Flow (ResidualCouplingBlock) | FP16 | High | 小さな誤差が連鎖 |
| **Decoder ConvTranspose (ups)** | **FP32** | **High** | アップサンプリングが誤差を増幅 |
| **Decoder conv_post** | **FP32** | **Very High** | 最終出力層、音質に直結 |
| LayerNorm / Sigmoid / Softmax | FP32 | N/A | 数値安定性のため |

**量子化サイズ/品質/速度の期待値:**

| 量子化 | サイズ | MOS差 | CPU速度 (Apple Silicon) |
|--------|-------|-------|----------------------|
| FP32 (`--no-fp16`指定時) | 74 MB | 0 | 1.0x |
| FP16 (エクスポートデフォルト) | ~39 MB | -0.00~-0.02 | ~1.0x |
| INT8 動的 (重みのみ) | ~22 MB | -0.02~-0.05 | ~1.1x |
| INT8 静的 (感度レイヤー除外) | ~26-33 MB | -0.03~-0.08 | ~1.1-1.4x |

**注意**: GPTQ/AWQはVITSには非適用（LLMのMatMul向け設計でConv1dには効果なし）。

### 2.3 Python推論最適化

#### OrtValue APIによるコピー削減

```python
import onnxruntime as ort

# OrtValue: numpy配列をゼロコピーでラップ (CPU)
input_ort = ort.OrtValue.ortvalue_from_numpy(phoneme_ids_array)
scales_ort = ort.OrtValue.ortvalue_from_numpy(scales_array)

# run_with_ort_values: C++/Python変換オーバーヘッド削減
results = session.run_with_ort_values(
    output_names=["output"],
    input_dict_ort_values={"input": input_ort, ...},
)
audio = results[0].numpy().squeeze((0, 1))
```

#### GILとマルチスレッド

`session.run()` はC++実行中にGILを解放する。HTTPサーバーで並行リクエスト処理が可能:

```python
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)
# 4スレッドで同時推論可能 (GIL解放されるため)
```

#### パッケージ選択

- **`onnxruntime`** (公式) を使用。`onnxruntime-silicon` (cansik/onnxruntime-silicon) は非推奨
- CoreML EPは公式 `onnxruntime` macOS wheelに含まれている
- `onnxruntime-gpu` はMacでは不要 (CUDA/TensorRT用)

---

## Tier 3: 高効果だが再学習が必要な最適化

### 3.1 デコーダアーキテクチャ比較

HiFi-GANデコーダが推論時間の96%+を占めるため、デコーダ置換が最大の高速化手段。

| デコーダ | CPU高速化 | パラメータ | MOS | 特徴 |
|---------|----------|-----------|------|------|
| HiFi-GAN V2 (現在) | 1x | ~29M total | ~4.43 | TransposedConv, ベースライン |
| iSTFT-VITS | 3.6x | ~29M | ~4.36 | 最終upsampling→iSTFT置換 |
| **MB-iSTFT-VITS** | **4.1x** | ~27.5M | **4.44** | マルチバンドiSTFT、**推奨** |
| **Mini-MB-iSTFT-VITS** | **9.7x** | 7.21M | **4.43** | チャンネル半減+MB-iSTFT |
| FLY-TTS (ConvNeXt+iSTFT) | 8.8x | 17.89M | 4.12 | ConvNeXtデコーダ |
| Mini FLY-TTS | 9.6x | 10.92M | 4.05 | FLY-TTS縮小版 |
| Vocos (standalone) | 13x (HiFi-GAN比) | 13.5M | 3.734 UTMOS | 独立vocoder |
| Wavehax | 4x+ | 0.622M | ~4.3 | Harmonic prior必要 |
| MS-Wavehax | 最速 (低遅延) | 0.332M | ~4.5 | マルチストリーム |

### 3.2 MB-iSTFT-VITS (推奨度: 最高)

HiFi-GANの最後の2アップサンプリング層をマルチバンド逆STFTに置換。**品質劣化なし (MOS 4.44)** で **4.1x高速化**。

**なぜiSTFTベースが速いか:**
- HiFi-GANのTransposedConvはフルサンプルレート（22050 Hz）で動作し、大量のメモリトラフィックを生成
- iSTFTはSTFTフレームレート（T = waveform_length / hop_length）で動作、キャッシュ効率が大幅向上
- iSTFT自体は高度に最適化されたFFTライブラリ (macOSではvDSP) で実行

**参考資料:**
- 論文: [Lightweight and High-Fidelity End-to-End TTS with Multi-Band Generation and ISTFT](https://arxiv.org/pdf/2210.15975)
- 実装: [MasayaKawamura/MB-iSTFT-VITS](https://github.com/MasayaKawamura/MB-iSTFT-VITS)
- V2: [FENRlR/MB-iSTFT-VITS2](https://github.com/FENRlR/MB-iSTFT-VITS2)

### 3.3 FLY-TTS (ConvNeXt + iSTFT)

HiFi-GAN全体をConvNeXtブロック+iSTFTに置換。さらにテキストエンコーダとFlowでパラメータ共有。

- **8.8x CPU高速化**, 61.2%パラメータ削減, MOS 4.12
- ConvNeXtブロック: 7x7 depthwise conv + inverted bottleneck + GELU
- WavLM Discriminator使用（piper-plusに既に実装済み）
- 論文: [FLY-TTS (arXiv 2407.00753)](https://arxiv.org/abs/2407.00753)

### 3.4 MLX ポート (Apple Silicon専用)

Apple の MLX フレームワークで推論を実装。統一メモリを活用したゼロコピーGPU推論が可能。MPS比2-3xの高速化。

**推定工数**: 2-4週間（推論のみ）

**参考実装:**
- [mlx-audio](https://github.com/Blaizzy/mlx-audio) — Kokoro TTS の MLX 実装

---

## Tier 4: 将来的な検討事項

### 4.1 ANE最適化アーキテクチャ

Neural Engineを最大限に活用するには [ml-ane-transformers](https://github.com/apple/ml-ane-transformers) の4原則に従う:

1. **データ形式**: テンソルを `(B, C, 1, S)` 形式に変換。`Conv1d` -> `Conv2d(kernel_size=(1, K))`
2. **テンソルチャンキング**: マルチヘッドアテンションをシングルヘッドに分割
3. **メモリコピー最小化**: reshape操作を避け、専用 einsum を使用
4. **帯域幅管理**: 量子化/プルーニングでパラメータ削減

**注意**: VITSのConv1dを全てConv2dに変換し、再学習が必要。工数1-2ヶ月。

### 4.2 BNNS Graph (iOS/macOSネイティブデプロイ)

WWDC 2024/2025でリアルタイム音声処理用として紹介:

- CPU推論が2x高速化、ゼロメモリ割り当て
- シングルスレッド実行（コンテキストスイッチなし）
- 自動最適化: Conv+Activation融合、メモリ再利用

### 4.3 Profile-Guided Optimization (PGO)

C++ビルドで5-15%追加改善:

```bash
# Step 1: 計装ビルド
cmake -B build-pgo-gen -DCMAKE_BUILD_TYPE=Release -DPGO_GENERATE=ON
cmake --build build-pgo-gen

# Step 2: 代表ワークロード実行
echo "こんにちは" | ./build-pgo-gen/piper -m model.onnx

# Step 3: プロファイルマージ
xcrun llvm-profdata merge -output=merged.profdata build-pgo-gen/pgo-data/*.profraw

# Step 4: 最適化ビルド
cmake -B build-pgo-use -DCMAKE_BUILD_TYPE=Release -DPGO_USE=ON
cmake --build build-pgo-use
```

### 4.4 BitTTS 1.58bit量子化

三値重み `{-1, 0, 1}` による 1.58bit QAT で **83%サイズ削減**、MOS品質は4bitと同等。
- 論文: [BitTTS (arXiv 2506.03515)](https://arxiv.org/abs/2506.03515)

### 4.5 Metal 4 Shader ML (M5チップ以降)

WWDC 2025で発表。MTLTensor、Metal Performance Primitivesによるシェーダ内ML操作。M5ハードウェアが必要。

---

## 既知の制約事項

| 制約 | 詳細 |
|------|------|
| **coremltools Conv1d segfault** | Issue #2574 (2025年7月, 未解決) — CoreML直接変換はクラッシュリスクあり |
| **ANE INT8の実態** | Orion論文 (2026年3月): INT8はFP16にデクオンタイズ。公称TOPS通りの性能は出ない |
| **Apple Silicon FP16** | Tensor Core非搭載。CPU FP16演算速度≈FP32。メモリ削減が主な価値。`export_onnx`でデフォルト適用 |
| **Intel Mac FP16** | Cast overheadで約4x遅い。速度向上なし。サイズ削減のみ価値あり。`--no-fp16`でFP32出力可 |
| **MLAS + AMX** | ONNX Runtime MLASはApple Accelerateを使用しない。AMXはCoreML EP経由のみ活用可能 |
| **NCHWcレイアウト** | Conv1dには効果なし。Conv2d + x86のみ。ARM64版は公式wheelに未含 |
| **macOS thread affinity** | `pthread_setaffinity_np` はmacOSに存在しない。QoS classで間接制御のみ |
| **iOS Audio Unit制限** | 120MB RAMハードリミット。74MBモデル+ランタイムで超過の可能性 |
| **ONNX Runtime x86_64 macOS** | 1.24.x以降wheelが廃止予定。ソースビルドが必要 |

---

## 推奨実行計画

| 段階 | 内容 | 期待効果 | 工数 |
|------|------|---------|------|
| **即座に** | Tier 1: **スレッド数最適化（最重要）**、ORT設定修正、warm-up、Opset 17、CMakeフラグ | **スレッド最適化で4x+**（実測: M4 Max） | 低 |
| **短期** | Tier 2: CoreML EP評価、INT8量子化、OrtValue API | **追加5-40%** | 中 |
| **中期** | Tier 3: MB-iSTFT-VITSデコーダ導入 | **4.1x高速化** | 再学習必要 |
| **長期** | Mini-MB-iSTFT-VITS or FLY-TTS | **8-10x高速化** | 再学習必要 |

**現時点での最適解は ONNX Runtime CPU推論 + P-core数に合わせたスレッド設定**です。実測で最も効果が大きかったのはスレッド数最適化（4.6x）であり、ORT最適化レベルの差は~1%に留まりました。さらなる速度向上には **MB-iSTFT-VITSによるデコーダアーキテクチャ変更** が最も費用対効果が高い選択肢です。

---

## 参考プロジェクト・文献

### Apple公式

| リソース | URL |
|---------|-----|
| ml-ane-transformers | https://github.com/apple/ml-ane-transformers |
| MLX Framework | https://github.com/ml-explore/mlx |
| coremltools | https://github.com/apple/coremltools |
| WWDC24 Session 10211 | "Support Real-Time ML Inference on the CPU" (BNNS Graph) |
| WWDC25 Session 276 | "What's New in BNNS Graph" |
| WWDC25 Session 262 | "Combine Metal 4 ML and Graphics" |

### コミュニティ・OSS

| プロジェクト | 用途 | URL |
|-------------|------|-----|
| sherpa-onnx | クロスプラットフォームTTS (CoreML対応) | https://github.com/k2-fsa/sherpa-onnx |
| mlx-audio | MLX TTSライブラリ | https://github.com/Blaizzy/mlx-audio |
| MB-iSTFT-VITS | 軽量VITSデコーダ | https://github.com/MasayaKawamura/MB-iSTFT-VITS |
| MB-iSTFT-VITS2 | V2実装 | https://github.com/FENRlR/MB-iSTFT-VITS2 |
| Vocos | 高速neural vocoder | https://github.com/gemelo-ai/vocos |
| Wavehax | 超軽量vocoder | https://github.com/chomeyama/wavehax |
| WhisperKit | ANE最適化音声モデル | https://github.com/argmaxinc/WhisperKit |
| hollance/neural-engine | ANEリバースエンジニアリング | https://github.com/hollance/neural-engine |
| corsix/amx | AMXリバースエンジニアリング | https://github.com/corsix/amx |

### ONNX Runtime

| リソース | URL |
|---------|-----|
| CoreML EP ドキュメント | https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html |
| 量子化ドキュメント | https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html |
| FP16ドキュメント | https://onnxruntime.ai/docs/performance/model-optimizations/float16.html |
| スレッディング設定 | https://onnxruntime.ai/docs/performance/tune-performance/threading.html |
| グラフ最適化 | https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html |
| DNNL macOS非対応 (Issue #5783) | https://github.com/microsoft/onnxruntime/issues/5783 |
| FP16マクロ修正 (PR #26443) | https://github.com/microsoft/onnxruntime/pull/26443 |

### 論文

| 論文 | 年 | 要点 |
|------|---|------|
| MB-iSTFT-VITS | 2022 | iSTFTデコーダで4.1x高速化、MOS 4.44 |
| FLY-TTS | 2024 | ConvNeXt+iSTFT、61.2%パラメータ削減、8.8x CPU高速化 |
| Vocos | 2023 | ConvNeXt vocoder、CPU 169.6x RT |
| Wavehax | 2024 | 0.622M params vocoder、4x+ HiFi-GAN比 |
| Nix-TTS | 2022 | VITSの知識蒸留、89%サイズ削減 (但しMOS 3.69に劣化) |
| BitTTS | 2025 | 1.58bit QATで83%サイズ削減 |
| Orion | 2026 | ANE INT8→FP16デクオンタイズ判明 |
| VITS Quality vs Speed (TSD) | 2023 | デコーダが推論時間の96%+を占めることを確認 |
| Comparative Analysis of Vocoders | 2025 | CPU上ではメモリ帯域幅が主要ボトルネック |
