# piper-plus ボコーダー高速化 総合調査レポート

> **調査日**: 2026-03-13
> **対象**: ONNX Runtime で動作する PC/スマホ/Unity 対応ボコーダー
> **調査範囲**: 30+ ボコーダー、最新論文 (2023-2025)、OSS実装、本番デプロイ事例
> **関連リポジトリ**: [mobile-vocoder](https://github.com/ayutaz/mobile-vocoder)

---

## 目次

1. [現状分析](#1-現状分析)
2. [重要な発見: Vocos ONNX膨張問題](#2-重要な発見-vocos-onnx膨張問題)
3. [ボコーダー候補の総合比較](#3-ボコーダー候補の総合比較)
4. [統合アプローチの比較](#4-統合アプローチの比較)
5. [プラットフォーム対応マトリクス](#5-プラットフォーム対応マトリクス)
6. [推奨戦略](#6-推奨戦略)
7. [mobile-vocoder リポジトリとの連携](#7-mobile-vocoder-リポジトリとの連携)
8. [結論](#8-結論)
9. [参考文献](#9-参考文献)

---

## 1. 現状分析

### piper-plus の現 HiFi-GAN デコーダー

| 項目 | 値 |
|------|-----|
| パラメータ数 | ~1.3-1.5M (モデル全体の 8-10%) |
| ONNX サイズ | ~6.4MB (全体 ~74MB 中) |
| ONNX MACs | 2.864B |
| アーキテクチャ | Conv1d(7) → ConvTranspose1d×3 (8,8,4) → ResBlock2×9 → Conv1d(7) → tanh |
| 入力 | `[batch, 192, time]` float32 (VITS 潜在空間) |
| 出力 | `[batch, 1, audio_samples]` float32 [-1,1] |
| ボトルネック | **全推論時間の 74%** (M4 Max 実測、Conv 53% + ConvTranspose 24.5%) |
| サンプルレート | 22,050 Hz |
| アップサンプリング倍率 | 256x (= 8 × 8 × 4) |

### 制約条件

- **VITS デコーダーは再学習なしで置換不可**: 潜在空間がエンコーダ・デコーダ間で共同学習されるため
- **ONNX に iSTFT オペレータなし**: DFT 行列方式 (mobile-vocoder で実証済み) or C++ 実装で回避必要
- **Unity Sentis**: ConvTranspose 性能問題あり ([Unity Discussions](https://discussions.unity.com/t/inference-running-slow-refer-to-convtranspose-and-upsample2d-ops-vs-onnxruntime/311194))、ONNX Runtime C# 推奨
- **INT8 量子化は VITS に有害**: 1.5-300 倍遅くなる報告多数 ([sherpa-onnx #575](https://github.com/k2-fsa/sherpa-onnx/issues/575), [Coqui TTS #2991](https://github.com/coqui-ai/TTS/discussions/2991))
- **NNAPI**: 1D Conv 非サポート (CPU フォールバック)

### 参照ファイル

| ファイル | 内容 |
|---------|------|
| `src/python/piper_train/vits/models.py` L301-380 | Generator (HiFi-GAN デコーダー) |
| `src/python/piper_train/vits/config.py` | quality presets (low/medium/high) |
| `src/python/piper_train/vits/modules.py` | ResBlock1, ResBlock2 |
| `src/python/piper_train/export_onnx.py` | ONNX エクスポート (opset 15) |
| `src/cpp/piper.cpp` L813-846 | C++ 推論 + 音声スケーリング |

---

## 2. 重要な発見: Vocos ONNX 膨張問題

mobile-vocoder リポジトリの実測で判明した**最も重要な発見**:

| ボコーダー | PyTorch MACs | ONNX MACs | 膨張率 |
|-----------|-------------|-----------|--------|
| **MS-Wavehax** | 0.986B | 1.056B | **1.07x** |
| MS-Vocos | 1.356B | 4.215B | 3.1x |
| Wavehax | — | 13.0B | — |
| **Vocos (vanilla)** | 1.348B | **46.96B** | **34.8x** |

### 原因

iSTFT 操作が ONNX グラフ内で DFT 行列演算に展開され、計算量が爆発する。ONNX には `iSTFT` オペレータが存在しないため ([onnx/onnx#4777](https://github.com/onnx/onnx/issues/4777))、`torch.istft` が低レベル演算 (MatMul, Slice 等) にアンロールされる。

### 解決策

1. **マルチストリーム分解** (MS-Wavehax): 膨張を 1.07x に抑制
2. **DFT 行列方式** (`OnnxSTFT`): `aten::fft_rfft` を DFT basis matrix の MatMul で置換 (mobile-vocoder `vocoder/models/stft_onnx.py` で実装済み)
3. **C++ 側で iSTFT 実行**: ONNX モデルは magnitude+phase を出力し、ホスト側で iSTFT (FFT ライブラリ利用)

> **教訓**: PyTorch ベンチマークだけでボコーダーを選ぶのは危険。**ONNX 実測が必須**。

---

## 3. ボコーダー候補の総合比較

### Tier S: 最有力候補

#### MS-Wavehax (Yoneyama et al., Interspeech 2025)

マルチストリーム分解 + 2D CNN + ハーモニック事前知識。

| 項目 | 値 |
|------|-----|
| パラメータ | **0.298M** (HiFi-GAN V1 の 2.4%) |
| ONNX MACs | **1.056B** |
| ONNX サイズ | **1.39 MB** (FP32) |
| CPU RTF | **16.7x** |
| ONNX 膨張率 | **1.07x** (最小) |
| F0 入力 | **必須** (最大ブロッカー) |
| 品質 | UTMOS 3.30 (JVS データ天井 3.52) |
| ライセンス | MIT ヘッダー (LICENSE ファイル未確認) |
| 論文 | [arXiv:2506.03554](https://arxiv.org/abs/2506.03554) |

piper-plus HiFi-GAN との比較:

| 指標 | HiFi-GAN | MS-Wavehax | 改善 |
|------|----------|-----------|------|
| パラメータ | 1.665M | 0.511M | **3.3x 削減** |
| ONNX サイズ | 6.36 MB | 2.22 MB | **2.9x 削減** |
| ONNX MACs | 2.864B | 1.139B | **2.5x 削減** |
| ONNX RTF | 26.9x | 24.4x | 同等 |

#### MB-iSTFT-VITS (Kawamura et al., ICASSP 2023)

VITS デコーダーの最終アップサンプリング層を iSTFT + マルチバンド生成で置換。

| 項目 | 値 |
|------|-----|
| 高速化 | **4.1x** vs VITS (CPU RTF 0.066) |
| 品質 | MOS 4.08 (人間レベル、統計的有意差なし) |
| F0 入力 | 不要 (End-to-End 学習維持) |
| ONNX 出力 | 1 ファイル (デプロイ変更なし) |
| 論文 | [arXiv:2210.15975](https://arxiv.org/abs/2210.15975) |
| コード | [GitHub](https://github.com/MasayaKawamura/MB-iSTFT-VITS) |

3 バリアント:

| バリアント | RTF | 高速化 vs VITS |
|-----------|-----|---------------|
| iSTFT-VITS | ~0.09 | ~3.0x |
| MS-iSTFT-VITS | ~0.08 | ~3.4x |
| **MB-iSTFT-VITS** | **~0.066** | **~4.1x** |

#### FLY-TTS (Guo et al., Interspeech 2024)

ConvNeXt ブロック + iSTFT ヘッド + WavLM 敵対的学習。

| 項目 | 値 |
|------|-----|
| 高速化 | **8.8x** vs VITS ベースライン (RTF 0.0139) |
| パラメータ圧縮 | 1.6x |
| 品質 | VITS 同等 |
| F0 入力 | 不要 |
| 論文 | [arXiv:2407.00753](https://arxiv.org/abs/2407.00753) |

### Tier A: 有望候補

| ボコーダー | パラメータ | CPU RTF | ONNX 膨張 | F0 | 品質 | 論文 |
|-----------|----------|---------|----------|-----|------|------|
| **Vocos** | 13.5M | 169.6x (PyTorch) | **34.8x** | No | PESQ 3.70 | [arXiv:2306.00814](https://arxiv.org/abs/2306.00814) |
| **MS-Vocos** | — | — | 3.1x | No | Vocos 同等 | — |
| **HiFTNet** | BigVGAN/6 | 4x vs BigVGAN | — | Yes | GT 級 | [arXiv:2309.09493](https://arxiv.org/abs/2309.09493) |
| **FARGAN** | 0.82M | 600M FLOPS | — | Yes | ~HiFi-GAN | [arXiv:2405.21069](https://arxiv.org/abs/2405.21069) |
| **iSTFTNet** | 13.3M | 14.4x | — | No | HiFi-GAN 同等 | [arXiv:2203.02395](https://arxiv.org/abs/2203.02395) |
| **APNet2** | — | ~8x | — | No | Vocos 超 | [arXiv:2311.11545](https://arxiv.org/abs/2311.11545) |

### Tier B: 条件付き候補

| ボコーダー | 特徴 | 制限 |
|-----------|------|------|
| HiFi-GAN V2 | 0.92M, MOS 4.23, 最低摩擦 | 速度改善は限定的 |
| HiFi-GAN V3 | ~1.46M, 13.44x RT, MOS 4.05 | 品質低下 |
| MB-MelGAN | 1.91M, RTF 0.03, MOS 4.34 | VITS 統合ではなく外部ボコーダー |
| DDSP Vocoder | **15 MFLOPS**, MOS 4.36 | アーキテクチャ変更大 |
| NSF | 非自己回帰, 10-50x RT | F0 必須 |
| RNDVoC | F0 不要、3.572M | MS-Wavehax 比 12x 大 |
| FARGAN | **600 MFLOPS**, Opus 統合済み | C 実装のみ、ONNX 非対応 |

### Tier C: 不推奨

| ボコーダー | 理由 |
|-----------|------|
| BigVGAN v2 | 112M params, CPU <1x RT (サーバー専用) |
| EVA-GAN | 200M params, 非公開 |
| WaveRNN | 自己回帰ボトルネック |
| LPCNet | FARGAN に後継、非推奨 |
| SqueezeWave | HiFi-GAN より大きく低品質 |
| FFTNet | 全指標で旧世代 |

---

## 4. 統合アプローチの比較

### アプローチ A: VITS 内部デコーダー置換 (推奨)

**MB-iSTFT-VITS / FLY-TTS 方式**

```
TextEncoder → Flow → [新デコーダー] → 波形
                      ↑ 潜在空間入力 (192ch)
```

| 項目 | MB-iSTFT-VITS | FLY-TTS |
|------|--------------|---------|
| 高速化 | 4.1x vs VITS | 8.8x vs VITS |
| 品質 | 人間レベル維持 (MOS 4.08) | VITS 同等 |
| 変更範囲 | デコーダーのみ | デコーダー + エンコーダ圧縮 |
| 再学習 | **必須** | **必須** |
| F0 不要 | Yes | Yes |
| ONNX 互換 | DFT 行列方式で解決可 | DFT 行列方式で解決可 |
| 実績 | ICASSP 2023, 複数 fork | Interspeech 2024 |

**利点**:
- F0 不要 (VITS の End-to-End 学習を維持)
- 1 つの ONNX ファイル (sherpa-onnx 互換性維持)
- iSTFT の ONNX 問題は `OnnxSTFT` で解決済み

**欠点**:
- 再学習必須 (20 話者 v2 データセットで ~90 時間)
- iSTFT 部分の ONNX 互換性テストが必要

### アプローチ B: 外部ボコーダー (mobile-vocoder 方式)

**MS-Wavehax 方式**

```
VITS → latent 抽出 → [MS-Wavehax] → 波形
                      ↑ F0 入力も必要
```

| 項目 | 値 |
|------|-----|
| パラメータ | 0.298M → 0.511M (piper-plus 用 192ch) |
| ONNX サイズ | 1.39 MB → 2.22 MB |
| ONNX MACs | 1.056B → 1.139B |
| ONNX 膨張率 | 1.07x (最小) |
| F0 | **必須** (最大ブロッカー) |

**利点**:
- 極小モデル (1.39-2.22 MB)
- ONNX 効率最良 (膨張 1.07x)
- ボコーダー独立学習可能

**欠点**:
- F0 供給問題 (推論時に F0 推定モジュールが必要)
- 2 段階パイプライン (2 つの ONNX モデル)
- 品質天井 (UTMOS 3.30, JVS データ制限)

**F0 問題の解決計画** (mobile-vocoder ロードマップ):
1. Analysis-synthesis テスト (再学習なし)
2. `preprocess_piper.py` で VITS latent + GT F0 抽出 → MS-Wavehax 学習
3. 推論時 F0 推定モジュール追加

### アプローチ C: HiFi-GAN 設定最適化 (最低摩擦)

現コードの `config.py` 設定変更:

| 設定 | パラメータ | MOS | 変更内容 |
|------|----------|-----|---------|
| 現状 (medium) | ~1.5M | — | — |
| V2 相当 | 0.92M | 4.23 | チャンネル削減 + 再学習 |
| V3 相当 | ~1.46M | 4.05 | 軽量 ResBlock + 再学習 |
| + Depthwise Sep Conv | -30-40% | +0.13 | ResBlock 改修 + 再学習 |

**利点**: 最低リスク、コード変更最小
**欠点**: 高速化は ~2x 程度に限定

---

## 5. プラットフォーム対応マトリクス

### 推論エンジン対応

| プラットフォーム | ONNX Runtime EP | Unity Sentis | CoreML | 備考 |
|---------------|----------------|-------------|--------|------|
| **Windows** | CPU / DirectML | GPUCompute | — | DirectML で GPU 活用可 |
| **macOS** | CPU / CoreML | GPUCompute | ANE 対応 | ANE で最速推論 |
| **iOS** | CPU / CoreML / XNNPACK | GPUCompute | **ANE 最速** | Kokoro: ~45ms/生成 |
| **Android** | CPU / NNAPI / XNNPACK | GPUCompute | — | NNAPI は 2D Conv のみ |
| **Linux** | CPU / CUDA | — | — | サーバー向け |
| **WebGL/Wasm** | WebAssembly | — | — | sherpa-onnx 対応 |

### Unity 統合の選択肢

| プロジェクト | エンジン | プラットフォーム | 備考 |
|------------|---------|-------------|------|
| [uPiper](https://github.com/ayutaz/uPiper) | Unity Inference Engine | Win/Mac/Linux/Android/iOS | 最広プラットフォーム |
| [piper.unity](https://github.com/Macoron/piper.unity) | Sentis | Windows x86-64 | 元祖ポート |
| [piper-unity](https://github.com/skykim/piper-unity) | Sentis | Win/Mac/Android | 20-30ms 推論 |

**推奨**: ConvTranspose ヘビーなボコーダーでは ONNX Runtime C# バインディングが Sentis より高速。
iSTFT ベースに置換すれば ConvTranspose 依存が減り、Sentis でも性能改善が期待できる。

### Sherpa-ONNX 対応

| モデル | アーキテクチャ | ボコーダー | 備考 |
|--------|-------------|----------|------|
| VITS (Piper) | End-to-End | HiFi-GAN 統合 | **現在対応済み** |
| Matcha-TTS | 2 段階 | Vocos / HiFi-GAN (別 ONNX) | 対応済み |
| Kokoro | StyleTTS2 + iSTFTNet | 統合 | 対応済み |
| Kitten TTS | 軽量 Transformer | 統合 | 15M params, 25MB INT8 |

---

## 6. 推奨戦略

### Phase 1: 即時改善 (再学習なし)

| 施策 | 効果 | 工数 |
|------|------|------|
| FP16 変換 (`export_onnx` デフォルト適用、`--no-fp16` で無効化) | モデルサイズ ~50% 削減 | 完了 |
| スレッド数最適化 (P-core 数に設定) | **4x+ 高速化** (M4 Max 実測) | 設定変更のみ |
| `ORT_ENABLE_ALL` 最適化レベル | ~1% 改善 (Conv ヘビーモデルでは微小) | 設定変更のみ |

### Phase 2: デコーダー置換 (中期目標、推奨)

**MB-iSTFT-VITS 方式**を採用:

1. `models.py` の `Generator` クラスを MB-iSTFT-VITS デコーダーに置換
2. 最終アップサンプリング層 (ConvTranspose1d ×1-2) を iSTFT に変更
3. マルチバンド生成 (PQMF) で並列化
4. `OnnxSTFT` (DFT 行列方式) で ONNX 互換性を確保
5. 20 話者 v2 データセットで再学習 (~90 時間)

**期待効果**:
- 推論速度: **4-9x 高速化** (MB-iSTFT-VITS 4.1x ～ FLY-TTS 8.8x)
- モデルサイズ: デコーダー部分 20-40% 削減
- 品質: 維持 (MOS 4.08, 統計的有意差なし)
- デプロイ: 1 ONNX ファイル維持 (sherpa-onnx 互換)

### Phase 3: MS-Wavehax 統合 (長期)

mobile-vocoder リポジトリと連携:

1. `preprocess_piper.py` で VITS latent + GT F0 を抽出
2. `ms_wavehax_piper_plus_22k.yaml` で 192ch/22050Hz ボコーダー学習
3. 推論時 F0 推定モジュールを追加
4. 2 段階パイプライン (VITS acoustic + MS-Wavehax vocoder) として ONNX 出力

### Phase 4: カスタムボコーダー (研究)

- **HarmoStream** (mobile-vocoder Phase 3 設計): <2M params, <3B MACs, UTMOS >= 3.7
- FTConv + FASA + HPE の 3 ピラー設計
- Phase 2 の結果に基づき Go/No-Go 判定

---

## 7. mobile-vocoder リポジトリとの連携

### 既存成果物

| 成果物 | ファイル | 状態 |
|--------|---------|------|
| VITS latent + GT F0 抽出 | `preprocess_piper.py` | 実装済み |
| piper-plus 用 MS-Wavehax 設定 | `ms_wavehax_piper_plus_22k.yaml` | 準備済み |
| Piper HiFi-GAN ラッパー | `piper_hifigan_wrapper.py` | 実装済み |
| iSTFTNet ラッパー | `istftnet_wrapper.py` | 実装済み |
| ONNX iSTFT 互換モジュール | `vocoder/models/stft_onnx.py` | 実装済み |
| ボコーダー比較スクリプト | `compare_piper_generators.py` | 実装済み |
| 25+ ボコーダー比較表 | `docs/vocoder_comparison.md` | 完了 |
| HarmoStream Phase 3 設計 | `feat/harmostream-design` ブランチ | 設計のみ |

### Phase 0 調査結果 (MS-Wavehax vs RNDVoC)

| 指標 | MS-Wavehax | RNDVoC | 比率 |
|------|-----------|--------|------|
| パラメータ | **0.298M** | 3.572M | 12x 小 |
| PyTorch MACs | **0.986B** | 7.884B | 8x 少 |
| CPU RTF | **25.6x** | 8.7x | 3x 速 |
| F0 入力 | 必須 | 不要 | トレードオフ |

### Phase 2 最終スコア

| 指標 | 目標 | 結果 | 判定 |
|------|------|------|------|
| ONNX MACs | < 3B | **1.056B** | **PASS** |
| ONNX サイズ | — | **1.39 MB** | 極めて軽量 |
| ONNX CPU RTF | >= 1x | **16.7x** | **PASS** |
| UTMOS | >= 3.5 | 3.297 | MISS (データ天井) |
| パラメータ | — | **0.298M** | iSTFTNet の 1/46 |

### piper-plus 統合ブロッカー

**F0 問題**: MS-Wavehax は F0 入力を必要とするが、VITS は推論時に F0 を生成しない。

**解決計画**:
1. Analysis-synthesis テスト (生成音声から mel + F0 抽出 → 既存 MS-Wavehax に入力)
2. 192ch VITS latent + GT F0 で MS-Wavehax を再学習
3. 軽量 F0 推定モジュールを ONNX グラフに統合

**フォールバック**: RNDVoC (F0 不要、MIT ライセンス、ただし 12x 大きい)

---

## 8. 結論

### 推奨優先順位

| 順位 | アプローチ | 高速化 | 工数 | リスク | 備考 |
|------|----------|--------|------|--------|------|
| **1** | MB-iSTFT-VITS デコーダー置換 | **4-9x** | 中 (再学習) | 低 | F0 不要、1 ONNX 維持 |
| **2** | MS-Wavehax 外部ボコーダー | **2.5x MACs** | 中 (F0 問題) | 中 | 最小モデル、ONNX 効率最良 |
| **3** | HiFi-GAN V2 + DSC | **~2x** | 低 (再学習) | 低 | 最低摩擦 |
| **4** | FP16 (エクスポート時デフォルト) + スレッド最適化 | **~1.5x** | 極低 | 極低 | 即時実行可 |

### 最終推奨

**MB-iSTFT-VITS 方式**を中期目標として採用する。

理由:
1. **F0 不要** — VITS の End-to-End 学習を維持でき、推論パイプラインに F0 推定が不要
2. **実績あり** — ICASSP 2023 で発表、MOS 4.08 で品質劣化なし
3. **4.1x 高速化** — デコーダーが推論時間の 74% を占めるため、全体で ~3x 高速化
4. **1 ONNX 維持** — sherpa-onnx, uPiper 等の既存デプロイインフラと互換
5. **iSTFT ONNX 問題は解決済み** — mobile-vocoder の `OnnxSTFT` を流用可能
6. **ConvTranspose 削減** — Unity Sentis の性能問題も緩和

短期は FP16 (エクスポート時にデフォルト適用済み) + スレッド最適化で即効性のある改善を得つつ、並行して MB-iSTFT-VITS の実装・学習を進める。MS-Wavehax は F0 問題解決後に外部ボコーダーとして統合し、最軽量デプロイオプションとする。

---

## 9. 参考文献

### iSTFT 系ボコーダー

| 論文 | 会議 | リンク |
|------|------|--------|
| iSTFTNet (Kaneko et al.) | ICASSP 2022 | [arXiv:2203.02395](https://arxiv.org/abs/2203.02395) |
| iSTFTNet2 (Kaneko et al.) | Interspeech 2023 | [arXiv:2308.07117](https://arxiv.org/abs/2308.07117) |
| MB-iSTFT-VITS (Kawamura et al.) | ICASSP 2023 | [arXiv:2210.15975](https://arxiv.org/abs/2210.15975) |
| FLY-TTS (Guo et al.) | Interspeech 2024 | [arXiv:2407.00753](https://arxiv.org/abs/2407.00753) |
| HiFTNet (Li et al.) | 2023 | [arXiv:2309.09493](https://arxiv.org/abs/2309.09493) |
| Vocos (Siuzdak) | ICLR 2024 | [arXiv:2306.00814](https://arxiv.org/abs/2306.00814) |
| Wavehax (Yoneyama et al.) | 2024 | [arXiv:2411.06807](https://arxiv.org/abs/2411.06807) |
| MS-Wavehax (Yoneyama et al.) | Interspeech 2025 | [arXiv:2506.03554](https://arxiv.org/abs/2506.03554) |
| APNet2 (Yang Ai et al.) | 2023 | [arXiv:2311.11545](https://arxiv.org/abs/2311.11545) |

### GAN 系ボコーダー

| 論文 | 会議 | リンク |
|------|------|--------|
| HiFi-GAN (Kong et al.) | NeurIPS 2020 | [arXiv:2010.05646](https://arxiv.org/abs/2010.05646) |
| BigVGAN v2 (NVIDIA) | ICLR 2023 + 2024 update | [arXiv:2206.04658](https://arxiv.org/abs/2206.04658) |
| MB-MelGAN | IEEE SLT 2021 | [arXiv:2005.05106](https://arxiv.org/abs/2005.05106) |
| MISRNet (Kaneko et al.) | Interspeech 2022 | [Project](https://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/misrnet/) |

### DSP ハイブリッド

| 論文 | リンク |
|------|--------|
| LPCNet (Valin et al.) | [GitHub](https://github.com/xiph/LPCNet) |
| FARGAN (Valin et al., 2024) | [arXiv:2405.21069](https://arxiv.org/abs/2405.21069) |
| Ultra-lightweight DDSP (ICASSP 2024) | [arXiv:2401.10460](https://arxiv.org/abs/2401.10460) |
| NSF (Wang et al.) | [arXiv:1904.12088](https://arxiv.org/abs/1904.12088) |
| WORLD Vocoder | [GitHub](https://github.com/mmorise/World) |

### End-to-End TTS

| システム | ボコーダー選択 | 備考 |
|---------|-------------|------|
| VITS / VITS2 | HiFi-GAN (統合) | デコーダー変更なし |
| Matcha-TTS | HiFi-GAN (外部) | [GitHub](https://github.com/shivammehta25/Matcha-TTS) |
| F5-TTS | **Vocos** (デフォルト) | [GitHub](https://github.com/SWivid/F5-TTS) |
| Kokoro | **iSTFTNet** | 82M params, 210x RT |
| CosyVoice | HiFi-GAN (軽量) | [arXiv:2407.05407](https://arxiv.org/abs/2407.05407) |
| GPT-SoVITS v4 | 改良 HiFi-GAN | BigVGAN → HiFi-GAN に戻した |
| Fish Speech | Firefly-GAN | コーデック統合 |

### デプロイ・フレームワーク

| リソース | リンク |
|---------|--------|
| Sherpa-ONNX | [GitHub](https://github.com/k2-fsa/sherpa-onnx) |
| ONNX Runtime Mobile | [Docs](https://onnxruntime.ai/docs/tutorials/mobile/) |
| Unity Sentis | [Docs](https://docs.unity3d.com/Packages/com.unity.sentis@2.1/manual/) |
| uPiper | [GitHub](https://github.com/ayutaz/uPiper) |
| piper.unity | [GitHub](https://github.com/Macoron/piper.unity) |
| istft-onnx workaround | [GitHub](https://github.com/mush42/istft-onnx) |
| ONNX iSTFT proposal | [onnx/onnx#4777](https://github.com/onnx/onnx/issues/4777) |

### 量子化・最適化

| トピック | リンク |
|---------|--------|
| ONNX Runtime 量子化 | [Docs](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) |
| BitTTS (Interspeech 2025) | [arXiv:2506.03515](https://arxiv.org/abs/2506.03515) |
| INT8 TTS の問題 | [sherpa-onnx #575](https://github.com/k2-fsa/sherpa-onnx/issues/575) |
| Comparative Analysis (Interspeech 2025) | [Yoneyama et al.](https://www.isca-archive.org/interspeech_2025/yoneyama25_interspeech.pdf) |
