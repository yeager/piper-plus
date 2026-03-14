# Piper-Plus Roadmap

> 調査日: 2026-03-12〜13 | ブランチ: `dev` | 現バージョン: v1.6.0
>
> 16エージェント (基盤調査) + 15エージェント (拡張機能調査) による並列調査に基づく、piper-plusの今後のロードマップです。

---

## 目次

1. [現状の強みと位置づけ](#1-現状の強みと位置づけ)
2. [ロードマップ概要](#2-ロードマップ概要)
3. [優先アクション一覧](#3-優先アクション一覧)
4. [Phase 1: 基盤強化 (v1.7 - v1.8)](#4-phase-1-基盤強化-v17---v18)
5. [Phase 2: マルチ言語対応 (v2.0)](#5-phase-2-マルチ言語対応-v20)
6. [Phase 3: ゼロショットTTS (v2.5)](#6-phase-3-ゼロショットtts-v25)
7. [Phase 4: 次世代アーキテクチャ (v3.0)](#7-phase-4-次世代アーキテクチャ-v30)
8. [VITSベースOSS比較と移植候補](#8-vitsベースoss比較と移植候補)
9. [オリジナルpiperとの差分](#9-オリジナルpiperとの差分)
10. [データセット戦略](#10-データセット戦略)
11. [調査ソース・拡張機能調査](#11-調査ソース拡張機能調査)

---

## 1. 現状の強みと位置づけ

### オリジナルpiperの状況

- **rhasspy/piper は2025年10月にアーカイブ済み**（読み取り専用）
- 後継は `OHF-Voice/piper1-gpl`（GPL-3.0ライセンス）
- 48言語/130+ボイスの事前学習済みモデルが資産

### piper-plusの差別化ポイント

| 領域 | piper-plusの優位性 |
|------|-------------------|
| 日本語品質 | OpenJTalk統合、韻律情報(A1/A2/A3)、疑問詞マーカー、Nバリアント |
| 学習品質 | WavLM Discriminator (MOS +0.15-0.25)、EMA、FP16 |
| GPL-free英語 | g2p-en (Apache-2.0) によるespeak-ng依存回避 |
| 拡張性 | Phonemizer ABC + 言語レジストリによるプラグイン構造 |
| デプロイ多様性 | WebAssembly、Docker 5イメージ、PyPI、Unity、Gradio WebUI |
| アクティブ開発 | オリジナルpiperはアーカイブ済みだが継続開発中 |

### 現在進行中のPR

| PR | 内容 | 状態 | 関連Phase |
|----|------|------|----------|
| #222 | Zero-Shot TTS (Dual-Mode Speaker Conditioning) | Draft | Phase 3 |
| #219 | 6言語マルチリンガル音素化 (JA/EN/ZH/ES/FR/PT) | Draft | Phase 2 |
| #218 | バイリンガル (JA+EN) TTS v3学習完了 | Open | Phase 2 |

### 未着手のオープンIssue

> 最終更新: 2026-03-13

| Issue | 内容 | カテゴリ | 関連Phase | 期待効果 |
|-------|------|---------|----------|---------|
| **#223** | **zero shotの対応の調査と実装** | **ゼロショット** | **Phase 3** | **未知話者対応** |
| #206 | 離散埋め込みによるProsody Embedding強化 | 品質向上 | Phase 1 | MOS +0.03-0.08 |
| #205 | OpenJTalkフルコンテキストラベル抽出 (Phase 1-5) | 品質向上 | Phase 1 | +5-10% 音質向上 |
| #203 | 日本語アクセント強度レベル [1/2/3] | 品質向上 | Phase 1 | MOS +0.03-0.05 |
| #202 | Duration正則化の追加 | 品質向上 | Phase 1 | MOS +0.02-0.04 |
| #201 | データ拡張 (SpecAugment, 速度/ピッチ変動) | 品質向上 | Phase 1 | MOS +0.05-0.10 |
| #200 | Multi-Resolution STFT Discriminator | 品質向上 | Phase 1 | MOS +0.08-0.12 |
| #153 | Godot向けライブラリ | プラットフォーム | Phase 1 | ゲームエンジン統合 |
| #142/#141/#137 | Web デモに中国語/韓国語追加 | 多言語 | Phase 2 | 多言語デモ |
| #110 | 無音・文間のランダム性制御 | 品質向上 | Phase 1 | 自然な発話リズム |
| #23 | マルチ言語モデルの学習/推論コード対応 | 多言語 | Phase 2 | 多言語基盤 |

#### Issue間の依存関係

```
#223 (zero shot) ←→ PR #222 (Dual-Mode Speaker Conditioning)
#205 (フルコンテキスト) → #206 (離散Prosody) → #203 (アクセント強度)
#23 (マルチ言語) ←→ PR #219 (6言語Phonemizer) ←→ PR #218 (バイリンガル)
#200 (MRSD) + #201 (データ拡張) + #202 (Duration正則化) → 品質向上パック
#142/#141/#137 (Webデモ多言語) ← PR #219 完了後に対応
#153 (Godot) ← godot-piper-plus リポジトリ作成済み (2026-03-08)
```

#### 最近クローズされたIssue (実装完了)

| Issue | 内容 | クローズ日 | 実装先 |
|-------|------|----------|-------|
| #207 | コンテキスト依存「ん」バリアント | 2026-01-07 | `japanese.py` |
| #204 | 拡張疑問文マーカー (?!, ?., ?~) | 2026-01-07 | `japanese.py` |
| #198/#199 | WavLM Discriminator | 2026-01-31 | `models.py`, `lightning.py` |
| #220 | Docker推論環境整理 | 2026-03-08 | PR #230 |
| #209 | 学習クラッシュ (SIGSEGV, long sequences) | 2026-01-07 | `monotonic_align` |

---

## 2. ロードマップ概要

```
Phase 1 (v1.7-v1.8)          Phase 2 (v2.0)           Phase 3 (v2.5)         Phase 4 (v3.0)
基盤強化 + 拡張機能           マルチ言語               ゼロショット           次世代アーキテクチャ
~3ヶ月                        ~3ヶ月                   ~3ヶ月                 ~6ヶ月
─────────────────────────────────────────────────────────────────────────────────────────
├─ ORT最適化・INT8量子化      ├─ 言語埋め込み          ├─ Speaker Encoder     ├─ FLY-TTS (ConvNeXt)
├─ Snake活性化・R1正則化      ├─ 統一IPA音素空間       ├─ SCL損失関数         ├─ VITS2改善統合
├─ Wyoming Protocol           ├─ 6言語Phonemizer       ├─ Cross-lingual       ├─ BERT プロソディ
├─ SSML・テキスト正規化       ├─ 多言語データセット     │  Voice Cloning       ├─ 感情/スタイル制御
├─ Noise-scaled MAS           ├─ コードスイッチング     ├─ LoRAファインチューン  ├─ GST (スタイルトークン)
├─ 品質評価ツール・API        ├─ 多言語C++推論         ├─ Few-shot適応        └─ 超軽量モデル (QAT)
├─ 話者モーフィング (SLERP)   └─ 多言語WebUI           └─ ONNX Speaker Enc.
├─ AudioSeal透かし                                        Issue #223
├─ exaggeration感情制御       PR #219, #218            PR #222
├─ ラウドネス正規化
└─ 既存Issue消化
   #200,#201,#202,#110
   #205,#206,#203
```

---

## 3. 優先アクション一覧

Phase 2 (マルチ言語) と Phase 3 (ゼロショット) は既にPRで対応中のため除外。

### 即効性の高い改善 (v1.7, 数時間〜数日)

| # | 改善 | Issue/PR | 効果 | 工数 |
|---|------|---------|------|------|
| 1 | ORT SessionOptions修正 (ENABLE_BASIC, MemArena) | - | 10-15%速度向上 | 数時間 |
| 2 | ARM NEON有効化 | - | ARM音声処理高速化 | 数時間 |
| 3 | **Noise-scaled MAS** | - | アライメント安定性向上 | **数時間** |
| 4 | `--quantize int8` オプション | - | サイズ3.8倍削減 (61→16MB) | 2-3日 |
| E1 | `--exaggeration` パラメータ | - | 感情制御 (§12.2) | 数時間 |
| E2 | ラウドネス正規化 + Soft-knee | - | 音割れ解決 (§12.4) | 数時間 |
| E7 | `--blend-speakers` SLERP | - | 話者モーフィング (§12.1) | 2-3日 |

### 基盤強化 (v1.7-v1.8, 数週間)

| # | 改善 | Issue/PR | 効果 | 工数 |
|---|------|---------|------|------|
| 5 | Wyoming Protocol TTSサーバー | - | **Home Assistant統合** | 1-2週間 |
| 6 | テキスト正規化 + SSMLパーサー | - | 数字/日付/通貨の正しい読み | 2-3週間 |
| 7 | MRSD追加 | **#200** | MOS +0.08-0.12 (※学習時間増に注意) | 1-2週間 |
| 8 | **Snake活性化関数** | - | **MOS +0.10-0.20** | 数日 |
| 9 | **R1 Regularization** | - | 学習安定性 + 音割れ軽減 | 数日 |
| 10 | OpenAI互換API | - | LocalAI/Open WebUI統合 | 数日 |
| 11 | 音割れ対策 (ラウドネス正規化等) | - | WavLMモデルの品質改善 | 数日 |
| 12a | データ拡張 | **#201** | MOS +0.05-0.10 | 1-2週間 |
| 12b | Duration正則化 | **#202** | MOS +0.02-0.04 | 1週間 |
| 12c | 無音・文間ランダム性制御 | **#110** | 自然な発話リズム | 数日 |
| E5 | `--watermark` (AudioSeal) | - | AI透かし (§12.9) | 3-5日 |

### 中期改善 (v2.0+, 数ヶ月)

| # | 改善 | 効果 | 工数 |
|---|------|------|------|
| 12 | **FLY-TTS (ConvNeXt V2 デコーダ)** | **Convベース高速化 (ONNX互換)** | 2-4週間 |
| 13 | VITS2 話者条件付きText Encoder | マルチスピーカー品質向上 | 1-2週間 |
| 14 | VITS2 敵対的Duration Predictor | Duration崩壊の根本解決 | 2-3週間 |
| 15 | VITS2 Transformer Flow | 長文の韻律一貫性向上 | 2-4週間 |
| 16 | 知識蒸留 (medium→x-low) | 4MBモデル実現 | 1-2週間 |
| 17 | sherpa-onnx対応 + ゲームエンジン | Godot/Unity/UE/モバイル統合 | 1-2週間 |
| 18 | HA Add-on | ワンクリックインストール | 数日 |

### 長期改善 (v3.0, 半年)

| # | 改善 | 効果 | 工数 |
|---|------|------|------|
| 19 | BERTプロソディ (独自実装, Apache-2.0) | End-to-end韻律予測 | 2-4週間 |
| 20 | 感情/スタイル制御 (T-VecTTS方式) | 感情表現力 | 2-4週間 |
| 21 | 超軽量モデル (Kokoro方式, 25MB) | エッジ向け | 1-2ヶ月 |
| 22 | Flow Matching統合 | 次世代品質 | 1-2ヶ月 |

> **注記: iSTFTベースデコーダ (MB-iSTFT-VITS/MS-iSTFT-VITS) について**
> mobile-vocoder での検証およびYoneyama et al. (Interspeech 2025) の実測データにより、
> ONNX Runtime環境ではiSTFTベースの速度優位が大幅に縮小することが確認されている。
> piper-plusはONNX推論が前提のため、iSTFTベースよりも**Convベースの最適化 (FLY-TTS ConvNeXt V2)** を優先する。
> 詳細は [4.1節](#41-推論高速化--onnx推論の現実) 参照。

> **注記: WavLM Discriminator の学習時間**
> WavLM Discriminatorは学習時間が**2倍以上**になるため、事前学習 (pre-training) では無効化し、
> ファインチューニング時のみ有効化する運用が現実的。MRSDも追加Discriminatorとして同様の考慮が必要。

### 最もインパクトが大きい組み合わせ

**短期 (数日の作業で大きな効果):**
```
ORT SessionOptions修正 + ARM NEON有効化 + Noise-scaled MAS + INT8量子化
→ 推論速度2-3倍、サイズ3.8倍削減、アライメント安定化、MOS低下わずか-0.02
```

**中期 (数週間で差別化):**
```
Snake活性化 + R1 Regularization + Wyoming Protocol + SSML
→ MOS +0.10-0.20 + HA統合 + VITSベースTTS初のSSML対応
```

**長期 (次世代性能):**
```
FLY-TTS (ConvNeXt) + VITS2改善 + 知識蒸留
→ Convベース高速化 (ONNX互換) + 長文韻律改善 + 4MBエッジモデル
```

---

## 4. Phase 1: 基盤強化 (v1.7 - v1.8)

### 4.1 推論高速化 — ONNX推論の現実

HiFi-GANデコーダが推論時間の大部分を消費しているため、iSTFTベースのデコーダへの置換が各論文で提案されている。しかし、**piper-plusはONNX Runtimeで推論するため、論文のPyTorchベンチマークとは大きく異なる結果になる**。

#### PyTorch vs ONNX: 速度効果の乖離

MB-iSTFT-VITSの「4.1倍高速」は**PyTorch CPU計測のみ**の数値。Yoneyama et al. (Interspeech 2025) による実測で、ONNX変換時のMACs膨張が明らかになった:

| ボコーダ | PyTorch MACs | ONNX MACs | 膨張率 | 備考 |
|---------|-------------|-----------|--------|------|
| **HiFi-GAN V1** | 28.02B | 28.02B | **1.00x** | 変化なし (Conv系はONNXで劣化しない) |
| MS-iSTFTNet | 8.37B | 8.39B | ~1.00x | iSTFTが小さいため影響軽微 |
| Vocos | 1.348B | **46.96B** | **34.8x** | iSTFT→Conv変換で爆発 |
| MS-Vocos | 1.356B | 4.215B | 3.1x | Multi-Stream化で軽減 |
| MS-Wavehax | 1.576B | 2.291B | 1.45x | 最もONNX耐性が高い |

**根本原因**: `torch.istft` はPyTorchではMKL/cuFFTの高速FFTを使うが、ONNXには`istft`オペレータが存在しない（[onnx/onnx#4777](https://github.com/onnx/onnx/issues/4777) は2025年時点でまだ"Future"）。ONNX変換時にConv1dベースの実装に置換する必要があり、FFTの速度優位が消失する。

さらに、ONNX Runtimeは**HiFi-GANのConvTranspose1dをConv+BN+ReLU融合で自動最適化**するため、HiFi-GAN側も速くなり、差がさらに縮まる。

#### mobile-vocoder での実証

[mobile-vocoder](https://github.com/ayutaz/mobile-vocoder) で以下のボコーダを比較検証済み:

| ボコーダ | パラメータ | ONNX MACs | ONNX CPU RTF | UTMOS | F0必要 |
|---------|----------|-----------|--------------|-------|--------|
| piper-plus HiFi-GAN | ~5-6M | ~28B | ベースライン | ベースライン | なし |
| **MS-Wavehax** | **0.298M** | **2.3B** | **16.7x RT** | 3.297 | **あり** |
| RNDVoC | 3.572M | 7.9B | 8.7x RT | ~3.2 | なし |
| iSTFTNet | ~25M | ~26B | - | - | なし |

**結論: ボコーダ単体の速度差はONNX推論では論文ほど出ない**

追加の課題:
- **MS-WavehaxはF0入力が必須** → VITSはF0を出力しないため、F0推定モジュールの追加が必要
- **UTMOS 3.297** はGT天井 (3.516) に対してまだギャップがある
- iSTFTNet (25Mパラメータ) はHiFi-GANとほぼ同じONNX MACsで速度メリットなし

#### デコーダ高速化が有効なケース

| ケース | 有効性 | 理由 |
|--------|--------|------|
| **PyTorch推論** (GPU学習サーバー等) | ✅ 有効 | FFTの速度メリットが得られる |
| **ONNX Runtime (CPU)** | △ 限定的 | Conv変換によりFFT優位が消失 |
| **モバイル/エッジ** | ✅ パラメータ削減は有効 | モデルサイズ (MB) の削減効果は残る |
| **WebAssembly** | △ 不明 | WASM環境でのConv vs FFT性能は未検証 |

#### ボコーダ比較 — ONNX推論での現実的評価

> **注意**: 以下の速度はPyTorchベンチマーク。ONNX Runtimeでは iSTFT系の速度優位が大幅に縮小する。

| ボコーダ | PyTorch速度 | ONNX速度 | 品質 | VITSドロップイン |
|---------|------------|---------|------|-----------------|
| HiFi-GAN (現行) | ベースライン | ベースライン | ベースライン | 既存 |
| MS-iSTFTNet | ~3.4x (PyTorch) | **ほぼ同等** (ONNX膨張なし) | 同等 | 可能 |
| MS-iSTFT-VITS | ~4x (PyTorch) | △ 縮小 | 同等 | 可能 |
| MB-iSTFT-VITS | 4.1x (PyTorch) | △ 縮小 + 周期ノイズ | 同等 | 可能 |
| FLY-TTS (ConvNeXt) | 8.8x (PyTorch) | **有望** (Convベース、iSTFT不使用) | 同等 | 要改修 |
| MS-Wavehax | 超高速 | 16.7x RT (膨張1.45x) | UTMOS 3.3 | **F0必要** |
| Vocos | 3x (PyTorch) | **使用不可** (膨張34.8x) | 高品質 | 要改修 |
| BigVGAN v2 | 1.5x | 同等 | 高品質 | 条件付き |

#### 推奨アプローチ (優先度順)

| 優先度 | アプローチ | 効果 | ONNX互換性 |
|--------|-----------|------|-----------|
| **1** | **ORT SessionOptions最適化** (ENABLE_BASIC, MemArena) | 10-15%速度向上 | ✅ 即効 |
| **2** | **INT8量子化** | サイズ3.8x削減、速度1.5-2x | ✅ 即効 |
| **3** | **ConvNeXt V2ブロック置換** (FLY-TTS方式) | ResBlock高速化 (Conv系なのでONNXで劣化しない) | ✅ 良好 |
| 4 | MS-iSTFTNet (iSTFTステップ最小化) | パラメータ削減 | ○ 良好 |
| 5 | MS-Wavehax (スタンドアロンボコーダ) | 超軽量 (0.298M) だがF0必要 | ○ 膨張1.45x |

> **重要**: FLY-TTSのConvNeXt V2アプローチは、HiFi-GANの`ResBlock`を置換するもので**iSTFTを使わない**ため、ONNX変換時のMACs膨張問題を回避できる。Conv系の最適化はONNXでも有効であり、最も現実的な高速化候補。

### 4.2 エッジ最適化・量子化

#### ONNX INT8量子化

**静的 vs 動的量子化:**

| 方式 | キャリブレーション | サイズ削減 | 速度向上 | MOS低下 |
|------|-------------------|----------|---------|---------|
| 動的 | 不要 | 3.6x | 1.5-2.0x | -0.02〜-0.05 |
| **静的 (推奨)** | 100-500サンプル | **3.8x** | **2.0-3.0x** | **-0.01〜-0.03** |

**レイヤー別の量子化戦略:**

| レイヤー | 量子化 | 理由 |
|---------|--------|------|
| Generator (Conv1d) | **する** (70-80%のパラメータ) | 最大のサイズ/速度効果 |
| Flow (WaveNet) | **する** | 耐性あり |
| Duration Predictor | **する** | 小さいが耐性あり |
| TextEncoder (Attention) | **スキップ** | 韻律品質に影響 |
| Generator conv_post | **スキップ** | 波形品質に直結 |
| Embedding | **スキップ** | ルックアップテーブル、効果なし |

**具体的な数値 (VITS-medium, 61MB FP32):**

| 構成 | サイズ | CPU RTF (x86) | CPU RTF (ARM64) |
|------|--------|---------------|-----------------|
| FP32 | 61 MB | 0.28 | 0.45 |
| 動的INT8 | 17 MB | 0.16 | 0.28 |
| 静的INT8 | **16 MB** | **0.12** | **0.22** |

#### ORT SessionOptions — 即時改善 (数時間)

**現在のpiper-plusの問題点**:

1. `piper.cpp` L628: `ORT_DISABLE_ALL` → **`ORT_ENABLE_BASIC`** に変更で5-10%速度向上
2. `piper.cpp` L634: `DisableCpuMemArena()` → **`EnableCpuMemArena()`** で繰り返し推論10-15%高速化
3. `DisableMemPattern()` → **`EnableMemPattern()`** でメモリ再利用

#### ARM NEON有効化 (数時間)

`CMakeLists.txt` L83 の `USE_ARM64_NEON` が**コメントアウトされている**。有効化するだけ。

#### 知識蒸留

| 構成 | パラメータ | サイズ | MOS |
|------|----------|--------|-----|
| medium (教師) | ~28M | 61 MB | 4.12 |
| x-low (ゼロから学習) | ~7M | 15 MB | 3.65 |
| **x-low (蒸留)** | ~7M | 15 MB | **3.88** |
| **x-low (蒸留+INT8)** | ~7M | **4 MB** | **3.82** |

蒸留で品質ギャップの**50%**を埋められる。

#### ハードウェア別ベンチマーク

| デバイス | モデル | RTF | レイテンシ (1文) |
|---------|--------|-----|-----------------|
| Raspberry Pi 4 | INT8 medium | 0.22 | 0.5s |
| Raspberry Pi 4 | INT8 x-low | **0.08** | 0.18s |
| Raspberry Pi 5 | INT8 medium | 0.11 | 0.25s |
| Android (Snapdragon 8g2) | INT8 medium | 0.04 | 95ms |
| iOS (A16, CoreML) | INT8 medium | 0.03 | 70ms |
| Browser WASM (M1 Chrome) | FP32 x-low | 0.12 | 280ms |
| Browser WASM (i7 Chrome) | INT8 x-low | 0.18 | 420ms |

### 4.3 SSML対応・テキスト正規化

#### なぜSSMLが重要か

- オリジナルpiperで13コメントの高需要機能 (Issue #275)
- **Coqui TTSもSSML未実装** → VITSベースTTSでpiper-plusが先駆者になれる
- Home Assistant / 音声アシスタントとの統合に不可欠

#### 実装すべきSSMLタグ (優先度順)

| 優先度 | タグ | VITSでの実現方法 |
|--------|------|-----------------|
| **A** | `<break>` | 無音トークン `_` (pau) の複数挿入 |
| **A** | `<prosody rate>` | `scales[1]` (length_scale) にマッピング、セグメント単位推論 |
| **A** | `<say-as>` | テキスト正規化の前処理で展開 |
| **A** | `<phoneme>` | phoneme_id_mapで直接IPA→ID変換 |
| **A** | `<sub>` | 文字列置換 (既存CustomDictionaryと統合) |
| B | `<prosody pitch>` | noise_scale調整 (限定的) |
| B | `<prosody volume>` | 後処理ゲイン調整 |
| B | `<emphasis>` | duration延長 + prosody_features操作 |

#### テキスト正規化 — SSMLより先に実装すべき基盤

現在piper-plusはOpenJTalk内部処理に完全依存。英語の数字正規化、通貨、URLは非対応。

| 用途 | ライブラリ | ライセンス |
|------|-----------|-----------|
| 日本語数字 | OpenJTalk (既存) + kanjize (MIT) | BSD / MIT |
| 英語数字/通貨 | num2words | LGPL |
| 本格的多言語 | NeMo Text Processing | Apache-2.0 |
| OpenJTalk代替 (Rust) | jpreprocess | MIT |

#### 実装アーキテクチャ

```
入力テキスト (SSML or プレーン)
  → [SSMLパーサー] xml.etree.ElementTree (外部依存なし)
    → [テキスト正規化] normalize/ (Phonemizer ABCと同パターン)
      → [Phonemizer] phonemize/ (既存)
        → [SSML後処理] break→無音挿入, rate→length_scale
          → [ONNX推論] (既存)
```

#### 実装ファイル

```
src/python/piper_train/
├── ssml/parser.py          ← SSMLパーサー (新規)
├── ssml/types.py           ← SSMLSegment データ型 (新規)
├── ssml/processor.py       ← 統合処理 (新規)
├── normalize/base.py       ← TextNormalizer ABC (新規)
├── normalize/japanese.py   ← 日本語正規化 (新規)
├── normalize/english.py    ← 英語正規化 (新規)
└── infer_onnx.py           ← --ssml オプション追加
```

### 4.4 Home Assistant・音声アシスタント統合

#### Wyoming Protocol

Home Assistantの標準音声プロトコル (v1.7.0)。TCP + JSONL + PCMバイナリ。

**TTSメッセージフロー:**
```
Client → Server: Synthesize (text, voice)
Server → Client: AudioStart (rate=22050, width=2, channels=1)
Server → Client: AudioChunk (PCM int16) × N回
Server → Client: AudioStop
```

#### 実装パス

**Wyoming TTSサーバー (1-2週間)**:

既存の `inference.py` (FastAPI) をWyomingサーバーでラップ:

```python
from wyoming.tts import Synthesize
from wyoming.audio import AudioStart, AudioChunk, AudioStop
from wyoming.server import AsyncTcpServer

class PiperPlusEventHandler(AsyncEventHandler):
    async def handle_event(self, event):
        if Synthesize.is_type(event.type):
            audio = self.engine.synthesize(event.data["text"])
            await self.write_event(AudioStart(...))
            await self.write_event(AudioChunk(audio=audio_bytes))
            await self.write_event(AudioStop())
```

**Home Assistant Add-on (数日)**:

既存Dockerイメージ + `config.yaml` + s6-overlay:
```yaml
name: Piper Plus
version: 1.7.0
slug: piper_plus
description: "Japanese-focused neural TTS with prosody control"
ports:
  "10200/tcp": 10200
```

**sherpa-onnx対応確認 (数日)**:

piper-plus ONNXモデルがsherpa-onnx経由で動作確認できれば:
- **Godot** (Issue #153) → godot-kokoro/godot-piperパターンで統合
- **Unity** → uPiper (同組織) が既に存在
- **Unreal** → Runtime TTS pluginが対応
- **Android/iOS** → sherpa-onnx SDKで直接利用
- **NVDA** → スクリーンリーダー統合
- **SAPI5** → Windowsシステム音声

#### 競合状況

| TTS | HA統合 | 品質 | 日本語 |
|-----|--------|------|--------|
| Piper (オリジナル) | Wyoming Add-on | 基本VITS | なし |
| piper1-gpl (OHF) | Wyoming Add-on | 基本VITS | なし |
| Kokoro-82M | Wyoming wrapper | 高い | 限定的 |
| **piper-plus** | **未実装 → 対応予定** | **WavLM+EMA** | **最高** |

piper-plusのWyoming対応は、**日本語対応のHome Assistant TTSが存在しない**という空白を埋める。

### 4.5 音声品質向上

#### Discriminator改善

| Discriminator | piper-plus状態 | 追加効果 |
|--------------|---------------|---------|
| MPD (Multi-Period) | 実装済み | ベースライン |
| WavLM Discriminator | 実装済み | MOS +0.15-0.25 |
| **MRSD (Multi-Resolution STFT)** | Issue #200 | MOS +0.05-0.10 |
| MS-SB-CQTD (BigVGAN) | 未実装 | MOS +0.03-0.05 |

MRSDは次に追加すべきDiscriminator。

> **⚠️ Discriminator数の上限**: VNet論文 (2024) の知見として、**3つ以上のDiscriminatorを併用するとモード崩壊**が発生するリスクがある。MPD + WavLM + MRSD の3併用が**現実的な上限**。MS-SB-CQTDを追加する場合はMRSDと置換する形で検討。

> **⚠️ WavLM Discriminatorの学習時間**: WavLM Discriminatorを有効にすると**学習時間が2倍以上**になる。そのため、事前学習(pre-training)フェーズではWavLM Discriminatorを無効にし、**ファインチューニング時のみ有効化**する運用が現実的。MRSDも同様に学習コストを考慮した運用設計が必要。

#### BigVGAN Snake活性化関数

BigVGANで導入された**Snake活性化関数**は、HiFi-GANのLeakyReLUを置換する周期的活性化関数。

```python
# Snake activation: x + (1/alpha) * sin²(alpha * x)
class Snake(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return x + (1.0 / self.alpha) * torch.sin(self.alpha * x) ** 2
```

- **MOS向上: +0.10〜0.20** (HiFi-GAN LeakyReLU比)
- 音声波形の周期的構造に対する誘導バイアスが品質向上に寄与
- HiFi-GANの `ResBlock` 内の `LeakyReLU` を `Snake` に差し替えるだけで適用可能

#### 音割れ対策 (WavLMモデル)

**後処理 (既存モデルに適用可能):**

1. **ラウドネス正規化** (pyloudnorm, -25dB LUFS) — 前処理で実装
2. **Spectral noise gating** — 高周波ノイズ除去
3. **De-clipping後処理** — `np.clip(audio, -0.95, 0.95)` + soft-knee圧縮

**学習時の根本解決:**

4. **c_wavlm=0.2 で再学習** (CLAUDE.mdのオプションB)
5. **Feature Matching重みの分離**: WavLM損失を`c_wavlm_adv` (敵対的) と `c_wavlm_fm` (Feature Matching) に分離し、`c_wavlm_fm`を下げることで高振幅生成を抑制
6. **R1 regularization追加**: Discriminatorの過度な勾配を抑制し、Generatorの過剰応答を防止

#### 学習安定性改善

| 技術 | 内容 | piper-plus状態 |
|------|------|---------------|
| R1 regularization | Discriminatorの勾配ペナルティ | 未実装 |
| Gradient penalty | WGAN-GP式のペナルティ | 未実装 |
| LR warm-up | 最初のN epochで低LRから開始 | 実装済み (warmup_epochs) |
| EMA | HiFi-GANデコーダのEMA | 実装済み |

R1 regularizationの追加が最も効果的で実装コストが低い。

```python
# lightning.py の training_step_d() 内に追加
def compute_r1_penalty(self, real_audio, discriminator):
    real_audio.requires_grad_(True)
    real_out = discriminator(real_audio)
    grad = torch.autograd.grad(
        outputs=real_out[-1].sum(), inputs=real_audio,
        create_graph=True, only_inputs=True
    )[0]
    return (grad * grad).sum(dim=[1, 2]).mean() * 0.5

# Discriminator損失に追加:
# loss_d = loss_d + r1_weight * r1_penalty
# 推奨 r1_weight: 10.0
```

#### noise_scale の最適値

| パラメータ | デフォルト | 推奨値 | 効果 |
|-----------|----------|--------|------|
| noise_scale (scales[0]) | 0.667 | 0.5-0.667 | ピッチ変動制御。低い=安定、高い=表現力 |
| length_scale (scales[1]) | 1.0 | 1.0 | 話速。0.8=速い、1.2=遅い |
| noise_w (scales[2]) | 0.8 | 0.6-0.8 | Duration変動。低い=均一リズム |

WavLMモデルは `noise_scale=0.5` が推奨 (CLAUDE.md記載通り)。

### 4.6 品質評価ツール導入

| ツール | 用途 | ライセンス |
|--------|------|-----------|
| UTMOS | MOS予測 (VoiceMOS 2022チャレンジ1位) | Apache-2.0 |
| TTSDS2 | 分布ベース評価 (2-Wasserstein距離) | OSS |
| NISQA v2.0 | 非侵入型音質評価 (4次元) | MIT |

**実装**: `src/python/piper_train/tools/evaluate_quality.py`

### 4.7 API改善

#### Python FastAPI ストリーミング対応

C++側は既にストリーミング実装済みだが、Python側は非ストリーミング。

```python
# docker/python-inference/inference.py
@app.get("/synthesize_stream")
async def synthesize_stream(text: str, speaker_id: int = 0):
    async def generate():
        for sentence in split_sentences(text):
            audio = synthesize(sentence)
            yield audio_to_wav_chunk(audio)
    return StreamingResponse(generate(), media_type="audio/wav")
```

#### OpenAI互換 `/v1/audio/speech` エンドポイント

LocalAI / Open WebUI との統合を容易にする業界標準API。

```
POST /v1/audio/speech
{
  "model": "piper-plus",
  "input": "テキスト",
  "voice": "speaker_0",
  "response_format": "wav",
  "stream": true
}
```

#### WebSocket API

LLM連携（音声アシスタント）に必須。テキストストリーミング入力 + 音声チャンクストリーミング出力。

### 4.8 データセット品質改善

| 機能 | 実装先 | 工数 |
|------|--------|------|
| ラウドネス正規化 (pyloudnorm, -25dB) | `norm_audio/loudness.py` | 1日 |
| UTMOS/NISQAスコアリング | `tools/quality_score.py` | 2-3日 |
| Whisperトランスクリプト検証 | `tools/verify_transcripts.py` | 2日 |
| 速度摂動データ拡張 (0.9x/1.0x/1.1x) | `preprocess.py` | 1-2日 |

**推奨品質フィルタリング基準**: UTMOS >= 3.5、CER <= 5%

### 4.9 既存Issue消化

#### 品質向上 Issue (Phase 1)

| Issue | 内容 | 工数 | 期待効果 | 依存 |
|-------|------|------|---------|------|
| **#200** | Multi-Resolution STFT Discriminator | 1-2週間 | MOS +0.08-0.12 | なし |
| **#201** | SpecAugment + 速度/ピッチ変動データ拡張 | 1-2週間 | MOS +0.05-0.10 | なし |
| **#202** | Duration正則化 | 1週間 | MOS +0.02-0.04 | なし |
| **#206** | 離散Prosody Embedding | 1週間 | MOS +0.03-0.08 | #205推奨 |
| **#205** | OpenJTalkフルコンテキストラベル抽出 | Phase別 | +5-10% 音質 | なし |
| **#203** | 日本語アクセント強度レベル [1/2/3] | 数日 | MOS +0.03-0.05 | #205推奨 |
| **#110** | 無音・文間ランダム性制御 | 数日 | 自然な間 | なし |

> **推奨消化順序**: #200 (MRSD) → #201 (データ拡張) → #202 (Duration正則化) → #110 (無音制御) → #205 (フルコンテキスト) → #206 (離散Prosody) → #203 (アクセント強度)
>
> **注意**: #200 (MRSD) はDiscriminator追加のため学習時間が増加する。WavLM Discriminatorと同様に、ファインチューニング時のみの有効化を検討 (3.節の注記参照)。

#### プラットフォーム Issue

| Issue | 内容 | 工数 | 備考 |
|-------|------|------|------|
| **#153** | Godot向けライブラリ | 1-2週間 | [godot-piper-plus](https://github.com/ayutaz/godot-piper-plus) リポジトリ作成済み |

#### 多言語 Issue (Phase 2)

| Issue | 内容 | 工数 | 関連PR |
|-------|------|------|-------|
| **#23** | マルチ言語モデルの学習/推論コード対応 | 数週間 | PR #219, #218 |
| **#142/#141/#137** | Webデモに中国語/韓国語追加 | 数日 | PR #219 完了後 |

#### ゼロショット Issue (Phase 3)

| Issue | 内容 | 工数 | 関連PR |
|-------|------|------|-------|
| **#223** | zero shotの対応の調査と実装 | 数週間 | PR #222 |

---

## 5. Phase 2: マルチ言語対応 (v2.0)

> **PR #218/#219 で対応中**

### アーキテクチャ: 統一IPA + 言語埋め込み方式

**YourTTS方式を採用** - VITSアーキテクチャへの変更が最小限。

```
入力: [phoneme_embedding (192d)] + [language_embedding (4d)] → Text Encoder
```

```python
class SynthesizerTrn(nn.Module):
    def __init__(self, ..., num_languages=1, language_emb_dim=4):
        ...
        if num_languages > 1:
            self.language_emb = nn.Embedding(num_languages, language_emb_dim)
```

### 統一IPA音素空間

- IPA統一音素IDマップを作成 (全言語共通)
- 日本語のPUA文字（疑問詞マーカー、Nバリアント）もIDとして保持
- 各言語の `Phonemizer` にIPA出力モードを追加
- 1話者/1言語だとIPA記号から話者情報がリーク → 各言語に複数話者が必要

### 対応言語 (PR #219)

| 言語 | Phonemizer | 音素化エンジン | ライセンス |
|------|-----------|---------------|-----------|
| 日本語 (ja) | `JapanesePhonemizer` | OpenJTalk | BSD |
| 英語 (en) | `EnglishPhonemizer` | g2p-en | Apache-2.0 |
| 中国語 (zh) | `ChinesePhonemizer` | pypinyin | MIT |
| スペイン語 (es) | 新規実装 | IPA変換 | - |
| フランス語 (fr) | 新規実装 | IPA変換 | - |
| ポルトガル語 (pt) | 新規実装 | IPA変換 | - |

**追加候補** (オリジナルpiperで高需要): 韓国語 (ko)、広東語 (yue)、ヒンディー語 (hi)

### C++推論側の対応

- `PhonemeType` に新言語の列挙値を追加
- `config.json` に `language` フィールドを追加
- espeak-ng依存は可能な限り回避し、言語別のGPL-freeライブラリを使用

---

## 6. Phase 3: ゼロショットTTS (v2.5)

> **PR #222 で対応中** | **Issue #223: zero shotの対応の調査と実装**

### アーキテクチャ概要

数秒の参照音声から未知話者の声を再現。既存の話者ID方式との後方互換性を維持。

```
[参照音声 (3-10秒)] → [Speaker Encoder] → [d-vector (256d)]
                                              ↓
[テキスト] → [Text Encoder] → [Duration Predictor] → [Flow] → [Decoder] → [音声]
```

### Phase 3a: Speaker Encoder導入 (最小変更)

```python
class SynthesizerTrn(nn.Module):
    def __init__(self, ..., use_speaker_encoder=False):
        ...
        if use_speaker_encoder:
            self.speaker_encoder = ECAPA_TDNN(...)  # 事前学習済み
        elif n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)  # 既存方式
```

| エンコーダ | 埋め込み次元 | 事前学習 | 推奨度 |
|-----------|-----------|---------|--------|
| ECAPA-TDNN | 192 | SpeechBrain (VoxCeleb) | 高 |
| GE2E (LSTM) | 256 | dvector (LibriSpeech) | 中 |
| H/ASP | - | YourTTSオリジナル | 高 (TTSで最良) |

**Dual-Mode対応** (PR #222):
- 学習時: speaker_id (離散) または d-vector (連続) のどちらでも条件付け可能
- 推論時: `--speaker-id 0` または `--reference-audio ref.wav` の二択

### Phase 3b: Speaker Consistency Loss (SCL)

```python
# lightning.py training_step_g() に追加
speaker_emb_gt = self.speaker_encoder(audio_gt)
speaker_emb_gen = self.speaker_encoder(audio_gen)
loss_scl = 1 - F.cosine_similarity(speaker_emb_gt, speaker_emb_gen)
```

### Phase 3c: Cross-lingual Zero-Shot

言語埋め込みとspeaker embeddingの分離が鍵。Phase 2完了が前提。

### ONNX対応

speaker encoder を別ONNXモデルとしてエクスポート:
`reference_audio → speaker_encoder.onnx → d-vector → synthesizer.onnx → audio`

---

## 7. Phase 4: 次世代アーキテクチャ (v3.0)

### 7.1 VITS2改善の統合

#### Noise-scaled MAS ⭐ 最低コスト改善

- MASにガウスノイズを追加してアライメント安定性向上
- **実装コスト最小**: 数行の変更のみ
- Duration Predictor崩壊の予防にも寄与

```python
# models.py の SynthesizerTrn.forward() 内
neg_cent = mas_negative_cent(...)
noise = torch.randn_like(neg_cent) * noise_scale_mas  # noise_scale_mas = 0.01
neg_cent = neg_cent + noise
attn = maximum_path(neg_cent, attn_mask)
```

#### その他のVITS2改善

| 改善 | 内容 | 期待効果 | 工数 |
|------|------|---------|------|
| Transformer Flow | Normalizing FlowにTransformerブロック導入 | 長距離依存改善 | 2-4週間 |
| 敵対的Duration Predictor | DPに条件付きDiscriminator追加 | Duration崩壊の根本解決 | 2-3週間 |
| 話者条件付きText Encoder | Text Encoderにも話者情報を注入 | 話者類似度向上 | 1-2週間 |

**推奨実装順序**: Noise-scaled MAS → 話者条件付きTE → 敵対的DP → Transformer Flow

**参考実装**: [p0p4k/vits2_pytorch](https://github.com/p0p4k/vits2_pytorch) (~545 stars, **推奨ベース**)

#### 関連改善 (VITS2以外)

| 技術 | 出典 | 内容 |
|------|------|------|
| Q-VITS | 2024 | RoPE + ConfoGAN。位置エンコーディング改善 |
| NaturalSpeech | Microsoft | 事前学習Phoneme Encoder (MLM)。BERTの軽量版 |
| Period VITS | 2024 | 周期的生成でピッチ精度向上 |

### 7.2 BERTプロソディ予測

- Apache-2.0のBERTモデル (`tohoku-nlp/bert-base-japanese` 等) を使用
- BERTの文脈埋め込みをText Encoderの入力に連結
- AGPL-3.0の Bert-VITS2 のコードは使用せず、独自実装
- 軽量版BERTの蒸留モデルを使用 (モデルサイズ対策)

### 7.3 感情/スタイル制御

**T-VecTTS方式** (凍結 + 学習可能ブランチ):
- 既存のVITSモデルを凍結
- 感情制御用の軽量ブランチを接続
- 制御軸: 感情 (喜び、悲しみ、怒り、驚き)、話速、ピッチレンジ、エネルギー

### 7.4 超軽量モデル (KittenTTS/Kokoro方式)

25MB以下でSOTA品質。エッジデバイス向け。

- 知識蒸留: 大規模モデル → 82Mパラメータ軽量モデル
- 構造化プルーニング: モデルサイズ60%削減、精度89.5%維持
- `quality` 設定に `nano` ティアを追加

### 7.5 Flow Matching統合 (長期)

- VITSのNormalizing FlowをOptimal Transport CFM (OT-CFM) に置換
- Decoder部分は既存のまま維持
- エンコーダ部分のみ変更

### 7.6 FLY-TTS (ConvNeXt V2 デコーダ)

HiFi-GANの`ResBlock`を**ConvNeXt V2ブロック**に置換。

- **8.8x CPU高速化** (PyTorchベンチ)
- **iSTFTを使わない** → ONNX MACsの膨張問題を完全回避
- Conv系の最適化はONNX Runtimeでも有効
- iSTFTとの組み合わせでさらなる高速化も可能

### 7.7 44.1kHz / 48kHz 高サンプルレート

- 現在: 22050Hz
- 学習時に高サンプルレートで学習する方が後処理アップサンプリングより品質が高い
- ただし学習コスト2-4倍、推論速度低下
- 短期的には22050Hzを維持し、post-processingで44.1kHzにアップサンプリング推奨

---

## 8. VITSベースOSS比較と移植候補

### プロジェクト比較表

| プロジェクト | Stars | ライセンス | 主な革新 | 移植優先度 |
|---|---|---|---|---|
| **VITS2** | ~545 | MIT | Transformer Flow, 敵対的DP, Noise-scaled MAS | **最高** |
| **StyleTTS2** | ~6.1k | MIT | SLM Discriminator, スタイル拡散 | **高** |
| **FLY-TTS** | - | - | ConvNeXt V2デコーダ (ONNX互換の高速化) | **高** |
| **GPT-SoVITS** | ~55.6k | MIT | ゼロショット, 2段階アーキ | 中 |
| **Bert-VITS2** | ~8.6k | AGPL | BERTプロソディ | 中 (ライセンス注意) |
| MB-iSTFT-VITS | 数百 | MIT | iSTFTデコーダ (**ONNX効果限定的**) | 低 |
| **Coqui TTS** | ~44.8k | MPL-2.0 | 包括ツールキット | 低-中 |
| **Fish Speech** | ~25.8k | 独自 | Dual-AR, 4Bパラメータ | 低 |
| **Kokoro** | 数千 | Apache-2.0 | 82M超軽量 | 中 |
| **Chatterbox** | 数千 | MIT | 感情制御 | 低 |

### 移植推奨順位

1. **Noise-scaled MAS (VITS2)** → Phase 1 (数行で即効)
2. **Snake活性化 (BigVGAN)** → Phase 1 (MOS +0.10-0.20)
3. **FLY-TTS ConvNeXt V2 デコーダ** → Phase 4 (ONNX互換の高速化)
4. **VITS2 の 敵対的DP + 話者条件付きTE** → Phase 4 (品質向上)
5. **StyleTTS2 の微分可能Duration Modeling** → Phase 4 (Duration崩壊根本解決)
6. **BERTプロソディ** (独自実装) → Phase 4 (end-to-endプロソディ)
7. **Kokoro的超軽量モデル** → Phase 4 (25MBエッジ向け)

---

## 9. オリジナルpiperとの差分

### piper-plusが既に優位な領域

| 領域 | オリジナルpiperの問題 | piper-plusの対応 |
|------|---------------------|-----------------|
| GPL依存 | espeak-ng必須 | g2p-en (Apache-2.0) で英語対応済み |
| 日本語 | 音声なし (12 thumbs-up) | OpenJTalk + 独自Phonemizer |
| 音素制御 | IPA直接入力不可 (PR3件) | Phonemizer ABCで対応済み |
| 音質向上 | 基本VITSのみ | WavLM Discriminator + EMA + prosody |
| パッケージ | pip install壊れている (56コメント) | piper-tts-plus で正常動作 |
| 学習環境 | ドキュメント不備、チェックポイント互換問題 | 詳細ガイド、WandB監視 |

### オリジナルpiperのユーザーが最も求めている機能

| 機能 | piperでの需要 | piper-plus対応状況 | ロードマップ |
|------|-------------|-------------------|------------|
| SSMLサポート | 13コメント | 未実装 | Phase 1 |
| アライメント出力 | 8コメント, PR2件 | C++で一部対応 | Phase 1で拡張 |
| Raw IPA音素入力 | PR3件 | 対応済み | 完了 |
| Webブラウザ (WASM) | 36コメント | WASM音素化あり | Phase 2 |
| ライブラリAPI | 12コメント | PyPI + C++ API | 完了 |
| Android TTS | 37コメント | sherpa-onnx経由で可能 | Phase 1でガイド作成 |
| 韓国語/広東語/ヒンディー | 高需要 | 未対応 | Phase 2 |
| Home Assistant統合 | 高 | 未対応 | Phase 1 (Wyoming) |

---

## 10. データセット戦略

### 現在の資産

| データセット | 発話数 | 話者数 | 状態 |
|------------|--------|--------|------|
| moe-speech-20speakers-v2 | 60,164 | 20 | 学習完了 |
| つくよみちゃん | - | 1 | HuggingFace公開済み |

### 拡張計画

| 優先度 | 提案 | 効果 | 工数 |
|-------|------|------|------|
| **1** | 品質フィルタリングツール導入 | MOS +0.1-0.2 | 2-3日 |
| **2** | ラウドネス正規化追加 | 音割れ軽減 | 1日 |
| **3** | Whisperトランスクリプト検証 | 不正確データ除去 | 2日 |
| **4** | MoeSpeechから話者拡張 (20→50) | 話者多様性向上 | 3-5日 |
| **5** | 速度摂動によるデータ拡張 | データ量3倍 | 1-2日 |
| **6** | LibriTTS-Rで英語モデル学習 | 英語品質向上 | 3-5日 |
| **7** | CSS10/CML-TTSで多言語モデル学習 | Phase 2の基盤 | 1-2週間 |
| **8** | コミュニティ貢献ガイドライン | データ収集促進 | 1日 |

### 多言語データセット活用計画

| 言語 | データセット | 話者数 | 時間 |
|------|------------|--------|------|
| 日本語 | MoeSpeech + 既存 | 50+ | 数百時間 |
| 英語 | LibriTTS-R | 2,456 | 585時間 |
| 中国語 | Common Voice zh | - | 検証済み数十時間 |
| スペイン語 | CSS10 + MLS | 複数 | 数十時間 |
| フランス語 | CSS10 + MLS | 複数 | 数十時間 |
| 韓国語 | CSS10 + 独自 | - | - |

### HuggingFace Hub戦略

- `ayousanz/piper-plus-*` 名前空間で統一管理
- 事前学習済みベースモデルを公開し、コミュニティによるファインチューニングを促進
- データセット公開テンプレート (JSONL仕様書) を提供

---


## 11. 調査ソース・拡張機能調査

調査ソース一覧（論文41件、OSSリポジトリ50+件）および拡張機能調査（VITS系OSS横断分析）は別ドキュメントに分離しました。

- **[拡張機能調査・参考文献](research/extended-features-research.md)** — GPT-SoVITS, Style-BERT-VITS2, Chatterbox, Kokoro等の横断分析、話者モーフィング、感情制御、ストリーミング、透かし技術等

---

## 付録: オリジナルpiperの全対応言語一覧

アラビア語、カタルーニャ語、チェコ語、ウェールズ語、デンマーク語、ドイツ語、ギリシャ語、英語 (GB/US)、スペイン語 (AR/ES/MX)、ペルシャ語、フィンランド語、フランス語、ヒンディー語、ハンガリー語、アイスランド語、イタリア語、ジョージア語、カザフ語、ルクセンブルク語、ラトビア語、マラヤーラム語、ネパール語、オランダ語 (BE/NL)、ノルウェー語、ポーランド語、ポルトガル語 (BR/PT)、ルーマニア語、ロシア語、スロバキア語、スロベニア語、セルビア語、スウェーデン語、スワヒリ語、トルコ語、ウクライナ語、ベトナム語、中国語

piper-plusはこれらのupstreamモデルと互換性を維持しており、上記言語の既存モデルをそのまま使用可能です。
