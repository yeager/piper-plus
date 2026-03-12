# Stack-chan + Module LLM CPU 検証ガイド

このドキュメントは、`Stack-chan` で `piper-plus` を使うために、
`Module LLM` の `CPU` 上で日本語モデルを動かせるかを検証するための手順書です。

更新日: 2026-03-11

## 目的

この検証の目的は次の 3 点です。

1. `Module LLM` 上で `piper-plus` の日本語モデルが `CPU` 実行できるか確認する
2. 実行時メモリと速度が、`Stack-chan` 連携の最低条件を満たすか判断する
3. `NPU 移植` に進む価値があるか、`CPU 実行時点` で見極める

## このドキュメントの前提

- `piper-plus` を動かすこと自体が目的
- `Stack-chan Core` 単体では動かさない
- `外部ホスト` は使わない
- `Module LLM` は使ってよい
- 対象は `日本語モデル` のみ
- 最初の検証対象は `NPU` ではなく `CPU`

## なぜ CPU 検証から始めるのか

`piper-plus` は `Linux x86_64 / ARM64` を正式対応範囲にしており、`C++` と `Python` の両方で通常の `onnxruntime` を使う設計です。

関連ファイル:

- [README.md](../../../README.md)
- [src/cpp/piper.cpp](../../../src/cpp/piper.cpp)
- [src/python_run/piper/voice.py](../../../src/python_run/piper/voice.py)

一方、`` は `Ubuntu 22.04` を内蔵した Linux モジュールで、`ADB`, `UART`, `SSH` を通じて通常の Linux マシンとして操作できます。

そのため、最初に検証すべきなのは:

`arm64 Linux に piper-plus をそのまま載せて CPU 実行できるか`

です。

`NPU` はその後です。CPU 実行が成立しない状態で `NPU 変換` に進むのは非効率です。

## Module LLM 側で確認済みの前提

公開ドキュメントで確認できる事実:

- SoC: `AX630C`
- CPU: `Dual Cortex-A53 1.2 GHz`
- メモリ: `4GB LPDDR4`
- ただし内訳は `1GB system memory + 3GB dedicated to hardware acceleration`
- ストレージ: `32GB eMMC`
- OS: `Built-in Ubuntu system`

重要なのは、`CPU 実行で自由に使えるメモリは 1GB と見なすべき` ことです。

参考資料:

- <https://docs.m5stack.com/en/module/Module-llm>
- <https://docs.m5stack.com/en/stackflow/module_llm/config>
- <https://docs.m5stack.com/en/stackflow/module_llm/software>

## piper-plus 側で確認済みの前提

### 対応範囲

- `Linux ARM64` は正式対応
- `GitHub Releases` に `arm64` バイナリ配布あり

関連ファイル:

- [README.md](../../../README.md)
- [docs/guides/japanese/japanese-usage.md](../japanese/japanese-usage.md)

### 日本語実行に必要なもの

- 日本語 ONNX モデル
- モデル設定ファイル (`.onnx.json`)
- OpenJTalk 系の音素化
- OpenJTalk 辞書

関連ファイル:

- [src/python_run/piper/voice.py](../../../src/python_run/piper/voice.py)
- [src/cpp/piper.cpp](../../../src/cpp/piper.cpp)
- [src/cpp/openjtalk_dictionary_manager.c](../../../src/cpp/openjtalk_dictionary_manager.c)

### サイズ上の注意

ローカル確認済みの日本語資産サイズ:

- [`test/models/ja_JP-test-medium.onnx`](../../../test/models/ja_JP-test-medium.onnx): 約 `60MB`
- Web 用日本語辞書全体: 約 `100MB+`

`Module LLM` の `1GB system memory` から見ると、
`ESP32-S3` よりは現実的ですが、軽量とは言えません。

## 検証のゴール

### 最低成功条件

次の 4 点を満たせば、`CPU 実行ルートは継続価値あり` と判断します。

1. `Module LLM` 上で `piper-plus` が起動する
2. 日本語 1 文を `CPU` で最後まで合成できる
3. `OOM` や異常終了が起きない
4. 速度を定量的に記録できる

### 実用判断の目安

最低成功の次に、以下を記録します。

- time to first audio
- total synthesis time
- 実行中のメモリ使用量
- 連続 5 回実行時の安定性
- 熱やスロットリングの兆候

## 検証順序

検証は次の順番で行います。

1. `C++ arm64 バイナリ` で 1 文合成
2. `C++` で複数文、長文、連続実行
3. 必要なら `Python` 経路を試す
4. その後に `Stack-chan` 連携を考える

最初に `Python` を選ばない理由:

- README 上、Python 推論は `Python 3.11+` 前提
- `Module LLM` は `Ubuntu 22.04` ベースで、追加整備なしでは Python バージョンがずれる可能性がある
- C++ バイナリの方が依存の数が少ない

## 検証パス A: C++ バイナリ

### 目的

最短で「日本語 1 文が CPU で合成できるか」を確認する。

### 必要なもの

- `piper-plus` Linux arm64 バイナリ
- 日本語モデル `.onnx`
- 日本語モデル設定 `.onnx.json`
- `adb`, `ssh`, または `uart` でのログイン手段
- `curl` または `wget`
- `tar`

### 理由

C++ 側には OpenJTalk 辞書の自動ダウンロード実装があります。

- `curl` または `wget` を使用
- `tar` で展開
- 環境変数で無効化しない限り自動取得する

関連ファイル:

- [src/cpp/openjtalk_dictionary_manager.c](../../../src/cpp/openjtalk_dictionary_manager.c)

### 手順

1. `Module LLM` にログインする
2. `uname -m` と `cat /etc/os-release` で `arm64 Ubuntu` 相当か確認する
3. `curl`, `wget`, `tar` が使えるか確認する
4. `piper-plus` arm64 バイナリと日本語モデルを転送する
5. 小さい日本語文を 1 文だけ合成する
6. 生成 WAV の存在とサイズを確認する
7. 標準出力/標準エラーのログを保存する

### 推奨テスト文

まずは短文から始めます。

```text
こんにちは
```

次に中程度の文を試します。

```text
こんにちは、今日は良い天気ですね。
```

### 記録すべき項目

- 起動に成功したか
- OpenJTalk 辞書の自動取得が成功したか
- 合成完了までの時間
- 生成 WAV サイズ
- 実行中のピーク RSS
- エラー内容

## 検証パス B: Python 推論

### 目的

将来的な API 化やアプリ内統合を見越して、Python 推論が成立するか確認する。

### このパスを後回しにする理由

- `piper-plus` README では Python 側に `Python 3.11+` を要求
- `Module LLM` の Ubuntu では追加整備が必要な可能性がある
- `onnxruntime` の Python 依存解決が C++ より重い

関連ファイル:

- [README.md](../../../README.md)
- [src/python_run/piper/voice.py](../../../src/python_run/piper/voice.py)

### Python 経路で確認する項目

1. `python3 --version`
2. `venv` の作成可否
3. `onnxruntime` の導入可否
4. `piper-tts-plus` の導入可否
5. 日本語文 1 文の合成可否

### このパスの判定

- `C++` が動き、`Python` だけ詰まる: まずは C++ で進める
- `C++` も `Python` も詰まる: `Module LLM CPU` の優先度を下げる

## 速度検証

### 測るべき指標

- 起動時間
- time to first audio
- total synthesis time
- 生成音声秒数
- `RTF = synthesis_time / audio_duration`

### 最低限の解釈

- `RTF < 1.0`: 実時間未満。かなり有望
- `RTF 1.0 - 3.0`: 要用途判断
- `RTF > 3.0`: `Stack-chan` 用途では厳しい可能性が高い

この閾値は一般的な運用目安であり、`piper-plus` や `Module LLM` の公式保証ではありません。

## メモリ検証

### 注目点

`Module LLM` は `4GB LPDDR4` 搭載ですが、CPU 実行で自由に使えるシステムメモリは `1GB` と見なすべきです。

そのため、次を必ず確認します。

- モデル読み込み直後のメモリ
- 合成中のメモリ増加
- 長文時の増加量
- 連続実行でメモリが戻るか

### 危険信号

- モデルロード時点で極端にメモリ逼迫する
- 2 回目以降で不安定になる
- 長文で `OOM Killer` 相当の挙動を示す

## Stack-chan 連携は後回しにする

この段階では、まだ `Stack-chan` との接続実装に入らない方がよいです。

理由:

- まず `Module LLM CPU` 上で `piper-plus` が成立するかを単独で確認すべき
- 連携を先に始めると、失敗原因が `TTS 側` か `通信側` か切り分けにくくなる

したがって順番は:

1. `Module LLM` 単体で `piper-plus`
2. 性能測定
3. その後に `Stack-chan` 連携

です。

## 途中で打ち切る条件

次のどれかが起きたら、`CPU 実行を本命にし続けるべきではない` と判断します。

1. 日本語 1 文が安定して最後まで合成できない
2. モデルロードで継続的にメモリ不足が出る
3. `RTF` が著しく悪い
4. 連続実行で安定しない

この場合の次の選択肢は:

- `NPU 変換` を研究テーマとして扱う
- `CM4/CM4Stack` のような別 Linux モジュールへ切り替える

## いまの推奨アクション

最初にやるべきことは次の通りです。

1. `Module LLM` にログインできる状態を作る
2. `piper-plus` arm64 C++ バイナリを配置する
3. 日本語モデルを配置する
4. 短文 1 文を合成する
5. 速度とメモリを記録する

この 5 ステップが、`Module LLM CPU ルート` の最小検証です。

## 参考資料

- <https://docs.m5stack.com/en/module/Module-llm>
- <https://docs.m5stack.com/en/stackflow/module_llm/config>
- <https://docs.m5stack.com/en/stackflow/module_llm/software>
- [README.md](../../../README.md)
- [docs/guides/japanese/japanese-usage.md](../japanese/japanese-usage.md)
- [src/cpp/piper.cpp](../../../src/cpp/piper.cpp)
- [src/python_run/piper/voice.py](../../../src/python_run/piper/voice.py)
- [src/cpp/openjtalk_dictionary_manager.c](../../../src/cpp/openjtalk_dictionary_manager.c)
