# Zero-Shot TTS ライセンス監査

**監査日**: 2026-03-08
**目的**: Zero-Shot TTS機能で使用する全外部依存がGPL-freeであることの確認

## 監査結果サマリー

| パッケージ | ライセンス | GPL-free | 用途 |
|-----------|-----------|----------|------|
| CAM++ (3D-Speaker) | Apache-2.0 | Yes | Speaker Encoderモデル |
| WeSpeaker (ツールキット) | Apache-2.0 | Yes | 評価用cross-encoder (コード部分) |
| WeSpeaker (事前学習済みモデル) | CC-BY-4.0 | Yes | 評価用cross-encoder (モデル重み) |
| onnxruntime | MIT | Yes | ONNX推論ランタイム |
| torchaudio | BSD-2-Clause | Yes | 音声処理・Fbank抽出 |
| soundfile (Pythonパッケージ) | BSD-3-Clause | Yes | WAV読み書き |
| soundfile (libsndfile依存) | LGPL-2.1 | 注意事項あり | soundfileの内部依存ライブラリ |
| g2p-en | Apache-2.0 | Yes | 英語G2P (既存使用、参考) |

## 結論

Zero-Shot TTS機能で使用する全パッケージは**GPL-free要件を満たす**。

ただし、`soundfile`が内部的に依存する`libsndfile`はLGPL-2.1ライセンスである点に注意が必要。LGPLはGPLとは異なり、動的リンク（共有ライブラリ経由の利用）であれば独自ライセンスのソフトウェアとの組み合わせが許可される。`soundfile`のPyPIパッケージはlibsndfileを共有ライブラリとしてバンドルしており、動的リンクの要件を満たしているため、実用上の問題はない。

全体として、GPL系ライセンス（GPL-2.0、GPL-3.0）に該当するパッケージは存在せず、商用利用を含むライセンス互換性に問題はない。

---

## 各パッケージ詳細

### 1. CAM++ (3D-Speaker)

- **リポジトリ**: https://github.com/modelscope/3D-Speaker
- **ModelScope**: `iic/speech_campplus_sv_zh-cn_16k-common`
- **ライセンス**: Apache License 2.0
- **確認ソース**: GitHubリポジトリのLICENSEファイルおよびREADME
- **用途**: Zero-Shot TTSのSpeaker Encoder（話者埋め込み抽出）
- **備考**: 3D-SpeakerはAlibaba/ModelScopeが開発するオープンソースツールキット。コード・モデル重みともにApache-2.0で提供されている。サードパーティコンポーネント（Speechbrain、WeSpeaker、D-TDNN、DINO、Vicreg等）を含むが、いずれもGPL-freeライセンスである。

### 2. WeSpeaker

- **リポジトリ**: https://github.com/wenet-e2e/wespeaker
- **HuggingFace**: `Wespeaker/wespeaker-voxceleb-resnet293-LM`
- **ライセンス (ツールキット)**: Apache License 2.0
- **ライセンス (事前学習済みモデル)**: CC-BY-4.0 (Creative Commons Attribution 4.0)
- **確認ソース**: GitHubリポジトリのLICENSEファイル、HuggingFaceモデルカード
- **用途**: 話者類似度評価用のcross-encoder（開発時の評価指標計算）
- **備考**: ツールキット自体はApache-2.0。事前学習済みモデルはVoxCelebデータセットのライセンスに従いCC-BY-4.0。CC-BY-4.0は帰属表示が必要だが、GPL互換性の問題はない。本モデルは評価用途のみであり、最終成果物には含まれない。

### 3. onnxruntime

- **リポジトリ**: https://github.com/microsoft/onnxruntime
- **PyPI**: https://pypi.org/project/onnxruntime/
- **ライセンス**: MIT License
- **確認ソース**: GitHubリポジトリのLICENSEファイル、PyPIパッケージメタデータ
- **用途**: ONNX形式モデルの推論ランタイム
- **備考**: Microsoft開発。最新バージョン1.24.3 (2026-03-05リリース)。MITライセンスは最も寛容なライセンスの一つであり、制限なく商用利用可能。なお、ONNX仕様自体はApache-2.0ライセンスであるが、これもGPL-freeである。

### 4. torchaudio

- **リポジトリ**: https://github.com/pytorch/audio
- **PyPI**: https://pypi.org/project/torchaudio/
- **ライセンス**: BSD-2-Clause License
- **確認ソース**: GitHubリポジトリのLICENSEファイル、PyPIパッケージメタデータ
- **用途**: 音声処理、Fbank特徴量抽出
- **備考**: PyTorchエコシステムの公式音声処理ライブラリ。BSD-2-Clauseは非常に寛容なライセンスであり、帰属表示のみで商用利用可能。PyTorch本体もBSD-3-Clauseライセンスであり、一貫してGPL-freeである。

### 5. soundfile

- **リポジトリ**: https://github.com/bastibe/python-soundfile
- **PyPI**: https://pypi.org/project/soundfile/
- **ライセンス (Pythonパッケージ)**: BSD-3-Clause License
- **ライセンス (libsndfile依存)**: LGPL-2.1
- **確認ソース**: GitHubリポジトリのLICENSEファイル、PyPIパッケージメタデータ、libsndfile FAQ
- **用途**: WAVファイルの読み書き
- **備考**: `soundfile` Python パッケージ自体はBSD-3-Clauseだが、内部的にlibsndfile（LGPL-2.1）に依存する。PyPIパッケージはlibsndfileを共有ライブラリ（.so/.dll/.dylib）としてバンドルしており、動的リンクの形態をとるため、LGPL-2.1の条件下でも独自ライセンスのソフトウェアとの組み合わせが許可される。静的リンクは行われていないため、GPL汚染の懸念はない。ただし、LGPL準拠として以下が推奨される:
  - ドキュメントにlibsndfileへの依存とそのライセンス（LGPL-2.1）を明記する
  - LGPLライセンス全文へのリンクを記載する

### 6. g2p-en (参考)

- **リポジトリ**: https://github.com/Kyubyong/g2p
- **PyPI**: https://pypi.org/project/g2p-en/
- **ライセンス**: Apache License 2.0
- **確認ソース**: PyPIパッケージメタデータ
- **用途**: 英語Grapheme-to-Phoneme変換（既存機能で使用中）
- **備考**: espeak-ng (GPL) の代替として既にPiper-plusで採用済み。Apache-2.0であり、GPL-free要件を満たしている。

---

## ライセンス分類まとめ

### Permissive (寛容型) ライセンス -- 制限最小

| ライセンス | 該当パッケージ | 主な条件 |
|-----------|--------------|---------|
| MIT | onnxruntime | 著作権表示の保持 |
| BSD-2-Clause | torchaudio | 著作権表示の保持 |
| BSD-3-Clause | soundfile | 著作権表示の保持、名前の無断使用禁止 |
| Apache-2.0 | CAM++, WeSpeaker (toolkit), g2p-en | 著作権表示、変更点の明記、特許ライセンス付与 |

### Creative Commons ライセンス

| ライセンス | 該当パッケージ | 主な条件 |
|-----------|--------------|---------|
| CC-BY-4.0 | WeSpeaker (事前学習済みモデル) | 帰属表示 |

### Weak Copyleft (弱コピーレフト) ライセンス -- 動的リンクで回避可能

| ライセンス | 該当パッケージ | 主な条件 |
|-----------|--------------|---------|
| LGPL-2.1 | libsndfile (soundfileの内部依存) | 動的リンクなら独自ライセンスと共存可能 |

### GPL (コピーレフト) ライセンス -- 該当なし

該当パッケージなし。

---

## 推奨アクション

1. **libsndfileのLGPL-2.1表記**: プロジェクトのライセンス関連ドキュメントに、soundfileがlibsndfile (LGPL-2.1) に依存する旨を記載する
2. **WeSpeakerモデルのCC-BY-4.0帰属表示**: 評価結果を公開する場合、WeSpeakerモデルへの帰属表示を含める
3. **Apache-2.0のNOTICEファイル確認**: CAM++、WeSpeaker toolkit、g2p-enのNOTICEファイルが存在する場合、配布時にその内容を保持する

---

## 調査情報源

- [3D-Speaker GitHub](https://github.com/modelscope/3D-Speaker)
- [WeSpeaker GitHub](https://github.com/wenet-e2e/wespeaker)
- [WeSpeaker HuggingFace](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet293-LM)
- [onnxruntime GitHub LICENSE](https://github.com/microsoft/onnxruntime/blob/main/LICENSE)
- [onnxruntime PyPI](https://pypi.org/project/onnxruntime/)
- [torchaudio GitHub](https://github.com/pytorch/audio)
- [torchaudio PyPI](https://pypi.org/project/torchaudio/)
- [soundfile GitHub](https://github.com/bastibe/python-soundfile)
- [soundfile PyPI](https://pypi.org/project/soundfile/)
- [libsndfile FAQ (ライセンス)](https://libsndfile.github.io/libsndfile/FAQ.html)
- [g2p-en PyPI](https://pypi.org/project/g2p-en/)
