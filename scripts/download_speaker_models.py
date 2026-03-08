#!/usr/bin/env python3
"""Zero-Shot TTS用Speaker Encoderモデルのダウンロードスクリプト

CAM++ および WeSpeaker ResNet293 の ONNX モデルをダウンロードします。

Usage:
    uv run python scripts/download_speaker_models.py --output-dir models/speaker_encoders
    uv run python scripts/download_speaker_models.py --model campplus --output-dir models/speaker_encoders
    uv run python scripts/download_speaker_models.py --model wespeaker --output-dir models/speaker_encoders
    uv run python scripts/download_speaker_models.py --model campplus-cosyvoice --output-dir models/speaker_encoders

Models:
    campplus          : CAM++ (3D-Speaker, zh-cn, 16k, 28MB)
                        Speaker Embedding抽出用。192次元embedding出力。
                        ソース: sherpa-onnx GitHub Releases (k2-fsa/sherpa-onnx)

    campplus-en       : CAM++ (3D-Speaker, en-voxceleb, 16k, 29MB)
                        英語VoxCeleb学習済み。192次元embedding出力。
                        ソース: sherpa-onnx GitHub Releases (k2-fsa/sherpa-onnx)

    campplus-bilingual: CAM++ (3D-Speaker, zh-en advanced, 16k, 28MB)
                        中英バイリンガル版。192次元embedding出力。
                        ソース: sherpa-onnx GitHub Releases (k2-fsa/sherpa-onnx)

    campplus-cosyvoice: CAM++ (CosyVoice-300M同梱版, 28MB)
                        CosyVoiceで使用されるCAM++。同一アーキテクチャ。
                        ソース: HuggingFace (model-scope/CosyVoice-300M)

    wespeaker         : WeSpeaker ResNet293-LM (VoxCeleb2, en, 114MB)
                        評価用cross-encoder。256次元embedding出力。
                        ソース: sherpa-onnx GitHub Releases (k2-fsa/sherpa-onnx)
                        オリジナル: HuggingFace (Wespeaker/wespeaker-voxceleb-resnet293-LM)
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelInfo:
    """ダウンロード対象モデルの定義"""

    key: str
    filename: str
    url: str
    expected_size: int  # bytes (概算。10%マージンでチェック)
    description: str
    sha256: str | None = None  # 既知の場合のみ


# sherpa-onnx GitHub Releases から取得可能なモデル
# https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
# NOTE: "recongition" is a typo in the official upstream tag name
_SHERPA_BASE = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models"
)

# HuggingFace から取得可能なモデル
_HF_COSYVOICE = "https://huggingface.co/model-scope/CosyVoice-300M/resolve/main"

MODELS: dict[str, ModelInfo] = {
    "campplus": ModelInfo(
        key="campplus",
        filename="3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx",
        url=f"{_SHERPA_BASE}/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx",
        expected_size=28_281_138,
        description="CAM++ (3D-Speaker, zh-cn, 16k) - 192dim speaker embedding",
    ),
    "campplus-en": ModelInfo(
        key="campplus-en",
        filename="3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
        url=f"{_SHERPA_BASE}/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
        expected_size=29_596_978,
        description="CAM++ (3D-Speaker, en-voxceleb, 16k) - 192dim speaker embedding",
    ),
    "campplus-bilingual": ModelInfo(
        key="campplus-bilingual",
        filename="3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
        url=f"{_SHERPA_BASE}/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
        expected_size=28_281_164,
        description="CAM++ (3D-Speaker, zh-en advanced, 16k) - 192dim bilingual",
    ),
    "campplus-cosyvoice": ModelInfo(
        key="campplus-cosyvoice",
        filename="campplus.onnx",
        url=f"{_HF_COSYVOICE}/campplus.onnx",
        expected_size=28_300_000,  # ~28.3MB
        description="CAM++ (CosyVoice-300M) - 192dim speaker embedding",
        sha256="a6ac6a63997761ae2997373e2ee1c47040854b4b759ea41ec48e4e42df0f4d73",
    ),
    "wespeaker": ModelInfo(
        key="wespeaker",
        filename="wespeaker_en_voxceleb_resnet293_LM.onnx",
        url=f"{_SHERPA_BASE}/wespeaker_en_voxceleb_resnet293_LM.onnx",
        expected_size=114_336_527,
        description="WeSpeaker ResNet293-LM (VoxCeleb2) - 256dim speaker embedding",
    ),
}

# デフォルトでダウンロードするモデル (Zero-Shot TTS推奨セット)
DEFAULT_MODELS = ["campplus", "wespeaker"]


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _format_size(size_bytes: int) -> str:
    """バイト数を人間が読みやすい形式に変換"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """urllib.request.urlretrieve 用プログレスコールバック"""
    if total_size > 0:
        downloaded = block_num * block_size
        percent = min(100.0, downloaded / total_size * 100)
        bar_len = 40
        filled = int(bar_len * percent / 100)
        bar = "=" * filled + "-" * (bar_len - filled)
        sys.stdout.write(
            f"\r  [{bar}] {percent:5.1f}%  "
            f"{_format_size(downloaded)} / {_format_size(total_size)}"
        )
        sys.stdout.flush()
    else:
        downloaded = block_num * block_size
        sys.stdout.write(f"\r  Downloaded: {_format_size(downloaded)}")
        sys.stdout.flush()


def _verify_size(filepath: Path, expected: int, margin: float = 0.10) -> bool:
    """ファイルサイズが期待値の margin (10%) 以内か検証"""
    actual = filepath.stat().st_size
    lower = int(expected * (1 - margin))
    upper = int(expected * (1 + margin))
    return lower <= actual <= upper


def _verify_sha256(filepath: Path, expected_hash: str) -> bool:
    """SHA256ハッシュを検証"""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest() == expected_hash


def download_model(
    model: ModelInfo,
    output_dir: Path,
    force: bool = False,
) -> bool:
    """モデルを1つダウンロードする。

    Returns:
        True: ダウンロード成功 or スキップ (既存)
        False: ダウンロード失敗
    """
    dest = output_dir / model.filename

    # 既存ファイルチェック
    if dest.exists() and not force:
        actual_size = dest.stat().st_size
        print(f"[SKIP] {model.filename} ({_format_size(actual_size)}) - already exists")
        print("       Use --force to re-download")
        return True

    print(f"[DOWN] {model.filename}")
    print(f"       {model.description}")
    print(f"       URL: {model.url}")

    # ダウンロード先ディレクトリ作成
    output_dir.mkdir(parents=True, exist_ok=True)

    # 一時ファイルに書き込み、成功時にリネーム
    tmp_dest = dest.with_suffix(dest.suffix + ".tmp")
    try:
        urllib.request.urlretrieve(model.url, str(tmp_dest), _progress_hook)
        print()  # プログレスバーの改行
    except urllib.error.URLError as e:
        print(f"\n[FAIL] Download failed: {e}")
        if tmp_dest.exists():
            tmp_dest.unlink()
        return False
    except KeyboardInterrupt:
        print("\n[CANCEL] Download interrupted")
        if tmp_dest.exists():
            tmp_dest.unlink()
        return False

    # ファイルサイズ検証
    actual_size = tmp_dest.stat().st_size
    if not _verify_size(tmp_dest, model.expected_size):
        print(
            f"[WARN] Size mismatch: expected ~{_format_size(model.expected_size)}, "
            f"got {_format_size(actual_size)}"
        )
        print("       File may be corrupted. Keeping file for manual inspection.")
        # サイズが大幅に異なる場合でもファイルは残す（HTML error page等の可能性）
        if actual_size < 1_000_000 and model.expected_size > 10_000_000:
            print("[FAIL] Downloaded file is suspiciously small. Removing.")
            tmp_dest.unlink()
            return False

    # SHA256検証 (ハッシュが既知の場合)
    if model.sha256:
        print("  Verifying SHA256...", end="")
        if _verify_sha256(tmp_dest, model.sha256):
            print(" OK")
        else:
            print(" MISMATCH")
            print("[FAIL] SHA256 hash does not match expected value.")
            print(f"       Expected: {model.sha256}")
            tmp_dest.unlink()
            return False

    # リネーム
    tmp_dest.rename(dest)
    print(f"[OK]   {model.filename} ({_format_size(actual_size)})")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-Shot TTS用Speaker Encoderモデルのダウンロード",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available models:
  campplus           CAM++ zh-cn (28MB)  - Speaker Embedding抽出用 [default]
  campplus-en        CAM++ en-voxceleb (29MB)
  campplus-bilingual CAM++ zh-en advanced (28MB)
  campplus-cosyvoice CAM++ CosyVoice版 (28MB)
  wespeaker          WeSpeaker ResNet293-LM (114MB) - 評価用 [default]

Examples:
  uv run %(prog)s --output-dir models/speaker_encoders
  uv run %(prog)s --model campplus --output-dir models/speaker_encoders
  uv run %(prog)s --model campplus wespeaker --output-dir models/speaker_encoders
  uv run %(prog)s --model all --output-dir models/speaker_encoders
  uv run %(prog)s --list
""",
    )
    parser.add_argument(
        "--model",
        nargs="*",
        default=None,
        help=(
            "ダウンロードするモデル名 (複数指定可)。"
            "未指定時はデフォルトセット (campplus, wespeaker)。"
            "'all' で全モデル。"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/speaker_encoders"),
        help="ダウンロード先ディレクトリ (default: models/speaker_encoders)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="既存ファイルを上書きダウンロード",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="利用可能なモデル一覧を表示して終了",
    )
    return parser.parse_args()


def list_models() -> None:
    """利用可能なモデル一覧を表示"""
    print("Available models:")
    print("-" * 80)
    for key, m in MODELS.items():
        default_mark = " [default]" if key in DEFAULT_MODELS else ""
        print(
            f"  {key:<22s} {_format_size(m.expected_size):>8s}  {m.description}{default_mark}"
        )
    print("-" * 80)
    print(f"\nDefault set: {', '.join(DEFAULT_MODELS)}")


def main() -> None:
    args = parse_args()

    if args.list:
        list_models()
        return

    # ダウンロード対象モデルの決定
    if args.model is None:
        model_keys = DEFAULT_MODELS
    elif "all" in args.model:
        model_keys = list(MODELS.keys())
    else:
        model_keys = args.model

    # モデルキーの検証
    invalid = [k for k in model_keys if k not in MODELS]
    if invalid:
        print(f"Error: Unknown model(s): {', '.join(invalid)}")
        print(f"Available: {', '.join(MODELS.keys())}")
        sys.exit(1)

    # 出力ディレクトリ (相対パスの場合はCWD基準)
    output_dir = args.output_dir.resolve()

    print("=" * 60)
    print("Zero-Shot TTS Speaker Encoder Model Downloader")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Models to download: {', '.join(model_keys)}")
    print(f"Force overwrite: {args.force}")
    print()

    # ダウンロード実行
    results: dict[str, bool] = {}
    for key in model_keys:
        model = MODELS[key]
        print("-" * 60)
        ok = download_model(model, output_dir, force=args.force)
        results[key] = ok
        print()

    # サマリー
    print("=" * 60)
    print("Summary:")
    success = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    for key, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  [{status:>6s}] {key}")

    print(f"\n  Total: {success} succeeded, {failed} failed")

    if failed:
        print("\nSome downloads failed. Please check your network and retry.")
        sys.exit(1)

    print(f"\nModels saved to: {output_dir}")


if __name__ == "__main__":
    main()
