# GPU Configuration Guide

> **注意**: GPU デバイス選択は C++ CLI (`piper` バイナリ) のみ対応。Python CLI では未対応。

## GPU Device Selection (C++ CLI)

`--gpu-device-id` パラメータで特定のGPUを選択できます:

```bash
# デフォルトGPU (device 0)
echo "Hello world" | piper --model en_US-lessac-medium.onnx --use-cuda -f output.wav

# GPU device 1 を使用
echo "Hello world" | piper --model en_US-lessac-medium.onnx --use-cuda --gpu-device-id 1 -f output.wav
```

### 環境変数

`PIPER_GPU_DEVICE_ID` で永続的に設定:

```bash
export PIPER_GPU_DEVICE_ID=1
```

**優先順位**: CLI引数 > 環境変数 > デフォルト (0)

## GPUの確認

```bash
nvidia-smi -L
```

## マルチGPU並列処理

```bash
# GPU 0 で処理
piper --model model.onnx --use-cuda --gpu-device-id 0 --input-file text1.txt -f output1.wav &

# GPU 1 で処理
piper --model model.onnx --use-cuda --gpu-device-id 1 --input-file text2.txt -f output2.wav &

wait
```

## Docker with GPU

```bash
docker run --gpus '"device=0"' -e PIPER_GPU_DEVICE_ID=0 piper-gpu
docker run --gpus '"device=1"' -e PIPER_GPU_DEVICE_ID=1 piper-gpu
```

## トラブルシューティング

| エラー | 対処法 |
|--------|--------|
| "Invalid GPU device ID" | `nvidia-smi -L` で利用可能なGPUを確認 |
| "CUDA out of memory" | 小さいモデルを使用、他のGPUアプリを終了 |
| "CUDA not available" | NVIDIAドライバとCUDAのインストールを確認 |

## 関連ドキュメント

- [CLI Enhancements](cli-enhancements.md) - CLIの各種機能
- [Multi-GPU Training](../guides/training/multi-gpu-training.md) - 学習時のマルチGPU設定
