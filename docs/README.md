# Piper Documentation

Piper Plus documentation. Guides and references for using and developing with Piper Plus.

## Getting Started
- [Windows Setup](getting-started/windows-setup.md) - Windows platform setup guide
- [Environment Variables](getting-started/environment-variables.md) - Configuration options
- [Troubleshooting](getting-started/troubleshooting.md) - Common issues and solutions

## Features
- [CLI Enhancements](features/cli-enhancements.md) - Enhanced command-line features
- [Custom Dictionary](features/custom_dictionary.md) - Custom dictionary for technical terms and proper nouns
- [Streaming Mode](features/streaming-mode.md) - Real-time streaming support
- [GPU Configuration](features/gpu-configuration.md) - GPU device selection (C++ CLI)
- [WebUI](features/webui.md) - Browser-based interface

## Guides

### Training
- [Training Guide](guides/training/training-guide.md) - General training instructions
- [Multi-GPU Training](guides/training/multi-gpu-training.md) - Training with multiple GPUs
- [Model Size Impact Analysis](guides/training/model-size-impact-analysis-ja.md) - Model size vs quality

### Japanese Language Support
- [Japanese Usage Guide](guides/japanese/japanese-usage.md) - Comprehensive Japanese TTS guide

### Optimization
- [Apple Silicon Optimization](guides/optimization/apple-silicon-optimization.md) - M1-M4 optimization

### Integration
- [Stack-chan](guides/integration/stack-chan.md) - Stack-chan validation guide

### Testing
- [Multilingual Testing](guides/testing/multilingual-testing.md) - Testing infrastructure

## Research
調査・リサーチ資料:
- [拡張機能調査・参考文献](research/extended-features-research.md) - VITS系OSS横断分析、論文・OSSリスト
- [ONNX量子化最適化](research/onnx-quantization-optimization.md) - INT8/FP16量子化の調査
- [ボコーダ置換リサーチ](research/vocoder-replacement-research.md) - HiFi-GAN代替の比較検証

## API Reference
- [Phoneme Mapping](api-reference/phoneme-mapping.md) - Phoneme reference for all languages

## Roadmap
- [Roadmap](roadmap.md) - Future development roadmap (Phase 1-4)

## Development
- [Contributing](/CONTRIBUTING.md) - Contribution guidelines
- [Changelog](/CHANGELOG.md) - Version history
- [License](/LICENSE.md) - Project license (MIT)
- [License Compliance](development/license-compliance.md) - License compliance info

## WebAssembly
Browser-based TTS implementation is in [src/wasm/openjtalk-web/](../src/wasm/openjtalk-web/).
