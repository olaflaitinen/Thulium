# Thulium

**State-of-the-art multilingual handwriting text recognition.**

[![PyPI](https://img.shields.io/pypi/v/thulium-htr)](https://pypi.org/project/thulium-htr/)
[![Python](https://img.shields.io/pypi/pyversions/thulium-htr)](https://pypi.org/project/thulium-htr/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen)](docs/)

Thulium is a production-ready Python library for offline handwritten text
recognition (HTR) supporting 52+ languages across Latin, Cyrillic, Greek,
Arabic, Hebrew, Devanagari, Chinese, Japanese, Korean, and Georgian scripts.

## Features

- **52+ Languages** — Comprehensive multilingual support with script-aware processing
- **Production Ready** — Optimized inference with ONNX export and mixed precision
- **State-of-the-Art** — CNN/ViT backbones with Transformer/LSTM sequence heads
- **Explainable AI** — Attention visualization, saliency maps, and confidence analysis
- **Flexible Decoding** — CTC beam search with n-gram and neural language models

## Installation

```bash
pip install thulium-htr
```

For GPU acceleration:

```bash
pip install thulium-htr[gpu]
```

## Quick Start

```python
from thulium import recognize_image

# Single image recognition
result = recognize_image("document.png", language="en")
print(result.text)

# Batch recognition with confidence scores
from thulium import HTRPipeline

pipeline = HTRPipeline.from_pretrained("thulium-base-multilingual")
results = pipeline.recognize_batch(images, languages=["en", "de", "fr"])

for r in results:
    print(f"{r.text} (confidence: {r.confidence:.2%})")
```

## Supported Languages

<details>
<summary>52+ languages across 10 scripts (click to expand)</summary>

| Region | Languages |
|--------|-----------|
| Western Europe | English, German, French, Spanish, Italian, Portuguese, Dutch |
| Scandinavia | Swedish, Norwegian, Danish, Finnish, Icelandic |
| Eastern Europe | Polish, Czech, Hungarian, Romanian, Bulgarian, Ukrainian, Russian |
| Baltic | Lithuanian, Latvian, Estonian |
| Caucasus | Georgian, Armenian, Azerbaijani |
| Middle East | Arabic, Hebrew, Persian, Turkish |
| South Asia | Hindi, Bengali, Tamil, Telugu, Urdu |
| East Asia | Chinese, Japanese, Korean |

</details>

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/getting_started.md) | Installation and first steps |
| [API Reference](docs/api/reference.md) | Complete API documentation |
| [Model Zoo](docs/models/model_zoo.md) | Pretrained model catalog |
| [Training Guide](docs/training/training_guide.md) | Train custom models |
| [Architecture](docs/architecture.md) | System design overview |

## Performance

Benchmarks on IAM Handwriting Database:

| Model | CER | WER | Latency |
|-------|-----|-----|---------|
| thulium-tiny | 5.2% | 14.1% | 12ms |
| thulium-base | 3.8% | 10.2% | 28ms |
| thulium-large | 2.9% | 7.8% | 65ms |

*Measured on NVIDIA A100, batch size 1, PyTorch 2.0+*

## Citation

```bibtex
@software{thulium2025,
  title={Thulium: Multilingual Handwriting Recognition},
  author={Thulium Authors},
  year={2025},
  url={https://github.com/thulium-dev/thulium}
}
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
