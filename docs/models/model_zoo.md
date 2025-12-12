# Model Zoo

Pretrained models for handwriting recognition.

## Available Models

| Model | Parameters | CER | WER | Languages | Download |
|-------|------------|-----|-----|-----------|----------|
| `thulium-tiny` | 5M | 5.2% | 14.1% | Latin | [Download](https://github.com/olaflaitinen/Thulium/releases/download/v1.2.1/thulium-tiny.pt) |
| `thulium-base` | 25M | 3.8% | 10.2% | Latin | [Download](https://github.com/olaflaitinen/Thulium/releases/download/v1.2.1/thulium-base.pt) |
| `thulium-large` | 100M | 2.9% | 7.8% | Latin | [Download](https://github.com/olaflaitinen/Thulium/releases/download/v1.2.1/thulium-large.pt) |
| `thulium-multilingual` | 150M | 4.1% | 11.5% | 52+ | [Download](https://github.com/olaflaitinen/Thulium/releases/download/v1.2.1/thulium-multilingual.pt) |

*Metrics measured on IAM test set, batch size 1, PyTorch 2.0+*
> **Note**: Models are hosted as GitHub Release assets. You can also load them directly via `torch.hub`.

## Model Architectures

### thulium-tiny

Lightweight model for edge deployment.

| Component | Configuration |
|-----------|---------------|
| Backbone | ResNet-18 (reduced) |
| Head | BiLSTM (128 hidden) |
| Decoder | CTC |
| Input | 32×256 grayscale |

```python
from thulium import HTRPipeline

pipeline = HTRPipeline.from_pretrained("thulium-tiny")
```

### thulium-base

Balanced accuracy and speed.

| Component | Configuration |
|-----------|---------------|
| Backbone | ResNet-34 |
| Head | BiLSTM (256 hidden, 2 layers) |
| Decoder | CTC + 3-gram LM |
| Input | 64×512 grayscale |

```python
pipeline = HTRPipeline.from_pretrained("thulium-base")
```

### thulium-large

Maximum accuracy for server deployment.

| Component | Configuration |
|-----------|---------------|
| Backbone | ViT-Base (patch 8) |
| Head | Transformer (6 layers) |
| Decoder | CTC + Neural LM |
| Input | 64×768 grayscale |

```python
pipeline = HTRPipeline.from_pretrained("thulium-large", device="cuda")
```

### thulium-multilingual

52+ languages with script detection.

| Component | Configuration |
|-----------|---------------|
| Backbone | ViT-Large |
| Head | Conformer (8 layers) |
| Decoder | Attention + Neural LM |
| Scripts | Latin, Cyrillic, Arabic, CJK, ... |

## Loading Models

### From Hub

```python
from thulium import HTRPipeline

# Download and cache automatically
pipeline = HTRPipeline.from_pretrained("thulium-base")
```

### From Local Path

```python
pipeline = HTRPipeline.from_pretrained("./models/my_model/")
```

### With Custom Config

```python
pipeline = HTRPipeline.from_config("config/custom.yaml")
```

## ONNX Export

Export for production deployment:

```python
from thulium.export import export_onnx

export_onnx(
    model=pipeline.model,
    output_path="model.onnx",
    input_shape=(1, 1, 64, 256),
    opset_version=17,
)
```

## Memory Requirements

| Model | GPU Memory | CPU RAM |
|-------|------------|---------|
| thulium-tiny | 512 MB | 256 MB |
| thulium-base | 1.5 GB | 512 MB |
| thulium-large | 4 GB | 2 GB |

## See Also

- [Language Support](language_support.md) — Supported languages
- [Training Guide](../training/training_guide.md) — Train custom models
- [Benchmarks](../evaluation/benchmarks.md) — Performance details
