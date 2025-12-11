# Thulium Architecture

## System Overview

Thulium implements a layered architecture for handwritten text recognition, designed for modularity, extensibility, and production reliability.

---

## High-Level Architecture

```mermaid
graph TB
    subgraph Input["Input Layer"]
        I1[Document Image]
        I2[PDF Document]
        I3[Batch Images]
    end
    
    subgraph Data["Data Layer"]
        D1[Image Loader]
        D2[PDF Processor]
        D3[Transform Pipeline]
        D4[Language Profile Registry]
    end
    
    subgraph Preprocessing["Preprocessing Layer"]
        P1[Normalization]
        P2[Binarization]
        P3[Deskewing]
        P4[Noise Reduction]
    end
    
    subgraph Segmentation["Segmentation Layer"]
        S1[Layout Analysis]
        S2[Line Detection]
        S3[Word Segmentation]
    end
    
    subgraph Recognition["Recognition Layer"]
        R1[Feature Extraction]
        R2[Sequence Modeling]
        R3[Output Decoding]
    end
    
    subgraph LM["Language Modeling"]
        L1[N-gram LM]
        L2[Neural LM]
        L3[Beam Rescoring]
    end
    
    subgraph Output["Output Layer"]
        O1[Structured Result]
        O2[Confidence Scores]
        O3[Export Formats]
    end
    
    I1 --> D1
    I2 --> D2
    I3 --> D1
    D1 --> D3
    D2 --> D3
    D3 --> P1
    D4 -.-> R3
    
    P1 --> P2 --> P3 --> P4
    P4 --> S1 --> S2 --> S3
    S3 --> R1 --> R2 --> R3
    R3 --> L3
    L1 --> L3
    L2 --> L3
    L3 --> O1
    O1 --> O2 --> O3
```

---

## Module Structure

| Module | Path | Responsibility |
|:-------|:-----|:---------------|
| API | `thulium.api` | High-level recognition interface |
| Data | `thulium.data` | Data loading, transforms, language profiles |
| Models | `thulium.models` | Neural network architectures |
| Pipeline | `thulium.pipeline` | End-to-end processing orchestration |
| Evaluation | `thulium.evaluation` | Metrics, benchmarking, reporting |
| XAI | `thulium.xai` | Explainability and error analysis |
| CLI | `thulium.cli` | Command-line interface |

---

## Data Layer

### Component Hierarchy

```mermaid
classDiagram
    class DataLoader {
        +load_image(path) Image
        +load_batch(paths) List~Image~
    }
    
    class PDFProcessor {
        +extract_pages(path) List~Image~
        +get_metadata(path) Dict
    }
    
    class TransformPipeline {
        +transforms: List~Transform~
        +apply(image) Tensor
    }
    
    class LanguageProfile {
        +code: str
        +name: str
        +script: str
        +alphabet: List~str~
        +direction: str
        +model_profile: str
    }
    
    DataLoader --> TransformPipeline
    PDFProcessor --> TransformPipeline
    TransformPipeline --> LanguageProfile
```

### Language Profile Schema

Each supported language is defined by a `LanguageProfile` dataclass:

| Field | Type | Description |
|:------|:-----|:------------|
| `code` | str | ISO 639-1 code |
| `name` | str | Human-readable name |
| `script` | str | Writing system |
| `alphabet` | List[str] | Character set |
| `direction` | str | LTR or RTL |
| `region` | str | Geographic region |
| `model_profile` | str | Default model config |
| `tokenizer_type` | str | char, bpe, or word |
| `default_decoder` | str | Decoder variant |

---

## Model Layer

### Architecture Variants

```mermaid
graph LR
    subgraph Backbones
        B1[ResNet-18]
        B2[ResNet-34]
        B3[ViT-Base]
        B4[Hybrid CNN-ViT]
    end
    
    subgraph Sequence
        S1[BiLSTM]
        S2[Transformer]
        S3[Conformer]
    end
    
    subgraph Decoders
        D1[CTC Greedy]
        D2[CTC Beam]
        D3[Attention Seq2Seq]
    end
    
    B1 --> S1
    B2 --> S2
    B3 --> S2
    B4 --> S3
    
    S1 --> D1
    S1 --> D2
    S2 --> D2
    S2 --> D3
    S3 --> D2
    S3 --> D3
```

### Backbone Architectures

#### Convolutional Neural Networks

ResNet-based feature extraction with asymmetric pooling for text line images:

```
Input: (B, C, H, W)
  --> Conv Layers --> Residual Blocks --> Pooling
Output: (B, C', H', W')
```

Feature map dimensions are computed as:

```
H' = H / s_h
W' = W / s_w
```

Where s_h and s_w are the total vertical and horizontal strides.

#### Vision Transformer

Patch-based encoding with positional embeddings:

```
Input: (B, C, H, W)
  --> Patch Embedding --> Positional Encoding --> Transformer Blocks
Output: (B, N, D)
```

Patch sequence length:

```
N = (H / P_h) * (W / P_w)
```

Where P_h and P_w are patch dimensions.

### Sequence Heads

#### BiLSTM

Bidirectional LSTM for sequential feature modeling:

```mermaid
graph LR
    A[Feature Sequence] --> B[Forward LSTM]
    A --> C[Backward LSTM]
    B --> D[Concatenate]
    C --> D
    D --> E[Output Sequence]
```

#### Transformer

Self-attention based sequence modeling:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

Multi-head attention with positional encoding:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### Decoders

#### CTC Decoder

Connectionist Temporal Classification enables alignment-free training:

```
P(y|x) = sum_{pi in B^{-1}(y)} P(pi|x)
```

Where B is the many-to-one mapping that removes blanks and repeated characters.

**Beam Search** maintains top-k hypotheses with optional language model rescoring:

```
score(h) = log P(h|x) + alpha * log P_LM(h) + beta * |h|
```

#### Attention Decoder

Autoregressive sequence generation with cross-attention:

```mermaid
sequenceDiagram
    participant E as Encoder
    participant D as Decoder
    participant O as Output
    
    E->>D: Encoded Features
    loop For each step
        D->>D: Self-Attention
        D->>E: Cross-Attention
        D->>O: Predict Next Token
        O->>D: Feed Back
    end
```

---

## Pipeline Layer

### Pipeline Architecture

```mermaid
flowchart TB
    subgraph Input
        A[Raw Image]
    end
    
    subgraph Preprocessing
        B1[Resize]
        B2[Normalize]
        B3[Binarize]
    end
    
    subgraph Segmentation
        C1[Layout Analysis]
        C2[Line Extraction]
        C3[Word Segmentation]
    end
    
    subgraph Recognition
        D1[Feature Extraction]
        D2[Sequence Modeling]
        D3[Decoding]
    end
    
    subgraph Postprocessing
        E1[Unicode Normalization]
        E2[Spell Check]
        E3[Format Output]
    end
    
    A --> B1 --> B2 --> B3
    B3 --> C1 --> C2 --> C3
    C3 --> D1 --> D2 --> D3
    D3 --> E1 --> E2 --> E3
```

### Configuration System

Pipelines are fully specified through YAML configuration:

```yaml
pipeline:
  name: htr_default
  
  preprocessing:
    target_height: 64
    normalize: true
    binarize: false
    
  model:
    backbone: resnet34
    sequence_head: transformer
    decoder: ctc_beam
    
  decoding:
    beam_width: 20
    lm_alpha: 0.5
    lm_beta: 0.1
    
  language:
    profile: en
```

---

## Evaluation Layer

### Metrics Framework

```mermaid
graph LR
    subgraph Inputs
        A[References]
        B[Hypotheses]
    end
    
    subgraph Metrics
        C[CER]
        D[WER]
        E[SER]
    end
    
    subgraph Analysis
        F[Edit Operations]
        G[Confusion Matrix]
        H[Error Categories]
    end
    
    subgraph Output
        I[Benchmark Report]
    end
    
    A --> C
    B --> C
    A --> D
    B --> D
    A --> E
    B --> E
    
    C --> F
    D --> G
    E --> H
    
    F --> I
    G --> I
    H --> I
```

### Metric Definitions

**Character Error Rate (CER)**:

```
CER = (S + D + I) / N
```

Where:
- S = substitutions
- D = deletions
- I = insertions
- N = reference length

**Word Error Rate (WER)**:

```
WER = (S_w + D_w + I_w) / N_w
```

Applied at word-level tokenization.

**Cross-Language Fairness**:

```
Delta_CER = max_l(CER_l) - min_l(CER_l)
Sigma_CER = sqrt(1/L * sum_l(CER_l - mean_CER)^2)
```

---

## XAI Layer

### Explainability Components

```mermaid
graph TB
    subgraph Model
        A[Recognition Model]
    end
    
    subgraph Attention
        B1[Encoder Attention]
        B2[Decoder Attention]
        B3[Cross Attention]
    end
    
    subgraph Analysis
        C1[Attention Visualization]
        C2[Confidence Heatmap]
        C3[Error Localization]
    end
    
    subgraph Output
        D[Explainability Report]
    end
    
    A --> B1
    A --> B2
    A --> B3
    
    B1 --> C1
    B2 --> C2
    B3 --> C3
    
    C1 --> D
    C2 --> D
    C3 --> D
```

---

## Design Principles

| Principle | Implementation |
|:----------|:---------------|
| **Modularity** | Components are independently replaceable |
| **Configurability** | YAML-based specification for reproducibility |
| **Type Safety** | Comprehensive type hints throughout |
| **Testability** | Unit and integration test coverage |
| **Extensibility** | New languages require only profile definition |
| **Performance** | Optimized inference with batching support |

---

## Technology Stack

| Component | Technology |
|:----------|:-----------|
| Deep Learning | PyTorch 2.0+ |
| Image Processing | Pillow, OpenCV |
| PDF Processing | pdf2image |
| CLI | Typer |
| Configuration | PyYAML, Pydantic |
| Testing | pytest |

---

## References

1. Graves, A., et al. (2006). Connectionist temporal classification.
2. Vaswani, A., et al. (2017). Attention is all you need.
3. Dosovitskiy, A., et al. (2020). An image is worth 16x16 words.
4. Gulati, A., et al. (2020). Conformer: Convolution-augmented Transformer.
