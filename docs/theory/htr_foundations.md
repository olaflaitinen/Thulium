# HTR Foundations

Background on handwriting text recognition.

## What is HTR?

**Handwriting Text Recognition (HTR)** is the task of converting
handwritten text images into machine-readable text. Unlike printed OCR,
HTR must handle significant variability in writing styles.

## HTR Pipeline

```mermaid
graph LR
    Image --> Pre(Preprocessing)
    Pre --> Feat(Feature Extraction)
    Feat --> Seq(Sequence Modeling)
    Seq --> Dec(Decoding)
    Dec --> Text(Text)
```

### 1. Preprocessing

- **Binarization** — Convert to binary (ink vs background)
- **Deskewing** — Correct rotation
- **Normalization** — Standardize height/width

### 2. Feature Extraction

CNNs or Vision Transformers extract visual features:

```mermaid
graph LR
    Input[Image<br>H x W x C] --> Backbone(Backbone)
    Backbone --> Features[Features<br>H' x W' x D]
```

### 3. Sequence Modeling

RNNs or Transformers model temporal dependencies:

```mermaid
graph LR
    Features --> Seq[BiLSTM / Transformer]
    Seq --> Out[Sequence<br>T x D]
```

### 4. Decoding

CTC or attention-based decoding produces text:

```mermaid
graph LR
    Seq(Sequence) --> Decoder(Decoder)
    Decoder --> Out(Characters)
```

## CTC (Connectionist Temporal Classification)

CTC enables training without character-level alignment.

### Loss Function

$$
L_{CTC} = -\log P(y | x) = -\log \sum_{\pi \in B^{-1}(y)} P(\pi | x)
$$

Where:
- $y$ = target label sequence
- $x$ = input sequence
- $\pi$ = all paths that decode to $y$
- $B^{-1}$ = inverse of the many-to-one mapping

### Decoding

**Greedy:** Take argmax at each timestep.

**Beam Search:** Maintain top-k hypotheses.

## Attention Mechanism

Attention-based decoders learn explicit alignments:

$$
\alpha_{t,i} = \text{softmax}(e_{t,i})
$$

$$
c_t = \sum_i \alpha_{t,i} h_i
$$

Where:
- $\alpha_{t,i}$ = attention weight for output $t$, input $i$
- $c_t$ = context vector
- $h_i$ = encoder hidden states

## Language Models

Language models improve accuracy by scoring hypotheses:

$$
\text{Score}(y) = \log P_{HTR}(y|x) + \alpha \cdot \log P_{LM}(y)
$$

### Types

- **N-gram** — Statistical, fast, limited context
- **Neural** — RNN/Transformer, better context, slower

## Evaluation Metrics

| Metric | Level | Formula |
|--------|-------|---------|
| CER | Character | $(S + D + I) / N$ |
| WER | Word | $(S_w + D_w + I_w) / N_w$ |
| SER | Sequence | 1 if error, 0 otherwise |

## Challenges

| Challenge | Description |
|-----------|-------------|
| Style Variation | Different writers have unique styles |
| Historical Scripts | Old documents use archaic letterforms |
| Low Quality | Faded ink, damaged paper |
| Mixed Scripts | Documents mixing languages |

## Further Reading

- [Graves et al., 2006](https://dl.acm.org/doi/10.1145/1143844.1143891) — CTC
- [Bahdanau et al., 2015](https://arxiv.org/abs/1409.0473) — Attention
- [Shi et al., 2017](https://arxiv.org/abs/1507.05717) — CRNN for OCR

## See Also

- [Architecture](../architecture.md) — Thulium architecture
- [Model Zoo](../models/model_zoo.md) — Pretrained models
