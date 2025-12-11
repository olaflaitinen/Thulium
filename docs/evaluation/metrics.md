# Evaluation Metrics

This document provides formal definitions of the evaluation metrics used in Thulium for assessing handwriting recognition quality.

## Character Error Rate (CER)

Character Error Rate measures the edit distance between reference and hypothesis strings at the character level.

### Definition

```
CER = (S + D + I) / N
```

Where:
- **S** = Number of character substitutions
- **D** = Number of character deletions
- **I** = Number of character insertions
- **N** = Total number of characters in the reference

### Properties

- Range: [0, infinity) - can exceed 1.0 if insertions dominate
- Lower is better; 0.0 indicates perfect recognition
- CER = 0.0 when hypothesis exactly matches reference

### Implementation

```python
from thulium.evaluation.metrics import cer

reference = "The quick brown fox"
hypothesis = "The quich brown fax"

error_rate = cer(reference, hypothesis)
# Substitutions: 'k'->'h', 'o'->'a' = 2
# Reference length: 19
# CER = 2/19 = 0.1053
```

---

## Word Error Rate (WER)

Word Error Rate applies the same edit distance computation at the word level.

### Definition

```
WER = (S_w + D_w + I_w) / N_w
```

Where:
- **S_w** = Number of word substitutions
- **D_w** = Number of word deletions
- **I_w** = Number of word insertions
- **N_w** = Total number of words in the reference

### Properties

- Range: [0, infinity)
- More interpretable for document-level quality
- Sensitive to word boundary detection

### Implementation

```python
from thulium.evaluation.metrics import wer

reference = "The quick brown fox"
hypothesis = "The quich brown fax"

error_rate = wer(reference, hypothesis)
# Word differences: 'quick'->'quich', 'fox'->'fax' = 2
# Reference words: 4
# WER = 2/4 = 0.50
```

---

## Sequence Error Rate (SER)

Sequence Error Rate is a binary metric indicating whether the entire sequence matches exactly.

### Definition

```
SER = 1 if reference != hypothesis else 0
```

### Properties

- Range: {0, 1}
- Useful for line-level or short-form recognition
- Very strict; any difference results in error

### Implementation

```python
from thulium.evaluation.metrics import ser

reference = "Hello World"
hypothesis = "Hello World"
print(ser(reference, hypothesis))  # 0.0

hypothesis = "Hello world"
print(ser(reference, hypothesis))  # 1.0 (case difference)
```

---

## CTC Loss

The Connectionist Temporal Classification loss function enables training sequence models without explicit alignment.

### Definition

```
L_CTC = -ln p(y | x)
```

Where the probability marginalizes over all valid alignments:

```
p(y | x) = sum over all alignments pi: p(pi | x)
```

### Key Concepts

1. **Blank Token**: CTC introduces a special blank token to handle variable-length outputs
2. **Collapsing**: Repeated characters and blanks are merged to produce final output
3. **Dynamic Programming**: Efficient computation via forward-backward algorithm

### Training Considerations

- Requires input sequence longer than target sequence
- Assumes conditional independence between time steps
- Works well for line-level recognition tasks

---

## Metric Selection Guidelines

| Metric | Best For | Considerations |
| :--- | :--- | :--- |
| CER | Fine-grained accuracy | Sensitive to character-level errors |
| WER | Document quality | More interpretable for users |
| SER | Short sequences | Very strict, use with caution |

### Typical Benchmarks

| Dataset | Language | Typical CER | Typical WER |
| :--- | :--- | :--- | :--- |
| IAM | English | 3-5% | 8-12% |
| RIMES | French | 2-4% | 6-10% |
| Washington | English (historical) | 5-10% | 15-25% |

---

## Visualization

To generate training curves or comparison plots:

```python
import matplotlib.pyplot as plt

# Example: Plot CER over evaluation
epochs = [1, 2, 3, 4, 5]
cer_values = [0.45, 0.32, 0.21, 0.15, 0.12]

plt.figure(figsize=(8, 5))
plt.plot(epochs, cer_values, marker='o')
plt.xlabel('Epoch')
plt.ylabel('CER')
plt.title('Character Error Rate Over Training')
plt.grid(True)
plt.savefig('cer_curve.png')
```
