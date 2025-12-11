# Theoretical Foundations of Handwriting Text Recognition

This document provides the theoretical background for handwriting text recognition (HTR) as implemented in Thulium. It covers the mathematical formulations, architectural principles, and algorithmic foundations that underpin modern deep learning-based HTR systems.

---

## 1. Problem Formulation

Handwriting text recognition is a sequence transduction problem where the goal is to convert an input image containing handwritten text into a corresponding character sequence.

### 1.1 Formal Definition

Given:
- An input image `x` of shape `(H, W, C)` containing handwritten text
- A vocabulary `V` of characters (alphabet, digits, punctuation)
- A target sequence `y = (y_1, y_2, ..., y_T)` where `y_t in V`

The objective is to learn a function:

```
f: X -> Y*
```

that maps images to variable-length character sequences.

### 1.2 Challenges

HTR differs from other sequence recognition tasks due to:

1. **Variable alignment**: The correspondence between image regions and output characters is not known a priori.
2. **Intra-class variation**: The same character can have vastly different visual appearances across writers.
3. **Inter-class similarity**: Different characters may look similar (e.g., 'l' and '1', 'e' and 'c').
4. **Degradation**: Historical documents may have noise, fading, or damage.
5. **Multilingual complexity**: Different scripts have different characteristics (directionality, ligatures, diacritics).

---

## 2. Connectionist Temporal Classification (CTC)

CTC is the foundational approach for alignment-free sequence recognition, enabling end-to-end training without explicit character-level annotations.

### 2.1 Path Formulation

For an input sequence of length `T_enc` and output vocabulary of size `K`, the network produces a probability distribution at each timestep:

```
P(pi_t | x) for t = 1, ..., T_enc
```

where `pi` is a "path" including blank tokens (denoted `phi`).

### 2.2 Many-to-One Mapping

CTC defines a mapping `B` that collapses paths to label sequences by:
1. Removing consecutive duplicate characters
2. Removing blank tokens

For example: `B("a-a--b") = "aab"` and `B("aa-b") = "ab"` where `-` is blank.

### 2.3 CTC Probability

The probability of a label sequence `y` is the sum over all valid paths:

```
P(y | x) = sum_{pi in B^{-1}(y)} P(pi | x)
```

where:

```
P(pi | x) = prod_{t=1}^{T_enc} P(pi_t | x)
```

### 2.4 CTC Loss

The CTC loss is the negative log-likelihood:

```
L_CTC = -ln P(y | x)
```

This is efficiently computed using the forward-backward algorithm in O(T_enc * T_out) time.

### 2.5 CTC Assumptions

CTC makes specific assumptions:
- Output sequence is shorter than input sequence
- Monotonic alignment (output order matches input spatial order)
- Conditional independence between frames given input

---

## 3. Attention-Based Sequence-to-Sequence

Attention mechanisms overcome CTC's monotonic alignment constraint by learning flexible alignments.

### 3.1 Encoder-Decoder Architecture

The encoder processes the input image:

```
H = Encoder(x)   # H: (T_enc, D)
```

The decoder generates output autoregressively:

```
P(y_t | y_{<t}, H) = Decoder(y_{<t}, H)
```

### 3.2 Attention Mechanism

At decoding step `t`, attention computes context vector `c_t`:

```
e_{t,i} = score(s_{t-1}, h_i)
alpha_{t,i} = softmax(e_{t,:})_i
c_t = sum_i alpha_{t,i} * h_i
```

Common scoring functions:
- Dot product: `score(s, h) = s^T h`
- Additive: `score(s, h) = v^T tanh(W_s s + W_h h)`
- Scaled dot product: `score(s, h) = (s^T h) / sqrt(d)`

### 3.3 Training with Teacher Forcing

During training, the decoder receives ground truth tokens:

```
L_CE = -sum_t ln P(y_t | y_{<t}, H)
```

This is cross-entropy loss with label smoothing option.

---

## 4. Language Model Integration

Language models provide prior knowledge about likely character sequences, improving recognition accuracy.

### 4.1 Shallow Fusion

During decoding, LM scores are combined with acoustic/visual scores:

```
score(y) = log P_HTR(y | x) + alpha * log P_LM(y) + beta * |y|
```

where:
- `alpha` is the LM weight (hyperparameter)
- `beta` is the word/character insertion bonus
- `|y|` is the sequence length

### 4.2 Beam Search with LM

For CTC, beam search maintains multiple hypotheses:

1. Initialize with empty sequence
2. At each timestep, extend all beams with all tokens
3. Score extensions: `new_score = old_score + log P_CTC(c) + alpha * log P_LM(c|prefix)`
4. Prune to top-K beams
5. Return highest-scoring complete sequence

### 4.3 N-gram Language Models

Character n-gram models estimate:

```
P(c_t | c_{t-n+1:t-1}) = count(c_{t-n+1:t}) / count(c_{t-n+1:t-1})
```

With smoothing to handle unseen sequences (add-k, interpolation, backoff).

---

## 5. Neural Network Architectures

### 5.1 CNN Feature Extraction

Convolutional networks extract local visual features:

```
F = CNN(x)   # F: (H', W', C')
```

Key design choices for HTR:
- Asymmetric pooling (more vertical than horizontal)
- Residual connections for depth
- Batch normalization for training stability

### 5.2 Recurrent Sequence Modeling

Bidirectional LSTMs process feature sequences:

```
h_t^{fwd} = LSTM_{fwd}(f_t, h_{t-1}^{fwd})
h_t^{bwd} = LSTM_{bwd}(f_t, h_{t+1}^{bwd})
h_t = [h_t^{fwd}; h_t^{bwd}]
```

This captures both left and right context for each position.

### 5.3 Transformer Architecture

Self-attention enables global context modeling:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

Multi-head attention:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
```

### 5.4 Vision Transformers (ViT)

ViT processes images as sequences of patches:

1. Split image into patches of size `(P, P)`
2. Flatten and project: `z_i = x_i^{patch} E + pos_i`
3. Apply Transformer encoder layers
4. Output sequence for CTC or as encoder for attention decoder

---

## 6. Evaluation Metrics

### 6.1 Character Error Rate (CER)

```
CER = (S + D + I) / N
```

where:
- S = substitutions
- D = deletions
- I = insertions
- N = total reference characters

### 6.2 Word Error Rate (WER)

Same formula applied at word level:

```
WER = (S_w + D_w + I_w) / N_w
```

### 6.3 Sequence Error Rate (SER)

```
SER = 1 if reference != hypothesis else 0
```

Binary indicator aggregated over samples.

---

## 7. Thulium Architecture

Thulium implements a modular architecture allowing flexible component composition:

```
                    +---------------+
                    |  Input Image  |
                    +-------+-------+
                            |
                    +-------v-------+
                    |   Backbone    |
                    | (CNN or ViT)  |
                    +-------+-------+
                            |
                    +-------v-------+
                    | Sequence Head |
                    |(LSTM/Transf.) |
                    +-------+-------+
                            |
              +-------------+-------------+
              |                           |
      +-------v-------+           +-------v-------+
      |  CTC Decoder  |           | Attn Decoder  |
      +-------+-------+           +-------+-------+
              |                           |
              +-------------+-------------+
                            |
                    +-------v-------+
                    | Language Model|
                    |   Rescoring   |
                    +-------+-------+
                            |
                    +-------v-------+
                    |    Output     |
                    +---------------+
```

This modularity enables:
- Easy experimentation with different components
- Language-specific configurations via profiles
- Trade-offs between accuracy and speed

---

## 8. Historical Perspective

HTR has evolved through several paradigms:

1. **Rule-based systems** (1960s-1980s): Hand-crafted features and rules
2. **Statistical methods** (1980s-2000s): HMMs with Gaussian mixture emissions
3. **Hybrid NN-HMM** (2000s-2010s): Neural networks for feature extraction, HMMs for sequence modeling
4. **CNN-RNN-CTC** (2014-2018): End-to-end deep learning with CTC
5. **Transformer-based** (2019-present): Self-attention replacing recurrence

Thulium supports architectures from the CNN-RNN-CTC era through modern Transformer approaches.

---

## References

The theoretical foundations draw from established literature in sequence recognition, including foundational work on CTC, attention mechanisms, and transformer architectures. Thulium integrates these established principles while providing a unified, language-aware implementation framework.
