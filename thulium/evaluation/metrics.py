"""
Evaluation metrics for handwriting text recognition.

This module provides implementations of standard metrics used
to evaluate HTR system performance:

- CER: Character Error Rate
- WER: Word Error Rate  
- SER: Sequence Error Rate
- Precision, Recall, F1 for detection tasks
- Latency and throughput utilities

All edit-distance metrics are based on the Levenshtein distance,
computed using the editdistance library for efficiency.

Mathematical Definitions
------------------------

Character Error Rate (CER):

    CER = (S + D + I) / N

where:
- S = number of character substitutions
- D = number of character deletions
- I = number of character insertions
- N = total characters in reference

Word Error Rate (WER):

    WER = (S_w + D_w + I_w) / N_w

Same formula applied at word level.

Sequence Error Rate (SER):

    SER = 1 if reference != hypothesis else 0

Binary indicator of exact match (0% or 100% per sample).
"""

from typing import List, Tuple, Dict, Optional
import time
from dataclasses import dataclass

import editdistance


def cer(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate (CER).
    
    CER measures the character-level edit distance between
    the reference and hypothesis, normalized by reference length.
    
    Args:
        reference: Ground truth text string.
        hypothesis: Predicted text string.
        
    Returns:
        CER value in range [0, 1] or greater if insertions exceed
        reference length. Returns 1.0 for empty reference with
        non-empty hypothesis, 0.0 for both empty.
        
    Example:
        >>> cer("hello", "hallo")
        0.2
        >>> cer("hello", "hello")
        0.0
    """
    if not reference:
        return 1.0 if hypothesis else 0.0
    
    distance = editdistance.eval(reference, hypothesis)
    return float(distance) / len(reference)


def wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER).
    
    WER measures the word-level edit distance between
    the reference and hypothesis, normalized by reference word count.
    
    Args:
        reference: Ground truth text string.
        hypothesis: Predicted text string.
        
    Returns:
        WER value in range [0, 1] or greater.
        
    Example:
        >>> wer("the quick fox", "the fast fox")
        0.333...
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    if not ref_words:
        return 1.0 if hyp_words else 0.0

    distance = editdistance.eval(ref_words, hyp_words)
    return float(distance) / len(ref_words)


def ser(reference: str, hypothesis: str) -> float:
    """
    Calculate Sequence Error Rate (SER).
    
    SER is a binary metric: 1 if the sequences differ, 0 if identical.
    Also known as sentence error rate for sentence-level evaluation.
    
    Args:
        reference: Ground truth text string.
        hypothesis: Predicted text string.
        
    Returns:
        1.0 if strings differ, 0.0 if identical.
    """
    return 1.0 if reference != hypothesis else 0.0


def cer_wer_batch(
    references: List[str],
    hypotheses: List[str]
) -> Tuple[float, float]:
    """
    Calculate CER and WER for a batch of samples.
    
    Computes micro-averaged metrics (sum of errors / sum of lengths)
    which gives more weight to longer sequences.
    
    Args:
        references: List of ground truth strings.
        hypotheses: List of predicted strings.
        
    Returns:
        Tuple of (micro-averaged CER, micro-averaged WER).
    """
    if len(references) != len(hypotheses):
        raise ValueError("References and hypotheses must have same length")
    
    total_char_dist = 0
    total_chars = 0
    total_word_dist = 0
    total_words = 0
    
    for ref, hyp in zip(references, hypotheses):
        total_char_dist += editdistance.eval(ref, hyp)
        total_chars += len(ref)
        
        ref_words = ref.split()
        hyp_words = hyp.split()
        total_word_dist += editdistance.eval(ref_words, hyp_words)
        total_words += len(ref_words)
    
    batch_cer = total_char_dist / total_chars if total_chars > 0 else 0.0
    batch_wer = total_word_dist / total_words if total_words > 0 else 0.0
    
    return batch_cer, batch_wer


@dataclass
class EditOperations:
    """Breakdown of edit operations between reference and hypothesis."""
    substitutions: int
    deletions: int
    insertions: int
    matches: int
    
    @property
    def total_errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions
    
    @property
    def total_reference(self) -> int:
        return self.substitutions + self.deletions + self.matches


def get_edit_operations(reference: str, hypothesis: str) -> EditOperations:
    """
    Compute detailed edit operation counts.
    
    Uses dynamic programming to compute the minimum edit distance
    and trace back to count individual operations.
    
    Args:
        reference: Ground truth string.
        hypothesis: Predicted string.
        
    Returns:
        EditOperations with counts of each operation type.
    """
    m, n = len(reference), len(hypothesis)
    
    # Build DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if reference[i-1] == hypothesis[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],      # deletion
                    dp[i][j-1],      # insertion
                    dp[i-1][j-1]     # substitution
                )
    
    # Trace back to count operations
    subs = dels = ins = matches = 0
    i, j = m, n
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and reference[i-1] == hypothesis[j-1]:
            matches += 1
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    
    return EditOperations(
        substitutions=subs,
        deletions=dels,
        insertions=ins,
        matches=matches
    )


def precision_recall_f1(
    true_positives: int,
    false_positives: int,
    false_negatives: int
) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 score.
    
    Used for detection tasks (e.g., line detection, word detection).
    
    Args:
        true_positives: Correctly detected items.
        false_positives: Incorrectly detected items.
        false_negatives: Missed items.
        
    Returns:
        Tuple of (precision, recall, F1).
    """
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


class LatencyMeter:
    """
    Utility for measuring inference latency.
    
    Tracks timing statistics including mean, std, and percentiles.
    
    Example:
        >>> meter = LatencyMeter()
        >>> for sample in dataset:
        ...     with meter.measure():
        ...         result = model(sample)
        >>> print(f"Avg: {meter.mean_ms:.2f}ms, P95: {meter.p95_ms:.2f}ms")
    """
    
    def __init__(self):
        self.times: List[float] = []
        self._start_time: Optional[float] = None
    
    def start(self) -> None:
        """Start timing."""
        self._start_time = time.perf_counter()
    
    def stop(self) -> float:
        """Stop timing and return elapsed time in ms."""
        if self._start_time is None:
            raise RuntimeError("Timer not started")
        elapsed = (time.perf_counter() - self._start_time) * 1000
        self.times.append(elapsed)
        self._start_time = None
        return elapsed
    
    def measure(self):
        """Context manager for timing a block."""
        return _LatencyContext(self)
    
    @property
    def count(self) -> int:
        return len(self.times)
    
    @property
    def mean_ms(self) -> float:
        return sum(self.times) / len(self.times) if self.times else 0.0
    
    @property
    def std_ms(self) -> float:
        if len(self.times) < 2:
            return 0.0
        mean = self.mean_ms
        variance = sum((t - mean) ** 2 for t in self.times) / len(self.times)
        return variance ** 0.5
    
    @property
    def p95_ms(self) -> float:
        if not self.times:
            return 0.0
        sorted_times = sorted(self.times)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times) - 1)]
    
    @property
    def p99_ms(self) -> float:
        if not self.times:
            return 0.0
        sorted_times = sorted(self.times)
        idx = int(len(sorted_times) * 0.99)
        return sorted_times[min(idx, len(sorted_times) - 1)]
    
    def summary(self) -> Dict[str, float]:
        """Return summary statistics."""
        return {
            'count': self.count,
            'mean_ms': self.mean_ms,
            'std_ms': self.std_ms,
            'p95_ms': self.p95_ms,
            'p99_ms': self.p99_ms,
        }


class _LatencyContext:
    """Context manager helper for LatencyMeter."""
    
    def __init__(self, meter: LatencyMeter):
        self.meter = meter
    
    def __enter__(self):
        self.meter.start()
        return self
    
    def __exit__(self, *args):
        self.meter.stop()


def throughput(num_samples: int, total_time_s: float) -> float:
    """
    Calculate throughput in samples per second.
    
    Args:
        num_samples: Number of samples processed.
        total_time_s: Total processing time in seconds.
        
    Returns:
        Throughput in samples/second.
    """
    if total_time_s <= 0:
        return 0.0
    return num_samples / total_time_s
