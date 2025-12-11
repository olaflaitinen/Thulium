"""
Error analysis tools for HTR evaluation.

This module provides utilities for analyzing recognition errors,
identifying patterns, and generating diagnostic reports. Key
capabilities include:

- Character confusion matrix generation
- Edit operation classification
- Per-language error aggregation
- Most frequent error pattern identification

These tools help researchers and engineers understand model
failure modes and guide improvements.
"""

from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter
from dataclasses import dataclass, field
import editdistance


@dataclass
class ErrorInstance:
    """
    Single error instance from HTR evaluation.
    
    Attributes:
        sample_id: Identifier for the source sample.
        position: Character position in reference.
        error_type: Type of error ('substitution', 'deletion', 'insertion').
        reference_char: Character in reference (None for insertion).
        hypothesis_char: Character in hypothesis (None for deletion).
        context: Surrounding context characters.
        language: Language code of the sample.
    """
    sample_id: str
    position: int
    error_type: str
    reference_char: Optional[str]
    hypothesis_char: Optional[str]
    context: str = ""
    language: Optional[str] = None


@dataclass
class ConfusionMatrix:
    """
    Character confusion matrix for analyzing substitution errors.
    
    Tracks how often each reference character is confused with
    each hypothesis character.
    """
    matrix: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    total_correct: int = 0
    total_errors: int = 0
    
    def add(self, reference_char: str, hypothesis_char: str) -> None:
        """Add a single observation to the matrix."""
        if reference_char == hypothesis_char:
            self.total_correct += 1
        else:
            self.matrix[reference_char][hypothesis_char] += 1
            self.total_errors += 1
    
    def get_confusions(self, min_count: int = 1) -> List[Tuple[str, str, int]]:
        """
        Get list of character confusions sorted by frequency.
        
        Args:
            min_count: Minimum count to include.
            
        Returns:
            List of (reference_char, hypothesis_char, count) tuples.
        """
        confusions = []
        for ref_char, hyp_dict in self.matrix.items():
            for hyp_char, count in hyp_dict.items():
                if count >= min_count:
                    confusions.append((ref_char, hyp_char, count))
        
        return sorted(confusions, key=lambda x: -x[2])
    
    def top_confusions(self, n: int = 10) -> List[Tuple[str, str, int]]:
        """Get top N most frequent confusions."""
        return self.get_confusions()[:n]
    
    def accuracy_for_char(self, char: str) -> float:
        """Get recognition accuracy for a specific character."""
        # Count correct as total - confusions FROM this char
        confused_count = sum(self.matrix[char].values())
        # Would need separate correct count tracking for accurate measure
        return 0.0  # Placeholder
    
    def to_dict(self) -> Dict[str, Dict[str, int]]:
        """Convert to regular dictionary."""
        return {k: dict(v) for k, v in self.matrix.items()}


@dataclass
class ErrorAnalysisResult:
    """
    Comprehensive error analysis result.
    
    Attributes:
        num_samples: Number of samples analyzed.
        total_chars: Total characters in references.
        total_errors: Total edit distance (errors).
        substitutions: Total substitution errors.
        deletions: Total deletion errors.
        insertions: Total insertion errors.
        confusion_matrix: Character confusion matrix.
        errors_by_language: Error counts per language.
        error_instances: Detailed error instances.
        common_patterns: Most common error patterns.
    """
    num_samples: int = 0
    total_chars: int = 0
    total_errors: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    confusion_matrix: ConfusionMatrix = field(default_factory=ConfusionMatrix)
    errors_by_language: Dict[str, Dict[str, int]] = field(default_factory=dict)
    error_instances: List[ErrorInstance] = field(default_factory=list)
    common_patterns: List[Tuple[str, int]] = field(default_factory=list)
    
    @property
    def cer(self) -> float:
        """Overall character error rate."""
        if self.total_chars == 0:
            return 0.0
        return self.total_errors / self.total_chars
    
    @property
    def substitution_rate(self) -> float:
        """Substitution rate as fraction of total characters."""
        if self.total_chars == 0:
            return 0.0
        return self.substitutions / self.total_chars
    
    @property
    def deletion_rate(self) -> float:
        """Deletion rate as fraction of total characters."""
        if self.total_chars == 0:
            return 0.0
        return self.deletions / self.total_chars
    
    @property
    def insertion_rate(self) -> float:
        """Insertion rate as fraction of total characters."""
        if self.total_chars == 0:
            return 0.0
        return self.insertions / self.total_chars


def align_sequences(reference: str, hypothesis: str) -> List[Tuple[Optional[str], Optional[str]]]:
    """
    Compute optimal alignment between reference and hypothesis.
    
    Uses dynamic programming to find the minimum edit distance
    alignment, returning paired characters for analysis.
    
    Args:
        reference: Ground truth string.
        hypothesis: Predicted string.
        
    Returns:
        List of (ref_char, hyp_char) tuples. None indicates gap.
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
    
    # Trace back to build alignment
    alignment = []
    i, j = m, n
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and reference[i-1] == hypothesis[j-1]:
            alignment.append((reference[i-1], hypothesis[j-1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            alignment.append((reference[i-1], hypothesis[j-1]))  # substitution
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            alignment.append((reference[i-1], None))  # deletion
            i -= 1
        else:
            alignment.append((None, hypothesis[j-1]))  # insertion
            j -= 1
    
    return list(reversed(alignment))


def analyze_errors(
    reference: str,
    hypothesis: str,
    sample_id: str = "",
    language: Optional[str] = None
) -> List[ErrorInstance]:
    """
    Analyze errors between reference and hypothesis.
    
    Args:
        reference: Ground truth text.
        hypothesis: Predicted text.
        sample_id: Identifier for the sample.
        language: Language code.
        
    Returns:
        List of ErrorInstance objects describing each error.
    """
    alignment = align_sequences(reference, hypothesis)
    errors = []
    
    ref_pos = 0
    for ref_char, hyp_char in alignment:
        if ref_char == hyp_char:
            ref_pos += 1 if ref_char else 0
            continue
        
        # Determine error type
        if ref_char is None:
            error_type = 'insertion'
        elif hyp_char is None:
            error_type = 'deletion'
        else:
            error_type = 'substitution'
        
        # Extract context
        context_start = max(0, ref_pos - 2)
        context_end = min(len(reference), ref_pos + 3)
        context = reference[context_start:context_end]
        
        errors.append(ErrorInstance(
            sample_id=sample_id,
            position=ref_pos,
            error_type=error_type,
            reference_char=ref_char,
            hypothesis_char=hyp_char,
            context=context,
            language=language
        ))
        
        if ref_char is not None:
            ref_pos += 1
    
    return errors


def analyze_batch(
    references: List[str],
    hypotheses: List[str],
    sample_ids: Optional[List[str]] = None,
    languages: Optional[List[str]] = None
) -> ErrorAnalysisResult:
    """
    Perform comprehensive error analysis on a batch of samples.
    
    Args:
        references: List of ground truth strings.
        hypotheses: List of predicted strings.
        sample_ids: Optional list of sample identifiers.
        languages: Optional list of language codes.
        
    Returns:
        ErrorAnalysisResult with aggregated analysis.
    """
    if len(references) != len(hypotheses):
        raise ValueError("References and hypotheses must have same length")
    
    if sample_ids is None:
        sample_ids = [f"sample_{i}" for i in range(len(references))]
    if languages is None:
        languages = [None] * len(references)
    
    result = ErrorAnalysisResult(num_samples=len(references))
    
    for ref, hyp, sid, lang in zip(references, hypotheses, sample_ids, languages):
        result.total_chars += len(ref)
        result.total_errors += editdistance.eval(ref, hyp)
        
        # Analyze individual errors
        errors = analyze_errors(ref, hyp, sid, lang)
        
        for error in errors:
            if error.error_type == 'substitution':
                result.substitutions += 1
                result.confusion_matrix.add(
                    error.reference_char, error.hypothesis_char
                )
            elif error.error_type == 'deletion':
                result.deletions += 1
            else:  # insertion
                result.insertions += 1
            
            result.error_instances.append(error)
            
            # Track per-language errors
            if lang:
                if lang not in result.errors_by_language:
                    result.errors_by_language[lang] = {
                        'substitutions': 0, 'deletions': 0,
                        'insertions': 0, 'total': 0
                    }
                result.errors_by_language[lang][error.error_type + 's'] += 1
                result.errors_by_language[lang]['total'] += 1
    
    # Identify common error patterns
    pattern_counter = Counter()
    for error in result.error_instances:
        if error.error_type == 'substitution':
            pattern = f"{error.reference_char}->{error.hypothesis_char}"
            pattern_counter[pattern] += 1
    
    result.common_patterns = pattern_counter.most_common(20)
    
    return result


def generate_error_report(result: ErrorAnalysisResult, format: str = 'markdown') -> str:
    """
    Generate a formatted error analysis report.
    
    Args:
        result: ErrorAnalysisResult to format.
        format: Output format ('markdown' or 'text').
        
    Returns:
        Formatted report string.
    """
    lines = [
        "# Error Analysis Report",
        "",
        "## Summary",
        "",
        f"- **Samples Analyzed**: {result.num_samples}",
        f"- **Total Characters**: {result.total_chars}",
        f"- **Total Errors**: {result.total_errors}",
        f"- **Character Error Rate**: {result.cer * 100:.2f}%",
        "",
        "## Error Breakdown",
        "",
        f"- **Substitutions**: {result.substitutions} ({result.substitution_rate * 100:.2f}%)",
        f"- **Deletions**: {result.deletions} ({result.deletion_rate * 100:.2f}%)",
        f"- **Insertions**: {result.insertions} ({result.insertion_rate * 100:.2f}%)",
        "",
    ]
    
    # Top confusions
    if result.common_patterns:
        lines.extend([
            "## Most Common Substitution Errors",
            "",
            "| Pattern | Count |",
            "|:--------|------:|"
        ])
        for pattern, count in result.common_patterns[:15]:
            lines.append(f"| `{pattern}` | {count} |")
        lines.append("")
    
    # Per-language breakdown
    if result.errors_by_language:
        lines.extend([
            "## Errors by Language",
            "",
            "| Language | Total | Substitutions | Deletions | Insertions |",
            "|:---------|------:|--------------:|----------:|-----------:|"
        ])
        for lang, counts in sorted(result.errors_by_language.items()):
            lines.append(
                f"| {lang} | {counts['total']} | {counts['substitutions']} | "
                f"{counts['deletions']} | {counts['insertions']} |"
            )
        lines.append("")
    
    return '\n'.join(lines)
