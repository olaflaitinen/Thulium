"""
Tests for evaluation metrics module.

Tests CER, WER, SER calculations and latency utilities.
"""

import pytest
from thulium.evaluation.metrics import (
    cer, wer, ser, cer_wer_batch,
    get_edit_operations, LatencyMeter,
    precision_recall_f1, throughput
)


class TestCER:
    """Tests for Character Error Rate."""
    
    def test_identical_strings(self):
        assert cer("hello", "hello") == 0.0
    
    def test_completely_different(self):
        assert cer("abc", "xyz") == 1.0
    
    def test_one_substitution(self):
        result = cer("hello", "hallo")
        assert result == pytest.approx(0.2)
    
    def test_insertion(self):
        result = cer("hello", "helllo")
        assert result == pytest.approx(0.2)
    
    def test_deletion(self):
        result = cer("hello", "helo")
        assert result == pytest.approx(0.2)
    
    def test_empty_reference(self):
        assert cer("", "abc") == 1.0
    
    def test_both_empty(self):
        assert cer("", "") == 0.0
    
    def test_empty_hypothesis(self):
        assert cer("abc", "") == 1.0


class TestWER:
    """Tests for Word Error Rate."""
    
    def test_identical_strings(self):
        assert wer("hello world", "hello world") == 0.0
    
    def test_one_word_substitution(self):
        result = wer("the quick fox", "the fast fox")
        assert result == pytest.approx(1/3)
    
    def test_word_insertion(self):
        result = wer("hello world", "hello big world")
        assert result == pytest.approx(0.5)
    
    def test_word_deletion(self):
        result = wer("hello big world", "hello world")
        assert result == pytest.approx(1/3)
    
    def test_empty_reference(self):
        assert wer("", "hello") == 1.0
    
    def test_both_empty(self):
        assert wer("", "") == 0.0


class TestSER:
    """Tests for Sequence Error Rate."""
    
    def test_identical(self):
        assert ser("hello", "hello") == 0.0
    
    def test_different(self):
        assert ser("hello", "hallo") == 1.0
    
    def test_empty_strings(self):
        assert ser("", "") == 0.0


class TestBatchMetrics:
    """Tests for batch metric computation."""
    
    def test_perfect_batch(self):
        refs = ["hello", "world"]
        hyps = ["hello", "world"]
        batch_cer, batch_wer = cer_wer_batch(refs, hyps)
        assert batch_cer == 0.0
        assert batch_wer == 0.0
    
    def test_mismatched_length(self):
        refs = ["hello"]
        hyps = ["hello", "world"]
        with pytest.raises(ValueError):
            cer_wer_batch(refs, hyps)


class TestEditOperations:
    """Tests for edit operation breakdown."""
    
    def test_substitution(self):
        ops = get_edit_operations("hello", "hallo")
        assert ops.substitutions == 1
        assert ops.deletions == 0
        assert ops.insertions == 0
        assert ops.matches == 4
    
    def test_deletion(self):
        ops = get_edit_operations("hello", "helo")
        assert ops.deletions == 1
        assert ops.matches == 4
    
    def test_insertion(self):
        ops = get_edit_operations("helo", "hello")
        assert ops.insertions == 1
        assert ops.matches == 4
    
    def test_identical(self):
        ops = get_edit_operations("hello", "hello")
        assert ops.total_errors == 0
        assert ops.matches == 5


class TestLatencyMeter:
    """Tests for latency measurement utility."""
    
    def test_basic_timing(self):
        meter = LatencyMeter()
        meter.start()
        meter.stop()
        assert meter.count == 1
        assert meter.mean_ms >= 0
    
    def test_context_manager(self):
        meter = LatencyMeter()
        with meter.measure():
            pass
        assert meter.count == 1
    
    def test_statistics(self):
        meter = LatencyMeter()
        meter.times = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert meter.mean_ms == 30.0
        assert meter.count == 5


class TestPrecisionRecallF1:
    """Tests for detection metrics."""
    
    def test_perfect_detection(self):
        p, r, f1 = precision_recall_f1(10, 0, 0)
        assert p == 1.0
        assert r == 1.0
        assert f1 == 1.0
    
    def test_no_detections(self):
        p, r, f1 = precision_recall_f1(0, 0, 10)
        assert p == 0.0
        assert r == 0.0
        assert f1 == 0.0
    
    def test_balanced(self):
        p, r, f1 = precision_recall_f1(8, 2, 2)
        assert p == pytest.approx(0.8)
        assert r == pytest.approx(0.8)


class TestThroughput:
    """Tests for throughput calculation."""
    
    def test_basic_throughput(self):
        assert throughput(100, 10.0) == 10.0
    
    def test_zero_time(self):
        assert throughput(100, 0.0) == 0.0
