"""
Tests for language model modules.

Tests character-level LM and n-gram LM implementations.
"""

import pytest
import torch


class TestCharacterLM:
    """Tests for neural character language model."""
    
    def test_char_lm_init(self):
        from thulium.models.language_models.char_lm import CharacterLM
        
        lm = CharacterLM(vocab_size=100, embedding_dim=64, hidden_size=128)
        assert lm.vocab_size == 100
        assert lm.hidden_size == 128
    
    def test_char_lm_forward(self):
        from thulium.models.language_models.char_lm import CharacterLM
        
        lm = CharacterLM(vocab_size=100, embedding_dim=64, hidden_size=128)
        x = torch.randint(0, 100, (4, 20))
        
        logits, hidden = lm(x)
        assert logits.shape == (4, 20, 100)
        assert hidden[0].shape[0] == 2  # num_layers
    
    def test_char_lm_score(self):
        from thulium.models.language_models.char_lm import CharacterLM
        
        lm = CharacterLM(vocab_size=100, embedding_dim=64, hidden_size=128)
        lm.eval()
        
        sequence = [5, 10, 15, 20, 25]
        score = lm.score(sequence)
        
        # Score should be negative log probability
        assert isinstance(score, float)
        assert score <= 0
    
    def test_char_lm_score_partial(self):
        from thulium.models.language_models.char_lm import CharacterLM
        
        lm = CharacterLM(vocab_size=100, embedding_dim=64, hidden_size=128)
        lm.eval()
        
        prefix = [5, 10, 15]
        next_token = 20
        
        score, state = lm.score_partial(prefix, next_token)
        assert isinstance(score, float)
    
    def test_short_sequence(self):
        from thulium.models.language_models.char_lm import CharacterLM
        
        lm = CharacterLM(vocab_size=100)
        lm.eval()
        
        assert lm.score([]) == 0.0
        assert lm.score([5]) == 0.0


class TestTransformerCharLM:
    """Tests for Transformer character language model."""
    
    def test_transformer_lm_init(self):
        from thulium.models.language_models.char_lm import TransformerCharLM
        
        lm = TransformerCharLM(vocab_size=100, d_model=64, num_layers=2)
        assert lm.vocab_size == 100
    
    def test_transformer_lm_forward(self):
        from thulium.models.language_models.char_lm import TransformerCharLM
        
        lm = TransformerCharLM(vocab_size=100, d_model=64, num_layers=2)
        x = torch.randint(0, 100, (2, 15))
        
        logits = lm(x)
        assert logits.shape == (2, 15, 100)


class TestCharacterNGramLM:
    """Tests for n-gram character language model."""
    
    def test_ngram_init(self):
        from thulium.models.language_models.ngram_lm import CharacterNGramLM
        
        lm = CharacterNGramLM(n=3, smoothing='interpolation')
        assert lm.n == 3
        assert lm.smoothing == 'interpolation'
    
    def test_ngram_train(self):
        from thulium.models.language_models.ngram_lm import CharacterNGramLM
        
        lm = CharacterNGramLM(n=3)
        texts = ["hello world", "hello there", "world hello"]
        lm.train(texts)
        
        assert len(lm.vocab) > 0
        assert lm.total_chars > 0
    
    def test_ngram_score(self):
        from thulium.models.language_models.ngram_lm import CharacterNGramLM
        
        lm = CharacterNGramLM(n=3)
        lm.train(["hello world", "hello there"])
        
        sequence = [ord(c) for c in "hello"]
        score = lm.score(sequence)
        
        assert isinstance(score, float)
        assert score <= 0  # Log probability
    
    def test_ngram_score_partial(self):
        from thulium.models.language_models.ngram_lm import CharacterNGramLM
        
        lm = CharacterNGramLM(n=3)
        lm.train(["hello world"])
        
        prefix = [ord('h'), ord('e')]
        next_token = ord('l')
        
        score, state = lm.score_partial(prefix, next_token)
        assert isinstance(score, float)
        assert state is not None
    
    def test_ngram_add_k_smoothing(self):
        from thulium.models.language_models.ngram_lm import CharacterNGramLM
        
        lm = CharacterNGramLM(n=2, smoothing='add_k', add_k=0.1)
        lm.train(["hello"])
        
        # Unseen sequence should still get non-zero probability
        unseen = [ord('x'), ord('y'), ord('z')]
        score = lm.score(unseen)
        assert score < 0  # Should be a valid log probability
    
    def test_empty_sequence(self):
        from thulium.models.language_models.ngram_lm import CharacterNGramLM
        
        lm = CharacterNGramLM(n=3)
        lm.train(["hello"])
        
        assert lm.score([]) == 0.0


class TestWordNGramLM:
    """Tests for word-level n-gram model."""
    
    def test_word_ngram_train(self):
        from thulium.models.language_models.ngram_lm import WordNGramLM
        
        lm = WordNGramLM(n=2)
        lm.train(["hello world", "world hello", "hello there"])
        
        assert "hello" in lm.vocab
        assert "world" in lm.vocab
    
    def test_word_ngram_score(self):
        from thulium.models.language_models.ngram_lm import WordNGramLM
        
        lm = WordNGramLM(n=2)
        lm.train(["hello world", "world hello"])
        
        score = lm.score(["hello", "world"])
        assert isinstance(score, float)
        assert score <= 0
