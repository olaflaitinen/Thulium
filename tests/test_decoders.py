"""
Tests for decoder modules.

Tests CTC decoder (greedy and beam search) and attention decoder.
"""

import pytest
import torch


class TestCTCDecoder:
    """Tests for CTC decoder module."""
    
    def test_ctc_decoder_init(self):
        from thulium.models.decoders.ctc_decoder import CTCDecoder
        
        decoder = CTCDecoder(input_size=256, num_classes=100)
        assert decoder.fc.in_features == 256
        assert decoder.fc.out_features == 101  # +1 for blank
    
    def test_ctc_decoder_forward(self):
        from thulium.models.decoders.ctc_decoder import CTCDecoder
        
        decoder = CTCDecoder(input_size=256, num_classes=100)
        x = torch.randn(4, 50, 256)  # (B, T, D)
        output = decoder(x)
        
        assert output.shape == (4, 50, 101)
        # log_softmax should sum to 0 (in log space)
        log_sum = output.logsumexp(dim=-1)
        assert torch.allclose(log_sum, torch.zeros_like(log_sum), atol=1e-5)
    
    def test_greedy_decode(self):
        from thulium.models.decoders.ctc_decoder import CTCDecoder
        
        decoder = CTCDecoder(input_size=256, num_classes=100)
        x = torch.randn(2, 30, 256)
        log_probs = decoder(x)
        
        results = decoder.decode_greedy(log_probs)
        assert len(results) == 2
        assert all(isinstance(r, list) for r in results)
    
    def test_beam_search_config(self):
        from thulium.models.decoders.ctc_decoder import BeamSearchConfig
        
        config = BeamSearchConfig(
            beam_width=10,
            lm_alpha=0.5,
            lm_beta=0.1,
            length_normalization=True
        )
        assert config.beam_width == 10
        assert config.lm_alpha == 0.5
    
    def test_null_lm_scorer(self):
        from thulium.models.decoders.ctc_decoder import NullLanguageModelScorer
        
        scorer = NullLanguageModelScorer()
        assert scorer.score([1, 2, 3]) == 0.0
        score, state = scorer.score_partial([1, 2], 3)
        assert score == 0.0


class TestAttentionDecoder:
    """Tests for attention-based decoder."""
    
    def test_attention_decoder_init(self):
        from thulium.models.decoders.attention_decoder import AttentionDecoder
        
        decoder = AttentionDecoder(
            vocab_size=100,
            d_model=256,
            encoder_dim=512
        )
        assert decoder.vocab_size == 100
        assert decoder.d_model == 256
    
    def test_attention_decoder_forward(self):
        from thulium.models.decoders.attention_decoder import AttentionDecoder
        
        decoder = AttentionDecoder(
            vocab_size=100,
            d_model=256,
            encoder_dim=512
        )
        
        encoder_out = torch.randn(4, 50, 512)
        target_tokens = torch.randint(0, 100, (4, 20))
        
        logits = decoder(target_tokens, encoder_out)
        assert logits.shape == (4, 20, 100)
    
    def test_greedy_decoding(self):
        from thulium.models.decoders.attention_decoder import AttentionDecoder
        
        decoder = AttentionDecoder(
            vocab_size=100,
            d_model=128,
            encoder_dim=256,
            num_layers=2
        )
        decoder.eval()
        
        encoder_out = torch.randn(2, 30, 256)
        results = decoder.decode_greedy(encoder_out, max_len=10)
        
        assert len(results) == 2
        assert all(isinstance(r, list) for r in results)


class TestPositionalEncoding:
    """Tests for positional encoding."""
    
    def test_positional_encoding_shape(self):
        from thulium.models.decoders.attention_decoder import PositionalEncoding
        
        pe = PositionalEncoding(d_model=256, max_len=100)
        x = torch.randn(4, 50, 256)
        output = pe(x)
        
        assert output.shape == x.shape
    
    def test_positional_encoding_adds_nonzero(self):
        from thulium.models.decoders.attention_decoder import PositionalEncoding
        
        pe = PositionalEncoding(d_model=256, max_len=100, dropout=0.0)
        x = torch.zeros(1, 10, 256)
        output = pe(x)
        
        # Output should have positional information added
        assert not torch.allclose(output, x)
