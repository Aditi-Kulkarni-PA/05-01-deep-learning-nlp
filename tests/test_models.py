"""
Unit tests for models module.
"""

import pytest
import torch

from deep_learning_nlp.models import TextClassifier, SequenceToSequence


class TestTextClassifier:
    """Tests for TextClassifier model."""
    
    def test_model_initialization(self):
        """Test model can be initialized."""
        model = TextClassifier(
            vocab_size=100,
            embedding_dim=50,
            hidden_dim=128,
            output_dim=2,
            n_layers=2,
            bidirectional=True,
            dropout=0.5
        )
        assert model is not None
    
    def test_forward_pass(self):
        """Test forward pass through the model."""
        vocab_size = 100
        batch_size = 4
        seq_len = 10
        output_dim = 2
        
        model = TextClassifier(
            vocab_size=vocab_size,
            embedding_dim=50,
            hidden_dim=64,
            output_dim=output_dim,
            n_layers=1,
            bidirectional=True,
            dropout=0.5
        )
        
        # Create dummy input
        text = torch.randint(0, vocab_size, (batch_size, seq_len))
        text_lengths = torch.LongTensor([10, 8, 6, 4])
        
        # Forward pass
        output = model(text, text_lengths)
        
        assert output.shape == (batch_size, output_dim)
    
    def test_unidirectional_lstm(self):
        """Test model with unidirectional LSTM."""
        vocab_size = 100
        batch_size = 2
        seq_len = 5
        output_dim = 3
        
        model = TextClassifier(
            vocab_size=vocab_size,
            embedding_dim=30,
            hidden_dim=32,
            output_dim=output_dim,
            n_layers=1,
            bidirectional=False,
            dropout=0.0
        )
        
        text = torch.randint(0, vocab_size, (batch_size, seq_len))
        text_lengths = torch.LongTensor([5, 3])
        
        output = model(text, text_lengths)
        
        assert output.shape == (batch_size, output_dim)


class TestSequenceToSequence:
    """Tests for SequenceToSequence model."""
    
    def test_model_initialization(self):
        """Test model can be initialized."""
        model = SequenceToSequence(
            input_vocab_size=100,
            output_vocab_size=100,
            embedding_dim=128,
            hidden_dim=256,
            n_layers=2,
            dropout=0.5
        )
        assert model is not None
    
    def test_forward_pass(self):
        """Test forward pass through the model."""
        input_vocab_size = 100
        output_vocab_size = 100
        batch_size = 4
        src_seq_len = 10
        trg_seq_len = 12
        
        model = SequenceToSequence(
            input_vocab_size=input_vocab_size,
            output_vocab_size=output_vocab_size,
            embedding_dim=64,
            hidden_dim=128,
            n_layers=1,
            dropout=0.3
        )
        
        # Create dummy input
        src = torch.randint(0, input_vocab_size, (batch_size, src_seq_len))
        trg = torch.randint(0, output_vocab_size, (batch_size, trg_seq_len))
        
        # Forward pass
        output = model(src, trg)
        
        assert output.shape == (batch_size, trg_seq_len, output_vocab_size)
