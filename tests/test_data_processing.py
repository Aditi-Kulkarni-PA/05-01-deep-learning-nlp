"""
Unit tests for data processing module.
"""

import pytest
import torch

from deep_learning_nlp.data_processing import TextPreprocessor, TextDataset


class TestTextPreprocessor:
    """Tests for TextPreprocessor class."""
    
    def test_clean_text_lowercase(self):
        """Test text cleaning with lowercase."""
        preprocessor = TextPreprocessor(lowercase=True, remove_punctuation=False)
        text = "Hello World!"
        cleaned = preprocessor.clean_text(text)
        assert cleaned == "hello world!"
    
    def test_clean_text_remove_punctuation(self):
        """Test text cleaning with punctuation removal."""
        preprocessor = TextPreprocessor(lowercase=False, remove_punctuation=True)
        text = "Hello, World!"
        cleaned = preprocessor.clean_text(text)
        assert cleaned == "Hello World"
    
    def test_tokenize(self):
        """Test text tokenization."""
        preprocessor = TextPreprocessor()
        text = "This is a test"
        tokens = preprocessor.tokenize(text)
        assert tokens == ["this", "is", "a", "test"]
    
    def test_build_vocab(self):
        """Test vocabulary building."""
        preprocessor = TextPreprocessor()
        texts = ["hello world", "hello there"]
        word2idx, idx2word = preprocessor.build_vocab(texts, min_freq=1)
        
        # Check special tokens
        assert '<pad>' in word2idx
        assert '<unk>' in word2idx
        assert '<sos>' in word2idx
        assert '<eos>' in word2idx
        
        # Check words
        assert 'hello' in word2idx
        assert 'world' in word2idx
        assert 'there' in word2idx
        
        # Check bidirectional mapping
        for word, idx in word2idx.items():
            assert idx2word[idx] == word
    
    def test_encode_text(self):
        """Test text encoding."""
        preprocessor = TextPreprocessor()
        word2idx = {'<pad>': 0, '<unk>': 1, 'hello': 2, 'world': 3}
        
        text = "hello world"
        encoded = preprocessor.encode_text(text, word2idx)
        assert encoded == [2, 3]
        
        # Test unknown word
        text = "hello unknown"
        encoded = preprocessor.encode_text(text, word2idx)
        assert encoded == [2, 1]  # 1 is <unk>
    
    def test_encode_text_with_max_length(self):
        """Test text encoding with max length."""
        preprocessor = TextPreprocessor()
        word2idx = {'<pad>': 0, '<unk>': 1, 'hello': 2, 'world': 3, 'test': 4}
        
        text = "hello world test"
        encoded = preprocessor.encode_text(text, word2idx, max_length=2)
        assert len(encoded) == 2
        assert encoded == [2, 3]


class TestTextDataset:
    """Tests for TextDataset class."""
    
    def test_dataset_length(self):
        """Test dataset length."""
        texts = ["hello world", "test"]
        labels = [0, 1]
        word2idx = {'<pad>': 0, '<unk>': 1, 'hello': 2, 'world': 3, 'test': 4}
        
        dataset = TextDataset(texts, labels, word2idx)
        assert len(dataset) == 2
    
    def test_dataset_getitem(self):
        """Test dataset item retrieval."""
        texts = ["hello world"]
        labels = [1]
        word2idx = {'<pad>': 0, '<unk>': 1, 'hello': 2, 'world': 3}
        
        dataset = TextDataset(texts, labels, word2idx)
        text_tensor, label_tensor = dataset[0]
        
        assert isinstance(text_tensor, torch.LongTensor)
        assert isinstance(label_tensor, torch.LongTensor)
        assert label_tensor.item() == 1
