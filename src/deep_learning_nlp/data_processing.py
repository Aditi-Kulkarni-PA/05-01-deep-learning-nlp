"""
Data processing utilities for NLP tasks.
"""

import re
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class TextPreprocessor:
    """
    Preprocessor for text data.
    """
    
    def __init__(self, lowercase=True, remove_punctuation=False):
        """
        Initialize the preprocessor.
        
        Args:
            lowercase (bool): Whether to convert text to lowercase
            remove_punctuation (bool): Whether to remove punctuation
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        
    def clean_text(self, text: str) -> str:
        """
        Clean the input text.
        
        Args:
            text: Input text string
            
        Returns:
            cleaned_text: Cleaned text string
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text.
        
        Args:
            text: Input text string
            
        Returns:
            tokens: List of tokens
        """
        text = self.clean_text(text)
        return text.split()
    
    def build_vocab(self, texts: List[str], min_freq=1) -> Tuple[dict, dict]:
        """
        Build vocabulary from a list of texts.
        
        Args:
            texts: List of text strings
            min_freq: Minimum frequency for a word to be included in vocabulary
            
        Returns:
            word2idx: Dictionary mapping words to indices
            idx2word: Dictionary mapping indices to words
        """
        word_freq = {}
        
        for text in texts:
            tokens = self.tokenize(text)
            for token in tokens:
                word_freq[token] = word_freq.get(token, 0) + 1
        
        # Filter by minimum frequency
        vocab = [word for word, freq in word_freq.items() if freq >= min_freq]
        
        # Add special tokens
        word2idx = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
        for word in vocab:
            if word not in word2idx:
                word2idx[word] = len(word2idx)
        
        idx2word = {idx: word for word, idx in word2idx.items()}
        
        return word2idx, idx2word
    
    def encode_text(self, text: str, word2idx: dict, max_length=None) -> List[int]:
        """
        Encode text to a list of indices.
        
        Args:
            text: Input text string
            word2idx: Dictionary mapping words to indices
            max_length: Maximum sequence length (optional)
            
        Returns:
            encoded: List of word indices
        """
        tokens = self.tokenize(text)
        encoded = [word2idx.get(token, word2idx['<unk>']) for token in tokens]
        
        if max_length is not None:
            encoded = encoded[:max_length]
        
        return encoded


class TextDataset(Dataset):
    """
    PyTorch Dataset for text data.
    """
    
    def __init__(self, texts, labels, word2idx, max_length=None):
        """
        Initialize the dataset.
        
        Args:
            texts: List of text strings
            labels: List of labels
            word2idx: Dictionary mapping words to indices
            max_length: Maximum sequence length (optional)
        """
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_length = max_length
        self.preprocessor = TextPreprocessor()
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoded = self.preprocessor.encode_text(text, self.word2idx, self.max_length)
        
        return torch.LongTensor(encoded), torch.LongTensor([label])


class DataLoader:
    """
    Custom data loader with padding collate function.
    """
    
    @staticmethod
    def collate_fn(batch):
        """
        Collate function to pad sequences in a batch.
        
        Args:
            batch: List of (text, label) tuples
            
        Returns:
            texts: Padded text tensor
            text_lengths: Tensor of sequence lengths
            labels: Label tensor
        """
        texts, labels = zip(*batch)
        
        # Get lengths before padding
        text_lengths = torch.LongTensor([len(text) for text in texts])
        
        # Pad sequences
        texts = pad_sequence(texts, batch_first=True, padding_value=0)
        labels = torch.cat(labels)
        
        return texts, text_lengths, labels
