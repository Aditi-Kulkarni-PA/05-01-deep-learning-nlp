"""
Deep learning models for NLP tasks.
"""

import torch
import torch.nn as nn


class TextClassifier(nn.Module):
    """
    A simple text classifier using LSTM.
    
    Args:
        vocab_size (int): Size of the vocabulary
        embedding_dim (int): Dimension of word embeddings
        hidden_dim (int): Dimension of LSTM hidden state
        output_dim (int): Number of output classes
        n_layers (int): Number of LSTM layers
        bidirectional (bool): Whether to use bidirectional LSTM
        dropout (float): Dropout rate
    """
    
    def __init__(
        self,
        vocab_size,
        embedding_dim=100,
        hidden_dim=256,
        output_dim=2,
        n_layers=2,
        bidirectional=True,
        dropout=0.5,
    ):
        super(TextClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
        )
        
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        """
        Forward pass of the model.
        
        Args:
            text: Input text tensor of shape (batch_size, seq_len)
            text_lengths: Lengths of sequences in the batch
            
        Returns:
            predictions: Output tensor of shape (batch_size, output_dim)
        """
        # Embed the text
        embedded = self.dropout(self.embedding(text))
        
        # Pack the sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # Pass through LSTM
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        # Concatenate the final forward and backward hidden states
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])
        
        # Pass through fully connected layer
        output = self.fc(hidden)
        
        return output


class SequenceToSequence(nn.Module):
    """
    A sequence-to-sequence model with attention for tasks like translation.
    
    Args:
        input_vocab_size (int): Size of the input vocabulary
        output_vocab_size (int): Size of the output vocabulary
        embedding_dim (int): Dimension of word embeddings
        hidden_dim (int): Dimension of LSTM hidden state
        n_layers (int): Number of LSTM layers
        dropout (float): Dropout rate
    """
    
    def __init__(
        self,
        input_vocab_size,
        output_vocab_size,
        embedding_dim=256,
        hidden_dim=512,
        n_layers=2,
        dropout=0.5,
    ):
        super(SequenceToSequence, self).__init__()
        
        self.encoder_embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.decoder_embedding = nn.Embedding(output_vocab_size, embedding_dim)
        
        self.encoder = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
        )
        
        self.decoder = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
        )
        
        self.fc = nn.Linear(hidden_dim, output_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, trg):
        """
        Forward pass of the model.
        
        Args:
            src: Source sequence tensor of shape (batch_size, src_seq_len)
            trg: Target sequence tensor of shape (batch_size, trg_seq_len)
            
        Returns:
            predictions: Output tensor of shape (batch_size, trg_seq_len, output_vocab_size)
        """
        # Encode the source sequence
        embedded_src = self.dropout(self.encoder_embedding(src))
        encoder_outputs, (hidden, cell) = self.encoder(embedded_src)
        
        # Decode the target sequence
        embedded_trg = self.dropout(self.decoder_embedding(trg))
        decoder_outputs, (hidden, cell) = self.decoder(embedded_trg, (hidden, cell))
        
        # Pass through fully connected layer
        predictions = self.fc(decoder_outputs)
        
        return predictions
