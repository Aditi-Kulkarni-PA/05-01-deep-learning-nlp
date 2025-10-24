"""
Example script for text classification using LSTM.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from deep_learning_nlp.models import TextClassifier
from deep_learning_nlp.data_processing import TextPreprocessor, TextDataset
from deep_learning_nlp.data_processing import DataLoader as CustomDataLoader
from deep_learning_nlp.utils import train_model, evaluate_model, save_model


def main():
    # Example data
    train_texts = [
        "This is a great movie",
        "I loved this film",
        "Terrible waste of time",
        "Not worth watching",
        "Amazing storyline and acting",
        "Boring and predictable",
    ]
    train_labels = [1, 1, 0, 0, 1, 0]  # 1 for positive, 0 for negative
    
    test_texts = [
        "Great entertainment",
        "Awful experience",
    ]
    test_labels = [1, 0]
    
    # Initialize preprocessor and build vocabulary
    preprocessor = TextPreprocessor(lowercase=True, remove_punctuation=False)
    word2idx, idx2word = preprocessor.build_vocab(train_texts, min_freq=1)
    
    print(f"Vocabulary size: {len(word2idx)}")
    
    # Create datasets
    train_dataset = TextDataset(train_texts, train_labels, word2idx, max_length=20)
    test_dataset = TextDataset(test_texts, test_labels, word2idx, max_length=20)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=CustomDataLoader.collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=CustomDataLoader.collate_fn
    )
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = TextClassifier(
        vocab_size=len(word2idx),
        embedding_dim=50,
        hidden_dim=128,
        output_dim=2,
        n_layers=2,
        bidirectional=True,
        dropout=0.5,
    )
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    print("\nTraining model...")
    model, train_losses = train_model(
        model, train_loader, criterion, optimizer, device, num_epochs=5
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, accuracy = evaluate_model(model, test_loader, criterion, device)
    
    # Save model
    save_model(model, "text_classifier.pth")
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
