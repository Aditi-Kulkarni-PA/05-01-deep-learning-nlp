#!/usr/bin/env python3
"""
End-to-end example demonstrating the complete workflow of the deep-learning-nlp toolkit.

This script shows:
1. Data preprocessing
2. Vocabulary building
3. Dataset creation
4. Model initialization
5. Training
6. Evaluation
7. Making predictions
"""

import sys
sys.path.insert(0, 'src')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from deep_learning_nlp.models import TextClassifier
from deep_learning_nlp.data_processing import TextPreprocessor, TextDataset
from deep_learning_nlp.data_processing import DataLoader as CustomDataLoader
from deep_learning_nlp.utils import train_model, evaluate_model, save_model


def main():
    print("=" * 70)
    print("Deep Learning NLP - End-to-End Example")
    print("=" * 70)
    
    # Step 1: Prepare sample data
    print("\n1. Preparing sample data...")
    train_texts = [
        "This movie is absolutely fantastic and amazing",
        "I loved every minute of this film",
        "Best movie I have seen this year",
        "Great acting and wonderful storyline",
        "Excellent cinematography and direction",
        "Terrible waste of time and money",
        "Not worth watching at all",
        "Boring and predictable plot",
        "Very disappointing experience",
        "Awful acting and terrible script",
    ]
    train_labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # 1: positive, 0: negative
    
    test_texts = [
        "Wonderful entertainment for the whole family",
        "Absolutely horrible movie",
        "Great performances by all actors",
        "Complete waste of my evening",
    ]
    test_labels = [1, 0, 1, 0]
    
    print(f"   Training samples: {len(train_texts)}")
    print(f"   Test samples: {len(test_texts)}")
    
    # Step 2: Initialize preprocessor and build vocabulary
    print("\n2. Building vocabulary...")
    preprocessor = TextPreprocessor(lowercase=True, remove_punctuation=False)
    word2idx, idx2word = preprocessor.build_vocab(train_texts, min_freq=1)
    
    print(f"   Vocabulary size: {len(word2idx)}")
    print(f"   Sample words: {list(word2idx.keys())[:10]}")
    
    # Step 3: Create datasets and data loaders
    print("\n3. Creating datasets and data loaders...")
    train_dataset = TextDataset(train_texts, train_labels, word2idx, max_length=20)
    test_dataset = TextDataset(test_texts, test_labels, word2idx, max_length=20)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=CustomDataLoader.collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=CustomDataLoader.collate_fn
    )
    
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # Step 4: Initialize model
    print("\n4. Initializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    
    model = TextClassifier(
        vocab_size=len(word2idx),
        embedding_dim=64,
        hidden_dim=128,
        output_dim=2,
        n_layers=2,
        bidirectional=True,
        dropout=0.5,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Step 5: Define loss function and optimizer
    print("\n5. Setting up training configuration...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"   Loss function: CrossEntropyLoss")
    print(f"   Optimizer: Adam (lr=0.001)")
    
    # Step 6: Train model
    print("\n6. Training model...")
    print("-" * 70)
    model, train_losses = train_model(
        model, train_loader, criterion, optimizer, device, num_epochs=10
    )
    print("-" * 70)
    
    # Step 7: Evaluate model
    print("\n7. Evaluating model on test set...")
    print("-" * 70)
    test_loss, accuracy = evaluate_model(model, test_loader, criterion, device)
    print("-" * 70)
    
    # Step 8: Make predictions on new examples
    print("\n8. Making predictions on new examples...")
    print("-" * 70)
    
    def predict_sentiment(text, model, word2idx, preprocessor, device):
        """Predict sentiment for a single text."""
        model.eval()
        
        encoded = preprocessor.encode_text(text, word2idx)
        text_tensor = torch.LongTensor(encoded).unsqueeze(0).to(device)
        text_length = torch.LongTensor([len(encoded)]).to(device)
        
        with torch.no_grad():
            output = model(text_tensor, text_length)
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
        
        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = probabilities[0][prediction].item()
        
        return sentiment, confidence
    
    new_examples = [
        "This is an amazing movie!",
        "I hated every second of it.",
        "Pretty good overall.",
        "Absolutely terrible.",
    ]
    
    print("\nPredictions:")
    for i, text in enumerate(new_examples, 1):
        sentiment, confidence = predict_sentiment(
            text, model, word2idx, preprocessor, device
        )
        print(f"{i}. Text: '{text}'")
        print(f"   Prediction: {sentiment} (confidence: {confidence:.2%})\n")
    
    # Step 9: Save model
    print("\n9. Saving model...")
    model_path = "/tmp/text_classifier_example.pth"
    save_model(model, model_path)
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)
    
    print("\nNext steps:")
    print("- Explore the notebooks/ directory for interactive tutorials")
    print("- Check examples/transformer_examples.py for pre-trained models")
    print("- Run tests with: pytest tests/")
    print("- Read the documentation in README.md")


if __name__ == "__main__":
    main()
