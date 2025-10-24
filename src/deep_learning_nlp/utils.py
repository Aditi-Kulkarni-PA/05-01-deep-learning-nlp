"""
Utility functions for training and evaluation.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    """
    Train a model.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cpu or cuda)
        num_epochs: Number of training epochs
        
    Returns:
        model: Trained model
        train_losses: List of training losses per epoch
    """
    model.to(device)
    model.train()
    
    train_losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            texts, text_lengths, labels = batch
            texts = texts.to(device)
            text_lengths = text_lengths.to(device)
            labels = labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(texts, text_lengths)
            
            # Calculate loss
            loss = criterion(predictions, labels)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")
    
    return model, train_losses


def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluate a model.
    
    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to evaluate on (cpu or cuda)
        
    Returns:
        test_loss: Average test loss
        accuracy: Test accuracy
    """
    model.to(device)
    model.eval()
    
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            texts, text_lengths, labels = batch
            texts = texts.to(device)
            text_lengths = text_lengths.to(device)
            labels = labels.to(device)
            
            # Forward pass
            predictions = model(texts, text_lengths)
            
            # Calculate loss
            loss = criterion(predictions, labels)
            test_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(predictions, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    
    print(f"Test Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%")
    
    return avg_loss, accuracy


def save_model(model, filepath):
    """
    Save a model to disk.
    
    Args:
        model: PyTorch model to save
        filepath: Path to save the model
    """
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def load_model(model, filepath, device):
    """
    Load a model from disk.
    
    Args:
        model: PyTorch model (architecture)
        filepath: Path to load the model from
        device: Device to load the model on
        
    Returns:
        model: Loaded model
    """
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    print(f"Model loaded from {filepath}")
    return model
