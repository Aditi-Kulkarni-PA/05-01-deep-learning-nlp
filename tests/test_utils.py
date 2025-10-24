"""
Unit tests for utils module.
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import os

from deep_learning_nlp.models import TextClassifier
from deep_learning_nlp.utils import save_model, load_model


class TestModelSaving:
    """Tests for model saving and loading."""
    
    def test_save_and_load_model(self):
        """Test saving and loading a model."""
        # Create a simple model
        model = TextClassifier(
            vocab_size=100,
            embedding_dim=50,
            hidden_dim=64,
            output_dim=2,
            n_layers=1,
            bidirectional=False,
            dropout=0.0
        )
        
        # Get initial parameters
        initial_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # Save model to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.pth') as f:
            filepath = f.name
        
        try:
            save_model(model, filepath)
            
            # Create new model instance
            new_model = TextClassifier(
                vocab_size=100,
                embedding_dim=50,
                hidden_dim=64,
                output_dim=2,
                n_layers=1,
                bidirectional=False,
                dropout=0.0
            )
            
            # Load saved weights
            device = torch.device('cpu')
            loaded_model = load_model(new_model, filepath, device)
            
            # Check that parameters match
            for name, param in loaded_model.named_parameters():
                assert torch.allclose(param, initial_params[name])
                
        finally:
            # Clean up
            if os.path.exists(filepath):
                os.remove(filepath)
