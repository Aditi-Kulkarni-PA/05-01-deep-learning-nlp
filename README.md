# Deep Learning NLP

A comprehensive toolkit for natural language processing tasks using deep learning frameworks including PyTorch and Transformers.

## Features

- **Text Classification**: LSTM-based and transformer-based text classifiers
- **Sequence-to-Sequence Models**: Neural machine translation and text generation
- **Pre-trained Models**: Easy integration with Hugging Face transformers
- **Data Processing**: Text preprocessing, tokenization, and vocabulary building
- **Training Utilities**: Ready-to-use training and evaluation functions
- **Examples**: Complete examples and Jupyter notebooks

## Project Structure

```
deep-learning-nlp/
├── src/
│   └── deep_learning_nlp/
│       ├── __init__.py           # Package initialization
│       ├── models.py              # Neural network models (LSTM, Seq2Seq)
│       ├── data_processing.py    # Data preprocessing and loading
│       └── utils.py               # Training and evaluation utilities
├── examples/
│   ├── train_classifier.py       # Text classification example
│   └── transformer_examples.py   # Pre-trained transformer examples
├── notebooks/
│   └── text_classification_tutorial.ipynb  # Interactive tutorial
├── tests/
│   ├── test_models.py            # Model unit tests
│   ├── test_data_processing.py   # Data processing tests
│   └── test_utils.py             # Utility tests
├── data/
│   ├── raw/                      # Raw data directory
│   └── processed/                # Processed data directory
├── requirements.txt              # Package dependencies
├── setup.py                      # Package setup file
└── README.md                     # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install from source

1. Clone the repository:
```bash
git clone https://github.com/Aditi-Kulkarni-PA/deep-learning-nlp.git
cd deep-learning-nlp
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package:
```bash
pip install -e .
```

### Install with development dependencies

```bash
pip install -e ".[dev]"
```

## Quick Start

### Text Classification Example

```python
import torch
from deep_learning_nlp.models import TextClassifier
from deep_learning_nlp.data_processing import TextPreprocessor

# Initialize preprocessor
preprocessor = TextPreprocessor(lowercase=True)

# Build vocabulary from your texts
texts = ["This is great!", "Not good at all"]
word2idx, idx2word = preprocessor.build_vocab(texts)

# Create model
model = TextClassifier(
    vocab_size=len(word2idx),
    embedding_dim=100,
    hidden_dim=256,
    output_dim=2
)

print(model)
```

### Using Pre-trained Transformers

```python
from transformers import pipeline

# Sentiment analysis
classifier = pipeline("sentiment-analysis")
result = classifier("I love this library!")
print(result)
```

## Examples

Run the provided examples to get started:

### Basic Text Classifier

```bash
cd examples
python train_classifier.py
```

### Transformer Examples

```bash
cd examples
python transformer_examples.py
```

### Jupyter Notebook Tutorial

```bash
jupyter notebook notebooks/text_classification_tutorial.ipynb
```

## Usage

### 1. Data Preprocessing

```python
from deep_learning_nlp.data_processing import TextPreprocessor

preprocessor = TextPreprocessor(lowercase=True, remove_punctuation=False)

# Clean text
cleaned = preprocessor.clean_text("Hello, World!")

# Tokenize
tokens = preprocessor.tokenize("This is a test")

# Build vocabulary
word2idx, idx2word = preprocessor.build_vocab(texts, min_freq=2)

# Encode text
encoded = preprocessor.encode_text("hello world", word2idx)
```

### 2. Training a Model

```python
from deep_learning_nlp.utils import train_model, evaluate_model
import torch.nn as nn

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train
model, losses = train_model(
    model, train_loader, criterion, optimizer, device, num_epochs=10
)

# Evaluate
test_loss, accuracy = evaluate_model(model, test_loader, criterion, device)
```

### 3. Saving and Loading Models

```python
from deep_learning_nlp.utils import save_model, load_model

# Save model
save_model(model, "my_model.pth")

# Load model
loaded_model = load_model(model, "my_model.pth", device)
```

## Testing

Run the test suite:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest --cov=deep_learning_nlp tests/
```

## Available Models

### TextClassifier

LSTM-based text classification model with the following features:
- Bidirectional LSTM support
- Configurable embedding and hidden dimensions
- Dropout for regularization
- Support for multi-class classification

### SequenceToSequence

Encoder-decoder architecture for sequence-to-sequence tasks:
- Machine translation
- Text summarization
- Question answering

## Dependencies

Core dependencies:
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- NumPy >= 1.24.0
- scikit-learn >= 1.3.0

See `requirements.txt` for complete list.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with PyTorch and Hugging Face Transformers
- Inspired by best practices in deep learning for NLP

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{deep_learning_nlp,
  title = {Deep Learning NLP Toolkit},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/Aditi-Kulkarni-PA/deep-learning-nlp}
}
```

## Contact

For questions or feedback, please open an issue on GitHub.
