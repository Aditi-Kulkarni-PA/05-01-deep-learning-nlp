"""
Example script for using pre-trained transformers.
"""

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch


def sentiment_analysis_example():
    """
    Example of sentiment analysis using a pre-trained transformer model.
    """
    print("=" * 50)
    print("Sentiment Analysis Example")
    print("=" * 50)
    
    # Load pre-trained sentiment analysis pipeline
    classifier = pipeline("sentiment-analysis")
    
    # Example texts
    texts = [
        "I love this product! It's amazing.",
        "This is the worst experience I've ever had.",
        "It's okay, nothing special.",
    ]
    
    print("\nAnalyzing sentiments...")
    for text in texts:
        result = classifier(text)[0]
        print(f"\nText: {text}")
        print(f"Sentiment: {result['label']} (confidence: {result['score']:.4f})")


def text_classification_example():
    """
    Example of zero-shot text classification.
    """
    print("\n" + "=" * 50)
    print("Zero-Shot Text Classification Example")
    print("=" * 50)
    
    # Load zero-shot classification pipeline
    classifier = pipeline("zero-shot-classification")
    
    # Example text and candidate labels
    text = "The new iPhone has an amazing camera and long battery life."
    candidate_labels = ["technology", "sports", "politics", "health"]
    
    print(f"\nText: {text}")
    print(f"Candidate labels: {candidate_labels}")
    
    result = classifier(text, candidate_labels)
    
    print("\nClassification results:")
    for label, score in zip(result['labels'], result['scores']):
        print(f"  {label}: {score:.4f}")


def text_generation_example():
    """
    Example of text generation using a pre-trained model.
    """
    print("\n" + "=" * 50)
    print("Text Generation Example")
    print("=" * 50)
    
    # Load text generation pipeline
    generator = pipeline("text-generation", model="gpt2")
    
    # Example prompt
    prompt = "Once upon a time in a distant galaxy"
    
    print(f"\nPrompt: {prompt}")
    print("\nGenerated text:")
    
    results = generator(prompt, max_length=50, num_return_sequences=1)
    print(results[0]['generated_text'])


def named_entity_recognition_example():
    """
    Example of named entity recognition.
    """
    print("\n" + "=" * 50)
    print("Named Entity Recognition Example")
    print("=" * 50)
    
    # Load NER pipeline
    ner = pipeline("ner", grouped_entities=True)
    
    # Example text
    text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
    
    print(f"\nText: {text}")
    print("\nEntities found:")
    
    entities = ner(text)
    for entity in entities:
        print(f"  {entity['word']}: {entity['entity_group']} (confidence: {entity['score']:.4f})")


def main():
    """
    Run all transformer examples.
    """
    print("Running Transformer Examples")
    print("These examples use pre-trained models from Hugging Face")
    print()
    
    try:
        sentiment_analysis_example()
        text_classification_example()
        text_generation_example()
        named_entity_recognition_example()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure you have installed the required packages:")
        print("  pip install transformers torch")


if __name__ == "__main__":
    main()
