import torch
import transformers
import numpy as np
from numerical_encoding import NumericalEncoder

def test_environment():
    print("Testing environment setup...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create encoder
    encoder = NumericalEncoder()
    encoder.to(device)
    
    # Test standalone number
    number = 3.14
    number_embedding = encoder.encode_number(number)
    print(f"\nSuccessfully encoded number: {number}")
    print(f"Embedding shape: {number_embedding.shape}")
    
    # Test text with number
    text = "Rated 5 stars"
    text_embedding = encoder.encode_number(text)
    print(f"\nSuccessfully encoded text: '{text}'")
    print(f"Embedding shape: {text_embedding.shape}")
    
    print("\nComputing similarity...")
    # Ensure both embeddings are on the same device
    number_embedding = number_embedding.to(device)
    text_embedding = text_embedding.to(device)
    
    similarity = encoder.compute_similarity(number_embedding, text_embedding)
    print(f"Similarity between embeddings: {similarity:.3f}")
    
    print("\nEnvironment test completed successfully! 🎉")

if __name__ == "__main__":
    test_environment()