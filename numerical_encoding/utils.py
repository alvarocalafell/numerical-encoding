import torch
import numpy as np
from typing import List, Tuple, Dict
import re

def extract_numbers_with_positions(text: str) -> List[Tuple[float, int, int]]:
    """Extract numbers and their positions from text"""
    number_pattern = r'-?\d*\.?\d+'
    matches = list(re.finditer(number_pattern, text))
    return [(float(m.group()), m.start(), m.end()) for m in matches]

def get_context_type(text: str) -> str:
    """Identify the context type of a number in text"""
    text = text.lower()
    contexts = {
        'rating': ['star', 'rate', 'score', 'rating'],
        'price': ['$', 'price', 'cost', 'dollar'],
        'quantity': ['item', 'unit', 'quantity', 'piece'],
        'time': ['hour', 'minute', 'day', 'year'],
        'percentage': ['%', 'percent']
    }
    
    for context, keywords in contexts.items():
        if any(keyword in text for keyword in keywords):
            return context
    return 'other'

def compute_cosine_similarity(
    emb1: torch.Tensor,
    emb2: torch.Tensor
) -> float:
    """Compute cosine similarity between two embeddings"""
    return torch.nn.functional.cosine_similarity(
        emb1.unsqueeze(0),
        emb2.unsqueeze(0)
    ).item()

def generate_synthetic_data(
    num_samples: int,
    ranges: List[Tuple[float, float]],
    seed: int = 42
) -> np.ndarray:
    """Generate synthetic numerical data for testing"""
    np.random.seed(seed)
    data = []
    samples_per_range = num_samples // len(ranges)
    
    for range_min, range_max in ranges:
        # Linear space samples
        data.extend(np.linspace(range_min, range_max, samples_per_range // 2))
        # Log space samples if applicable
        if range_min > 0 and range_max > 0:
            data.extend(np.logspace(
                np.log10(range_min),
                np.log10(range_max),
                samples_per_range // 2
            ))
    
    return np.array(data)

def create_text_templates() -> Dict[str, List[str]]:
    """Create templates for generating text data"""
    return {
        'rating': [
            'rated {} stars',
            'gave it {} points',
            'score of {}'
        ],
        'price': [
            'costs ${}.00',
            'priced at ${}.99',
            '${} total'
        ],
        'quantity': [
            '{} items',
            '{} units available',
            'quantity: {}'
        ],
        'time': [
            '{} hours',
            '{} minutes',
            '{} days old'
        ],
        'percentage': [
            '{}% complete',
            '{}% increase',
            'grew by {}%'
        ]
    }
