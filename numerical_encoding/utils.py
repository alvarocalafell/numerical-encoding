"""
Utility functions for numerical encoding and evaluation.
"""

import re
from typing import List, Tuple, Optional
import torch
import numpy as np

def extract_numbers_with_positions(text: str) -> List[Tuple[float, int, int]]:
    """Extract numbers and their positions from text.
    
    Args:
        text: Input text containing numbers
        
    Returns:
        List[Tuple[float, int, int]]: List of (number, start_pos, end_pos)
    """
    number_pattern = r"""
        (-?\d*\.?\d+)     # Basic number format with optional decimal
        ([eE][+-]?\d+)?   # Optional scientific notation
    """
    
    matches = []
    for match in re.finditer(number_pattern, text, re.VERBOSE):
        try:
            value = float(match.group())
            matches.append((value, match.start(), match.end()))
        except ValueError:
            continue
    
    return matches

def get_context_type(text: str) -> str:
    """Identify the context type of a number in text.
    
    Args:
        text: Input text containing numbers
        
    Returns:
        str: Context type identifier
    """
    text = text.lower()
    context_patterns = {
        'rating': ['star', 'rate', 'score', 'rating'],
        'price': ['$', 'price', 'cost', 'dollar'],
        'quantity': ['item', 'unit', 'quantity', 'piece'],
        'time': ['hour', 'minute', 'day', 'year'],
        'percentage': ['%', 'percent', 'percentage']
    }
    
    for context, patterns in context_patterns.items():
        if any(pattern in text for pattern in patterns):
            return context
    return 'other'

def compute_cosine_similarity(
    emb1: torch.Tensor,
    emb2: torch.Tensor
) -> float:
    """Compute cosine similarity between two embeddings.
    
    Args:
        emb1: First embedding
        emb2: Second embedding
        
    Returns:
        float: Cosine similarity score
    """
    return torch.nn.functional.cosine_similarity(
        emb1.unsqueeze(0),
        emb2.unsqueeze(0)
    ).item()

def generate_synthetic_data(
    num_samples: int,
    ranges: List[Tuple[float, float]],
    seed: Optional[int] = None
) -> np.ndarray:
    """Generate synthetic numerical data for testing.
    
    Args:
        num_samples: Number of samples to generate
        ranges: List of (min, max) ranges
        seed: Random seed for reproducibility
        
    Returns:
        np.ndarray: Generated data
    """
    if seed is not None:
        np.random.seed(seed)
    
    samples = []
    samples_per_range = num_samples // len(ranges)
    
    for range_min, range_max in ranges:
        # Linear space samples
        samples.extend(np.linspace(
            range_min, range_max,
            samples_per_range // 2
        ))
        
        # Log space samples if applicable
        if range_min > 0 and range_max > 0:
            samples.extend(np.logspace(
                np.log10(range_min),
                np.log10(range_max),
                samples_per_range // 2
            ))
    
    return np.array(sorted(samples))