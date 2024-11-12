from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class EncoderConfig:
    """Configuration for the numerical encoder"""
    embedding_dim: int = 768  # Matches BERT dimension
    num_heads: int = 8
    dropout: float = 0.1
    max_sequence_length: int = 512
    positional_encoding_base: int = 10000
    num_magnitude_bins: int = 10
    
@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics"""
    num_clusters: int = 10
    num_synthetic_samples: int = 1000
    seed: int = 42
    batch_size: int = 32
    numerical_test_ranges: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0, 1),           # Small numbers
        (1, 1000),       # Medium numbers
        (1000, 1e6),     # Large numbers
        (-1000, 1000),   # Mixed positive/negative
        (0.0001, 0.1)    # Small decimals
    ])
