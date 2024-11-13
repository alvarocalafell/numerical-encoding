"""
Configuration module for the numerical encoding system.
Provides dataclass-based configurations for both encoder and evaluation components.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import torch

@dataclass
class EncoderConfig:
    """Configuration settings for the numerical encoder.
    
    Attributes:
        embedding_dim (int): Dimension of output embeddings, matching BERT
        num_heads (int): Number of attention heads for multi-head attention
        dropout (float): Dropout rate for regularization
        max_sequence_length (int): Maximum sequence length for text input
        positional_encoding_base (int): Base for positional encoding calculations
        num_magnitude_bins (int): Number of bins for magnitude-aware encoding
        device (torch.device): Device to run computations on
    """
    embedding_dim: int = 768  # Matches BERT dimension
    num_heads: int = 8
    dropout: float = 0.1
    max_sequence_length: int = 512
    positional_encoding_base: int = 10000
    num_magnitude_bins: int = 10
    device: torch.device = field(
        default_factory=lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be divisible "
                f"by num_heads ({self.num_heads})"
            )
        if self.dropout < 0 or self.dropout > 1:
            raise ValueError(f"dropout must be between 0 and 1, got {self.dropout}")
    
    @property
    def head_dim(self) -> int:
        """Calculate dimension per attention head."""
        return self.embedding_dim // self.num_heads

@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics and testing.
    
    Attributes:
        num_clusters (int): Number of clusters for clustering evaluation
        num_synthetic_samples (int): Number of synthetic samples to generate
        seed (int): Random seed for reproducibility
        batch_size (int): Batch size for evaluation
        numerical_test_ranges (List[Tuple[float, float]]): Ranges for numerical testing
    """
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
    
    def __post_init__(self):
        """Validate evaluation configuration parameters."""
        if self.num_synthetic_samples < self.batch_size:
            raise ValueError(
                f"num_synthetic_samples ({self.num_synthetic_samples}) must be >= "
                f"batch_size ({self.batch_size})"
            )
        if not self.numerical_test_ranges:
            raise ValueError("numerical_test_ranges cannot be empty")