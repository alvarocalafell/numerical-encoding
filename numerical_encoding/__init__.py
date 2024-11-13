"""
Numerical Encoding System for Machine Learning
============================================

This package provides advanced encoding schemes for representing numerical data
in both standalone and textual contexts, making it suitable for various
downstream machine learning tasks.

Main Components
--------------
- NumericalEncoder: Core encoder class for handling numbers and text
- EncoderConfig: Configuration settings for the encoder
- EvaluationConfig: Configuration for evaluation metrics
- NumericalEncodingEvaluator: Comprehensive evaluation framework

Example
-------
>>> from numerical_encoding import NumericalEncoder
>>> encoder = NumericalEncoder()
>>> embedding = encoder.encode_number(3.14)
>>> text_embedding = encoder.encode_number("Rated 5 stars")
"""

from numerical_encoding.encoder import NumericalEncoder
from numerical_encoding.config import EncoderConfig
from numerical_encoding.evaluation import (
    EvaluationConfig,
    NumericalEncodingEvaluator,
    NumericalEncodingDataset,
)

__version__ = "1.0.0"
__author__ = "Alvaro Calafell"

__all__ = [
    "NumericalEncoder",
    "EncoderConfig",
    "EvaluationConfig",
    "NumericalEncodingEvaluator",
    "NumericalEncodingDataset",
]