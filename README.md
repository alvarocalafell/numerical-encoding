# Numerical Encoding System for Machine Learning

An advanced encoding scheme for representing numerical data in both standalone and textual contexts. This project implements novel techniques for preserving numerical relationships while maintaining contextual understanding, making it suitable for various downstream machine learning tasks.

## 🔍 Overview

This project addresses the challenge of effectively encoding numerical data for machine learning applications. It combines recent advances in transformer architectures with specialized numerical representation techniques to create a versatile encoding system.

Key features:
- Magnitude-aware numerical encoding
- Contextual understanding of numbers in text
- Scale-invariant representations
- Cross-context consistency preservation

## 🏗️ Architecture

The system consists of two main components:

1. **Numerical Encoder:**
   - Periodic embeddings for continuous numerical values
   - Magnitude-aware binning system
   - Custom attention mechanism for numerical awareness

2. **Evaluation Framework:**
   - Comprehensive metric suite for numerical properties
   - Contextual understanding evaluation
   - Synthetic data generation for thorough testing

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/alvarocalafell/numerical-encoding.git
cd numerical-encoding

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📦 Requirements

```
torch>=1.9.0
transformers>=4.5.0
numpy>=1.19.5
scipy>=1.7.0
scikit-learn>=0.24.2
pandas>=1.3.0
matplotlib>=3.4.2
seaborn>=0.11.1
```

## 💻 Usage

Basic usage example:

```python
from numerical_encoding import NumericalEncoder

# Initialize encoder
encoder = NumericalEncoder()

# Encode standalone number
embedding = encoder.encode_number(3.14)

# Encode number in text
text_embedding = encoder.encode_number("Rated 5 stars")

# Compare embeddings
similarity = encoder.compute_similarity(embedding1, embedding2)
```

Run evaluation:

```python
from numerical_encoding.evaluation import run_evaluation

# Run complete evaluation suite
run_evaluation()
```

## 📊 Evaluation Metrics

The evaluation framework includes:

1. **Numerical Properties:**
   - Monotonicity preservation
   - Scale invariance
   - Sign preservation
   - Relative magnitude preservation

2. **Contextual Understanding:**
   - Context differentiation
   - Semantic clustering
   - Cross-context consistency

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{calafell2024numerical,
  author = {Calafell, Alvaro},
  title = {Numerical Encoding System for Machine Learning},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/alvarocalafell/numerical-encoding}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
