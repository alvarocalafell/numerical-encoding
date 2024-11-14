# Numerical Encoding System for Machine Learning

An advanced encoding scheme for representing numerical data in both standalone and textual contexts, designed to preserve numerical relationships while maintaining contextual understanding. This system achieves state-of-the-art performance in numerical encoding tasks with an overall score of 0.722 across various evaluation metrics.

## 🔥 Key Features

- **Magnitude-aware numerical encoding** with 0.862 preservation score
- **Strong contextual separation** (1.000) between different number use cases
- **High classification performance** (0.988) in downstream tasks
- **Robust scale invariance** (0.772) across numerical ranges
- **Effective semantic preservation** for numbers in context

## 📊 Performance Metrics

### Numerical Understanding
| Metric | Score |
|--------|--------|
| Magnitude Preservation | 0.862 |
| Numerical Continuity | 0.847 |
| Scale Invariance | 0.772 |
| Interval Preservation | 0.506 |
| Relative Distance | 0.502 |
| **Category Average** | **0.698** |

### Contextual Processing
| Metric | Score |
|--------|--------|
| Context Separation | 1.000 |
| Cross Context Understanding | 0.772 |
| Semantic Preservation | 0.581 |
| **Category Average** | **0.784** |

### Downstream Applications
| Metric | Score |
|--------|--------|
| Classification Performance | 0.988 |
| Clustering Quality | 0.595 |
| Semantic Similarity | 0.512 |
| **Category Average** | **0.699** |

**Overall System Score: 0.722**

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/alvarocalafell/numerical-encoding.git
cd numerical-encoding

# Create virtual environment
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
pytest>=7.0.0
black>=22.0.0
isort>=5.10.1
mypy>=0.950
rich>=13.3.1
```

## 💻 Basic Usage

```python
from numerical_encoding import NumericalEncoder

# Initialize encoder
encoder = NumericalEncoder()

# Encode standalone numbers
num_embedding = encoder.encode_number(3.14)

# Encode numbers in text
text_embedding = encoder.encode_number("Rated 5 stars")
```


## 🔍 Architecture Overview

The system employs a three-component architecture:

1. **MagnitudeAwareEncoding**
   - Preserves numerical scale and relationships
   - Handles numbers from very small to very large
   - Maintains sign and decimal precision

2. **ContextualEncoder**
   - Processes numbers in textual context
   - Preserves semantic meaning
   - Differentiates between usage contexts

3. **IntegratedAttention**
   - Combines numerical and contextual information
   - Maintains consistency across representations
   - Enables downstream task performance

## 📈 Evaluation

Run the complete evaluation suite:

```bash
python run_evaluation.py --num-samples 1000 --save-results
```

## 🔧 Advanced Configuration

```python
from numerical_encoding import EncoderConfig

config = EncoderConfig(
    embedding_dim=768,
    num_heads=8,
    dropout=0.1,
    max_sequence_length=512
)

encoder = NumericalEncoder(config)
```

## 🤝 Contributing

Contributions welcome! Please check our [Contributing Guidelines](CONTRIBUTING.md).

## 📝 Citation

```bibtex
@software{calafell2024numerical,
  author = {Calafell, Alvaro},
  title = {Numerical Encoding System for Machine Learning},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/username/numerical-encoding}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Performance Highlights

The system achieves particularly strong results in:
- Context separation (1.000)
- Classification tasks (0.988)
- Magnitude preservation (0.862)
- Numerical continuity (0.847)
- Scale invariance (0.772)

Areas for future improvement include:
- Interval preservation (0.506)
- Semantic similarity (0.512)
- Clustering quality (0.595)

