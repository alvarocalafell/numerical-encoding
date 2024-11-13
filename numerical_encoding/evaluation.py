"""
Evaluation framework for numerical encoding system.
Provides comprehensive metrics and evaluation tools for both numerical and contextual aspects.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path

from .utils import extract_numbers_with_positions, get_context_type
from .templates import create_text_templates, get_context_templates
from numerical_encoding.encoder import NumericalEncoder



@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics and testing.
    
    Attributes:
        num_clusters: Number of clusters for clustering evaluation
        num_synthetic_samples: Number of synthetic samples to generate
        seed: Random seed for reproducibility
        batch_size: Batch size for evaluation
        numerical_test_ranges: Ranges for numerical testing
        context_types: Types of contexts to evaluate
        semantic_test_cases: Test cases for semantic evaluation
        ordinal_sequences: Sequences for ordinal evaluation
        downstream_tasks: Configuration for downstream tasks
        evaluation_output_dir: Directory for saving evaluation results
        visualization_config: Configuration for result visualization
    """
    
    # Basic configuration
    num_clusters: int = 10
    num_synthetic_samples: int = 1000
    seed: int = 42
    batch_size: int = 32
    
    # Numerical ranges for testing
    numerical_test_ranges: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0, 1),           # Small numbers
        (1, 1000),       # Medium numbers
        (1000, 1e6),     # Large numbers
        (-1000, 1000),   # Mixed positive/negative
        (0.0001, 0.1)    # Small decimals
    ])
    
    # Context configuration
    context_types: List[str] = field(default_factory=lambda: [
        'rating',
        'price',
        'quantity',
        'time',
        'percentage'
    ])
    
    # Semantic test configuration
    semantic_test_cases: List[Tuple[str, ...]] = field(default_factory=lambda: [
        ("5 stars", "4 stars", "3 stars"),
        ("$5", "$50", "$500"),
        ("5 items", "50 items", "500 items"),
        ("5 minutes", "50 minutes", "500 minutes"),
        ("5%", "50%", "100%")
    ])
    
    # Ordinal evaluation configuration
    ordinal_sequences: Dict[str, List[str]] = field(default_factory=lambda: {
        "rating": ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"],
        "price": ["$10", "$20", "$50", "$100", "$200"],
        "quantity": ["5 units", "10 units", "20 units", "50 units", "100 units"],
        "time": ["1 minute", "5 minutes", "10 minutes", "30 minutes", "60 minutes"],
        "percentage": ["10%", "25%", "50%", "75%", "100%"]
    })
    
    # Downstream task configuration
    downstream_tasks: Dict[str, Dict] = field(default_factory=lambda: {
        "classification": {
            "enabled": True,
            "test_size": 0.2,
            "metrics": ["accuracy", "f1"]
        },
        "range_prediction": {
            "enabled": True,
            "ranges": [(0, 10), (10, 50), (50, 100), (100, float('inf'))],
            "test_size": 0.2
        },
        "magnitude_comparison": {
            "enabled": True,
            "similarity_threshold": 0.9
        },
        "contextual_regression": {
            "enabled": True,
            "min_samples_per_context": 10,
            "test_size": 0.2
        },
        "semantic_grouping": {
            "enabled": True,
            "min_samples_per_group": 5
        }
    })
    
    # Output configuration
    evaluation_output_dir: Optional[Path] = None
    
    # Visualization configuration
    visualization_config: Dict[str, Dict] = field(default_factory=lambda: {
        "figures": {
            "width": 12,
            "height": 6,
            "dpi": 100
        },
        "colors": {
            "numerical": "blue",
            "contextual": "green",
            "downstream": "orange",
            "ordinal": "purple"
        },
        "plot_types": ["bar", "heatmap", "scatter"],
        "save_formats": ["png", "pdf"]
    })
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        # Validate basic parameters
        if self.num_synthetic_samples < self.batch_size:
            raise ValueError(
                f"num_synthetic_samples ({self.num_synthetic_samples}) must be >= "
                f"batch_size ({self.batch_size})"
            )
        
        if not self.numerical_test_ranges:
            raise ValueError("numerical_test_ranges cannot be empty")
        
        # Convert output directory to Path if provided
        if self.evaluation_output_dir is not None:
            self.evaluation_output_dir = Path(self.evaluation_output_dir)
            self.evaluation_output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_context_template(self, context_type: str) -> str:
        """Get template string for a given context type."""
        templates = {
            'rating': '{} stars',
            'price': '${}'.format,
            'quantity': '{} units',
            'time': '{} minutes',
            'percentage': '{}%'
        }
        return templates.get(context_type, '{}')
    
    def get_task_config(self, task_name: str) -> Dict:
        """Get configuration for a specific downstream task."""
        if task_name not in self.downstream_tasks:
            raise ValueError(f"Unknown task: {task_name}")
        return self.downstream_tasks[task_name]
    
    def get_visualization_config(self, plot_type: str) -> Dict:
        """Get visualization configuration for a specific plot type."""
        if plot_type not in self.visualization_config["plot_types"]:
            raise ValueError(f"Unknown plot type: {plot_type}")
        return {
            **self.visualization_config["figures"],
            "color": self.visualization_config["colors"].get(plot_type, "blue")
        }
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary format."""
        return {
            "basic_config": {
                "num_clusters": self.num_clusters,
                "num_synthetic_samples": self.num_synthetic_samples,
                "seed": self.seed,
                "batch_size": self.batch_size
            },
            "numerical_config": {
                "test_ranges": self.numerical_test_ranges
            },
            "context_config": {
                "types": self.context_types,
                "semantic_tests": self.semantic_test_cases,
                "ordinal_sequences": self.ordinal_sequences
            },
            "downstream_config": self.downstream_tasks,
            "visualization_config": self.visualization_config
        }



@dataclass
class EvaluationMetric:
    """Container for evaluation metric results and metadata.
    
    Attributes:
        name: Name of the metric
        value: Computed metric value
        description: Description of what the metric measures
        category: Category of the metric (numerical/contextual/etc)
    """
    name: str
    value: float
    description: str
    category: str
    
    def to_dict(self) -> Dict[str, Union[str, float]]:
        """Convert metric to dictionary format."""
        return {
            "name": self.name,
            "value": self.value,
            "description": self.description,
            "category": self.category
        }

class NumericalEncodingDataset(Dataset):
    """Dataset for evaluating numerical encoding performance.
    
    Generates synthetic data covering various numerical ranges and contexts.
    
    Attributes:
        config: Evaluation configuration
        numerical_data: Generated numerical samples
        text_data: Generated text samples with numbers
        total_samples: Total number of samples in dataset
    """
    
    def __init__(self, config: 'EvaluationConfig'):
        """
        Initialize dataset with given configuration.
        
        Args:
            config: Evaluation configuration object
        """
        self.config = config
        
        # Set seeds for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Generate data
        self._generate_data()
    
    def _generate_data(self) -> None:
        """Generate and store synthetic evaluation data."""
        self.numerical_data = self._generate_numerical_samples()
        self.text_data = self._generate_textual_samples()
        
        # Store sample counts
        self.total_numerical = len(self.numerical_data)
        self.total_text = len(self.text_data)
        self.total_samples = self.total_numerical + self.total_text
    
    def _generate_numerical_samples(self) -> List[float]:
        """Generate numerical samples across different ranges."""
        samples = []
        samples_per_range = self.config.num_synthetic_samples // (
            len(self.config.numerical_test_ranges) * 2
        )
        
        for range_min, range_max in self.config.numerical_test_ranges:
            # Linear space samples
            linear_samples = np.linspace(
                range_min, range_max,
                samples_per_range
            )
            samples.extend(linear_samples)
            
            # Log space samples for positive ranges
            if range_min > 0 and range_max > 0:
                log_samples = np.logspace(
                    np.log10(range_min),
                    np.log10(range_max),
                    samples_per_range
                )
                samples.extend(log_samples)
        
        return sorted(samples)
    
    def _generate_textual_samples(self) -> List[str]:
        """Generate text samples with embedded numbers."""
        templates = create_text_templates()
        samples = []
        
        # Ensure deterministic order
        for context_type in sorted(templates.keys()):
            templates_for_context = sorted(templates[context_type])
            
            for template in templates_for_context:
                # Generate deterministic number ranges
                if context_type == 'rating':
                    numbers = np.linspace(1, 5, 20)  # Ratings from 1-5
                elif context_type == 'price':
                    numbers = np.logspace(0, 4, 20)  # Prices from 1-10000
                elif context_type == 'quantity':
                    numbers = np.linspace(1, 100, 20)  # Quantities from 1-100
                elif context_type == 'time':
                    numbers = np.linspace(1, 1000, 20)  # Time values
                elif context_type == 'percentage':
                    numbers = np.linspace(0, 100, 20)  # Percentages from 0-100
                
                # Format numbers and create samples
                samples.extend([
                    template.format(f"{num:.2f}") 
                    for num in numbers
                ])
        
        return samples
    
    def __len__(self) -> int:
        """Get total number of samples."""
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Tuple[Optional[float], Optional[str]]:
        """Get sample by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple containing either (number, None) or (None, text)
        """
        if idx < self.total_numerical:
            return self.numerical_data[idx], None
        else:
            text_idx = idx - self.total_numerical
            return None, self.text_data[text_idx]

def custom_collate_fn(
    batch: List[Tuple[Optional[float], Optional[str]]]
) -> Tuple[Optional[torch.Tensor], List[str]]:
    """Custom collate function for batching mixed numerical and text data.
    
    Args:
        batch: List of (number, text) tuples
        
    Returns:
        Tuple of numerical tensor and list of text samples
    """
    numerical_samples = []
    text_samples = []
    
    for num, text in batch:
        if num is not None:
            numerical_samples.append(num)
        if text is not None:
            text_samples.append(text)
    
    # Convert numerical samples to tensor if present
    if numerical_samples:
        numerical_samples = torch.tensor(
            numerical_samples,
            dtype=torch.float32
        )
    else:
        numerical_samples = None
    
    return numerical_samples, text_samples


class NumericalEncodingEvaluator:
    """
    Comprehensive evaluation framework for numerical encoding.
    
    Provides metrics for:
    - Numerical relationship preservation
    - Contextual understanding
    - Cross-context consistency
    - Downstream task performance
    
    Attributes:
        encoder: Numerical encoder model
        config: Evaluation configuration
        device: Computation device (CPU/CUDA)
        metrics: Dictionary of computed metrics
    """
    
    def __init__(
        self,
        encoder: NumericalEncoder,
        config: EvaluationConfig = EvaluationConfig()
    ):
        """
        Initialize evaluator with encoder and configuration.
        
        Args:
            encoder: Numerical encoder to evaluate
            config: Evaluation configuration
        """
        self.encoder = encoder
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder.to(self.device)
        self.metrics: Dict[str, EvaluationMetric] = {}
    
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """
        Run complete evaluation suite.
        
        Args:
            loader: DataLoader containing evaluation samples
            
        Returns:
            Dictionary of metric names and values
        """
        print("Starting evaluation...")
        
        # Evaluate numerical properties
        numerical_results = self.evaluate_numerical_properties(loader)
        self.metrics.update(numerical_results)
        
        # Evaluate contextual understanding
        contextual_results = self.evaluate_contextual_understanding(loader)
        self.metrics.update(contextual_results)
        
        # Evaluate ordinal preservation
        ordinal_score = self.evaluate_ordinal_preservation()
        self.metrics["ordinal_preservation"] = EvaluationMetric(
            name="ordinal_preservation",
            value=ordinal_score,
            description="Preservation of ordinal relationships",
            category="numerical"
        )
        
        # Evaluate interval preservation
        interval_score = self.evaluate_interval_preservation()
        self.metrics["interval_preservation"] = EvaluationMetric(
            name="interval_preservation",
            value=interval_score,
            description="Preservation of interval relationships",
            category="numerical"
        )
        
        # Evaluate cross-context discrimination
        cross_context_score = self.evaluate_cross_context_discrimination()
        self.metrics["cross_context_discrimination"] = EvaluationMetric(
            name="cross_context_discrimination",
            value=cross_context_score,
            description="Discrimination between contexts",
            category="contextual"
        )
        
        # Evaluate downstream tasks
        downstream_results = self.evaluate_downstream_task()
        self.metrics.update(downstream_results)
        
        # Convert metrics to simple dictionary
        return {
            name: metric.value 
            for name, metric in self.metrics.items()
        }
    
    def save_results(self, save_dir: Union[str, Path]) -> None:
        """
        Save evaluation results and visualizations.
        
        Args:
            save_dir: Directory to save results
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics to file
        metrics_file = save_dir / "metrics.txt"
        with open(metrics_file, "w") as f:
            for name, metric in sorted(self.metrics.items()):
                f.write(f"{name}:\n")
                f.write(f"  Value: {metric.value:.4f}\n")
                f.write(f"  Description: {metric.description}\n")
                f.write(f"  Category: {metric.category}\n\n")
        
        # Create visualizations
        self.visualize_results(save_dir)
    
    def visualize_results(self, save_dir: Optional[Path] = None) -> None:
        """
        Create and optionally save visualization of results.
        
        Args:
            save_dir: Optional directory to save visualizations
        """
        # Create category-based visualization
        categories = {
            metric.category for metric in self.metrics.values()
        }
        
        for category in sorted(categories):
            category_metrics = {
                name: metric.value 
                for name, metric in self.metrics.items() 
                if metric.category == category
            }
            
            plt.figure(figsize=(10, 6))
            sns.barplot(
                x=list(category_metrics.keys()),
                y=list(category_metrics.values()),
                palette="husl"
            )
            
            plt.title(f"{category.title()} Metrics")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(save_dir / f"{category}_metrics.png")
                plt.close()
            else:
                plt.show()

    def _extract_context(self, text: str) -> str:
        """
        Extract the context type from text containing numbers.
        
        Args:
            text: Input text string
            
        Returns:
            str: Context type ('rating', 'price', 'quantity', 'time', 'percentage', or 'other')
        """
        text = text.lower()
        
        # Define context patterns with their associated keywords
        context_patterns = {
            'rating': ['star', 'rate', 'score', 'rating'],
            'price': ['$', 'price', 'cost', 'dollar'],
            'quantity': ['item', 'unit', 'quantity', 'piece'],
            'time': ['hour', 'minute', 'day', 'year'],
            'percentage': ['%', 'percent', 'percentage']
        }
        
        # Check each context pattern
        for context, patterns in context_patterns.items():
            if any(pattern in text for pattern in patterns):
                return context
                
        # Default context if no pattern matches
        return 'other'

    def evaluate_numerical_properties(
        self,
        loader: DataLoader
    ) -> Dict[str, EvaluationMetric]:
        """
        Evaluate numerical property preservation and relationships.
        
        Evaluates:
        1. Relative distance preservation between numbers
        2. Scale invariance across different magnitudes
        3. Sign sensitivity and preservation
        4. Magnitude order preservation
        5. Numerical continuity
        
        Args:
            loader: DataLoader containing numerical samples
            
        Returns:
            Dict[str, EvaluationMetric]: Dictionary of evaluation metrics
        """
        print("Evaluating numerical properties...")
        metrics = {}
        
        try:
            # Collect all numbers and their embeddings
            all_numbers = []
            all_embeddings = []
            
            for nums, _ in loader:
                if nums is not None:
                    for num in nums.numpy():
                        try:
                            # Get embedding and store with number
                            embedding = self.encoder.encode_number(float(num))
                            all_embeddings.append(embedding.detach().cpu().numpy())
                            all_numbers.append(num)
                        except Exception as e:
                            print(f"Error processing number {num}: {e}")
                            continue
            
            if all_numbers:
                all_numbers = np.array(all_numbers)
                all_embeddings = np.array([e.squeeze() for e in all_embeddings])
                
                # 1. Relative Distance Preservation
                relative_distance_score = self._evaluate_relative_distances(
                    all_numbers, all_embeddings
                )
                metrics["relative_distance"] = EvaluationMetric(
                    name="relative_distance",
                    value=relative_distance_score,
                    description="Preservation of relative distances between numbers",
                    category="numerical"
                )
                
                # 2. Scale Invariance
                scale_invariance_score = self._evaluate_scale_invariance(
                    all_numbers, all_embeddings
                )
                metrics["scale_invariance"] = EvaluationMetric(
                    name="scale_invariance",
                    value=scale_invariance_score,
                    description="Consistency across different scales",
                    category="numerical"
                )
                
                # 3. Sign Preservation
                sign_preservation_score = self._evaluate_sign_preservation(
                    all_numbers, all_embeddings
                )
                metrics["sign_preservation"] = EvaluationMetric(
                    name="sign_preservation",
                    value=sign_preservation_score,
                    description="Discrimination between positive and negative numbers",
                    category="numerical"
                )
                
                # 4. Magnitude Order Preservation
                magnitude_score = self._evaluate_magnitude_preservation(
                    all_numbers, all_embeddings
                )
                metrics["magnitude_preservation"] = EvaluationMetric(
                    name="magnitude_preservation",
                    value=magnitude_score,
                    description="Preservation of magnitude ordering",
                    category="numerical"
                )
                
                # 5. Numerical Continuity
                continuity_score = self._evaluate_numerical_continuity(
                    all_numbers, all_embeddings
                )
                metrics["numerical_continuity"] = EvaluationMetric(
                    name="numerical_continuity",
                    value=continuity_score,
                    description="Smoothness of numerical representations",
                    category="numerical"
                )
        
        except Exception as e:
            print(f"Error in numerical properties evaluation: {e}")
        
        return metrics

    def _evaluate_relative_distances(
        self,
        numbers: np.ndarray,
        embeddings: np.ndarray
    ) -> float:
        """
        Evaluate how well the embedding preserves relative distances between numbers.
        
        Args:
            numbers: Array of input numbers
            embeddings: Array of corresponding embeddings
            
        Returns:
            float: Correlation score between numerical and embedding distances
        """
        # Calculate pairwise differences in number space (log scale)
        num_diffs = np.log1p(np.abs(
            numbers[:, np.newaxis] - numbers[np.newaxis, :]
        ))
        
        # Calculate pairwise distances in embedding space
        emb_dists = np.linalg.norm(
            embeddings[:, np.newaxis] - embeddings[np.newaxis, :],
            axis=-1
        )
        
        # Compute rank correlation
        correlation, _ = spearmanr(num_diffs.flatten(), emb_dists.flatten())
        return max(0, correlation)  # Ensure non-negative score

    def evaluate_interval_preservation(self) -> float:
        """
        Evaluate how well the encoder preserves numerical intervals between numbers.
        Tests if the distance between embeddings is proportional to the numerical
        difference between the corresponding numbers.
        
        Returns:
            float: Interval preservation score between 0 and 1
        """
        print("Evaluating interval preservation...")
        
        try:
            # Generate sequence of numbers with known intervals
            intervals = [
                # Small intervals
                [1, 2, 3, 4, 5],
                # Medium intervals
                [10, 20, 30, 40, 50],
                # Large intervals
                [100, 200, 300, 400, 500],
                # Mixed intervals
                [1, 5, 10, 50, 100],
                # Decimal intervals
                [0.1, 0.2, 0.3, 0.4, 0.5]
            ]
            
            interval_scores = []
            
            for sequence in intervals:
                try:
                    # Get embeddings for sequence
                    embeddings = []
                    for num in sequence:
                        emb = self.encoder.encode_number(float(num))  # [1, 768]
                        embeddings.append(emb.detach().cpu().numpy().squeeze())
                    embeddings = np.array(embeddings)  # [seq_len, 768]
                    
                    # Calculate numerical differences
                    num_diffs = np.array([
                        abs(sequence[i+1] - sequence[i])
                        for i in range(len(sequence)-1)
                    ])  # [seq_len-1]
                    
                    # Calculate embedding distances
                    emb_dists = np.array([
                        np.linalg.norm(embeddings[i+1] - embeddings[i])
                        for i in range(len(embeddings)-1)
                    ])  # [seq_len-1]
                    
                    # Normalize differences and distances
                    num_diffs = num_diffs / np.max(num_diffs)
                    emb_dists = emb_dists / np.max(emb_dists)
                    
                    # Calculate correlation between differences and distances
                    correlation, _ = spearmanr(num_diffs, emb_dists)
                    interval_scores.append(max(0, correlation))
                    
                except Exception as e:
                    print(f"Error processing interval sequence {sequence}: {e}")
                    continue
            
            # Calculate final score as mean of all interval correlations
            final_score = np.mean(interval_scores) if interval_scores else 0.0
            
            return final_score
        
        except Exception as e:
            print(f"Error in interval preservation evaluation: {e}")
            return 0.0
        
    def evaluate_cross_context_discrimination(self) -> float:
        """
        Evaluate how well the encoder discriminates between the same numbers in different contexts.
        Tests if the same number is encoded differently based on its context (e.g., "5 stars" vs "$5").
        
        Returns:
            float: Cross-context discrimination score between 0 and 1
        """
        print("Evaluating cross-context discrimination...")
        
        try:
            # Define test cases with same numbers in different contexts
            test_cases = [
                # Test with small numbers
                {
                    'number': 5,
                    'contexts': [
                        "5 stars",          # rating
                        "$5",               # price
                        "5 items",          # quantity
                        "5 minutes",        # time
                        "5%"                # percentage
                    ]
                },
                # Test with medium numbers
                {
                    'number': 50,
                    'contexts': [
                        "rated 50 points",  # rating
                        "$50",              # price
                        "50 units",         # quantity
                        "50 minutes",       # time
                        "50%"               # percentage
                    ]
                },
                # Test with large numbers
                {
                    'number': 100,
                    'contexts': [
                        "100 point rating", # rating
                        "$100",             # price
                        "100 pieces",       # quantity
                        "100 hours",        # time
                        "100%"             # percentage
                    ]
                }
            ]
            
            discrimination_scores = []
            
            for test_case in test_cases:
                try:
                    # Get embeddings for each context
                    embeddings = []
                    for text in test_case['contexts']:
                        emb = self.encoder.encode_number(text)  # [1, 768]
                        embeddings.append(emb.detach().cpu().numpy().squeeze())
                    embeddings = np.array(embeddings)  # [num_contexts, 768]
                    
                    # Calculate pairwise similarities between contexts
                    similarities = np.zeros((len(embeddings), len(embeddings)))
                    for i in range(len(embeddings)):
                        for j in range(i + 1, len(embeddings)):
                            sim = np.dot(embeddings[i], embeddings[j]) / (
                                np.linalg.norm(embeddings[i]) * 
                                np.linalg.norm(embeddings[j])
                            )
                            similarities[i, j] = similarities[j, i] = sim
                    
                    # Calculate discrimination score
                    # We want similarities to be low (good discrimination between contexts)
                    # Convert similarities to distances: higher is better
                    distances = 1 - similarities
                    
                    # Calculate mean distance between different contexts
                    # Exclude diagonal (self-similarities)
                    mean_distance = np.sum(distances) / (len(embeddings) * (len(embeddings) - 1))
                    
                    # Scale to [0, 1] range and add to scores
                    discrimination_scores.append(mean_distance)
                    
                except Exception as e:
                    print(f"Error processing test case {test_case['number']}: {e}")
                    continue
            
            # Calculate final score as mean of all discrimination scores
            final_score = np.mean(discrimination_scores) if discrimination_scores else 0.0
            
            # Normalize score to ensure it's between 0 and 1
            final_score = min(max(final_score, 0.0), 1.0)
            
            return final_score
        
        except Exception as e:
            print(f"Error in cross-context discrimination evaluation: {e}")
            return 0.0

    def _evaluate_scale_invariance(
        self,
        numbers: np.ndarray,
        embeddings: np.ndarray,
        scales: np.ndarray = np.array([0.01, 0.1, 1.0, 10.0, 100.0])
    ) -> float:
        """
        Evaluate how consistent embeddings are across different scales.
        
        Args:
            numbers: Array of input numbers
            embeddings: Array of corresponding embeddings
            scales: Array of scale factors to test
            
        Returns:
            float: Scale invariance score
        """
        # Take a subset of numbers for efficiency
        subset_size = min(100, len(numbers))
        base_numbers = numbers[:subset_size]
        base_embeddings = embeddings[:subset_size]
        
        scale_similarities = []
        
        for scale in scales:
            scaled_embs = []
            for num in base_numbers * scale:
                try:
                    embed = self.encoder.encode_number(float(num))
                    scaled_embs.append(embed.detach().cpu().numpy().squeeze())
                except Exception:
                    continue
            
            if scaled_embs:
                scaled_embs = np.array(scaled_embs)
                similarities = []
                for base_emb, scaled_emb in zip(base_embeddings, scaled_embs):
                    similarity = np.dot(base_emb, scaled_emb) / (
                        np.linalg.norm(base_emb) * np.linalg.norm(scaled_emb)
                    )
                    similarities.append(similarity)
                scale_similarities.append(np.mean(similarities))
        
        return np.mean(scale_similarities) if scale_similarities else 0.0

    def _evaluate_sign_preservation(
        self,
        numbers: np.ndarray,
        embeddings: np.ndarray
    ) -> float:
        """
        Evaluate how well the embedding discriminates between positive and negative numbers.
        
        Args:
            numbers: Array of input numbers
            embeddings: Array of corresponding embeddings
            
        Returns:
            float: Sign preservation score
        """
        positive_mask = numbers > 0
        negative_mask = numbers < 0
        
        if not (np.any(positive_mask) and np.any(negative_mask)):
            return 0.0
        
        pos_embeddings = embeddings[positive_mask]
        neg_embeddings = embeddings[negative_mask]
        
        # Calculate centroids
        pos_centroid = pos_embeddings.mean(axis=0)
        neg_centroid = neg_embeddings.mean(axis=0)
        
        # Calculate separation between centroids
        centroid_separation = np.linalg.norm(pos_centroid - neg_centroid)
        
        # Calculate average within-class distance
        pos_variance = np.mean([
            np.linalg.norm(emb - pos_centroid) for emb in pos_embeddings
        ])
        neg_variance = np.mean([
            np.linalg.norm(emb - neg_centroid) for emb in neg_embeddings
        ])
        avg_variance = (pos_variance + neg_variance) / 2
        
        # Return normalized score
        return centroid_separation / (avg_variance + 1e-6)

    def _evaluate_magnitude_preservation(
        self,
        numbers: np.ndarray,
        embeddings: np.ndarray
    ) -> float:
        """
        Evaluate how well the embedding preserves magnitude ordering.
        
        Args:
            numbers: Array of input numbers
            embeddings: Array of corresponding embeddings
            
        Returns:
            float: Magnitude preservation score
        """
        # Sort numbers by absolute magnitude
        magnitude_order = np.argsort(np.abs(numbers))
        sorted_numbers = numbers[magnitude_order]
        sorted_embeddings = embeddings[magnitude_order]
        
        # Calculate embedding similarities
        similarities = np.zeros((len(numbers), len(numbers)))
        for i in range(len(numbers)):
            for j in range(i + 1, len(numbers)):
                sim = np.dot(sorted_embeddings[i], sorted_embeddings[j]) / (
                    np.linalg.norm(sorted_embeddings[i]) * 
                    np.linalg.norm(sorted_embeddings[j])
                )
                similarities[i, j] = similarities[j, i] = sim
        
        # Check if similarities decrease with magnitude difference
        correct_order = 0
        total_comparisons = 0
        
        for i in range(len(numbers) - 2):
            for j in range(i + 1, len(numbers) - 1):
                for k in range(j + 1, len(numbers)):
                    if similarities[i, j] > similarities[i, k]:
                        correct_order += 1
                    total_comparisons += 1
        
        return correct_order / total_comparisons if total_comparisons > 0 else 0.0

    def _evaluate_numerical_continuity(
        self,
        numbers: np.ndarray,
        embeddings: np.ndarray
    ) -> float:
        """
        Evaluate the smoothness/continuity of the numerical embeddings.
        
        Args:
            numbers: Array of input numbers
            embeddings: Array of corresponding embeddings
            
        Returns:
            float: Continuity score
        """
        # Sort by numerical value
        order = np.argsort(numbers)
        sorted_numbers = numbers[order]
        sorted_embeddings = embeddings[order]
        
        # Calculate embedding differences for consecutive numbers
        embedding_diffs = np.linalg.norm(
            sorted_embeddings[1:] - sorted_embeddings[:-1],
            axis=1
        )
        
        # Calculate number differences
        number_diffs = np.abs(sorted_numbers[1:] - sorted_numbers[:-1])
        
        # Calculate correlation between differences
        correlation, _ = spearmanr(number_diffs, embedding_diffs)
        
        # Calculate smoothness penalty
        smoothness = 1.0 - np.std(embedding_diffs) / (np.mean(embedding_diffs) + 1e-6)
        
        # Combine correlation and smoothness
        return (correlation + smoothness) / 2
    
    
    def evaluate_contextual_understanding(
        self,
        loader: DataLoader
    ) -> Dict[str, EvaluationMetric]:
        """
        Evaluate understanding of numbers in different contexts.
        
        Tests:
        1. Context clustering quality
        2. Context separation
        3. Semantic consistency
        4. Inter-context relationships
        5. Intra-context coherence
        
        Args:
            loader: DataLoader with text samples
            
        Returns:
            Dict[str, EvaluationMetric]: Dictionary of evaluation metrics
        """
        print("Evaluating contextual understanding...")
        metrics = {}
        
        try:
            # Collect embeddings by context
            context_data = self._collect_context_embeddings(loader)
            
            if context_data:
                # 1. Context Clustering Quality
                clustering_score = self._evaluate_context_clustering(context_data)
                metrics["clustering_quality"] = EvaluationMetric(
                    name="clustering_quality",
                    value=clustering_score,
                    description="Quality of context-based clustering",
                    category="contextual"
                )
                
                # 2. Context Separation
                separation_score = self._evaluate_context_separation(context_data)
                metrics["context_separation"] = EvaluationMetric(
                    name="context_separation",
                    value=separation_score,
                    description="Separation between different contexts",
                    category="contextual"
                )
                
                # 3. Semantic Consistency
                consistency_score = self._evaluate_semantic_consistency()
                metrics["semantic_consistency"] = EvaluationMetric(
                    name="semantic_consistency",
                    value=consistency_score,
                    description="Consistency of semantic relationships",
                    category="contextual"
                )
                
                # 4. Inter-context Relationships
                relationship_score = self._evaluate_intercontext_relationships(context_data)
                metrics["intercontext_relationships"] = EvaluationMetric(
                    name="intercontext_relationships",
                    value=relationship_score,
                    description="Preservation of relationships across contexts",
                    category="contextual"
                )
                
                # 5. Intra-context Coherence
                coherence_score = self._evaluate_intracontext_coherence(context_data)
                metrics["intracontext_coherence"] = EvaluationMetric(
                    name="intracontext_coherence",
                    value=coherence_score,
                    description="Coherence within each context",
                    category="contextual"
                )
        
        except Exception as e:
            print(f"Error in contextual understanding evaluation: {e}")
        
        return metrics

    def _collect_context_embeddings(
        self,
        loader: DataLoader
    ) -> Dict[str, Dict[str, Any]]:
        """
        Collect embeddings and metadata for each context.
        
        Args:
            loader: DataLoader with text samples
            
        Returns:
            Dict containing embeddings and metadata for each context
        """
        context_data = {}
        
        for _, texts in loader:
            if texts:
                for text in texts:
                    if text is not None:
                        try:
                            # Extract number and context
                            numbers = [float(match[0]) for match in 
                                    extract_numbers_with_positions(text)]
                            if not numbers:
                                continue
                                
                            number = numbers[0]  # Take first number if multiple
                            context = self._extract_context(text)
                            
                            # Get embedding
                            embedding = self.encoder.encode_number(text)
                            embedding = embedding.detach().cpu().numpy().squeeze()
                            
                            # Initialize context data if needed
                            if context not in context_data:
                                context_data[context] = {
                                    'embeddings': [],
                                    'numbers': [],
                                    'texts': []
                                }
                            
                            # Store data
                            context_data[context]['embeddings'].append(embedding)
                            context_data[context]['numbers'].append(number)
                            context_data[context]['texts'].append(text)
                            
                        except Exception as e:
                            print(f"Error processing text {text}: {e}")
                            continue
        
        # Convert lists to arrays
        for context in context_data:
            context_data[context]['embeddings'] = np.array(
                context_data[context]['embeddings']
            )
            context_data[context]['numbers'] = np.array(
                context_data[context]['numbers']
            )
        
        return context_data

    def _evaluate_context_clustering(
        self,
        context_data: Dict[str, Dict[str, Any]]
    ) -> float:
        """
        Evaluate how well embeddings cluster by context.
        
        Args:
            context_data: Dictionary of context embeddings and metadata
            
        Returns:
            float: Clustering quality score
        """
        if len(context_data) < 2:
            return 0.0
        
        # Collect all embeddings and labels
        all_embeddings = []
        labels = []
        
        for idx, (context, data) in enumerate(context_data.items()):
            all_embeddings.extend(data['embeddings'])
            labels.extend([idx] * len(data['embeddings']))
        
        all_embeddings = np.array(all_embeddings)
        labels = np.array(labels)
        
        # Perform clustering
        kmeans = KMeans(
            n_clusters=len(context_data),
            random_state=self.config.seed,
            n_init=10
        )
        pred_labels = kmeans.fit_predict(all_embeddings)
        
        # Calculate clustering quality
        return adjusted_rand_score(labels, pred_labels)

    def _evaluate_context_separation(
        self,
        context_data: Dict[str, Dict[str, Any]]
    ) -> float:
        """
        Evaluate separation between different contexts.
        
        Args:
            context_data: Dictionary of context embeddings and metadata
            
        Returns:
            float: Context separation score
        """
        if len(context_data) < 2:
            return 0.0
        
        # Calculate context centroids
        centroids = {
            context: np.mean(data['embeddings'], axis=0)
            for context, data in context_data.items()
        }
        
        # Calculate inter-context distances
        distances = []
        for c1 in centroids:
            for c2 in centroids:
                if c1 < c2:  # Avoid duplicate pairs
                    dist = np.linalg.norm(centroids[c1] - centroids[c2])
                    distances.append(dist)
        
        # Calculate intra-context spreads
        spreads = []
        for data in context_data.values():
            if len(data['embeddings']) > 1:
                centroid = np.mean(data['embeddings'], axis=0)
                spread = np.mean([
                    np.linalg.norm(emb - centroid)
                    for emb in data['embeddings']
                ])
                spreads.append(spread)
        
        # Calculate separation score
        avg_distance = np.mean(distances) if distances else 0
        avg_spread = np.mean(spreads) if spreads else 1
        
        return avg_distance / (avg_spread + 1e-6)

    def _evaluate_semantic_consistency(self) -> float:
        """
        Evaluate consistency of semantic relationships across contexts.
        
        Returns:
            float: Semantic consistency score
        """
        semantic_test_cases = [
            ("5 stars", "4 stars", "3 stars"),
            ("$5", "$50", "$500"),
            ("5 items", "50 items", "500 items"),
            ("5 minutes", "50 minutes", "500 minutes"),
            ("5%", "50%", "100%")
        ]
        
        consistency_scores = []
        
        for test_case in semantic_test_cases:
            try:
                # Get embeddings for test case
                embeddings = []
                for text in test_case:
                    emb = self.encoder.encode_number(text)
                    embeddings.append(emb.detach().cpu().numpy().squeeze())
                
                embeddings = np.array(embeddings)
                
                # Calculate pairwise similarities
                similarities = np.zeros((len(embeddings), len(embeddings)))
                for i in range(len(embeddings)):
                    for j in range(i+1, len(embeddings)):
                        sim = np.dot(embeddings[i], embeddings[j]) / (
                            np.linalg.norm(embeddings[i]) *
                            np.linalg.norm(embeddings[j])
                        )
                        similarities[i,j] = similarities[j,i] = sim
                
                # Score based on expected similarity pattern
                # Closer numbers should have higher similarity
                expected_similarities = []
                for i in range(len(embeddings)):
                    for j in range(i+1, len(embeddings)):
                        expected_sim = 1.0 / (1.0 + abs(i - j))
                        expected_similarities.append(expected_sim)
                
                actual_similarities = similarities[np.triu_indices(
                    len(embeddings), k=1
                )]
                
                # Calculate correlation between expected and actual similarities
                correlation, _ = spearmanr(
                    expected_similarities,
                    actual_similarities
                )
                consistency_scores.append(max(0, correlation))
                
            except Exception as e:
                print(f"Error in semantic test case {test_case}: {e}")
                continue
        
        return np.mean(consistency_scores) if consistency_scores else 0.0

    def _evaluate_intercontext_relationships(
        self,
        context_data: Dict[str, Dict[str, Any]]
    ) -> float:
        """
        Evaluate preservation of numerical relationships across contexts.
        
        Args:
            context_data: Dictionary of context embeddings and metadata
            
        Returns:
            float: Inter-context relationship score
        """
        if len(context_data) < 2:
            return 0.0
        
        relationship_scores = []
        
        # Compare relationships between pairs of contexts
        for c1 in context_data:
            for c2 in context_data:
                if c1 < c2:  # Avoid duplicate pairs
                    nums1 = context_data[c1]['numbers']
                    nums2 = context_data[c2]['numbers']
                    embs1 = context_data[c1]['embeddings']
                    embs2 = context_data[c2]['embeddings']
                    
                    # Find common numbers
                    common_nums = np.intersect1d(nums1, nums2)
                    if len(common_nums) < 2:
                        continue
                    
                    # Get embeddings for common numbers
                    embs1_common = np.array([
                        embs1[np.where(nums1 == num)[0][0]]
                        for num in common_nums
                    ])
                    embs2_common = np.array([
                        embs2[np.where(nums2 == num)[0][0]]
                        for num in common_nums
                    ])
                    
                    # Calculate similarity between relationship patterns
                    dists1 = np.linalg.norm(
                        embs1_common[:, np.newaxis] - embs1_common, axis=2
                    )
                    dists2 = np.linalg.norm(
                        embs2_common[:, np.newaxis] - embs2_common, axis=2
                    )
                    
                    correlation, _ = spearmanr(
                        dists1.flatten(),
                        dists2.flatten()
                    )
                    relationship_scores.append(max(0, correlation))
        
        return np.mean(relationship_scores) if relationship_scores else 0.0

    def _evaluate_intracontext_coherence(
        self,
        context_data: Dict[str, Dict[str, Any]]
    ) -> float:
        """
        Evaluate coherence of embeddings within each context.
        
        Args:
            context_data: Dictionary of context embeddings and metadata
            
        Returns:
            float: Intra-context coherence score
        """
        coherence_scores = []
        
        for context, data in context_data.items():
            if len(data['numbers']) < 2:
                continue
                
            # Sort by number value
            sort_idx = np.argsort(data['numbers'])
            sorted_nums = data['numbers'][sort_idx]
            sorted_embs = data['embeddings'][sort_idx]
            
            # Calculate embedding distances
            emb_dists = np.linalg.norm(
                sorted_embs[1:] - sorted_embs[:-1],
                axis=1
            )
            
            # Calculate number differences
            num_diffs = np.abs(sorted_nums[1:] - sorted_nums[:-1])
            
            # Calculate correlation
            correlation, _ = spearmanr(num_diffs, emb_dists)
            
            # Calculate smoothness
            smoothness = 1.0 - np.std(emb_dists) / (np.mean(emb_dists) + 1e-6)
            
            # Combine metrics
            coherence = (max(0, correlation) + smoothness) / 2
            coherence_scores.append(coherence)
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    def evaluate_ordinal_preservation(self) -> float:
        """
        Evaluate preservation of ordinal relationships within contexts.
        
        Tests ordinal preservation across different contexts:
        - Ratings (1-5 stars)
        - Prices ($10-$200)
        - Quantities (5-100 units)
        - Time periods
        - Percentages
        
        Returns:
            float: Ordinal preservation score between 0 and 1
        """
        print("Evaluating ordinal preservation...")
        
        ordinal_sequences = [
            ("rating", [
                "1 star", "2 stars", "3 stars", "4 stars", "5 stars"
            ]),
            ("price", [
                "$10", "$20", "$50", "$100", "$200"
            ]),
            ("quantity", [
                "5 units", "10 units", "20 units", "50 units", "100 units"
            ]),
            ("time", [
                "1 minute", "5 minutes", "10 minutes", "30 minutes", "60 minutes"
            ]),
            ("percentage", [
                "10%", "25%", "50%", "75%", "100%"
            ])
        ]
        
        ordinal_scores = []
        
        for context, sequence in ordinal_sequences:
            try:
                # Get embeddings for sequence
                embeddings = []
                for text in sequence:
                    emb = self.encoder.encode_number(text)
                    embeddings.append(emb.detach().cpu().numpy().squeeze())
                embeddings = np.array(embeddings)
                
                # Calculate pairwise scores
                correct_order = 0
                total_pairs = 0
                
                for i in range(len(sequence) - 1):
                    for j in range(i + 1, len(sequence)):
                        # Calculate similarities
                        sim_i_next = np.dot(
                            embeddings[i],
                            embeddings[i + 1]
                        ) / (
                            np.linalg.norm(embeddings[i]) *
                            np.linalg.norm(embeddings[i + 1])
                        )
                        
                        sim_i_j = np.dot(
                            embeddings[i],
                            embeddings[j]
                        ) / (
                            np.linalg.norm(embeddings[i]) *
                            np.linalg.norm(embeddings[j])
                        )
                        
                        # Check if closer numbers have higher similarity
                        if sim_i_next > sim_i_j:
                            correct_order += 1
                        total_pairs += 1
                
                # Calculate score for this context
                context_score = correct_order / total_pairs if total_pairs > 0 else 0
                ordinal_scores.append(context_score)
                
            except Exception as e:
                print(f"Error evaluating ordinal preservation for {context}: {e}")
                continue
        
        return np.mean(ordinal_scores) if ordinal_scores else 0.0

    def evaluate_downstream_task(self) -> Dict[str, EvaluationMetric]:
        """
        Evaluate performance on synthetic downstream tasks.
        
        Tasks:
        1. Context Classification
        2. Number Range Prediction
        3. Magnitude Comparison
        4. Context-aware Regression
        5. Semantic Grouping
        
        Returns:
            Dict[str, EvaluationMetric]: Dictionary of downstream task metrics
        """
        print("Evaluating downstream task performance...")
        metrics = {}
        
        try:
            # Generate synthetic data
            contexts = ['rating', 'price', 'quantity', 'time', 'percentage']
            numbers = [1, 2, 5, 10, 20, 50, 100]
            
            # Prepare data structures
            data = []
            labels = []
            embeddings = []
            number_labels = []  # For magnitude comparison
            range_labels = []   # For range prediction
            
            # Generate examples for each context
            for ctx_idx, context in enumerate(contexts):
                for num in numbers:
                    # Generate appropriate text based on context
                    if context == 'rating':
                        text = f"{num} stars"
                        range_label = 0 if num <= 3 else 1  # Binary range (low/high)
                    elif context == 'price':
                        text = f"${num}"
                        range_label = 0 if num < 50 else 1
                    elif context == 'quantity':
                        text = f"{num} units"
                        range_label = 0 if num < 30 else 1
                    elif context == 'time':
                        text = f"{num} minutes"
                        range_label = 0 if num < 30 else 1
                    else:  # percentage
                        text = f"{num}%"
                        range_label = 0 if num < 50 else 1
                    
                    # Get embedding
                    emb = self.encoder.encode_number(text)
                    embeddings.append(emb.detach().cpu().numpy().squeeze())
                    
                    # Store labels
                    labels.append(ctx_idx)
                    number_labels.append(num)
                    range_labels.append(range_label)
                    data.append(text)
            
            # Convert to arrays
            embeddings = np.array(embeddings)
            labels = np.array(labels)
            number_labels = np.array(number_labels)
            range_labels = np.array(range_labels)
            
            # 1. Context Classification
            clf_score = self._evaluate_classification(embeddings, labels)
            metrics["classification_accuracy"] = EvaluationMetric(
                name="classification_accuracy",
                value=clf_score,
                description="Accuracy on context classification",
                category="downstream"
            )
            
            # 2. Range Prediction
            range_score = self._evaluate_range_prediction(embeddings, range_labels)
            metrics["range_prediction"] = EvaluationMetric(
                name="range_prediction",
                value=range_score,
                description="Accuracy on number range prediction",
                category="downstream"
            )
            
            # 3. Magnitude Comparison
            magnitude_score = self._evaluate_magnitude_comparison(
                embeddings, number_labels
            )
            metrics["magnitude_comparison"] = EvaluationMetric(
                name="magnitude_comparison",
                value=magnitude_score,
                description="Accuracy on magnitude comparison",
                category="downstream"
            )
            
            # 4. Context-aware Regression
            regression_score = self._evaluate_contextual_regression(
                embeddings, number_labels, labels
            )
            metrics["contextual_regression"] = EvaluationMetric(
                name="contextual_regression",
                value=regression_score,
                description="Performance on context-aware regression",
                category="downstream"
            )
            
            # 5. Semantic Grouping
            grouping_score = self._evaluate_semantic_grouping(embeddings, labels)
            metrics["semantic_grouping"] = EvaluationMetric(
                name="semantic_grouping",
                value=grouping_score,
                description="Quality of semantic grouping",
                category="downstream"
            )
        
        except Exception as e:
            print(f"Error in downstream task evaluation: {e}")
        
        return metrics

    def _evaluate_classification(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """Evaluate context classification performance."""
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=0.2, random_state=42
        )
        
        clf = LogisticRegression(
            random_state=42,
            multi_class='multinomial',
            max_iter=1000
        )
        clf.fit(X_train, y_train)
        return clf.score(X_test, y_test)

    def _evaluate_range_prediction(
        self,
        embeddings: np.ndarray,
        range_labels: np.ndarray
    ) -> float:
        """Evaluate number range prediction performance."""
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, range_labels, test_size=0.2, random_state=42
        )
        
        clf = LogisticRegression(random_state=42)
        clf.fit(X_train, y_train)
        return clf.score(X_test, y_test)

    def _evaluate_magnitude_comparison(
        self,
        embeddings: np.ndarray,
        numbers: np.ndarray
    ) -> float:
        """Evaluate magnitude comparison performance."""
        correct = 0
        total = 0
        
        for i in range(len(numbers)):
            for j in range(i + 1, len(numbers)):
                # Calculate similarity
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) *
                    np.linalg.norm(embeddings[j])
                )
                
                # Check if similarity reflects magnitude difference
                if (numbers[i] < numbers[j] and sim < 0.9) or \
                (numbers[i] == numbers[j] and sim > 0.9):
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0.0

    def _evaluate_contextual_regression(
        self,
        embeddings: np.ndarray,
        numbers: np.ndarray,
        contexts: np.ndarray
    ) -> float:
        """Evaluate context-aware regression performance."""
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score
        
        scores = []
        
        for context in np.unique(contexts):
            mask = contexts == context
            if np.sum(mask) > 1:
                X = embeddings[mask]
                y = numbers[mask]
                
                if len(X) > 10:  # Ensure enough samples
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    
                    reg = Ridge(alpha=1.0)
                    reg.fit(X_train, y_train)
                    y_pred = reg.predict(X_test)
                    
                    score = r2_score(y_test, y_pred)
                    scores.append(max(0, score))  # Ensure non-negative score
        
        return np.mean(scores) if scores else 0.0

    def _evaluate_semantic_grouping(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """Evaluate semantic grouping performance."""
        # Use clustering to group embeddings
        n_clusters = len(np.unique(labels))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        pred_labels = kmeans.fit_predict(embeddings)
        
        # Calculate clustering metrics
        ari_score = adjusted_rand_score(labels, pred_labels)
        
        # Calculate silhouette score
        sil_score = silhouette_score(embeddings, labels)
        
        # Combine scores
        return (ari_score + sil_score) / 2