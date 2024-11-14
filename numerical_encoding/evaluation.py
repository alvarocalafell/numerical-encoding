"""
Evaluation framework for numerical encoding system.
Provides comprehensive metrics and evaluation tools.
"""

# 1. Imports
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Union, Any, Optional
from dataclasses import dataclass, field
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    silhouette_score,
    f1_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from .utils import extract_numbers_with_positions
from numerical_encoding.encoder import NumericalEncoder
from rich.console import Console

console = Console()

@dataclass
class EvaluationConfig:
    """Configuration for numerical encoding evaluation.
    
    Organized around three core evaluation aspects:
    1. Numerical Relationship Preservation
    2. Contextual Understanding
    3. Downstream Task Performance
    """
    # Basic configuration
    num_synthetic_samples: int = 1000
    seed: int = 42
    batch_size: int = 32
    embedding_dim: int = 768
    
    # Numerical evaluation ranges
    numerical_test_ranges: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0, 1),          # Small numbers
        (1, 1000),       # Medium numbers
        (1000, 1e6),     # Large numbers
        (-1000, 1000),   # Mixed positive/negative
        (0.0001, 0.1)    # Small decimals
    ])
    
    # Context evaluation settings
    context_types: List[str] = field(default_factory=lambda: [
        'rating',
        'price',
        'quantity',
        'time',
        'percentage'
    ])
    
    # Semantic test cases
    semantic_test_cases: List[Dict[str, Any]] = field(default_factory=lambda: [
        {
            "context": "rating",
            "sequence": ["1 star", "3 stars", "5 stars"],
            "expected_relations": [(0, 1, 0.5), (1, 2, 0.5)]
        },
        {
            "context": "price",
            "sequence": ["$10", "$50", "$100"],
            "expected_relations": [(0, 1, 0.2), (1, 2, 0.5)]
        }
    ])
    
    # Evaluation weights
    metric_weights: Dict[str, float] = field(default_factory=lambda: {
        "numerical": 0.4,
        "contextual": 0.4,
        "downstream": 0.2
    })
    
    # Output settings
    evaluation_output_dir: Optional[Path] = None
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.num_synthetic_samples < self.batch_size:
            raise ValueError(
                f"num_synthetic_samples ({self.num_synthetic_samples}) must be >= "
                f"batch_size ({self.batch_size})"
            )
        
        if not self.numerical_test_ranges:
            raise ValueError("numerical_test_ranges cannot be empty")
            
        # Ensure weights sum to 1
        total_weight = sum(self.metric_weights.values())
        if not np.isclose(total_weight, 1.0):
            raise ValueError(f"Metric weights must sum to 1.0, got {total_weight}")
        
class NumericalEncodingDataset(Dataset):
    """Dataset for evaluating numerical encoding performance."""
    
    def __init__(self, config: 'EvaluationConfig'):
        self.config = config
        
        # Set seeds for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Generate data
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
        samples = []
        
        # Increased samples per context type
        samples_per_context = 50 #100
        
        for context_type in self.config.context_types:
            # Generate numbers based on context
            if context_type == 'rating':
                numbers = np.concatenate([
                    np.linspace(1, 5, samples_per_context//2),
                    np.random.uniform(1, 5, samples_per_context//2)
                ])
            elif context_type == 'price':
                numbers = np.concatenate([
                    np.logspace(0, 4, samples_per_context//2),
                    np.random.uniform(1, 10000, samples_per_context//2)
                ])
            elif context_type == 'quantity':
                numbers = np.concatenate([
                    np.linspace(1, 1000, samples_per_context//2),
                    np.random.uniform(1, 1000, samples_per_context//2)
                ])
            elif context_type == 'time':
                numbers = np.concatenate([
                    np.linspace(1, 24, samples_per_context//4),  # Hours
                    np.linspace(1, 60, samples_per_context//4),  # Minutes
                    np.random.uniform(1, 24, samples_per_context//4),
                    np.random.uniform(1, 60, samples_per_context//4)
                ])
            elif context_type == 'percentage':
                numbers = np.concatenate([
                    np.linspace(0, 100, samples_per_context//2),
                    np.random.uniform(0, 100, samples_per_context//2)
                ])
            else:
                continue
                
            # Add variety in text formats
            templates = {
                'rating': [
                    "{} stars", "rated {}", "rating: {}", 
                    "gave it {} stars", "score of {}", "{}/5",
                    "{} star rating", "review score: {}"
                ],
                'price': [
                    "${}", "price: ${}", "costs ${}", 
                    "${} dollars", "priced at ${}", "${}.00",
                    "value: ${}", "worth ${}"
                ],
                'quantity': [
                    "{} items", "{} pieces", "quantity: {}", 
                    "{} units", "count: {}", "total: {} items",
                    "{} in stock", "quantity of {}"
                ],
                'time': [
                    "{} hours", "{} minutes", "duration: {} hrs", 
                    "{} mins", "time: {} hours", "{} hour duration",
                    "{} minute span", "period: {} hours"
                ],
                'percentage': [
                    "{}%", "{} percent", "percentage: {}", 
                    "{} pct", "{}% complete", "progress: {}%",
                    "{} percent done", "completion: {}%"
                ]
            }
            
            context_templates = templates.get(context_type, ["{} {}"])
            
            # Generate samples with different templates
            for num in numbers:
                template = np.random.choice(context_templates)
                text = template.format(
                    round(num, 2) if context_type != 'price' else format(num, '.2f')
                )
                samples.append(text)
        
        return samples
    
    def _format_number_for_context(self, number: float, context: str) -> str:
        """Format a number according to context."""
        context_templates = {
            'rating': '{:.1f} stars',
            'price': '${:.2f}',
            'quantity': '{:.0f} items',
            'time': '{:.0f} minutes',
            'percentage': '{:.1f}%'
        }
        template = context_templates.get(context, '{:.2f}')
        return template.format(number)
    
    def __len__(self) -> int:
        """Get total number of samples."""
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Tuple[Optional[float], Optional[str]]:
        """Get sample by index."""
        if idx < self.total_numerical:
            return self.numerical_data[idx], None
        else:
            text_idx = idx - self.total_numerical
            return None, self.text_data[text_idx]
        
def custom_collate_fn(
    batch: List[Tuple[Optional[float], Optional[str]]]
) -> Tuple[Optional[torch.Tensor], List[str]]:
    """Custom collate function for batching mixed numerical and text data."""
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
    """Evaluator for numerical encoding scheme focusing on three core aspects:
    1. Numerical Relationship Preservation
    2. Contextual Understanding
    3. Downstream Task Performance
    """
    
    def __init__(
        self,
        encoder: NumericalEncoder,
        config: EvaluationConfig = EvaluationConfig()
    ):
        self.encoder = encoder
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder.to(self.device)
        self.metrics = {}  # Store metrics for visualization


        
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Run complete evaluation suite."""
        console.print("\n[bold]Starting Numerical Encoding Evaluation...[/bold]")
        
        try:
            # Collect all results
            numerical_metrics = self._evaluate_numerical_relationships(loader)
            console.print("\n[cyan]Numerical Metrics:[/cyan]")
            for metric, value in numerical_metrics.items():
                console.print(f"  {metric}: {value:.3f}")
            
            contextual_metrics = self._evaluate_contextual_understanding(loader)
            console.print("\n[cyan]Contextual Metrics:[/cyan]")
            for metric, value in contextual_metrics.items():
                console.print(f"  {metric}: {value:.3f}")
            
            downstream_metrics = self._evaluate_downstream_tasks(loader)
            console.print("\n[cyan]Downstream Metrics:[/cyan]")
            for metric, value in downstream_metrics.items():
                console.print(f"  {metric}: {value:.3f}")
            
            # Combine all metrics
            metrics = {
                **numerical_metrics,
                **contextual_metrics,
                **downstream_metrics
            }
            
            # Calculate overall score if we have valid metrics
            valid_scores = [v for v in metrics.values() if not np.isnan(v)]
            if valid_scores:
                metrics["overall_score"] = np.mean(valid_scores)
            
            console.print("\n[bold green]Evaluation Complete![/bold green]")
            
            self.metrics = metrics
            return metrics
            
        except Exception as e:
            console.print(f"[red]Error in evaluation: {str(e)}[/red]")
            return {}
    
    def _evaluate_numerical_relationships(
        self,
        loader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate preservation of numerical relationships."""
        print("Evaluating numerical relationships...")

                
        try:
            # Collect embeddings for numerical evaluation with enhanced sampling
            numbers, embeddings = self._collect_numerical_embeddings(loader)
            
            if not len(numbers):
                return {}
                
            metrics = {}
            
            # Use operations and pre-computed distances
            metrics["relative_distance"] = self._evaluate_relative_distances(
                numbers, embeddings
            )
            
            metrics["interval_preservation"] = self._evaluate_interval_preservation(
                numbers, embeddings
            )
            
            # Keep other evaluations
            metrics["scale_invariance"] = self._evaluate_scale_invariance(
                numbers, embeddings,
                scales=[0.01, 0.1, 1.0, 10.0, 100.0]
            )
            
            metrics["magnitude_preservation"] = self._evaluate_magnitude_preservation(
                numbers, embeddings
            )
            
            metrics["numerical_continuity"] = self._evaluate_numerical_continuity(
                numbers, embeddings
            )
            
            return metrics
            
        except Exception as e:
            print(f"Error in numerical relationship evaluation: {e}")
            return {}

    def _evaluate_relative_distances(
        self,
        numbers: np.ndarray,
        embeddings: np.ndarray
    ) -> float:
        """Evaluate preservation of relative distances between numbers."""
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
        return max(0, correlation)

    def _evaluate_scale_invariance(
        self,
        numbers: np.ndarray,
        embeddings: np.ndarray,
        scales: List[float]
    ) -> float:
        """Evaluate consistency across different scales."""
        scale_similarities = []
        
        for scale in scales:
            scaled_nums = numbers * scale
            scaled_embs = []
            
            for num in scaled_nums:
                try:
                    emb = self.encoder.encode_number(float(num))
                    scaled_embs.append(emb.detach().cpu().numpy().squeeze())
                except Exception:
                    continue
            
            if scaled_embs:
                scaled_embs = np.array(scaled_embs)
                similarities = []
                
                for base_emb, scaled_emb in zip(embeddings, scaled_embs):
                    sim = np.dot(base_emb, scaled_emb) / (
                        np.linalg.norm(base_emb) * np.linalg.norm(scaled_emb)
                    )
                    similarities.append(sim)
                    
                scale_similarities.append(np.mean(similarities))
        
        return np.mean(scale_similarities) if scale_similarities else 0.0

    def _evaluate_magnitude_preservation(
        self,
        numbers: np.ndarray,
        embeddings: np.ndarray
    ) -> float:
        """
        Evaluate how well the embedding preserves magnitude relationships.
        
        Args:
            numbers: Array of input numbers
            embeddings: Array of corresponding embeddings
            
        Returns:
            float: Magnitude preservation score between 0 and 1
        """
        try:
            if len(numbers) < 2:
                return 0.0
            
            # Sort numbers by magnitude
            magnitude_order = np.argsort(np.abs(numbers))
            sorted_numbers = np.abs(numbers[magnitude_order])
            sorted_embeddings = embeddings[magnitude_order]
            
            # Calculate pairwise magnitude differences
            magnitude_diffs = np.abs(
                sorted_numbers[:, np.newaxis] - sorted_numbers[np.newaxis, :]
            )
            
            # Calculate pairwise embedding distances
            embedding_dists = np.zeros((len(embeddings), len(embeddings)))
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    # Calculate cosine similarity
                    sim = np.dot(sorted_embeddings[i], sorted_embeddings[j]) / (
                        np.linalg.norm(sorted_embeddings[i]) * 
                        np.linalg.norm(sorted_embeddings[j])
                    )
                    # Convert similarity to distance
                    dist = 1 - sim
                    embedding_dists[i, j] = embedding_dists[j, i] = dist
            
            # Check preservation of magnitude relationships
            correct_order = 0
            total_comparisons = 0
            
            for i in range(len(numbers) - 2):
                for j in range(i + 1, len(numbers) - 1):
                    for k in range(j + 1, len(numbers)):
                        # Compare magnitude differences
                        mag_diff_ij = magnitude_diffs[i, j]
                        mag_diff_ik = magnitude_diffs[i, k]
                        
                        # Compare embedding distances
                        emb_dist_ij = embedding_dists[i, j]
                        emb_dist_ik = embedding_dists[i, k]
                        
                        # Check if relative magnitudes are preserved
                        if mag_diff_ij < mag_diff_ik and emb_dist_ij < emb_dist_ik:
                            correct_order += 1
                        elif mag_diff_ij > mag_diff_ik and emb_dist_ij > emb_dist_ik:
                            correct_order += 1
                        elif np.isclose(mag_diff_ij, mag_diff_ik) and np.isclose(emb_dist_ij, emb_dist_ik):
                            correct_order += 1
                            
                        total_comparisons += 1
            
            if total_comparisons == 0:
                return 0.0
                
            # Calculate final score
            preservation_score = correct_order / total_comparisons
            
            # Add penalty for very different scales
            scale_variance = np.std(np.linalg.norm(embeddings, axis=1))
            scale_penalty = np.exp(-scale_variance)
            
            final_score = (preservation_score + scale_penalty) / 2
            
            return max(0.0, min(1.0, final_score))  # Ensure score is between 0 and 1
            
        except Exception as e:
            print(f"Error in magnitude preservation evaluation: {e}")
            return 0.0

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
    
    def _evaluate_interval_preservation(
        self,
        numbers: np.ndarray,
        embeddings: np.ndarray
    ) -> float:
        """Evaluate preservation of numerical intervals."""
        # Generate sequence pairs with known intervals
        pairs = []
        intervals = []
        
        for i in range(len(numbers) - 1):
            for j in range(i + 1, min(i + 5, len(numbers))):
                pairs.append((i, j))
                intervals.append(abs(numbers[j] - numbers[i]))
        
        if not pairs:
            return 0.0
        
        # Calculate embedding distances for each pair
        emb_dists = []
        for i, j in pairs:
            dist = np.linalg.norm(embeddings[j] - embeddings[i])
            emb_dists.append(dist)
        
        # Normalize intervals and distances
        intervals = np.array(intervals)
        emb_dists = np.array(emb_dists)
        
        intervals = intervals / np.max(intervals)
        emb_dists = emb_dists / np.max(emb_dists)
        
        # Calculate correlation between intervals and distances
        correlation, _ = spearmanr(intervals, emb_dists)
        return max(0, correlation)

    def _evaluate_contextual_understanding(self, loader: DataLoader) -> Dict[str, float]:
            """Evaluate understanding of numbers in different contexts.
            
            Tests:
            1. Context separation
            2. Cross-context understanding
            3. Semantic preservation
            """
            print("Evaluating contextual understanding...")
            
            try:
                # Collect context-specific embeddings
                context_data = self._collect_context_embeddings(loader)
                
                if not context_data:
                    return {}
                    
                metrics = {}
                
                # 1. Context Separation
                metrics["context_separation"] = self._evaluate_context_separation(
                    context_data
                )
                
                # 2. Cross-context Understanding
                metrics["cross_context_understanding"] = self._evaluate_cross_context_understanding(
                    context_data
                )
                
                # 3. Semantic Preservation
                metrics["semantic_preservation"] = self._evaluate_semantic_preservation(
                    context_data
                )
                
                return metrics
                
            except Exception as e:
                print(f"Error in contextual understanding evaluation: {e}")
                return {}

    def _evaluate_context_separation(
        self,
        context_data: Dict[str, Dict[str, Any]]
    ) -> float:
        if len(context_data) < 2:
            return 0.0
        
        try:
            # Enhanced context encoding with sub-context clustering
            context_clusters = {}
            for context, data in context_data.items():
                if len(data['embeddings']) > 1:
                    # Cluster embeddings within each context
                    kmeans = KMeans(n_clusters=min(3, len(data['embeddings'])))
                    clusters = kmeans.fit_predict(data['embeddings'])
                    context_clusters[context] = {
                        'centroids': kmeans.cluster_centers_,
                        'embeddings': data['embeddings']
                    }
            
            # Calculate inter-context distances using cluster centroids
            inter_distances = []
            for c1, data1 in context_clusters.items():
                for c2, data2 in context_clusters.items():
                    if c1 < c2:
                        for centroid1 in data1['centroids']:
                            for centroid2 in data2['centroids']:
                                sim = cosine_similarity([centroid1], [centroid2])[0][0]
                                dist = 1 - sim
                                inter_distances.append(dist)
            
            # Calculate intra-context cohesion
            intra_cohesion = []
            for data in context_clusters.values():
                for embedding in data['embeddings']:
                    centroid_dists = [
                        1 - cosine_similarity([embedding], [centroid])[0][0]
                        for centroid in data['centroids']
                    ]
                    intra_cohesion.append(min(centroid_dists))
            
            score = np.mean(inter_distances) / (np.mean(intra_cohesion) + 1e-6)
            return np.clip(score, 0, 1)
            
        except Exception as e:
            print(f"Error in context separation evaluation: {e}")
            return 0.0

    def _evaluate_cross_context_understanding(
        self,
        context_data: Dict[str, Dict[str, Any]]
    ) -> float:
        """Evaluate preservation of numerical relationships across contexts."""
        test_numbers = [1, 5, 10, 50, 100]  # Test across different magnitudes
        contexts = list(context_data.keys())
        
        if len(contexts) < 2:
            return 0.0
        
        relationship_scores = []
        
        for num in test_numbers:
            context_embeddings = {}
            
            # Get embeddings for each context
            for context in contexts:
                text = self._format_number_for_context(num, context)
                try:
                    emb = self.encoder.encode_number(text)
                    context_embeddings[context] = emb.detach().cpu().numpy().squeeze()
                except Exception:
                    continue
            
            if len(context_embeddings) < 2:
                continue
            
            # Calculate relationship preservation scores
            for c1 in context_embeddings:
                for c2 in context_embeddings:
                    if c1 < c2:
                        # Compare relationship preservation
                        emb1 = context_embeddings[c1]
                        emb2 = context_embeddings[c2]
                        
                        # Numerical relationship should be preserved
                        # despite different contexts
                        sim = np.dot(emb1, emb2) / (
                            np.linalg.norm(emb1) * np.linalg.norm(emb2)
                        )
                        relationship_scores.append(sim)
        
        return np.mean(relationship_scores) if relationship_scores else 0.0

    def _evaluate_semantic_preservation(
        self,
        context_data: Dict[str, Dict[str, Any]]
    ) -> float:
        """Evaluate preservation of semantic relationships within contexts."""
        semantic_scores = []
        
        for test_case in self.config.semantic_test_cases:
            try:
                context = test_case["context"]
                sequence = test_case["sequence"]
                relations = test_case["expected_relations"]
                
                # Get embeddings for sequence
                embeddings = []
                for text in sequence:
                    emb = self.encoder.encode_number(text)
                    embeddings.append(emb.detach().cpu().numpy().squeeze())
                embeddings = np.array(embeddings)
                
                # Evaluate semantic relationships
                for i, j, expected_sim in relations:
                    sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) *
                        np.linalg.norm(embeddings[j])
                    )
                    
                    # Score based on how close similarity is to expected
                    score = 1.0 - abs(sim - expected_sim)
                    semantic_scores.append(score)
                    
            except Exception as e:
                print(f"Error in semantic test case {test_case}: {e}")
                continue
        
        return np.mean(semantic_scores) if semantic_scores else 0.0

    def _format_number_for_context(self, number: float, context: str) -> str:
        """Format a number according to its context."""
        context_templates = {
            'rating': '{} stars',
            'price': '${:g}',
            'quantity': '{} items',
            'time': '{} minutes',
            'percentage': '{}%'
        }
        template = context_templates.get(context, '{}')
        return template.format(number)

    def _evaluate_downstream_tasks(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate performance on downstream ML tasks.
        
        Tests:
        1. Classification
        2. Clustering
        3. Semantic similarity
        """
        print("Evaluating downstream task performance...")
        
        try:
            # Collect data for downstream tasks
            embeddings, labels, texts = self._collect_downstream_data(loader)
            
            if not len(embeddings):
                return {}
                
            metrics = {}
            
            # 1. Classification Performance
            metrics["classification_performance"] = self._evaluate_classification_task(
                embeddings, labels
            )
            
            # 2. Clustering Quality
            metrics["clustering_quality"] = self._evaluate_clustering_task(
                embeddings, labels
            )
            
            # 3. Semantic Similarity
            metrics["semantic_similarity"] = self._evaluate_semantic_similarity(
                embeddings, texts
            )
            
            return metrics
            
        except Exception as e:
            print(f"Error in downstream task evaluation: {e}")
            return {}

    def _collect_downstream_data(
        self,
        loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Collect embeddings and labels for downstream task evaluation."""
        all_embeddings = []
        all_labels = []
        all_texts = []
        
        for batch in loader:
            numbers, texts = batch
            
            # Process standalone numbers
            if numbers is not None:
                for num in numbers:
                    try:
                        emb = self.encoder.encode_number(float(num))
                        all_embeddings.append(emb.detach().cpu().numpy().squeeze())
                        all_labels.append(self._get_number_category(float(num)))
                        all_texts.append(str(num))
                    except Exception:
                        continue
            
            # Process text samples
            if texts:
                for text in texts:
                    try:
                        emb = self.encoder.encode_number(text)
                        all_embeddings.append(emb.detach().cpu().numpy().squeeze())
                        all_labels.append(self._get_context_category(text))
                        all_texts.append(text)
                    except Exception:
                        continue
        
        if not all_embeddings:
            return np.array([]), np.array([]), []
            
        return (
            np.array(all_embeddings),
            np.array(all_labels),
            all_texts
        )

    def _evaluate_classification_task(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """Evaluate embeddings on a classification task."""
        if len(embeddings) < 20:  # Minimum samples for meaningful split
            return 0.0
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                embeddings, labels,
                test_size=0.2,
                random_state=self.config.seed,
                stratify=labels
            )
            
            # Train classifier
            clf = LogisticRegression(
                random_state=self.config.seed,
                multi_class='multinomial',
                max_iter=1000
            )
            clf.fit(X_train, y_train)
            
            # Evaluate
            y_pred = clf.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            return (accuracy + f1) / 2
            
        except Exception as e:
            print(f"Error in classification evaluation: {e}")
            return 0.0

    def _evaluate_clustering_task(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """Evaluate embeddings on a clustering task."""
        if len(embeddings) < 20:
            return 0.0
        
        try:
            # Feature normalization for better clustering
            normalized_embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
            
            # Determine optimal number of clusters using silhouette analysis
            n_clusters_range = range(max(2, len(np.unique(labels)) - 2), 
                                min(len(np.unique(labels)) + 3, len(embeddings) // 5))
            best_score = -1
            best_n_clusters = len(np.unique(labels))
            
            for n_clusters in n_clusters_range:
                # Initialize kmeans with better parameters
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    random_state=self.config.seed,
                    n_init=20,  # Increased from 10
                    max_iter=500,  # Increased from default
                    tol=1e-5  # Increased precision
                )
                pred_labels = kmeans.fit_predict(normalized_embeddings)
                
                # Calculate metrics
                current_ari = adjusted_rand_score(labels, pred_labels)
                current_silhouette = silhouette_score(normalized_embeddings, pred_labels)
                
                # Weighted combination of metrics
                current_score = (0.6 * current_ari + 0.4 * current_silhouette)
                
                if current_score > best_score:
                    best_score = current_score
                    best_n_clusters = n_clusters
            
            # Final clustering with best parameters
            final_kmeans = KMeans(
                n_clusters=best_n_clusters,
                random_state=self.config.seed,
                n_init=30,  # Even more initializations for final clustering
                max_iter=500
            )
            final_pred_labels = final_kmeans.fit_predict(normalized_embeddings)
            
            # Calculate final metrics
            ari_score = adjusted_rand_score(labels, final_pred_labels)
            silhouette = silhouette_score(normalized_embeddings, final_pred_labels)
            
            # Add cluster cohesion metric
            cluster_cohesion = self._calculate_cluster_cohesion(normalized_embeddings, final_pred_labels)
            
            # Weighted combination of all metrics
            return (0.5 * ari_score + 0.3 * silhouette + 0.2 * cluster_cohesion)
            
        except Exception as e:
            print(f"Error in clustering evaluation: {e}")
            return 0.0

    def _calculate_cluster_cohesion(
        self,
        embeddings: np.ndarray,
        cluster_labels: np.ndarray
    ) -> float:
        """Calculate cluster cohesion score."""
        unique_clusters = np.unique(cluster_labels)
        cohesion_scores = []
        
        for cluster in unique_clusters:
            cluster_points = embeddings[cluster_labels == cluster]
            if len(cluster_points) > 1:
                # Calculate mean pairwise distance within cluster
                centroid = np.mean(cluster_points, axis=0)
                distances = np.linalg.norm(cluster_points - centroid, axis=1)
                cohesion = 1.0 / (1.0 + np.mean(distances))
                cohesion_scores.append(cohesion)
        
        return np.mean(cohesion_scores) if cohesion_scores else 0.0

    def _evaluate_semantic_similarity(
        self,
        embeddings: np.ndarray,
        texts: List[str]
    ) -> float:
        """Evaluate semantic similarity preservation."""
        if len(texts) < 2:
            return 0.0
        
        try:
            similarity_scores = []
            
            # Define semantic similarity test cases
            test_cases = [
                {
                    "similar": ["5 stars", "rated 5", "rating: 5"],
                    "dissimilar": ["5 dollars", "5 items"]
                },
                {
                    "similar": ["$10", "10 dollars", "price: 10"],
                    "dissimilar": ["10 minutes", "10 stars"]
                }
            ]
            
            for test_case in test_cases:
                # Get embeddings for similar texts
                similar_embs = []
                for text in test_case["similar"]:
                    try:
                        emb = self.encoder.encode_number(text)
                        similar_embs.append(emb.detach().cpu().numpy().squeeze())
                    except Exception:
                        continue
                
                # Get embeddings for dissimilar texts
                dissimilar_embs = []
                for text in test_case["dissimilar"]:
                    try:
                        emb = self.encoder.encode_number(text)
                        dissimilar_embs.append(emb.detach().cpu().numpy().squeeze())
                    except Exception:
                        continue
                
                if not (similar_embs and dissimilar_embs):
                    continue
                
                # Calculate within-group similarity
                similar_sims = []
                for i in range(len(similar_embs)):
                    for j in range(i + 1, len(similar_embs)):
                        sim = np.dot(similar_embs[i], similar_embs[j]) / (
                            np.linalg.norm(similar_embs[i]) *
                            np.linalg.norm(similar_embs[j])
                        )
                        similar_sims.append(sim)
                
                # Calculate between-group similarity
                dissimilar_sims = []
                for emb1 in similar_embs:
                    for emb2 in dissimilar_embs:
                        sim = np.dot(emb1, emb2) / (
                            np.linalg.norm(emb1) * np.linalg.norm(emb2)
                        )
                        dissimilar_sims.append(sim)
                
                # Score based on separation between similar and dissimilar
                if similar_sims and dissimilar_sims:
                    avg_similar = np.mean(similar_sims)
                    avg_dissimilar = np.mean(dissimilar_sims)
                    score = (avg_similar - avg_dissimilar + 1) / 2  # Normalize to [0,1]
                    similarity_scores.append(score)
            
            return np.mean(similarity_scores) if similarity_scores else 0.0
            
        except Exception as e:
            print(f"Error in semantic similarity evaluation: {e}")
            return 0.0

    def _get_number_category(self, number: float) -> int:
        """Categorize numbers into ranges for classification."""
        if number < 0:
            return 0  # negative
        elif number < 1:
            return 1  # small
        elif number < 100:
            return 2  # medium
        elif number < 10000:
            return 3  # large
        else:
            return 4  # very large

    def _get_context_category(self, text: str) -> int:
        """Get category label for text based on context."""
        text = text.lower()
        if any(word in text for word in ['star', 'rate', 'score']):
            return 5  # rating
        elif any(word in text for word in ['$', 'dollar', 'price']):
            return 6  # price
        elif any(word in text for word in ['item', 'unit', 'piece']):
            return 7  # quantity
        elif any(word in text for word in ['hour', 'minute', 'time']):
            return 8  # time
        elif any(word in text for word in ['%', 'percent']):
            return 9  # percentage
        return 10  # other

    def _collect_numerical_embeddings(
        self,
        loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Collect standalone number embeddings for numerical evaluation.
        
        Returns:
            Tuple of (numbers, embeddings) arrays
        """
        all_numbers = []
        all_embeddings = []
        
        for nums, _ in loader:
            if nums is not None:
                for num in nums.numpy():
                    try:
                        embedding = self.encoder.encode_number(float(num))
                        all_embeddings.append(embedding.detach().cpu().numpy())
                        all_numbers.append(num)
                    except Exception as e:
                        print(f"Error processing number {num}: {e}")
                        continue
        
        if not all_numbers:
            return np.array([]), np.array([])
            
        return (
            np.array(all_numbers),
            np.array([e.squeeze() for e in all_embeddings])
        )


    def _collect_context_embeddings(
        self,
        loader: DataLoader
    ) -> Dict[str, Dict[str, Any]]:
        """Collect embeddings organized by context type.
        
        Returns:
            Dict mapping context types to their data
        """
        context_data = {}
        
        for _, texts in loader:
            if texts:
                for text in texts:
                    try:
                        # Extract number and context
                        numbers = extract_numbers_with_positions(text)
                        if not numbers:
                            continue
                            
                        number = float(numbers[0][0])
                        context = self._get_context_type(text)
                        
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

    def _get_context_type(self, text: str) -> str:
        """Determine context type from text.
        
        Returns:
            str: Context type identifier
        """
        text = text.lower()
        
        # Context patterns with keywords
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

    def _calculate_category_score(
        self,
        metrics: Dict[str, float]
    ) -> float:
        """Calculate weighted average score for a category of metrics.
        
        Returns:
            float: Category score between 0 and 1
        """
        if not metrics:
            return 0.0
            
        return sum(metrics.values()) / len(metrics)

    def save_results(self, save_dir: Union[str, Path]) -> None:
        """Save evaluation results and visualizations.
        
        Args:
            save_dir: Directory to save results
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics to file
        results_path = save_dir / f"evaluation_results_{timestamp}.txt"
        with open(results_path, "w") as f:
            f.write("Numerical Encoding Evaluation Results\n")
            f.write("===================================\n\n")
            
            # Write category scores
            f.write("Category Scores:\n")
            f.write("--------------\n")
            for category in ['numerical', 'contextual', 'downstream']:
                score = self._calculate_category_score(
                    {k:v for k,v in self.metrics.items() if k.startswith(category)}
                )
                f.write(f"{category.title()}: {score:.3f}\n")
            f.write("\n")
            
            # Write individual metrics
            f.write("Detailed Metrics:\n")
            f.write("----------------\n")
            for name, value in sorted(self.metrics.items()):
                f.write(f"{name}: {value:.3f}\n")
        
        print(f"Results saved to {results_path}")
        
        # Create visualizations
        self._create_visualizations(save_dir, timestamp)

    def _create_visualizations(
        self,
        save_dir: Path,
        timestamp: str
    ) -> None:
        """Create and save visualization of results.
        
        Args:
            save_dir: Directory to save visualizations
            timestamp: Timestamp string for filenames
        """
        # Group metrics by category
        metric_groups = {
            'Numerical': [m for m in self.metrics if m.startswith('numerical')],
            'Contextual': [m for m in self.metrics if m.startswith('contextual')],
            'Downstream': [m for m in self.metrics if m.startswith('downstream')]
        }
        
        # Create subplot for each category
        fig, axes = plt.subplots(
            len(metric_groups),
            1,
            figsize=(12, 6 * len(metric_groups)),
            squeeze=False
        )
        
        for i, (category, metrics) in enumerate(metric_groups.items()):
            values = [self.metrics[m] for m in metrics]
            names = [m.replace('_', ' ').title() for m in metrics]
            
            ax = axes[i, 0]
            bars = ax.bar(names, values)
            ax.set_title(f'{category} Metrics')
            ax.set_ylim(0, 1)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height,
                    f'{height:.3f}',
                    ha='center',
                    va='bottom'
                )
            
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = save_dir / f"evaluation_results_{timestamp}.png"
        plt.savefig(viz_path)
        plt.close()
        
        print(f"Visualization saved to {viz_path}")    