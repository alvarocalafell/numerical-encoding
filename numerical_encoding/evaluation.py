import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics"""
    num_clusters: int = 10
    num_synthetic_samples: int = 1000
    seed: int = 42
    batch_size: int = 32
    numerical_test_ranges: List[Tuple[float, float]] = None
    
    def __post_init__(self):
        if self.numerical_test_ranges is None:
            self.numerical_test_ranges = [
                (0, 1),           # Small numbers
                (1, 1000),       # Medium numbers
                (1000, 1e6),     # Large numbers
                (-1000, 1000),   # Mixed positive/negative
                (0.0001, 0.1)    # Small decimals
            ]

def custom_collate_fn(batch):
    """Custom collate function to handle mixed numerical and text data"""
    numerical_samples = []
    text_samples = []
    
    for num, text in batch:
        if num is not None:
            numerical_samples.append(num)
        if text is not None:
            text_samples.append(text)
    
    # Convert to tensors if we have data
    if numerical_samples:
        numerical_samples = torch.tensor(numerical_samples, dtype=torch.float32)
    else:
        numerical_samples = None
        
    return numerical_samples, text_samples

class NumericalEncodingDataset(Dataset):
    def __init__(self, config: EvaluationConfig):
        self.config = config
        np.random.seed(config.seed)
        
        # Generate samples
        self.numerical_data = self._generate_numerical_samples()
        self.text_data = self._generate_textual_samples()
        
        # Create index mapping
        self.total_numerical = len(self.numerical_data)
        self.total_text = len(self.text_data)
        self.total_samples = self.total_numerical + self.total_text
    
    def _generate_numerical_samples(self) -> List[float]:
        samples = []
        for range_min, range_max in self.config.numerical_test_ranges:
            # Linear space samples
            linear_samples = np.linspace(
                range_min, range_max, 
                self.config.num_synthetic_samples // (len(self.config.numerical_test_ranges) * 2)
            )
            samples.extend(linear_samples)
            
            # Log space samples if applicable
            if range_min > 0 and range_max > 0:
                log_samples = np.logspace(
                    np.log10(range_min),
                    np.log10(range_max),
                    self.config.num_synthetic_samples // (len(self.config.numerical_test_ranges) * 2)
                )
                samples.extend(log_samples)
        
        return samples
    
    def _generate_textual_samples(self) -> List[str]:
        contexts = {
            'rating': ['rated {} stars', 'gave it {} points', 'score of {}'],
            'price': ['costs ${}.00', 'priced at ${}.99', '${} total'],
            'quantity': ['{} items', '{} units available', 'quantity: {}'],
            'time': ['{} hours', '{} minutes', '{} days old'],
            'percentage': ['{}% complete', '{}% increase', 'grew by {}%']
        }
        
        samples = []
        for context_type, templates in contexts.items():
            for template in templates:
                if context_type == 'rating':
                    numbers = np.random.uniform(1, 5, size=20)
                elif context_type == 'price':
                    numbers = np.random.lognormal(4, 1, size=20)
                elif context_type == 'quantity':
                    numbers = np.random.randint(1, 100, size=20)
                elif context_type == 'time':
                    numbers = np.random.randint(1, 1000, size=20)
                elif context_type == 'percentage':
                    numbers = np.random.uniform(0, 100, size=20)
                
                samples.extend([template.format(f"{num:.2f}") for num in numbers])
        
        return samples
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        if idx < self.total_numerical:
            return self.numerical_data[idx], None
        else:
            text_idx = idx - self.total_numerical
            return None, self.text_data[text_idx]

class NumericalEncodingEvaluator:
    def __init__(self, encoder, config: EvaluationConfig = EvaluationConfig()):
        self.encoder = encoder
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder.to(self.device)
    
    def evaluate_contextual_understanding(self, loader: DataLoader) -> Dict[str, float]:
        print("Evaluating contextual understanding...")
        # Initialize results dictionary with default values
        results = {
            'clustering_quality': 0.0,
            'context_separation': 0.0,
            'semantic_consistency': 0.0
        }
        
        try:
            context_embeddings = {}
            
            # Define semantic test cases
            semantic_test_cases = [
                ("5 stars", "4 stars", "3 stars"),
                ("$5", "$50", "$500"),
                ("5 items", "50 items", "500 items"),
                ("5 minutes", "50 minutes", "500 minutes"),
                ("5%", "50%", "100%")
            ]
            
            # Collect embeddings by context
            for _, texts in loader:
                if texts:
                    for text in texts:
                        if text is not None:
                            try:
                                embedding = self.encoder.encode_number(text)
                                context = self._extract_context(text)
                                if context not in context_embeddings:
                                    context_embeddings[context] = []
                                context_embeddings[context].append(
                                    embedding.detach().cpu().numpy().squeeze()
                                )
                            except Exception as e:
                                print(f"Error processing text {text}: {e}")
                                continue

            # 1. Context Clustering Quality
            if context_embeddings:
                all_embeddings = []
                labels = []
                for context, embeddings in context_embeddings.items():
                    if embeddings:
                        all_embeddings.extend(embeddings)
                        labels.extend([list(context_embeddings.keys()).index(context)] * len(embeddings))
                
                if len(all_embeddings) > 0:
                    all_embeddings = np.array(all_embeddings)
                    labels = np.array(labels)
                    
                    if len(np.unique(labels)) > 1:
                        kmeans = KMeans(
                            n_clusters=len(context_embeddings),
                            random_state=self.config.seed
                        )
                        pred_labels = kmeans.fit_predict(all_embeddings)
                        
                        results['clustering_quality'] = adjusted_rand_score(labels, pred_labels)
                        results['context_separation'] = silhouette_score(all_embeddings, labels)
            
            # 2. Semantic Consistency
            semantic_scores = []
            for test_case in semantic_test_cases:
                embeddings = []
                for text in test_case:
                    try:
                        emb = self.encoder.encode_number(text)
                        embeddings.append(emb.detach().cpu().numpy().squeeze())
                    except Exception as e:
                        print(f"Error processing test case text {text}: {e}")
                        continue
                
                if len(embeddings) == len(test_case):
                    embeddings = np.array(embeddings)
                    # Calculate pairwise cosine similarities
                    similarities = np.zeros((len(embeddings), len(embeddings)))
                    for i in range(len(embeddings)):
                        for j in range(i+1, len(embeddings)):
                            similarity = np.dot(embeddings[i], embeddings[j]) / (
                                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                            )
                            similarities[i,j] = similarities[j,i] = similarity
                    
                    # Take average of upper triangle
                    upper_tri = similarities[np.triu_indices(len(embeddings), k=1)]
                    semantic_scores.append(np.mean(upper_tri))
            
            if semantic_scores:
                results['semantic_consistency'] = np.mean(semantic_scores)

        except Exception as e:
            print(f"Error in contextual understanding evaluation: {e}")
        
        return results

    def evaluate_numerical_properties(self, loader: DataLoader) -> Dict[str, float]:
        print("Evaluating numerical properties...")
        # Initialize results with default values
        results = {
            'relative_distance': 0.0,
            'scale_invariance': 0.0
        }
        
        try:
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
            
            if all_numbers:
                all_numbers = np.array(all_numbers)
                all_embeddings = np.array([e.squeeze() for e in all_embeddings])
                
                # 1. Relative Distance Preservation
                try:
                    num_diffs = np.log1p(np.abs(all_numbers[:, np.newaxis] - all_numbers[np.newaxis, :]))
                    emb_dists = np.linalg.norm(all_embeddings[:, np.newaxis] - all_embeddings[np.newaxis, :], axis=-1)
                    rel_dist_corr, _ = spearmanr(num_diffs.flatten(), emb_dists.flatten())
                    results['relative_distance'] = rel_dist_corr
                except Exception as e:
                    print(f"Error computing relative distance preservation: {e}")

                # 2. Scale Invariance
                try:
                    scale_factors = np.array([0.01, 0.1, 1.0, 10.0, 100.0])
                    scale_similarities = []
                    base_embeddings = all_embeddings[:100]
                    base_numbers = all_numbers[:100]
                    
                    for scale in scale_factors:
                        scaled_embs = []
                        for num in base_numbers * scale:
                            try:
                                embed = self.encoder.encode_number(float(num))
                                scaled_embs.append(embed.detach().cpu().numpy().squeeze())
                            except Exception as e:
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
                    
                    if scale_similarities:
                        results['scale_invariance'] = np.mean(scale_similarities)
                except Exception as e:
                    print(f"Error computing scale invariance: {e}")
        
        except Exception as e:
            print(f"Error in numerical properties evaluation: {e}")
        
        return results

    def _extract_context(self, text: str) -> str:
        """Extract context type from text"""
        text = text.lower()
        if any(word in text for word in ['star', 'rate', 'score']):
            return 'rating'
        elif any(word in text for word in ['$', 'price', 'cost']):
            return 'price'
        elif any(word in text for word in ['item', 'unit', 'quantity']):
            return 'quantity'
        elif any(word in text for word in ['hour', 'minute', 'day']):
            return 'time'
        elif '%' in text:
            return 'percentage'
        return 'other'

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Run complete evaluation suite"""
        numerical_results = self.evaluate_numerical_properties(loader)
        contextual_results = self.evaluate_contextual_understanding(loader)
        
        return {**numerical_results, **contextual_results}

    def visualize_results(self, results: Dict[str, float]):
        plt.figure(figsize=(12, 6))
        metrics = list(results.keys())
        scores = list(results.values())
        
        plt.bar(range(len(results)), scores)
        plt.xticks(range(len(results)), metrics, rotation=45)
        plt.title('Numerical Encoding Evaluation Results')
        plt.ylabel('Score')
        plt.tight_layout()
        
        return plt