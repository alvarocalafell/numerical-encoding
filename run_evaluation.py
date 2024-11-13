import torch
import numpy as np
import random
from numerical_encoding import NumericalEncoder
from numerical_encoding.evaluation import (
    NumericalEncodingEvaluator,
    EvaluationConfig,
    NumericalEncodingDataset,
    custom_collate_fn
)
from torch.utils.data import DataLoader

def set_all_seeds(seed):
    """Set all seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def run_evaluation():
    """Run complete evaluation suite"""
    print("Initializing evaluation...")
    
    # Initialize configurations
    config = EvaluationConfig()
    
    # Set all seeds for reproducibility
    set_all_seeds(config.seed)
    
    # Initialize encoder with deterministic behavior
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    
    # Initialize models with deterministic settings
    encoder = NumericalEncoder()
    encoder.tokenizer.init_weights = False  # Disable random init of new tokens
    evaluator = NumericalEncodingEvaluator(encoder, config)
    
    # Create dataset and dataloader with fixed seed
    print("Creating evaluation dataset...")
    dataset = NumericalEncodingDataset(config)
    
    # Use generator for dataloader
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=0,
        generator=generator,
        worker_init_fn=lambda x: np.random.seed(config.seed)
    )
    
    # Run evaluation
    print("Running evaluation...")
    results = evaluator.evaluate(loader)
    
    # Print results
    print("\nEvaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.3f}")
    
    # Visualize results
    if results:
        plt = evaluator.visualize_results(results)
        plt.show()
    else:
        print("No results to visualize")

if __name__ == "__main__":
    run_evaluation()