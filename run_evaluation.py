from numerical_encoding import NumericalEncoder
from numerical_encoding.evaluation import (
    NumericalEncodingEvaluator,
    EvaluationConfig,
    NumericalEncodingDataset,
    custom_collate_fn
)
from torch.utils.data import DataLoader

def run_evaluation():
    """Run complete evaluation suite"""
    print("Initializing evaluation...")
    
    # Initialize configurations and models
    config = EvaluationConfig()
    encoder = NumericalEncoder()
    evaluator = NumericalEncodingEvaluator(encoder, config)
    
    # Create dataset and dataloader
    print("Creating evaluation dataset...")
    dataset = NumericalEncodingDataset(config)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=0
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