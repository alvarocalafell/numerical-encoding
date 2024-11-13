"""
Main evaluation script for the numerical encoding system.
Runs comprehensive evaluation suite and generates visualization of results.
"""

import torch
import numpy as np
import random
import argparse
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.progress import Progress
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional

from numerical_encoding import (
    NumericalEncoder,
    EvaluationConfig,
    NumericalEncodingDataset,
    NumericalEncodingEvaluator
)
from numerical_encoding.evaluation import custom_collate_fn
from torch.utils.data import DataLoader

console = Console()

def set_all_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def create_evaluation_dataset(
    config: EvaluationConfig,
    generator: torch.Generator
) -> DataLoader:
    """Create evaluation dataset and dataloader.
    
    Args:
        config: Evaluation configuration
        generator: PyTorch random number generator
        
    Returns:
        DataLoader: Configured data loader for evaluation
    """
    dataset = NumericalEncodingDataset(config)
    
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=0,
        generator=generator,
        worker_init_fn=lambda x: np.random.seed(config.seed)
    )

def visualize_results(
    results: Dict[str, float],
    save_path: Optional[Path] = None
) -> None:
    """Create and save visualization of evaluation results.
    
    Args:
        results: Dictionary of evaluation results
        save_path: Optional path to save the visualization
    """
    plt.figure(figsize=(12, 6))
    
    # Create color palette
    colors = sns.color_palette("husl", len(results))
    
    # Create bar plot
    bars = plt.bar(range(len(results)), 
                  list(results.values()), 
                  color=colors)
    
    # Customize plot
    plt.xticks(range(len(results)), 
               list(results.keys()), 
               rotation=45, 
               ha='right')
    
    plt.title('Numerical Encoding Evaluation Results', 
              fontsize=14, 
              pad=20)
    
    plt.ylabel('Score', fontsize=12)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., 
                height,
                f'{height:.3f}',
                ha='center', 
                va='bottom',
                rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        console.print(f"[green]Results visualization saved to {save_path}[/green]")
    
    plt.close()

def run_evaluation(args: argparse.Namespace) -> Dict[str, float]:
    """Run complete evaluation suite.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dict[str, float]: Evaluation results
    """
    console.print("[bold blue]Initializing Numerical Encoding Evaluation[/bold blue]")
    
    # Initialize configurations
    config = EvaluationConfig(
        num_synthetic_samples=args.num_samples,
        seed=args.seed,
        batch_size=args.batch_size
    )
    
    # Set seeds
    set_all_seeds(config.seed)
    
    # Initialize models
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    
    with Progress() as progress:
        # Initialize encoder
        task1 = progress.add_task("[cyan]Initializing encoder...", total=1)
        encoder = NumericalEncoder()
        encoder.tokenizer.init_weights = False
        progress.update(task1, completed=1)
        
        # Create dataset
        task2 = progress.add_task("[cyan]Creating evaluation dataset...", total=1)
        loader = create_evaluation_dataset(config, generator)
        progress.update(task2, completed=1)
        
        # Initialize evaluator
        task3 = progress.add_task("[cyan]Initializing evaluator...", total=1)
        evaluator = NumericalEncodingEvaluator(encoder, config)
        progress.update(task3, completed=1)
        
        # Run evaluation
        task4 = progress.add_task("[cyan]Running evaluation...", total=1)
        results = evaluator.evaluate(loader)
        progress.update(task4, completed=1)
    
    # Print results
    console.print("\n[bold green]Evaluation Results:[/bold green]")
    for metric, value in results.items():
        console.print(f"{metric}: {value:.3f}")
    
    # Save results
    if args.save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path("results")
        save_dir.mkdir(exist_ok=True)
        
        # Save visualization
        vis_path = save_dir / f"evaluation_results_{timestamp}.png"
        visualize_results(results, vis_path)
        
        # Save numeric results
        results_path = save_dir / f"evaluation_results_{timestamp}.txt"
        with open(results_path, "w") as f:
            for metric, value in results.items():
                f.write(f"{metric}: {value:.3f}\n")
        
        console.print(f"[green]Results saved to {results_path}[/green]")
    
    return results

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run numerical encoding evaluation"
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of synthetic samples to generate"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save evaluation results and visualization"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)