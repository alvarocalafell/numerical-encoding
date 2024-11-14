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
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.table import Table
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional

from numerical_encoding import (
    NumericalEncoder,
    EncoderConfig,
    EvaluationConfig,
    NumericalEncodingDataset,
    NumericalEncodingEvaluator
)
from numerical_encoding.evaluation import custom_collate_fn
from torch.utils.data import DataLoader

console = Console()

def set_all_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def create_evaluation_dataset(
    config: EvaluationConfig,
    generator: torch.Generator
) -> DataLoader:
    """Create evaluation dataset and dataloader."""
    dataset = NumericalEncodingDataset(config)
    
    # Add debug logging
    console.print(f"Generated {len(dataset)} total samples")
    console.print(f"Numerical samples: {dataset.total_numerical}")
    console.print(f"Text samples: {dataset.total_text}")
    
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=0,
        generator=generator,
        worker_init_fn=lambda x: np.random.seed(config.seed)
    )
    
def display_results_table(results: Dict[str, float]) -> None:
    """Display evaluation results in a formatted table."""
    if not results:
        console.print("[red]No results to display[/red]")
        return
        
    table = Table(title="Numerical Encoding Evaluation Results")
    table.add_column("Category", style="cyan", no_wrap=True)
    table.add_column("Metric", style="magenta")
    table.add_column("Score", justify="right", style="green")
    
    # Group metrics by category
    categories = {
        "Numerical": {
            "prefix": "numerical",
            "metrics": ["relative_distance", "scale_invariance", "sign_preservation",
                       "magnitude_preservation", "numerical_continuity", "interval_preservation"]
        },
        "Contextual": {
            "prefix": "contextual",
            "metrics": ["context_separation", "cross_context_understanding", 
                       "semantic_preservation"]
        },
        "Downstream": {
            "prefix": "downstream",
            "metrics": ["classification_performance", "clustering_quality", 
                       "semantic_similarity"]
        }
    }
    
    # Process each category
    for category_name, category_info in categories.items():
        category_metrics = []
        
        # Get all metrics for this category
        for metric in category_info["metrics"]:
            full_metric_name = metric
            if full_metric_name in results:
                category_metrics.append((metric, results[full_metric_name]))
        
        if category_metrics:
            # Add category header
            table.add_row(
                f"[bold]{category_name}[/bold]",
                "",
                "",
                style="bright_black"
            )
            
            # Add individual metrics
            for metric_name, score in category_metrics:
                name = metric_name.replace("_", " ").title()
                if not np.isnan(score):
                    table.add_row(
                        "",
                        name,
                        f"{score:.3f}"
                    )
            
            # Calculate and add category average
            valid_scores = [score for _, score in category_metrics if not np.isnan(score)]
            if valid_scores:
                avg_score = np.mean(valid_scores)
                table.add_row(
                    "",
                    "[bold italic]Category Average[/bold italic]",
                    f"[bold italic]{avg_score:.3f}[/bold italic]",
                    style="bright_black"
                )
            
            # Add separator
            table.add_row("", "", "")
    
    # Add overall score if present and valid
    if "overall_score" in results and not np.isnan(results["overall_score"]):
        table.add_row(
            "[bold red]Overall[/bold red]",
            "[bold red]Final Score[/bold red]",
            f"[bold red]{results['overall_score']:.3f}[/bold red]"
        )
    
    console.print(table)

def run_evaluation(args: argparse.Namespace) -> Dict[str, float]:
    """Run complete evaluation suite."""
    console.rule("[bold blue]Numerical Encoding Evaluation[/bold blue]")
    
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
    
    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        # Initialize encoder
        encoder_task = progress.add_task(
            "[cyan]Initializing encoder...",
            total=1
        )
        encoder = NumericalEncoder()
        progress.update(encoder_task, completed=1)
        
        # Create dataset
        data_task = progress.add_task(
            "[cyan]Creating evaluation dataset...",
            total=1
        )
        loader = create_evaluation_dataset(config, generator)
        progress.update(data_task, completed=1)
        
        # Initialize evaluator
        eval_init_task = progress.add_task(
            "[cyan]Initializing evaluator...",
            total=1
        )
        evaluator = NumericalEncodingEvaluator(encoder, config)
        progress.update(eval_init_task, completed=1)
        
        # Run complete evaluation
        eval_task = progress.add_task(
            "[cyan]Running evaluation...",
            total=1
        )
        results = evaluator.evaluate(loader)  # Use the main evaluate method
        progress.update(eval_task, completed=1)
    
    # Display results
    console.print("\n")
    if results:  # Only display if we have results
        display_results_table(results)
    else:
        console.print("[red]No results were generated during evaluation.[/red]")
    
    # Save results
    if args.save_results and results:
        save_dir = Path("results")
        save_dir.mkdir(exist_ok=True)
        evaluator.save_results(save_dir)
        console.print(f"\n[green]Results saved in {save_dir}[/green]")
    
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