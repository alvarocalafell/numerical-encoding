"""
Environment test script to verify the numerical encoding setup and basic functionality.
"""

import torch
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Any
from numerical_encoding import NumericalEncoder
from rich.console import Console
from rich.table import Table

console = Console()

def test_cuda_availability() -> Dict[str, Any]:
    """Test CUDA availability and get device information."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }
    return info

def test_basic_encoding() -> Dict[str, bool]:
    """Test basic encoding functionality for numbers and text."""
    try:
        encoder = NumericalEncoder()
        results = {}
        
        # Test standalone number
        number = 3.14
        number_embedding = encoder.encode_number(number)
        results["standalone_number"] = (
            number_embedding.shape == torch.Size([1, 768]) and 
            not torch.isnan(number_embedding).any()
        )
        
        # Test text with number
        text = "Rated 5 stars"
        text_embedding = encoder.encode_number(text)
        results["text_with_number"] = (
            text_embedding.shape == torch.Size([1, 768]) and 
            not torch.isnan(text_embedding).any()
        )
        
        # Test similarity computation
        similarity = encoder.compute_similarity(number_embedding, text_embedding)
        results["similarity_computation"] = isinstance(similarity, float)
        
        return results
    
    except Exception as e:
        console.print(f"[red]Error in basic encoding test: {str(e)}[/red]")
        return {"error": str(e)}

def run_environment_test() -> None:
    """Run complete environment test suite and display results."""
    console.print("\n[bold blue]Running Numerical Encoding Environment Test[/bold blue]\n")
    
    # Test CUDA availability
    console.print("[yellow]Testing CUDA Setup...[/yellow]")
    cuda_info = test_cuda_availability()
    
    device_table = Table(title="Device Information")
    device_table.add_column("Property", style="cyan")
    device_table.add_column("Value", style="green")
    
    for key, value in cuda_info.items():
        device_table.add_row(key.replace("_", " ").title(), str(value))
    
    console.print(device_table)
    
    # Test Python environment
    console.print("\n[yellow]Testing Python Environment...[/yellow]")
    env_table = Table(title="Python Environment")
    env_table.add_column("Component", style="cyan")
    env_table.add_column("Version", style="green")
    
    env_table.add_row("Python", sys.version.split()[0])
    env_table.add_row("PyTorch", torch.__version__)
    env_table.add_row("NumPy", np.__version__)
    
    console.print(env_table)
    
    # Test basic encoding functionality
    console.print("\n[yellow]Testing Basic Encoding Functionality...[/yellow]")
    encoding_results = test_basic_encoding()
    
    results_table = Table(title="Encoding Tests")
    results_table.add_column("Test", style="cyan")
    results_table.add_column("Status", style="green")
    
    for test_name, passed in encoding_results.items():
        status = "[green]✓ Passed" if passed else "[red]✗ Failed"
        results_table.add_row(test_name.replace("_", " ").title(), status)
    
    console.print(results_table)
    
    # Overall status
    all_passed = all(encoding_results.values())
    if all_passed:
        console.print("\n[green]✨ All tests passed successfully![/green]")
    else:
        console.print("\n[red]❌ Some tests failed. Please check the results above.[/red]")

if __name__ == "__main__":
    run_environment_test()