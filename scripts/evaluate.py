"""
Standalone evaluation on test set with generation examples.
"""

import argparse
import yaml
import torch
import torch.nn as nn
from typing import Dict
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.gpt2_wrapper import load_gpt2_model_and_tokenizer
from src.data.dataset import E2EDataset, get_dataloader
from src.evaluation.metrics import (
    compute_perplexity, 
    generate_texts, 
    compute_generation_metrics,
    evaluate_model_comprehensive
)
from src.lora.model import LoRAGPT2


def load_checkpoint(
    checkpoint_path: str,
    config: Dict,
    device: str
) -> nn.Module:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Configuration dictionary
        device: Device to load model on
        
    Returns:
        Loaded model ready for evaluation
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    # Load base model and tokenizer
    model, tokenizer = load_gpt2_model_and_tokenizer(config["model_name"])
    
    # Apply LoRA if needed
    if config.get("training_mode") == "lora":
        model = LoRAGPT2(
            base_model=model,
            rank=config["lora"]["rank"],
            alpha=config["lora"]["alpha"],
            target_modules=config["lora"]["target_modules"],
            dropout=config["lora"]["dropout"]
        )
    
    # Load checkpoint weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model state from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        # Try loading directly (for saved models, not trainer checkpoints)
        model.load_state_dict(checkpoint)
        print("Loaded model state directly from checkpoint")
    
    model.to(device)
    model.eval()
    
    return model, tokenizer


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_test_data(config: Dict, tokenizer):
    """Create test dataset and dataloader."""
    print("Setting up test data...")
    
    # Create test dataset
    test_dataset = E2EDataset(
        split="test",
        tokenizer=tokenizer,
        max_length=config["max_length"],
        sample_percentage=config.get("sample_percentage", 1.0),
        device=config["device"] if config.get("device") else "cpu"
    )
    
    # Create test dataloader
    test_loader = get_dataloader(
        test_dataset,
        batch_size=config.get("val_batch_size", config.get("batch_size", 16)),
        shuffle=False
    )
    
    print(f"Test dataset size: {len(test_dataset)} samples")
    return test_loader, test_dataset


def main() -> None:
    """
    Main evaluation function.
    1. Parse arguments (--checkpoint, --config)
    2. Load model
    3. Load test data
    4. Compute perplexity
    5. Generate sample outputs
    6. Compute BLEU/ROUGE
    7. Save evaluation report
    """
    parser = argparse.ArgumentParser(description="Evaluate trained LoRA model")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to config file (default: config.yaml)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory for evaluation results")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of samples for text generation evaluation")
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}...")
    config = load_config(args.config)
    
    # Set device
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed(seed)
    
    # Load model from checkpoint
    model, tokenizer = load_checkpoint(args.checkpoint, config, device)
    
    # Setup test data
    test_loader, test_dataset = setup_test_data(config, tokenizer)
    
    # Run comprehensive evaluation
    print("\n" + "="*60)
    print("Starting comprehensive evaluation...")
    print("="*60)
    
    results = evaluate_model_comprehensive(
        model=model,
        tokenizer=tokenizer,
        test_loader=test_loader,
        test_dataset=test_dataset,
        device=device,
        num_samples=args.num_samples
    )
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    for metric, value in results.items():
        if metric != "_examples":
            print(f"{metric:15}: {value:.4f}")
    
    # Print example generations
    if "_examples" in results:
        print("\n" + "="*60)
        print("EXAMPLE GENERATIONS")
        print("="*60)
        
        for i, example in enumerate(results["_examples"]):
            print(f"\nExample {i+1}:")
            print(f"Prompt:     {example['prompt']}")
            print(f"Prediction: {example['prediction'][:100]}...")
            print(f"Reference:  {example['reference'][:100]}...")
    
    # Save evaluation report
    output_dir = args.output or config.get("paths", {}).get("evaluation_dir", "./results/evaluations")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = Path(args.checkpoint).stem
    output_file = os.path.join(output_dir, f"eval_{checkpoint_name}_{timestamp}.json")
    
    # Prepare results for saving
    save_results = {
        "checkpoint": args.checkpoint,
        "config": {k: v for k, v in config.items() if k != "paths"},
        "timestamp": timestamp,
        "metrics": {k: v for k, v in results.items() if k != "_examples"}
    }
    
    if "_examples" in results:
        save_results["examples"] = results["_examples"]
    
    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    
    # Also save a simple text summary
    summary_file = os.path.join(output_dir, f"eval_summary_{checkpoint_name}_{timestamp}.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Evaluation Results for {checkpoint_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("="*50 + "\n\n")
        
        f.write("METRICS:\n")
        for metric, value in results.items():
            if metric != "_examples":
                f.write(f"  {metric:15}: {value:.4f}\n")
        
        if "_examples" in results:
            f.write("\nEXAMPLES:\n")
            for i, example in enumerate(results["_examples"]):
                f.write(f"\nExample {i+1}:\n")
                f.write(f"  Prompt:     {example['prompt']}\n")
                f.write(f"  Prediction: {example['prediction']}\n")
                f.write(f"  Reference:  {example['reference']}\n")
    
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
