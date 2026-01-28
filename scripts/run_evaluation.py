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
from transformers import BitsAndBytesConfig, GPT2LMHeadModel, GPT2Tokenizer

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
) -> tuple[nn.Module, any]:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Configuration dictionary
        device: Device to load model on
        
    Returns:
        Loaded model and tokenizer ready for evaluation
    """
    # Zero-shot evaluation
    if checkpoint_path.replace('-', '').lower().startswith('gpt2'):
        model_name = config["model_name"]
        print(f"Zero-shot with {model_name}")
        model, tokenizer = load_gpt2_model_and_tokenizer(model_name)
        model.to(device)
        model.eval()
        return model, tokenizer

    print(f"Loading checkpoint from {checkpoint_path}...")
    
    # Check if QLoRA was used
    training_config = config.get("training_config", {})
    use_qlora = training_config.get("use_qlora", False)
    
    # Load base model (with or without quantization)
    if use_qlora:
        print("Loading quantized model for QLoRA evaluation...")
        
        qlora_compute_dtype = training_config.get("qlora_compute_dtype", "float16")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16 if qlora_compute_dtype == "float16" else torch.float32,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["lm_head"]
        )
        
        base_model = GPT2LMHeadModel.from_pretrained(
            config["model_name"],
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        tokenizer = GPT2Tokenizer.from_pretrained(config["model_name"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        base_model, tokenizer = load_gpt2_model_and_tokenizer(config["model_name"])
    
    training_mode = config.get("training_mode") or training_config.get("training_mode")
    if training_mode == "lora":
        lora_config = config.get("lora") or config.get("lora_config", {})
        model = LoRAGPT2(
            base_model=base_model,
            rank=lora_config["rank"],
            alpha=lora_config["alpha"],
            target_modules=lora_config["target_modules"],
            dropout=lora_config.get("dropout", 0.0)
        )
    else:
        model = base_model
    
    # Load checkpoint weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if "model_state_dict" in checkpoint:
        # For QLoRA: use strict=False since checkpoint only has LoRA params
        model.load_state_dict(checkpoint["model_state_dict"], strict=not use_qlora)
        print(f"Loaded model state from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        model.load_state_dict(checkpoint, strict=not use_qlora)
        print("Loaded model state directly from checkpoint")
    
    #  quantized models already placed by device_map)
    if not use_qlora:
        model.to(device)
    
    model.eval()
    
    return model, tokenizer


def load_config(config_path: str) -> Dict:
    """Load configuration file (YAML or JSON)."""
    with open(config_path, 'r') as f:
        if config_path.endswith('.json'):
            config = json.load(f)
        else:
            config = yaml.safe_load(f)
    return config


def setup_test_data(config: Dict, tokenizer):
    """Create test dataset and dataloader."""
    print("Setting up test data...")
    
    # Get max_length from config, supporting both config.yaml and training_config.json formats
    max_length = config.get("max_length", 256)  # Default from config.yaml
    
    # Get sample_percentage, checking both top-level and nested 'dataset' key
    sample_percentage = config.get("sample_percentage") or config.get("dataset", {}).get("sample_percentage", 1.0)
    
    test_dataset = E2EDataset(
        split="test",
        tokenizer=tokenizer,
        max_length=max_length,
        sample_percentage=sample_percentage
        # NB: device parameter not needed here - data is moved to device in evaluation functions
    )
    
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
    parser.add_argument("--num_samples", type=int, default=-1,
                       help="Number of samples for text generation evaluation")
    parser.add_argument("--use_qlora", action="store_true",
                      help="Force QLoRA mode (if not detected from config)")
    
    args = parser.parse_args()
    
    print(f"Loading configuration from {args.config}...")
    config = load_config(args.config)
    
    if hasattr(args, 'use_qlora') and args.use_qlora:
        if "training_config" not in config:
            config["training_config"] = {}
        config["training_config"]["use_qlora"] = True    
    
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed(seed)
    
    model, tokenizer = load_checkpoint(args.checkpoint, config, device)
    
    test_loader, test_dataset = setup_test_data(config, tokenizer)
    
    print("Starting comprehensive evaluation...")
    
    # Extract generation config from config file (inference parameters from LoRA paper)
    eval_config = config.get("evaluation", {})
    inference_config = eval_config.get("inference", {})
    generation_config = eval_config.get("generation", {})
    
    # Merge inference config into generation config (inference params take precedence)
    # This maps beam_size -> num_beams for compatibility
    merged_generation_config = {
        "max_new_tokens": generation_config.get("max_new_tokens", 50),
        "num_beams": inference_config.get("beam_size", 10),
        "length_penalty": inference_config.get("length_penalty", 0.9),
        "no_repeat_ngram_size": inference_config.get("no_repeat_ngram_size", 4),
        "do_sample": False,  # Use beam search, not sampling
    }
    
    print(f"Generation config: {merged_generation_config}")
    
    results = evaluate_model_comprehensive(
        model=model,
        tokenizer=tokenizer,
        test_loader=test_loader,
        test_dataset=test_dataset,
        device=device,
        num_samples=args.num_samples,
        generation_config=merged_generation_config
    )
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    for metric, value in results.items():
        # Skip metadata fields (those starting with '_')
        if not metric.startswith("_"):
            print(f"{metric:15}: {value:.4f}")
    
    # Print evaluation info (multi-reference BLEU context)
    if "_eval_info" in results:
        print("\n" + "-"*60)
        print("EVALUATION INFO (Multi-Reference BLEU)")
        print("-"*60)
        info = results["_eval_info"]
        print(f"Unique MRs evaluated: {info.get('unique_mrs_evaluated', 'N/A')}")
        print(f"Total unique MRs:     {info.get('total_unique_mrs', 'N/A')}")
        print(f"Total test samples:   {info.get('total_test_samples', 'N/A')}")
        print(f"Avg refs per MR:      {info.get('avg_refs_per_mr', 'N/A'):.1f}")
    
    if "_examples" in results:
        print("\n" + "="*60)
        print("EXAMPLE GENERATIONS")
        print("="*60)
        
        for i, example in enumerate(results["_examples"]):
            print(f"\nExample {i+1}:")
            print(f"MR:              {example['mr']}")
            print(f"Prediction:      {example['prediction'][:100]}...")
            print(f"Num references:  {example['num_references']}")
            print(f"Sample ref:      {example['sample_reference'][:100]}...")
    
    output_dir = args.output or config.get("paths", {}).get("evaluation_dir", "./results/evaluations")
    os.makedirs(output_dir, exist_ok=True)
    
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = Path(args.checkpoint).stem
    output_file = os.path.join(output_dir, f"eval_{checkpoint_name}_{timestamp}.json")
    
    save_results = {
        "checkpoint": args.checkpoint,
        "config": {k: v for k, v in config.items() if k != "paths"},
        "timestamp": timestamp,
        "metrics": {k: v for k, v in results.items() if not k.startswith("_")}
    }
    
    if "_examples" in results:
        save_results["examples"] = results["_examples"]
    
    if "_eval_info" in results:
        save_results["eval_info"] = results["_eval_info"]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")



if __name__ == "__main__":
    main()