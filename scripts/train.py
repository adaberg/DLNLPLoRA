#!/usr/bin/env python3
"""
Training script for LoRA fine-tuning following the original paper.
(TODO: Possibly needs adaptation for better fit our data changes if necessary!)

Usage:
    # LoRA training (default)
    python scripts/train.py --config config.yaml --mode lora

    # Full fine-tuning baseline
    python scripts/train.py --config config.yaml --mode full

    # With custom hyperparameters
    python scripts/train.py --config config.yaml --mode lora --lr 2e-4 --epochs 5 --batch_size 8

Cloud Deployment:
    # For GPU/TPU training, the script automatically detects available devices
"""

import argparse
import logging
import os
import sys
import json
from datetime import datetime
from pathlib import Path

import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.gpt2_wrapper import load_gpt2_model_and_tokenizer, count_parameters
from src.data.dataset import E2EDataset, get_dataloader
from src.lora.model import LoRAGPT2
from src.training.trainer import Trainer, TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_model(config: dict, training_mode: str):
    """
    Setup model based on training mode.

    Args:
        config: Configuration dictionary
        training_mode: "lora", "full", or "none"

    Returns:
        model, tokenizer
    """
    logger.info(f"Loading model: {config['model_name']}")
    base_model, tokenizer = load_gpt2_model_and_tokenizer(config['model_name'])

    if training_mode == "lora":
        logger.info("Setting up LoRA model...")
        model = LoRAGPT2(
            base_model=base_model,
            rank=config['lora']['rank'],
            alpha=config['lora']['alpha'],
            target_modules=config['lora']['target_modules'],
            dropout=config['lora']['dropout']
        )
        logger.info(f"LoRA modules injected: {model.lora_modules}")
    elif training_mode == "full":
        logger.info("Setting up full fine-tuning...")
        model = base_model
        for param in model.parameters():
            param.requires_grad = True
    else:  # none - evaluation only
        logger.info("Setting up model for evaluation only (no training)...")
        model = base_model
        for param in model.parameters():
            param.requires_grad = False

    # Log parameter counts
    total, trainable = count_parameters(model)
    logger.info(f"Total parameters: {total:,}")
    logger.info(f"Trainable parameters: {trainable:,}")
    logger.info(f"Trainable percentage: {100 * trainable / total:.4f}%")

    return model, tokenizer


def setup_data(config: dict, tokenizer, training_config: TrainingConfig):
    """Setup datasets and dataloaders."""
    logger.info("Setting up datasets...")

    #max_length = config.get('max_length', 256)
    max_length = config.get('max_length', 128)
    sample_percentage = config.get('sample_percentage', 1.0)

    train_dataset = E2EDataset(
        split="train",
        tokenizer=tokenizer,
        max_length=max_length,
        sample_percentage=sample_percentage
    )

    val_dataset = E2EDataset(
        split="validation",
        tokenizer=tokenizer,
        max_length=max_length,
        sample_percentage=sample_percentage
    )

    train_loader = get_dataloader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True
    )

    val_loader = get_dataloader(
        val_dataset,
        batch_size=training_config.batch_size * 2, # Larger batch for evaluation
        shuffle=False
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")

    return train_loader, val_loader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train LoRA model following the original paper methodology"
    )

    # Config file
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )

    # Training mode
    parser.add_argument(
        "--mode", type=str, default="lora",
        choices=["lora", "full", "none"],
        help="Training mode: lora, full fine-tuning, or none (eval only)"
    )

    # Hyperparameters (override config)
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--warmup_steps", type=int, default=None, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=None, help="Max gradient norm")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None,
                       help="Gradient accumulation steps")

    # Output
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name")

    # Device
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda/cpu/mps)")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 mixed precision")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 mixed precision")

    # Checkpointing
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint path")
    parser.add_argument("--save_steps", type=int, default=None, help="Save every N steps")
    parser.add_argument("--eval_steps", type=int, default=None, help="Evaluate every N steps")

    # Logging
    parser.add_argument("--logging_steps", type=int, default=None, help="Log every N steps")

    # Seed
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    DEBUG_MODE = True

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Determine device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
        logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using Apple MPS")
    else:
        device = "cpu"
        logger.info("Using CPU")

    # Set seed
    seed = args.seed if args.seed is not None else config.get('seed', 42)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed: {seed}")

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or config.get('experiment_name', 'lora_training')
    base_output_dir = args.output_dir or config.get('output_dir', './results')
    output_dir = os.path.join(base_output_dir, f"{experiment_name}_{args.mode}_{timestamp}")

    # Create training configuration
    # Use hyperparameters from paper (Table 11) as defaults
    # TODO: Possibly needs adaptation for better fit our data changes if necessary!
    # NB: YAML may parse scientific notation (e.g., 2e-4) as strings, so we explicitly convert to float/int
    training_config = TrainingConfig(
        learning_rate=float(args.lr or config.get('training', {}).get('learning_rate', 2e-4)),
        weight_decay=float(args.weight_decay or config.get('training', {}).get('weight_decay', 0.01)),
        num_epochs=int(args.epochs or config.get('training', {}).get('num_epochs', 5)),
        batch_size=int(args.batch_size or config.get('training', {}).get('batch_size', 8)),
        warmup_steps=int(args.warmup_steps or config.get('training', {}).get('warmup_steps', 500)),
        max_grad_norm=float(args.max_grad_norm or config.get('training', {}).get('max_grad_norm', 1.0)),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps or
            config.get('training', {}).get('gradient_accumulation_steps', 1)),
        output_dir=output_dir,
        logging_steps=args.logging_steps or config.get('training', {}).get('logging_steps', 100),
        eval_steps=args.eval_steps or config.get('training', {}).get('eval_steps', 500),
        save_steps=args.save_steps or config.get('training', {}).get('save_steps', 500),
        device=device,
        fp16=args.fp16,
        bf16=args.bf16,
        seed=seed,
        training_mode=args.mode,
        lora_dropout=config.get('lora', {}).get('dropout', 0.1),
    )

    # Setup model
    model, tokenizer = setup_model(config, args.mode)

    # Setup data
    train_loader, val_loader = setup_data(config, tokenizer, training_config)

    # Create trainer
    trainer = Trainer(
        model=model,
        config=training_config,
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
        tokenizer=tokenizer,
        debug_mode=DEBUG_MODE
    )

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Save training configuration
    os.makedirs(output_dir, exist_ok=True)
    config_save_path = os.path.join(output_dir, "training_config.json")
    with open(config_save_path, 'w') as f:
        json.dump({
            "training_mode": args.mode,
            "model_name": config['model_name'],
            "lora_config": config.get('lora', {}),
            "training_config": {
                "learning_rate": training_config.learning_rate,
                "weight_decay": training_config.weight_decay,
                "num_epochs": training_config.num_epochs,
                "batch_size": training_config.batch_size,
                "warmup_steps": training_config.warmup_steps,
                "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
                "fp16": training_config.fp16,
                "bf16": training_config.bf16,
            },
            "dataset": {
                "name": config.get('dataset_name', 'e2e_nlg'),
                "sample_percentage": config.get('sample_percentage', 1.0),
            },
            "seed": seed,
            "device": device,
        }, f, indent=2)
    logger.info(f"Configuration saved to {config_save_path}")

    # Add file handler for logging
    file_handler = logging.FileHandler(os.path.join(output_dir, "training.log"))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

    # Train
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)

    results = trainer.train()

    # Save final results
    results_path = os.path.join(output_dir, "training_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info("=" * 60)
    logger.info("Training completed!")
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"Best validation loss: {results['best_eval_loss']:.4f}")
    logger.info(f"Total steps: {results['total_steps']}")
    logger.info(f"Epochs completed: {results['epochs_completed']}")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    main()
