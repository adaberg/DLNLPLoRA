"""
Trainer for LoRA fine-tuning following the original paper.
TODO: Possibly adapt to better fit our data changes if necessary!
Hyperparameters from Table 11 (GPT-2 on E2E NLG):
- Optimizer: AdamW
- Weight Decay: 0.01
- Batch Size: 8
- Epochs: 5
- Warmup Steps: 500
- LR Schedule: Linear
- Learning Rate: 2e-4
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
import os
import json
import logging
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import numpy as np
import random

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration following LoRA paper Table 11."""

    # Training hyperparameters
    #  (reused from paper! TODO: Possibly adapt to better fit our data changes if necessary!)
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    num_epochs: int = 5
    batch_size: int = 8
    warmup_steps: int = 500

    # LoRA specific
    lora_dropout: float = 0.1

    # Logging and checkpointing
    output_dir: str = "./results"
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Gradient accumulation for larger effective batch sizes
    gradient_accumulation_steps: int = 1

    # Seed
    seed: int = 42

    # Training mode: "lora", "full", or "none" (evaluation only)
    training_mode: str = "lora"

    # Early stopping
    early_stopping_patience: Optional[int] = None

    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Linear warmup then linear decay scheduler (as used in LoRA paper).
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


class Trainer:
    """
    Trainer class for LoRA fine-tuning.
    Implements the training procedure from the LoRA paper.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        tokenizer: Any = None,
    ) -> None:

        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)

        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.tokenizer = tokenizer

        # Move model to device
        self.model.to(config.device)

        # Setup optimizer (only for trainable parameters)
        self.optimizer = self._create_optimizer()

        # Calculate total training steps (account for gradient accumulation properly)
        steps_per_epoch = math.ceil(
            len(train_dataloader) / config.gradient_accumulation_steps
        )
        self.total_steps = steps_per_epoch * config.num_epochs

        # Setup scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=self.total_steps,
        )

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_loss = float("inf")
        self.patience_counter = 0

        # History
        self.train_losses: List[float] = []
        self.eval_losses: List[float] = []

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

        # Log configuration
        self._log_config()

    def _create_optimizer(self) -> AdamW:
        """Create AdamW optimizer for trainable parameters only."""
        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        # Log parameter counts
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_count = sum(p.numel() for p in trainable_params)

        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_count:,}")
        logger.info(
            f"Trainable percentage: {100 * trainable_count / total_params:.2f}%"
        )

        return AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),  # PyTorch defaults, not specified in paper
            eps=1e-8,  # PyTorch default, not specified in paper
        )

    def _log_config(self) -> None:
        """Log training configuration."""
        logger.info("=" * 60)
        logger.info("Training Configuration")
        logger.info("=" * 60)
        logger.info(f"Training mode: {self.config.training_mode}")
        logger.info(f"Learning rate: {self.config.learning_rate}")
        logger.info(f"Weight decay: {self.config.weight_decay}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(
            f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}"
        )
        logger.info(f"Effective batch size: {self.config.effective_batch_size()}")
        logger.info(f"Number of epochs: {self.config.num_epochs}")
        logger.info(f"Warmup steps: {self.config.warmup_steps}")
        logger.info(f"Total training steps: {self.total_steps}")
        logger.info(f"Device: {self.config.device}")
        logger.info("=" * 60)

    def train(self) -> Dict[str, Any]:
        """
        Main training loop.

        Returns:
            Dictionary with training results and metrics.
        """
        logger.info("Starting training...")

        self.model.train()

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            epoch_loss = self._train_epoch(epoch)
            self.train_losses.append(epoch_loss)

            logger.info(
                f"Epoch {epoch + 1}/{self.config.num_epochs} - Train Loss: {epoch_loss:.4f}"
            )

            # Evaluation
            if self.eval_dataloader is not None:
                eval_loss = self.evaluate()
                self.eval_losses.append(eval_loss)
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.num_epochs} - Eval Loss: {eval_loss:.4f}"
                )

                # Save best model
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self.save_checkpoint("best_model")
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                # Early stopping
                if (
                    self.config.early_stopping_patience is not None
                    and self.patience_counter >= self.config.early_stopping_patience
                ):
                    logger.info(f"Early stopping after {epoch + 1} epochs")
                    break

            # Save checkpoint at end of epoch
            self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}")

        # Save final model
        self.save_checkpoint("final_model")

        return {
            "train_losses": self.train_losses,
            "eval_losses": self.eval_losses,
            "best_eval_loss": self.best_eval_loss,
            "total_steps": self.global_step,
            "epochs_completed": self.current_epoch + 1,
        }

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            self.train_dataloader, desc=f"Epoch {epoch + 1}", disable=False
        )

        self.optimizer.zero_grad()

        for step, batch in enumerate(progress_bar):
            loss = self._training_step(batch)

            # Store original loss for logging before scaling
            loss_value = loss.item()

            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            total_loss += loss_value
            num_batches += 1

            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:

                # Optimizer step
                self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    avg_loss = total_loss / num_batches
                    lr = self.scheduler.get_last_lr()[0]
                    logger.info(
                        f"Step {self.global_step} - Loss: {avg_loss:.4f} - LR: {lr:.2e}"
                    )

                # Evaluation during training
                if (
                    self.config.eval_steps > 0
                    and self.global_step % self.config.eval_steps == 0
                    and self.eval_dataloader is not None
                ):
                    eval_loss = self.evaluate()
                    logger.info(f"Step {self.global_step} - Eval Loss: {eval_loss:.4f}")

                # Save checkpoint
                if (
                    self.config.save_steps > 0
                    and self.global_step % self.config.save_steps == 0
                ):
                    self.save_checkpoint(f"checkpoint_step_{self.global_step}")

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "loss": f"{total_loss / num_batches:.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                }
            )

        # After the for loop ends, apply accumulated gradients if any remain
        if len(self.train_dataloader) % self.config.gradient_accumulation_steps != 0:

            self.optimizer.step()

            self.scheduler.step()
            self.optimizer.zero_grad()
            self.global_step += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a single training step."""
        # Move batch to device
        batch = {
            k: v.to(self.config.device)
            for k, v in batch.items()
            if isinstance(v, torch.Tensor)
        }

        # Forward pass
        outputs = self.model(**batch)
        loss = outputs.loss

        return loss

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate the model on the evaluation dataset."""
        if self.eval_dataloader is None:
            return float("inf")

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            batch = {
                k: v.to(self.config.device)
                for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }

            outputs = self.model(**batch)
            total_loss += outputs.loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")

        # Restore training mode
        self.model.train()

        return avg_loss

    def save_checkpoint(self, name: str) -> str:
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.config.output_dir, name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model state
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_eval_loss": self.best_eval_loss,
            "train_losses": self.train_losses,
            "eval_losses": self.eval_losses,
            "config": {
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "num_epochs": self.config.num_epochs,
                "batch_size": self.config.batch_size,
                "warmup_steps": self.config.warmup_steps,
                "training_mode": self.config.training_mode,
            },
        }

        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        torch.save(checkpoint, checkpoint_path)

        # Save config as JSON
        config_path = os.path.join(checkpoint_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(checkpoint["config"], f, indent=2)

        logger.info(f"Checkpoint saved to {checkpoint_dir}")

        # Cleanup old checkpoints if needed
        self._cleanup_checkpoints()

        return checkpoint_dir

    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints if exceeding save_total_limit."""
        if self.config.save_total_limit is None or self.config.save_total_limit <= 0:
            return

        checkpoints = []
        for name in os.listdir(self.config.output_dir):
            path = os.path.join(self.config.output_dir, name)
            # Match both step and epoch checkpoints
            if os.path.isdir(path) and (
                name.startswith("checkpoint_step_")
                or name.startswith("checkpoint_epoch_")
            ):
                # Use modification time for sorting instead of numeric suffix
                mtime = os.path.getmtime(path)
                checkpoints.append((mtime, path))

        # Sort by modification time (oldest first)
        checkpoints.sort(key=lambda x: x[0])

        # Remove oldest checkpoints
        while len(checkpoints) > self.config.save_total_limit:
            _, path = checkpoints.pop(0)
            logger.info(f"Removing old checkpoint: {path}")
            import shutil

            shutil.rmtree(path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_eval_loss = checkpoint.get("best_eval_loss", float("inf"))
        self.train_losses = checkpoint.get("train_losses", [])
        self.eval_losses = checkpoint.get("eval_losses", [])

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(
            f"Resuming from epoch {self.current_epoch}, step {self.global_step}"
        )
