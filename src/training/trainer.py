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
    max_grad_norm: float = 1.0
    
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
    
    # Mixed precision
    fp16: bool = False
    bf16: bool = False
    
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
    last_epoch: int = -1
) -> LambdaLR:
    """
    Linear warmup then linear decay scheduler (as used in LoRA paper).
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / 
            float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


class Trainer:
    """
    Trainer class for LoRA fine-tuning.
    Implements the training procedure from the LoRA paper.
    """

    _DEBUG = False

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        tokenizer: Any = None,
        debug_mode: bool = False
    ) -> None:
        if debug_mode:
            self._DEBUG = True

        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.tokenizer = tokenizer
        
        # Move model to device
        self.model.to(config.device)
        
        # Setup optimizer (only for trainable parameters)
        self.optimizer = self._create_optimizer()
        
        # Calculate total training steps
        self.total_steps = (
            len(train_dataloader) // config.gradient_accumulation_steps
        ) * config.num_epochs
        
        # Setup scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=self.total_steps
        )
        
        # Mixed precision
        self.scaler = None
        if config.fp16:
            self.scaler = torch.amp.GradScaler('cuda')
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_loss = float('inf')
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
        logger.info(f"Trainable percentage: {100 * trainable_count / total_params:.2f}%")

        optimizer = AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        return optimizer

    
    def _log_config(self) -> None:
        """Log training configuration."""
        logger.info("=" * 60)
        logger.info("Training Configuration")
        logger.info("=" * 60)
        logger.info(f"Training mode: {self.config.training_mode}")
        logger.info(f"Learning rate: {self.config.learning_rate}")
        logger.info(f"Weight decay: {self.config.weight_decay}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        logger.info(f"Effective batch size: {self.config.effective_batch_size()}")
        logger.info(f"Number of epochs: {self.config.num_epochs}")
        logger.info(f"Warmup steps: {self.config.warmup_steps}")
        logger.info(f"Total training steps: {self.total_steps}")
        logger.info(f"Device: {self.config.device}")
        logger.info(f"FP16: {self.config.fp16}")
        logger.info(f"BF16: {self.config.bf16}")
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

            #train_loss = self._train_epoch(epoch) # batch-weighted 
            train_loss = self._train_epoch_token_weighted(epoch) # paper-conform
            self.train_losses.append(train_loss)
            
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs} - Train Loss: {train_loss:.4f}")
            
            ## Evaluation:
            eval_loss = None
            if self.eval_dataloader is not None:
                #eval_loss = self.evaluate() # batch-weighted
                eval_loss = self.evaluate_token_weighted() # paper-conform
                self.eval_losses.append(eval_loss)
                logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs} - Eval Loss: {eval_loss:.4f}")
                
                # Save best model
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self.save_checkpoint("best_model")
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                # Early stopping
                if (self.config.early_stopping_patience is not None and 
                    self.patience_counter >= self.config.early_stopping_patience):
                    logger.info(f"Early stopping after {epoch + 1} epochs")
                    break
            
            # Save checkpoint at end of epoch
            self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}")
            self.model.train()
        
        # Save final model
        self.save_checkpoint("final_model")
        
        return {
            "train_losses": self.train_losses,
            "eval_losses": self.eval_losses,
            "best_eval_loss": self.best_eval_loss,
            "total_steps": self.global_step,
            "epochs_completed": self.current_epoch + 1
        }

    def _train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch with batch-weighted averaging (each batch
        contributes equally to the average.).
        Returns:
            Average loss per batch for the entire epoch.        
        """
        if self._DEBUG:
            batch_losses = []

        if not self.model.training:
            self.model.train()

        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch + 1}",
            disable=False
        )

        self.optimizer.zero_grad()

        for step, batch in enumerate(progress_bar):
            # Forward pass:
            loss = self._training_step(batch)

            if self._DEBUG:
                batch_losses.append(loss.item())

            # Scale loss for gradient accumulation:
            loss = loss / self.config.gradient_accumulation_steps

            # Backward pass:
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Debug mode: Check gradient flow
            # (The base weights must not have a gradient.)
            if self._DEBUG:
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        if param.grad is None:
                            print(f"No grad for {name}")

            # Batch-weighted accumulation (each batch counts equally):
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1

            # Aaccumlation of the gradients:
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.max_grad_norm > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )

                # Optimizer step:
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                # Logging of the average loss and LR:
                if self.global_step % self.config.logging_steps == 0:
                    avg_loss = total_loss / num_batches
                    lr = self.scheduler.get_last_lr()[0]
                    logger.info(
                        f"Step {self.global_step} - Avg. Batch Loss: {avg_loss:.4f} - LR: {lr:.2e}"
                    )
                # Note: The evaluation part is moved to a clearly defined, epoch-based position.
                #       Here, the evaluation would be calculated in the middle of the gradient
                #       flow and at the step level.

                if self._DEBUG:
                    batch_loss = sum(batch_losses) / len(batch_losses)
                    print(f"Batch-loss: {batch_loss:.4f}")

            # Update progress bar:
            progress_bar.set_postfix({
                "batch_loss": f"{total_loss / num_batches:.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
            })

        # Batch-weighted average:
        return total_loss / num_batches if num_batches > 0 else 0.0

    def _train_epoch_token_weighted(self, epoch: int) -> float:
        """
        Train for one epoch with token-weighted loss averaging (each token
        contributes equally to the average).
        Returns:
            Average loss per token for the entire epoch. 
        """
        if self._DEBUG:
            batch_losses = []
            token_losses = []
            token_counts = []

        if not self.model.training:
            self.model.train()

        total_loss = 0.0
        total_tokens = 0

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch + 1}",
            disable=False
        )

        self.optimizer.zero_grad()

        for step, batch in enumerate(progress_bar):
            # Forward pass:
            # (loss: mean CE per token, returned by the model)
            loss, num_tokens = self._training_step_token_weighted(batch) 

            if self._DEBUG:
                batch_losses.append(loss.item())
                token_losses.append(loss.item() * num_tokens)
                token_counts.append(num_tokens)

            # Scale loss for gradient accumulation:
            # (This eliminates the need for a multiplication to scale back the total loss.)
            loss_scaled = loss / self.config.gradient_accumulation_steps # only for backward pass
            
            # Backward pass (scaled):
            if self.scaler is not None:
                self.scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            # Debug mode: Check gradient flow
            # (The base weights must not have a gradient.)
            if self._DEBUG:
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        if param.grad is None:
                            print(f"No grad for {name}")

            # Token-weighted accumulation (is more consistent):
            # (Multiplication by 'gradient_accumulation_steps' to reverse the scaling.)
            #total_loss += loss.item() * num_tokens * self.config.gradient_accumulation_steps
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            # Accumlation of the gradients:
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.max_grad_norm > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )

                # Optimizer step:
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                # Logging of the average loss and LR:
                if self.global_step % self.config.logging_steps == 0:
                    avg_loss = total_loss / total_tokens
                    lr = self.scheduler.get_last_lr()[0]
                    logger.info(
                        f"Step {self.global_step} - Avg. Token Loss: {avg_loss:.4f} - LR: {lr:.2e}"
                    )
                # Note: The evaluation part is moved to a clearly defined, epoch-based position.
                #       Here, the evaluation would be calculated in the middle of the gradient
                #       flow and at the step level.

                if self._DEBUG:
                    batch_loss = sum(batch_losses) / len(batch_losses)
                    token_loss = sum(token_losses) / sum(token_counts)

                    print(f"Batch-loss: {batch_loss:.4f}")
                    print(f"Token-loss: {token_loss:.4f}")
                    print(f"Δ: {batch_loss - token_loss:.4f}")

            # Update progress bar:
            progress_bar.set_postfix({
                "token_loss": f"{total_loss / total_tokens:.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
            })

        # Token-weighted average:
        return total_loss / total_tokens if total_tokens > 0 else 0.0

    def _training_step_token_weighted(self, batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, int]:
        """
        Perform a single training step and returns loss and token count.
        Returns:
            tuple: (loss, num_tokens)
        """

        # Move batch to device:
        batch = {k: v.to(self.config.device) for k, v in batch.items()
                if isinstance(v, torch.Tensor)}

        if self._DEBUG:
            assert "labels" in batch
            assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()

        attention_mask = batch.get("attention_mask", None)

        # Forward pass:
        if self.config.fp16:
            with torch.amp.autocast("cuda"):
                outputs = self.model(**batch)
                loss = outputs.loss
        elif self.config.bf16:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.model(**batch)
                loss = outputs.loss
        else:
            outputs = self.model(**batch)
            loss = outputs.loss

        # Determine the number of valid tokens (excluding padding):
        if attention_mask is not None:
            num_tokens = attention_mask.sum().item()
        else:
            # Fallback: count all tokens if no attention mask
            num_tokens = batch["input_ids"].numel()

        return loss, num_tokens

    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a single training step and returns the loss."""
        # Move batch to device
        batch = {k: v.to(self.config.device) for k, v in batch.items() 
                 if isinstance(v, torch.Tensor)}
        
        # Forward pass
        if self.config.fp16:
            with torch.amp.autocast('cuda'):
                outputs = self.model(**batch)
                loss = outputs.loss
        elif self.config.bf16:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = self.model(**batch)
                loss = outputs.loss
        else:
            outputs = self.model(**batch)
            loss = outputs.loss
        
        return loss
    
    @torch.no_grad()
    def evaluate_token_weighted(self) -> float:
        """
        Evaluate the model on the evaluation dataset with
        token-weighted averaging (matching training).
        """
        if self.eval_dataloader is None:
            return float("inf")

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            batch = {k: v.to(self.config.device) for k, v in batch.items()
                    if isinstance(v, torch.Tensor)}

            attention_mask = batch.get("attention_mask", None)

            outputs = self.model(**batch)
            loss = outputs.loss

            # Count number of valid tokens:
            if attention_mask is not None:
                num_tokens = attention_mask.sum().item()
            else:
                num_tokens = batch["input_ids"].numel()

            # Token-weighted accumulation:
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

        return total_loss / total_tokens
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Evaluate the model on the evaluation dataset with
        batch-weighted averaging.
        """
        if self.eval_dataloader is None:
            return float('inf')
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            batch = {k: v.to(self.config.device) for k, v in batch.items()
                     if isinstance(v, torch.Tensor)}
            
            outputs = self.model(**batch)
            total_loss += outputs.loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else float('inf')
    
    def save_checkpoint(self, name: str) -> str:
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.config.output_dir, name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model state
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_eval_loss': self.best_eval_loss,
            'train_losses': self.train_losses,
            'eval_losses': self.eval_losses,
            'config': {
                'learning_rate': self.config.learning_rate,
                'weight_decay': self.config.weight_decay,
                'num_epochs': self.config.num_epochs,
                'batch_size': self.config.batch_size,
                'warmup_steps': self.config.warmup_steps,
                'training_mode': self.config.training_mode,
            }
        }
        
        checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save config as JSON
        config_path = os.path.join(checkpoint_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(checkpoint['config'], f, indent=2)
        
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
            if os.path.isdir(path) and name.startswith("checkpoint_step_"):
                try:
                    step = int(name.split("_")[-1])
                    checkpoints.append((step, path))
                except ValueError:
                    continue
        
        # Sort by step number
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
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_eval_loss = checkpoint.get('best_eval_loss', float('inf'))
        self.train_losses = checkpoint.get('train_losses', [])
        self.eval_losses = checkpoint.get('eval_losses', [])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")
