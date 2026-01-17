# Dataset Module Usage Guide

## Quick Start
```python
from dataset import create_datasets_and_loaders

# Get all data loaders with one command
loaders = create_datasets_and_loaders(tokenizer)

# Use in training
train_loader = loaders["train"]      # Training data
val_loader = loaders["validation"]   # Validation data
test_loader = loaders["test"]        # Test data

# For Debugging

from dataset import E2EDataset, get_dataloader

# For debugging or custom setups
dataset = E2EDataset(
    split="train",
    tokenizer=tokenizer,
    max_length=256,
    sample_percentage=0.1  # Use 10% data for quick testing
)

# Inspect data
print(dataset.get_raw_sample(0))

# Create custom loader
loader = get_dataloader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=2
)

Notes
create_datasets_and_loaders(): For normal training

E2EDataset() + get_dataloader(): For debugging or custom needs
