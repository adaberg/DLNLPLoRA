"""
E2E NLG Challenge dataset loader with proper EOS handling.
"""

from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizer
import logging
import os

logger = logging.getLogger(__name__)


class E2EDataset(Dataset):
    """
    E2E NLG dataset for language modeling.

    IMPORTANT: Adds EOS token after reference text so model learns to stop!
    """

    def __init__(
        self,
        split: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128, # reduced by half from 256, E2E samples are short!
        sample_percentage: float = 1.0,
        device: Optional[str] = None,
        add_eos_token: bool = True  # add EOS after reference
    ) -> None:
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.add_eos_token = add_eos_token

        # Ensure tokenizer has EOS token
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer must have an EOS token!")

        # Local CSV file paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        e2e_dir = os.path.join(current_dir, "e2e_data")

        split_files = {
            "train": os.path.join(e2e_dir, "train.csv"),
            "validation": os.path.join(e2e_dir, "dev.csv"),
            "test": os.path.join(e2e_dir, "test.csv")
        }

        csv_path = split_files[split]
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File not found: {csv_path}\nRun download_data.py first")

        dataset = load_dataset("csv", data_files={split: csv_path})
        dataset = dataset[split]

        if 'mr' in dataset.column_names:
            dataset = dataset.rename_column('mr', 'meaning_representation')
        if 'ref' in dataset.column_names:
            dataset = dataset.rename_column('ref', 'human_reference')

        # Sample if needed
        if sample_percentage < 1.0:
            num_samples = int(len(dataset) * sample_percentage)
            dataset = dataset.select(range(num_samples))

        self.dataset = dataset

        logger.info(f"Loaded {len(self.dataset)} samples from {split}")
        if add_eos_token:
            logger.info(f"EOS token will be added after references")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.dataset[idx]
        meaning_rep = sample["meaning_representation"]
        reference = sample["human_reference"]

        # Format with EOS token:
        prompt = f"MR: {meaning_rep}\nREF:"
        if self.add_eos_token:
            text = f"{prompt} {reference}{self.tokenizer.eos_token}" # see (**) 
        else:
            text = f"{prompt} {reference}"

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        attention_mask = encoding["attention_mask"].squeeze(0)
        labels = encoding["input_ids"].squeeze(0).clone()

        # Ignore padding tokens in loss:
        labels[attention_mask == 0] = -100

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": attention_mask,
            "labels": labels
        }

        if self.device:
            item = {k: v.to(self.device) for k, v in item.items()}

        return item

    @staticmethod
    def format_sample(meaning_rep: str, reference: str, eos_token: str = "") -> str:
        """Format a single sample for language modeling."""
        # (**) Note: Do not use the rarely used pipe character "|", it is a strong
        #            and rare token and GPT-2 has hardly been trained on it.
        #            --> It strongly influences the BLEU score.
        # return f"meaning_representation: {meaning_rep.strip()} | reference: {reference.strip()}{eos_token}"
        prompt = f"MR: {meaning_rep.strip()}\nREF:"
        text = f"{prompt} {reference.strip()}{eos_token}"
        return text

    def get_raw_sample(self, idx: int) -> Dict[str, str]:
        """Get raw sample for inspection."""
        sample = self.dataset[idx]
        return {
            "meaning_representation": sample["meaning_representation"],
            "human_reference": sample["human_reference"]
        }

    def get_grouped_data(self) -> Dict[str, List[str]]:
        """
        Group references by meaning representation.

        Returns:
            Dictionary mapping each unique MR to a list of all its references.
            NB: Needed for multi-reference BLEU evaluation.
        """
        from collections import defaultdict
        grouped = defaultdict(list)
        for sample in self.dataset:
            mr = sample["meaning_representation"]
            ref = sample["human_reference"]
            grouped[mr].append(ref)
        return dict(grouped)


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """Create a DataLoader for the dataset."""
    def collate_fn(batch):
        collated = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], torch.Tensor):
                collated[key] = torch.stack([item[key] for item in batch])
            else:
                collated[key] = [item[key] for item in batch]
        return collated

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )


def create_datasets_and_loaders(
    tokenizer: PreTrainedTokenizer,
    train_batch_size: int = 8,
    val_batch_size: int = 16,
    max_length: int = 128, # reduced from 256
    sample_percentage: float = 1.0,
    device: Optional[str] = None,
    add_eos_token: bool = True
) -> dict:
    """Create all datasets and dataloaders."""

    logger.info(f"Creating datasets with max_length={max_length}, add_eos_token={add_eos_token}")

    train_dataset = E2EDataset(
        split="train",
        tokenizer=tokenizer,
        max_length=max_length,
        sample_percentage=sample_percentage,
        device=device,
        add_eos_token=add_eos_token
    )

    val_dataset = E2EDataset(
        split="validation",
        tokenizer=tokenizer,
        max_length=max_length,
        sample_percentage=sample_percentage,
        device=device,
        add_eos_token=add_eos_token
    )

    test_dataset = E2EDataset(
        split="test",
        tokenizer=tokenizer,
        max_length=max_length,
        sample_percentage=sample_percentage,
        device=device,
        add_eos_token=add_eos_token
    )

    train_loader = get_dataloader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=val_batch_size, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=val_batch_size, shuffle=False)

    # Log some statistics
    logger.info("\n=== Dataset Statistics ===")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    # Check average length
    sample_lengths = []
    for i in range(min(100, len(train_dataset))):
        sample = train_dataset[i]
        length = (sample["attention_mask"] == 1).sum().item()
        sample_lengths.append(length)

    logger.info(f"Average token length (first 100 samples): {sum(sample_lengths)/len(sample_lengths):.1f}")
    logger.info(f"Max token length (first 100 samples): {max(sample_lengths)}")
    logger.info("=" * 50)

    return {
        "train": train_loader,
        "validation": val_loader,
        "test": test_loader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset
    }