"""
E2E NLG Challenge dataset loader.
Format: "meaning_representation: <mr> | reference: <ref>"
"""

from typing import Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizer
import logging
import os

logger = logging.getLogger(__name__)


class E2EDataset(Dataset):
    """E2E NLG dataset for language modeling."""
    
    def __init__(
        self,
        split: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,
        sample_percentage: float = 1.0,
        device: Optional[str] = None
    ) -> None:
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        
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
        
        # Load from local CSV
        dataset = load_dataset("csv", data_files={split: csv_path})
        dataset = dataset[split]
        
        # Rename columns to match our format
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
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.dataset[idx]
        meaning_rep = sample["meaning_representation"]
        reference = sample["human_reference"]
        
        # Format the text
        text = f"meaning_representation: {meaning_rep} | reference: {reference}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0).clone()
        }
        
        if self.device:
            item = {k: v.to(self.device) for k, v in item.items()}
        
        return item
    
    @staticmethod
    def format_sample(meaning_rep: str, reference: str) -> str:
        """Format a single sample for language modeling."""
        return f"meaning_representation: {meaning_rep.strip()} | reference: {reference.strip()}"
    
    def get_raw_sample(self, idx: int) -> Dict[str, str]:
        """Get raw sample for inspection."""
        sample = self.dataset[idx]
        return {
            "meaning_representation": sample["meaning_representation"],
            "human_reference": sample["human_reference"]
        }


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
    max_length: int = 256,
    sample_percentage: float = 1.0,
    device: Optional[str] = None
) -> dict:
    """Create all datasets and dataloaders."""
    train_dataset = E2EDataset(
        split="train",
        tokenizer=tokenizer,
        max_length=max_length,
        sample_percentage=sample_percentage,
        device=device
    )
    
    val_dataset = E2EDataset(
        split="validation",
        tokenizer=tokenizer,
        max_length=max_length,
        sample_percentage=sample_percentage,
        device=device
    )
    
    test_dataset = E2EDataset(
        split="test",
        tokenizer=tokenizer,
        max_length=max_length,
        sample_percentage=sample_percentage,
        device=device
    )
    
    train_loader = get_dataloader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=val_batch_size, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=val_batch_size, shuffle=False)
    
    return {
        "train": train_loader,
        "validation": val_loader,
        "test": test_loader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset
    }


def test_dataset():
    """Quick test function."""
    from transformers import GPT2TokenizerFast
    
    print("Testing E2EDataset...")
    
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = E2EDataset(
        split="train",
        tokenizer=tokenizer,
        max_length=128,
        sample_percentage=0.01
    )
    
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    
    raw = dataset.get_raw_sample(0)
    print(f"\nRaw sample: {raw['meaning_representation'][:50]}...")
    
    # 新增详细检查
    print(f"\n📊 Detailed check:")
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Actual length (non-padding): {(sample['attention_mask'] == 1).sum().item()}")
    
    # 解码查看
    decoded = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
    print(f"\nDecoded (first 100 chars): {decoded[:100]}")
    
    print("\n✅ Test passed!")


if __name__ == "__main__":
    test_dataset()