"""
Unit tests for data loading module.
"""

import pytest
import torch
import os
import sys
from transformers import GPT2TokenizerFast

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.insert(0, os.path.abspath(project_root))

from src.data.dataset import E2EDataset, get_dataloader


def test_format_sample():
    """Test E2E format function produces expected string."""
    mr = "name[The Eagle], eatType[coffee shop]"
    ref = "The Eagle is a coffee shop."
    
    formatted = E2EDataset.format_sample(mr, ref)
    expected = "meaning_representation: name[The Eagle], eatType[coffee shop] | reference: The Eagle is a coffee shop."
    
    assert formatted == expected, f"Expected: {expected}, Got: {formatted}"
    print("format_sample test passed")


def test_e2e_dataset_loading():
    """Test E2E dataset loads from local CSV files."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    e2e_dir = os.path.join(current_dir, "..", "src", "data", "e2e_data")
    train_csv = os.path.join(e2e_dir, "train.csv")
    
    if not os.path.exists(train_csv):
        pytest.skip("E2E data files not found. Run download_data.py first.")
    
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = E2EDataset(
        split="train",
        tokenizer=tokenizer,
        max_length=128,
        sample_percentage=0.01
    )
    
    assert len(dataset) > 0, "Dataset should not be empty"
    print(f"✅ Dataset loaded with {len(dataset)} samples")


def test_dataset_length():
    """Test sample_percentage correctly reduces dataset size."""
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    full_dataset = E2EDataset(
        split="validation",
        tokenizer=tokenizer,
        max_length=128,
        sample_percentage=1.0
    )
    full_size = len(full_dataset)
    
    partial_dataset = E2EDataset(
        split="validation",
        tokenizer=tokenizer,
        max_length=128,
        sample_percentage=0.1
    )
    partial_size = len(partial_dataset)
    
    expected_size = int(full_size * 0.1)
    tolerance = max(1, expected_size * 0.1)
    
    assert abs(partial_size - expected_size) <= tolerance, \
        f"Expected ~{expected_size} samples, got {partial_size}"
    
    print(f"Dataset sampling: full={full_size}, 10%={partial_size}")


def test_tokenization_length():
    """Verify tokenized sequences don't exceed max_length."""
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    max_length = 64
    dataset = E2EDataset(
        split="test",
        tokenizer=tokenizer,
        max_length=max_length,
        sample_percentage=0.01
    )
    
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        
        # Check shapes
        assert sample["input_ids"].shape == (max_length,), \
            f"Input IDs shape mismatch: {sample['input_ids'].shape}"
        assert sample["attention_mask"].shape == (max_length,), \
            f"Attention mask shape mismatch: {sample['attention_mask'].shape}"
        assert sample["labels"].shape == (max_length,), \
            f"Labels shape mismatch: {sample['labels'].shape}"
        
        # Check actual length doesn't exceed max_length
        actual_length = sample["attention_mask"].sum().item()
        assert actual_length <= max_length, \
            f"Actual length {actual_length} > max_length {max_length}"
    
    print("Tokenization length test passed")


def test_dataloader():
    """Test DataLoader creation and batching."""
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = E2EDataset(
        split="train",
        tokenizer=tokenizer,
        max_length=128,
        sample_percentage=0.01
    )
    
    batch_size = 4
    dataloader = get_dataloader(dataset, batch_size=batch_size, shuffle=False)
    
    first_batch = next(iter(dataloader))
    
    assert first_batch["input_ids"].shape == (batch_size, 128), \
        f"Batch input IDs shape: {first_batch['input_ids'].shape}"
    assert first_batch["attention_mask"].shape == (batch_size, 128), \
        f"Batch attention mask shape: {first_batch['attention_mask'].shape}"
    assert first_batch["labels"].shape == (batch_size, 128), \
        f"Batch labels shape: {first_batch['labels'].shape}"
    
    print("DataLoader test passed")


def test_raw_sample():
    """Test raw sample extraction."""
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = E2EDataset(
        split="train",
        tokenizer=tokenizer,
        max_length=128,
        sample_percentage=0.01
    )
    
    raw = dataset.get_raw_sample(0)
    
    assert "meaning_representation" in raw, "Missing meaning_representation"
    assert "human_reference" in raw, "Missing human_reference"
    assert isinstance(raw["meaning_representation"], str), \
        "meaning_representation should be string"
    assert isinstance(raw["human_reference"], str), \
        "human_reference should be string"
    assert len(raw["meaning_representation"]) > 0, \
        "meaning_representation should not be empty"
    assert len(raw["human_reference"]) > 0, \
        "human_reference should not be empty"
    
    print("Raw sample test passed")


def run_all_tests():
    """Run all data tests."""
    print("=" * 60)
    print("Running all data module tests...")
    print("=" * 60)
    
    tests = [
        test_format_sample,
        test_e2e_dataset_loading,
        test_dataset_length,
        test_tokenization_length,
        test_raw_sample,
        test_dataloader,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"{test_func.__name__} failed: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"Test summary: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
