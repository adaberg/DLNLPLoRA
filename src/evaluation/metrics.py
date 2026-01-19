"""
Evaluation metrics for LoRA project.
"""

from typing import List, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
import numpy as np
import evaluate
from transformers import GPT2TokenizerFast
import logging

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)


def compute_perplexity(model: nn.Module, dataloader: DataLoader, device: str) -> float:
    """
    Compute perplexity: exp(average_cross_entropy_loss)
<<<<<<< HEAD
        
    Returns:
        Perplexity score (float)
=======

    Args:
        model: Language model (should return loss in forward pass)
        dataloader: DataLoader with evaluation data
        device: Device to run computation on ("cuda" or "cpu")

    Returns:
        Perplexity score (float)

    Raises:
        ValueError: If no valid batches processed or model doesn't return loss
>>>>>>> 246de1b5342db25998f170be1257df5cb6290139
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
<<<<<<< HEAD
            batch = {k: v.to(device) for k, v in batch.items() 
                    if isinstance(v, torch.Tensor)}
            
=======
            # Move batch to device
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }

            # Forward pass - model should return loss
>>>>>>> 246de1b5342db25998f170be1257df5cb6290139
            outputs = model(**batch)

            if not hasattr(outputs, "loss") or outputs.loss is None:
                raise ValueError(
                    "Model must return loss in forward pass. "
                    "Ensure model is a language model with LM head."
                )

            loss = outputs.loss
<<<<<<< HEAD
            
=======

            # Accumulate loss (weighted by batch size)
>>>>>>> 246de1b5342db25998f170be1257df5cb6290139
            batch_size = batch["input_ids"].size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("No valid batches processed for perplexity computation")
<<<<<<< HEAD
    
=======

    # Compute average cross-entropy loss
>>>>>>> 246de1b5342db25998f170be1257df5cb6290139
    avg_loss = total_loss / total_samples

    # Perplexity = exp(average_loss)
    perplexity = np.exp(avg_loss)

    logger.info(
        f"Perplexity computation: avg_loss={avg_loss:.4f}, perplexity={perplexity:.2f}"
    )
    return float(perplexity)


def generate_texts(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    max_new_tokens: int = 50,
    device: str = "cuda",
) -> List[str]:
    """
    Generate text completions for given prompts.
<<<<<<< HEAD
    """
    model.eval()
    generated_texts = []
    
=======

    Args:
        model: Language model
        tokenizer: Tokenizer for encoding/decoding
        prompts: List of prompt strings
        max_new_tokens: Maximum number of tokens to generate
        device: Device to run generation on

    Returns:
        List of generated text strings (same length as prompts)
    """
    model.eval()
    generated_texts = []

    # Ensure tokenizer has pad token
>>>>>>> 246de1b5342db25998f170be1257df5cb6290139
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with torch.no_grad():
        for prompt in prompts:
            try:
                inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

                # Generate text
                outputs = model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=1,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
<<<<<<< HEAD
                
=======

                # Decode generated part (skip prompt)
>>>>>>> 246de1b5342db25998f170be1257df5cb6290139
                generated = tokenizer.decode(
                    outputs[0][len(inputs[0]) :], skip_special_tokens=True
                ).strip()

                generated_texts.append(generated)

            except Exception as e:
                logger.error(
                    f"Error generating text for prompt '{prompt[:50]}...': {e}"
                )
                generated_texts.append("")  # Return empty string on error

    return generated_texts


def compute_generation_metrics(
    predictions: List[str], references: List[str]
) -> Dict[str, float]:
    """
    Compute BLEU and ROUGE using HuggingFace evaluate library.
<<<<<<< HEAD
=======

    Args:
        predictions: List of generated/predicted texts
        references: List of reference/target texts

    Returns:
        Dictionary with metric names and scores:
        {
            "bleu": float,
            "rouge1": float,
            "rouge2": float,
            "rougeL": float
        }

    Raises:
        ValueError: If predictions and references have different lengths
>>>>>>> 246de1b5342db25998f170be1257df5cb6290139
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: predictions ({len(predictions)}) != "
            f"references ({len(references)})"
        )
<<<<<<< HEAD
    
=======

    # Clean inputs (remove extra whitespace)
>>>>>>> 246de1b5342db25998f170be1257df5cb6290139
    predictions = [str(p).strip() for p in predictions]
    references = [str(r).strip() for r in references]

    results = {}

    # 1. Compute BLEU
    try:
        bleu = evaluate.load("bleu")
        # BLEU expects list of references for each prediction
        references_list = [[ref] for ref in references]
        bleu_result = bleu.compute(predictions=predictions, references=references_list)
        results["bleu"] = bleu_result["bleu"]
    except Exception as e:
        logger.warning(f"BLEU computation failed: {e}")
        results["bleu"] = 0.0

    # 2. Compute ROUGE
    try:
        rouge = evaluate.load("rouge")
        rouge_result = rouge.compute(
            predictions=predictions, references=references, use_stemmer=True
        )
        # Extract ROUGE-1, ROUGE-2, ROUGE-L
        results["rouge1"] = rouge_result["rouge1"]
        results["rouge2"] = rouge_result["rouge2"]
        results["rougeL"] = rouge_result["rougeL"]
    except Exception as e:
        logger.warning(f"ROUGE computation failed: {e}")
        results.update({"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0})

    logger.info(
        f"Generation metrics: BLEU={results.get('bleu', 0):.4f}, "
        f"ROUGE-1={results.get('rouge1', 0):.4f}, "
        f"ROUGE-2={results.get('rouge2', 0):.4f}, "
        f"ROUGE-L={results.get('rougeL', 0):.4f}"
    )

    return results


# Additional utility function (not in original spec but useful)
def evaluate_model_comprehensive(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    test_loader: DataLoader,
    test_dataset,
    device: str = "cuda",
    num_samples: int = 10,
) -> Dict[str, float]:
    """
    Comprehensive evaluation combining perplexity and generation metrics.
<<<<<<< HEAD
=======

    Args:
        model: Trained language model
        tokenizer: Tokenizer
        test_loader: DataLoader for test set
        test_dataset: E2EDataset instance (to extract references)
        device: Device to run evaluation on
        num_samples: Number of samples for generation evaluation

    Returns:
        Dictionary with all evaluation metrics
>>>>>>> 246de1b5342db25998f170be1257df5cb6290139
    """
    results = {}

    # 1. Compute perplexity
    logger.info("Computing perplexity...")
    try:
        perplexity = compute_perplexity(model, test_loader, device)
        results["perplexity"] = perplexity
    except Exception as e:
        logger.error(f"Perplexity computation failed: {e}")
        results["perplexity"] = float("inf")

    # 2. Generate texts for evaluation
    logger.info(f"Generating texts for {num_samples} samples...")

    # Prepare prompts and references
    prompts = []
    references = []

    for i in range(min(num_samples, len(test_dataset))):
<<<<<<< HEAD
        if hasattr(test_dataset, 'get_raw_sample'):
=======
        # Get raw sample from dataset
        # Assuming test_dataset has get_raw_sample method from dataset.py
        if hasattr(test_dataset, "get_raw_sample"):
>>>>>>> 246de1b5342db25998f170be1257df5cb6290139
            raw = test_dataset.get_raw_sample(i)
            mr = raw.get("meaning_representation", "")
            ref = raw.get("human_reference", "")
        else:
            try:
                mr = test_dataset.dataset[i]["meaning_representation"]
                ref = test_dataset.dataset[i]["human_reference"]
            except:
                logger.warning(f"Could not extract sample {i}, skipping")
                continue
<<<<<<< HEAD
        
=======

        # Create prompt: only the meaning representation part
>>>>>>> 246de1b5342db25998f170be1257df5cb6290139
        prompt = f"meaning_representation: {mr} | reference:"
        prompts.append(prompt)
        references.append(ref)

    if len(prompts) > 0:
        predictions = generate_texts(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=50,
            device=device,
        )
<<<<<<< HEAD
        
        logger.info("Computing generation metrics...")
        gen_metrics = compute_generation_metrics(predictions, references)
        results.update(gen_metrics)
        
=======

        # 3. Compute generation metrics
        logger.info("Computing generation metrics...")
        gen_metrics = compute_generation_metrics(predictions, references)
        results.update(gen_metrics)

        # Add example outputs for debugging
>>>>>>> 246de1b5342db25998f170be1257df5cb6290139
        results["_examples"] = []
        for i in range(min(3, len(prompts))):
            results["_examples"].append(
                {
                    "prompt": (
                        prompts[i][:100] + "..."
                        if len(prompts[i]) > 100
                        else prompts[i]
                    ),
                    "prediction": predictions[i],
                    "reference": references[i],
                }
            )
    else:
        logger.warning("No valid prompts extracted for generation evaluation")

    return results


# Test function to verify implementation
def _test_metrics():
    """Internal test function to verify metrics work correctly."""
    print("Testing evaluation metrics...")
<<<<<<< HEAD
    
=======

    # Mock a simple model for testing
>>>>>>> 246de1b5342db25998f170be1257df5cb6290139
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.randn(1))

        def forward(self, input_ids, attention_mask=None, labels=None):
            batch_size = input_ids.shape[0]
            loss = torch.tensor(1.0, requires_grad=True)

            class MockOutput:
                def __init__(self, loss):
                    self.loss = loss

            return MockOutput(loss)

        def generate(self, input_ids, **kwargs):
            return torch.cat([input_ids, input_ids], dim=-1)
<<<<<<< HEAD
    
    from transformers import GPT2TokenizerFast
    
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
=======

    # Create tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Test generate_texts
>>>>>>> 246de1b5342db25998f170be1257df5cb6290139
    prompts = ["Hello world", "Test prompt"]
    model = MockModel()

    print("Testing generate_texts...")
    generated = generate_texts(model, tokenizer, prompts, device="cpu")
    print(f"Generated: {generated}")
<<<<<<< HEAD
    
=======

    # Test compute_generation_metrics
>>>>>>> 246de1b5342db25998f170be1257df5cb6290139
    print("\nTesting compute_generation_metrics...")
    predictions = ["The cat sits on the mat", "I love programming"]
    references = ["The cat sits on the mat", "I enjoy coding"]

    metrics = compute_generation_metrics(predictions, references)
    print(f"Metrics: {metrics}")

    print("\nAll tests passed!")


if __name__ == "__main__":
<<<<<<< HEAD
    import logging
    logging.basicConfig(level=logging.INFO)
    _test_metrics()
=======
    _test_metrics()
>>>>>>> 246de1b5342db25998f170be1257df5cb6290139
