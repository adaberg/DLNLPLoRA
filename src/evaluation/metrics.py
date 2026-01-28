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
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)


def compute_perplexity(model: nn.Module, dataloader: DataLoader, device: str) -> float:
    """
    Compute perplexity: exp(average_cross_entropy_loss)

    Returns:
        Perplexity score (float)
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()
                    if isinstance(v, torch.Tensor)}

            # Modifiction
            outputs = model(**batch, loss_type="ForCausalLMLoss")

            if not hasattr(outputs, "loss") or outputs.loss is None:
                raise ValueError(
                    "Model must return loss in forward pass. "
                    "Ensure model is a language model with LM head."
                )

            loss = outputs.loss

            batch_size = batch["input_ids"].size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("No valid batches processed for perplexity computation")

    avg_loss = total_loss / total_samples
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
    num_beams: int = 10,
    length_penalty: float = 0.9,
    no_repeat_ngram_size: int = 4,
    do_sample: bool = False,
) -> List[str]:
    """
    Generate text completions for given prompts.

    Uses beam search by default (as specified in LoRA paper Table 11/Section D.3):
    - num_beams: 10
    - length_penalty: 0.9 (for E2E)
    - no_repeat_ngram_size: 4

    Args:
        model: The language model
        tokenizer: Tokenizer for encoding/decoding
        prompts: List of input prompts
        max_new_tokens: Maximum tokens to generate
        device: Device to run on
        num_beams: Number of beams for beam search (paper: 10)
        length_penalty: Length penalty for beam search (paper: 0.9 for E2E)
        no_repeat_ngram_size: Prevent n-gram repetition (paper: 4)
        do_sample: Use sampling instead of beam search (default: False for deterministic output)

    Returns:
        List of generated text completions
    """
    model.eval()
    generated_texts = []

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with torch.no_grad():
        for prompt in prompts:
            try:
                inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
                attention_mask = torch.ones_like(inputs)

                # Generate text using beam search (paper-specified parameters)
                outputs = model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=1,
                    # paper states reuse of params of https://arxiv.org/pdf/2101.00190 (beam size = 5)
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    do_sample=do_sample,
                    early_stopping=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

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
    predictions: List[str], references: List[List[str]]
) -> Dict[str, float]:
    """
    Compute BLEU and ROUGE metrics for E2E NLG evaluation.

    Uses NLTK corpus_bleu with smoothing to match the E2E NLG benchmark methodology.
    This is important because SacreBLEU (HuggingFace evaluate) gives significantly
    lower scores than NLTK with smoothing for the same predictions.

    Args:
        predictions: List of generated texts (one per unique MR)
        references: List of reference lists (multiple references per MR)
                   Format: [[ref1_mr1, ref2_mr1, ...], [ref1_mr2, ref2_mr2, ...], ...]

    Returns:
        Dictionary with BLEU and ROUGE scores
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: predictions ({len(predictions)}) != "
            f"references ({len(references)})"
        )

    predictions = [str(p).strip() for p in predictions]
    # Clean up references (each is a list of strings)
    references = [[str(r).strip() for r in ref_list] for ref_list in references]

    results = {}

    # 1. Compute BLEU with multiple references using NLTK (E2E benchmark standard)
    # NLTK corpus_bleu with smoothing matches the E2E NLG Challenge evaluation
    try:
        # Tokenize predictions and references (lowercase + split on whitespace)
        pred_tokens = [p.lower().split() for p in predictions]
        # NLTK expects references as: [[[ref1_tokens], [ref2_tokens], ...], ...]
        ref_tokens = [[r.lower().split() for r in ref_list] for ref_list in references]

        # Use smoothing method7 which is commonly used in NLG evaluation
        # This handles cases where higher-order n-grams have zero matches
        smoother = SmoothingFunction()
        bleu_score = corpus_bleu(
            ref_tokens,
            pred_tokens,
            smoothing_function=smoother.method7
        )
        results["bleu"] = bleu_score
    except Exception as e:
        logger.warning(f"BLEU computation failed: {e}")
        results["bleu"] = 0.0

    # 2. Compute ROUGE (use first reference from each list for ROUGE)
    # ROUGE doesn't natively support multiple references well
    try:
        rouge = evaluate.load("rouge")
        # Use first reference for ROUGE computation
        first_references = [ref_list[0] for ref_list in references]
        rouge_result = rouge.compute(
            predictions=predictions, references=first_references, use_stemmer=True
        )
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


def evaluate_model_comprehensive(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    test_loader: DataLoader,
    test_dataset,
    device: str = "cuda",
    num_samples: int = -1,
    generation_config: Dict = None,
) -> Dict[str, float]:
    """
    Comprehensive evaluation combining perplexity and generation metrics.

    E2E NLG evaluation methodology:
    - Test samples are grouped by unique meaning representation (MR)
    - Generate one prediction per unique MR
    - Compute multi-reference BLEU
        (validate output against ALL references for each MR)

    Args:
        model: The language model to evaluate
        tokenizer: Tokenizer for encoding/decoding
        test_loader: DataLoader for perplexity computation
        test_dataset: Dataset with get_grouped_data() method
        device: Device to run on
        num_samples: Number of unique MRs to evaluate (-1 for all)
        generation_config: Dict with generation parameters:
            - max_new_tokens (default: 50)
            - num_beams (default: 10, from LoRA paper)
            - length_penalty (default: 0.9, from LoRA paper for E2E)
            - no_repeat_ngram_size (default: 4, from LoRA paper)
            - do_sample (default: False for beam search)
    """
    if generation_config is None:
        generation_config = {}

    # Default generation parameters from LoRA paper (Table 11 / Section D.3)
    max_new_tokens = generation_config.get("max_new_tokens", 50)
    num_beams = generation_config.get("num_beams", generation_config.get("beam_size", 10))
    length_penalty = generation_config.get("length_penalty", 0.9)
    no_repeat_ngram_size = generation_config.get("no_repeat_ngram_size", 4)
    do_sample = generation_config.get("do_sample", False)

    results = {}

    # 1. Compute perplexity
    logger.info("Computing perplexity...")
    try:
        perplexity = compute_perplexity(model, test_loader, device)
        results["perplexity"] = perplexity
    except Exception as e:
        logger.error(f"Perplexity computation failed: {e}")
        results["perplexity"] = float("inf")

    # 2. Get grouped data (unique MRs with all their references)
    if not hasattr(test_dataset, 'get_grouped_data'):
        logger.error("test_dataset must have get_grouped_data() method for proper E2E evaluation")
        return results

    grouped_data = test_dataset.get_grouped_data()
    total_unique_mrs = len(grouped_data)

    # Determine how many MRs to evaluate
    if num_samples == -1 or num_samples > total_unique_mrs:
        num_samples = total_unique_mrs

    logger.info(f"E2E Evaluation: {num_samples} unique MRs (out of {total_unique_mrs} total)")
    logger.info(f"Total test samples: {len(test_dataset)}, Average refs per MR: {len(test_dataset)/total_unique_mrs:.1f}")
    logger.info(f"Generation config: num_beams={num_beams}, length_penalty={length_penalty}, no_repeat_ngram_size={no_repeat_ngram_size}, do_sample={do_sample}")

    # 3. Generate texts for evaluation (ONE per unique MR)
    prompts = []
    all_references = []  # List[List[str]] - multiple refs per MR
    mr_list = []  # Keep track of MRs for examples

    for i, (mr, refs) in enumerate(grouped_data.items()):
        if i >= num_samples:
            break
        prompt = f"meaning_representation: {mr} | reference:"
        prompts.append(prompt)
        all_references.append(refs)  # All references for this MR
        mr_list.append(mr)

    if len(prompts) > 0:
        logger.info(f"Generating {len(prompts)} predictions (one per unique MR)...")
        predictions = generate_texts(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            device=device,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            do_sample=do_sample,
        )

        logger.info("Computing generation metrics with multi-reference BLEU...")
        gen_metrics = compute_generation_metrics(predictions, all_references)
        results.update(gen_metrics)

        # Store some examples for inspection
        results["_examples"] = []
        for i in range(min(3, len(prompts))):
            results["_examples"].append(
                {
                    "mr": mr_list[i][:100] + "..." if len(mr_list[i]) > 100 else mr_list[i],
                    "prediction": predictions[i],
                    "num_references": len(all_references[i]),
                    "sample_reference": all_references[i][0],  # Show first reference
                }
            )

        # Add evaluation metadata
        results["_eval_info"] = {
            "unique_mrs_evaluated": len(prompts),
            "total_unique_mrs": total_unique_mrs,
            "total_test_samples": len(test_dataset),
            "avg_refs_per_mr": len(test_dataset) / total_unique_mrs,
        }
    else:
        logger.warning("No valid prompts extracted for generation evaluation")

    return results


# Test function to verify implementation
def _test_metrics():
    """Internal test function to verify metrics work correctly."""
    print("Testing evaluation metrics...")

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

    from transformers import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    prompts = ["Hello world", "Test prompt"]
    model = MockModel()

    print("Testing generate_texts...")
    generated = generate_texts(model, tokenizer, prompts, device="cpu")
    print(f"Generated: {generated}")

    print("\nTesting compute_generation_metrics with multi-reference BLEU...")
    predictions = ["The cat sits on the mat", "I love programming"]
    # NB: Multi-reference format: List[List[str]]
    references = [
        ["The cat sits on the mat", "A cat is sitting on the mat", "The cat sat on the mat"],
        ["I enjoy coding", "I love programming", "Programming is fun"]
    ]

    metrics = compute_generation_metrics(predictions, references)
    print(f"Metrics: {metrics}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    _test_metrics()