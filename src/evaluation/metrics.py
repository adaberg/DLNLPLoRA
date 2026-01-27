"""
Evaluation metrics for LoRA project.
"""

from typing import List, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
import numpy as np
import evaluate
from tqdm import trange
from bert_score import score as bertscore
from transformers import GPT2TokenizerFast
import logging
from absl import logging as absl_logging
from transformers import logging as hf_logging

# Suppresses the annoying info messages
# from the bertscore function:
absl_logging.set_verbosity(absl_logging.ERROR)
hf_logging.set_verbosity_error()

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)


def compute_perplexity_batch_weighted(model: nn.Module, dataloader: DataLoader, device: str) -> float:
    """
    Compute perplexity with batch-weighted aggregation: exp(average_cross_entropy_loss)

    Returns:
        Perplexity score (float)
    """
    # Note: This method is poor for the E2E dataset because batch lengths
    #       vary significantly. This means sequences have different lengths,
    #       the padding skews the result, and long sequences are underweighted.
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()
                    if isinstance(v, torch.Tensor)}

            outputs = model(**batch)
            # Note: The standard HuggingFace GPT-2 model does not support the "loss_type" parameter.
            #       This will likely lead to an error or incorrect loss calculation.

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

    avg_batch_loss = total_loss / total_samples
    perplexity = np.exp(avg_batch_loss)

    logger.info(
        f"Perplexity computation (batch-weighted): avg_batch_loss={avg_batch_loss:.4f}, perplexity={perplexity:.2f}"
    )
    return float(perplexity)


def compute_perplexity_token_weighted(model: nn.Module, dataloader: DataLoader, device: str) -> float:
    """
    Compute perplexity with token-weighted aggregation: exp(average_cross_entropy_loss)

    Returns:
        Perplexity score (float)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()
                    if isinstance(v, torch.Tensor)}

            outputs = model(**batch)
            # Note: The standard HuggingFace GPT-2 model does not support the "loss_type" parameter.
            #       This will likely lead to an error or incorrect loss calculation.

            if not hasattr(outputs, "loss") or outputs.loss is None:
                raise ValueError(
                    "Model must return loss in forward pass. "
                    "Ensure model is a language model with LM head."
                )

            loss = outputs.loss
            num_tokens = (batch["labels"] != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    if total_tokens == 0:
        raise ValueError("No valid tokens processed for perplexity computation")

    avg_token_loss = total_loss / total_tokens
    perplexity = np.exp(avg_token_loss)

    logger.info(
        f"Perplexity computation (token-weighted): avg_token_loss={avg_token_loss:.4f}, perplexity={perplexity:.2f}"
    )
    return float(perplexity)


def generate_texts(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    max_new_tokens: int = 64,  # increased from 50
    length_penalty: float = 0.9,  # model applies a penalty based on the sequence length
    no_repeat_ngram_size: int = 4,
    num_beams: int = 10,
    use_beam_search: bool = True,
    use_greedy: bool = True,
    device: str = "cuda"
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
        use_beam_search: If True, the model uses beam search if use_greedy is also
                         set to True. If 'use_greedy' is set to False, the model
                         uses sampling instead of beam search (default: True).
        use_greedy: If True, the model generates a deterministic output (greedy decoding);
                    otherwise, the model uses non-deterministic sampling (default: True).

    Returns:
        List of generated text completions
    """
    model.eval()
    outputs = None
    generated_texts = []

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with torch.no_grad():
        for prompt in prompts:
            try:
                inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

                # Attention mask for the inputs:
                attention_mask = torch.ones_like(inputs)

                # Generate text:
                if use_beam_search:
                    # Generate text using beam search (paper-specified parameters)
                    # When greedy decoding is enabled, the model is deterministic,
                    # which is better for BLEU evaluation.
                    outputs = model.generate(
                        inputs,
                        attention_mask=attention_mask,
                        do_sample=False if use_greedy else True,
                        early_stopping=True,
                        min_new_tokens=1,
                        max_new_tokens=max_new_tokens,
                        length_penalty=length_penalty,
                        no_repeat_ngram_size=no_repeat_ngram_size,
                        # paper states reuse of params of https://arxiv.org/pdf/2101.00190 (beam size = 5)
                        num_beams=num_beams,
                        num_return_sequences = 1,  # to avoid unnecessary internal competition between the beams
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                else:
                    # Non-deterministic sampling (nucleus sampling for diversity),
                    # not recommended for BLEU (BLEU/ROUGE fluctuate from run to run).
                    outputs = model.generate(
                        inputs,
                        attention_mask=attention_mask,
                        do_sample=True,
                        min_new_tokens=1,
                        max_new_tokens=max_new_tokens,
                        length_penalty=length_penalty,
                        no_repeat_ngram_size=no_repeat_ngram_size,
                        temperature=0.8,
                        top_p=0.9,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
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


def compute_rouge_multi_ref(
    predictions: List[str], references: List[List[str]]
) -> Dict[str, float]:
    """
    Computes ROUGE-1, ROUGE-2, and ROUGE-L F1 scores by evaluating each prediction
    against all references and selecting the maximum score per example.
    """
    try:
        rouge = evaluate.load("rouge")
    except Exception as e:
        logger.warning(f"ROUGE F1 calculation failed: {e}")
        return {"rouge1_f1": 0.0, "rouge2_f1": 0.0, "rougeL_f1": 0.0}

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    # Select the best ROUGE values ​​from each reference example:
    for pred, refs in zip(predictions, references):
        # one prediction and N references
        best_r1 = 0.0
        best_r2 = 0.0
        best_rL = 0.0

        for ref in refs:
            scores = rouge.compute(
                predictions=[pred],
                references=[ref],
                use_stemmer=True
            )
            best_r1 = max(best_r1, scores["rouge1"])
            best_r2 = max(best_r2, scores["rouge2"])
            best_rL = max(best_rL, scores["rougeL"])

        rouge1_scores.append(best_r1)
        rouge2_scores.append(best_r2)
        rougeL_scores.append(best_rL)

    # Calculate and return the arithmetic mean of the
    # ROUGE F1 scores across all examples:
    return {
        "rouge1_f1": sum(rouge1_scores) / len(rouge1_scores),
        "rouge2_f1": sum(rouge2_scores) / len(rouge2_scores),
        "rougeL_f1": sum(rougeL_scores) / len(rougeL_scores),
    }


def compute_bertscore_multi_ref(
    predictions: List[str], references: List[List[str]], lang: str = "en"
) -> Dict[str, float]:
    """
    Computes BERTScore F1 by scoring each prediction against all references
    and taking the maximum score per example.
    """
    best_scores = []

    for pred, refs in zip(predictions, references):
        # one prediction and N references
        P, R, F1 = bertscore(
            [pred] * len(refs),  # duplicate by the number of references
            refs,
            lang=lang,
            verbose=False
        )
        # Select the best matching reference:
        best_scores.append(F1.max().item())

    # Calculate and return the arithmetic mean of the
    # BERTScore F1 values across all examples:
    return {
        "bertscore_f1": sum(best_scores) / len(best_scores)
    }


def compute_generation_metrics(
    predictions: List[str],
    references: List[List[str]],
    num_samples: int | None = None,
    do_bootstrap_eval: bool = False
) -> Dict[str, float]:
    """
    Compute BLEU and ROUGE F1 scores using HuggingFace evaluate library
    (fast, single-pass metrics for monitoring).

    Args:
        predictions: List of generated texts (one per unique MR)
        references: List of reference lists (multiple references per MR)
                   Format: [[ref1_mr1, ref2_mr1, ...], [ref1_mr2, ref2_mr2, ...], ...]

    Returns:
        Dictionary with BLEU and ROUGE F1 scores
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: predictions ({len(predictions)}) != "
            f"references ({len(references)})"
        )

    predictions = [str(p).strip() for p in predictions]
    # Clean up references (each is a list of strings):
    references = [[str(r).strip() for r in ref_list] for ref_list in references]
    results = {}

    # 1. Compute BLEU score with multiple references (corpus-level):
    try:
        bleu = evaluate.load("bleu")
        # 'references' is already in the correct format: List[List[str]]
        bleu_result = bleu.compute(predictions=predictions, references=references)
        results["bleu"] = bleu_result["bleu"]  # corpus BLEU
        logger.info("BLEU (corpus-level, precision-based)")
    except Exception as e:
        logger.warning(f"BLEU calculation failed: {e}")
        results["bleu"] = 0.0

    # 2. Compute ROUGE F1 scores (mean of the best scores):
    #    Note: ROUGE F1 does not natively support multiple references well, therefore the
    #          ROUGE values ​​must be calculated as the arithmetic mean across all references.
    rouge_scores = compute_rouge_multi_ref(predictions, references)
    results.update(rouge_scores)

    # 3. Compute BERTScore F1 (mean of the best matching scores):
    #    (Measures the semantic similarity between generated and reference text.)
    #    Note: In the case of multiple references, as with the ROUGE values, the
    #          BertScore must be calculated as the arithmetic mean across all references.
    bert_score = compute_bertscore_multi_ref(predictions, references)
    results.update(bert_score)

    logger.info(
        f"Generation metrics: BLEU={results.get('bleu', 0.0):.4f}, "
        f"ROUGE-1 F1={results.get('rouge1_f1', 0.0):.4f}, "
        f"ROUGE-2 F1={results.get('rouge2_f1', 0.0):.4f}, "
        f"ROUGE-L F1={results.get('rougeL_f1', 0.0):.4f}, "
        f"BERTScore F1={results.get('bertscore_f1', 0.0):.4f}, "
    )

    if num_samples and do_bootstrap_eval:
        boot_result = compute_bootstrap_generation_metrics(
            predictions=predictions,
            references=references,
            num_samples=num_samples
        )
        results["bootstrap"] = boot_result

    return results


def compute_bootstrap_generation_metrics(
    predictions: List[str],
    references: List[str],
    num_samples: int,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Compute bootstrapped BLEU and ROUGE F1 scores with confidence intervals
    (slow, statistical evaluation with CI).

    Returns mean and CI bounds.
    """
    rng = np.random.default_rng(seed)

    try:
        bleu = evaluate.load("bleu")
        rouge = evaluate.load("rouge")
    except Exception as e:
        logger.warning(f"Bootstrap metrics calculation failed: {e}")
        return {}

    bleu_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    n = len(predictions)
    assert n == len(references)

    for _ in trange(num_samples, desc="Bootstrapping metrics"):
        idx = rng.integers(0, n, size=n)

        preds_sample = [predictions[i] for i in idx]
        refs_sample = [references[i] for i in idx]

        bleu_result = bleu.compute(
            predictions=preds_sample,
            references=[[r] for r in refs_sample]
        )
        rouge_result = rouge.compute(
            predictions=preds_sample,
            references=refs_sample,
            use_stemmer=True
        )

        bleu_scores.append(bleu_result["bleu"])
        rouge1_scores.append(rouge_result["rouge1"])
        rouge2_scores.append(rouge_result["rouge2"])
        rougeL_scores.append(rouge_result["rougeL"])

    # Calculation of the confidence intervalls:
    def ci(scores):
        lower = np.percentile(scores, (1.0 - confidence) / 2.0 * 100.0)
        upper = np.percentile(scores, (1.0 + confidence) / 2.0 * 100.0)
        return {
            "mean": float(np.mean(scores)),
            "lower": float(lower),
            "upper": float(upper)
        }

    return {
        "boot_bleu": ci(bleu_scores),
        "boot_rouge1_f1": ci(rouge1_scores),
        "boot_rouge2_f1": ci(rouge2_scores),
        "boot_rougeL_f1": ci(rougeL_scores)
    }


def evaluate_model_comprehensive(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    test_loader: DataLoader,
    test_dataset,
    device: str = "cuda",
    num_samples: int = -1,
    max_new_tokens = 64,
    generation_config: Dict[str, Any] | None = None,
    do_bootstrap_eval: bool = False
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
            - max_new_tokens (default: 64)
            - num_beams (default: 10, from LoRA paper)
            - length_penalty (default: 0.9, from LoRA paper for E2E)
            - no_repeat_ngram_size (default: 4, from LoRA paper)
            - use_beam_search (default: True)
            - use_greedy (default: True)
        do_bootstrap_eval: Compute bootstrapped BLEU and ROUGE F1 with
                           confidence intervals
    """
    if generation_config is None:
        generation_config = {}

    # Default generation parameters from LoRA paper (Table 11 / Section D.3)
    max_new_tokens = generation_config.get("max_new_tokens", 64)
    num_beams = generation_config.get("num_beams", generation_config.get("beam_size", 10))
    length_penalty = generation_config.get("length_penalty", 0.9)
    no_repeat_ngram_size = generation_config.get("no_repeat_ngram_size", 4)
    use_beam_search = generation_config.get("use_beam_search", True)
    use_greedy = generation_config.get("use_greedy", True)

    results = {}

    # 1. Compute perplexity
    logger.info("Computing perplexity...")
    try:
        #perplexity = compute_perplexity_batch_weighted(model, test_loader, device)
        perplexity = compute_perplexity_token_weighted(model, test_loader, device)  # paper-conform
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
    logger.info(
        f"Generation config: num_beams={num_beams}, length_penalty={length_penalty}, no_repeat_ngram_size={no_repeat_ngram_size}, "
        f"use_beam_search={use_beam_search}, use_greedy={use_greedy}"
    )

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
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            num_beams=num_beams,
            use_beam_search=use_beam_search,
            use_greedy=use_greedy,
            device=device
        )

        logger.info("Computing generation metrics with multi-reference BLEU...")
        gen_metrics = compute_generation_metrics(
            predictions=predictions,
            references=all_references,
            num_samples=num_samples,
            do_bootstrap_eval=do_bootstrap_eval
        )
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
    generated = generate_texts(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=64,
        use_beam_search=True,
        use_greedy=True,  # always true during testing
        device="cpu"
    )
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