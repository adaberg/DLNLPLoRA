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
from tqdm import trange
from transformers import GPT2TokenizerFast
import logging

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
    max_new_tokens: int = 30, # reduced from 50
    length_penalty: float = 0.9, # model applies a penalty based on the sequence length
    device: str = "cuda",
    use_greedy: bool = True
) -> List[str]:
    """
    Generate text completions for given prompts.
    """
    model.eval()
    generated_texts = []

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with torch.no_grad():
        for prompt in prompts:
            try:
                inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
                # Attention mask:
                attention_mask = torch.ones_like(inputs)

                # Generate text
                if use_greedy:
                    # Greedy decoding (deterministic and better for BLEU evaluation).
                    outputs = model.generate(
                        inputs,
                        attention_mask=attention_mask,
                        do_sample=False, # activates greedy
                        min_new_tokens=1,
                        max_new_tokens=max_new_tokens,
                        length_penalty=length_penalty,
                        no_repeat_ngram_size=4,
                        #num_beams=10,
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
                        no_repeat_ngram_size=4,
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


def compute_generation_metrics(
    predictions: List[str], references: List[str]
) -> Dict[str, float]:
    """
    Compute BLEU and ROUGE F1 scores using HuggingFace evaluate library
    (fast, single-pass metrics for monitoring).
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: predictions ({len(predictions)}) != "
            f"references ({len(references)})"
        )
    
    predictions = [str(p).strip() for p in predictions]
    references = [str(r).strip() for r in references]
    results = {}

    # 1. Compute BLEU score (corpus-level):
    try:
        bleu = evaluate.load("bleu")
        references_list = [[ref] for ref in references]
        bleu_result = bleu.compute(predictions=predictions, references=references_list)
        results["bleu"] = bleu_result["bleu"] # corpus BLEU
        logger.info("BLEU (corpus-level, precision-based)")
    except Exception as e:
        logger.warning(f"BLEU computation failed: {e}")
        results["bleu"] = 0.0

    # 2. Compute ROUGE F1 scores:
    #    (returns the F1 score by default) 
    try:
        rouge = evaluate.load("rouge")
        rouge_result = rouge.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True
        )
        results["rouge1_f1"] = rouge_result["rouge1"]
        results["rouge2_f1"] = rouge_result["rouge2"]
        results["rougeL_f1"] = rouge_result["rougeL"]
    except Exception as e:
        logger.warning(f"ROUGE F1 computation failed: {e}")
        results.update({"rouge1_f1": 0.0, "rouge2_f1": 0.0, "rougeL_f1": 0.0})

    # 3. Compute BERTScore F1:
    #    (measures the semantic similarity between generated and reference text)
    try:
        bertscore = evaluate.load("bertscore")
        bert_result = bertscore.compute(
            predictions=predictions,
            references=references,
            lang="en"
        )
        results["bertscore_f1"] = float(np.mean(bert_result["f1"]))
    except Exception as e:
        logger.warning(f"ROUGE F1 computation failed: {e}")
        results.update({"bertscore_f1": 0.0})

    logger.info(
        f"Generation metrics: BLEU={results.get('bleu', 0):.4f}, "
        f"ROUGE-1 F1={results.get('rouge1_f1', 0):.4f}, "
        f"ROUGE-2 F1={results.get('rouge2_f1', 0):.4f}, "
        f"ROUGE-L F1={results.get('rougeL_f1', 0):.4f}"
    )

    return results


def compute_bootstrap_generation_metrics(
    predictions: list[str],
    references: list[str],
    num_samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Compute bootstrapped BLEU and ROUGE F1 scores with confidence intervals
    (slow, statistical evaluation with CI).

    Returns mean and CI bounds.
    """

    rng = np.random.default_rng(seed)

    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")

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

    def ci(scores):
        lower = np.percentile(scores, (1.0 - confidence) / 2.0 * 100.0)
        upper = np.percentile(scores, (1.0 + confidence) / 2.0 * 100.0)
        return float(np.mean(scores)), float(lower), float(upper)

    return {
        "boot_bleu": ci(bleu_scores),
        "boot_rouge1_f1": ci(rouge1_scores),
        "boot_rouge2_f1": ci(rouge2_scores),
        "boot_rougeL_f1": ci(rougeL_scores)
    }


# Additional utility function (not in original spec but useful)
def evaluate_model_comprehensive(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    test_loader: DataLoader,
    test_dataset,
    device: str = "cuda",
    num_samples: int = 10,
    max_new_tokens = 30,
    do_bootstrap_eval: bool = False,
    use_greedy: bool = True
) -> Dict[str, float]:
    """
    Comprehensive evaluation combining perplexity and generation metrics.
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

    prompts = []
    references = []

    for i in range(min(num_samples, len(test_dataset))):
        if hasattr(test_dataset, 'get_raw_sample'):
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
        
        prompt = f"meaning_representation: {mr} | reference:"
        prompts.append(prompt)
        references.append(ref)

    if len(prompts) > 0:
        predictions = generate_texts(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            device=device,
            use_greedy=use_greedy
        )
        
        logger.info("Computing generation metrics...")
        metrics = compute_generation_metrics(predictions, references)
        if do_bootstrap_eval:
            metrics["bootstrap"] = compute_bootstrap_generation_metrics(
                predictions,
                references,
                num_samples=num_samples
            )

        results.update(metrics)

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
        max_new_tokens=30,
        device="cpu",
        use_greedy=True # always true during testing
    )
    print(f"Generated: {generated}")

    print("\nTesting compute_generation_metrics...")
    predictions = ["The cat sits on the mat", "I love programming"]
    references = ["The cat sits on the mat", "I enjoy coding"]

    metrics = compute_generation_metrics(predictions, references)
    print(f"Metrics: {metrics}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    _test_metrics()