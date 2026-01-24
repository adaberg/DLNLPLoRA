## Evaluation Module
This module provides evaluation metrics for the LoRA reproduction project.

## Overview
The evaluation module includes functions to compute perplexity, generate text, and calculate NLG metrics (BLEU, ROUGE) for language models.

## Available Functions
Core Metrics
```python
from src.evaluation import (
    compute_perplexity,
    generate_texts,
    compute_generation_metrics,
    evaluate_model_comprehensive
)
```
#1. compute_perplexity(model, dataloader, device)
Compute perplexity: exp(average_cross_entropy_loss)

#Usage:

```python
perplexity = compute_perplexity(model, test_loader, device="cuda")
# Returns: float (lower is better)
```

#2. generate_texts(model, tokenizer, prompts, max_new_tokens=50, device="cuda")
Generate text completions for given prompts.
```
#Usage:

```python
prompts = ["meaning_representation: name[The Eagle] | reference:"]
generated = generate_texts(model, tokenizer, prompts, max_new_tokens=50)
# Returns: List[str] of generated texts
```

#3. compute_generation_metrics(predictions, references)
Compute BLEU and ROUGE scores using HuggingFace evaluate library.

#Usage:

```python
metrics = compute_generation_metrics(predictions, references)
# Returns: {"bleu": float, "rouge1_f1": float, "rouge2_f1": float, "rougeL_f1": float, "bertscore_f1": float}
```

#4. evaluate_model_comprehensive(model, tokenizer, test_loader, test_dataset, device="cuda", num_samples=10)
Complete evaluation combining all metrics.

#Usage:

```python
results = evaluate_model_comprehensive(
    model, tokenizer, test_loader, test_dataset, 
    device="cuda", num_samples=20
)
# Returns: Dict with perplexity, BLEU, ROUGE F1 and BERTScore as well as example generations
```

## Quick Example
```python
from src.evaluation import compute_generation_metrics

# Compare generated texts with references
predictions = ["The restaurant serves French food"]
references = ["This restaurant serves French cuisine"]

metrics = compute_generation_metrics(predictions, references)
print(f"BLEU: {metrics['bleu']:.4f}, ROUGE-L F1: {metrics['rougeL_f1']:.4f}")
```

## Standalone Evaluation Script
Use scripts/evaluate.py for end-to-end evaluation:

```bash
python scripts/evaluate.py \
  --checkpoint results/checkpoints/model_epoch_3.pt \
  --config config.yaml \
  --num_samples 20
```


## Testing
Run the built-in test:

```bash
python src/evaluation/metrics.py
```