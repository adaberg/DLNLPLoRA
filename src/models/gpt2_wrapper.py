# wrapper
from typing import Tuple
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def load_gpt2_model_and_tokenizer(
    model_name: str = "gpt2-medium",
) -> Tuple[GPT2LMHeadModel, GPT2TokenizerFast]:

    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)

    # GPT-2 doesn't have a pad token by default
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def freeze_model(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total, trainable
