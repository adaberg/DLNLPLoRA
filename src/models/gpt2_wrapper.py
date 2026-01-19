# wrapper
from typing import Tuple
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def load_gpt2_model_and_tokenizer(
    model_name: str = "gpt2-medium"
) -> Tuple[GPT2LMHeadModel, GPT2TokenizerFast]:
  
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    
    return model, tokenizer

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total, trainable