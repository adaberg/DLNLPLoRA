# wrapper
from typing import Tuple
import torch
import torch.nn as nn
from transformers import BitsAndBytesConfig, GPT2LMHeadModel, GPT2TokenizerFast


def load_gpt2_model_and_tokenizer(
    model_name: str = "gpt2-medium",
) -> Tuple[GPT2LMHeadModel, GPT2TokenizerFast]:

    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)

    # GPT-2 doesn't have a pad token by default
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer

def load_gpt2_model_and_tokenizer_4bit(
    model_name: str = "gpt2-medium",
) -> Tuple[GPT2LMHeadModel, GPT2TokenizerFast]:

    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["lm_head"],
        )

    model = GPT2LMHeadModel.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)

    # GPT-2 doesn't have a pad token by default
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer
  
def load_gpt2_model_and_tokenizer_8bit(
    model_name: str = "gpt2-medium",
) -> Tuple[GPT2LMHeadModel, GPT2TokenizerFast]:

    bnb_config = BitsAndBytesConfig(
            load_in_8bit=True, llm_int8_skip_modules=["lm_head"]
        )

    model = GPT2LMHeadModel.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    
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
