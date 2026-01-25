from typing import List, Union
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, BitsAndBytesConfig
from transformers.pytorch_utils import Conv1D
from .layer import LoRALayer, DoRALayer
from bitsandbytes.nn import Params4bit


class LoRALinear(nn.Module):

    def __init__(
        self,
        base_layer: Union[nn.Linear, Conv1D],
        rank: int,
        alpha: float,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer

        # Handle both Linear and Conv1D if in feature we test other models
        if isinstance(base_layer, nn.Linear):
            in_features = base_layer.in_features
            out_features = base_layer.out_features
        else:
            in_features = base_layer.weight.shape[0]
            out_features = base_layer.nf

        self.lora = LoRALayer(in_features, out_features, rank, alpha, dropout)
        
        if hasattr(base_layer.weight, 'quant_state'):  # Check if quantized
          device = base_layer.weight.device
          self.lora.lora_A.data = self.lora.lora_A.data.to(torch.float16).to(device)
          self.lora.lora_B.data = self.lora.lora_B.data.to(torch.float16).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """W₀x + BAx più facile di così non si può"""
        return self.base_layer(x) + self.lora(x)


class LoRAGPT2(nn.Module):
    """GPT-2 with LoRA adapters in attention layers"""

    def __init__(
        self,
        base_model: GPT2LMHeadModel,
        rank: int,
        alpha: float,
        target_modules: List[str],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.base_model = base_model

        for param in base_model.parameters():
          if not isinstance(param, Params4bit):
            param.requires_grad = False

        self.lora_modules = []
        for name, module in base_model.named_modules():
            if isinstance(module, (nn.Linear, Conv1D)):
                if any(target in name for target in target_modules):
                    self._inject_lora(name, module, rank, alpha, dropout)

        assert len(self.lora_modules) > 0, f"No modules matched {target_modules}"

    def _inject_lora(
        self, name: str, module: nn.Linear, rank: int, alpha: float, dropout: float
    ) -> None:

        parent_name, child_name = name.rsplit(".", 1)
        parent = self.base_model.get_submodule(parent_name)

        lora_linear = LoRALinear(module, rank, alpha, dropout)
        
        # make sure lora parameters are in float16
        lora_linear.lora.lora_A = nn.Parameter(lora_linear.lora.lora_A.to(torch.float16))
        lora_linear.lora.lora_B = nn.Parameter(lora_linear.lora.lora_B.to(torch.float16))

        # says to the parent "when you call child_name you get lora_linear"
        setattr(parent, child_name, lora_linear)
        self.lora_modules.append(name)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        **kwargs,
    ):
        return self.base_model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs
        )

    def generate(self, *args, **kwargs):
        """Delegate text generation to the base model."""
        return self.base_model.generate(*args, **kwargs)

    def get_lora_parameters(self) -> List[nn.Parameter]:
        """Return A and B"""
        params = []
        for module in self.base_model.modules():
            if isinstance(module, LoRALinear):
                params.extend([module.lora.lora_A, module.lora.lora_B])
                
        for param in params:
          assert param.dtype in [torch.float16, torch.float32], \
              f"LoRA param has wrong dtype: {param.dtype}"
        
        return params


class DoRALinear(LoRALinear):
    def __init__(self, base_layer, rank, alpha, dropout):
        super().__init__(base_layer, rank, alpha, dropout)
        base_w = (
            self.base_layer.weight
            if isinstance(base_layer, nn.Linear)
            else self.base_layer.weight.T
        )
        in_f, out_f = self.lora.lora_A.shape[1], self.lora.lora_B.shape[0]
        self.lora = DoRALayer(in_f, out_f, rank, alpha, base_w, dropout)

    def forward(self, x):
        base_w = (
            self.base_layer.weight
            if isinstance(self.base_layer, nn.Linear)
            else self.base_layer.weight.T
        )
        return self.lora(x, base_w)


class DoRAGPT2(LoRAGPT2):
    def _inject_lora(self, name, module, rank, alpha, dropout):
        parent_name, child_name = name.rsplit(".", 1)
        parent = self.base_model.get_submodule(parent_name)

        dora_linear = DoRALinear(module, rank, alpha, dropout)
        setattr(parent, child_name, dora_linear)
        self.lora_modules.append(name)
        
    def get_lora_parameters(self) -> List[nn.Parameter]:
        params = []
        for module in self.base_model.modules():
            if isinstance(module, DoRALinear):
                params.extend([module.lora.lora_A, module.lora.lora_B, module.lora.magnitude])
        return params


if __name__ == "__main__":
  
    bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.float16,
      bnb_4bit_use_double_quant=True,
      llm_int8_skip_modules=["lm_head"]  # Explicitly skip LM head
    )

    model = GPT2LMHeadModel.from_pretrained(
        "gpt2-medium",
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16  # Non-quantized params use fp16
    )
  
    lora = LoRAGPT2(
      model, rank=8, alpha=16, target_modules=["c_attn", "c_proj"]
    )
    
    for name, module in lora.named_modules():
      if isinstance(module, nn.Linear):
          has_4bit = any(isinstance(p, Params4bit) for p in module.parameters())
          print(f"{name}: quantized={has_4bit}")
          
    #print()
    #dora_layer = DoRALayer(
    #    in_features=128,
    #    out_features=64,
    #    rank=8,
    #    alpha=16,
    #    base_weight=torch.randn(64, 128),
    #)
    #print(dora_layer)
    #print("--" * 40)
    #print(dora_layer.extra_repr())
