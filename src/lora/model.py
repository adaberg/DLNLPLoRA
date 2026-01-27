from typing import List, Union
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from transformers.pytorch_utils import Conv1D
from .layer import LoRALayer, DoRALayer
from os import mkdir
from torchinfo import summary

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
                params.extend(
                    [module.lora.lora_A, module.lora.lora_B, module.lora.magnitude]
                )
        return params


if __name__ == "__main__":
    lora_layer = LoRALayer(in_features=128, out_features=64, rank=8, alpha=16)
    print(lora_layer)
    print("--" * 40)
    print(lora_layer.extra_repr)
    print("--" * 40)
    print()
    dora_layer = DoRALayer(
        in_features=128,
        out_features=64,
        rank=8,
        alpha=16,
        base_weight=torch.randn(64, 128),
    )
    print(dora_layer)
    print("--" * 40)
    print(dora_layer.extra_repr())
    
    mkdir("visualization")
    
    base_model = GPT2LMHeadModel.from_pretrained(
        "gpt2-medium")
    with open("visualization/base.txt", "w") as f:
      f.write(str(summary(base_model, input_size=(1, 128), dtypes=[torch.long], verbose=2, col_names=[])))

    lora = LoRAGPT2(
      base_model,
      rank=8,
      alpha=16,
      target_modules=["c_attn", "c_proj"]
    )
    with open("visualization/lora.txt", "w") as f:
      f.write(str(summary(lora, input_size=(1, 128), dtypes=[torch.long], verbose=2, col_names=[])))
      
    dora = DoRAGPT2(
      base_model=base_model,
      rank=4,
      alpha=16.0,
      dropout=0.0,
      target_modules=["c_attn", "c_proj"],
    )
    with open("visualization/dora.txt", "w") as f:
      f.write(str(summary(dora, input_size=(1, 128), dtypes=[torch.long], verbose=2, col_names=[])))