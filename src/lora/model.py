from typing import List
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from .layer import LoRALayer


class LoRALinear(nn.Module):

    def __init__(
        self, base_layer: nn.Linear, rank: int, alpha: float, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.lora = LoRALayer(
            base_layer.in_features, base_layer.out_features, rank, alpha, dropout
        )

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
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base_model = base_model

        # Freeze all base parameters
        for param in base_model.parameters():
            param.requires_grad = False

        ### debug
        print("Available attention modules:")
        for name, module in base_model.named_modules():
            if isinstance(module, nn.Linear) and 'attn' in name:
                print(f"  {name}")
        ###
        
        # Inject LoRA into target modules
        self.lora_modules = []
        for name, module in base_model.named_modules():
            if isinstance(module, nn.Linear):
                if any(target in name for target in target_modules):
                    self._inject_lora(name, module, rank, alpha, dropout)

        assert len(self.lora_modules) > 0, f"No modules matched {target_modules}"

    def _inject_lora(
        self, name: str, module: nn.Linear, rank: int, alpha: float, dropout: float
    ) -> None:
        """Replace Linear with LoRALinear"""
        parent_name, child_name = name.rsplit(".", 1)
        parent = self.base_model.get_submodule(parent_name)

        lora_linear = LoRALinear(module, rank, alpha, dropout)
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

    def get_lora_parameters(self) -> List[nn.Parameter]:
        """Return A and B"""
        params = []
        for module in self.base_model.modules():
            if isinstance(module, LoRALinear):
                params.extend([module.lora.lora_A, module.lora.lora_B])
        return params
