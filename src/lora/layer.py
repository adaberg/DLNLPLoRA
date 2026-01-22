import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """
    Attributes:
        rank: Rank of decomposition
        alpha: Scaling factor for LoRA updates
        scaling: Computed scaling factor (alpha / rank)
        lora_A: Down-projection matrix (rank × in_features)
        lora_B: Up-projection matrix (out_features × rank)
        dropout: Dropout layer applied before low-rank projection
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        """
        importnant
            in_features: Input dimension (must match base layer)
            out_features: Output dimension (must match base layer)
        """
        super().__init__()

        if rank > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {rank} cannot exceed min(in_features={in_features}, "
                f"out_features={out_features})={min(in_features, out_features)}"
            )

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Original LoRA paper: "random Gaussian initialization for A"
        # self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        # Using Kaiming ensures proper gradient flow at initialization
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: ONLY the LoRA adaptation, not the base layer output.
        """
        # Apply dropout for regularization
        x_dropped = self.dropout(x)

        down_proj = x_dropped @ self.lora_A.T
        up_proj = down_proj @ self.lora_B.T

        # Apply scaling factor α/r
        # This scaling is crucial for controlling the magnitude of LoRA updates
        return up_proj * self.scaling

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def extra_repr(self) -> str:
        """
        String representation for debugging and logging.
        """
        return (
            f"in_features={self.lora_A.shape[1]}, "
            f"out_features={self.lora_B.shape[0]}, "
            f"rank={self.rank}, "
            f"alpha={self.alpha}, "
            f"scaling={self.scaling:.4f}, "
            f"dropout={self.dropout.p if isinstance(self.dropout, nn.Dropout) else 0.0}"
        )

    # this decorator makes it possible to access weight as a property
    # leyer.weight instead of layer.weight()
    @property
    def weight(self) -> torch.Tensor:
        # (out_features × rank) @ (rank × in_features)
        return (self.lora_B @ self.lora_A) * self.scaling

    def merge_with_base_weight(self, base_weight: torch.Tensor) -> torch.Tensor:
        """
        Merge LoRA parameters with base weight for efficient inference.
        """
        if base_weight.shape != (self.lora_B.shape[0], self.lora_A.shape[1]):
            raise ValueError(
                f"Base weight shape {base_weight.shape} doesn't match LoRA shape "
                f"({self.lora_B.shape[0]}, {self.lora_A.shape[1]})"
            )

        # W' = W₀ + (α/r)BA
        return base_weight + self.weight

    def get_num_parameters(self) -> tuple[int, int]:
        lora_params = self.rank * (self.lora_A.shape[1] + self.lora_B.shape[0])
        full_params = self.lora_A.shape[1] * self.lora_B.shape[0]
        return lora_params, full_params


class DoRALayer(LoRALayer):

    def __init__(
        self, in_features, out_features, rank, alpha, base_weight, dropout=0.0
    ):
        super().__init__(in_features, out_features, rank, alpha, dropout)
        self.magnitude = nn.Parameter(torch.norm(base_weight, dim=0))

    def forward(self, x, base_weight):
        x = self.dropout(x)
        merged = base_weight + (self.lora_B @ self.lora_A) * self.scaling
        norm = torch.norm(merged, dim=0, keepdim=True) + 1e-8
        # following the paper description
        return x @ (self.magnitude * merged / norm).T

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        # it effectively resets, it does not count as initialization (for my understanding)
        nn.init.ones_(self.magnitude)

    def extra_repr(self) -> str:
        """
        String representation for debugging and logging.
        """
        return (
            f"in_features={self.lora_A.shape[1]}, "
            f"out_features={self.lora_B.shape[0]}, "
            f"rank={self.rank}, "
            f"alpha={self.alpha}, "
            f"scaling={self.scaling:.4f}, "
            f"magnitude={self.magnitude}, "
            f"dropout={self.dropout.p if isinstance(self.dropout, nn.Dropout) else 0.0}"
        )

    @property
    def magnitude(self) -> torch.Tensor:
        return self._magnitude

    def get_num_parameters(self) -> tuple[int, int]:
        lora_params = (
            self.rank * (self.lora_A.shape[1] + self.lora_B.shape[0])
            + self.magnitude.numel()
        )
        full_params = self.lora_A.shape[1] * self.lora_B.shape[0]
        return lora_params, full_params
