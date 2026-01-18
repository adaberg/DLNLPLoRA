import torch
import torch.nn as nn
import pytest
from transformers import GPT2LMHeadModel
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.lora.layer import LoRALayer
from src.lora.model import LoRALinear, LoRAGPT2


class TestLoRALayer:
    """Test LoRALayer implementation"""

    @pytest.mark.layer
    def test_initialization(self):
        layer = LoRALayer(in_features=128, out_features=64, rank=8, alpha=16)
        assert not torch.allclose(layer.lora_A, torch.zeros_like(layer.lora_A))
        assert torch.allclose(layer.lora_B, torch.zeros_like(layer.lora_B))

    @pytest.mark.layer
    def test_output_shape(self):
        batch_size, seq_len, in_features = 2, 10, 128
        out_features = 64

        layer = LoRALayer(
            in_features=in_features, out_features=out_features, rank=8, alpha=16
        )
        x = torch.randn(batch_size, seq_len, in_features)

        output = layer(x)

        assert output.shape == (batch_size, seq_len, out_features)

    @pytest.mark.layer
    def test_scaling_factor(self):
        """Verify scaling factor alpha/rank is applied"""
        rank, alpha = 8, 16
        layer = LoRALayer(in_features=128, out_features=64, rank=rank, alpha=alpha)

        assert layer.scaling == alpha / rank

    @pytest.mark.layer
    def test_zero_output_when_B_is_zero(self):
        layer = LoRALayer(in_features=128, out_features=64, rank=8, alpha=16)
        x = torch.randn(2, 10, 128)

        output = layer(x)

        assert torch.allclose(output, torch.zeros_like(output))


class TestLoRALinear:

    @pytest.mark.linear
    def test_wraps_base_layer(self):
        base = nn.Linear(128, 64)
        lora_linear = LoRALinear(base, rank=8, alpha=16)

        assert lora_linear.base_layer is base
        assert isinstance(lora_linear.lora, LoRALayer)

    @pytest.mark.linear
    def test_forward_combines_base_and_lora(self):
        base = nn.Linear(128, 64)
        lora_linear = LoRALinear(base, rank=8, alpha=16)

        x = torch.randn(2, 10, 128)

        output = lora_linear(x)
        expected = base(x) + lora_linear.lora(x)

        assert torch.allclose(output, expected)


class TestLoRAGPT2:
    """Test LoRAGPT2 model"""

    # this annotation makes it so that evert test with gpt2_model as paramether
    # gets the initialized gpt2 model without writing it explicitly every time
    @pytest.fixture
    def gpt2_model(self):
        return GPT2LMHeadModel.from_pretrained("gpt2-medium")

    @pytest.mark.model
    def test_freezes_base_parameters(self, gpt2_model):
        lora_model = LoRAGPT2(
            gpt2_model, rank=8, alpha=16, target_modules=["c_attn", "c_proj"]
        )

        lora_param_ids = {id(p) for p in lora_model.get_lora_parameters()}
        for param in gpt2_model.parameters():
            if id(param) not in lora_param_ids:
                assert param.requires_grad == False
    
    @pytest.mark.model
    def test_injects_lora_modules(self, gpt2_model):
        lora_model = LoRAGPT2(
            gpt2_model, rank=8, alpha=16, target_modules=["c_attn", "c_proj"]
        )

        assert len(lora_model.lora_modules) > 0

        for name in lora_model.lora_modules:
            assert any(target in name for target in ["c_attn", "c_proj"])

    @pytest.mark.model
    def test_only_lora_parameters_trainable(self, gpt2_model):
        lora_model = LoRAGPT2(
            gpt2_model, rank=8, alpha=16, target_modules=["c_attn", "c_proj"]
        )

        lora_params = lora_model.get_lora_parameters()

        assert len(lora_params) > 0
        for param in lora_params:
            assert param.requires_grad == True

    @pytest.mark.model
    def test_forward_pass(self, gpt2_model):
        lora_model = LoRAGPT2(
            gpt2_model, rank=8, alpha=16, target_modules=["c_attn", "c_proj"]
        )

        # dummy input
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        output = lora_model(input_ids=input_ids)

        assert hasattr(output, "logits")
        assert output.logits.shape == (
            batch_size,
            seq_len,
            gpt2_model.config.vocab_size,
        )

    @pytest.mark.model
    def test_fails_on_invalid_target_modules(self, gpt2_model):
        with pytest.raises(AssertionError):
            LoRAGPT2(
                gpt2_model, rank=8, alpha=16, target_modules=["invalid_module_name"]
            )

    @pytest.mark.model
    def test_parameter_count(self, gpt2_model):
        lora_model = LoRAGPT2(
            gpt2_model, rank=8, alpha=16, target_modules=["c_attn", "c_proj"]
        )

        total = sum(p.numel() for p in gpt2_model.parameters())
        trainable = sum(p.numel() for p in lora_model.get_lora_parameters())

        assert total / trainable > 100


# tagging of tests probably not required but i am flexing
