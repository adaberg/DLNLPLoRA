import torch
import torch.nn as nn
import pytest
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import sys
from pathlib import Path
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.lora.layer import DoRALayer
from src.lora.model import DoRALinear, DoRAGPT2


class TestDoRALayer:

    @pytest.mark.layer
    def test_initialization(self):
        layer = DoRALayer(
            in_features=128,
            out_features=64,
            rank=8,
            alpha=16,
            base_weight=torch.randn(64, 128),
        )
        assert not torch.allclose(layer.lora_A, torch.zeros_like(layer.lora_A))
        assert torch.allclose(layer.lora_B, torch.zeros_like(layer.lora_B))
        assert layer.magnitude.shape == (128,)

    @pytest.mark.layer
    def test_output_shape(self):
        batch_size, seq_len, in_features = 2, 10, 128
        out_features = 64

        layer = DoRALayer(
            in_features=in_features,
            out_features=out_features,
            rank=8,
            alpha=16,
            base_weight=torch.randn(64, 128),
        )
        x = torch.randn(batch_size, seq_len, in_features)

        output = layer(x, torch.randn(64, 128))

        assert output.shape == (batch_size, seq_len, out_features)

    @pytest.mark.layer
    def test_scaling_factor(self):
        """Verify scaling factor alpha/rank is applied"""
        rank, alpha = 8, 16
        layer = DoRALayer(
            in_features=128,
            out_features=64,
            rank=rank,
            alpha=alpha,
            base_weight=torch.randn(64, 128),
        )

        assert layer.scaling == alpha / rank

    @pytest.mark.layer
    def test_identity_when_B_is_zero(self):
        base_weight = torch.randn(64, 128)
        layer = DoRALayer(
            in_features=128,
            out_features=64,
            rank=8,
            alpha=16,
            base_weight=base_weight,
            dropout=0.0,
        )
        print(f"Dropout in layer: {layer.dropout}")
        x = torch.randn(2, 10, 128)
        output = layer(x, base_weight)
        expected = F.linear(x, base_weight)

        assert torch.allclose(output, expected, atol=1e-6)


class TestDoRALinear:

    @pytest.mark.linear
    def test_wraps_base_layer(self):
        base = nn.Linear(128, 64)
        dora_linear = DoRALinear(base, rank=8, alpha=16, dropout=0.0)

        assert dora_linear.base_layer is base
        assert isinstance(dora_linear.lora, DoRALayer)

    @pytest.mark.linear
    def test_forward_applies_dora_formula(self):
        base = nn.Linear(128, 64)
        dora_linear = DoRALinear(base, rank=8, alpha=16, dropout=0.0)

        x = torch.randn(2, 10, 128)

        output = dora_linear(x)
        # Manually compute DoRA formula
        merged = (
            base.weight
            + (dora_linear.lora.lora_B @ dora_linear.lora.lora_A)
            * dora_linear.lora.scaling
        )
        norm = torch.norm(merged, dim=0, keepdim=True)
        W = dora_linear.lora.magnitude * merged / norm
        expected = x @ W.T

        assert torch.allclose(output, expected)


class TestDoRAGPT2:

    # this annotation makes it so that evert test with gpt2_model as paramether
    # gets the initialized gpt2 model without writing it explicitly every time
    @pytest.fixture
    def gpt2_model(self):
        return GPT2LMHeadModel.from_pretrained("gpt2-medium")

    @pytest.mark.model
    def test_freezes_base_parameters(self, gpt2_model):
        dora_model = DoRAGPT2(
            gpt2_model, rank=8, alpha=16, target_modules=["c_attn", "c_proj"]
        )

        dora_param_ids = {id(p) for p in dora_model.get_lora_parameters()}
        for param in gpt2_model.parameters():
            if id(param) not in dora_param_ids:
                assert param.requires_grad == False

    @pytest.mark.model
    def test_injects_lora_modules(self, gpt2_model):
        dora_model = DoRAGPT2(
            gpt2_model, rank=8, alpha=16, target_modules=["c_attn", "c_proj"]
        )

        assert len(dora_model.lora_modules) > 0

        for name in dora_model.lora_modules:
            assert any(target in name for target in ["c_attn", "c_proj"])

    @pytest.mark.model
    def test_only_lora_parameters_trainable(self, gpt2_model):
        dora_model = DoRAGPT2(
            gpt2_model, rank=8, alpha=16, target_modules=["c_attn", "c_proj"]
        )

        dora_params = dora_model.get_lora_parameters()
        assert len(dora_params) > 0
        for param in dora_params:
            assert param.requires_grad == True

    @pytest.mark.model
    def test_forward_pass(self, gpt2_model):
        dora_model = DoRAGPT2(
            gpt2_model, rank=8, alpha=16, target_modules=["c_attn", "c_proj"]
        )

        # dummy input
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        output = dora_model(input_ids=input_ids)

        assert hasattr(output, "logits")
        assert output.logits.shape == (
            batch_size,
            seq_len,
            gpt2_model.config.vocab_size,
        )

    @pytest.mark.model
    def test_fails_on_invalid_target_modules(self, gpt2_model):
        with pytest.raises(AssertionError):
            DoRAGPT2(
                gpt2_model, rank=8, alpha=16, target_modules=["invalid_module_name"]
            )

    @pytest.mark.model
    def test_parameter_count(self, gpt2_model):
        dora_model = DoRAGPT2(
            gpt2_model, rank=8, alpha=16, target_modules=["c_attn", "c_proj"]
        )

        total = sum(p.numel() for p in gpt2_model.parameters())
        trainable = sum(p.numel() for p in dora_model.get_lora_parameters())

        assert total / trainable > 100


class TestDoRALearning:

    @pytest.fixture
    def model_and_inputs(self):
        base_model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
        model = DoRAGPT2(
            base_model=base_model,
            rank=4,
            alpha=16.0,
            dropout=0.0,
            target_modules=["c_attn", "c_proj"],
        )

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
        tokenizer.pad_token = tokenizer.eos_token

        text = ["The students love deep learning for nlp"]
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs["labels"] = inputs["input_ids"].clone()

        return model, inputs

    @pytest.mark.learning
    def test_only_dora_receives_gradients(self, model_and_inputs):
        model, inputs = model_and_inputs

        dora_params = model.get_lora_parameters()
        optimizer = torch.optim.SGD(dora_params, lr=1e-3)
        
        # warmup step if we want gradient to pass the B
        optimizer.zero_grad()
        model(**inputs).loss.backward()
        optimizer.step()

        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()

        for name, param in model.base_model.named_parameters():
            if "lora" in name:
                assert param.grad is not None, f"LoRA param {name} missing gradient"
                assert not torch.allclose(
                    param.grad, torch.zeros_like(param.grad)
                ), f"LoRA param {name} has zero gradient"
            else:
                assert param.grad is None or torch.allclose(
                    param.grad, torch.zeros_like(param.grad)
                ), f"Base param {name} should not receive gradients"

    @pytest.mark.learning
    def test_parameters_update_after_step(self, model_and_inputs):
        model, inputs = model_and_inputs

        lora_params = model.get_lora_parameters()
        initial_params = [p.data.clone() for p in lora_params]

        optimizer = torch.optim.AdamW(lora_params, lr=1e-3)

        # again warmp up step
        for _ in range(2):
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        for initial, current in zip(initial_params, lora_params):
            assert not torch.allclose(
                initial, current.data
            ), "LoRA parameters did not update after optimizer step"

    @pytest.mark.learning
    def test_loss_decreases_on_overfit(self, model_and_inputs):
        model, inputs = model_and_inputs

        lora_params = model.get_lora_parameters()
        optimizer = torch.optim.AdamW(lora_params, lr=1e-3)

        initial_loss = model(**inputs).loss.item()

        for _ in range(10):
            optimizer.zero_grad()
            loss = model(**inputs).loss
            loss.backward()
            optimizer.step()

        final_loss = model(**inputs).loss.item()

        assert (
            final_loss < initial_loss
        ), f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"


# tagging of tests definetlye not required but i am flexing
# actually it is useful to not run layer tests again
