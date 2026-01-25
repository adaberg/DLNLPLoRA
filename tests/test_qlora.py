import torch
import torch.nn as nn
import pytest
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BitsAndBytesConfig
import sys
from pathlib import Path
from bitsandbytes.nn import Params4bit

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.lora.layer import LoRALayer
from src.lora.model import LoRALinear, LoRAGPT2


class TestLoRALayer:

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
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Quantization requires CUDA")
    def test_wraps_quantized_base_layer(self):
        
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = GPT2LMHeadModel.from_pretrained("gpt2", quantization_config=bnb_config, device_map="auto")
        
        # Get a quantized layer from the model
        base = model.transformer.h[0].attn.c_attn
        lora_linear = LoRALinear(base, rank=8, alpha=16)
        
        assert lora_linear.base_layer is base
        assert lora_linear.lora.lora_A.dtype == torch.float16

    @pytest.mark.linear
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Quantization requires CUDA")
    def test_quantized_output_dtype(self):
        from transformers import GPT2LMHeadModel, BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = GPT2LMHeadModel.from_pretrained("gpt2", quantization_config=bnb_config, device_map="auto")
        
        base = model.transformer.h[0].attn.c_attn
        lora_linear = LoRALinear(base, rank=8, alpha=16)
        
        x = torch.randn(2, 10, 768, dtype=torch.float16, device="cuda")
        output = lora_linear(x)
        
        assert output.dtype == torch.float16

class TestLoRAGPT2:
    """Test LoRAGPT2 model"""

    @pytest.fixture
    def gpt2_model(self):
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
            dtype=torch.float16  # Non-quantized params use fp16
        )
      
        return model

    @pytest.mark.model
    def test_freezes_base_parameters(self, gpt2_model):
      lora_model = LoRAGPT2(
          gpt2_model, rank=8, alpha=16, target_modules=["c_attn", "c_proj"]
      )

      lora_param_ids = {id(p) for p in lora_model.get_lora_parameters()}
      for name, param in gpt2_model.named_parameters():
          if id(param) not in lora_param_ids:
              if isinstance(param, Params4bit):
                  continue  # Quantized params inherently frozen
              assert param.requires_grad == False, f"{name} not frozen"
    
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

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to("cuda")  # Device fix

        output = lora_model(input_ids=input_ids)

        assert hasattr(output, "logits")
        assert output.logits.shape == (batch_size, seq_len, gpt2_model.config.vocab_size)

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

        assert total / trainable > 50 # depends on rank and model size


    @pytest.mark.model
    def test_base_weights_quantized(self, gpt2_model):              
        quantized_count = sum(1 for p in gpt2_model.parameters() if isinstance(p, Params4bit))
        assert quantized_count > 0, "No quantized parameters found"
        
        for name, module in gpt2_model.named_modules():
            if 'c_attn' in name or 'c_fc' in name:
                assert isinstance(module.weight, Params4bit), f"{name} not quantized"
                
    @pytest.mark.model
    def test_lora_params_fp16(self, gpt2_model): 
        lora_model = LoRAGPT2(
            gpt2_model, rank=8, alpha=16, target_modules=["c_attn", "c_proj"]
        )
        for param in  lora_model.get_lora_parameters():
            assert param.dtype == torch.float16, f"LoRA param dtype is {param.dtype}, expected fp16"

class TestLoRALearning:
    
    @pytest.fixture
    def model_and_inputs(self):
        bnb_config = BitsAndBytesConfig(
          load_in_4bit=True,
          bnb_4bit_quant_type="nf4",
          bnb_4bit_compute_dtype=torch.float16,
          bnb_4bit_use_double_quant=True,
          llm_int8_skip_modules=["lm_head"]  # Explicitly skip LM head
        )

        base_model = GPT2LMHeadModel.from_pretrained(
            "gpt2-medium",
            quantization_config=bnb_config,
            torch_dtype=torch.float16
        )
    
        model = LoRAGPT2(base_model, rank=8, alpha=16, target_modules=["c_attn", "c_proj"])
        
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
        tokenizer.pad_token = tokenizer.eos_token
        
        text = ["The students love deep learning for nlp"]
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs["labels"] = inputs["input_ids"].clone()
        
        # Move inputs to CUDA (quantized model is on CUDA)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        return model, inputs
    
    @pytest.mark.learning
    def test_only_lora_receives_gradients(self, model_and_inputs):
        model, inputs = model_and_inputs
        
        lora_params = model.get_lora_parameters()
        optimizer = torch.optim.SGD(lora_params, lr=1e-3)
        
        # warmup step if we want gradient to pass the B
        optimizer.zero_grad()
        model(**inputs).loss.backward()
        optimizer.step()
        
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        
        lora_param_ids = {id(p) for p in lora_params}
        for name, param in model.base_model.named_parameters():
            if id(param) in lora_param_ids:
                assert param.grad is not None, f"LoRA param {name} missing gradient"
                assert not torch.allclose(param.grad, torch.zeros_like(param.grad)), \
                    f"LoRA param {name} has zero gradient"
            elif not isinstance(param, Params4bit):
                assert param.grad is None, f"Base param {name} should not receive gradients"
    
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
            assert not torch.allclose(initial, current.data), \
                "LoRA parameters did not update after optimizer step"
    
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
        
        assert final_loss < initial_loss, \
            f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"
            
            
# tagging of tests definetlye not required but i am flexing
# actually it is useful to not run layer tests again
