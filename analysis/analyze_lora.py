import os
from datetime import datetime
import yaml
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from src.lora.model import LoRAGPT2

def load_lora_model_for_analysis(checkpoint_path, config_path):
    """Load trained LoRA model and tokenizer."""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load base GPT-2 from local cache
    model_name = config["model_name"]
    model = GPT2LMHeadModel.from_pretrained(
        model_name, 
        cache_dir="Path/to/huggingface/hub",
        local_files_only=True
    )
    tokenizer = GPT2TokenizerFast.from_pretrained(
        model_name,
        cache_dir="Path/to/huggingface/hub",
        local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    
    # Wrap with LoRA structure
    lora_config = config["lora"]
    lora_model = LoRAGPT2(
        base_model=model,
        rank=lora_config["rank"],
        alpha=lora_config["alpha"],
        target_modules=lora_config["target_modules"],
        dropout=lora_config.get("dropout", 0.0)
    )
    
    # Load trained weights
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in checkpoint:
        lora_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        lora_model.load_state_dict(checkpoint)
    
    lora_model.eval()
    print(f"Model loaded from: {checkpoint_path}")
    return lora_model, tokenizer


def analyze_sampled_lora_layers(rank, model, sample_config: Dict[str, List[int]] = None):
    """
    Analyze sampled LoRA layers and generate visualizations.
    
    Args:
        model: Loaded LoRA model
        sample_config: Layer sampling config, e.g., {'attn.c_attn': [0, 11, 23], ...}
                      Default: three types, each with low/mid/high layers.
    """
    if sample_config is None:
        # three types, each with low(0), mid(11), high(23) layers
        sample_config = {
            'attn.c_attn': [0, 11, 23],  # QKV projection
            'attn.c_proj': [0, 11, 23],  # Attention output
            'mlp.c_proj': [0, 11, 23]    # MLP output
        }
    
    results = {}
    
    print("=" * 60)
    print("ANALYZING SAMPLED LoRA LAYERS")
    print("=" * 60)
    
    # Analyze each sampled layer
    for layer_type, layer_indices in sample_config.items():
        results[layer_type] = {}
        
        for layer_idx in layer_indices:
            try:
                if layer_type == 'mlp.c_proj':
                    module = model.base_model.transformer.h[layer_idx].mlp.c_proj
                else:
                    attn_module = model.base_model.transformer.h[layer_idx].attn
                    module = getattr(attn_module, layer_type.split('.')[1])
                
                # Analyze adapter
                delta_w = module.lora.weight.detach()
                base_w = module.base_layer.weight.detach()
                
                # 1. Singular value analysis (top 8 only)
                sv = torch.linalg.svdvals(delta_w)
                effective_rank = (sv > 0.05 * sv[0]).sum().item()
                energy_in_rank = (sv[:rank]**2).sum() / (sv**2).sum()
                energy_ratios = []
                for i in range(1, rank+1):
                    ratio = (sv[:i]**2).sum() / (sv[:rank]**2).sum()
                    energy_ratios.append(ratio.item())
                
                # 2. Update magnitude analysis
                rel_norm = delta_w.norm() / base_w.norm()
                
                # 3. Sparsity analysis (threshold = 0.1% of max)
                threshold = 0.001 * delta_w.abs().max()
                sparsity = (delta_w.abs() < threshold).float().mean().item()
                
                layer_key = f"L{layer_idx}"
                results[layer_type][layer_key] = {
                    'shape': delta_w.shape,
                    'effective_rank': effective_rank,
                    'energy_ratio': energy_ratios,
                    'relative_norm': rel_norm.item(),
                    'sparsity': sparsity,
                    'singular_values': sv[:10].cpu().numpy(),
                    'layer_idx': layer_idx
                }
                
                print(f"{layer_type:15} L{layer_idx:2} | "
                      f"Shape: {delta_w.shape} | "
                      f"EffRank: {effective_rank} | "
                      f"Energy: {energy_ratios} | "
                      f"RelNorm: {rel_norm:.4f} | "
                      f"Sparsity: {sparsity:.3f}")
                      
            except AttributeError as e:
                print(f"Warning: Cannot access {layer_type} at layer {layer_idx}: {e}")
                continue
    
    return results


    
class LoRAAnalysisStorage:
    """Store analysis results using pickle."""
    
    def __init__(self, storage_file="lora_analyses.pkl"):
        self.storage_file = storage_file
        self.analyses = {}
        self._load()
    
    def _load(self):
        """Load existing analyses."""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'rb') as f:
                    self.analyses = pickle.load(f)
                print(f"✓ Loaded {len(self.analyses)} historical analyses")
            except Exception as e:
                print(f"Failed to load analysis file: {e}")
                self.analyses = {}
        else:
            print("No historical analysis file found, creating new")
            self.analyses = {}
    
    def save(self):
        """Save all analyses."""
        try:
            with open(self.storage_file, 'wb') as f:
                pickle.dump(self.analyses, f)
            print(f"✓ Saved {len(self.analyses)} analyses to {self.storage_file}")
        except Exception as e:
            print(f"Save failed: {e}")
    
    def save_analysis(self, experiment_name: str, analysis_results: dict):
        """Save a single analysis result."""
        analysis_results['_save_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        self.analyses[experiment_name] = analysis_results
        self.save()
        
        print(f"✓ Saved analysis: {experiment_name}")
    
    def get_analysis(self, experiment_name):
        if experiment_name in self.analyses:
            return self.analyses[experiment_name]
        else:
            available = list(self.analyses.keys())
            raise KeyError(f"Analysis '{experiment_name}' not found. Available: {available}")
    
    def list_analyses(self):
        return list(self.analyses.keys())
    
    def get_analysis_info(self):
        info = {}
        for name, data in self.analyses.items():
            info[name] = {
                'save_time': data.get('_save_time', 'Unknown'),
                'layers': list(data.keys()) if isinstance(data, dict) else []
            }
        return info
    
    def delete_analysis(self, experiment_name):
        if experiment_name in self.analyses:
            del self.analyses[experiment_name]
            self.save()
            print(f"Deleted analysis: {experiment_name}")
        else:
            print(f"Analysis '{experiment_name}' not found")