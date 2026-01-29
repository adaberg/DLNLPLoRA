import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import pandas as pd

def load_lora_analyses(pkl_path="lora_analyses.pkl"):
    """Load analysis data from pickle file."""
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"File not found: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        analyses = pickle.load(f)
    
    print(f" Loaded {len(analyses)} experiment analyses")
    return analyses

def extract_analysis_data(analyses: Dict, layer_types=None):
    """
    Extract structured data from analysis results.
    
    Args:
        analyses: Dictionary loaded from pickle
        layer_types: List of layer types to analyze
    
    Returns:
        Structured data as DataFrame
    """
    if layer_types is None:
        layer_types = ['attn.c_attn', 'attn.c_proj', 'mlp.c_proj']
    
    all_data = []
    
    for exp_name, exp_data in analyses.items():
        if not isinstance(exp_data, dict) or '_save_time' in exp_name:
            continue
        
        data_percentage = None
        rank = None
        
        import re
        pct_match = re.search(r'(\d+)p', exp_name)
        if pct_match:
            data_percentage = int(pct_match.group(1))
        
        rank_match = re.search(r'_r(\d+)', exp_name)
        if rank_match:
            rank = int(rank_match.group(1))
        
        for layer_type in layer_types:
            if layer_type not in exp_data:
                continue
            
            layer_results = exp_data[layer_type]
            
            for layer_key, layer_data in layer_results.items():
                if not isinstance(layer_data, dict):
                    continue
                
                layer_idx = layer_data.get('layer_idx', 
                                          int(layer_key.replace('L', '')) if 'L' in layer_key else 0)
                
                data_point = {
                    'experiment': exp_name,
                    'layer_type': layer_type,
                    'layer': f"L{layer_idx}",
                    'layer_idx': layer_idx,
                    'data_percentage': data_percentage, 
                    'effective_rank': layer_data.get('effective_rank', 0),
                    'effective_rank_ratio': layer_data.get('effective_rank', 0) / rank if rank and rank > 0 else 0,
                    'relative_norm': layer_data.get('relative_norm', 0),
                    'sparsity': layer_data.get('sparsity', 0),
                    'rank': rank
                }
                
                # Extract energy ratios (take first 8 or available)
                energy_ratios = layer_data.get('energy_ratio', [])
                for i, ratio in enumerate(energy_ratios[:8]):
                    data_point[f'energy_ratio_r{i+1}'] = ratio

                # Calculate singular values needed for 95% energy
                if 'singular_values' in layer_data:
                    sv = layer_data['singular_values']
                    if len(sv) > 0:
                        total_energy = np.sum(sv**2)
                        
                        cumulative_energy = 0
                        sv_for_95 = len(sv) 
                        for k, s in enumerate(sv, 1):
                            cumulative_energy += s**2
                            if cumulative_energy / total_energy >= 0.95:
                                sv_for_95 = k
                                break
                        
                        data_point['sv_for_95percent'] = sv_for_95
                        data_point['energy_95percent'] = cumulative_energy / total_energy
                    else:
                        data_point['sv_for_95percent'] = 0
                        data_point['energy_95percent'] = 0
                else:
                    data_point['sv_for_95percent'] = 0
                    data_point['energy_95percent'] = 0
                
                all_data.append(data_point)
    
    return pd.DataFrame(all_data)


def plot_effective_rank_ratio_by_layer_type_split(df, save_dir="effective_rank_analysis"):
    """
    Plot effective rank ratio separated by data percentage.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if 'data_percentage' not in df.columns:
        print("Warning: Missing 'data_percentage' column")
        return
    
    data_percentages = sorted(df['data_percentage'].unique())
    
    for data_pct in data_percentages:
        pct_data = df[df['data_percentage'] == data_pct]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Effective Rank Ratio Analysis (Data: {data_pct}%)', 
                    fontsize=16, fontweight='bold')
        
        layer_types = pct_data['layer_type'].unique()
        
        for idx, layer_type in enumerate(layer_types):
            ax = axes[idx]
            layer_data = pct_data[pct_data['layer_type'] == layer_type]
            
            # Use seaborn barplot
            sns.barplot(data=layer_data, x='layer', y='effective_rank_ratio', 
                       hue='rank', ax=ax, palette='Set3')
            
            ax.set_xlabel('Layer', fontsize=12)
            ax.set_ylabel('Effective Rank Ratio', fontsize=12)
            ax.set_title(f'{layer_type}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(title='Rank', fontsize=10, title_fontsize=11)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'effective_rank_ratio_{data_pct}p.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f" Effective rank ratio plot ({data_pct}%) saved to: {save_path}")

def plot_relative_norm_by_layer_type_split(df, save_dir="relative_norm_analysis"):
    """
    Plot relative norm separated by data percentage.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if 'data_percentage' not in df.columns:
        print("Warning: Missing 'data_percentage' column")
        return
    
    data_percentages = sorted(df['data_percentage'].unique())
    
    for data_pct in data_percentages:
        pct_data = df[df['data_percentage'] == data_pct]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Relative Norm Analysis (Data: {data_pct}%)', 
                    fontsize=16, fontweight='bold')
        
        layer_types = pct_data['layer_type'].unique()
        
        for idx, layer_type in enumerate(layer_types):
            ax = axes[idx]
            layer_data = pct_data[pct_data['layer_type'] == layer_type]
            
            layers = sorted(layer_data['layer'].unique())
            ranks = sorted(layer_data['rank'].unique())
            
            heatmap_data = np.zeros((len(layers), len(ranks)))
            
            for i, layer in enumerate(layers):
                for j, rank in enumerate(ranks):
                    subset = layer_data[(layer_data['layer'] == layer) & (layer_data['rank'] == rank)]
                    if not subset.empty:
                        heatmap_data[i, j] = subset['relative_norm'].mean()

            im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
            
            ax.set_xlabel('Rank', fontsize=12)
            ax.set_ylabel('Layer', fontsize=12)
            ax.set_title(f'{layer_type}', fontsize=14, fontweight='bold')
            ax.set_xticks(np.arange(len(ranks)))
            ax.set_yticks(np.arange(len(layers)))
            ax.set_xticklabels(ranks)
            ax.set_yticklabels(layers)
            
            for i in range(len(layers)):
                for j in range(len(ranks)):
                    text = ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
                                  ha="center", va="center", 
                                  color="black" if heatmap_data[i, j] < 0.5 else "white",
                                  fontsize=9)
            
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'relative_norm_{data_pct}p.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f" Relative norm plot ({data_pct}%) saved to: {save_path}")

def plot_sparsity_by_layer_type_split(df, save_dir="sparsity_analysis"):
    """
    Plot sparsity separated by data percentage.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if 'data_percentage' not in df.columns:
        print("Warning: Missing 'data_percentage' column")
        return
    
    data_percentages = sorted(df['data_percentage'].unique())
    
    for data_pct in data_percentages:
        pct_data = df[df['data_percentage'] == data_pct]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Sparsity Analysis (Data: {data_pct}%)', 
                    fontsize=16, fontweight='bold')
        
        layer_types = pct_data['layer_type'].unique()
        
        for idx, layer_type in enumerate(layer_types):
            ax = axes[idx]
            layer_data = pct_data[pct_data['layer_type'] == layer_type]
            
            layers = sorted(layer_data['layer'].unique())
            layers_pos = np.arange(len(layers))
            
            for rank in sorted(pct_data['rank'].unique()):
                rank_data = layer_data[layer_data['rank'] == rank]
                
                sparsity_vals = []
                for layer in layers:
                    layer_rank_data = rank_data[rank_data['layer'] == layer]
                    if not layer_rank_data.empty:
                        sparsity_vals.append(layer_rank_data['sparsity'].mean())
                    else:
                        sparsity_vals.append(0)
                
                ax.plot(layers_pos, sparsity_vals, marker='o', linewidth=2, 
                       label=f'Rank={rank}', markersize=8)
            
            ax.set_xlabel('Layer', fontsize=12)
            ax.set_ylabel('Sparsity', fontsize=12)
            ax.set_title(f'{layer_type}', fontsize=14, fontweight='bold')
            ax.set_xticks(layers_pos)
            ax.set_xticklabels(layers, rotation=45)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(title='Rank', fontsize=10, title_fontsize=11)
            
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'sparsity_{data_pct}p.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Sparsity plot ({data_pct}%) saved to: {save_path}")

def plot_first_energy_ratio_by_rank_split(df, save_dir="first_energy_ratio_analysis"):
    """
    Plot first energy ratio separated by data percentage.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if 'data_percentage' not in df.columns:
        print("Warning: Missing 'data_percentage' column")
        return
    
    data_percentages = sorted(df['data_percentage'].unique())
    
    for data_pct in data_percentages:
        pct_data = df[df['data_percentage'] == data_pct]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'First Energy Ratio Analysis (Data: {data_pct}%)', 
                    fontsize=16, fontweight='bold')
        
        layer_types = pct_data['layer_type'].unique()
        
        for idx, layer_type in enumerate(layer_types):
            ax = axes[idx]
            layer_data = pct_data[pct_data['layer_type'] == layer_type]
            
            layers = sorted(layer_data['layer'].unique())
            ranks = sorted(layer_data['rank'].unique())
            
            heatmap_data = np.zeros((len(layers), len(ranks)))
            
            for i, layer in enumerate(layers):
                for j, rank in enumerate(ranks):
                    subset = layer_data[(layer_data['layer'] == layer) & (layer_data['rank'] == rank)]
                    if not subset.empty:
                        if 'energy_ratio_r1' in subset.columns:
                            heatmap_data[i, j] = subset['energy_ratio_r1'].mean()
            
            im = ax.imshow(heatmap_data, cmap='RdYlBu', aspect='auto', vmin=0, vmax=1)
            
            ax.set_xlabel('Rank', fontsize=12)
            ax.set_ylabel('Layer', fontsize=12)
            ax.set_title(f'{layer_type}', fontsize=14, fontweight='bold')
            ax.set_xticks(np.arange(len(ranks)))
            ax.set_yticks(np.arange(len(layers)))
            ax.set_xticklabels(ranks)
            ax.set_yticklabels(layers)
            
            for i in range(len(layers)):
                for j in range(len(ranks)):
                    text = ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
                                  ha="center", va="center", 
                                  color="black" if heatmap_data[i, j] < 0.5 else "white",
                                  fontsize=10, fontweight='bold')
            
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('First Energy Ratio', fontsize=12)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'first_energy_ratio_{data_pct}p.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ First energy ratio plot ({data_pct}%) saved to: {save_path}")

def plot_sv_for_95percent_energy_split(df, save_dir="sv_for_95percent_energy"):
    """
    Plot singular values needed for 95% cumulative energy, separated by data percentage.
    """
    if 'sv_for_95percent' not in df.columns:
        print("Warning: Missing 'sv_for_95percent' column")
        return
    
    if 'data_percentage' not in df.columns:
        print("Warning: Missing 'data_percentage' column")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    data_percentages = sorted(df['data_percentage'].unique())
    
    for data_pct in data_percentages:
        pct_data = df[df['data_percentage'] == data_pct]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Singular Values Needed for 95% Energy (Data: {data_pct}%)', 
                    fontsize=16, fontweight='bold')
        
        layer_types = pct_data['layer_type'].unique()
        
        for idx, layer_type in enumerate(layer_types):
            ax = axes[idx]
            layer_data = pct_data[pct_data['layer_type'] == layer_type]
            
            pivot_data = layer_data.pivot_table(
                values='sv_for_95percent',
                index='layer',
                columns='rank',
                aggfunc='mean'
            )
            
            pivot_data = pivot_data.fillna(0).astype(int)
            
            heatmap_data = pivot_data.values
            
            valid_data = heatmap_data[~np.isnan(heatmap_data)]
            if len(valid_data) > 0:
                vmin = np.min(valid_data)
                vmax = np.max(valid_data)
            else:
                vmin, vmax = 0, 1
            
            if vmin == vmax:
                vmin = vmin - 0.5
                vmax = vmax + 0.5
            
            im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto',
                          vmin=vmin, vmax=vmax)
            
            ax.set_xlabel('Rank', fontsize=12)
            ax.set_ylabel('Layer', fontsize=12)
            ax.set_title(f'{layer_type}', fontsize=14, fontweight='bold')
            
            ax.set_xticks(np.arange(len(pivot_data.columns)))
            ax.set_yticks(np.arange(len(pivot_data.index)))
            ax.set_xticklabels(pivot_data.columns.astype(int))
            ax.set_yticklabels(pivot_data.index)
            
            for i in range(len(pivot_data.index)):
                for j in range(len(pivot_data.columns)):
                    value = pivot_data.iloc[i, j]
                    if vmax > vmin:
                        norm_value = (value - vmin) / (vmax - vmin)
                    else:
                        norm_value = 0.5
                    
                    text_color = "white" if norm_value > 0.5 else "black"
                    ax.text(j, i, f'{value:d}',
                           ha="center", va="center", 
                           color=text_color,
                           fontsize=10, fontweight='bold')
            
            plt.colorbar(im, ax=ax, label='Singular Values Needed')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'sv_for_95percent_{data_pct}p.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Singular values for 95% energy plot ({data_pct}%) saved to: {save_path}")
        
        
def generate_all_visualizations(pkl_path="lora_analyses.pkl", output_dir="lora_analysis_plots"):
    """
    Generate all visualization plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    analyses = load_lora_analyses(pkl_path)
    
    df = extract_analysis_data(analyses)
    
    if df.empty:
        print("Warning: No valid data extracted, check pickle file format")
        return
    
    print(f"\nData analysis statistics:")
    print(f"Total data points: {len(df)}")
    print(f"Data percentages: {sorted(df['data_percentage'].unique())}")
    print(f"Layer types: {df['layer_type'].unique().tolist()}")
    print(f"Layers: {sorted(df['layer'].unique())}")
    print(f"Rank values: {sorted(df['rank'].unique())}")
    
    print("\n" + "="*60)
    print("Generating visualizations by data percentage...")
    print("="*60)
    
    # 1. Effective rank analysis
    plot_effective_rank_ratio_by_layer_type_split(
        df, 
        save_dir=os.path.join(output_dir, "effective_rank")
    )
    
    # 2. Relative norm analysis
    plot_relative_norm_by_layer_type_split(
        df,
        save_dir=os.path.join(output_dir, "relative_norm")
    )
    
    # 3. Sparsity analysis
    plot_sparsity_by_layer_type_split(
        df,
        save_dir=os.path.join(output_dir, "sparsity")
    )
    
    # 4. First energy ratio analysis
    plot_first_energy_ratio_by_rank_split(
        df,
        save_dir=os.path.join(output_dir, "first_energy_ratio")
    )
    
    # 5. Singular values for 95% energy analysis
    plot_sv_for_95percent_energy_split(
        df,
        save_dir=os.path.join(output_dir, "sv_for_95percent")
    )
    
    print("\n" + "="*60)
    print(f"✓ All visualizations saved to: {output_dir}/")
    print("="*60)
    
    return df


if __name__ == "__main__":
    # Generate all visualizations
    df = generate_all_visualizations(
        pkl_path="lora_analyses.pkl",
        output_dir="lora_analysis_results"
    )