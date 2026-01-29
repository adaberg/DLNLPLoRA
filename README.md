# LoRA

tuwien project 2025/2026

## Installation

1. **Clone the repository**
   ```bash
   git clone git@github.com:adaberg/DLNLPLoRA.git
   cd DLNLPLoRA
   ```

2. **Create and activate a virtual environment**
   ```bash
   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate
   
   # Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the E2E NLG dataset**
   ```bash
   python src/data/download_data.py
   ```

## Training

### Quick Start

```bash
# LoRA training (default, following paper hyperparameters)
python scripts/train.py --config config.yaml --mode lora

# Full fine-tuning baseline
python scripts/train.py --config config.yaml --mode full

# With custom hyperparameters
python scripts/train.py --config config.yaml --mode lora --lr 2e-4 --epochs 5 --batch_size 8
```

### Training Modes

| Mode | Description | Trainable Params |
|------|-------------|------------------|
| `lora` | LoRA fine-tuning (paper method) | ~0.35M (0.1%) |
| `full` | Full fine-tuning baseline | ~355M (100%) |
| `none` | Evaluation only | 0 |

### Command Line Options

```bash
python scripts/train.py --help

# Key options:
--config CONFIG       # Path to config file (default: config.yaml)
--mode {lora,full,none}  # Training mode
--lr LR               # Learning rate (default: 2e-4)
--epochs EPOCHS       # Number of epochs (default: 5)
--batch_size N        # Batch size (default: 8)
--warmup_steps N      # Warmup steps (default: 500)
--fp16                # Enable FP16 mixed precision
--bf16                # Enable BF16 mixed precision
--resume PATH         # Resume from checkpoint
--output_dir DIR      # Output directory
```

### Default Hyperparameters (from LoRA Paper Table 11)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 2e-4 | AdamW optimizer |
| Weight Decay | 0.01 | L2 regularization |
| Batch Size | 8 | Per-device batch size |
| Epochs | 5 | Training epochs |
| Warmup Steps | 500 | Linear warmup |
| LoRA Rank | 4 | Rank of adaptation matrices |
| LoRA Alpha | 32 | Scaling factor |
| LoRA Dropout | 0.0 | Dropout on LoRA layers |

## Cloud Deployment

### GPU Training (Recommended!)

The training script automatically detects available GPUs/TPUs.

#### Cloud (e.g. JupyterHub) VM Setup
```bash
# 1. Launch a GPU instance (recommended: NVIDIA A100, V100, or T4)
# 2. Clone the repo and install dependencies
git clone git@github.com:adaberg/DLNLPLoRA.git
cd DLNLPLoRA
pip install -r requirements.txt

# 3. Download data
python src/data/download_data.py

# 4. Run training
python scripts/train.py --config config.yaml --mode lora --fp16
```

## Evaluation

```bash
# Evaluate a trained checkpoint
python scripts/run_evaluation.py --checkpoint results/best_model/checkpoint.pt --config results/training_config.json

# Evaluate Zero-shot
python scripts/run_evaluation.py --checkpoint gpt2-medium --config config.yaml
```

## Tests

```bash
source venv/bin/activate
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_lora.py -v  # LoRA tests
pytest tests/test_data.py -v  # Dataset tests

# Test with markers
pytest tests/test_lora.py -v -m layer   # Only layer tests
pytest tests/test_lora.py -v -m linear  # Only linear tests
pytest tests/test_lora.py -v -m model   # Only model tests
pytest tests/test_lora.py -v -m learning
```

## project structure 

```
lora-reproduction/
├── README.md
├── requirements.txt
├── config.yaml
├── src/
│   ├── __init__.py
│   ├── lora/
│   │   ├── __init__.py
│   │   ├── layer.py
│   │   └── model.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── gpt2_wrapper.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py
│   └── evaluation/
│       ├── __init__.py
│       └── metrics.py
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── results/
│   └── .gitkeep
└── tests/
    ├── __init__.py
    ├── test_lora.py
    └── test_data.py
```


## possible extentions

now ordered by feasibility and easeness of implementation in the current project

[] Weightwatcher
[] rank ablation: run multiple runs on different ranks
[x] QLoRA: quantization + LoRA (could run in parallel on personal hardware)
[x] DoRA: LoRA updates magnitude and direction of the weight matrix concurrently. it is not good for small nuanced changes in one of the two directions. DoRA solves this by decoupling magnitude from direction through weight decomposition
[x] Selective LoRA: enable injection only in selected layers, to use in combination with weightwatcher maybe and highlight overfitting layers

bonus:
- LoRAFA : frozen A matrix (minimal change of code but almost full processing needed and maybe not that interesting compared to the others)

# Extention: DoRA

## Tests

```bash
pytest tests/test_dora.py -v

# Test with markers
pytest tests/test_dora.py -v -m layer   # Only layer tests
pytest tests/test_dora.py -v -m linear  # Only linear tests
pytest tests/test_dora.py -v -m model   # Only model tests
pytest tests/test_dora.py -v -m learning
```

## Train & evaluate

```bash
python scripts/train.py --config config.yaml --mode dora
```

```bash
python scripts/run_evaluation.py --checkpoint results/{model}/best_model/checkpoint.pt --config results/{model}/training_config.json
```



# Extention: QLoRA

## Tests

```bash
pytest tests/test_qlora.py -v

# Test with markers
pytest tests/test_qlora.py -v -m layer   # Only layer tests
pytest tests/test_qlora.py -v -m linear  # Only linear tests
pytest tests/test_qlora.py -v -m model   # Only model tests
pytest tests/test_qlora.py -v -m learning
```

# Extention: Selective LoRA

## Tests

```bash
# Test with markers
pytest tests/test_qlora.py -v -m selective
```