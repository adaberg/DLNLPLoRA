# LoRA

tuwien project 2025/2026

### Installation

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

### pytests

#### LoRA

```bash
# can also use -m layer / linear / model to split testing
pytest tests/test_lora.py -v
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

- Weightwatcher
- rank ablation: run multiple runs on different ranks
- QLoRA: quantization + LoRA (could run in parallel on personal hardware)
- DoRA: LoRA updates magnitude and direction of the weight matrix concurrently. it is not good for small nuanced changes in one of the two directions. DoRA solves this by decoupling magnitude from direction through weight decomposition

bonus:
- LoRAFA : frozen A matrix (minimal change of code but almost full processing needed and maybe not that interesting compared to the others)