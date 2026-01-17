# LoRA

tuwien project 2025/2026

## possible extentions

- Weightwatcher
- DoRA

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