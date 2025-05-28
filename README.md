# RASP Generalization Experiments

Implementation of "What Algorithms can Transformers Learn? A Study in Length Generalization" focusing on the counting task to demonstrate the RASP-Generalization Conjecture.

## Quick Start

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Generate Data
```bash
python data_generation.py
```

### 3. Train Model (PyTorch Lightning - Recommended)
```bash
python train_lightning.py --config counting_config
```

### 4. Alternative Training (Legacy)
```bash
python train.py --config counting_config
```

## Features

- ✅ **PyTorch Lightning Integration**: Modern training framework with built-in logging, checkpointing, and multi-GPU support
- ✅ **Generation-Based Evaluation**: Primary metric focuses on autoregressive generation accuracy (not teacher-forcing)
- ✅ **Length Generalization Testing**: Comprehensive evaluation on sequences longer than training
- ✅ **Wandb + TensorBoard Logging**: Full experiment tracking
- ✅ **Automatic Mixed Precision**: Faster training on modern GPUs
- ✅ **Smart Checkpointing**: Saves best models based on generation accuracy

## Repository Structure

```
├── train_lightning.py      # Main PyTorch Lightning training script ⭐
├── lightning_model.py      # Lightning module implementation
├── train.py               # Legacy training script (still functional)
├── model.py               # Transformer architecture
├── data_generation.py     # Dataset creation
├── evaluate.py            # Evaluation utilities
├── config.py              # Configuration
├── utils.py               # Utilities
├── debug/                 # Debug and test scripts
│   ├── quick_test.py
│   ├── quick_train_test.py
│   ├── debug_accuracy.py
│   └── setup_wandb.py
└── intro.md               # Detailed explanation of the approach
```

## Key Results

The model learns counting patterns that generalize to longer sequences:

- **Training**: Sequences ≤60 tokens
- **Testing**: Sequences 70-150 tokens  
- **Expected Results**: >90% accuracy on length generalization

### Sample Predictions
```
Input:  SoS 5 16 > 
Output: 5 6 7 8 9 10 11 12 13 14 15 16 ✅

Input:  SoS 24 18 >
Output: 24 23 22 21 20 19 18 ✅

Input:  SoS 15 15 >
Output: 15 ✅
```

## Training Options

### PyTorch Lightning (Recommended)
```bash
# Basic training
python train_lightning.py

# With custom settings
python train_lightning.py --seed 123 --output_dir results/

# Debug mode (fast run)
python train_lightning.py --fast_dev_run

# Resume from checkpoint
python train_lightning.py --resume_from output/checkpoints/last.ckpt
```

### Legacy Training
```bash
python train.py --config counting_config
```

## Monitoring

- **Wandb**: Real-time metrics and generation examples
- **TensorBoard**: Local logging and visualizations
- **Generation Examples**: Printed during training to monitor progress

## Configuration

Key parameters in `config.py`:
- Model size: 6 layers, 8 heads, 512 dimensions
- Training: 5000 samples, batch size 32
- Length generalization: Tests on 70-150 token sequences

## Debug Tools

See `debug/` folder for:
- Functionality tests
- Quick training verification  
- Accuracy calculation debugging
- Wandb setup utilities

## Paper Reference

Based on "What Algorithms can Transformers Learn? A Study in Length Generalization" - focuses on the counting task as an example of easily learnable algorithms that exhibit strong length generalization. 