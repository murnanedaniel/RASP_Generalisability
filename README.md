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
# Train on flat distribution (recommended for proper generalization testing)
python train_lightning.py --config counting_config_flat

# Or train on biased distribution (original data)
python train_lightning.py --config counting_config_biased
```

### 4. Alternative Training (Legacy)
```bash
python train.py --config counting_config
```

## Dataset Configuration Options

### 🎯 Flat Distribution (`counting_config_flat`) - **Recommended**
- **Equal representation** of all sequence lengths (6-60 tokens)
- **Perfect length coverage** for robust generalization testing
- **Data location**: `data_flat/` directory
- **Usage**: `--config counting_config_flat`

### 📊 Biased Distribution (`counting_config_biased`) - Original
- **Natural length distribution** with bias toward shorter sequences
- **Limited long sequence coverage** (lengths 40-60 underrepresented)
- **Data location**: `data/` directory  
- **Usage**: `--config counting_config_biased`

### Configuration Comparison
| Config | Avg Length | Length 50-60 Coverage | Generalization Quality |
|--------|------------|----------------------|----------------------|
| Flat   | 33.0       | 18% (996 samples)    | ⭐⭐⭐ Excellent |
| Biased | 20.2       | 0% (0 samples)       | ⭐ Limited |

## Training Options

### PyTorch Lightning (Recommended)
```bash
# Flat distribution training (best for generalization)
python train_lightning.py --config counting_config_flat

# Biased distribution training (original data)
python train_lightning.py --config counting_config_biased

# Manual data directory override
python train_lightning.py --config counting_config --data_dir data_flat

# With custom settings
python train_lightning.py --config counting_config_flat --seed 123 --output_dir results/

# Debug mode (fast run)
python train_lightning.py --fast_dev_run

# Resume from checkpoint
python train_lightning.py --resume_from output/checkpoints/last.ckpt
```

### Legacy Training
```bash
python train.py --config counting_config
```

## Length Generalization Testing

### Evaluation Options

#### 🚀 Quick Evaluation (Recommended)
```bash
# Fast length generalization test on Lightning checkpoints
python quick_evaluate.py
```
- **Purpose**: Quick test of length generalization performance
- **Input**: Automatically uses best Lightning checkpoint  
- **Output**: Clear pass/fail results for different sequence lengths
- **Best for**: Getting immediate results after training

#### 📊 Comprehensive Evaluation  
```bash
# Detailed analysis with visualizations
python evaluate.py --checkpoint output/checkpoints/best-epoch=41-val_generation_exact_match=0.990.ckpt

# Test specific lengths
python evaluate.py --checkpoint your_model.ckpt --test_lengths 70,80,90,100

# Full generalization suite with plots
python evaluate.py --checkpoint your_model.ckpt --comprehensive
```
- **Purpose**: In-depth analysis with failure case examination and plots
- **Input**: Works with both Lightning and regular checkpoints
- **Output**: Detailed metrics, visualizations, and error analysis
- **Best for**: Research analysis and paper figures

### Automatic Evaluation

The training script automatically performs length generalization testing on completion:

1. **Training Phase**: Sequences 6-60 tokens
2. **Validation**: Real-time generation accuracy monitoring  
3. **Final Testing**: Sequences 70-150 tokens (out-of-distribution)

### Expected Results Pattern
```
Length  70: 0.950 (95/100)  ✅ Strong generalization
Length  80: 0.920 (92/100)  ✅ Good generalization  
Length  90: 0.890 (89/100)  ✅ Decent generalization
Length 100: 0.850 (85/100)  ⚠️  Degrading
Length 120: 0.700 (70/100)  ⚠️  Significant drop
Length 150: 0.500 (50/100)  ❌ Poor generalization
```

## Features

- ✅ **PyTorch Lightning Integration**: Modern training framework with built-in logging, checkpointing, and multi-GPU support
- ✅ **Generation-Based Evaluation**: Primary metric focuses on autoregressive generation accuracy (not teacher-forcing)
- ✅ **Length Generalization Testing**: Comprehensive evaluation on sequences longer than training
- ✅ **Dual Dataset Support**: Both flat and biased length distributions available
- ✅ **Wandb + TensorBoard Logging**: Full experiment tracking with dataset type logging
- ✅ **Automatic Mixed Precision**: Faster training on modern GPUs
- ✅ **Smart Checkpointing**: Saves best models based on generation accuracy

## Repository Structure

```
├── train_lightning.py      # Main PyTorch Lightning training script ⭐
├── lightning_model.py      # Lightning module implementation
├── train.py               # Legacy training script (still functional)
├── model.py               # Transformer architecture
├── data_generation.py     # Dataset creation (both distributions)
├── quick_evaluate.py      # Quick length generalization testing ⭐
├── evaluate.py            # Comprehensive evaluation with analysis
├── config.py              # Configuration with dataset options ⭐
├── utils.py               # Utilities
├── data/                  # Biased distribution data
├── data_flat/             # Flat distribution data ⭐
├── debug/                 # Debug and test scripts
│   ├── quick_test.py
│   ├── quick_train_test.py
│   ├── debug_accuracy.py
│   └── setup_wandb.py
└── intro.md               # Detailed explanation of the approach
```

## Key Results

The model learns counting patterns that generalize to longer sequences:

- **Training**: Sequences ≤60 tokens (flat distribution recommended)
- **Testing**: Sequences 70-150 tokens  
- **Expected Results**: >90% accuracy on length generalization with flat distribution

### Sample Predictions
```
Input:  SoS 5 16 > 
Output: 5 6 7 8 9 10 11 12 13 14 15 16 ✅

Input:  SoS 24 18 >
Output: 24 23 22 21 20 19 18 ✅

Input:  SoS 15 15 >
Output: 15 ✅

# Longer sequences (out-of-distribution)
Input:  SoS 45 75 >
Output: 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 ✅
```

## Monitoring

- **Wandb**: Real-time metrics and generation examples (includes dataset type in run name)
- **TensorBoard**: Local logging and visualizations
- **Generation Examples**: Printed during training to monitor progress
- **Length-wise Results**: Detailed breakdown by sequence length

## Configuration

Key parameters in `config.py`:
- Model size: 6 layers, 8 heads, 512 dimensions (~19M parameters)
- Training: 5000 samples, batch size 32
- **Dataset options**: `counting_config_flat` vs `counting_config_biased`
- Length generalization: Tests on 70-150 token sequences

## Reproducing Paper Results

### Recommended Workflow
1. **Generate flat distribution data**: `python data_generation.py` (creates `data_flat/`)
2. **Train with flat config**: `python train_lightning.py --config counting_config_flat`
3. **Analyze results**: Check `length_generalization_results.json` in output directory
4. **Compare distributions**: Train both configs and compare generalization curves

### Research Questions
- How does training data distribution affect length generalization?
- What is the critical sequence length where generalization breaks down?
- Can the model learn the recursive counting algorithm from flat vs biased data?

## Debug Tools

See `debug/` folder for:
- Functionality tests
- Quick training verification  
- Accuracy calculation debugging
- Wandb setup utilities

## Paper Reference

Based on "What Algorithms can Transformers Learn? A Study in Length Generalization" - focuses on the counting task as an example of easily learnable algorithms that exhibit strong length generalization. 