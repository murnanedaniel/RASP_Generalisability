# Debug Scripts

This folder contains debugging and testing scripts used during development.

## Scripts

- **`debug_accuracy.py`** - Comprehensive accuracy calculation testing to understand teacher-forcing vs generation differences
- **`quick_test.py`** - Basic functionality test for all components
- **`quick_train_test.py`** - Quick training test with small dataset to verify learning
- **`setup_wandb.py`** - Wandb setup and authentication helper

## Usage

These scripts are primarily for development and debugging purposes. For normal training and evaluation, use the main scripts in the parent directory.

```bash
# Run quick functionality test
python debug/quick_test.py

# Run quick training test
python debug/quick_train_test.py

# Debug accuracy calculations
python debug/debug_accuracy.py

# Setup wandb
python debug/setup_wandb.py 