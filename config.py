"""
Configuration file for RASP generalization experiments.
Based on the paper "What Algorithms can Transformers Learn? A Study in Length Generalization"
"""

import dataclasses
from typing import Optional

@dataclasses.dataclass
class ModelConfig:
    """Transformer model configuration"""
    vocab_size: int = 100  # Will be set based on data
    max_seq_length: int = 200  # Maximum sequence length to support
    n_layers: int = 6  # Number of transformer layers
    n_heads: int = 8   # Number of attention heads
    d_model: int = 512 # Hidden dimension
    d_ff: int = 2048   # Feed-forward dimension
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    
@dataclasses.dataclass
class TrainingConfig:
    """Training configuration"""
    # Data
    max_train_length: int = 60  # Maximum training sequence length
    num_train_samples: int = 50000
    num_val_samples: int = 5000
    
    # Training
    batch_size: int = 32
    learning_rate: float = 5e-4  # Increase learning rate for faster convergence
    weight_decay: float = 1e-5
    max_epochs: int = 100
    warmup_steps: int = 500  # Reduce warmup steps
    grad_clip_norm: float = 1.0
    
    # Evaluation
    eval_every: int = 500  # Steps between evaluations
    save_every: int = 1000 # Steps between checkpoints
    early_stopping_patience: int = 20  # Increase patience
    
    # Length generalization evaluation
    eval_lengths: list = dataclasses.field(default_factory=lambda: [70, 80, 90, 100, 120, 150])
    
    # Logging
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    use_wandb: bool = True  # Enable wandb by default
    wandb_project: str = "rasp-generalization"
    
@dataclasses.dataclass
class CountingTaskConfig:
    """Configuration specific to the counting task"""
    min_number: int = 0
    max_number: int = 50  # Will be increased for longer sequences
    sos_token: str = "SoS"
    eos_token: str = "EoS"
    separator_token: str = ">"
    
# Combine all configs
@dataclasses.dataclass 
class Config:
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)
    task: CountingTaskConfig = dataclasses.field(default_factory=CountingTaskConfig)

# Default configuration
counting_config = Config() 