"""
PyTorch Lightning training script for RASP generalization experiments.
"""

import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from torch.utils.data import DataLoader, Dataset

from config import counting_config
from lightning_model import CountingLightningModule
from data_generation import load_dataset
from utils import Tokenizer, collate_fn, get_device, set_seed


class CountingDataset(Dataset):
    """Dataset class for counting task"""
    
    def __init__(self, samples: list, tokenizer: Tokenizer):
        self.samples = samples
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def create_dataloader(samples: list, tokenizer: Tokenizer, 
                     batch_size: int, shuffle: bool = True) -> DataLoader:
    """Create DataLoader for training/evaluation"""
    dataset = CountingDataset(samples, tokenizer)
    
    def collate_wrapper(batch):
        return collate_fn(batch, tokenizer)
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_wrapper,
        num_workers=4,  # Lightning can handle multiple workers better
        pin_memory=True
    )


class GenerationExamplesCallback(pl.Callback):
    """Callback to show generation examples during training"""
    
    def __init__(self, train_samples: list, val_samples: list, show_every_n_epochs: int = 5):
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.show_every_n_epochs = show_every_n_epochs
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Show training examples at epoch end"""
        if trainer.current_epoch < 5 or trainer.current_epoch % self.show_every_n_epochs == 0:
            print(f"\n🏋️ TRAINING EXAMPLES - Epoch {trainer.current_epoch}")
            correct, total = pl_module.show_generation_examples(self.train_samples, num_examples=10)
            
            # Log to wandb
            if trainer.logger and hasattr(trainer.logger, 'experiment'):
                trainer.logger.experiment.log({
                    'train_generation_examples_accuracy': correct / total if total > 0 else 0.0,
                    'epoch': trainer.current_epoch
                })
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Show validation examples at epoch end"""
        if trainer.current_epoch < 5 or trainer.current_epoch % self.show_every_n_epochs == 0:
            print(f"\n🎯 VALIDATION EXAMPLES - Epoch {trainer.current_epoch}")
            correct, total = pl_module.show_generation_examples(self.val_samples, num_examples=10)


def main():
    parser = argparse.ArgumentParser(description="Train counting transformer with PyTorch Lightning")
    parser.add_argument("--config", type=str, default="counting_config", 
                       help="Configuration name: counting_config, counting_config_biased, counting_config_flat")
    parser.add_argument("--data_dir", type=str, default=None,
                       help="Data directory (overrides config)")
    parser.add_argument("--output_dir", type=str, default="output",
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Resume training from checkpoint")
    parser.add_argument("--fast_dev_run", action="store_true",
                       help="Fast dev run for debugging")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    pl.seed_everything(args.seed)
    
    # Load configuration
    if args.config == "counting_config":
        config = counting_config
    elif args.config == "counting_config_biased":
        from config import counting_config_biased
        config = counting_config_biased
        print("🔸 Using BIASED length distribution (original data)")
    elif args.config == "counting_config_flat":
        from config import counting_config_flat
        config = counting_config_flat
        print("🔸 Using FLAT length distribution (new corrected data)")
    else:
        raise ValueError(f"Unknown config: {args.config}")
    
    # Override data_dir if provided
    if args.data_dir is not None:
        config.training.data_dir = args.data_dir
        print(f"🔸 Overriding data directory: {args.data_dir}")
    
    print(f"📁 Data directory: {config.training.data_dir}")
    print(f"📊 Dataset type: {config.training.dataset_type}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    train_samples = load_dataset(os.path.join(config.training.data_dir, "train.json"))
    val_samples = load_dataset(os.path.join(config.training.data_dir, "val.json"))
    test_samples = load_dataset(os.path.join(config.training.data_dir, "test_length_gen.json"))
    
    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")
    
    # Create tokenizer
    tokenizer = Tokenizer()
    
    # Update config with actual vocab size
    config.model.vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create dataloaders
    train_dataloader = create_dataloader(
        train_samples, tokenizer, config.training.batch_size, shuffle=True
    )
    val_dataloader = create_dataloader(
        val_samples, tokenizer, config.training.batch_size, shuffle=False
    )
    
    # Create Lightning module
    print("Creating Lightning module...")
    model = CountingLightningModule(
        config=config,
        tokenizer=tokenizer,
        val_samples=val_samples,
        generation_eval_samples=100
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup loggers
    loggers = []
    
    # Wandb logger
    if config.training.use_wandb:
        try:
            wandb_logger = WandbLogger(
                project=config.training.wandb_project,
                name=f"counting-transformer-{config.training.dataset_type}-{args.seed}",
                save_dir=args.output_dir,
                version=f"{config.training.dataset_type}-seed-{args.seed}"
            )
            loggers.append(wandb_logger)
            print("✅ Wandb logging enabled")
        except Exception as e:
            print(f"⚠️  Wandb setup failed: {e}")
    
    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name="lightning_logs",
        version=f"seed-{args.seed}"
    )
    loggers.append(tb_logger)
    print("✅ TensorBoard logging enabled")
    
    # Setup callbacks
    callbacks = []
    
    # Model checkpoint callback (saves best model based on generation accuracy)
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename="best-{epoch}-{val_generation_exact_match:.3f}",
        monitor="val_generation_exact_match",
        mode="max",
        save_top_k=1,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping based on generation accuracy
    early_stopping_callback = EarlyStopping(
        monitor="val_generation_exact_match",
        mode="max",
        patience=config.training.early_stopping_patience,
        verbose=True
    )
    callbacks.append(early_stopping_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    
    # Generation examples callback
    examples_callback = GenerationExamplesCallback(
        train_samples=train_samples,
        val_samples=val_samples,
        show_every_n_epochs=5
    )
    callbacks.append(examples_callback)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator="auto",  # Automatically choose GPU/CPU
        devices="auto",      # Use all available devices
        precision="16-mixed" if torch.cuda.is_available() else "32",  # Use mixed precision on GPU
        gradient_clip_val=config.training.grad_clip_norm,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=50,
        val_check_interval=1.0,  # Validate every epoch
        enable_progress_bar=True,
        enable_model_summary=True,
        fast_dev_run=args.fast_dev_run,  # For debugging
        deterministic=True  # For reproducibility
    )
    
    # Print training info
    print(f"\n🚀 Starting Lightning training...")
    print(f"Device: {trainer.strategy.root_device}")
    print(f"Precision: {trainer.precision}")
    print(f"Max epochs: {config.training.max_epochs}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    
    # Train the model
    if args.resume_from:
        trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=args.resume_from)
    else:
        trainer.fit(model, train_dataloader, val_dataloader)
    
    # Load best model for final evaluation
    if not args.fast_dev_run:
        print("\n" + "="*50)
        print("FINAL EVALUATION")
        print("="*50)
        
        # Load best checkpoint
        best_model = CountingLightningModule.load_from_checkpoint(
            checkpoint_callback.best_model_path,
            config=config,
            tokenizer=tokenizer,
            val_samples=val_samples
        )
        best_model.eval()
        
        print(f"Loaded best model from: {checkpoint_callback.best_model_path}")
        print(f"Best validation generation accuracy: {checkpoint_callback.best_model_score:.3f}")
        
        # Comprehensive length generalization evaluation
        print("\nLength Generalization Results:")
        print("-" * 40)
        
        # Group test samples by length
        length_groups = {}
        for sample in test_samples:
            length = sample['length']
            if length not in length_groups:
                length_groups[length] = []
            length_groups[length].append(sample)
        
        from evaluate import evaluate_generation
        
        final_results = {}
        for length, samples in sorted(length_groups.items()):
            show_examples = (length == sorted(length_groups.items())[0][0])  # Show examples for first length only
            result = evaluate_generation(best_model.model, samples, tokenizer, trainer.strategy.root_device, show_examples=show_examples)
            final_results[length] = {
                'exact_match': result['exact_match'],
                'total_samples': result['total_samples'],
                'correct_samples': result['correct_samples']
            }
            
            print(f"Length {length:3d}: {result['exact_match']:.3f} "
                  f"({result['correct_samples']}/{result['total_samples']})")
        
        # Save final results
        import json
        with open(os.path.join(args.output_dir, "length_generalization_results.json"), 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n✅ Training completed!")
        print(f"📊 Best validation generation accuracy: {checkpoint_callback.best_model_score:.3f}")
        print(f"💾 Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 