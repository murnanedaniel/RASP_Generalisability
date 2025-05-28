"""
Training script for RASP generalization experiments.
"""

import os
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
import numpy as np
from tqdm import tqdm
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not available. Install with: pip install wandb")
from typing import Dict, List

from config import counting_config, Config
from model import CountingTransformer
from data_generation import load_dataset
from utils import (
    Tokenizer, collate_fn, calculate_sequence_accuracy, 
    count_parameters, get_device, set_seed, save_results
)

class CountingDataset(Dataset):
    """Dataset class for counting task"""
    
    def __init__(self, samples: List[dict], tokenizer: Tokenizer):
        self.samples = samples
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def create_dataloader(samples: List[dict], tokenizer: Tokenizer, 
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
        num_workers=0,  # Set to 0 for compatibility
        pin_memory=True if torch.cuda.is_available() else False
    )

def evaluate_model(model: CountingTransformer, dataloader: DataLoader, 
                  tokenizer: Tokenizer, device: torch.device, 
                  max_new_tokens: int = 100) -> Dict[str, float]:
    """Evaluate model on a dataset"""
    model.eval()
    
    total_loss = 0.0
    total_exact_match = 0.0
    total_token_accuracy = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Prepare labels (same as input_ids for autoregressive training)
            labels = input_ids.clone()
            labels[~attention_mask] = -100  # Ignore padding tokens in loss
            
            # Forward pass
            logits, loss = model(input_ids, attention_mask, labels)
            
            if loss is not None:
                total_loss += loss.item()
            
            # Calculate accuracy metrics (same as in train_epoch)
            predicted_ids = torch.argmax(logits, dim=-1)
            
            # Create target tokens (what the model should predict at each position)
            # At position i, model should predict input_ids[i+1]
            batch_size, seq_len = input_ids.shape
            
            # Targets are input_ids shifted left by 1
            targets = input_ids[:, 1:]  # Remove first token
            predictions = predicted_ids[:, :-1]  # Remove last prediction (no target for it)
            
            # Create mask for valid positions (non-padding)
            target_mask = attention_mask[:, 1:]  # Corresponding mask for targets
            
            # Calculate token accuracy
            correct_tokens = (predictions == targets) & target_mask
            total_valid_tokens = target_mask.sum()
            token_accuracy = correct_tokens.sum().float() / total_valid_tokens.float() if total_valid_tokens > 0 else 0.0
            
            # For sequence accuracy, check if entire sequence prediction is correct
            sequence_matches = 0
            for i in range(batch_size):
                seq_len_i = attention_mask[i].sum().item()
                if seq_len_i > 1:  # Need at least 2 tokens (one to predict from)
                    pred_seq = predictions[i][:seq_len_i-1]
                    target_seq = targets[i][:seq_len_i-1]
                    if torch.equal(pred_seq, target_seq):
                        sequence_matches += 1
            
            exact_match = sequence_matches / batch_size
            
            total_exact_match += exact_match
            total_token_accuracy += token_accuracy.item()
            num_batches += 1
    
    # Calculate averages
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_exact_match = total_exact_match / num_batches if num_batches > 0 else 0.0
    avg_token_accuracy = total_token_accuracy / num_batches if num_batches > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'exact_match': avg_exact_match,
        'token_accuracy': avg_token_accuracy
    }

def evaluate_generation(model: CountingTransformer, samples: List[dict], 
                       tokenizer: Tokenizer, device: torch.device, 
                       max_new_tokens: int = 100, show_examples: bool = False) -> Dict[str, float]:
    """
    Evaluate model using generation (more realistic evaluation)
    
    Args:
        model: Model to evaluate
        samples: List of samples to evaluate on
        tokenizer: Tokenizer instance
        device: Device to run on
        max_new_tokens: Maximum tokens to generate
        show_examples: Whether to print example generations
        
    Returns:
        Dictionary with evaluation metrics and examples
    """
    model.eval()
    
    correct_sequences = 0
    total_sequences = len(samples)
    examples = []
    
    with torch.no_grad():
        for i, sample in enumerate(tqdm(samples, desc="Generation eval", leave=False)):
            sequence = sample['sequence']
            tokens = sequence.split()
            
            # Find the separator '>' to split prompt and target
            separator_idx = tokens.index('>')
            prompt_tokens = tokens[:separator_idx + 1]  # Include the '>'
            target_tokens = tokens[separator_idx + 1:]   # Exclude the '>'
            
            # Encode prompt
            prompt = " ".join(prompt_tokens)
            prompt_ids = torch.tensor([tokenizer.encode_sequence(prompt)], device=device)
            
            # Generate
            generated_ids = model.generate(
                prompt_ids, 
                max_new_tokens=max_new_tokens,
                temperature=1.0,  # Use sampling for more realistic evaluation
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Extract generated part
            generated_part = generated_ids[0][len(prompt_ids[0]):]
            generated_tokens = tokenizer.decode(generated_part.cpu().tolist())
            
            # Remove EoS if present and clean up
            generated_text = " ".join(generated_tokens)
            if "EoS" in generated_text:
                generated_text = generated_text.split("EoS")[0].strip()
            
            target_text = " ".join(target_tokens[:-1])  # Remove EoS from target
            
            # Check if generated matches target
            is_correct = generated_text.strip() == target_text.strip()
            if is_correct:
                correct_sequences += 1
            
            # Collect examples for the first few samples or if showing examples
            if show_examples and (i < 5 or (i < 20 and not is_correct)):
                examples.append({
                    'prompt': prompt,
                    'target': target_text,
                    'generated': generated_text.strip(),
                    'correct': is_correct,
                    'full_sequence': sequence
                })
    
    # Print examples if requested
    if show_examples and examples:
        print(f"\n📝 Generation Examples (showing first few):")
        print("-" * 80)
        for i, ex in enumerate(examples[:8]):  # Show up to 8 examples
            status = "✅" if ex['correct'] else "❌"
            print(f"Example {i+1} {status}")
            print(f"  Prompt:    {ex['prompt']}")
            print(f"  Target:    {ex['target']}")
            print(f"  Generated: {ex['generated']}")
            if not ex['correct']:
                print(f"  Expected:  {ex['full_sequence']}")
            print()
    
    result = {
        'exact_match': correct_sequences / total_sequences,
        'total_samples': total_sequences,
        'correct_samples': correct_sequences
    }
    
    if examples:
        result['examples'] = examples[:10]  # Keep first 10 examples
    
    return result

def train_epoch(model: CountingTransformer, dataloader: DataLoader, 
               optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler,
               device: torch.device, tokenizer: Tokenizer, grad_clip_norm: float = 1.0,
               show_examples: bool = False, epoch: int = 0) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    
    # Show training examples at the start of epoch
    generation_exact_matches = 0
    total_generation_examples = 0
    
    if show_examples:
        print(f"\n📝 TRAINING EXAMPLES - Epoch {epoch + 1}")
        print("-" * 80)
        
        # Get a few samples for demonstration
        demo_samples = []
        for i, batch in enumerate(dataloader):
            if i >= 2:  # Get samples from first 2 batches
                break
            sequences = batch['sequences']
            for j, seq in enumerate(sequences):
                if len(demo_samples) >= 10:
                    break
                demo_samples.append({'sequence': seq})
            if len(demo_samples) >= 10:
                break
        
        # Generate examples with current model
        model.eval()
        with torch.no_grad():
            for i, sample in enumerate(demo_samples):
                sequence = sample['sequence']
                tokens = sequence.split()
                
                # Find the separator '>' to split prompt and target
                if '>' in tokens:
                    separator_idx = tokens.index('>')
                    prompt_tokens = tokens[:separator_idx + 1]  # Include the '>'
                    target_tokens = tokens[separator_idx + 1:]   # Exclude the '>'
                    
                    # Encode prompt
                    prompt = " ".join(prompt_tokens)
                    prompt_ids = torch.tensor([tokenizer.encode_sequence(prompt)], device=device)
                    
                    # Generate
                    try:
                        generated_ids = model.generate(
                            prompt_ids, 
                            max_new_tokens=50,
                            temperature=1.0,
                            eos_token_id=tokenizer.eos_token_id
                        )
                        
                        # Extract generated part
                        generated_part = generated_ids[0][len(prompt_ids[0]):]
                        generated_tokens = tokenizer.decode(generated_part.cpu().tolist())
                        
                        # Remove EoS if present and clean up
                        generated_text = " ".join(generated_tokens)
                        if "EoS" in generated_text:
                            generated_text = generated_text.split("EoS")[0].strip()
                        
                        target_text = " ".join(target_tokens[:-1]) if target_tokens and target_tokens[-1] == "EoS" else " ".join(target_tokens)
                        
                        # Check if correct
                        is_correct = generated_text.strip() == target_text.strip()
                        if is_correct:
                            generation_exact_matches += 1
                        total_generation_examples += 1
                        
                        status = "✅" if is_correct else "❌"
                        
                        print(f"Ex {i+1:2d} {status}")
                        print(f"  Prompt:    {prompt}")
                        print(f"  Target:    {target_text}")
                        print(f"  Generated: {generated_text.strip()}")
                        print()
                        
                    except Exception as e:
                        print(f"Ex {i+1:2d} ❌ (Generation failed: {str(e)[:50]})")
                        print(f"  Prompt: {prompt}")
                        print()
                        total_generation_examples += 1
        
        model.train()  # Switch back to training mode
        
        # Print generation accuracy summary
        if total_generation_examples > 0:
            gen_accuracy = generation_exact_matches / total_generation_examples
            print(f"🎯 TRAINING GENERATION ACCURACY: {generation_exact_matches}/{total_generation_examples} = {gen_accuracy:.3f}")
        print("-" * 80)
    
    total_loss = 0.0
    total_exact_match = 0.0
    total_token_accuracy = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Prepare labels
        labels = input_ids.clone()
        labels[~attention_mask] = -100
        
        # Forward pass
        logits, loss = model(input_ids, attention_mask, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        
        optimizer.step()
        scheduler.step()
        
        # Calculate metrics
        with torch.no_grad():
            predicted_ids = torch.argmax(logits, dim=-1)
            
            # For next-token prediction, calculate token-level accuracy
            # Compare predictions with the actual next tokens (input_ids shifted)
            
            # Create target tokens (what the model should predict at each position)
            # At position i, model should predict input_ids[i+1]
            batch_size, seq_len = input_ids.shape
            
            # Targets are input_ids shifted left by 1
            targets = input_ids[:, 1:]  # Remove first token
            predictions = predicted_ids[:, :-1]  # Remove last prediction (no target for it)
            
            # Create mask for valid positions (non-padding)
            target_mask = attention_mask[:, 1:]  # Corresponding mask for targets
            
            # Calculate token accuracy
            correct_tokens = (predictions == targets) & target_mask
            total_valid_tokens = target_mask.sum()
            token_accuracy = correct_tokens.sum().float() / total_valid_tokens.float() if total_valid_tokens > 0 else 0.0
            
            # For sequence accuracy, check if entire sequence prediction is correct
            # This is more approximate since we're using teacher forcing
            sequence_matches = 0
            for i in range(batch_size):
                seq_len_i = attention_mask[i].sum().item()
                if seq_len_i > 1:  # Need at least 2 tokens (one to predict from)
                    pred_seq = predictions[i][:seq_len_i-1]
                    target_seq = targets[i][:seq_len_i-1]
                    if torch.equal(pred_seq, target_seq):
                        sequence_matches += 1
            
            exact_match = sequence_matches / batch_size
        
        total_loss += loss.item()
        total_exact_match += exact_match
        total_token_accuracy += token_accuracy.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'exact_match': f"{exact_match:.3f}",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}"
        })
    
    return {
        'loss': total_loss / num_batches,
        'exact_match': total_exact_match / num_batches,
        'token_accuracy': total_token_accuracy / num_batches,
        'generation_exact_match': generation_exact_matches / total_generation_examples if total_generation_examples > 0 else 0.0,
        'generation_samples': total_generation_examples
    }

def main():
    parser = argparse.ArgumentParser(description="Train counting transformer")
    parser.add_argument("--config", type=str, default="counting_config", 
                       help="Configuration name")
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Data directory")
    parser.add_argument("--output_dir", type=str, default="output",
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Resume training from checkpoint")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Load configuration
    if args.config == "counting_config":
        config = counting_config
    else:
        raise ValueError(f"Unknown config: {args.config}")
    
    # Setup device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    train_samples = load_dataset(os.path.join(args.data_dir, "train.json"))
    val_samples = load_dataset(os.path.join(args.data_dir, "val.json"))
    test_samples = load_dataset(os.path.join(args.data_dir, "test_length_gen.json"))
    
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
    
    # Create model
    print("Creating model...")
    model = CountingTransformer(config).to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Create optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Calculate total training steps
    total_steps = len(train_dataloader) * config.training.max_epochs
    scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=config.training.warmup_steps
    )
    
    # Initialize wandb if enabled and available
    if config.training.use_wandb and WANDB_AVAILABLE:
        try:
            wandb.init(
                project=config.training.wandb_project,
                config=config.__dict__,
                name=f"counting-transformer-{args.seed}",
                mode="online"  # Try online first
            )
            print("✅ wandb tracking enabled")
        except Exception as e:
            print(f"⚠️  wandb online initialization failed: {e}")
            try:
                # Fallback to offline mode
                wandb.init(
                    project=config.training.wandb_project,
                    config=config.__dict__,
                    name=f"counting-transformer-{args.seed}",
                    mode="offline"
                )
                print("✅ wandb tracking enabled (offline mode)")
            except Exception as e2:
                print(f"⚠️  wandb offline initialization also failed: {e2}")
                print("   Continuing without wandb tracking...")
                config.training.use_wandb = False
    elif config.training.use_wandb and not WANDB_AVAILABLE:
        print("⚠️  wandb requested but not available. Install with: pip install wandb")
        print("   Continuing without wandb tracking...")
        config.training.use_wandb = False
    
    # Training loop
    print("Starting training...")
    # NOTE: We use generation accuracy as primary metric because:
    # 1. It reflects real-world performance (autoregressive generation)
    # 2. It's what the paper evaluates for length generalization
    # 3. Teacher-forcing accuracy can be misleading during training
    best_val_generation_accuracy = 0.0  # Changed from best_val_exact_match
    best_val_loss = float('inf')
    patience_counter = 0
    step = 0
    model_saved = False
    
    for epoch in range(config.training.max_epochs):
        print(f"\nEpoch {epoch + 1}/{config.training.max_epochs}")
        
        # Train for one epoch
        train_metrics = train_epoch(
            model, train_dataloader, optimizer, scheduler, device, tokenizer,
            config.training.grad_clip_norm, 
            show_examples=(epoch < 5 or epoch % 10 == 0),  # Show examples for first 5 epochs and every 10th epoch
            epoch=epoch
        )
        
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Teacher-Force Exact Match: {train_metrics['exact_match']:.3f}, "
              f"Generation Exact Match: {train_metrics['generation_exact_match']:.3f} "
              f"({train_metrics['generation_samples']} samples)")
        
        # Show validation examples
        if epoch < 5 or epoch % 10 == 0:  # Show examples for first 5 epochs and every 10th epoch
            print(f"\n📝 VALIDATION EXAMPLES - Epoch {epoch + 1}")
            print("-" * 80)
            
            # Get a few validation samples for demonstration
            val_demo_samples = val_samples[:10]  # First 10 validation samples
            
            result = evaluate_generation(
                model, val_demo_samples, tokenizer, device, 
                max_new_tokens=50, show_examples=True
            )
            
            # Extract exact match count from result
            val_generation_accuracy = result['exact_match']
            val_correct_samples = result['correct_samples']
            val_total_samples = result['total_samples']
            
            print(f"🎯 VALIDATION GENERATION ACCURACY: {val_correct_samples}/{val_total_samples} = {val_generation_accuracy:.3f}")
            print("-" * 80)
        
        # Evaluate on validation set (teacher-forcing)
        val_metrics = evaluate_model(model, val_dataloader, tokenizer, device)
        
        # Evaluate on validation set (generation-based) - this is our primary metric
        val_generation_result = evaluate_generation(
            model, val_samples[:100], tokenizer, device, 
            max_new_tokens=100, show_examples=False
        )
        val_generation_accuracy = val_generation_result['exact_match']
        
        print(f"Val - Loss: {val_metrics['loss']:.4f}, "
              f"Teacher-Force Exact Match: {val_metrics['exact_match']:.3f}, "
              f"Generation Exact Match: {val_generation_accuracy:.3f} "
              f"({val_generation_result['correct_samples']}/{val_generation_result['total_samples']} samples)")
        
        # Log metrics
        if config.training.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'train_teacher_forcing_exact_match': train_metrics['exact_match'],
                'train_generation_exact_match': train_metrics['generation_exact_match'],
                'val_loss': val_metrics['loss'],
                'val_teacher_forcing_exact_match': val_metrics['exact_match'],
                'val_generation_exact_match': val_generation_accuracy,
                'learning_rate': scheduler.get_last_lr()[0]
            })
        
        # Save checkpoint if best model (USE GENERATION ACCURACY as primary metric)
        save_model = False
        if val_generation_accuracy > best_val_generation_accuracy:
            best_val_generation_accuracy = val_generation_accuracy
            best_val_loss = val_metrics['loss']  # Update loss too for consistency
            save_model = True
        
        if save_model:
            patience_counter = 0
            model_saved = True
            
            # Save best model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config,
                'tokenizer_state': {
                    'token_to_id': tokenizer.token_to_id,
                    'id_to_token': tokenizer.id_to_token
                },
                'best_val_generation_accuracy': best_val_generation_accuracy,
                'best_val_loss': best_val_loss
            }
            
            torch.save(checkpoint, os.path.join(args.output_dir, "checkpoints", "best_model.pt"))
            print(f"💾 Saved best model with validation generation accuracy: {best_val_generation_accuracy:.3f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.training.early_stopping_patience:
            print(f"Early stopping after {patience_counter} epochs without improvement")
            break
        
        # Evaluate length generalization periodically
        if (epoch + 1) % 5 == 0:
            print("\nEvaluating length generalization...")
            
            # Group test samples by length
            length_groups = {}
            for sample in test_samples:
                length = sample['length']
                if length not in length_groups:
                    length_groups[length] = []
                length_groups[length].append(sample)
            
            # Evaluate each length
            length_results = {}
            for length, samples in sorted(length_groups.items()):
                if len(samples) > 10:  # Only evaluate if we have enough samples
                    eval_samples = samples[:50]  # Limit for speed
                    show_examples = (length == sorted(length_groups.keys())[0])  # Show examples for first length only
                    result = evaluate_generation(model, eval_samples, tokenizer, device, show_examples=show_examples)
                    length_results[length] = result['exact_match']
                    print(f"Length {length}: {result['exact_match']:.3f} "
                          f"({result['correct_samples']}/{result['total_samples']})")
            
            # Log length generalization results
            if config.training.use_wandb:
                wandb_log = {}
                for length, acc in length_results.items():
                    wandb_log[f'length_gen/{length}'] = acc
                wandb.log(wandb_log)
    
    # Final evaluation
    print("\n" + "="*50)
    print("FINAL EVALUATION")
    print("="*50)
    
    # Load best model if one was saved
    checkpoint_path = os.path.join(args.output_dir, "checkpoints", "best_model.pt")
    if model_saved and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded best model for final evaluation")
    else:
        print("No model checkpoint found, using current model state")
    
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
    
    final_results = {}
    for length, samples in sorted(length_groups.items()):
        show_examples = (length == sorted(length_groups.items())[0][0])  # Show examples for first length only
        result = evaluate_generation(model, samples, tokenizer, device, show_examples=show_examples)
        final_results[length] = {
            'exact_match': result['exact_match'],
            'total_samples': result['total_samples'],
            'correct_samples': result['correct_samples']
        }
        
        print(f"Length {length:3d}: {result['exact_match']:.3f} "
              f"({result['correct_samples']}/{result['total_samples']})")
    
    # Save final results
    save_results(final_results, os.path.join(args.output_dir, "length_generalization_results.json"))
    
    # Cleanup wandb
    if config.training.use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    print(f"\nTraining completed! Best validation generation accuracy: {best_val_generation_accuracy:.3f}")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 