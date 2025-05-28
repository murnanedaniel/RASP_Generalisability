"""
PyTorch Lightning module for RASP generalization experiments.
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from typing import Dict, List, Optional, Any
import wandb

from model import CountingTransformer
from utils import Tokenizer, calculate_sequence_accuracy
from evaluate import evaluate_generation


class CountingLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for counting task.
    
    This module handles:
    - Training and validation steps
    - Generation-based evaluation
    - Optimizer and scheduler configuration
    - Logging to wandb and tensorboard
    """
    
    def __init__(
        self,
        config,
        tokenizer: Tokenizer,
        val_samples: List[dict] = None,
        generation_eval_samples: int = 100
    ):
        """
        Initialize Lightning module.
        
        Args:
            config: Configuration object
            tokenizer: Tokenizer instance
            val_samples: Validation samples for generation evaluation
            generation_eval_samples: Number of samples to use for generation evaluation
        """
        super().__init__()
        
        # Save config and tokenizer
        self.config = config
        self.tokenizer = tokenizer
        self.val_samples = val_samples or []
        self.generation_eval_samples = generation_eval_samples
        
        # Create model
        self.model = CountingTransformer(config)
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['tokenizer', 'val_samples'])
        
        # Metrics tracking
        self.best_val_generation_accuracy = 0.0
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass through the model"""
        return self.model(input_ids, attention_mask, labels)
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        # Prepare labels
        labels = input_ids.clone()
        labels[~attention_mask] = -100
        
        # Forward pass
        logits, loss = self.model(input_ids, attention_mask, labels)
        
        # Calculate teacher-forcing accuracy (for monitoring)
        with torch.no_grad():
            predicted_ids = torch.argmax(logits, dim=-1)
            
            # Teacher-forcing accuracy calculation
            batch_size, seq_len = input_ids.shape
            targets = input_ids[:, 1:]  # Next tokens
            predictions = predicted_ids[:, :-1]  # Remove last prediction
            target_mask = attention_mask[:, 1:]  # Mask for targets
            
            # Token accuracy
            correct_tokens = (predictions == targets) & target_mask
            total_valid_tokens = target_mask.sum()
            token_accuracy = correct_tokens.sum().float() / total_valid_tokens.float() if total_valid_tokens > 0 else 0.0
            
            # Sequence accuracy
            sequence_matches = 0
            for i in range(batch_size):
                seq_len_i = attention_mask[i].sum().item()
                if seq_len_i > 1:
                    pred_seq = predictions[i][:seq_len_i-1]
                    target_seq = targets[i][:seq_len_i-1]
                    if torch.equal(pred_seq, target_seq):
                        sequence_matches += 1
            exact_match = sequence_matches / batch_size
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_teacher_forcing_exact_match', exact_match, on_step=False, on_epoch=True)
        self.log('train_token_accuracy', token_accuracy, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        # Prepare labels
        labels = input_ids.clone()
        labels[~attention_mask] = -100
        
        # Forward pass
        logits, loss = self.model(input_ids, attention_mask, labels)
        
        # Calculate teacher-forcing accuracy
        with torch.no_grad():
            predicted_ids = torch.argmax(logits, dim=-1)
            
            batch_size, seq_len = input_ids.shape
            targets = input_ids[:, 1:]
            predictions = predicted_ids[:, :-1]
            target_mask = attention_mask[:, 1:]
            
            correct_tokens = (predictions == targets) & target_mask
            total_valid_tokens = target_mask.sum()
            token_accuracy = correct_tokens.sum().float() / total_valid_tokens.float() if total_valid_tokens > 0 else 0.0
            
            sequence_matches = 0
            for i in range(batch_size):
                seq_len_i = attention_mask[i].sum().item()
                if seq_len_i > 1:
                    pred_seq = predictions[i][:seq_len_i-1]
                    target_seq = targets[i][:seq_len_i-1]
                    if torch.equal(pred_seq, target_seq):
                        sequence_matches += 1
            exact_match = sequence_matches / batch_size
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_teacher_forcing_exact_match', exact_match, on_step=False, on_epoch=True)
        self.log('val_token_accuracy', token_accuracy, on_step=False, on_epoch=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch"""
        # Perform generation-based evaluation
        if self.val_samples and len(self.val_samples) > 0:
            # Use a subset for faster evaluation
            eval_samples = self.val_samples[:self.generation_eval_samples]
            
            # Evaluate generation
            result = evaluate_generation(
                self.model, eval_samples, self.tokenizer, self.device,
                max_new_tokens=100, show_examples=False
            )
            
            generation_accuracy = result['exact_match']
            
            # Log generation accuracy
            self.log('val_generation_exact_match', generation_accuracy, on_epoch=True, prog_bar=True)
            
            # Update best accuracy
            if generation_accuracy > self.best_val_generation_accuracy:
                self.best_val_generation_accuracy = generation_accuracy
                self.log('best_val_generation_accuracy', self.best_val_generation_accuracy, on_epoch=True)
            
            # Print progress
            if self.current_epoch % 5 == 0 or self.current_epoch < 5:
                correct_samples = result['correct_samples']
                total_samples = result['total_samples']
                print(f"\n📊 Epoch {self.current_epoch}: Generation accuracy: {generation_accuracy:.3f} "
                      f"({correct_samples}/{total_samples})")
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        optimizer = AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        # Linear warmup scheduler
        scheduler = {
            'scheduler': LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.config.training.warmup_steps
            ),
            'interval': 'step',
            'frequency': 1,
        }
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }
    
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50,
                 temperature: float = 1.0, eos_token_id: Optional[int] = None) -> torch.Tensor:
        """Generate tokens using the model"""
        return self.model.generate(input_ids, max_new_tokens, temperature, eos_token_id)
    
    def show_generation_examples(self, samples: List[dict], num_examples: int = 10):
        """Show generation examples during training"""
        self.eval()
        
        print(f"\n📝 GENERATION EXAMPLES - Epoch {self.current_epoch}")
        print("-" * 80)
        
        correct_count = 0
        total_count = 0
        
        with torch.no_grad():
            for i, sample in enumerate(samples[:num_examples]):
                sequence = sample['sequence']
                tokens = sequence.split()
                
                if '>' in tokens:
                    separator_idx = tokens.index('>')
                    prompt_tokens = tokens[:separator_idx + 1]
                    target_tokens = tokens[separator_idx + 1:]
                    
                    prompt = " ".join(prompt_tokens)
                    prompt_ids = torch.tensor([self.tokenizer.encode_sequence(prompt)], device=self.device)
                    
                    try:
                        generated_ids = self.generate(
                            prompt_ids,
                            max_new_tokens=50,
                            temperature=1.0,
                            eos_token_id=self.tokenizer.eos_token_id
                        )
                        
                        generated_part = generated_ids[0][len(prompt_ids[0]):]
                        generated_tokens = self.tokenizer.decode(generated_part.cpu().tolist())
                        
                        generated_text = " ".join(generated_tokens)
                        if "EoS" in generated_text:
                            generated_text = generated_text.split("EoS")[0].strip()
                        
                        target_text = " ".join(target_tokens[:-1]) if target_tokens and target_tokens[-1] == "EoS" else " ".join(target_tokens)
                        
                        is_correct = generated_text.strip() == target_text.strip()
                        if is_correct:
                            correct_count += 1
                        total_count += 1
                        
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
                        total_count += 1
        
        if total_count > 0:
            accuracy = correct_count / total_count
            print(f"🎯 Example accuracy: {correct_count}/{total_count} = {accuracy:.3f}")
        print("-" * 80)
        
        self.train()
        return correct_count, total_count 