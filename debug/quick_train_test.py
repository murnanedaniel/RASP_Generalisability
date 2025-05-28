#!/usr/bin/env python3
"""
Quick training test - train for just a few epochs and show examples.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from tqdm import tqdm

from config import counting_config
from model import CountingTransformer
from data_generation import load_dataset, generate_dataset
from utils import Tokenizer, collate_fn, get_device, set_seed
from train import CountingDataset, evaluate_generation

def main():
    print("🚀 Quick Training Test")
    print("=" * 50)
    
    # Setup
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")
    
    # Create small config
    config = counting_config
    config.model.vocab_size = 106
    config.model.n_layers = 3  # Small model
    config.model.d_model = 256
    config.model.n_heads = 4
    config.model.d_ff = 1024
    
    # Create tokenizer and model
    tokenizer = Tokenizer()
    model = CountingTransformer(config).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Generate small dataset
    print("\nGenerating training data...")
    train_samples = generate_dataset(
        num_samples=1000,
        max_length=40,
        min_number=0,
        max_number=20,
        seed=42
    )
    
    # Generate test samples
    test_samples = generate_dataset(
        num_samples=100,
        max_length=50,
        min_number=0,
        max_number=25,
        seed=43
    )
    
    print(f"Training samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")
    
    # Create dataloader
    train_dataloader = DataLoader(
        CountingDataset(train_samples, tokenizer),
        batch_size=16,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )
    
    # Test initial generation
    print("\n" + "="*50)
    print("INITIAL GENERATION (before training)")
    print("="*50)
    result = evaluate_generation(model, test_samples[:10], tokenizer, device, show_examples=True)
    print(f"Initial accuracy: {result['exact_match']:.3f}")
    
    # Train for a few epochs
    print("\n" + "="*50)
    print("TRAINING")
    print("="*50)
    
    model.train()
    for epoch in range(5):
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            labels = input_ids.clone()
            labels[~attention_mask] = -100
            
            optimizer.zero_grad()
            logits, loss = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")
    
    # Test after training
    print("\n" + "="*50)
    print("GENERATION AFTER TRAINING")
    print("="*50)
    result = evaluate_generation(model, test_samples[:10], tokenizer, device, show_examples=True)
    print(f"Final accuracy: {result['exact_match']:.3f}")
    
    print("\n🎉 Quick training test completed!")
    print(f"Loss should have decreased and examples should look better.")

if __name__ == "__main__":
    main() 