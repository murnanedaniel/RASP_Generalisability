#!/usr/bin/env python3
"""
Quick evaluation script for testing length generalization of the counting model.
"""

import torch
import os
from typing import List, Dict
from tqdm import tqdm

from model import CountingTransformer
from data_generation import generate_length_specific_dataset
from utils import Tokenizer, get_device

def load_lightning_checkpoint(checkpoint_path: str, device: torch.device):
    """Load model from Lightning checkpoint"""
    print(f"Loading Lightning checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config and create tokenizer
    config = checkpoint['hyper_parameters']['config']
    tokenizer = Tokenizer()
    
    # Create model
    model = CountingTransformer(config).to(device)
    
    # Handle Lightning state dict - remove "model." prefix
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            new_key = key[6:]  # Remove "model." prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    # Load state dict
    model.load_state_dict(new_state_dict)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Training length: {config.training.max_train_length}")
    
    return model, tokenizer, config

def evaluate_length(model: CountingTransformer, tokenizer: Tokenizer, 
                   device: torch.device, length: int, num_samples: int = 50) -> Dict:
    """Evaluate model on a specific sequence length"""
    
    # Generate test samples
    samples = generate_length_specific_dataset(
        target_lengths=[length],
        samples_per_length=num_samples,
        min_number=0,
        max_number=min(100, length + 20),  # Adapt range to length
        seed=42 + length  # Different seed per length
    )
    
    correct = 0
    total = len(samples)
    examples = []
    
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(tqdm(samples, desc=f"Length {length}", leave=False)):
            sequence = sample['sequence']
            tokens = sequence.split()
            
            # Split prompt and target
            separator_idx = tokens.index('>')
            prompt_tokens = tokens[:separator_idx + 1]
            target_tokens = tokens[separator_idx + 1:-1]  # Remove EoS
            
            # Encode and generate
            prompt = " ".join(prompt_tokens)
            prompt_ids = torch.tensor([tokenizer.encode_sequence(prompt)], device=device)
            
            generated_ids = model.generate(
                prompt_ids, 
                max_new_tokens=length + 10,
                temperature=0.1,  # Low temperature for consistency
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Decode generated part
            generated_part = generated_ids[0][len(prompt_ids[0]):]
            generated_tokens = tokenizer.decode(generated_part.cpu().tolist())
            
            # Clean up generation
            generated_text = " ".join(generated_tokens)
            if "EoS" in generated_text:
                generated_text = generated_text.split("EoS")[0].strip()
            
            target_text = " ".join(target_tokens)
            
            # Check correctness
            is_correct = generated_text.strip() == target_text.strip()
            if is_correct:
                correct += 1
            
            # Store examples (first 3 and first 3 failures)
            if i < 3 or (not is_correct and len(examples) < 6):
                examples.append({
                    'prompt': prompt,
                    'target': target_text,
                    'generated': generated_text.strip(),
                    'correct': is_correct
                })
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'examples': examples
    }

def main():
    # Configuration
    checkpoint_path = "output/checkpoints/best-epoch=41-val_generation_exact_match=0.990.ckpt"
    test_lengths = [20, 30, 40, 50, 60, 70, 80, 90, 100, 120]  # Mix of in-dist and OOD
    samples_per_length = 30  # Smaller for quick testing
    
    # Setup
    device = get_device()
    print(f"Device: {device}")
    
    # Load model
    model, tokenizer, config = load_lightning_checkpoint(checkpoint_path, device)
    max_train_length = config.training.max_train_length
    
    print(f"\nTesting length generalization:")
    print(f"Training range: 6-{max_train_length} tokens")
    print(f"Test lengths: {test_lengths}")
    print("-" * 60)
    
    results = {}
    
    # Test each length
    for length in test_lengths:
        result = evaluate_length(model, tokenizer, device, length, samples_per_length)
        results[length] = result
        
        # Display result
        status = "✓ TRAIN" if length <= max_train_length else "✗ OOD"
        print(f"Length {length:3d} ({status}): {result['accuracy']:.3f} ({result['correct']}/{result['total']})")
        
        # Show a few examples for OOD lengths or failures
        if length > max_train_length or result['accuracy'] < 0.8:
            print("  Examples:")
            for i, ex in enumerate(result['examples'][:2]):
                status_icon = "✅" if ex['correct'] else "❌"
                print(f"    {status_icon} {ex['prompt']} → {ex['generated']}")
                if not ex['correct']:
                    print(f"      Expected: {ex['target']}")
            print()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    in_dist = [(l, r['accuracy']) for l, r in results.items() if l <= max_train_length]
    ood = [(l, r['accuracy']) for l, r in results.items() if l > max_train_length]
    
    if in_dist:
        avg_in_dist = sum(acc for _, acc in in_dist) / len(in_dist)
        print(f"📊 In-distribution (≤{max_train_length}): {avg_in_dist:.3f}")
    
    if ood:
        avg_ood = sum(acc for _, acc in ood) / len(ood)
        print(f"🚀 Out-of-distribution (>{max_train_length}): {avg_ood:.3f}")
        
        excellent_ood = sum(1 for _, acc in ood if acc > 0.95)
        good_ood = sum(1 for _, acc in ood if acc > 0.8)
        
        print(f"   • >95% accuracy: {excellent_ood}/{len(ood)} lengths")
        print(f"   • >80% accuracy: {good_ood}/{len(ood)} lengths")
        
        if avg_ood > 0.9:
            print("✅ STRONG length generalization!")
        elif avg_ood > 0.7:
            print("🔶 MODERATE length generalization")
        else:
            print("❌ POOR length generalization")
    
    print(f"\nCheckpoint: {checkpoint_path}")

if __name__ == "__main__":
    main() 