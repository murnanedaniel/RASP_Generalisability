"""
Evaluate length generalization using PyTorch Lightning checkpoint.
"""

import os
import argparse
import torch
import json
from typing import Dict, List

from config import counting_config
from lightning_model import CountingLightningModule
from data_generation import generate_length_specific_dataset
from utils import Tokenizer, get_device
from evaluate import evaluate_generation


def load_lightning_model(checkpoint_path: str, device: torch.device):
    """Load trained Lightning model from checkpoint"""
    print(f"Loading Lightning model from: {checkpoint_path}")
    
    # Load checkpoint to get hyperparameters
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create tokenizer
    tokenizer = Tokenizer()
    
    # Get config and update vocab_size from checkpoint
    config = counting_config
    if 'hyper_parameters' in checkpoint and 'config' in checkpoint['hyper_parameters']:
        saved_config = checkpoint['hyper_parameters']['config']
        if hasattr(saved_config, 'model') and hasattr(saved_config.model, 'vocab_size'):
            config.model.vocab_size = saved_config.model.vocab_size
            print(f"Using vocab_size from checkpoint: {config.model.vocab_size}")
    else:
        # Fallback: try to infer from model weights
        if 'state_dict' in checkpoint:
            for key in checkpoint['state_dict'].keys():
                if 'token_embedding.weight' in key:
                    vocab_size = checkpoint['state_dict'][key].shape[0]
                    config.model.vocab_size = vocab_size
                    print(f"Inferred vocab_size from weights: {vocab_size}")
                    break
    
    # Load Lightning module
    model = CountingLightningModule.load_from_checkpoint(
        checkpoint_path,
        config=config,
        tokenizer=tokenizer,
        val_samples=[]  # Not needed for evaluation
    )
    model.to(device)
    model.eval()
    
    print(f"Lightning model loaded successfully!")
    
    return model, tokenizer


def evaluate_length_generalization_lightning(model: CountingLightningModule, tokenizer: Tokenizer, 
                                           device: torch.device, max_train_length: int = 60,
                                           test_lengths: List[int] = None, 
                                           samples_per_length: int = 100) -> Dict[int, Dict]:
    """Comprehensive length generalization evaluation for Lightning model"""
    
    if test_lengths is None:
        test_lengths = [30, 40, 50, 60, 70, 80, 90, 100, 120, 150]
    
    print("Generating test data for length generalization...")
    
    # Generate test samples at specific lengths
    test_samples = generate_length_specific_dataset(
        target_lengths=test_lengths,
        samples_per_length=samples_per_length,
        min_number=0,
        max_number=100,
        seed=42
    )
    
    # Group by length
    length_groups = {}
    for sample in test_samples:
        length = sample['length']
        if length not in length_groups:
            length_groups[length] = []
        length_groups[length].append(sample)
    
    # Evaluate each length
    results = {}
    print(f"\nEvaluating length generalization (max train length: {max_train_length}):")
    print("="*70)
    
    for length in sorted(length_groups.keys()):
        samples = length_groups[length]
        
        print(f"Evaluating length {length} ({len(samples)} samples)...")
        
        # Use the underlying transformer model for evaluation
        result = evaluate_generation(
            model.model, samples, tokenizer, device, 
            max_new_tokens=length + 20,  # Give some extra tokens
            show_examples=(length == sorted(length_groups.keys())[0])  # Show examples for first length
        )
        
        results[length] = result
        
        # Print results
        status = "✓ IN-DIST" if length <= max_train_length else "✗ OUT-OF-DIST"
        print(f"Length {length:3d} ({status}): {result['exact_match']:.3f} "
              f"({result['correct_samples']}/{result['total_samples']})")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Lightning model on length generalization")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to Lightning checkpoint")
    parser.add_argument("--max_eval_length", type=int, default=150,
                       help="Maximum sequence length to evaluate")
    parser.add_argument("--samples_per_length", type=int, default=100,
                       help="Number of samples per length")
    parser.add_argument("--output_dir", type=str, default="length_gen_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Setup
    device = get_device()
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Lightning model and tokenizer
    model, tokenizer = load_lightning_model(args.checkpoint, device)
    
    # Define test lengths (training was up to 60 tokens)
    max_train_length = 60
    test_lengths = [30, 40, 50, 60, 70, 80, 90, 100, 120, 150]
    if args.max_eval_length != 150:
        # Adjust test lengths based on max_eval_length
        test_lengths = [l for l in test_lengths if l <= args.max_eval_length]
    
    print(f"Max training length: {max_train_length}")
    print(f"Test lengths: {test_lengths}")
    
    # Evaluate length generalization
    results = evaluate_length_generalization_lightning(
        model, tokenizer, device, 
        max_train_length=max_train_length,
        test_lengths=test_lengths,
        samples_per_length=args.samples_per_length
    )
    
    # Save results
    results_path = os.path.join(args.output_dir, "lightning_length_generalization.json")
    
    # Convert results to JSON-serializable format
    json_results = {}
    for length, result in results.items():
        json_results[str(length)] = {
            'exact_match': result['exact_match'],
            'total_samples': result['total_samples'],
            'correct_samples': result['correct_samples']
        }
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("LENGTH GENERALIZATION SUMMARY")
    print("="*70)
    
    # Calculate performance on in-distribution vs out-of-distribution
    in_dist_results = [(l, r['exact_match']) for l, r in results.items() if l <= max_train_length]
    ood_results = [(l, r['exact_match']) for l, r in results.items() if l > max_train_length]
    
    if in_dist_results:
        avg_in_dist = sum(acc for _, acc in in_dist_results) / len(in_dist_results)
        print(f"📊 Average in-distribution accuracy:     {avg_in_dist:.3f}")
        
        # Show individual in-dist results
        print(f"   In-distribution lengths: {[l for l, _ in in_dist_results]}")
        for length, acc in in_dist_results:
            print(f"   Length {length:3d}: {acc:.3f}")
    
    if ood_results:
        avg_ood = sum(acc for _, acc in ood_results) / len(ood_results)
        print(f"🚀 Average out-of-distribution accuracy: {avg_ood:.3f}")
        
        # Show individual OOD results
        print(f"   Out-of-distribution lengths: {[l for l, _ in ood_results]}")
        for length, acc in ood_results:
            print(f"   Length {length:3d}: {acc:.3f}")
        
        # Check if length generalization is successful
        successful_ood = sum(1 for _, acc in ood_results if acc > 0.9)
        print(f"\n🎯 Lengths with >90% accuracy: {successful_ood}/{len(ood_results)}")
        
        if avg_ood > 0.9:
            print("🎉 EXCELLENT: Strong length generalization achieved!")
            print("   The model demonstrates the RASP-Generalization Conjecture!")
        elif avg_ood > 0.7:
            print("🔶 GOOD: Moderate length generalization achieved.")
        else:
            print("❌ POOR: Limited length generalization.")
    
    print(f"\n💾 Results saved to: {results_path}")


if __name__ == "__main__":
    main() 