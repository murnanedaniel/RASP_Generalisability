"""
Comprehensive evaluation script for RASP generalization experiments.
"""

import os
import argparse
import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from tqdm import tqdm

from config import counting_config
from model import CountingTransformer
from data_generation import load_dataset, generate_length_specific_dataset
from utils import Tokenizer, get_device, plot_length_generalization, save_results

def load_model_and_tokenizer(checkpoint_path: str, device: torch.device):
    """Load trained model and tokenizer from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Use weights_only=False for PyTorch Lightning checkpoints
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle PyTorch Lightning checkpoint format
    if 'hyper_parameters' in checkpoint:
        # This is a Lightning checkpoint
        print("Detected PyTorch Lightning checkpoint")
        
        # Get config from hyperparameters
        config = checkpoint['hyper_parameters']['config']
        
        # Create tokenizer (use default since we don't save tokenizer state in Lightning)
        from config import counting_config_flat
        tokenizer = Tokenizer()
        
        # Create and load model
        model = CountingTransformer(config).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        
        # Get validation accuracy if available
        best_val_exact_match = checkpoint.get('best_val_exact_match', 'N/A')
        
    else:
        # This is a regular checkpoint
        print("Detected regular PyTorch checkpoint")
        
        # Load config
        config = checkpoint['config']
        
        # Create tokenizer
        tokenizer = Tokenizer()
        if 'tokenizer_state' in checkpoint:
            tokenizer.token_to_id = checkpoint['tokenizer_state']['token_to_id']
            tokenizer.id_to_token = checkpoint['tokenizer_state']['id_to_token']
        
        # Create and load model
        model = CountingTransformer(config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        best_val_exact_match = checkpoint.get('best_val_exact_match', 'N/A')
    
    print(f"Model loaded successfully!")
    print(f"Best validation exact match from training: {best_val_exact_match}")
    
    return model, tokenizer, config

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

def evaluate_generation_detailed(model: CountingTransformer, samples: List[dict], 
                               tokenizer: Tokenizer, device: torch.device, 
                               max_new_tokens: int = 100, verbose: bool = False) -> Dict:
    """Detailed evaluation with generation examples"""
    model.eval()
    
    correct_sequences = 0
    total_sequences = len(samples)
    examples = []
    error_types = {'wrong_sequence': 0, 'incomplete': 0, 'invalid_format': 0}
    
    with torch.no_grad():
        for i, sample in enumerate(tqdm(samples, desc="Evaluating", leave=False)):
            sequence = sample['sequence']
            tokens = sequence.split()
            
            # Find the separator '>' to split prompt and target
            try:
                separator_idx = tokens.index('>')
                prompt_tokens = tokens[:separator_idx + 1]  # Include the '>'
                target_tokens = tokens[separator_idx + 1:]   # Exclude the '>'
            except ValueError:
                error_types['invalid_format'] += 1
                continue
            
            # Encode prompt
            prompt = " ".join(prompt_tokens)
            prompt_ids = torch.tensor([tokenizer.encode_sequence(prompt)], device=device)
            
            # Generate
            try:
                generated_ids = model.generate(
                    prompt_ids, 
                    max_new_tokens=max_new_tokens,
                    temperature=0.1,  # Low temperature for more consistent results
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
                else:
                    if len(generated_text.strip()) == 0:
                        error_types['incomplete'] += 1
                    else:
                        error_types['wrong_sequence'] += 1
                
                # Store examples for analysis
                if i < 10 or not is_correct:  # Store first 10 examples and all errors
                    examples.append({
                        'prompt': prompt,
                        'target': target_text,
                        'generated': generated_text,
                        'correct': is_correct,
                        'length': sample['length']
                    })
                
            except Exception as e:
                error_types['invalid_format'] += 1
                if verbose:
                    print(f"Error generating for sample {i}: {e}")
    
    accuracy = correct_sequences / total_sequences if total_sequences > 0 else 0.0
    
    return {
        'exact_match': accuracy,
        'total_samples': total_sequences,
        'correct_samples': correct_sequences,
        'error_types': error_types,
        'examples': examples
    }

def evaluate_length_generalization(model: CountingTransformer, tokenizer: Tokenizer, 
                                 device: torch.device, max_train_length: int = 60,
                                 test_lengths: List[int] = None, 
                                 samples_per_length: int = 100) -> Dict[int, Dict]:
    """Comprehensive length generalization evaluation"""
    
    if test_lengths is None:
        test_lengths = [30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200]
    
    print("Generating test data for length generalization...")
    
    # Generate test samples at specific lengths
    test_samples = generate_length_specific_dataset(
        target_lengths=test_lengths,
        samples_per_length=samples_per_length,
        min_number=0,
        max_number=100,  # Increase range for longer sequences
        seed=123
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
    print("-" * 60)
    
    for length in sorted(length_groups.keys()):
        samples = length_groups[length]
        
        print(f"Evaluating length {length} ({len(samples)} samples)...")
        result = evaluate_generation_detailed(
            model, samples, tokenizer, device, 
            max_new_tokens=length + 20  # Give some extra tokens
        )
        
        results[length] = result
        
        # Print results
        status = "✓ TRAIN" if length <= max_train_length else "✗ OOD"
        print(f"Length {length:3d} ({status}): {result['exact_match']:.3f} "
              f"({result['correct_samples']}/{result['total_samples']})")
        
        # Show error breakdown
        if result['total_samples'] > 0:
            for error_type, count in result['error_types'].items():
                if count > 0:
                    pct = count / result['total_samples'] * 100
                    print(f"  {error_type}: {count} ({pct:.1f}%)")
    
    return results

def analyze_failure_cases(results: Dict[int, Dict], max_examples: int = 5):
    """Analyze and display failure cases"""
    print("\n" + "="*60)
    print("FAILURE CASE ANALYSIS")
    print("="*60)
    
    for length, result in results.items():
        examples = result['examples']
        failed_examples = [ex for ex in examples if not ex['correct']]
        
        if failed_examples:
            print(f"\nLength {length} failures (showing up to {max_examples}):")
            print("-" * 40)
            
            for i, example in enumerate(failed_examples[:max_examples]):
                print(f"Example {i+1}:")
                print(f"  Prompt:    {example['prompt']}")
                print(f"  Target:    {example['target']}")
                print(f"  Generated: {example['generated']}")
                print()

def visualize_results(results: Dict[int, Dict], max_train_length: int, 
                     save_path: str = None):
    """Create visualizations of length generalization results"""
    
    # Prepare data
    lengths = sorted(results.keys())
    accuracies = [results[length]['exact_match'] for length in lengths]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Length generalization curve
    colors = ['green' if l <= max_train_length else 'red' for l in lengths]
    ax1.plot(lengths, accuracies, 'o-', linewidth=2, markersize=8, color='blue')
    ax1.scatter(lengths, accuracies, c=colors, s=100, alpha=0.7, edgecolors='black')
    ax1.axvline(x=max_train_length, color='red', linestyle='--', 
                label=f'Max Training Length ({max_train_length})')
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Exact Match Accuracy')
    ax1.set_title('Length Generalization Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # Add annotations for in-distribution vs out-of-distribution
    ax1.text(max_train_length - 15, 0.9, 'In-Distribution', 
             rotation=90, ha='center', va='center', color='green', fontweight='bold')
    ax1.text(max_train_length + 15, 0.9, 'Out-of-Distribution', 
             rotation=90, ha='center', va='center', color='red', fontweight='bold')
    
    # Plot 2: Error type breakdown for OOD lengths
    ood_lengths = [l for l in lengths if l > max_train_length]
    if ood_lengths:
        error_types = ['wrong_sequence', 'incomplete', 'invalid_format']
        error_data = {error_type: [] for error_type in error_types}
        
        for length in ood_lengths:
            total_samples = results[length]['total_samples']
            for error_type in error_types:
                count = results[length]['error_types'][error_type]
                error_data[error_type].append(count / total_samples if total_samples > 0 else 0)
        
        # Stacked bar chart
        bottom = [0] * len(ood_lengths)
        colors_errors = ['red', 'orange', 'yellow']
        
        for i, error_type in enumerate(error_types):
            ax2.bar(ood_lengths, error_data[error_type], bottom=bottom, 
                   label=error_type.replace('_', ' ').title(), 
                   color=colors_errors[i], alpha=0.7)
            bottom = [b + e for b, e in zip(bottom, error_data[error_type])]
        
        # Add correct samples on top
        correct_rates = [results[length]['exact_match'] for length in ood_lengths]
        ax2.bar(ood_lengths, correct_rates, bottom=bottom, 
               label='Correct', color='green', alpha=0.7)
        
        ax2.set_xlabel('Sequence Length (Out-of-Distribution)')
        ax2.set_ylabel('Proportion')
        ax2.set_title('Error Type Breakdown (OOD Lengths)')
        ax2.legend()
        ax2.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plots saved to: {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Evaluate counting transformer")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--test_data", type=str, default=None,
                       help="Path to test dataset (optional)")
    parser.add_argument("--max_eval_length", type=int, default=150,
                       help="Maximum sequence length to evaluate")
    parser.add_argument("--samples_per_length", type=int, default=100,
                       help="Number of samples per length")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Setup
    device = get_device()
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer, config = load_model_and_tokenizer(args.model_path, device)
    
    # Get max training length from config
    max_train_length = config.training.max_train_length
    print(f"Max training length: {max_train_length}")
    
    # Define test lengths
    test_lengths = list(range(max_train_length + 10, args.max_eval_length + 1, 10))
    if max_train_length not in test_lengths and max_train_length > 0:
        test_lengths.insert(0, max_train_length)  # Include training length for comparison
    
    print(f"Test lengths: {test_lengths}")
    
    # Evaluate length generalization
    results = evaluate_length_generalization(
        model, tokenizer, device, 
        max_train_length=max_train_length,
        test_lengths=test_lengths,
        samples_per_length=args.samples_per_length
    )
    
    # Analyze failure cases
    if args.verbose:
        analyze_failure_cases(results)
    
    # Visualize results
    plot_path = os.path.join(args.output_dir, "length_generalization_plot.png")
    visualize_results(results, max_train_length, plot_path)
    
    # Save detailed results
    results_path = os.path.join(args.output_dir, "detailed_results.json")
    
    # Convert results to JSON-serializable format
    json_results = {}
    for length, result in results.items():
        json_results[str(length)] = {
            'exact_match': result['exact_match'],
            'total_samples': result['total_samples'],
            'correct_samples': result['correct_samples'],
            'error_types': result['error_types']
            # Note: examples are not saved to avoid large file sizes
        }
    
    save_results(json_results, results_path)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    # Calculate average performance on in-distribution vs out-of-distribution
    in_dist_results = [(l, r['exact_match']) for l, r in results.items() if l <= max_train_length]
    ood_results = [(l, r['exact_match']) for l, r in results.items() if l > max_train_length]
    
    if in_dist_results:
        avg_in_dist = sum(acc for _, acc in in_dist_results) / len(in_dist_results)
        print(f"Average in-distribution accuracy: {avg_in_dist:.3f}")
    
    if ood_results:
        avg_ood = sum(acc for _, acc in ood_results) / len(ood_results)
        print(f"Average out-of-distribution accuracy: {avg_ood:.3f}")
        
        # Check if length generalization is successful (>90% accuracy on OOD)
        successful_ood = sum(1 for _, acc in ood_results if acc > 0.9)
        print(f"Lengths with >90% accuracy: {successful_ood}/{len(ood_results)}")
        
        if avg_ood > 0.9:
            print("✅ Strong length generalization achieved!")
        elif avg_ood > 0.7:
            print("🔶 Moderate length generalization achieved.")
        else:
            print("❌ Poor length generalization.")
    
    print(f"\nResults saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 