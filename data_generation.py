"""
Data generation for the counting task.
Format: SoS a b > count_sequence EoS
Example: SoS 3 7 > 3 4 5 6 7 EoS
"""

import random
import argparse
import os
from typing import List, Tuple
import json
from tqdm import tqdm

from config import counting_config

def generate_counting_sequence(start: int, end: int, sos_token: str = "SoS", 
                             eos_token: str = "EoS", separator: str = ">") -> str:
    """
    Generate a single counting sequence.
    
    Args:
        start: Starting number
        end: Ending number (inclusive)
        sos_token: Start of sequence token
        eos_token: End of sequence token
        separator: Separator token
    
    Returns:
        Formatted sequence string
    """
    if start > end:
        # If start > end, count down
        count_sequence = list(range(start, end - 1, -1))
    else:
        # Count up
        count_sequence = list(range(start, end + 1))
    
    # Format: SoS start end > count_sequence EoS
    sequence_parts = [sos_token, str(start), str(end), separator]
    sequence_parts.extend([str(num) for num in count_sequence])
    sequence_parts.append(eos_token)
    
    return " ".join(sequence_parts)

def calculate_sequence_length(start: int, end: int) -> int:
    """Calculate the total length of the generated sequence"""
    # SoS + start + end + > + count_sequence + EoS
    count_length = abs(end - start) + 1
    return 4 + count_length + 1  # 5 fixed tokens + count_sequence

def generate_dataset_flat_distribution(num_samples: int, max_length: int, min_length: int = 6, 
                                     min_number: int = 0, seed: int = 42) -> List[dict]:
    """
    Generate a dataset with flat distribution of sequence lengths.
    
    Each length from min_length to max_length gets equal representation.
    
    Args:
        num_samples: Total number of samples to generate
        max_length: Maximum sequence length (e.g., 60)
        min_length: Minimum sequence length (e.g., 6 - minimum meaningful counting sequence)
        min_number: Minimum number value
        seed: Random seed
    
    Returns:
        List of samples with sequence and metadata
    """
    random.seed(seed)
    samples = []
    
    # Calculate samples per length for flat distribution
    num_lengths = max_length - min_length + 1
    samples_per_length = num_samples // num_lengths
    remaining_samples = num_samples % num_lengths
    
    print(f"Generating flat distribution: {num_lengths} lengths, ~{samples_per_length} samples each")
    
    for target_length in range(min_length, max_length + 1):
        # Add one extra sample to some lengths to handle remainder
        current_samples_needed = samples_per_length
        if target_length - min_length < remaining_samples:
            current_samples_needed += 1
        
        length_samples = []
        attempts = 0
        max_attempts = current_samples_needed * 50  # Be generous with attempts
        
        while len(length_samples) < current_samples_needed and attempts < max_attempts:
            attempts += 1
            
            # Calculate required count length: target_length = 5 + count_length  
            # (SoS + start + > + count + EoS = 5 + count_length)
            desired_count_length = target_length - 5
            
            if desired_count_length <= 0:
                continue
            
            # Dynamically adjust number range to ensure we can create the desired length
            # We need enough range to count from start to end
            required_range = desired_count_length + 20  # Add buffer
            adjusted_max_number = max(min_number + required_range, 100)
            
            # Sample start point
            max_start = adjusted_max_number - desired_count_length
            if max_start < min_number:
                continue
                
            start = random.randint(min_number, max_start)
            
            # Randomly choose to count up or down
            if random.random() < 0.5:
                # Count up: start -> start + (desired_count_length - 1)
                end = start + desired_count_length - 1
            else:
                # Count down: start -> start - (desired_count_length - 1)
                end = start - desired_count_length + 1
            
            # Ensure end is in valid range
            if end < min_number:
                continue
            
            # Verify the actual sequence length matches target
            actual_length = calculate_sequence_length(start, end)
            if actual_length != target_length:
                continue
            
            # Generate the sequence
            sequence = generate_counting_sequence(start, end)
            
            sample = {
                'sequence': sequence,
                'start': start,
                'end': end,
                'length': actual_length,
                'count_length': abs(end - start) + 1
            }
            
            length_samples.append(sample)
        
        samples.extend(length_samples)
        success_rate = len(length_samples) / attempts * 100 if attempts > 0 else 0
        print(f"Length {target_length:2d}: generated {len(length_samples):3d}/{current_samples_needed:3d} samples "
              f"(success rate: {success_rate:.1f}%)")
        
        if len(length_samples) < current_samples_needed:
            print(f"  ⚠️  Warning: Could not generate enough samples for length {target_length}")
    
    # Shuffle the final dataset
    random.shuffle(samples)
    print(f"\nGenerated {len(samples)} total samples with flat length distribution")
    
    return samples

def generate_dataset(num_samples: int, max_length: int, min_number: int = 0, 
                    max_number: int = 50, seed: int = 42) -> List[dict]:
    """
    Generate a dataset of counting sequences with FLAT length distribution.
    
    This replaces the old random sampling approach with one that ensures
    equal representation of all sequence lengths from 6 to max_length.
    
    Args:
        num_samples: Number of samples to generate
        max_length: Maximum sequence length
        min_number: Minimum number value (legacy parameter, dynamically adjusted)
        max_number: Maximum number value (legacy parameter, dynamically adjusted)
        seed: Random seed
    
    Returns:
        List of samples with sequence and metadata
    """
    return generate_dataset_flat_distribution(
        num_samples=num_samples,
        max_length=max_length,
        min_length=6,  # Minimum meaningful counting sequence
        min_number=min_number,
        seed=seed
    )

def split_dataset(samples: List[dict], train_ratio: float = 0.8, 
                 val_ratio: float = 0.1) -> Tuple[List[dict], List[dict], List[dict]]:
    """Split dataset into train/val/test sets"""
    random.shuffle(samples)
    
    n_train = int(len(samples) * train_ratio)
    n_val = int(len(samples) * val_ratio)
    
    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]
    
    return train_samples, val_samples, test_samples

def save_dataset(samples: List[dict], filepath: str):
    """Save dataset to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(samples, f, indent=2)
    print(f"Saved {len(samples)} samples to {filepath}")

def load_dataset(filepath: str) -> List[dict]:
    """Load dataset from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def generate_length_specific_dataset(target_lengths: List[int], 
                                   samples_per_length: int = 100,
                                   min_number: int = 0, max_number: int = 50,
                                   seed: int = 42) -> List[dict]:
    """
    Generate dataset with specific target sequence lengths for evaluation.
    """
    random.seed(seed)
    samples = []
    
    for target_length in target_lengths:
        length_samples = []
        attempts = 0
        max_attempts = samples_per_length * 20  # Increase max attempts
        
        while len(length_samples) < samples_per_length and attempts < max_attempts:
            attempts += 1
            
            # Sample start and end to try to hit target length
            # target_length = 5 + count_length
            # count_length = abs(end - start) + 1
            desired_count_length = target_length - 5
            
            if desired_count_length <= 0:
                continue
            
            # For longer sequences, we need a larger number range
            # Dynamically increase max_number if needed
            adjusted_max_number = max(max_number, min_number + desired_count_length + 10)
                
            start = random.randint(min_number, adjusted_max_number - desired_count_length)
            
            # Randomly choose to count up or down
            if random.random() < 0.5:
                end = start + desired_count_length - 1
            else:
                end = start - desired_count_length + 1
            
            # Ensure numbers are in valid range
            if end < min_number or start < min_number:
                continue
            
            seq_length = calculate_sequence_length(start, end)
            if seq_length != target_length:
                continue
            
            sequence = generate_counting_sequence(start, end)
            
            sample = {
                'sequence': sequence,
                'start': start,
                'end': end,
                'length': seq_length,
                'count_length': abs(end - start) + 1
            }
            
            length_samples.append(sample)
        
        samples.extend(length_samples)
        print(f"Generated {len(length_samples)} samples for length {target_length}")
    
    return samples

def print_dataset_stats(samples: List[dict]):
    """Print statistics about the dataset"""
    lengths = [sample['length'] for sample in samples]
    count_lengths = [sample['count_length'] for sample in samples]
    
    print(f"Dataset Statistics:")
    print(f"  Total samples: {len(samples)}")
    print(f"  Sequence length range: {min(lengths)} - {max(lengths)}")
    print(f"  Count length range: {min(count_lengths)} - {max(count_lengths)}")
    print(f"  Average sequence length: {sum(lengths) / len(lengths):.1f}")
    
    # Show some examples
    print(f"\nExample sequences:")
    for i, sample in enumerate(samples[:3]):
        print(f"  {i+1}: {sample['sequence']}")

def main():
    parser = argparse.ArgumentParser(description="Generate counting task dataset")
    parser.add_argument("--max_train_length", type=int, default=60, 
                       help="Maximum training sequence length")
    parser.add_argument("--num_samples", type=int, default=50000,
                       help="Number of training samples")
    parser.add_argument("--num_val_samples", type=int, default=5000,
                       help="Number of validation samples")
    parser.add_argument("--output_dir", type=str, default="data",
                       help="Output directory for datasets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    print("Generating counting task dataset...")
    print(f"Max training length: {args.max_train_length}")
    print(f"Number of samples: {args.num_samples}")
    
    # Generate training dataset
    train_samples = generate_dataset(
        num_samples=args.num_samples,
        max_length=args.max_train_length,
        min_number=0,
        max_number=50,
        seed=args.seed
    )
    
    # Generate validation dataset
    val_samples = generate_dataset(
        num_samples=args.num_val_samples,
        max_length=args.max_train_length,
        min_number=0,
        max_number=50,
        seed=args.seed + 1
    )
    
    # Generate test datasets at various lengths for length generalization
    test_lengths = [70, 80, 90, 100, 120, 150]
    test_samples = generate_length_specific_dataset(
        target_lengths=test_lengths,
        samples_per_length=200,
        min_number=0,
        max_number=80,  # Increase range for longer sequences
        seed=args.seed + 2
    )
    
    # Save datasets
    os.makedirs(args.output_dir, exist_ok=True)
    save_dataset(train_samples, os.path.join(args.output_dir, "train.json"))
    save_dataset(val_samples, os.path.join(args.output_dir, "val.json"))
    save_dataset(test_samples, os.path.join(args.output_dir, "test_length_gen.json"))
    
    # Print statistics
    print("\nTraining dataset:")
    print_dataset_stats(train_samples)
    
    print("\nValidation dataset:")
    print_dataset_stats(val_samples)
    
    print("\nLength generalization test dataset:")
    print_dataset_stats(test_samples)

if __name__ == "__main__":
    main() 