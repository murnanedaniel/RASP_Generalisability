"""
Utility functions for RASP generalization experiments.

This module contains the core utilities for tokenization, data processing, 
evaluation metrics, and helper functions used throughout the project.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Any
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

class Tokenizer:
    """
    Simple tokenizer for the counting task.
    
    The tokenizer handles the conversion between text sequences and token IDs.
    It includes special tokens for sequence boundaries and supports numbers 0-100.
    
    Vocabulary structure:
    - Special tokens: ["SoS", "EoS", ">", "<PAD>", "<UNK>"]
    - Number tokens: ["0", "1", "2", ..., "100"]
    - Total vocabulary size: ~106 tokens
    
    Example usage:
        tokenizer = Tokenizer()
        sequence = "SoS 3 7 > 3 4 5 6 7 EoS"
        ids = tokenizer.encode_sequence(sequence)
        decoded = tokenizer.decode_sequence(ids)
    """
    
    def __init__(self, special_tokens=None):
        """
        Initialize the tokenizer with special tokens and number tokens.
        
        Args:
            special_tokens: List of special tokens. If None, uses default set.
        """
        if special_tokens is None:
            special_tokens = ["SoS", "EoS", ">", "<PAD>", "<UNK>"]
        
        self.special_tokens = special_tokens
        self.token_to_id = {}
        self.id_to_token = {}
        
        # Add special tokens first
        for token in special_tokens:
            self._add_token(token)
            
        # Add number tokens (0-100 should be enough for most experiments)
        for i in range(101):
            self._add_token(str(i))
    
    def _add_token(self, token: str):
        """Add a token to the vocabulary if it doesn't exist."""
        if token not in self.token_to_id:
            token_id = len(self.token_to_id)
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
    
    def encode(self, tokens: List[str]) -> List[int]:
        """
        Convert a list of tokens to a list of token IDs.
        
        Args:
            tokens: List of string tokens
            
        Returns:
            List of integer token IDs
        """
        return [self.token_to_id.get(token, self.token_to_id["<UNK>"]) for token in tokens]
    
    def decode(self, ids: List[int]) -> List[str]:
        """
        Convert a list of token IDs to a list of tokens.
        
        Args:
            ids: List of integer token IDs
            
        Returns:
            List of string tokens
        """
        return [self.id_to_token.get(id, "<UNK>") for id in ids]
    
    def encode_sequence(self, sequence: str) -> List[int]:
        """
        Encode a space-separated sequence string to token IDs.
        
        Args:
            sequence: Space-separated string (e.g., "SoS 3 7 > 3 4 5 6 7 EoS")
            
        Returns:
            List of token IDs
        """
        tokens = sequence.strip().split()
        return self.encode(tokens)
    
    def decode_sequence(self, ids: List[int]) -> str:
        """
        Decode token IDs to a space-separated sequence string.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Space-separated string
        """
        tokens = self.decode(ids)
        return " ".join(tokens)
    
    @property
    def vocab_size(self):
        """Get the vocabulary size."""
        return len(self.token_to_id)
    
    @property
    def pad_token_id(self):
        """Get the padding token ID."""
        return self.token_to_id["<PAD>"]
    
    @property
    def eos_token_id(self):
        """Get the end-of-sequence token ID."""
        return self.token_to_id["EoS"]

def collate_fn(batch, tokenizer, max_length=None):
    """
    Collate function for DataLoader that handles padding and creates attention masks.
    
    This function processes a batch of samples, tokenizes them, and pads them to
    the same length for efficient batch processing.
    
    Args:
        batch: List of sample dictionaries with 'sequence' key
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length for padding (None = use batch max)
        
    Returns:
        Dictionary containing:
        - input_ids: Tensor of shape [batch_size, seq_len] with token IDs
        - attention_mask: Tensor of shape [batch_size, seq_len] with masks
        - sequences: List of original sequence strings
    """
    sequences = [item['sequence'] for item in batch]
    
    # Encode sequences
    encoded_sequences = [tokenizer.encode_sequence(seq) for seq in sequences]
    
    # Find max length in batch
    if max_length is None:
        max_len = max(len(seq) for seq in encoded_sequences)
    else:
        max_len = max_length
    
    # Pad sequences
    padded_sequences = []
    attention_masks = []
    
    for seq in encoded_sequences:
        if len(seq) > max_len:
            seq = seq[:max_len]
        
        padding_length = max_len - len(seq)
        padded_seq = seq + [tokenizer.pad_token_id] * padding_length
        attention_mask = [1] * len(seq) + [0] * padding_length
        
        padded_sequences.append(padded_seq)
        attention_masks.append(attention_mask)
    
    return {
        'input_ids': torch.tensor(padded_sequences, dtype=torch.long),
        'attention_mask': torch.tensor(attention_masks, dtype=torch.bool),
        'sequences': sequences
    }

def calculate_exact_match_accuracy(predictions: List[str], targets: List[str]) -> float:
    """
    Calculate exact match accuracy between predictions and targets.
    
    A prediction is considered correct if and only if it exactly matches
    the target string after stripping whitespace.
    
    Args:
        predictions: List of predicted strings
        targets: List of target strings
        
    Returns:
        Accuracy as a float between 0 and 1
        
    Raises:
        ValueError: If predictions and targets have different lengths
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have same length")
    
    correct = sum(1 for pred, target in zip(predictions, targets) if pred.strip() == target.strip())
    return correct / len(predictions)

def calculate_sequence_accuracy(pred_ids: torch.Tensor, target_ids: torch.Tensor, 
                              pad_token_id: int, eos_token_id: int) -> Tuple[float, float]:
    """
    Calculate sequence-level accuracy metrics from token ID tensors.
    
    This function computes both exact match accuracy (entire sequence correct)
    and token-level accuracy (fraction of correct tokens).
    
    Args:
        pred_ids: Predicted token IDs [batch_size, seq_len]
        target_ids: Target token IDs [batch_size, seq_len]
        pad_token_id: ID of padding token to ignore
        eos_token_id: ID of end-of-sequence token
        
    Returns:
        Tuple of (exact_match_accuracy, token_accuracy)
        
    Note:
        Sequences are compared up to the first EoS token or end of sequence.
        Padding tokens are ignored in the comparison.
    """
    batch_size = pred_ids.size(0)
    exact_matches = 0
    total_tokens = 0
    correct_tokens = 0
    
    for i in range(batch_size):
        pred = pred_ids[i]
        target = target_ids[i]
        
        # Find EoS positions
        pred_eos_pos = (pred == eos_token_id).nonzero(as_tuple=True)[0]
        target_eos_pos = (target == eos_token_id).nonzero(as_tuple=True)[0]
        
        if len(pred_eos_pos) > 0:
            pred_end = pred_eos_pos[0].item() + 1
        else:
            pred_end = len(pred)
            
        if len(target_eos_pos) > 0:
            target_end = target_eos_pos[0].item() + 1
        else:
            target_end = len(target)
        
        # Check exact match
        pred_seq = pred[:pred_end]
        target_seq = target[:target_end]
        
        if torch.equal(pred_seq, target_seq):
            exact_matches += 1
        
        # Token accuracy (up to minimum length)
        min_len = min(pred_end, target_end)
        total_tokens += min_len
        correct_tokens += (pred[:min_len] == target[:min_len]).sum().item()
    
    exact_match_acc = exact_matches / batch_size
    token_acc = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    
    return exact_match_acc, token_acc

def plot_length_generalization(results: Dict[int, Dict[str, float]], 
                             max_train_length: int, 
                             save_path: str = None):
    """
    Plot length generalization results showing accuracy vs sequence length.
    
    Creates a line plot showing how exact match accuracy varies with
    sequence length, with a vertical line indicating the maximum training length.
    
    Args:
        results: Dictionary mapping length to metrics dict with 'exact_match' key
        max_train_length: Maximum training sequence length (for vertical line)
        save_path: Optional path to save the plot
    """
    lengths = sorted(results.keys())
    exact_match_accs = [results[length]['exact_match'] for length in lengths]
    
    plt.figure(figsize=(10, 6))
    plt.plot(lengths, exact_match_accs, 'o-', linewidth=2, markersize=8)
    plt.axvline(x=max_train_length, color='red', linestyle='--', 
                label=f'Max Training Length ({max_train_length})')
    plt.xlabel('Sequence Length')
    plt.ylabel('Exact Match Accuracy')
    plt.title('Length Generalization Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def save_results(results: Dict[str, Any], filepath: str):
    """
    Save results dictionary to JSON file.
    
    Creates the directory if it doesn't exist and saves the results
    as a formatted JSON file.
    
    Args:
        results: Dictionary to save
        filepath: Path to save the JSON file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

def load_results(filepath: str) -> Dict[str, Any]:
    """
    Load results dictionary from JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Dictionary loaded from JSON
    """
    with open(filepath, 'r') as f:
        return json.load(f)

def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a PyTorch model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_device() -> torch.device:
    """
    Get the best available device (CUDA if available, otherwise CPU).
    
    Returns:
        PyTorch device object
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def set_seed(seed: int):
    """
    Set random seeds for reproducibility across PyTorch, NumPy, and CUDA.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 