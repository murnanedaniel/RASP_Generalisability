#!/usr/bin/env python3
"""
Quick test script to verify the counting task implementation.

This script performs basic sanity checks:
1. Test data generation 
2. Test tokenization
3. Test model initialization
4. Test training loop (few steps)
5. Test generation

This helps verify our "easily solvable" task is implemented correctly.
"""

import torch
from data_generation import generate_counting_sequence, calculate_sequence_length
from utils import Tokenizer, collate_fn
from model import CountingTransformer
from config import counting_config

def test_data_generation():
    """Test basic data generation functionality."""
    print("Testing data generation...")
    
    # Test counting up
    seq1 = generate_counting_sequence(3, 7)
    expected1 = "SoS 3 7 > 3 4 5 6 7 EoS"
    print(f"Count up: {seq1}")
    assert seq1 == expected1, f"Expected {expected1}, got {seq1}"
    
    # Test counting down  
    seq2 = generate_counting_sequence(10, 6)
    expected2 = "SoS 10 6 > 10 9 8 7 6 EoS"
    print(f"Count down: {seq2}")
    assert seq2 == expected2, f"Expected {expected2}, got {seq2}"
    
    # Test single number
    seq3 = generate_counting_sequence(5, 5)
    expected3 = "SoS 5 5 > 5 EoS"
    print(f"Single number: {seq3}")
    assert seq3 == expected3, f"Expected {expected3}, got {seq3}"
    
    # Test sequence length calculation
    length1 = calculate_sequence_length(3, 7)
    print(f"Length of '3 to 7': {length1}")
    assert length1 == 10, f"Expected 10, got {length1}"  # SoS + 3 + 7 + > + [3,4,5,6,7] + EoS = 10
    
    print("✅ Data generation tests passed!")

def test_tokenization():
    """Test tokenizer functionality."""
    print("\nTesting tokenization...")
    
    tokenizer = Tokenizer()
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Test sequence encoding/decoding
    sequence = "SoS 3 7 > 3 4 5 6 7 EoS"
    tokens = sequence.split()
    print(f"Original: {sequence}")
    print(f"Tokens: {tokens}")
    
    # Encode and decode
    ids = tokenizer.encode_sequence(sequence)
    decoded = tokenizer.decode_sequence(ids)
    print(f"Encoded: {ids}")
    print(f"Decoded: {decoded}")
    
    assert decoded == sequence, f"Round-trip failed: {decoded} != {sequence}"
    
    # Test special tokens
    print(f"SoS ID: {tokenizer.token_to_id['SoS']}")
    print(f"EoS ID: {tokenizer.token_to_id['EoS']}")
    print(f"PAD ID: {tokenizer.pad_token_id}")
    
    print("✅ Tokenization tests passed!")

def test_model_initialization():
    """Test model can be created and forward pass works."""
    print("\nTesting model initialization...")
    
    # Update config for small test
    config = counting_config
    config.model.vocab_size = 106  # Should match tokenizer
    config.model.n_layers = 2  # Smaller for testing
    config.model.d_model = 128
    config.model.n_heads = 4
    
    model = CountingTransformer(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")
    
    # Test forward pass
    tokenizer = Tokenizer()
    sequence = "SoS 3 7 > 3 4 5 6 7 EoS"
    ids = torch.tensor([tokenizer.encode_sequence(sequence)])
    
    print(f"Input shape: {ids.shape}")
    logits, loss = model(ids, labels=ids)
    print(f"Output logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    assert logits.shape == (1, ids.shape[1], config.model.vocab_size)
    assert loss is not None
    
    print("✅ Model initialization tests passed!")

def test_generation():
    """Test model generation (should be random initially)."""
    print("\nTesting generation...")
    
    config = counting_config
    config.model.vocab_size = 106
    config.model.n_layers = 2
    config.model.d_model = 128
    config.model.n_heads = 4
    
    model = CountingTransformer(config)
    tokenizer = Tokenizer()
    
    # Test prompt
    prompt = "SoS 3 7 >"
    prompt_ids = torch.tensor([tokenizer.encode_sequence(prompt)])
    print(f"Prompt: {prompt}")
    print(f"Prompt IDs: {prompt_ids}")
    
    # Generate (should be random for untrained model)
    generated = model.generate(
        prompt_ids, 
        max_new_tokens=10, 
        eos_token_id=tokenizer.eos_token_id
    )
    
    generated_text = tokenizer.decode_sequence(generated[0].tolist())
    print(f"Generated: {generated_text}")
    
    # Just check it generated something
    assert generated.shape[1] > prompt_ids.shape[1], "Model should generate some tokens"
    
    print("✅ Generation tests passed!")

def test_simple_training():
    """Test a few training steps work."""
    print("\nTesting simple training...")
    
    # Create small model
    config = counting_config
    config.model.vocab_size = 106
    config.model.n_layers = 2
    config.model.d_model = 128
    config.model.n_heads = 4
    
    model = CountingTransformer(config)
    tokenizer = Tokenizer()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Create a small batch
    sequences = [
        "SoS 3 7 > 3 4 5 6 7 EoS",
        "SoS 10 6 > 10 9 8 7 6 EoS",
        "SoS 5 5 > 5 EoS"
    ]
    
    samples = [{'sequence': seq} for seq in sequences]
    batch = collate_fn(samples, tokenizer)
    
    print(f"Batch input shape: {batch['input_ids'].shape}")
    
    # Training step
    model.train()
    initial_loss = None
    
    for step in range(5):
        optimizer.zero_grad()
        
        logits, loss = model(batch['input_ids'], labels=batch['input_ids'])
        loss.backward()
        optimizer.step()
        
        if step == 0:
            initial_loss = loss.item()
        
        print(f"Step {step}: Loss = {loss.item():.4f}")
    
    final_loss = loss.item()
    print(f"Loss change: {initial_loss:.4f} -> {final_loss:.4f}")
    
    # Loss should generally decrease (though not guaranteed in 5 steps)
    print("✅ Simple training tests passed!")

def main():
    """Run all tests."""
    print("🚀 Running quick tests for counting task implementation...")
    print("=" * 60)
    
    try:
        test_data_generation()
        test_tokenization()
        test_model_initialization()
        test_generation()
        test_simple_training()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed! The implementation looks good.")
        print("\nKey observations:")
        print("• Data generation produces correct counting sequences")
        print("• Tokenization works bidirectionally") 
        print("• Model initializes and forward pass works")
        print("• Generation produces output (random for untrained model)")
        print("• Training loop executes without errors")
        print("\n💡 The task appears to be correctly implemented.")
        print("   You can now run the full experiment with confidence!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 