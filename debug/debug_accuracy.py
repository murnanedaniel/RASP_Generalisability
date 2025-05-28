#!/usr/bin/env python3
"""
Debug script to understand why exact match is always 0
"""

import torch
import torch.nn as nn
from model import CountingTransformer
from config import counting_config
from utils import Tokenizer, get_device

def test_accuracy_calculation():
    """Test the accuracy calculation logic"""
    
    # Create a simple tokenizer
    tokenizer = Tokenizer()
    device = get_device()
    
    # Create a simple test case
    # Let's say we have: "SoS 1 3 > 1 2 3 EoS"
    sequence = "SoS 1 3 > 1 2 3 EoS"
    tokens = sequence.split()
    print(f"Test sequence: {sequence}")
    print(f"Tokens: {tokens}")
    
    # Encode the sequence
    input_ids = torch.tensor([tokenizer.encode_sequence(sequence)], device=device)
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Input IDs: {input_ids[0].tolist()}")
    
    # Decode to verify
    decoded = tokenizer.decode(input_ids[0].tolist())
    print(f"Decoded: {' '.join(decoded)}")
    
    # Create attention mask (all ones for this example)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    
    print("\n--- Understanding Next-Token Prediction ---")
    print("For autoregressive training:")
    for i in range(input_ids.shape[1] - 1):
        context = ' '.join(tokenizer.decode(input_ids[0, :i+1].tolist()))
        next_token = tokenizer.decode([input_ids[0, i+1].item()])[0]
        print(f"  After seeing '{context}' -> should predict '{next_token}'")
    
    # CORRECT way to simulate perfect predictions
    # The model output at position i predicts the token at position i+1
    perfect_logits = torch.zeros(1, input_ids.shape[1], tokenizer.vocab_size, device=device)
    for i in range(input_ids.shape[1] - 1):
        target_token_id = input_ids[0, i+1].item()
        perfect_logits[0, i, target_token_id] = 100.0  # High confidence for correct token
    
    perfect_predictions = torch.argmax(perfect_logits, dim=-1)
    
    print("\n--- Testing CORRECT accuracy calculation ---")
    
    # For next-token prediction:
    # - Model outputs predictions for positions 0 to n-1
    # - We compare prediction[i] with input_ids[i+1] (the actual next token)
    batch_size, seq_len = input_ids.shape
    
    # Targets: what the model should predict at each position
    targets = input_ids[:, 1:]  # tokens 1 to n (next tokens)
    
    # Predictions: what the model actually predicted at each position
    predictions = perfect_predictions[:, :-1]  # predictions 0 to n-2 (we can't predict beyond sequence)
    
    print(f"Input tokens: {' '.join(tokenizer.decode(input_ids[0].tolist()))}")
    print(f"Targets shape: {targets.shape}")
    print(f"Predictions shape: {predictions.shape}")
    
    print(f"\nPosition-by-position perfect prediction:")
    for i in range(min(targets.shape[1], predictions.shape[1])):
        target_token = tokenizer.decode([targets[0, i].item()])[0]
        pred_token = tokenizer.decode([predictions[0, i].item()])[0]
        match = targets[0, i].item() == predictions[0, i].item()
        print(f"  Pos {i}: target='{target_token}' pred='{pred_token}' match={match}")
    
    # Check if they match
    matches = (predictions == targets)
    print(f"All matches: {matches[0].tolist()}")
    print(f"Perfect exact match: {torch.all(matches).item()}")
    
    # Now test with a model
    print("\n--- Testing with actual model ---")
    config = counting_config
    config.model.vocab_size = tokenizer.vocab_size
    model = CountingTransformer(config).to(device)
    model.eval()
    
    # Prepare labels
    labels = input_ids.clone()
    labels[~attention_mask] = -100
    
    # Forward pass
    with torch.no_grad():
        logits, loss = model(input_ids, attention_mask, labels)
        print(f"Logits shape: {logits.shape}")
        print(f"Loss: {loss.item():.4f}")
        
        # Get predictions
        predicted_ids = torch.argmax(logits, dim=-1)
        print(f"Predicted IDs shape: {predicted_ids.shape}")
        
        # CORRECT accuracy calculation
        targets = input_ids[:, 1:]  # What should be predicted
        predictions = predicted_ids[:, :-1]  # What was actually predicted
        
        print(f"\nModel prediction analysis:")
        for i in range(min(5, targets.shape[1])):
            target_token = tokenizer.decode([targets[0, i].item()])[0]
            pred_token = tokenizer.decode([predictions[0, i].item()])[0]
            match = targets[0, i].item() == predictions[0, i].item()
            print(f"  Pos {i}: should predict '{target_token}' -> predicted '{pred_token}' match={match}")
        
        # Check matches
        matches = (predictions == targets)
        print(f"Token matches: {matches[0].tolist()}")
        print(f"Exact match: {torch.all(matches).item()}")
        
        # Also test with attention mask
        target_mask = attention_mask[:, 1:]  # Mask for target positions
        valid_matches = matches & target_mask
        total_valid = target_mask.sum()
        token_accuracy = valid_matches.sum().float() / total_valid.float() if total_valid > 0 else 0.0
        print(f"Token accuracy: {token_accuracy.item():.3f}")
        
        # Sequence accuracy (all tokens must match within valid positions)
        sequence_correct = torch.all(valid_matches | ~target_mask)  # All valid positions match
        print(f"Sequence correct: {sequence_correct.item()}")

def test_model_output_shift():
    """Test if model output is already shifted"""
    tokenizer = Tokenizer()
    device = get_device()
    
    print("\n=== Testing Model Output Shift ===")
    
    # Simple sequence
    sequence = "SoS 1 2 > 1 2 EoS"
    input_ids = torch.tensor([tokenizer.encode_sequence(sequence)], device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    
    print(f"Input sequence: {sequence}")
    print(f"Input IDs: {input_ids[0].tolist()}")
    
    # Create model
    config = counting_config
    config.model.vocab_size = tokenizer.vocab_size
    model = CountingTransformer(config).to(device)
    
    # Get raw logits without labels
    with torch.no_grad():
        # Just pass input_ids and attention_mask
        outputs = model.model(input_ids, attention_mask)
        raw_logits = model.output_projection(outputs)
        
        print(f"Raw logits shape: {raw_logits.shape}")
        print(f"Input length: {input_ids.shape[1]}")
        
        # Check if logits are already shifted
        if raw_logits.shape[1] == input_ids.shape[1] - 1:
            print("✓ Logits are already shifted (one less than input)")
        elif raw_logits.shape[1] == input_ids.shape[1]:
            print("✗ Logits have same length as input (not shifted)")
        
        # Now with labels
        labels = input_ids.clone()
        labels[~attention_mask] = -100
        
        logits, loss = model(input_ids, attention_mask, labels)
        print(f"\nWith labels - Logits shape: {logits.shape}")
        
        # What does the model predict at each position?
        predicted_ids = torch.argmax(logits, dim=-1)
        print(f"\nPosition-by-position analysis:")
        for i in range(min(5, input_ids.shape[1])):
            input_token = tokenizer.decode([input_ids[0, i].item()])[0]
            if i < predicted_ids.shape[1]:
                pred_token = tokenizer.decode([predicted_ids[0, i].item()])[0]
                if i + 1 < input_ids.shape[1]:
                    target_token = tokenizer.decode([input_ids[0, i+1].item()])[0]
                    print(f"Pos {i}: input='{input_token}' -> pred='{pred_token}' (target='{target_token}')")
                else:
                    print(f"Pos {i}: input='{input_token}' -> pred='{pred_token}' (no target)")
            else:
                print(f"Pos {i}: input='{input_token}' (no prediction)")

def test_model_shifting_behavior():
    """Test to understand exactly how the model shifts logits and labels"""
    tokenizer = Tokenizer()
    device = get_device()
    
    print("\n=== Testing Model Shifting Behavior ===")
    
    # Simple sequence
    sequence = "SoS 1 2 3 EoS"
    input_ids = torch.tensor([tokenizer.encode_sequence(sequence)], device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    
    print(f"Input sequence: {sequence}")
    print(f"Input tokens: {' '.join(tokenizer.decode(input_ids[0].tolist()))}")
    print(f"Input IDs: {input_ids[0].tolist()}")
    print(f"Input shape: {input_ids.shape}")
    
    # Create model
    config = counting_config
    config.model.vocab_size = tokenizer.vocab_size
    model = CountingTransformer(config).to(device)
    model.eval()
    
    # Test 1: Get logits WITHOUT labels (no shifting)
    print("\n--- Test 1: Forward pass WITHOUT labels ---")
    with torch.no_grad():
        logits_no_labels, loss_no_labels = model(input_ids, attention_mask, labels=None)
        print(f"Logits shape: {logits_no_labels.shape}")
        print(f"Loss: {loss_no_labels}")
        
        # What does each position predict?
        predictions_no_labels = torch.argmax(logits_no_labels, dim=-1)
        print(f"Predictions shape: {predictions_no_labels.shape}")
        print("\nPosition-by-position (no labels):")
        for i in range(input_ids.shape[1]):
            input_token = tokenizer.decode([input_ids[0, i].item()])[0]
            if i < predictions_no_labels.shape[1]:
                pred_token = tokenizer.decode([predictions_no_labels[0, i].item()])[0]
                print(f"  Pos {i}: input='{input_token}' -> predicts '{pred_token}'")
    
    # Test 2: Get logits WITH labels (check if shifting happens)
    print("\n--- Test 2: Forward pass WITH labels ---")
    labels = input_ids.clone()
    labels[~attention_mask] = -100
    
    with torch.no_grad():
        logits_with_labels, loss_with_labels = model(input_ids, attention_mask, labels=labels)
        print(f"Logits shape: {logits_with_labels.shape}")
        print(f"Loss: {loss_with_labels.item():.4f}")
        
        # Check if logits are the same
        print(f"\nAre logits identical? {torch.allclose(logits_no_labels, logits_with_labels)}")
        
        # What does each position predict?
        predictions_with_labels = torch.argmax(logits_with_labels, dim=-1)
        print(f"Predictions shape: {predictions_with_labels.shape}")
        
    # Test 3: Manual shifting to understand the loss calculation
    print("\n--- Test 3: Understanding the loss calculation ---")
    print("The model internally does:")
    print("  shift_logits = logits[:, :-1, :]  # Predictions for positions 0 to n-2")
    print("  shift_labels = labels[:, 1:]      # Targets for positions 1 to n-1")
    
    shift_logits = logits_with_labels[:, :-1, :]
    shift_labels = labels[:, 1:]
    
    print(f"\nShifted logits shape: {shift_logits.shape}")
    print(f"Shifted labels shape: {shift_labels.shape}")
    
    # Decode what this means
    shift_predictions = torch.argmax(shift_logits, dim=-1)
    print("\nWhat the loss is actually comparing:")
    for i in range(shift_predictions.shape[1]):
        if shift_labels[0, i] != -100:
            pred_token = tokenizer.decode([shift_predictions[0, i].item()])[0]
            target_token = tokenizer.decode([shift_labels[0, i].item()])[0]
            input_context = ' '.join(tokenizer.decode(input_ids[0, :i+1].tolist()))
            print(f"  After '{input_context}' -> predicts '{pred_token}', target '{target_token}'")
    
    # Test 4: Correct accuracy calculation
    print("\n--- Test 4: CORRECT accuracy calculation ---")
    print("Since logits are NOT shifted, we should compare:")
    print("  predictions = argmax(logits)[:, :-1]  # Use all but last prediction")
    print("  targets = input_ids[:, 1:]            # Next tokens")
    
    predictions = predictions_with_labels[:, :-1]
    targets = input_ids[:, 1:]
    
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")
    
    print("\nComparison:")
    for i in range(predictions.shape[1]):
        pred_token = tokenizer.decode([predictions[0, i].item()])[0]
        target_token = tokenizer.decode([targets[0, i].item()])[0]
        match = predictions[0, i] == targets[0, i]
        print(f"  Pos {i}: pred='{pred_token}' vs target='{target_token}' -> match={match}")
    
    # Calculate accuracy
    matches = (predictions == targets)
    exact_match = torch.all(matches).item()
    token_accuracy = matches.float().mean().item()
    
    print(f"\nExact match: {exact_match}")
    print(f"Token accuracy: {token_accuracy:.3f}")

def test_trained_model_simulation():
    """Test accuracy calculation with a simulated well-trained model"""
    tokenizer = Tokenizer()
    device = get_device()
    
    print("\n=== Testing With Simulated Well-Trained Model ===")
    
    # Simple sequence
    sequence = "SoS 1 3 > 1 2 3 EoS"
    input_ids = torch.tensor([tokenizer.encode_sequence(sequence)], device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    
    print(f"Input sequence: {sequence}")
    print(f"Input tokens: {' '.join(tokenizer.decode(input_ids[0].tolist()))}")
    
    # Create model
    config = counting_config
    config.model.vocab_size = tokenizer.vocab_size
    model = CountingTransformer(config).to(device)
    model.eval()
    
    # Get the model's current logits
    with torch.no_grad():
        original_logits, _ = model(input_ids, attention_mask, labels=None)
        
        # Manually create "perfect" logits where each position correctly predicts the next token
        perfect_logits = original_logits.clone()
        batch_size, seq_len, vocab_size = perfect_logits.shape
        
        # Clear all logits and set high values for correct predictions
        perfect_logits.fill_(-100.0)  # Low probability for all tokens
        
        # Set high probability for correct next token at each position
        for i in range(seq_len - 1):  # Don't predict past the sequence
            next_token_id = input_ids[0, i + 1].item()
            perfect_logits[0, i, next_token_id] = 100.0  # High confidence for correct token
        
        # Test accuracy calculation with perfect logits
        perfect_predictions = torch.argmax(perfect_logits, dim=-1)
        
        print("\nWith perfect logits:")
        print("What each position predicts:")
        for i in range(seq_len - 1):
            input_context = ' '.join(tokenizer.decode(input_ids[0, :i+1].tolist()))
            predicted_token = tokenizer.decode([perfect_predictions[0, i].item()])[0]
            target_token = tokenizer.decode([input_ids[0, i+1].item()])[0]
            print(f"  After '{input_context}' -> predicts '{predicted_token}' (target: '{target_token}')")
        
        # Apply the SAME accuracy calculation as in training
        targets = input_ids[:, 1:]  # Next tokens
        predictions = perfect_predictions[:, :-1]  # Remove last prediction
        
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
        
        print(f"\n✅ Perfect model results:")
        print(f"Token accuracy: {token_accuracy.item():.3f}")
        print(f"Exact match: {exact_match:.3f}")
        
        # Now test the same calculation but with a small error
        print("\n--- Testing with one wrong prediction ---")
        imperfect_logits = perfect_logits.clone()
        # Make position 1 predict wrong token
        imperfect_logits[0, 1, :] = -100.0  # Clear all
        wrong_token_id = (input_ids[0, 2].item() + 1) % vocab_size  # Pick a wrong token
        imperfect_logits[0, 1, wrong_token_id] = 100.0
        
        imperfect_predictions = torch.argmax(imperfect_logits, dim=-1)
        predictions_imp = imperfect_predictions[:, :-1]
        
        # Calculate accuracy
        correct_tokens_imp = (predictions_imp == targets) & target_mask
        token_accuracy_imp = correct_tokens_imp.sum().float() / total_valid_tokens.float()
        
        # Sequence accuracy
        sequence_matches_imp = 0
        for i in range(batch_size):
            seq_len_i = attention_mask[i].sum().item()
            if seq_len_i > 1:
                pred_seq = predictions_imp[i][:seq_len_i-1]
                target_seq = targets[i][:seq_len_i-1]
                if torch.equal(pred_seq, target_seq):
                    sequence_matches_imp += 1
        
        exact_match_imp = sequence_matches_imp / batch_size
        
        print(f"With 1 wrong prediction:")
        print(f"Token accuracy: {token_accuracy_imp.item():.3f}")
        print(f"Exact match: {exact_match_imp:.3f}")

if __name__ == "__main__":
    test_accuracy_calculation()
    # test_model_output_shift()  # This has an error, commenting out
    test_model_shifting_behavior()
    test_trained_model_simulation() 