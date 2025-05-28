# RASP Generalization Study: Understanding When Transformers Learn Algorithms

## Overview

This repository implements a reproduction of the key experiments from ["What Algorithms can Transformers Learn? A Study in Length Generalization"](https://arxiv.org/pdf/2310.16028) (Zhou et al., 2023). The paper's central claim is the **RASP-Generalization Conjecture**: 

> *Transformers tend to length generalize on a task if and only if the task can be solved by a short program in RASP-L (a restricted version of RASP).*

## What is the Task?

We focus on the **counting task**, which demonstrates excellent length generalization and serves as a clear example of the conjecture.

### Task Definition: Counting
- **Input Format**: `SoS <start> <end> >`
- **Output Format**: `<sequence_of_numbers> EoS`
- **Rule**: Count from `start` to `end` (inclusive), either up or down

### Examples:
```
Input:  SoS 3 7 >
Output: 3 4 5 6 7 EoS

Input:  SoS 10 6 >  
Output: 10 9 8 7 6 EoS

Input:  SoS 5 5 >
Output: 5 EoS
```

### Why This Task?
1. **Simple to understand**: Clear algorithmic pattern
2. **RASP-L solvable**: Can be implemented using position-based attention patterns
3. **Length generalization**: Should work on longer sequences than seen in training
4. **Clear success metric**: Exact sequence match

## What Makes This "Easily Solvable"?

The counting task can be solved by a Transformer using position-based attention patterns because:

1. **Position-based patterns**: The model can learn to attend to positions rather than content
2. **Local computation**: Each output token depends on simple arithmetic with the input numbers
3. **Regular structure**: The pattern is highly regular and doesn't require complex reasoning

In RASP-L terms, this task can be solved with:
- Position-based selectors (attending to the start/end positions)
- Simple arithmetic operations
- Sequential generation patterns

## Repository Structure

```
├── config.py              # Model and training configurations
├── data_generation.py     # Generate counting task datasets
├── model.py               # Transformer implementation
├── train.py               # Training script with evaluation
├── evaluate.py            # Comprehensive length generalization evaluation
├── utils.py               # Tokenizer, metrics, and utility functions
├── run_experiment.py      # Complete experimental pipeline
├── requirements.txt       # Dependencies
├── README.md              # Usage instructions
└── intro.md               # This file
```

## Data Format and Tokenization

### Tokenization
- **Special tokens**: `["SoS", "EoS", ">", "<PAD>", "<UNK>"]`
- **Number tokens**: `["0", "1", "2", ..., "100"]`
- **Vocabulary size**: ~106 tokens

### Sequence Structure
```
[SoS] [start_num] [end_num] [>] [count_1] [count_2] ... [count_n] [EoS]
```

### Training Data Generation
- **Length control**: Generate sequences up to a maximum length (e.g., 60 tokens)
- **Random sampling**: Random start/end numbers within a range
- **Balanced data**: Mix of counting up and counting down

## Model Architecture

### Transformer Details
- **Type**: Decoder-only autoregressive Transformer
- **Architecture**: 6 layers, 8 attention heads, 512 hidden dimensions
- **Position encoding**: Sinusoidal (crucial for position-based patterns)
- **Training objective**: Next-token prediction (causal language modeling)

### Key Components
1. **Token embeddings**: Map tokens to vectors
2. **Positional encoding**: Enable position-based attention
3. **Multi-head attention**: Learn attention patterns
4. **Feed-forward layers**: Local computation
5. **Output projection**: Predict next token

## Training Process

### Objective
The model learns to predict the next token given the previous tokens:
```
Input:  [SoS] [3] [7] [>] [3] [4]
Target: [3] [7] [>] [3] [4] [5]
```

### Loss Function
- **Cross-entropy loss** on next-token prediction
- **Ignore padding** tokens in loss computation
- **Autoregressive** training (each token predicts the next)

### Training Data
- **Training set**: Sequences up to length L (e.g., L=60)
- **Validation set**: Same length distribution as training
- **Test set**: Longer sequences (L+10 to L+90) for generalization testing

## Evaluation: What Constitutes a "Match"?

### Exact Match Accuracy
A prediction is considered correct if and only if:
1. **Complete sequence match**: Generated sequence exactly matches target
2. **Proper termination**: Model generates `EoS` token
3. **No extra tokens**: No additional tokens after `EoS`

### Example Evaluation
```
Target:    3 4 5 6 7 EoS
Generated: 3 4 5 6 7 EoS     ✅ MATCH (100% correct)
Generated: 3 4 5 6 8 EoS     ❌ NO MATCH (wrong number)
Generated: 3 4 5 6 7         ❌ NO MATCH (no EoS)
Generated: 3 4 5 6 7 EoS 9   ❌ NO MATCH (extra tokens)
```

### Generation Process
1. **Prompt**: Give model `SoS start end >`
2. **Generate**: Let model autoregressively generate tokens
3. **Stop**: When model generates `EoS` or reaches max length
4. **Evaluate**: Compare generated sequence to ground truth

## Length Generalization Study

### The Core Question
Can a model trained on sequences up to length L correctly solve the same task on sequences of length L+k?

### Experimental Setup
1. **Training**: Train on sequences up to length 60
2. **Testing**: Evaluate on sequences of length 70, 80, 90, 100, 120, 150
3. **Metric**: Exact match accuracy at each test length
4. **Success criterion**: >90% accuracy on out-of-distribution lengths

### Expected Results (Based on Paper)
- **In-distribution** (≤60): ~100% accuracy
- **Out-of-distribution** (>60): >95% accuracy
- **Pattern**: Accuracy should remain high across all test lengths

### Why Should This Work?
According to the RASP-Generalization Conjecture:
1. Counting has a short RASP-L solution
2. RASP-L solutions use position-based patterns
3. Position-based patterns generalize to longer sequences
4. Therefore, counting should exhibit strong length generalization

## Key Insights from the Paper

### RASP-L Characteristics
- **Position-based attention**: Attention patterns depend on token positions
- **Limited depth**: Shallow computation graphs
- **Simple aggregations**: Basic operations like mean, max
- **No content-based reasoning**: Doesn't depend on complex token interactions

### Why Some Tasks Fail
Tasks that fail length generalization typically require:
- **Content-based attention**: Attention based on token values
- **Complex reasoning**: Multi-step logical inference
- **Variable binding**: Tracking relationships between distant tokens
- **Deep computation**: Many layers of processing

### The Counting Task in RASP-L
The counting task can be solved with:
1. **Position selectors**: Identify start/end positions
2. **Arithmetic**: Simple increment/decrement operations
3. **Sequential generation**: Output tokens in order
4. **Termination**: Detect when to stop

## Running the Experiments

### Quick Test
```bash
python run_experiment.py --quick
```
- Small dataset (5K training samples)
- Shorter sequences (max length 40)
- 20 epochs training
- Test up to length 100

### Full Experiment
```bash
python run_experiment.py
```
- Large dataset (50K training samples)
- Standard sequences (max length 60)
- 100 epochs training
- Test up to length 150

### Expected Timeline
- **Quick test**: ~30 minutes
- **Full experiment**: 2-4 hours (depending on hardware)

## Success Criteria

### Training Success
- Training loss decreases to <0.1
- Validation exact match reaches >95%
- Model generates correct sequences on validation set

### Length Generalization Success
- Out-of-distribution accuracy >90%
- Performance degrades gracefully (if at all)
- Clear demonstration of the RASP-Generalization Conjecture

## Common Issues and Debugging

### Model Not Learning
- Check learning rate (try 5e-4)
- Verify data format is correct
- Ensure tokenization is working
- Check for tensor shape mismatches

### Poor Generalization
- Verify positional encoding is working
- Check if model is overfitting to content vs. positions
- Ensure test data follows same format as training

### Generation Issues
- Check EoS token handling
- Verify autoregressive generation
- Look for infinite generation loops

This implementation serves as both a reproduction of important research results and a clear demonstration of when and why Transformers can learn algorithmic tasks that generalize beyond their training distribution. 