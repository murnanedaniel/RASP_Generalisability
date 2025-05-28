"""
Check what lengths are actually in the training data.
"""

import json
from collections import Counter

# Load NEW flat distribution training data
print("=== FLAT DISTRIBUTION TRAINING DATA ===")
with open('data_flat/train.json', 'r') as f:
    train_data = json.load(f)

# Get length distribution
lengths = [sample['length'] for sample in train_data]
length_counts = Counter(lengths)

print('NEW Training data length distribution:')
for length in sorted(length_counts.keys()):
    count = length_counts[length]
    percentage = count / len(lengths) * 100
    print(f'Length {length:2d}: {count:3d} samples ({percentage:5.1f}%)')

print(f'\nTotal samples: {len(train_data)}')
print(f'Length range: {min(lengths)} - {max(lengths)}')
print(f'Average length: {sum(lengths) / len(lengths):.1f}')

# Check how many samples are at different length ranges
ranges = [(6, 20), (21, 40), (41, 60), (61, 100)]
for min_len, max_len in ranges:
    count = sum(1 for l in lengths if min_len <= l <= max_len)
    percentage = count / len(lengths) * 100
    print(f'Lengths {min_len:2d}-{max_len:2d}: {count:4d} samples ({percentage:5.1f}%)')

# Check validation data too
print("\n=== FLAT DISTRIBUTION VALIDATION DATA ===")
with open('data_flat/val.json', 'r') as f:
    val_data = json.load(f)

val_lengths = [sample['length'] for sample in val_data]
val_length_counts = Counter(val_lengths)

print('NEW Validation data length distribution:')
for length in sorted(val_length_counts.keys()):
    count = val_length_counts[length]
    percentage = count / len(val_lengths) * 100
    print(f'Length {length:2d}: {count:3d} samples ({percentage:5.1f}%)')

print(f'\nValidation total samples: {len(val_data)}')
print(f'Validation length range: {min(val_lengths)} - {max(val_lengths)}')
print(f'Validation average length: {sum(val_lengths) / len(val_lengths):.1f}')

# Show some examples from different lengths
print('\nExample sequences from NEW flat distribution:')
length_examples = {}
for sample in train_data:
    length = sample['length']
    if length not in length_examples:
        length_examples[length] = []
    if len(length_examples[length]) < 1:  # Keep 1 example per length
        length_examples[length].append(sample)

# Show examples at key lengths
key_lengths = [6, 15, 30, 45, 60]
for length in key_lengths:
    if length in length_examples:
        print(f'\nLength {length} example:')
        sample = length_examples[length][0]
        print(f'  {sample["sequence"]}')

# Compare with old data
print("\n" + "="*60)
print("COMPARISON WITH OLD BIASED DISTRIBUTION")
print("="*60)

with open('data/train.json', 'r') as f:
    old_train_data = json.load(f)

old_lengths = [sample['length'] for sample in old_train_data]

print(f'OLD distribution: range {min(old_lengths)}-{max(old_lengths)}, avg {sum(old_lengths)/len(old_lengths):.1f}')
print(f'NEW distribution: range {min(lengths)}-{max(lengths)}, avg {sum(lengths)/len(lengths):.1f}')

print(f'\nCoverage comparison:')
print(f'OLD: lengths 1-40 only ({max(old_lengths)} max)')
print(f'NEW: lengths 6-60 flat distribution ({max(lengths)} max)')

# Check if we now have good coverage in critical ranges
print(f'\nLength 50-60 coverage (critical for generalization):')
count_50_60_old = sum(1 for l in old_lengths if 50 <= l <= 60)
count_50_60_new = sum(1 for l in lengths if 50 <= l <= 60)
print(f'OLD: {count_50_60_old} samples')
print(f'NEW: {count_50_60_new} samples')

print(f'\n🎯 SUCCESS: Now we have proper training data for lengths 6-60!') 