"""
Prepare Orca Math Word Problems dataset for training.
Downloads from HuggingFace and converts to binary token files.
"""
import os
import numpy as np
import tiktoken
from datasets import load_dataset

print("Downloading Orca Math Word Problems dataset from HuggingFace...")
print("This is a large dataset (200k examples), download may take several minutes...")

# Just load normally - the timeout may be a network issue
try:
    dataset = load_dataset("microsoft/orca-math-word-problems-200k")
except Exception as e:
    print(f"Error downloading dataset: {e}")
    print("\nTroubleshooting:")
    print("1. Check your internet connection")
    print("2. Try again later (HuggingFace may be experiencing issues)")
    print("3. Consider using a smaller dataset for testing")
    raise

print(f"Dataset loaded. Train split has {len(dataset['train'])} examples")

# Combine question and answer into training format
print("Processing examples and filtering by length (max 256 tokens)...")
text_data = []
enc = tiktoken.get_encoding("gpt2")
skipped = 0

for i, example in enumerate(dataset['train']):
    # Format: Q: [question]\nA: [answer]\n\n
    question = example['question'].strip()
    answer = example['answer'].strip()
    
    formatted = f"Q: {question}\nA: {answer}\n\n"
    
    # Check token length
    tokens = enc.encode_ordinary(formatted)
    if len(tokens) <= 256:
        text_data.append(formatted)
    else:
        skipped += 1
    
    if (i + 1) % 10000 == 0:
        print(f"Processed {i + 1} examples, kept {len(text_data)}, skipped {skipped}...")

print(f"\nOriginal examples: {len(dataset['train'])}")
print(f"Kept (<=256 tokens): {len(text_data)}")
print(f"Skipped (>256 tokens): {skipped}")

# Combine all text
full_text = ''.join(text_data)
print(f"Total characters: {len(full_text)}")

# Tokenize with GPT-2 BPE
print("Tokenizing filtered dataset...")
tokens = enc.encode_ordinary(full_text)
print(f"Total tokens: {len(tokens)}")

# Split into train (90%) and val (10%)
n = len(tokens)
train_tokens = tokens[:int(n*0.9)]
val_tokens = tokens[int(n*0.9):]

print(f"Train tokens: {len(train_tokens)}")
print(f"Val tokens: {len(val_tokens)}")

# Save as binary files
train_ids = np.array(train_tokens, dtype=np.uint16)
val_ids = np.array(val_tokens, dtype=np.uint16)

output_dir = os.path.dirname(__file__)
train_ids.tofile(os.path.join(output_dir, 'train.bin'))
val_ids.tofile(os.path.join(output_dir, 'val.bin'))

print(f"\nSaved train.bin and val.bin to {output_dir}")

# Calculate training iterations
batch_size = 2
gradient_accumulation_steps = 16
block_size = 256  # Using 256 since we filtered to this length
tokens_per_iter = batch_size * gradient_accumulation_steps * block_size
iters_per_epoch = len(train_tokens) / tokens_per_iter

print(f"\nWith batch_size={batch_size}, gradient_accumulation_steps={gradient_accumulation_steps}, block_size={block_size}:")
print(f"Tokens per iteration: {tokens_per_iter}")
print(f"Iterations per epoch: {iters_per_epoch:.1f}")
print(f"For 1 epoch, set max_iters to ~{int(iters_per_epoch)}")
print(f"For 3 epochs, set max_iters to ~{int(iters_per_epoch * 3)}")
