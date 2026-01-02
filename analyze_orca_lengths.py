"""
Analyze Orca Math dataset to find problem lengths
"""
from datasets import load_dataset
import tiktoken

print("Loading dataset...")
dataset = load_dataset("microsoft/orca-math-word-problems-200k")

enc = tiktoken.get_encoding("gpt2")

lengths = []
max_example = None
max_length = 0

print("Analyzing problem lengths...")
for i, example in enumerate(dataset['train']):
    question = example['question'].strip()
    answer = example['answer'].strip()
    formatted = f"Q: {question}\nA: {answer}\n\n"
    
    tokens = enc.encode(formatted)
    length = len(tokens)
    lengths.append(length)
    
    if length > max_length:
        max_length = length
        max_example = formatted
    
    if (i + 1) % 50000 == 0:
        print(f"Processed {i + 1} examples...")

# Statistics
lengths_sorted = sorted(lengths)
print(f"\nToken Length Statistics:")
print(f"Minimum: {lengths_sorted[0]}")
print(f"Maximum: {lengths_sorted[-1]}")
print(f"Average: {sum(lengths) / len(lengths):.1f}")
print(f"Median: {lengths_sorted[len(lengths)//2]}")
print(f"95th percentile: {lengths_sorted[int(len(lengths)*0.95)]}")
print(f"99th percentile: {lengths_sorted[int(len(lengths)*0.99)]}")

print(f"\nExamples by length:")
print(f"  <= 256 tokens: {sum(1 for l in lengths if l <= 256)} ({100*sum(1 for l in lengths if l <= 256)/len(lengths):.1f}%)")
print(f"  <= 512 tokens: {sum(1 for l in lengths if l <= 512)} ({100*sum(1 for l in lengths if l <= 512)/len(lengths):.1f}%)")
print(f"  <= 1024 tokens: {sum(1 for l in lengths if l <= 1024)} ({100*sum(1 for l in lengths if l <= 1024)/len(lengths):.1f}%)")

print(f"\nLongest example ({max_length} tokens):")
print(max_example[:500] + "..." if len(max_example) > 500 else max_example)
