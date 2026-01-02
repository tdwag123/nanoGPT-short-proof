"""
Interactive question-answering with the fine-tuned model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# Configuration
out_dir = 'out-math-qa'
device = 'cpu'
temperature = 0.3  # Lower temperature for more focused answers
top_k = 40
max_new_tokens = 100
seed = 1337

print("Loading model from", out_dir)
torch.manual_seed(seed)
device_type = 'cpu'
ctx = nullcontext()

# Load model
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)

# Setup encoder/decoder
print("No meta.pkl found, assuming GPT-2 encodings...")
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

print("\n" + "="*60)
print("Math Q&A Interactive Session")
print("="*60)
print("Ask elementary number theory questions!")
print("Type 'quit' or 'exit' to end the session.")
print("Examples:")
print("  - What is the GCD of 24 and 36?")
print("  - What is 15 mod 7?")
print("  - Is 17 a prime number?")
print("="*60 + "\n")

while True:
    # Get question from user
    question = input("Q: ").strip()
    
    if question.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        break
    
    if not question:
        continue
    
    # Format as training data format
    prompt = f"Q: {question}\nA:"
    start_ids = encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    
    # Generate answer
    with torch.no_grad():
        with ctx:
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            output = decode(y[0].tolist())
            
            # Extract just the answer part
            if "\nA:" in output:
                answer_part = output.split("\nA:", 1)[1]
                # Stop at next question or double newline
                if "\nQ:" in answer_part:
                    answer_part = answer_part.split("\nQ:", 1)[0]
                elif "\n\n" in answer_part:
                    answer_part = answer_part.split("\n\n", 1)[0]
                
                print(f"A:{answer_part.strip()}\n")
            else:
                print(f"A: {output}\n")
