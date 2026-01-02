"""
Interactive math problem solver with the Orca Math fine-tuned model
"""
import os
import torch
import tiktoken
from model import GPTConfig, GPT

# Configuration
out_dir = 'out-orca-math'
device = 'cpu'
temperature = 0.3
top_k = 40
max_new_tokens = 200
seed = 1337

print("Loading Orca Math fine-tuned model from", out_dir)
torch.manual_seed(seed)

# Check if model exists
if not os.path.exists(os.path.join(out_dir, 'ckpt.pt')):
    print(f"\nError: No checkpoint found in {out_dir}")
    print("The model hasn't been trained yet. Please run training first:")
    print('  python train.py config/finetune_orca_math.py')
    exit(1)

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
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

print("\n" + "="*60)
print("Math Word Problem Solver - Interactive Session")
print("="*60)
print("Ask math word problems and get step-by-step solutions!")
print("Type 'quit' or 'exit' to end the session.")
print("\nExamples:")
print("  - A store has 24 apples. If 1/3 are sold, how many remain?")
print("  - John has 5 times as many marbles as Mary. If Mary has")
print("    8 marbles, how many does John have?")
print("="*60 + "\n")

while True:
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
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        output = decode(y[0].tolist())
        
        # Extract answer part
        if "\nA:" in output:
            answer_part = output.split("\nA:", 1)[1]
            if "\nQ:" in answer_part:
                answer_part = answer_part.split("\nQ:", 1)[0]
            elif "\n\n" in answer_part:
                answer_part = answer_part.split("\n\n", 1)[0]
            
            print(f"A:{answer_part.strip()}\n")
        else:
            print(f"A: {output}\n")
