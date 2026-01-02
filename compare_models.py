import os
import torch
import tiktoken
from model import GPTConfig, GPT

# Test questions
questions = [
    "What is the GCD of 24 and 36?",
    "What is 15 mod 7?",
    "Is 17 a prime number?",
]

print("Loading base model...")
base_model = GPT.from_pretrained('gpt2-medium', dict(dropout=0.0))
base_model.eval()
base_model.to('cpu')

print("Loading fine-tuned model...")
checkpoint = torch.load('out-math-qa/ckpt.pt', map_location='cpu')
ft_model = GPT(GPTConfig(**checkpoint['model_args']))
ft_model.load_state_dict(checkpoint['model'])
ft_model.eval()
ft_model.to('cpu')

enc = tiktoken.get_encoding("gpt2")

def ask(model, question):
    prompt = f"Q: {question}\nA:"
    tokens = enc.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long)[None, ...]
    y = model.generate(x, 100, temperature=0.3, top_k=40)
    output = enc.decode(y[0].tolist())
    answer = output.split("\nA:", 1)[1].split("\n\n", 1)[0].strip()
    return answer

for question in questions:
    print(f"\n{'='*80}")
    print(f"Q: {question}")
    print(f"{'='*80}")
    print(f"Base model:       {ask(base_model, question)}")
    print(f"Fine-tuned model: {ask(ft_model, question)}")
