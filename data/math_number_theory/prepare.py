"""
Prepare elementary number theory Q&A dataset for training.
Generates synthetic problem-answer pairs and saves as binary token files.
"""
import os
import random
import numpy as np
import tiktoken

random.seed(42)
np.random.seed(42)

def gcd(a, b):
    """Calculate greatest common divisor."""
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    """Calculate least common multiple."""
    return abs(a * b) // gcd(a, b)

def is_prime(n):
    """Check if number is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def prime_factorization(n):
    """Get prime factorization as list of factors."""
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

def generate_gcd_problems(n_samples=500):
    """Generate GCD problems."""
    problems = []
    for _ in range(n_samples):
        a = random.randint(2, 200)
        b = random.randint(2, 200)
        answer = gcd(a, b)
        problems.append(f"Q: What is the greatest common divisor of {a} and {b}?\nA: {answer}\n\n")
    return problems

def generate_lcm_problems(n_samples=500):
    """Generate LCM problems."""
    problems = []
    for _ in range(n_samples):
        a = random.randint(2, 50)
        b = random.randint(2, 50)
        answer = lcm(a, b)
        problems.append(f"Q: What is the least common multiple of {a} and {b}?\nA: {answer}\n\n")
    return problems

def generate_divisibility_problems(n_samples=500):
    """Generate divisibility problems."""
    problems = []
    for _ in range(n_samples):
        a = random.randint(2, 200)
        b = random.randint(2, 50)
        if random.random() < 0.5:
            # Make divisible
            a = b * random.randint(1, 20)
            answer = "yes"
        else:
            # Make not divisible
            while a % b == 0:
                a = random.randint(2, 200)
            answer = "no"
        problems.append(f"Q: Is {a} divisible by {b}?\nA: {answer}\n\n")
    return problems

def generate_modular_arithmetic_problems(n_samples=500):
    """Generate modular arithmetic problems."""
    problems = []
    for _ in range(n_samples):
        a = random.randint(1, 200)
        m = random.randint(2, 50)
        answer = a % m
        problems.append(f"Q: What is {a} mod {m}?\nA: {answer}\n\n")
    return problems

def generate_prime_check_problems(n_samples=500):
    """Generate prime checking problems."""
    problems = []
    # Mix of primes and non-primes
    test_numbers = []
    primes = [n for n in range(2, 300) if is_prime(n)]
    non_primes = [n for n in range(4, 300) if not is_prime(n)]
    
    for _ in range(n_samples):
        if random.random() < 0.4:  # 40% primes
            n = random.choice(primes)
            answer = "yes"
        else:
            n = random.choice(non_primes)
            answer = "no"
        problems.append(f"Q: Is {n} a prime number?\nA: {answer}\n\n")
    return problems

def generate_prime_factorization_problems(n_samples=300):
    """Generate prime factorization problems."""
    problems = []
    for _ in range(n_samples):
        n = random.randint(4, 200)
        factors = prime_factorization(n)
        answer = " × ".join(map(str, factors))
        problems.append(f"Q: What is the prime factorization of {n}?\nA: {answer}\n\n")
    return problems

def generate_simple_arithmetic_problems(n_samples=500):
    """Generate simple arithmetic in number theory context."""
    problems = []
    for _ in range(n_samples):
        op = random.choice(['+', '-', '*'])
        if op == '+':
            a, b = random.randint(1, 100), random.randint(1, 100)
            answer = a + b
            problems.append(f"Q: What is {a} + {b}?\nA: {answer}\n\n")
        elif op == '-':
            a, b = random.randint(1, 100), random.randint(1, 100)
            if a < b:
                a, b = b, a  # Keep positive
            answer = a - b
            problems.append(f"Q: What is {a} - {b}?\nA: {answer}\n\n")
        else:  # multiplication
            a, b = random.randint(2, 20), random.randint(2, 20)
            answer = a * b
            problems.append(f"Q: What is {a} × {b}?\nA: {answer}\n\n")
    return problems

def generate_exponent_problems(n_samples=300):
    """Generate exponentiation problems."""
    problems = []
    for _ in range(n_samples):
        base = random.randint(2, 12)
        exp = random.randint(2, 6)
        answer = base ** exp
        problems.append(f"Q: What is {base}^{exp}?\nA: {answer}\n\n")
    return problems

def generate_dataset():
    """Generate complete dataset."""
    print("Generating number theory problems...")
    
    all_problems = []
    all_problems.extend(generate_gcd_problems(500))
    all_problems.extend(generate_lcm_problems(500))
    all_problems.extend(generate_divisibility_problems(500))
    all_problems.extend(generate_modular_arithmetic_problems(500))
    all_problems.extend(generate_prime_check_problems(500))
    all_problems.extend(generate_prime_factorization_problems(300))
    all_problems.extend(generate_simple_arithmetic_problems(500))
    all_problems.extend(generate_exponent_problems(300))
    
    print(f"Generated {len(all_problems)} problems")
    
    # Shuffle
    random.shuffle(all_problems)
    
    # Combine into single text
    text = ''.join(all_problems)
    
    print(f"Total characters: {len(text)}")
    
    return text

def prepare_data():
    """Prepare and save tokenized data."""
    # Generate dataset
    text = generate_dataset()
    
    # Tokenize with GPT-2 BPE
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode_ordinary(text)
    
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
    
    train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
    
    print(f"Saved train.bin and val.bin")
    
    # Calculate approximate iterations per epoch
    # Assuming batch_size=2, gradient_accumulation_steps=16, block_size=256
    tokens_per_iter = 2 * 16 * 256
    iters_per_epoch = len(train_tokens) / tokens_per_iter
    print(f"\nWith batch_size=2, gradient_accumulation_steps=16, block_size=256:")
    print(f"Tokens per iteration: {tokens_per_iter}")
    print(f"Iterations per epoch: {iters_per_epoch:.1f}")
    print(f"For 3 epochs, set max_iters to ~{int(iters_per_epoch * 3)}")

if __name__ == '__main__':
    prepare_data()
