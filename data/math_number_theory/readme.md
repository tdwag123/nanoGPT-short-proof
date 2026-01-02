# Elementary Number Theory Q&A Dataset

This dataset contains synthetically generated elementary number theory problems with answers.

## Problem Types

- **GCD (Greatest Common Divisor)**: Finding GCD of two numbers
- **LCM (Least Common Multiple)**: Finding LCM of two numbers  
- **Divisibility**: Checking if one number divides another
- **Modular Arithmetic**: Computing remainders (mod operations)
- **Prime Checking**: Determining if a number is prime
- **Prime Factorization**: Breaking numbers into prime factors
- **Basic Arithmetic**: Addition, subtraction, multiplication
- **Exponentiation**: Computing powers

## Dataset Size

- ~3,600 total problems
- Format: "Q: [question]\nA: [answer]\n\n"
- 90/10 train/val split

## Usage

Run the preparation script to generate the dataset:

```bash
python data/math_number_theory/prepare.py
```

This will create `train.bin` and `val.bin` files containing tokenized data.
