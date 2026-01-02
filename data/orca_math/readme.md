# Orca Math Word Problems Dataset

This dataset contains 200k math word problems from Microsoft's Orca-Math dataset.

Source: https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k

## Dataset Format

Each example contains:
- **question**: A math word problem
- **answer**: The solution with step-by-step reasoning

## Usage

Run the preparation script to download and tokenize the dataset:

```bash
python data/orca_math/prepare.py
```

This will:
1. Download the dataset from HuggingFace
2. Format it as "Q: [question]\nA: [answer]\n\n"
3. Tokenize with GPT-2 BPE encoding
4. Create `train.bin` and `val.bin` files (90/10 split)

Note: The download is ~200k examples and may take several minutes.
