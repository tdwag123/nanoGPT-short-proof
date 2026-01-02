import time

out_dir = 'out-math-qa'
eval_interval = 10
eval_iters = 20
device = 'cpu'  # Force CPU since no CUDA available
compile = False  # Disable compilation (needs C++ compiler on Windows)
wandb_log = False  # feel free to turn on
wandb_project = 'math-number-theory'
wandb_run_name = 'ft-math-qa-' + str(time.time())

dataset = 'math_number_theory'
init_from = 'gpt2-medium'  # 355M params, good balance

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 2 batch_size * 16 grad_accum * 256 tokens = 8,192 tokens/iter
# Dataset has 53,224 train tokens (3,600 problems)
# so 1 epoch ~= 6.5 iters
batch_size = 2
gradient_accumulation_steps = 16
block_size = 256  # shorter context since answers are concise
max_iters = 40  # ~6 epochs for pilot training

# finetune at constant LR
learning_rate = 1e-5  # conservative starting point
decay_lr = False

# slight regularization to prevent overfitting
dropout = 0.1
