import time

out_dir = 'out-orca-math'
eval_interval = 10
eval_iters = 20
device = 'cpu'
compile = False
wandb_log = False
wandb_project = 'orca-math'
wandb_run_name = 'ft-orca-' + str(time.time())

dataset = 'orca_math'
init_from = 'gpt2-medium'

always_save_checkpoint = False

# Filtered Orca dataset: 11.2M train tokens (70k examples, <=256 tokens each)
# With these settings: 1 epoch = ~1366 iters
batch_size = 2
gradient_accumulation_steps = 16
block_size = 256  # Filtered to this length
max_iters = 50  # Small pilot run

# Fine-tuning settings
learning_rate = 1e-5
decay_lr = False
dropout = 0.1
