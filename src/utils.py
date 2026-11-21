import random
import numpy as np
import torch
import math

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_lr(step,cfg):
    if step < cfg.warmup_steps:
        lr = cfg.learning_rate *(step+1)/cfg.warmup_steps
    else:
        progress = (step - cfg.warmup_steps)/(cfg.max_steps - cfg.warmup_steps)
        factor = 0.5 *(1+math.cos(math.pi * progress))
        lr = cfg.min_lr + factor *(cfg.learning_rate-cfg.min_lr)
    return lr
        

