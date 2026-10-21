import random
import numpy as np
import torch
from datasets import load_dataset
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
ds = load_dataset("roneneldan/TinyStories", split="train[:1%]")


