from dataclasses import dataclass

@dataclass
class Config:
    seed: int = 42
    data_slice: str = "train[:1%]"
    val_fraction: float = 0.1
    batch_size: int = 32
    block_size: int = 64
    n_embd: int = 64
    n_heads: int = 4
    n_layer: int = 4
    learning_rate: float = 1e-3
    max_steps: int = 1000
    eval_interval: int = 100
    eval_iters: int = 10
    run_name: str = "baseline_seed42"
    log_dir: str = "/content/mydrive/MyDrive/runs"
   
    
