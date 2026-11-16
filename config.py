from dataclasses import dataclass

@dataclass
class Config:
    seed: int = 42
    data_slice: str = "train[:1%]"
    val_fraction: float = 0.1
    batch_size: int = 32
    block_size: int = 64
    n_embd = 64
    n_heads: int = 4
    head_size = n_embd // n_heads
    n_layer: int = 4
    learning_rate: float = 1e-3
   
    
