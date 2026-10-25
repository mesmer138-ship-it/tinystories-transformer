from dataclasses import dataclass

@dataclass
class Config:
    seed: int = 42
    data_slice: str = "train[:1%]"
    val_fraction: float = 0.1
    batch_size: int = 32
    block_size: int = 64
