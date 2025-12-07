from dataclasses import dataclass
import torch

@dataclass
class ModelConfig:
    d_model = 768
    n_h = 12
    seq_len = 128
    batch_size = 8
    dff = 3072
    dp = 0.1
    n_layers = 12
    vocab_size = 30522
    type_vocab_size = 2
