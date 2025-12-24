import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class ModelArgs:
    dim :int = 4096
    n_layers: int = 32
    n_heads:int = 32
    vocab_size:int = -1
    norm_eps:float = 1e-5
    max_batch_size:int = 32
    max_seq_len:int = 2048


class SwiGLU(nn.Module):
    def __init__(self,dimension) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(dimension,dimension)
        self.linear_2 = nn.Linear(dimension,dimension)
        

    def forward(self,x):
        output = self.linear_1(x)
        swish = output * torch.sigmoid(output)
        swish = swish * self.linear_2(x)

        return swish


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d))  

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True))
        x_norm = x / (rms + self.eps)
        return x_norm * self.gamma


