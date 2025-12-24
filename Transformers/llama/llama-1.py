import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple,Optional
import math


@dataclass
class ModelArgs:
    d_model :int = 4096
    n_layers: int = 32
    n_heads:int = 32
    vocab_size:int = -1
    norm_eps:float = 1e-5
    max_batch_size:int = 32
    max_seq_len:int = 2048
    n_kv_heads: Optional[int] = None # Number of head for the k and v

    device: str = "cuda" if torch.cuda.is_available else "cpu"



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




def calculate_thetas(d_model: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    i = torch.arange(0, d_model // 2) 
    w_i = 1 / torch.pow(torch.tensor(1e4), (2 * i) / d_model)  
    p = torch.arange(0, seq_len).unsqueeze(1)  
    theta = p * w_i  
    return torch.sin(theta), torch.cos(theta)  

class RotaryPosEmb(nn.Module):
    def __init__(self, config):
        super().__init__()
        sin, cos = calculate_thetas(config.d_model, config.max_seq_len)  
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)
    
    def forward(self, x):
        
        seq_len = x.shape[-2]
        cos = self.cos[:seq_len]   # type: ignore
        sin = self.sin[:seq_len]   # type: ignore
        
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_even_rot = x_even * cos - x_odd * sin  
        x_odd_rot = x_even * sin + x_odd * cos   
        return torch.stack((x_even_rot, x_odd_rot), dim=-1).flatten(-2)        



