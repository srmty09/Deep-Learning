import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple,Optional
import math
import torch.nn.functional as F


@dataclass
class ModelArgs:
    d_model: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    vocab_size: int = -1
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048
    n_kv_heads: Optional[int] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"



class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))



class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))



def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device: str = "cpu"):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin

def apply_rotary_emb(x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    cos = freqs_cos[: x.shape[1], None, :]
    sin = freqs_sin[: x.shape[1], None, :]
    x_out_even = x_even * cos - x_odd * sin
    x_out_odd = x_even * sin + x_odd * cos
    return torch.stack((x_out_even, x_out_odd), dim=-1).flatten(-2)


