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

class FFN(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        hidden_dim = int(8 * args.d_model / 3)
        hidden_dim = 256 * ((hidden_dim + 256 - 1) // 256)
        self.swiglu = SwiGLU(args.d_model, hidden_dim)

    def forward(self, x):
        return self.swiglu(x)


class CausalAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.n_heads = args.n_heads
        self.head_dim = args.d_model // args.n_heads
        self.wq = nn.Linear(args.d_model, args.d_model, bias=False)
        self.wk = nn.Linear(args.d_model, args.d_model, bias=False)
        self.wv = nn.Linear(args.d_model, args.d_model, bias=False)
        self.wo = nn.Linear(args.d_model, args.d_model, bias=False)

    def forward(self, x, freqs_cos, freqs_sin):
        B, T, C = x.size()
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(B, T, self.n_heads, self.head_dim)
        xk = xk.view(B, T, self.n_heads, self.head_dim)
        xv = xv.view(B, T, self.n_heads, self.head_dim)

        xq = apply_rotary_emb(xq, freqs_cos, freqs_sin)
        xk = apply_rotary_emb(xk, freqs_cos, freqs_sin)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(output)


class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.attention = CausalAttention(args)
        self.ffn = FFN(args)
        self.attention_norm = RMSNorm(args.d_model, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.d_model, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):
        h = x + self.attention(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.ffn(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.d_model)
        self.layers = nn.ModuleList([EncoderBlock(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.d_model, eps=args.norm_eps)
        self.output = nn.Linear(args.d_model, args.vocab_size, bias=False)

        freqs_cos, freqs_sin = precompute_freqs_cis(
            args.d_model // args.n_heads, args.max_seq_len, device=args.device
        )
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    def forward(self, x):
        h = self.tok_embeddings(x)
        for layer in self.layers:
            h = layer(h, self.freqs_cos, self.freqs_sin)
        h = self.norm(h)
        return self.output(h)
    
