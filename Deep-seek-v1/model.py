import torch
import torch.nn as nn

def precompute_rope_cache(positions: torch.Tensor, d: int, base: float = 10000.0):
    k = torch.arange(0, d // 2, device=positions.device)
    w_k = 1.0 / torch.pow(base, 2 * k / d)
    angles = positions.reshape(-1, 1) * w_k
    return torch.sin(angles), torch.cos(angles)

def apply_rope(freq, x):
    sin, cos = freq
    sin = sin[None, :, None, :]
    cos = cos[None, :, None, :]
    x_even = x[..., 0::2]
    x_odd  = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out


class MLA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.hidden_dim = self.cfg.hidden_dim
        self.n_heads = self.cfg.n_heads
        self.head_dim = self.cfg.head_dim
        self.cq_dim = self.cfg.cq_dim
        self.ckv_dim = self.cfg.ckv_dim
        self.dim_rope = self.cfg.dim_rope

        self.w_q_down = nn.Linear(self.hidden_dim, self.cq_dim)
        self.w_q_up = nn.Linear(self.cq_dim, self.n_heads * self.head_dim)
        self.w_qr = nn.Linear(self.cq_dim, self.n_heads * self.dim_rope)

        self.w_kv_down = nn.Linear(self.hidden_dim, self.ckv_dim)
        self.w_kv_up = nn.Linear(self.ckv_dim, self.n_heads * self.head_dim + self.n_heads * self.head_dim)

        self.w_kr = nn.Linear(self.hidden_dim, self.dim_rope)

        self.w_o = nn.Linear(self.n_heads * self.head_dim, self.hidden_dim, bias=False)

    def forward(self, x, freqs, attention_mask):
        b, sl, c = x.shape
        c_q = self.w_q_down(x)

        q_nope = self.w_q_up(c_q).view(b, sl, self.n_heads, self.head_dim)
        q_rope = self.w_qr(c_q).view(b, sl, self.n_heads, self.dim_rope)

        c_kv = self.w_kv_down(x)
        k_nope, v = self.w_kv_up(c_kv).split([self.n_heads * self.head_dim, self.n_heads * self.head_dim], dim=-1)
        k_nope = k_nope.view(b, sl, self.n_heads, self.head_dim)
        v = v.view(b, sl, self.n_heads, self.head_dim)

        k_rope = self.w_kr(x).unsqueeze(2).expand(b, sl, self.n_heads, self.dim_rope)

        k_rope = apply_rope(freqs, k_rope)
        q_rope = apply_rope(freqs, q_rope)

        q = torch.cat((q_nope, q_rope), dim=-1).transpose(1, 2)
        k = torch.cat((k_nope, k_rope), dim=-1).transpose(1, 2)

        v = v.transpose(1, 2)
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attention_mask, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, sl, -1)

        attn_out = self.w_o(attn_out)
        return attn_out
    

class FFN(nn.Module):
def __init__(self, config):
    super().__init__()
    self.cfg = config
    self.gate_proj = nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim * 2, bias=False)
    self.up_proj = nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim * 2, bias=False)
    self.down_proj = nn.Linear(self.cfg.hidden_dim * 2, self.cfg.hidden_dim, bias=False)
    self.silu = nn.SiLU()

def forward(self, x):
    return self.down_proj(self.silu(self.gate_proj(x)) * self.up_proj(x))



class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight
    

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = MLA(config)
        self.ffn = FFN(config)
        self.norm1 = RMSNorm(config.hidden_dim)
        self.norm2 = RMSNorm(config.hidden_dim)

    def forward(self, x, freqs, attention_mask):
        x = x + self.attn(self.norm1(x), freqs, attention_mask)
        x = x + self.ffn(self.norm2(x))
        return x
    


class Deepseek(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.embed = nn.Embedding(self.cfg.vocab_size, self.cfg.hidden_dim)
        self.blocks = nn.ModuleList([Block(config) for _ in range(self.cfg.n_layers)])
        self.norm = RMSNorm(self.cfg.hidden_dim)
        self.head = nn.Linear(self.cfg.hidden_dim, self.cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, x, attention_mask):
        b, sl = x.shape
        positions = torch.arange(sl, device=x.device)
        freqs = precompute_rope_cache(positions, self.cfg.dim_rope)
        x = self.embed(x)
        for block in self.blocks:
            x = block(x, freqs, attention_mask)
        x = self.norm(x)
        x = self.head(x)
        return x
    

class Deepseek(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.embed = nn.Embedding(self.cfg.vocab_size, self.cfg.hidden_dim)
        self.blocks = nn.ModuleList([Block(config) for _ in range(self.cfg.n_layers)])
        self.norm = RMSNorm(self.cfg.hidden_dim)
        # Removed self.head = nn.Linear(...) and self.head.weight = self.embed.weight
        # Weights will be tied by directly using self.embed.weight in the forward pass

    def forward(self, x, attention_mask):
        b, sl = x.shape
        positions = torch.arange(sl, device=x.device)
        freqs = precompute_rope_cache(positions, self.cfg.dim_rope)
        x = self.embed(x) # Shape: (b, sl, hidden_dim)
        for block in self.blocks:
            x = block(x, freqs, attention_mask) # Shape: (b, sl, hidden_dim)
        x = self.norm(x) # Shape: (b, sl, hidden_dim)
        # Direct projection using tied embedding weights
        logits = x @ self.embed.weight.T # (b, sl, hidden_dim) @ (hidden_dim, vocab_size) -> (b, sl, vocab_size)
        return logits
    

from dataclasses import dataclass

@dataclass
class Config:
    vocab_size: int = 32000
    hidden_dim: int = 512
    n_heads: int = 8
    head_dim: int = 64
    cq_dim: int = 256
    ckv_dim: int = 256
    dim_rope: int = 64
    n_layers: int = 12
    eps: float = 1e-6
    max_seq_len: int = 768


model_config = Config()
model = Deepseek(model_config)