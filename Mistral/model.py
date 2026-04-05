import math
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class TinyMistralConfig:
    hidden_dim: int = 512
    num_experts: int = 5
    num_experts_per_tok: int = 2
    intermediate_dim: int = 1344
    n_heads: int = 8
    n_kv_heads: int = 2
    head_dim: int = 64
    n_layers: int = 6

    batch_size: int = 32
    max_seq_len: int = 256
    learning_rate: float = 3e-4
    attention_dropout: float = 0.1
    aux_loss_weight: float = 0.01

    vocab_size: int = 8000
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class RMSNorm(nn.Module):
    def __init__(self, config: TinyMistralConfig):
        super().__init__()
        self.w = nn.Parameter(torch.ones(config.hidden_dim))
        self.eps = config.rms_norm_eps

    def forward(self, x):
        x_float = x.float()
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_float * torch.rsqrt(variance + self.eps)
        return (self.w * x_normed).type_as(x)


class RotaryEmbedding(nn.Module):
    def __init__(self, config: TinyMistralConfig):
        super().__init__()
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, config.head_dim, 2).float() / config.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, position_ids):
        with torch.amp.autocast(device_type=position_ids.device.type, enabled=False):
            freqs = torch.einsum("i, j -> ij", position_ids.reshape(-1).float(), self.inv_freq.float())
            freqs = freqs.view(position_ids.shape[0], position_ids.shape[1], -1)
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
            return freqs_cis


def apply_rotary_emb(xq, xk, freq_cis):
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freq_cis = freq_cis.unsqueeze(2)
    xq_out = torch.view_as_real(xq_complex * freq_cis).flatten(3)
    xk_out = torch.view_as_real(xk_complex * freq_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(hidden_state, n_rep):
    bd, sl, kvh, hd = hidden_state.shape
    return hidden_state.unsqueeze(3).expand(bd, sl, kvh, n_rep, hd).reshape(bd, sl, kvh * n_rep, hd)


class Router(nn.Module):
    def __init__(self, config: TinyMistralConfig):
        super().__init__()
        self.router = nn.Linear(config.hidden_dim, config.num_experts, bias=False)

    def forward(self, x):
        return self.router(x)


class Expert(nn.Module):
    def __init__(self, config: TinyMistralConfig):
        super().__init__()
        self.gate = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.up = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.down = nn.Linear(config.intermediate_dim, config.hidden_dim, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.down(self.act(self.gate(x)) * self.up(x))


class MOE(nn.Module):
    def __init__(self, config: TinyMistralConfig):
        super().__init__()
        self.cfg = config
        self.router = Router(config)
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.num_experts)])

    def forward(self, x):
        batch, seq_len, h_dim = x.shape
        x_flat = x.view(-1, h_dim)

        router_logits = self.router(x_flat)
        router_probs = torch.nn.functional.softmax(router_logits, dim=-1)

        values, indices = torch.topk(router_probs, k=self.cfg.num_experts_per_tok, dim=-1)
        values = values / (values.sum(dim=-1, keepdim=True) + 1e-9)

        expert_probability = router_probs.mean(dim=0)
        expert_load = torch.nn.functional.one_hot(indices, num_classes=self.cfg.num_experts).sum(dim=1).float().mean(dim=0)
        router_loss = self.cfg.num_experts * torch.sum(expert_probability * expert_load)

        final_output = torch.zeros_like(x_flat)

        for i, expert in enumerate(self.experts):
            mask = (indices == i)
            if mask.any():
                token_idx, topk_idx = torch.where(mask)
                expert_out = expert(x_flat[token_idx])
                weighted_out = expert_out * values[token_idx, topk_idx].unsqueeze(-1)
                final_output.index_add_(0, token_idx, weighted_out)

        return final_output.view(batch, seq_len, h_dim), router_loss * self.cfg.aux_loss_weight


class SlidingWindowAttention(nn.Module):
    def __init__(self, config: TinyMistralConfig):
        super().__init__()
        self.cfg = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads
        self.window_size = 64

        self.wq = nn.Linear(config.hidden_dim, config.n_heads * config.head_dim, bias=False)
        self.wk = nn.Linear(config.hidden_dim, config.n_kv_heads * config.head_dim, bias=False)
        self.wv = nn.Linear(config.hidden_dim, config.n_kv_heads * config.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * config.head_dim, config.hidden_dim, bias=False)
        self.drop = nn.Dropout(config.attention_dropout)

    def forward(self, x, freqs_cis, attention_mask=None):
        bd, sl, _ = x.shape

        q = self.wq(x).view(bd, sl, self.n_heads, self.head_dim)
        k = self.wk(x).view(bd, sl, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(bd, sl, self.n_kv_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cis)
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        positions = torch.arange(sl, device=q.device)
        distance = positions.unsqueeze(0) - positions.unsqueeze(1)
        combined = (distance > 0) | (distance < -self.window_size)

        mask = torch.zeros(sl, sl, device=q.device).masked_fill(combined, float("-inf"))
        attn_weights = attn_weights + mask.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, float("-inf"))

        attn_weights = attn_weights.clamp(min=-1e9)
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(q)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.drop(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(bd, sl, self.n_heads * self.head_dim)

        return self.wo(out)


class Decoder(nn.Module):
    def __init__(self, config: TinyMistralConfig):
        super().__init__()
        self.attn = SlidingWindowAttention(config)
        self.moe = MOE(config)
        self.pre_layer_norm = RMSNorm(config)
        self.post_layer_norm = RMSNorm(config)

    def forward(self, hidden_state, freqs_cis, attention_mask=None):
        residual = hidden_state
        hidden_state = self.pre_layer_norm(hidden_state)
        hidden_state = self.attn(hidden_state, freqs_cis, attention_mask) + residual

        residual = hidden_state
        hidden_state = self.post_layer_norm(hidden_state)
        out_moe, router_loss = self.moe(hidden_state)

        return residual + out_moe, router_loss


class TinyMistral(nn.Module):
    def __init__(self, config: TinyMistralConfig):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.layers = nn.ModuleList([Decoder(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config)
        self.rotary_emb = RotaryEmbedding(config)

    def forward(self, tokens, position_ids=None, attention_mask=None):
        batch, seq_len = tokens.shape

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch, seq_len)

        h = self.embed(tokens)
        freqs_cis = self.rotary_emb(position_ids)

        total_router_loss = 0.0

        for layer in self.layers:
            h, layer_router_loss = layer(h, freqs_cis, attention_mask)
            total_router_loss += layer_router_loss

        return self.norm(h), total_router_loss


class TinyMistralForCausalLM(nn.Module):
    def __init__(self, config: TinyMistralConfig):
        super().__init__()
        self.model = TinyMistral(config)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        self.lm_head.weight = self.model.embed.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.w)

    def forward(self, tokens, position_ids=None, attention_mask=None):
        hidden_states, router_loss = self.model(tokens, position_ids, attention_mask)
        return self.lm_head(hidden_states), router_loss


cfg = TinyMistralConfig()
model = TinyMistralForCausalLM(cfg).to(cfg.device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")