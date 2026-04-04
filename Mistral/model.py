import math
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class TinyMistralConfig():
    hidden_dim: int = 768
    num_experts: int = 5
    rms_norm_eps: float = 1e-8
    intermediate_dim: int = 1024
    n_heads: int = 8
    n_kv_heads: int = 2
    head_dim: int = 96
    n_layers: int = 12
    attention_dropout: float = 0.4
    rope_theta: float = 10000.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size: int = 32000


class RMSNorm(nn.Module):
    def __init__(self, config: TinyMistralConfig):
        super().__init__()
        self.cfg = config
        self.w = nn.Parameter(torch.ones(self.cfg.hidden_dim))
        self.eps = self.cfg.rms_norm_eps

    def forward(self, x):
        sqr_x = x * x
        rms_x = torch.rsqrt(sqr_x.mean(dim=-1, keepdim=True) + self.eps)
        out = x * rms_x * self.w
        return out


class RotaryEmbedding(nn.Module):
    def __init__(self, config: TinyMistralConfig):
        super().__init__()
        self.config = config
        inv_freq = self._compute_default_rope_parameters(self.config, device=self.config.device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _compute_default_rope_parameters(self, config, device):
        base = config.rope_theta
        head_dim = config.head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).to(device, dtype=torch.float) / head_dim))
        return inv_freq

    @torch.no_grad()
    def forward(self, position_ids):
        with torch.amp.autocast(device_type=position_ids.device.type, enabled=False):
            freqs = torch.einsum("i, j -> ij", position_ids.reshape(-1).float(), self.inv_freq.float())
            freqs = freqs.view(position_ids.shape[0], position_ids.shape[1], -1)
            freqs_cis = torch.polar(torch.ones_like(freqs), angle=freqs)
            return freqs_cis


def apply_rotary_emb(xq, xk, freq_cis):
    # xq shape: (batch, seq, n_heads, head_dim)
    # freq_cis shape: (batch, seq, head_dim/2)

    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # freq_cis needs to broadcast from (batch, seq, head_dim/2) to (batch, seq, n_heads, head_dim/2)
    freq_cis = freq_cis.unsqueeze(2)

    xq_out = torch.view_as_real(xq_complex * freq_cis).flatten(3)
    xk_out = torch.view_as_real(xk_complex * freq_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)  


class Router(nn.Module):
    def __init__(self, config: TinyMistralConfig):
        super().__init__()
        self.cfg = config
        self.router = nn.Linear(self.cfg.hidden_dim, self.cfg.num_experts)
        self.act = nn.Softmax(dim=-1)

    def forward(self, x):
        router_logits = self.router(x)
        router_scores = self.act(router_logits)
        return router_scores


class Expert(nn.Module):
    def __init__(self, config: TinyMistralConfig):
        super().__init__()
        self.cfg = config
        self.gate = nn.Linear(self.cfg.hidden_dim, self.cfg.intermediate_dim, bias=False)
        self.up = nn.Linear(self.cfg.hidden_dim, self.cfg.intermediate_dim, bias=False)
        self.down = nn.Linear(self.cfg.intermediate_dim, self.cfg.hidden_dim, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.down(self.act(self.gate(x)) * self.up(x))


class MOE(nn.Module):
    def __init__(self, config: TinyMistralConfig):
        super().__init__()
        self.cfg = config
        self.router = Router(self.cfg)
        self.experts = nn.ModuleList(
            [Expert(self.cfg) for _ in range(self.cfg.num_experts)]
        )

    def forward(self, x):
        batch, seq_len, h_dim = x.shape
        x_flat = x.view(-1, h_dim)

        router_score = self.router(x_flat)
        values, indices = torch.topk(router_score, k=2, dim=-1)
        values = values / values.sum(dim=-1, keepdim=True)

        final_output = torch.zeros_like(x_flat)

        for i, expert in enumerate(self.experts):
            mask = (indices == i)
            if mask.any():
                token_idx, topk_idx = torch.where(mask)
                expert_out = expert(x_flat[token_idx])
                weighted_out = expert_out * values[token_idx, topk_idx].unsqueeze(-1)
                final_output.index_add_(0, token_idx, weighted_out)

        return final_output.view(batch, seq_len, h_dim)


def repeat_kv(hidden_state, n_rep):
    bd, sl, kvh, hd = hidden_state.shape
    hidden_state = hidden_state.unsqueeze(3)
    hidden_state = hidden_state.expand(bd, sl, kvh, n_rep, hd)
    hidden_state = hidden_state.reshape(bd, sl, kvh * n_rep, hd)
    return hidden_state


class SlidingWindowAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.hidden_dim = config.hidden_dim
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim  
        self.n_rep = self.n_heads // self.n_kv_heads
        self.window_size = 3

        self.wq = nn.Linear(self.hidden_dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.hidden_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.hidden_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.hidden_dim, bias=False)

        self.drop_out = nn.Dropout(config.attention_dropout)

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

        mask = torch.full((sl, sl), float("-inf"), device=q.device)
        mask = torch.triu(mask, diagonal=1)

        sliding_mask = torch.tril(torch.ones(sl, sl, device=q.device), diagonal=-self.window_size)
        mask.masked_fill_(sliding_mask.bool(), float("-inf"))

        attn_weights = attn_weights + mask

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(q)
        attn_weights = self.drop_out(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(bd, sl, self.n_heads * self.head_dim)

        return self.wo(out), attn_weights


class Decoder(nn.Module):
    def __init__(self, cfg: TinyMistralConfig):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = cfg.hidden_dim
        self.attn_layer = SlidingWindowAttention(cfg)
        self.moe = MOE(cfg)
        self.pre_layer_norm = RMSNorm(cfg)
        self.post_layer_norm = RMSNorm(cfg)

    def forward(self, hidden_state, freqs_cis, attention_mask=None):
        residual = hidden_state
        hidden_state = self.pre_layer_norm(hidden_state)
        atten_out, _ = self.attn_layer(hidden_state, freqs_cis, attention_mask)
        hidden_state = atten_out + residual

        residual = hidden_state
        hidden_state = self.post_layer_norm(hidden_state)
        out_moe = self.moe(hidden_state)

        return residual + out_moe


class TinyMistral(nn.Module):
    def __init__(self, cfg: TinyMistralConfig):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = cfg.vocab_size

        self.embed = nn.Embedding(self.vocab_size, cfg.hidden_dim)
        self.layers = nn.ModuleList([
            Decoder(cfg) for _ in range(cfg.n_layers)
        ])
        self.norm = RMSNorm(cfg)
        self.rotary_emb = RotaryEmbedding(cfg)

    def forward(self, tokens, position_ids=None, attention_mask=None):
        batch, seq_len = tokens.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch, seq_len)

        h = self.embed(tokens)
        freqs_cis = self.rotary_emb(position_ids)

        for layer in self.layers:
            h = layer(h, freqs_cis, attention_mask)

        h = self.norm(h)
        return h


class TinyMistralForCausalLM(nn.Module):
    def __init__(self, cfg: TinyMistralConfig):
        super().__init__()
        self.model = TinyMistral(cfg)
        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.w)

    def forward(self, tokens, position_ids=None, attention_mask=None):
        hidden_states = self.model(tokens, position_ids, attention_mask)
        logits = self.lm_head(hidden_states)
        return logits