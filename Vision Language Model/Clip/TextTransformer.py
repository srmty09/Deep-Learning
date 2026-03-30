import torch
import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F

@dataclass
class TextTransformerConfig:
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    hidden_dim: int = 512
    max_seq_len: int = 256
    n_vocab: int = 10000
    eos_token_id: int = 0

    @property
    def head_dim(self):
        return self.d_model // self.n_heads

class TextEmbedding(nn.Module):
    def __init__(self, config: TextTransformerConfig):
        super().__init__()
        self.cfg = config
        self.token_embedding = nn.Embedding(self.cfg.n_vocab, self.cfg.d_model)
        self.pos_embedding = nn.Embedding(self.cfg.max_seq_len, self.cfg.d_model)

    def forward(self, x):
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        return self.token_embedding(x) + self.pos_embedding(positions)

class TextTransformerEncoder(nn.Module):
    def __init__(self, config: TextTransformerConfig):
        super().__init__()
        self.cfg = config
        self.layernorm1 = nn.LayerNorm(self.cfg.d_model)
        self.layernorm2 = nn.LayerNorm(self.cfg.d_model)
        self.mlp = nn.Sequential(
            *[
                nn.Linear(self.cfg.d_model, self.cfg.hidden_dim),
                nn.GELU(),
                nn.Linear(self.cfg.hidden_dim, self.cfg.d_model)
            ]
        )

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        residual1 = x
        x = self.layernorm1(x)
        x = x.reshape(batch_size, seq_len, self.cfg.n_heads, self.cfg.head_dim).transpose(1, 2)
        x = F.scaled_dot_product_attention(
            x,
            x,
            x,
            attn_mask=mask,
            dropout_p=0.1
        )
        x = x.transpose(1, 2).reshape(batch_size, seq_len, d_model)
        x = x + residual1
        residual2 = x
        x = self.layernorm2(x)
        x = self.mlp(x)
        x = x + residual2
        return x

class TextTransformer(nn.Module):
    def __init__(self, config: TextTransformerConfig):
        super().__init__()
        self.cfg = config
        self.embedding = TextEmbedding(self.cfg)
        self.encoder = nn.ModuleList(
            [TextTransformerEncoder(self.cfg) for _ in range(self.cfg.n_layers)]
        )
        self.layernorm = nn.LayerNorm(self.cfg.d_model)

    def forward(self, x):
        batch_size, seq_len = x.shape
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
        eos_positions = (x == self.cfg.eos_token_id).int().argmax(dim=-1)
        x = self.embedding(x)
        for layer in self.encoder:
            x = layer(x, causal_mask)
        x = self.layernorm(x)
        return x[torch.arange(batch_size), eos_positions]

# cfg = TextTransformerConfig()
# m = TextTransformer(cfg)
# x = torch.randint(0, cfg.n_vocab, (2, 128))
# print(m(x).shape)