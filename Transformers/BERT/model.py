import torch
import torch.nn as nn
from torch.nn import functional as F
import random 
import math
from typing import Optional


class BertEmbeddings(nn.Module):
    """Token, Position, and Token-Type Embeddings combined."""
    def __init__(self, config):
        super().__init__()
        self.position_embeddings = nn.Embedding(config.seq_len, config.d_model)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.d_model)
        self.word_embeddings = nn.Embedding(config.vocab_size, config.d_model)

        self.layernorm = nn.LayerNorm(config.d_model, eps=1e-12)
        self.dropout = nn.Dropout(config.dp)

    def forward(self, input_ids, token_type_ids):
        seq_len = input_ids.size(1)

        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    # ... (Keep this class as is)
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.d_model % config.n_h == 0
        self.c_attn = nn.Linear(config.d_model, config.d_model * 3)
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dp)

    def forward(self, x, attention_mask):
        batch_size, seq_len, d_model = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(d_model, dim=2)
        q = q.view(batch_size, seq_len, self.config.n_h, d_model // self.config.n_h).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.config.n_h, d_model // self.config.n_h).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.config.n_h, d_model // self.config.n_h).transpose(1, 2)
        attention_scores = q @ k.transpose(-2, -1) * (1 / math.sqrt(k.size(-1)))

        pad = (attention_mask == 0).float()
        pad = pad.unsqueeze(1).unsqueeze(2)
        attention_scores = attention_scores.masked_fill(pad.bool(), -1e9)

        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)
        out = attention_scores @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        out = self.c_proj(out)
        return out


class Residual(nn.Module):
    def __init__(self, sublayer, config):
        super().__init__()
        self.sublayer = sublayer(config)
        self.ln = nn.LayerNorm(config.d_model)

    def forward(self, x, attention_mask: Optional[torch.Tensor] = None):
        if attention_mask is not None:
            return self.sublayer(x, attention_mask)
        return self.sublayer(x)


class ffn(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(config.d_model, config.dff),
            nn.GELU(),
            nn.Linear(config.dff, config.d_model),
            nn.Dropout(config.dp)
        )

    def forward(self, x, attention_mask=None):
        return self.layer(x)


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_ln = nn.LayerNorm(config.d_model)
        self.feed_forward_ln = nn.LayerNorm(config.d_model)

        self.attn = Residual(lambda cfg: BertSelfAttention(cfg), config=config)
        self.ff = Residual(ffn, config=config)

    def forward(self, x, attention_mask):

        attn_out = self.attn(x, attention_mask)
        x = self.attention_ln(x + attn_out)

        ff_out = self.ff(x)
        x = self.feed_forward_ln(x + ff_out)

        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.layers = nn.ModuleList([EncoderBlock(config=config) for _ in range(config.n_layers)])

    def forward(self, input_ids, token_type_ids, attention_mask):
        x = self.embeddings(input_ids, token_type_ids)

        for layer in self.layers:
            x = layer(x, attention_mask)

        return x


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.shared_embedding = self.encoder.embeddings.word_embeddings

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.shared_embedding.weight
        self.lm_bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.lm_head.bias = self.lm_bias

        self.nsp_head = nn.Linear(config.d_model, 2)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.d_model**-0.5)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, token_type_ids, attention_mask): # New: accepts token_type_ids
        enc_out = self.encoder(input_ids, token_type_ids, attention_mask)

        logits_mlm = self.lm_head(enc_out) + self.lm_bias

        # Only use the [CLS] token (first token) for NSP
        cls_token = enc_out[:, 0, :]
        logits_nsp = self.nsp_head(cls_token)
        return logits_mlm, logits_nsp